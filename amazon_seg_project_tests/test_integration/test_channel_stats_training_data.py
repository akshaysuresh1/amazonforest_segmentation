"""
Integration test for channel_stats_training_data()
"""

from unittest.mock import patch, MagicMock
from io import BytesIO
import boto3
from moto import mock_aws
import numpy as np
import pytest
from amazon_seg_project.ops import simulate_mock_multispec_data
from amazon_seg_project.resources import AMAZON_TIF_BUCKET
from amazon_seg_project.assets import channel_stats_training_data


@mock_aws
@patch("amazon_seg_project.assets.statistics.s3_resource")
@patch("amazon_seg_project.assets.statistics.write_stats")
def test_channel_stats_training_data_success(
    mock_write_stats: MagicMock, mock_s3_resource: MagicMock
) -> None:
    """
    Test successful execution of channel_stats_training_data()
    """
    # Create a mock S3 client.
    s3_client = boto3.client("s3", region_name="us-east-1")
    bucket_name = AMAZON_TIF_BUCKET.get_value()

    # Create the bucket.
    s3_client.create_bucket(Bucket=bucket_name)

    # Set up mock .tif files.
    tif_channels = {"file1.tif": 3, "file2.tif": 4, "file3.tif": 4}
    channel_sums = np.zeros(4, dtype=np.float64)
    channel_squared_sums = np.zeros(4, dtype=np.float64)
    count_valid_tif = 0
    for object_key, channel_count in tif_channels.items():
        dataset = simulate_mock_multispec_data(n_bands=channel_count, n_y=64, n_x=64)

        # Convert dataset to GeoTiff in memory.
        buffer = BytesIO()
        dataset.rio.to_raster(buffer, driver="GTiff")
        buffer.seek(0)  # Reset buffer position

        # Upload GeoTiff to S3.
        s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=buffer.getvalue())

        # Update channel-wise sums and squared sums.
        if channel_count == 4:
            count_valid_tif += 1
            dataset = dataset.astype(np.float64)
            channel_sums += dataset.mean(dim=("y", "x")).values
            channel_squared_sums += (dataset**2).mean(dim=("y", "x")).values

    # Compute expected results.
    expected_means = channel_sums / count_valid_tif
    expected_sigma = np.sqrt(
        (channel_squared_sums / count_valid_tif) - expected_means**2
    )

    # Connect the mock Dagster AWS S3Resource to boto3 client.
    mock_s3_resource.get_client.return_value = s3_client

    # Call the test function.
    outputs = list(channel_stats_training_data(list(tif_channels.keys())))
    actual_means = outputs[0].value
    actual_sigma = outputs[1].value

    # Validate the outputs.
    assert (
        actual_means is not None
    ), "Channel-wise mean of training data should not be None."
    assert (
        actual_sigma is not None
    ), "Per-channel standard deviation of training data should not be None."

    # Ensure values are within reasonable bounds.
    assert np.all(actual_means >= 0), "Mean values should be non-negative."
    assert np.all(
        actual_sigma >= 0
    ), "Standard deviation values should be non-negative."

    # Verify that outputs match expected results.
    np.testing.assert_array_equal(actual_means, expected_means)
    np.testing.assert_array_equal(actual_sigma, expected_sigma)

    # Assert that the write_stats() function is called exactly once.
    mock_write_stats.assert_called_once_with(
        actual_means, actual_sigma, ["Red", "Green", "Blue", "NIR"]
    )


@mock_aws
@patch("amazon_seg_project.assets.statistics.s3_resource")
def test_channel_stats_training_data_invalid_inputs(
    mock_s3_resource: MagicMock,
) -> None:
    """
    Validate the response of channel_stats_training_data() to fully invalid input data.

    In other words, none of the input .tif files have exactly four color channels.
    """
    with pytest.raises(
        ValueError, match="No valid GeoTIFF images found in training data."
    ):
        # Create a mock S3 client.
        s3_client = boto3.client("s3", region_name="us-east-1")
        bucket_name = AMAZON_TIF_BUCKET.get_value()

        # Create the bucket.
        s3_client.create_bucket(Bucket=bucket_name)

        # Set up mock .tif files of which none are valid inputs.
        tif_channels = {"file1.tif": 3, "file2.tif": 5}
        for object_key, channel_count in tif_channels.items():
            dataset = simulate_mock_multispec_data(
                n_bands=channel_count, n_y=64, n_x=64
            )

            # Convert dataset to GeoTiff in memory.
            buffer = BytesIO()
            dataset.rio.to_raster(buffer, driver="GTiff")
            buffer.seek(0)  # Reset buffer position

            # Upload GeoTiff to S3.
            s3_client.put_object(
                Bucket=bucket_name, Key=object_key, Body=buffer.getvalue()
            )

        # Connect the mock Dagster AWS S3Resource to boto3 client.
        mock_s3_resource.get_client.return_value = s3_client

        # Call the test function.
        list(channel_stats_training_data(list(tif_channels.keys())))
