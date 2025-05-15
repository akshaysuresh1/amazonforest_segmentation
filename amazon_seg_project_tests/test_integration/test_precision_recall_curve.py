"""
Unit test for asset "precision_recall_curve"
"""

from unittest.mock import patch, MagicMock
from io import BytesIO
import numpy as np
import pandas as pd
import boto3
from moto import mock_aws
from segmentation_models_pytorch import Unet
from amazon_seg_project.assets import precision_recall_curve, SegmentationDataset
from amazon_seg_project.config import PrecRecallCurveConfig
from amazon_seg_project.ops.tif_utils import (
    simulate_mock_multispec_data,
    simulate_mock_binary_mask,
)
from amazon_seg_project.data_paths import OUTPUT_PATH


@mock_aws
@patch("amazon_seg_project.assets.precision_recall_metrics.plot_precision_recall_curve")
@patch("amazon_seg_project.assets.precision_recall_metrics.write_precision_recall_data")
@patch("amazon_seg_project.assets.precision_recall_metrics.compute_f1_scores")
@patch("amazon_seg_project.assets.precision_recall_metrics.smp_metrics")
@patch("amazon_seg_project.assets.dataset_definition.s3_resource")
@patch("logging.info")
def test_precision_recall_curve_execution(
    mock_logging: MagicMock,
    mock_s3_resource: MagicMock,
    mock_smp_metrics: MagicMock,
    mock_compute_f1_scores: MagicMock,
    mock_write_precision_recall_data: MagicMock,
    mock_plot_precision_recall_curve: MagicMock,
) -> None:
    """
    Test for successful execution of precision_recall_curve().
    """
    # Multispectral image data properties
    color_channels = 4  # No. of color channels
    img_height = 16  # pixel count
    img_width = 16  # pixel count

    # Set up test config and dummy model.
    test_config = PrecRecallCurveConfig(thresholds_list=[0.1, 0.5, 0.9])
    model = Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=color_channels,
        activation="sigmoid",
    )

    # Inputs to mock validation dataset
    s3_bucket = "mock-bucket"
    images_list = ["images/file1.tif", "images/file2.tif", "images/file3.tif"]
    masks_list = ["masks/file1.tif", "masks/file2.tif", "masks/file3.tif"]

    # Create mock S3 client.
    s3_client = boto3.client("s3", region_name="us-east-1")
    # Create S3 bucket.
    s3_client.create_bucket(Bucket=s3_bucket)

    n_samples = 0  # No. of pixels across all images
    n_positive_samples = 0  # No. of pixels labeled as "1" in ground truth masks
    for index in range(len(images_list)):
        # Create and upload image data to S3.
        image_ds = simulate_mock_multispec_data(
            n_bands=4, n_y=img_height, n_x=img_width
        )
        n_samples += img_height * img_width
        # Convert dataset to GeoTiff in memory.
        buffer = BytesIO()
        image_ds.rio.to_raster(buffer, driver="GTiff")
        buffer.seek(0)  # Reset buffer position
        # Upload GeoTiff to S3.
        s3_client.put_object(
            Bucket=s3_bucket, Key=images_list[index], Body=buffer.getvalue()
        )

        # Create and upload segmentation mask to S3 bucket.
        mask_ds = simulate_mock_binary_mask(n_y=img_height, n_x=img_width)
        n_positive_samples += np.sum(mask_ds.values)
        # Convert dataset to GeoTiff in memory.
        buffer = BytesIO()
        mask_ds.rio.to_raster(buffer, driver="GTiff")
        buffer.seek(0)  # Reset buffer position
        # Upload GeoTiff to S3.
        s3_client.put_object(
            Bucket=s3_bucket, Key=masks_list[index], Body=buffer.getvalue()
        )

    # Connect the mock Dagster AWS S3Resource to boto3 client.
    mock_s3_resource.get_client.return_value = s3_client

    # Create a SegmentationDataset object for mock validation dataset.
    val_dataset = SegmentationDataset(images_list, masks_list, s3_bucket)

    # Return fixed outputs from mocked functions for metric computation
    mock_smp_metrics.return_value = {"Precision": 0.6, "Recall": 0.4}
    mock_compute_f1_scores.return_value = np.array(
        [0.48] * len(test_config.thresholds_list)
    )

    # Call the test function.
    output_df = precision_recall_curve(test_config, val_dataset, model)

    # Expected intermediate compute products
    threshold_values = np.array(test_config.thresholds_list)
    recall_values = np.array(
        [mock_smp_metrics.return_value.get("Recall")] * len(threshold_values)
    )
    precision_values = np.array(
        [mock_smp_metrics.return_value.get("Precision")] * len(threshold_values)
    )
    expected_df = pd.DataFrame(
        {
            "Binarization threshold": threshold_values,
            "Recall": recall_values,
            "Precision": precision_values,
            "F1 score": mock_compute_f1_scores.return_value,
        }
    )

    # Assertions for output_df
    assert isinstance(output_df, pd.DataFrame)
    pd.testing.assert_frame_equal(output_df, expected_df)

    # Assertion for write_precision_recall_data()
    write_prec_recall_data_args, _ = mock_write_precision_recall_data.call_args
    np.testing.assert_array_almost_equal(write_prec_recall_data_args[0], precision_values)
    np.testing.assert_array_almost_equal(write_prec_recall_data_args[1], recall_values)
    np.testing.assert_array_almost_equal(write_prec_recall_data_args[2], threshold_values)
    assert (
        write_prec_recall_data_args[3] == OUTPUT_PATH / "val_precision_recall_curve.csv"
    )

    # Assertion for plot_precision_recall_curve()
    plot_prec_recall_curve_args, plot_prec_recall_curve_kwargs = (
        mock_plot_precision_recall_curve.call_args
    )
    np.testing.assert_array_almost_equal(plot_prec_recall_curve_args[0], precision_values)
    np.testing.assert_array_almost_equal(plot_prec_recall_curve_args[1], recall_values)
    np.testing.assert_array_almost_equal(plot_prec_recall_curve_args[2], threshold_values)
    assert plot_prec_recall_curve_kwargs["n_positive_samples"] == n_positive_samples
    assert plot_prec_recall_curve_kwargs["n_samples"] == n_samples
    assert plot_prec_recall_curve_kwargs["basename"] == str(OUTPUT_PATH / "val")
