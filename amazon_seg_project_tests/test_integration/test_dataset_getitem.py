"""
Integration tests for the __getitem__() method of SegmentationDataset
"""

from unittest.mock import patch, MagicMock
from io import BytesIO
import pytest
import numpy as np
import boto3
from moto import mock_aws
from amazon_seg_project.ops import simulate_mock_multispec_data
from amazon_seg_project.assets import SegmentationDataset
from amazon_seg_project.resources import torch


@mock_aws
@patch("dagster_aws.s3.S3Resource")
def test_seg_dataset_getitem_unequal_row_counts(mock_s3_resource: MagicMock) -> None:
    """
    Test the __getitem__() method for an image-mask pair with unequal row counts.
    """
    index = 0  # Index of file to simulate in images_list

    with pytest.raises(
        ValueError,
        match=f"Index {index}: The image and the mask have unequal row counts.",
    ):
        # Create a mock S3 client.
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_bucket = "test-bucket"
        images_list = ["data/image1.tif"]
        masks_list = ["data/mask1.tif"]

        # Create the bucket.
        s3_client.create_bucket(Bucket=s3_bucket)

        # Create and upload image to S3 bucket.
        image_ds = simulate_mock_multispec_data(n_bands=4, n_y=64, n_x=64)
        # Convert dataset to GeoTiff in memory.
        buffer = BytesIO()
        image_ds.rio.to_raster(buffer, driver="GTiff")
        buffer.seek(0)  # Reset buffer position
        # Upload GeoTiff to S3.
        s3_client.put_object(
            Bucket=s3_bucket, Key=images_list[index], Body=buffer.getvalue()
        )

        # Create and upload segmentation mask to S3 bucket.
        mask_ds = simulate_mock_multispec_data(n_bands=1, n_y=32, n_x=64)
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

        # Create the SegmentationDataset object.
        seg_dataset = SegmentationDataset(
            images_list, masks_list, mock_s3_resource, s3_bucket
        )

        # Call the __getitem__() method.
        _ = seg_dataset[index]


@mock_aws
@patch("dagster_aws.s3.S3Resource")
def test_seg_dataset_getitem_unequal_column_counts(mock_s3_resource: MagicMock) -> None:
    """
    Test the __getitem__() method for an image-mask pair with unequal column counts.
    """
    index = 0  # Index of file to simulate in images_list

    with pytest.raises(
        ValueError,
        match=f"Index {index}: The image and the mask have unequal column counts.",
    ):
        # Create a mock S3 client.
        s3_client = boto3.client("s3", region_name="us-east-1")
        s3_bucket = "test-bucket"
        images_list = ["data/image1.tif"]
        masks_list = ["data/mask1.tif"]

        # Create the bucket.
        s3_client.create_bucket(Bucket=s3_bucket)

        # Create and upload image to S3 bucket.
        image_ds = simulate_mock_multispec_data(n_bands=4, n_y=64, n_x=64)
        # Convert dataset to GeoTiff in memory.
        buffer = BytesIO()
        image_ds.rio.to_raster(buffer, driver="GTiff")
        buffer.seek(0)  # Reset buffer position
        # Upload GeoTiff to S3.
        s3_client.put_object(
            Bucket=s3_bucket, Key=images_list[index], Body=buffer.getvalue()
        )

        # Create and upload segmentation mask to S3 bucket.
        mask_ds = simulate_mock_multispec_data(n_bands=1, n_y=64, n_x=32)
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

        # Create the SegmentationDataset object.
        seg_dataset = SegmentationDataset(
            images_list, masks_list, mock_s3_resource, s3_bucket
        )

        # Call the __getitem__() method.
        _ = seg_dataset[index]


@mock_aws
@patch("dagster_aws.s3.S3Resource")
@patch(
    "amazon_seg_project.assets.dataset_definition.robust_scaling",
    side_effect=lambda x: x,
)
def test_seg_dataset_getitem_valid_without_aug(
    mock_scaling: MagicMock, mock_s3_resource: MagicMock
) -> None:
    """
    Test response of __getitem__() to valid inputs without data augmentation.

    Here, the function robust_scaling() is mocked as the identity operator.
    """
    index = 0  # Index of file to simulate in images_list
    images_list = ["data/image1.tif"]
    masks_list = ["data/mask1.tif"]

    # Create mock S3 client.
    s3_client = boto3.client("s3", region_name="us-east-1")

    # Create S3 bucket.
    s3_bucket = "test-bucket"
    s3_client.create_bucket(Bucket=s3_bucket)

    # Create and upload image data to S3.
    image_ds = simulate_mock_multispec_data(n_bands=4, n_y=64, n_x=64)
    # Convert dataset to GeoTiff in memory.
    buffer = BytesIO()
    image_ds.rio.to_raster(buffer, driver="GTiff")
    buffer.seek(0)  # Reset buffer position
    # Upload GeoTiff to S3.
    s3_client.put_object(
        Bucket=s3_bucket, Key=images_list[index], Body=buffer.getvalue()
    )

    # Create and upload segmentation mask to S3 bucket.
    mask_ds = simulate_mock_multispec_data(n_bands=1, n_y=64, n_x=64)
    # Convert dataset to GeoTiff in memory.
    buffer = BytesIO()
    mask_ds.rio.to_raster(buffer, driver="GTiff")
    buffer.seek(0)  # Reset buffer position
    # Upload GeoTiff to S3.
    s3_client.put_object(
        Bucket=s3_bucket, Key=masks_list[index], Body=buffer.getvalue()
    )

    # Connect the mock Dagster AWS S3Resource to boto3 client
    mock_s3_resource.get_client.return_value = s3_client

    # Expected results
    expected_image = torch.from_numpy(image_ds.to_numpy().astype(np.float64))
    expected_mask = torch.from_numpy(mask_ds.to_numpy().astype(np.float64))

    # Create a SegmentationDataset object and call its test method.
    seg_dataset = SegmentationDataset(
        images_list, masks_list, mock_s3_resource, s3_bucket
    )
    output_image, output_mask = seg_dataset[index]

    # Assert that robust_scaling() was called once.
    mock_scaling.assert_called_once()

    # Verify that outputs match expected results.
    assert torch.equal(
        output_image, expected_image
    ), "Output image does not match expected image."
    assert torch.equal(
        output_mask, expected_mask
    ), "Output mask does not match expected mask."
