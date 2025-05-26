"""
Unit tests for performance metrics computed on test dataset
"""

from unittest.mock import patch, MagicMock, call, ANY
from io import BytesIO
import numpy as np
import boto3
from moto import mock_aws
from segmentation_models_pytorch import Unet
from amazon_seg_project.assets import (
    afs_test_dataset_metrics,
    SegmentationDataset,
)
from amazon_seg_project.config import ModelEvaluationConfig
from amazon_seg_project.ops.tif_utils import (
    simulate_mock_multispec_data,
    simulate_mock_binary_mask,
)
from amazon_seg_project.data_paths import OUTPUT_PATH


@mock_aws
@patch("amazon_seg_project.assets.performance_metrics_test_dataset.write_dict_to_csv")
@patch(
    "amazon_seg_project.assets.performance_metrics_test_dataset.visualize_and_save_model_predictions"
)
@patch("amazon_seg_project.assets.performance_metrics_test_dataset.smp_metrics")
@patch("amazon_seg_project.assets.performance_metrics_test_dataset.create_directories")
@patch("amazon_seg_project.assets.dataset_definition.s3_resource")
@patch("logging.info")
def test_afs_test_dataset_metrics_computation(
    mock_logging: MagicMock,
    mock_s3_resource: MagicMock,
    mock_create_directories: MagicMock,
    mock_smp_metrics: MagicMock,
    mock_plot_model_predictions: MagicMock,
    mock_write_dict_to_csv: MagicMock,
) -> None:
    """
    Test successful computation of test dataset metrics using mocked test dataset.
    """
    # Multispectral image data properties
    color_channels = 4  # No. of color channels
    img_height = 64  # pixel count
    img_width = 64  # pixel count

    # Set up test config and dummy model.
    test_config = ModelEvaluationConfig(threshold=0.73)
    model = Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=color_channels,
        activation="sigmoid",
    )

    # Inputs to mock test dataset
    s3_bucket = "mock-bucket"
    images_list = ["images/file1.tif", "images/file2.tif", "images/file3.tif"]
    masks_list = ["masks/file1.tif", "masks/file2.tif", "masks/file3.tif"]

    # Create mock S3 client.
    s3_client = boto3.client("s3", region_name="us-east-1")
    # Create S3 bucket.
    s3_client.create_bucket(Bucket=s3_bucket)

    for index, image_file in enumerate(images_list):
        # Create and upload image data to S3.
        image_ds = simulate_mock_multispec_data(
            n_bands=4, n_y=img_height, n_x=img_width
        )
        # Convert dataset to GeoTiff in memory.
        buffer = BytesIO()
        image_ds.rio.to_raster(buffer, driver="GTiff")
        buffer.seek(0)  # Reset buffer position
        # Upload GeoTiff to S3.
        s3_client.put_object(Bucket=s3_bucket, Key=image_file, Body=buffer.getvalue())

        # Create and upload segmentation mask to S3 bucket.
        mask_ds = simulate_mock_binary_mask(n_y=img_height, n_x=img_width)
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

    # Create a SegmentationDataset object for mock test dataset.
    test_dataset = SegmentationDataset(images_list, masks_list, s3_bucket)

    # Return fixed outputs from mocked functions for metric computation
    mock_smp_metrics.return_value = {
        "Precision": 0.6,
        "Recall": 0.4,
        "F1 score": 0.48,
        "IoU": 0.27,
        "Accuracy": 0.73,
    }

    # Call the test function.
    output_dict = afs_test_dataset_metrics(test_config, test_dataset, model)

    # Assertions for output dictionary
    assert isinstance(output_dict, dict)
    assert output_dict.keys() == mock_smp_metrics.return_value.keys()
    for key in output_dict:
        np.testing.assert_array_equal(
            np.array([mock_smp_metrics.return_value[key]] * len(test_dataset)),
            output_dict[key],
        )

    # Assertion for output base path creation
    mock_create_directories.assert_called_once_with(
        OUTPUT_PATH / "test_dataset_plots" / "test_data_index"
    )

    # Assertions for calls to visualize_and_save_model_predictions()
    mock_plot_model_predictions.assert_called_with(
        ANY,  # image_plot
        ANY,  # ground_truth_mask_plot
        ANY,  # predicted_mask_plot
        basename=str(OUTPUT_PATH / "test_dataset_plots" / f"test_data_index{index:03d}"),
        accuracy=mock_smp_metrics.return_value.get("Accuracy"),
        precision=mock_smp_metrics.return_value.get("Precision"),
        recall=mock_smp_metrics.return_value.get("Recall"),
        iou_value=mock_smp_metrics.return_value.get("IoU"),
    )

    # Assertion for write_dict_to_csv()
    mock_write_dict_to_csv.assert_called_once_with(
        output_dict,
        str(
            OUTPUT_PATH
            / f"test_dataset_metrics_threshold_{test_config.threshold:.2f}.csv"
        ),
    )
    # Logging assertions
    calls = [
        call(f"Computing metrics over test dataset of size {len(test_dataset)}"),
        call("Metrics computed for test dataset."),
    ]
    mock_logging.assert_has_calls(calls)
