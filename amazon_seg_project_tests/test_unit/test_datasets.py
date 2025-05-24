"""
Unit tests for assets defined in amazon_seg_project.assets.datasets
"""

from unittest.mock import patch, MagicMock
from amazon_seg_project.assets import (
    afs_training_dataset,
    afs_validation_dataset,
    afs_test_dataset,
)
from amazon_seg_project.ops.scaling_utils import robust_scaling
from amazon_seg_project.resources import AMAZON_TIF_BUCKET
from amazon_seg_project.config import TrainingDatasetConfig


@patch("amazon_seg_project.assets.datasets.SegmentationDataset")
@patch("amazon_seg_project.assets.datasets.get_aug_pipeline")
def test_afs_training_dataset_success(
    mock_get_aug_pipeline: MagicMock, mock_seg_dataset: MagicMock
) -> None:
    """
    Test successful initialization of training dataset.
    """
    # Define mock config and inputs.
    config = TrainingDatasetConfig(
        horizontal_flip_prob=0.7,
        vertical_flip_prob=0.2,
        rotate90_prob=0.4,
        augmentation_seed=45,
    )
    mock_train_image_files = ["train/images/file1.tif", "train/images/file2.tif"]
    mock_train_mask_files = ["train/masks/file1.tif", "train/masks/file2.tif"]
    # Set up mock outputs.
    mock_train_dataset = MagicMock()
    mock_train_dataset.__len__.return_value = len(mock_train_image_files)
    mock_seg_dataset.return_value = mock_train_dataset
    mock_get_aug_pipeline.return_value = "mocked_transform_function"

    # Call the test function.
    output_generator = afs_training_dataset(
        config, mock_train_image_files, mock_train_mask_files
    )
    # Exhaust the generator into a list.
    output = list(output_generator)  # type: ignore

    # Assertion for mocked get_aug_pipeline()
    mock_get_aug_pipeline.assert_called_once_with(
        horizontal_flip_prob=config.horizontal_flip_prob,
        vertical_flip_prob=config.vertical_flip_prob,
        rotate90_prob=config.rotate90_prob,
        augmentation_seed=config.augmentation_seed,
    )

    # Assertion for mocked SegmentationDataset class
    mock_seg_dataset.assert_called_once_with(
        images_list=mock_train_image_files,
        masks_list=mock_train_mask_files,
        s3_bucket=AMAZON_TIF_BUCKET.get_value() or "",
        scaling_func=robust_scaling,
        transform=mock_get_aug_pipeline.return_value,
    )

    # Assertions for output
    assert len(output) == 1, f"Expected 1 output, found {len(output)} output(s)."
    assert output[0].value == mock_train_dataset
    assert output[0].metadata.get("Training dataset size").value == len(
        mock_train_dataset
    ), f"""Expected training dataset size = {len(mock_train_dataset)},
        Actual dataset size = {output[0].metadata.get("Training dataset size").value}"""


@patch("amazon_seg_project.assets.datasets.SegmentationDataset")
def test_afs_validation_dataset_success(mock_seg_dataset: MagicMock) -> None:
    """
    Test successful initialization of validation dataset.
    """
    # Define mock config and inputs.
    mock_val_image_files = ["val/images/file1.tif", "val/images/file2.tif"]
    mock_val_mask_files = ["val/masks/file1.tif", "val/masks/file2.tif"]
    # Set up mock output.
    mock_val_dataset = MagicMock()
    mock_val_dataset.__len__.return_value = len(mock_val_image_files)
    mock_seg_dataset.return_value = mock_val_dataset

    # Call the test function.
    output_generator = afs_validation_dataset(mock_val_image_files, mock_val_mask_files)
    # Exhaust the generator into a list.
    output = list(output_generator)  # type: ignore

    # Assertion for mocked SegmentationDataset class
    mock_seg_dataset.assert_called_once_with(
        images_list=mock_val_image_files,
        masks_list=mock_val_mask_files,
        s3_bucket=AMAZON_TIF_BUCKET.get_value() or "",
        scaling_func=robust_scaling,
        transform=None,
    )

    # Assertions for output
    assert len(output) == 1, f"Expected 1 output, found {len(output)} output(s)."
    assert output[0].value == mock_val_dataset
    assert output[0].metadata.get("Validation dataset size").value == len(
        mock_val_dataset
    ), f"""Expected validation dataset size = {len(mock_val_dataset)},
        Actual dataset size = {output[0].metadata.get("Validation dataset size").value}"""


@patch("amazon_seg_project.assets.datasets.SegmentationDataset")
def test_afs_test_dataset_success(mock_seg_dataset: MagicMock) -> None:
    """
    Verify successful initialization of test dataset.
    """
    # Define mock config and inputs.
    mock_test_image_files = ["test/images/file1.tif", "test/images/file2.tif"]
    mock_test_mask_files = ["test/masks/file1.tif", "test/masks/file2.tif"]
    # Set up mock output.
    mock_test_dataset = MagicMock()
    mock_test_dataset.__len__.return_value = len(mock_test_image_files)
    mock_seg_dataset.return_value = mock_test_dataset

    # Call the test function.
    output_generator = afs_test_dataset(mock_test_image_files, mock_test_mask_files)
    # Exhaust the generator into a list.
    output = list(output_generator)  # type: ignore

    # Assertion for mocked SegmentationDataset class
    mock_seg_dataset.assert_called_once_with(
        images_list=mock_test_image_files,
        masks_list=mock_test_mask_files,
        s3_bucket=AMAZON_TIF_BUCKET.get_value() or "",
        scaling_func=robust_scaling,
        transform=None,
    )

    # Assertions for output
    assert len(output) == 1, f"Expected 1 output, found {len(output)} output(s)."
    assert output[0].value == mock_test_dataset
    assert output[0].metadata.get("Test dataset size").value == len(
        mock_test_dataset
    ), f"""Expected test dataset size = {len(mock_test_dataset)},
        Actual dataset size = {output[0].metadata.get("Test dataset size").value}"""
