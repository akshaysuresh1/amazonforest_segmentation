"""
Unit tests for assets defined in amazon_seg_project/assets/data_products.py
"""

from unittest.mock import patch, MagicMock
import pytest
from amazon_seg_project.assets import (
    data_training,
    data_validation,
    data_test,
)
from amazon_seg_project.data_paths import (
    TRAINING_IMAGES_PREFIX,
    TRAINING_MASKS_PREFIX,
    VALIDATION_IMAGES_PREFIX,
    VALIDATION_MASKS_PREFIX,
    TEST_IMAGES_PREFIX,
)
from amazon_seg_project.resources import s3_resource, AMAZON_TIF_BUCKET


@patch("amazon_seg_project.assets.data_products.load_labeled_data")
def test_training_data_success(mock_load_labeled_data: MagicMock) -> None:
    """
    Test for successful execution of data_training()
    """
    # Mock outputs
    mock_train_image_files = ["train/images/file1.tif", "train/images/file2.tif"]
    mock_train_mask_files = ["train/masks/file1.tif", "train/masks/file2.tif"]
    mock_load_labeled_data.return_value = (
        mock_train_image_files,
        mock_train_mask_files,
    )

    # Call the test function.
    output_generator = data_training()
    # Exhaust the generator into a list.
    outputs = list(output_generator)  # type: ignore

    # Assertion for mocked load_labeled_data()
    mock_load_labeled_data.assert_called_once_with(
        s3_resource,
        AMAZON_TIF_BUCKET.get_value(),
        str(TRAINING_IMAGES_PREFIX),
        str(TRAINING_MASKS_PREFIX),
        ".tif",
        ".tif",
    )

    assert len(outputs) == 2, (
        f"Expected number of outputs = 2, Actual number of outputs = {len(outputs)}"
    )
    # First output element: train_image_files
    assert outputs[0].output_name == "train_image_files", (
        "The name of the first output is not 'train_image_files'."
    )
    assert outputs[0].value == mock_train_image_files, (
        f"Expected output: {mock_train_image_files}, Actual output: {outputs[0].value}'"
    )
    assert outputs[0].metadata.get("Count of images").value == len(
        mock_train_image_files
    ), f"""Expected image count = {len(mock_train_image_files)},
        Actual image count = {len(outputs[0].value)}"""

    # Second output element: train_mask_files
    assert outputs[1].output_name == "train_mask_files", (
        "The name of the second output is not 'train_mask_files'."
    )
    assert outputs[1].value == mock_train_mask_files, (
        f"Expected output: {mock_train_mask_files}, Actual output: {outputs[1].value}'"
    )
    assert outputs[1].metadata.get("Count of masks").value == len(
        mock_train_mask_files
    ), f"""Expected image count = {len(mock_train_mask_files)},
        Actual image count = {len(outputs[1].value)}"""


@patch("amazon_seg_project.assets.data_products.load_labeled_data")
def test_training_data_absent(mock_labeled_data: MagicMock) -> None:
    """
    Verify correct response of data_training() to an empty list of data files
    """
    with pytest.raises(
        ValueError,
        match=f"Zero .tif images with masks found at {str(TRAINING_IMAGES_PREFIX)}",
    ):
        # Set up empty list of data files and mask files.
        mock_labeled_data.return_value = ([], [])

        # Call the test function.
        generator_outputs = data_training()
        # Exhaust the generator via a list.
        _ = list(generator_outputs)  # type: ignore


@patch("amazon_seg_project.assets.data_products.load_labeled_data")
def test_validation_data_success(mock_load_labeled_data: MagicMock) -> None:
    """
    Test for successful execution of data_validation()
    """
    # Mock outputs
    mock_val_image_files = ["val/images/file1.tif", "val/images/file2.tif"]
    mock_val_mask_files = ["val/masks/file1.tif", "val/masks/file2.tif"]
    mock_load_labeled_data.return_value = (
        mock_val_image_files,
        mock_val_mask_files,
    )

    # Call the test function.
    output_generator = data_validation()
    # Exhaust the generator into a list.
    outputs = list(output_generator)  # type: ignore

    # Assertion for mocked load_labeled_data()
    mock_load_labeled_data.assert_called_once_with(
        s3_resource,
        AMAZON_TIF_BUCKET.get_value(),
        str(VALIDATION_IMAGES_PREFIX),
        str(VALIDATION_MASKS_PREFIX),
        ".tif",
        ".tif",
    )

    assert len(outputs) == 2, (
        f"Expected number of outputs = 2, Actual number of outputs = {len(outputs)}"
    )
    # First output element: train_image_files
    assert outputs[0].output_name == "val_image_files", (
        "The name of the first output is not 'val_image_files'."
    )
    assert outputs[0].value == mock_val_image_files, (
        f"Expected output: {mock_val_image_files}, Actual output: {outputs[0].value}'"
    )
    assert outputs[0].metadata.get("Count of images").value == len(
        mock_val_image_files
    ), f"""Expected image count = {len(mock_val_image_files)},
        Actual image count = {len(outputs[0].value)}"""

    # Second output element: train_mask_files
    assert outputs[1].output_name == "val_mask_files", (
        "The name of the second output is not 'val_mask_files'."
    )
    assert outputs[1].value == mock_val_mask_files, (
        f"Expected output: {mock_val_mask_files}, Actual output: {outputs[1].value}'"
    )
    assert outputs[1].metadata.get("Count of masks").value == len(
        mock_val_mask_files
    ), f"""Expected image count = {len(mock_val_mask_files)},
        Actual image count = {len(outputs[1].value)}"""


@patch("amazon_seg_project.assets.data_products.load_labeled_data")
def test_validation_data_absent(mock_labeled_data: MagicMock) -> None:
    """
    Verify response of data_validation() to an empty list of data files
    """
    with pytest.raises(
        ValueError,
        match=f"Zero .tif images with masks found at {str(VALIDATION_IMAGES_PREFIX)}",
    ):
        # Set up empty list of data files and mask files.
        mock_labeled_data.return_value = ([], [])

        # Call the test function.
        generator_outputs = data_validation()
        # Exhaust the generator via a list.
        _ = list(generator_outputs)  # type: ignore


@patch("amazon_seg_project.assets.data_products.list_objects")
def test_testdata_success(mock_list_objects: MagicMock) -> None:
    """
    Test for successful execution of data_test()
    """
    mock_test_image_files = ["test/images/file1.tif", "test/images/file2.tif"]
    mock_list_objects.return_value = mock_test_image_files

    # Call the test function.
    output_generator = data_test()
    # Exhaust the generator into a list.
    output = list(output_generator)  # type: ignore

    # Assertion for mocked list_objects()
    mock_list_objects.assert_called_once_with(
        s3_resource, AMAZON_TIF_BUCKET.get_value(), str(TEST_IMAGES_PREFIX), ".tif"
    )

    # Assertion for output
    assert len(output) == 1, f"Expected 1 output, found {len(output)} output(s)."
    assert (
        output[0].value == mock_test_image_files
    ), f"""Expected output: {mock_test_image_files},
        Actual output: {output[0].value}"""
    assert output[0].metadata.get("Count of images").value == len(
        mock_test_image_files
    ), f"""Expected image count = {len(mock_test_image_files)},
        Actual image count = {len(output[0].value)}"""


@patch("amazon_seg_project.assets.data_products.list_objects")
def test_testdata_absent(mock_list_objects: MagicMock) -> None:
    """
    Evaluate response of data_test() to an empty list of image files
    """
    with pytest.raises(
        ValueError,
        match=f"Zero .tif images found at {str(TEST_IMAGES_PREFIX)}",
    ):
        # Set up empty list of test image files.
        mock_list_objects.return_value = []

        # Call the test function.
        generator_outputs = data_test()
        # Exhaust the generator via a list.
        _ = list(generator_outputs)  # type: ignore
