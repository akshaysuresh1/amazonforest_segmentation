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
    VALIDATION_IMAGES_PREFIX,
    TEST_IMAGES_PREFIX,
)


@patch("amazon_seg_project.assets.data_products.load_labeled_data")
def test_training_data_absent(mock_labeled_data: MagicMock) -> None:
    """
    Check correct response of training_data() to an empty list of data files
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
def test_validation_data_absent(mock_labeled_data: MagicMock) -> None:
    """
    Verify response of validation_data() to an empty list of data files
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
def test_testdata_absent(mock_list_objects: MagicMock) -> None:
    """
    Evaluate response of test
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
