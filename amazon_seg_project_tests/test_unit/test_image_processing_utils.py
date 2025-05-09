"""
Unit tests for ops defined in image_processing_utils.py
"""

import pytest
import numpy as np
from amazon_seg_project.ops.image_processing_utils import compute_ndvi, apply_mask


def test_compute_ndvi_success() -> None:
    """
    Test for correct computation of NDVI for valid inputs.
    """
    #  Input arrays
    nir_data = np.array([[0.8, 0.6], [0.4, 0.2]])
    red_data = np.array([[0.4, 0.2], [0.2, 0.1]])
    eps = 1.0e-9
    # Expected NDVI array
    expected_ndvi = (nir_data - red_data) / (nir_data + red_data + eps)

    # Call the test function.
    result = compute_ndvi(nir_data, red_data, eps=eps)

    # Assertion
    np.testing.assert_allclose(result, expected_ndvi, rtol=1e-7, atol=1e-10)


def test_compute_ndvi_invalid_nir_data_dim() -> None:
    """
    Check for correct raise of ValueError when nir_data.dim != 2.
    """
    with pytest.raises(
        ValueError, match="NIR raster data must be a 2-dimensional array."
    ):
        # Input arrays
        nir_data = np.random.randn(16, 16, 3)
        red_data = np.random.randn(16, 16)

        # Call the test function.
        _ = compute_ndvi(nir_data, red_data)


def test_compute_ndvi_invalid_red_data_dim() -> None:
    """
    Check for correct raise of ValueError when red_data.dim != 2.
    """
    with pytest.raises(
        ValueError, match="Red channel raster data must be a 2-dimensional array."
    ):
        # Input arrays
        nir_data = np.random.randn(16, 16)
        red_data = np.random.randn(16, 16, 3)

        # Call the test function.
        _ = compute_ndvi(nir_data, red_data)


def test_compute_ndvi_unequal_input_data_shapes() -> None:
    """
    Check for correct raise of ValueError when red_data.shape != nir_data.shape.
    """
    with pytest.raises(
        ValueError,
        match="Supplied data for NIR and Red channels must have the same shape.",
    ):
        # Input arrays
        nir_data = np.random.randn(16, 16)
        red_data = np.random.randn(32, 32)

        # Call the test function.
        _ = compute_ndvi(nir_data, red_data)


def test_apply_mask_success() -> None:
    """
    Test to verify correct execution of apply_mask().
    """
    # Input arrays
    img = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    binary_mask = np.array(([1, 0], [0, 1]))
    # Expected output
    expected_masked_img = np.array([[[1, 2, 3], [0, 0, 0]], [[0, 0, 0], [10, 11, 12]]])

    # Call the test function.
    result = apply_mask(img, binary_mask)

    # Assertion
    np.testing.assert_allclose(result, expected_masked_img)


def test_apply_mask_invalid_img_dim() -> None:
    """
    Check for correct raise of ValueError when img.ndim != 3.
    """
    with pytest.raises(
        ValueError, match="Multispectral image data must be 3-dimensional."
    ):
        img = np.random.randn(16, 16)
        binary_mask = np.random.randint(low=0, high=2, size=(16, 16))

        # Call the test function.
        _ = apply_mask(img, binary_mask)


def test_apply_mask_invalid_mask_shape() -> None:
    """
    Check for correct raise of ValueError upon encountering invalid mask shape.
    """
    with pytest.raises(ValueError, match=""):
        img = np.random.randn(16, 16, 3)
        binary_mask = np.random.randint(low=0, high=2, size=(32, 32))

        # Call the test function.
        _ = apply_mask(img, binary_mask)
