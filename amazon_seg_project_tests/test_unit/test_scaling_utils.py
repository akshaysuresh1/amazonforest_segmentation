"""
Unit tests for modules defined in amazon_seg_project/ops/scaling_utils.py
"""

import pytest
import numpy as np
from amazon_seg_project.ops.scaling_utils import robust_scaling, min_max_scaling


def test_robust_scaling_success() -> None:
    """
    Test successful execution of robust_scaling().
    """
    data = 4.0 * (np.random.randn(64, 64, 4) + 1000.0)  # pylint: disable=E1101
    means = np.mean(data, axis=(0, 1))
    sigma = np.std(data, axis=(0, 1))
    expected = (data - means[np.newaxis, np.newaxis, :]) / sigma[
        np.newaxis, np.newaxis, :
    ]

    # Call the test function.
    result = robust_scaling(data, means, sigma)

    # Use numpy's testing assert to compare arrays for exact equality.
    np.testing.assert_equal(result, expected)


def test_robust_scaling_invalid_data() -> None:
    """
    Test response of robust_scaling() to data.ndim != 3.
    """
    with pytest.raises(ValueError, match="Input data array must be 3-dimensional."):
        data = np.random.randn(64, 64)  # pylint: disable=E1101
        robust_scaling(data)


def test_robust_scaling_unequal_channels() -> None:
    """
    Test response of robust_scaling() to inputs with different channel counts.
    """
    with pytest.raises(
        ValueError,
        match="Data, means, and standard deviation arrays have unequal number of color channels.",
    ):
        data = np.random.randn(64, 64, 3)  # pylint: disable=E1101
        means = np.zeros(4)
        sigma = np.ones(5)

        robust_scaling(data, means, sigma)


def test_minmax_scaling_success() -> None:
    """
    Test successful execution of min_max_scaling().
    """
    # Create a 3D numpy array (2x2x3) with known values.
    data = np.array(
        [[[1, 10, 100], [2, 20, 200]], [[3, 30, 300], [4, 40, 400]]], dtype=float
    )
    # Expected min and max per channel
    # Channel 0: min=1, max=4, range=3
    # Channel 1: min=10, max=40, range=30
    # Channel 2: min=100, max=400, range=300
    expected_result = np.array(
        [[[0, 0, 0], [1 / 3, 1 / 3, 1 / 3]], [[2 / 3, 2 / 3, 2 / 3], [1, 1, 1]]],
        dtype=float,
    )

    # Call the test function.
    output = min_max_scaling(data)

    # Assert for closeness between output and expected result.
    np.testing.assert_allclose(output, expected_result, rtol=1e-6, atol=1e-8)


def test_minmax_scaling_invalid_data() -> None:
    """
    Test response of min_max_scaling() to data.ndim != 3.
    """
    with pytest.raises(ValueError, match="Input data array must be 3-dimensional."):
        data = np.random.randn(32, 32)  # pylint: disable=E1101
        min_max_scaling(data)
