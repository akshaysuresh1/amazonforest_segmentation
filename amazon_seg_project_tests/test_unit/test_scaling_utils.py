"""
Unit tests for modules defined in amazon_seg_project/ops/scaling_utils.py
"""

import pytest
import numpy as np
from amazon_seg_project.ops.scaling_utils import robust_scaling


def test_robust_scaling_success() -> None:
    """
    Test successful execution of robust_scaling()
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
    Test response of robust_scaling() to data.ndim != 3
    """
    with pytest.raises(ValueError, match="Input data array must be 3-dimensional."):
        data = np.random.randn(64, 64)  # pylint: disable=E1101
        robust_scaling(data)


def test_robust_scaling_unequal_channels() -> None:
    """
    Test response of robust_scaling() to inputs with different channel counts
    """
    with pytest.raises(
        ValueError,
        match="Data, means, and standard deviation arrays have unequal number of color channels.",
    ):
        data = np.random.randn(64, 64, 3)  # pylint: disable=E1101
        means = np.zeros(4)
        sigma = np.ones(5)

        robust_scaling(data, means, sigma)
