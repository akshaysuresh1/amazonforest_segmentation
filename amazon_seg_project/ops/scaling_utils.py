"""
Utility functions for data scaling
"""

import numpy as np
import numpy.typing as npt
from dagster import op, In, Out
from dagster import Any as dg_Any
from ..resources import ScalarTypeT


# Avoid using the op decorator here to allow for module pickling.
def robust_scaling(
    data: npt.NDArray[ScalarTypeT],
    means: npt.NDArray[ScalarTypeT] = np.array([622.59, 683.14, 436.69, 2951.97]),
    sigma: npt.NDArray[ScalarTypeT] = np.array([541.08, 368.43, 342.26, 633.47]),
) -> npt.NDArray[ScalarTypeT]:
    """
    Normalize image data using input channel-wise means and standard deviations.

    Default values of "means" and "sigma" are pre-computed estimates from training data.
    Link to training data: https://zenodo.org/records/4498086

    Args:
        data: 3D numpy array of shape (n_y, n_x, n_bands)
        means: Per-channel mean pixel values, array shape = (n_bands,)
        sigma: Per-channel standard deviation values, array shape = (n_bands,)

    Returns: Normalized data array
    """
    if data.ndim != 3:
        raise ValueError("Input data array must be 3-dimensional.")

    if not data.shape[-1] == len(means) == len(sigma):
        raise ValueError(
            "Data, means, and standard deviation arrays have unequal number of color channels."
        )
    # Normalization = (data - mean) / standard deviation
    normalized_data = (data - means[np.newaxis, np.newaxis, :]) / sigma[
        np.newaxis, np.newaxis, :
    ]
    return normalized_data


@op(ins={"data": In(dg_Any)}, out=Out(dg_Any))
def min_max_scaling(data: npt.NDArray[ScalarTypeT]) -> npt.NDArray[ScalarTypeT]:
    """
    Apply min-maxing scaling on a per-channel basis for image normalization to [0, 1].

    Args:
        data: 3D numpy array of shape (n_y, n_x, n_bands)

    Returns: Normalized data array
    """
    if data.ndim != 3:
        raise ValueError("Input data array must be 3-dimensional.")

    # Evaluate maximum and minimum of data values per color channel.
    min_vals = np.nanmin(data, axis=(0, 1))
    max_vals = np.nanmax(data, axis=(0, 1))
    range_vals = max_vals - min_vals

    # Perform min-max scaling separately for each color channel.
    normalized_data = (data - min_vals[np.newaxis, np.newaxis, :]) / range_vals[
        np.newaxis, np.newaxis, :
    ]
    return normalized_data
