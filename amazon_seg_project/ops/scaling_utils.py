"""
Utility functions for data scaling
"""

import numpy as np


def robust_scaling(
    data: np.ndarray,
    means: np.ndarray = np.array([622.59, 683.14, 436.69, 2951.97]),
    sigma: np.ndarray = np.array([541.08, 368.43, 342.26, 633.47]),
) -> np.ndarray:
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
