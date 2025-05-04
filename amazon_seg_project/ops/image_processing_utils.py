"""
Modules for image processing in Python
"""

import numpy as np
import numpy.typing as npt
from dagster import op, In, Out
from dagster import Any as dg_Any
from ..resources import ScalarTypeT


@op(
    ins={"nir_data": In(dg_Any), "red_data": In(dg_Any), "eps": In(float)},
    out=Out(dg_Any),
)
def compute_ndvi(
    nir_data: npt.NDArray[ScalarTypeT],
    red_data: npt.NDArray[ScalarTypeT],
    eps: float = 1.0e-10,
) -> npt.NDArray[ScalarTypeT]:
    """
    Compute the normalized difference vegetation index (NDVI).

    NDVI = (NIR - Red) / (NIR + Red)

    Args:
        nir_data: NIR raster data of shape (n_y, n_x)
        red_data: Red channel raster data of shape (n_y, n_x)
        eps: Small floating-point value to avoid division by zero

    Returns: NDVI raster of shape (n_y, n_x)
    """
    if nir_data.ndim != 2:
        raise ValueError("NIR raster data must be a 2-dimensional array.")

    if red_data.ndim != 2:
        raise ValueError("Red channel raster data must be a 2-dimensional array.")

    if nir_data.shape != red_data.shape:
        raise ValueError(
            "Supplied data for NIR and Red channels must have the same shape."
        )
    ndvi = (nir_data - red_data) / (nir_data + red_data + eps)
    return ndvi


@op(ins={"img": In(dg_Any), "bin_mask": In(dg_Any)}, out=Out(dg_Any))
def apply_mask(
    img: npt.NDArray[ScalarTypeT], bin_mask: npt.NDArray[ScalarTypeT]
) -> npt.NDArray[ScalarTypeT]:
    """
    Apply a binary mask on a multispectral image.

    Args:
        img: Image data of shape (n_y, n_x, n_bands)
        bin_mask: Binary mask of shape (n_y, n_x)

    Returns: Masked data array with same shape as input
    """
    if img.ndim != 3:
        raise ValueError("Multispectral image data must be 3-dimensional.")
    img_height, img_width, _ = img.shape

    if bin_mask.shape != (img_height, img_width):
        raise ValueError(
            "Binary mask does not have the same spatial grid as the image data."
        )

    masked_img = img * bin_mask[:, :, np.newaxis]
    return masked_img
