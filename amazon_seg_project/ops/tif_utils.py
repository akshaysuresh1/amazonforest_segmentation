"""
Modules to read/write multispectral data stored in .tif files
"""

from io import BytesIO
from typing import List
import numpy as np
import rioxarray as rxr
import xarray as xr
from dagster import op, Out, Any
from dagster_aws.s3 import S3Resource


@op(out=Out(Any))
def simulate_mock_multispec_data(
    n_bands: int, n_y: int, n_x: int, bit_depth: int = 16
) -> xr.DataArray:
    """
    Simulate a mock multispectral data array.
    Pixel value are unsigned integers with 8, 16, 32, or 64 bits.

    Args:
        n_bands: No. of spectral channels
        n_y: No. of samples along the vertical or y-axis
        n_x: No. of samples along the horizontal or x-axis
        bit_depth: No. of bits per data sample

    Returns: Multispectral data array of shape (n_bands, n_y, n_x)

    Raises:
        ValueError: If any of the input parameters are not integers or are less than 1.
    """
    bitdepths_to_dtype = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
    if not isinstance(n_bands, int) or n_bands < 1:
        raise ValueError(
            "Number of spectral channels must be an integer greater than 0."
        )
    if not isinstance(n_y, int) or n_y < 1:
        raise ValueError(
            "Number of pixels along y-dimension must be an integer greater than 0."
        )
    if not isinstance(n_x, int) or n_x < 1:
        raise ValueError(
            "Number of pixels along x-dimension must be an integer greater than 0."
        )
    if not isinstance(bit_depth, int) or (bit_depth not in bitdepths_to_dtype):
        raise ValueError(
            f"""
        Unsupported bit depth: {bit_depth}.
        Supported values are integers 8, 16, 32, and 64."""
        )

    # Simulate mock data as 3d numpy array of shape (n_bands, n_y, n_x).
    data_type = bitdepths_to_dtype[bit_depth]
    data_vals = np.random.randint(  # pylint: disable=no-member
        low=0,
        high=np.iinfo(data_type).max + 1,  # type: ignore
        size=(n_bands, n_y, n_x),
        dtype=data_type,  # type: ignore
    )

    # Create coordinate arrays.
    band_coords = np.arange(n_bands)
    y_coords = np.arange(n_y)
    x_coords = np.arange(n_x)

    # Build DataArray.
    dims = ("band", "y", "x")
    coords = {"band": band_coords, "y": y_coords, "x": x_coords}
    data_array = xr.DataArray(data_vals, dims=dims, coords=coords)
    return data_array


@op(out=Out(Any))
def load_tif_from_s3(
    s3_resource: S3Resource, s3_bucket: str, object_key: str
) -> xr.DataArray | xr.Dataset | List[xr.Dataset]:
    """
    Load multispectral data from .tif file accessed via an S3 object key

    Args:
        s3_resource: Dagster-AWS S3 resource
        s3_bucket: Name of S3 bucket
        object_key: Object key of .tif file stored in S3 bucket

    Returns: xarray DataArray object

    Raises:
        ValueError: If the object key does not end with the ".tif" file extension
        ValueError: If dataset stored in object is empty
        AttributeError: If object key does not exist in S3 bucket
        ValueError: If opening GeoTIFF fails for any reason
    """
    if not object_key.endswith(".tif"):
        raise ValueError("File extension does not correspond to a GeoTIFF file.")

    try:
        response = s3_resource.get_client().get_object(Bucket=s3_bucket, Key=object_key)
    except Exception as exc:
        raise AttributeError(f"Failed to fetch {object_key} from S3.") from exc

    if "Body" in response:
        tif_bytes = response.get("Body").read()
        if not tif_bytes:
            raise ValueError(f"Empty dataset in {object_key}")
        dataset = rxr.open_rasterio(BytesIO(tif_bytes))
    else:
        raise AttributeError(f"{object_key} not found in {s3_bucket}.")

    return dataset
