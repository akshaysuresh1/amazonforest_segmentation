"""
Modules for handling multispectral data stored in .tif files
"""

from io import BytesIO
import boto3
import numpy as np
import rioxarray as rxr
import xarray as xr
from .s3_utils import initialize_s3_client


def simulate_mock_multispec_data(
    N_bands: int, N_y: int, N_x: int, bit_depth: int = 16
) -> xr.DataArray:
    """
    Simulate a mock multispectral data array. Pixel value are unsigned integers with 8, 16, 32, or 64 bits.

    Args:
        N_bands: No. of spectral channels
        N_y: No. of samples along vertical or y-axish
        N_x: No. of samples along orizontal or x-axis
        bit_depth: No. of bits per data sample

    Returns: Multispectral data array of shape (N_bands, N_y, N_x)

    Raises:
        ValueError: If any of the input parameters are not integers or are less than 1.
    """
    bitdepths_to_dtype = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
    if not isinstance(N_bands, int) or N_bands < 1:
        raise ValueError(
            "Number of spectral channels must be an integer greater than 0."
        )
    if not isinstance(N_y, int) or N_y < 1:
        raise ValueError(
            "Number of pixels along y-dimension must be an integer greater than 0."
        )
    if not isinstance(N_x, int) or N_x < 1:
        raise ValueError(
            "Number of pixels along x-dimension must be an integer greater than 0."
        )
    if not isinstance(bit_depth, int) or (bit_depth not in bitdepths_to_dtype):
        raise ValueError(
            f"Unsupported bit depth: {bit_depth}. Supported values are integers 8, 16, 32, and 64."
        )

    # Simulate mock data as 3d numpy array of shape (N_bands, N_y, N_x).
    data_type = bitdepths_to_dtype[bit_depth]
    data_vals = np.random.randint(
        low=0,
        high=np.iinfo(data_type).max + 1,
        size=(N_bands, N_y, N_x),
        dtype=data_type,
    )

    # Create coordinate arrays
    band_coords = np.arange(N_bands)
    y_coords = np.arange(N_y)
    x_coords = np.arange(N_x)

    # Build DataArray.
    dims = ("band", "y", "x")
    coords = {"band": band_coords, "y": y_coords, "x": x_coords}
    data_array = xr.DataArray(data_vals, dims=dims, coords=coords)
    return data_array


def load_tif_from_s3(object_key: str, bucket: str, s3_client: boto3.client) -> xr.DataArray:
    """
    Load multispectral data from .tif file accessed via an S3 object key

    Args:
        object_key: Object key of .tif file stored in S3 bucket
        bucket: Name of S3 bucket
        s3_client: S3 client

    Returns: Multispectral dataset

    Raises:
        ValueError: If the object key does not end with the ".tif" file extension
        ValueError: If dataset stored in object is empty
        AttributeError: If object key does not exist in S3 bucket
        ValueError: If opening GeoTIFF fails for any reason
    """
    if not object_key.endswith(".tif"):
        raise ValueError("File extension does not correspond to a GeoTIFF file.")

    try:
        response = s3_client.get_object(Bucket=bucket, Key=object_key)
    except Exception:
        raise AttributeError(f"Failed to fetch {object_key} from S3.")

    if "Body" in response:
        tif_bytes = response["Body"].read()
        if not tif_bytes:
            raise ValueError(f"Empty dataset in {object_key}")
        dataset = rxr.open_rasterio(BytesIO(tif_bytes))
    else:
        raise AttributeError(f"{object_key} not found in {bucket}.")

    return dataset
