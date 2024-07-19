"""
Modules for handling multispectral data stored in .tif files
"""

import rioxarray as rxr
from io import BytesIO
import boto3
import xarray as xr


def load_tif_from_s3(
    object_key: str, bucket: str, s3_client: boto3.client
) -> xr.DataArray:
    """
    Load multispectral data from .tif file accessed via an S3 object key

    Args:
        object_key (str): Object key of .tif file stored in S3 bucket
        bucket (str): Bucket name
        s3_client (boto3.client): S3 client

    Returns:
        xr.DataArray: Multispectral dataset

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
