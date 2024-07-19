"""
Tests for modules defined in tif_utils.py
"""

import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
import xarray as xr
import numpy as np
from botocore.exceptions import ClientError
from botocore.response import StreamingBody
from amazon_seg_project.scripts.tif_utils import load_tif_from_s3


def test_load_tif_from_s3_success() -> None:
    """
    Test successful execution of load_tif_from_s3()
    """
    # Set parameters for mock xarray multispectral dataset.
    N_bands = np.random.randint(low=1, high=6)  # No. of spectral channels
    N_x = np.random.randint(low=5, high=20)  # No. of x samples
    N_y = np.random.randint(low=5, high=20)  # No. of y samples
    dims = {"band", "y", "x"}

    # Simulate mock xarray dataset.
    data = np.random.randn(N_bands, N_y, N_x)
    dataset = xr.DataArray(data, dims=dims)

    # Create mock S3 client.
    mock_s3_client = MagicMock()

    # Set up mock dependencies.
    content_bytes = dataset.values.tobytes()
    mock_streaming_body = StreamingBody(BytesIO(content_bytes), len(content_bytes))
    mock_response = {"Body": mock_streaming_body}
    mock_s3_client.get_object.return_value = mock_response

    # NOTE: "rioxarray.open_rasterio" requires driver access to read a .tif file from storage.
    # Since a temporary .tif file is not created here, a mock ""rioxarray.open_rasterio"" is used.
    mock_open_rasterio = MagicMock(return_value=dataset)
    with patch("rioxarray.open_rasterio", mock_open_rasterio):
        result = load_tif_from_s3("example.tif", "bucket", mock_s3_client)

    assert result.equals(dataset)


def test_load_tif_from_s3_invalid_extension() -> None:
    """
    Test response of load_tif_from_s3() to a file extension that is not ".tif"
    """
    with pytest.raises(
        ValueError, match="File extension does not correspond to a GeoTIFF file."
    ):
        load_tif_from_s3("sample.txt", "bucket", MagicMock())


def test_load_tif_from_s3_failed_response() -> None:
    """
    Test behavior of load_tif_from_s3() upon encountering a failed response from an S3 client.
    """
    mock_object_key = "empty.tif"
    mock_bucket = "bucket"
    with pytest.raises(
        AttributeError, match=f"Failed to fetch {mock_object_key} from S3."
    ):
        # Create mock S3 client.
        mock_s3_client = MagicMock()

        # Simulate an exception when fetching object from S3
        mock_s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Key not found"}}, "GetObject"
        )

        # Call test function.
        load_tif_from_s3(mock_object_key, mock_bucket, mock_s3_client)


def test_load_tif_from_s3_empty_dataset() -> None:
    """
    Test response of load_tif_from_s3() to an empty .tif file.
    """
    mock_object_key = "empty.tif"
    mock_bucket = "bucket"
    with pytest.raises(ValueError, match=f"Empty dataset in {mock_object_key}"):
        # Create mock S3 client.
        mock_s3_client = MagicMock()

        # Create an empty StreamingBody object.
        content_bytes = b""
        empty_body = StreamingBody(BytesIO(content_bytes), len(content_bytes))
        # Mock get_object() method of S3 client.
        mock_s3_client.get_object.return_value = {"Body": empty_body}

        # Call test function.
        load_tif_from_s3(mock_object_key, mock_bucket, mock_s3_client)


def test_load_tif_from_s3_key_not_found() -> None:
    """
    Test response of load_tif_from_s3() to a scenario where a non-existent object key is supplied.
    """
    mock_object_key = "nonexistent.tif"
    mock_bucket = "bucket"
    with pytest.raises(
        AttributeError, match=f"{mock_object_key} not found in {mock_bucket}."
    ):
        # Create mock S3 client.
        mock_s3_client = MagicMock()

        # Create a mock StreamingBody object.
        content_bytes = b"Magic"
        mock_streaming_body = StreamingBody(BytesIO(content_bytes), len(content_bytes))
        # Mock get_object() method of S3 client.
        mock_s3_client.get_object.return_value = {"No_Body": mock_streaming_body}

        # Call test function.
        load_tif_from_s3(mock_object_key, mock_bucket, mock_s3_client)
