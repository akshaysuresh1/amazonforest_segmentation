"""
Tests for modules defined in tif_utils.py
"""

import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
import boto3
import numpy as np
import xarray as xr
import pickle
from botocore.exceptions import ClientError
from botocore.response import StreamingBody
from amazon_seg_project.scripts.tif_utils import (
    load_tif_from_s3,
    simulate_mock_multispec_data,
)


def test_simulate_mock_multispec_data_success() -> None:
    """
    Test successful execution of simulate_mock_multispec_data()
    """
    # Test parameters
    N_bands = np.random.randint(low=1, high=6)  # No. of spectral channels
    N_y = np.random.randint(low=5, high=20)  # No. of y samples
    N_x = np.random.randint(low=5, high=20)  # No. of x samples
    bit_depth = 16  # No. of bits per pixel

    # Call test function.
    result = simulate_mock_multispec_data(N_bands, N_y, N_x, bit_depth)

    # Assertions
    assert isinstance(result, xr.DataArray)
    assert result.shape == (N_bands, N_y, N_x)
    assert result.dtype == np.uint16

    # Check range of data values.
    min_value = 0
    max_value = 2**bit_depth - 1
    assert np.min(result.values) >= min_value
    assert np.max(result.values) <= max_value


def test_simulate_mock_multispec_data_nonpositive_N_bands() -> None:
    """
    Test the response of simulate_mock_multispec_data() to a non-positive N_bands argument
    """
    with pytest.raises(
        ValueError,
        match="Number of spectral channels must be an integer greater than 0.",
    ):
        simulate_mock_multispec_data(-3, 4, 5)


def test_simulate_mock_multispec_data_noninteger_N_bands() -> None:
    """
    Test the response of simulate_mock_multispec_data() to a non-integer N_bands argument
    """
    with pytest.raises(
        ValueError,
        match="Number of spectral channels must be an integer greater than 0.",
    ):
        simulate_mock_multispec_data(3.0, 4, 5)


def test_simulate_mock_multispec_data_nonpositive_N_y() -> None:
    """
    Test the response of simulate_mock_multispec_data() to a non-positive N_y argument
    """
    with pytest.raises(
        ValueError,
        match="Number of pixels along y-dimension must be an integer greater than 0.",
    ):
        simulate_mock_multispec_data(3, -4, 5)


def test_simulate_mock_multispec_data_noninteger_N_y() -> None:
    """
    Test the response of simulate_mock_multispec_data() to a non-integer N_y argument
    """
    with pytest.raises(
        ValueError,
        match="Number of pixels along y-dimension must be an integer greater than 0.",
    ):
        simulate_mock_multispec_data(3, 4.0, 5)


def test_simulate_mock_multispec_data_nonpositive_N_x() -> None:
    """
    Test the response of simulate_mock_multispec_data() to a non-positive N_x argument
    """
    with pytest.raises(
        ValueError,
        match="Number of pixels along x-dimension must be an integer greater than 0.",
    ):
        simulate_mock_multispec_data(3, 4, -5)


def test_simulate_mock_multispec_data_noninteger_N_x() -> None:
    """
    Test the response of simulate_mock_multispec_data() to a non-integer N_x argument
    """
    with pytest.raises(
        ValueError,
        match="Number of pixels along x-dimension must be an integer greater than 0.",
    ):
        simulate_mock_multispec_data(3, 4, 5.0)


def test_simulate_mock_multispec_data_unsupported_bit_depth() -> None:
    """
    Test the response of simulate_mock_multispec_data() to an unsupported bit_depth argument value.
    """
    bit_depth = 10
    with pytest.raises(
        ValueError,
        match=f"Unsupported bit depth: {bit_depth}. Supported values are integers 8, 16, 32, and 64.",
    ):
        simulate_mock_multispec_data(3, 4, 5, bit_depth)


def test_simulate_mock_multispec_data_noninteger_bit_depth() -> None:
    """
    Test the response of simulate_mock_multispec_data() to a non-integer bit_depth argument value.
    """
    bit_depth = 16.0
    with pytest.raises(
        ValueError,
        match=f"Unsupported bit depth: {bit_depth}. Supported values are integers 8, 16, 32, and 64.",
    ):
        simulate_mock_multispec_data(3, 4, 5, bit_depth)


@patch("rioxarray.open_rasterio")
def test_load_tif_from_s3_success(mock_open_rasterio: MagicMock) -> None:
    """
    Test successful execution of load_tif_from_s3()
    """
    # Create MagicMock() instances for mock objects.
    mock_s3_client = MagicMock()
    mock_rasterio = MagicMock()
    # Configure return value for mock_open_rasterio.
    mock_open_rasterio.return_value = mock_rasterio

    # Create mock data set for testing.
    N_bands = np.random.randint(low=1, high=6)  # No. of spectral channels
    N_y = np.random.randint(low=5, high=20)  # No. of y samples
    N_x = np.random.randint(low=5, high=20)  # No. of x samples
    bit_depth = int(np.random.choice([8, 16, 32, 64]))  # No. of bits per pixel
    dataset = simulate_mock_multispec_data(N_bands, N_y, N_x, bit_depth)

    # Serialize dataset to bytes.
    mock_tif_bytes = pickle.dumps(dataset)

    # Mock response from S3 client.
    stream = BytesIO(mock_tif_bytes)  # Wrap data bytes in a BytesIO object.
    mock_response = {"Body": StreamingBody(stream, len(mock_tif_bytes))}
    mock_s3_client.get_object.return_value = mock_response

    # Set dependencies for mock_open_rasterio.
    mock_rasterio.return_value = pickle.loads(mock_tif_bytes)

    # Call test function.
    result = load_tif_from_s3("example.tif", "bucket", mock_s3_client)

    assert result.equals(dataset)


def test_load_tif_from_s3_invalid_extension() -> None:
    """
    Test response of load_tif_from_s3() to a file extension that is not ".tif"
    """
    with pytest.raises(
        ValueError, match="File extension does not correspond to a GeoTIFF file."
    ):
        load_tif_from_s3("sample.txt", "mock_bucket", MagicMock())


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
