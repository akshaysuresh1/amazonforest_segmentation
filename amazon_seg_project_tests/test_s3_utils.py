"""
Tests for modules defined in s3_utils.py
"""

import pytest
from unittest.mock import patch, MagicMock
from amazon_seg_project.scripts.s3_utils import (
    initialize_s3_client,
    paginate_s3_objects,
    filter_object_keys,
    list_objects,
)
from botocore.exceptions import ClientError


@patch("boto3.client")
@patch("os.getenv")
def test_initialize_s3_client(
    mock_getenv: MagicMock, mock_boto_client: MagicMock
) -> None:
    """
    Test for successful initialization of S3 client
    """
    # Set mock values for environment variables.
    mock_region = "us-west-1"
    mock_access_key_id = "fake"
    mock_secret_access_key = "secret"

    # Set mock environment variable values.
    mock_getenv.side_effect = lambda key: {
        "AWS_REGION_NAME": mock_region,
        "S3_ACCESS_KEY_ID": mock_access_key_id,
        "S3_SECRET_ACCESS_KEY": mock_secret_access_key,
    }.get(key, None)

    # Mock boto3 client to return a mock client instance
    mock_client_instance = MagicMock()
    mock_boto_client.return_value = mock_client_instance

    # Call the function to test
    client = initialize_s3_client()

    # Assertions
    mock_boto_client.assert_called_once_with(
        service_name="s3",
        region_name=mock_region,
        aws_access_key_id=mock_access_key_id,
        aws_secret_access_key=mock_secret_access_key,
    )
    assert client == mock_client_instance


def test_paginate_s3_objects() -> None:
    """
    Test successful pagination of prefix path in S3 bucket across multiple non-empty pages.
    """
    # Define mock bucket and prefix path.
    mock_bucket = "bucket"
    mock_prefix = "prefix"

    # Define mock response pages.
    page1 = {"Contents": [{"Key": "file1.tif"}, {"Key": "file2.tif"}]}
    page2 = {"Contents": [{"Key": "file3.tif"}, {"Key": "file4.tif"}]}
    page3 = {"Contents": [{"Key": "file5.tif"}]}

    # Set up mock S3 client and paginator.
    mock_s3_client = MagicMock()
    mock_paginator = MagicMock()
    mock_s3_client.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = iter([page1, page2, page3])

    # Call test function.
    result = paginate_s3_objects(mock_s3_client, mock_bucket, mock_prefix)

    # Check if result is iterable.
    assert hasattr(result, "__iter__")

    # Convert result to a list of pages for quick assertions.
    result_pages = list(result)

    # Assert that the correct pages were returned in the expected order.
    assert len(result_pages) == 3
    assert result_pages[0] == page1
    assert result_pages[1] == page2
    assert result_pages[2] == page3


def test_filter_object_keys_empty() -> None:
    """
    Test for return of [] by filter_object_keys() for an input empty list of pages.
    """
    mock_pages = []
    assert [] == filter_object_keys(mock_pages)


def test_filter_object_keys_nocontents() -> None:
    """
    Test for the response of filter_object_keys() in the absence of "Contents" key in pages.
    """
    pages_without_contents = [{"NoContents": [{"Key": "file1.tif"}]}]
    assert [] == filter_object_keys(pages_without_contents)


def test_filter_object_keys_contents_only() -> None:
    """
    Test for the response of filter_object_keys() to multiple pages, each having a "Contents" keyword.
    Pages contain keys to objects with different extensions.
    """
    pages_with_contents = [
        {"Contents": [{"Key": "file1.tif"}, {"Key": "folder/"}, {"Key": "file2.txt"}]},
        {"Contents": [{"Key": "file3.tif"}, {"Key": "file4.txt"}]},
    ]
    expected_result = ["file1.tif", "file3.tif"]
    assert expected_result == filter_object_keys(pages_with_contents, ".tif")


def test_filter_object_keys_contents_mixed() -> None:
    """
    Test for the response of filter_object_keys() to multiple pages, of which only a subset contain the "Contents" keyword.
    """
    pages_mixed = [
        {"Contents": [{"Key": "file1.tif"}, {"Key": "folder/"}]},
        {"NoContents": [{"Key": "file_bogus.tif"}]},
        {"Contents": [{"Key": "file2.tif"}, {"Key": "file3.txt"}]},
    ]
    expected_result = ["file1.tif", "file2.tif"]
    assert expected_result == filter_object_keys(pages_mixed, ".tif")


@patch("amazon_seg_project.scripts.s3_utils.initialize_s3_client")
def test_list_objects_success(
    mock_initialize_s3_client: MagicMock,
) -> None:
    """
    Test for successful execution of list_objects()
    """
    # Create mock S3 client.
    mock_s3_client = MagicMock()
    mock_initialize_s3_client.return_value = mock_s3_client

    # Create paginator for mock client.
    mock_paginator = MagicMock()
    mock_s3_client.get_paginator.return_value = mock_paginator

    # Define mock pages for paginator.
    mock_pages = [
        {"Contents": [{"Key": "file1.tif"}, {"Key": "folder/"}]},
        {"NoContents": [{"Key": "file_bogus.tif"}]},
        {"Contents": [{"Key": "file2.tif"}, {"Key": "file3.txt"}]},
    ]
    mock_paginator.paginate.return_value = iter(mock_pages)
    expected_result = ["file1.tif", "file2.tif"]

    # Call the test function.
    result = list_objects(
        "mock_bucket", prefix="test_prefix", file_extension=".tif"
    )

    assert expected_result == result


@patch("amazon_seg_project.scripts.s3_utils.initialize_s3_client")
@patch("amazon_seg_project.scripts.s3_utils.paginate_s3_objects")
@patch("amazon_seg_project.scripts.s3_utils.filter_object_keys")
@patch("logging.error")
def test_list_objects_client_error(
    mock_logging: MagicMock,
    mock_filter_keys: MagicMock,
    mock_paginate: MagicMock,
    mock_init_client: MagicMock,
) -> None:
    """
    Tests response of list_objects() to client-end pagination failure
    """
    with pytest.raises(ClientError):
        # Mock ClientError
        error = ClientError(
            {"Error": {"Code": "code", "Message": "message"}}, "operation_name"
        )
        mock_paginate.side_effect = error
        mock_filter_keys.return_value = [
            "object_key1",
            "object_key2",
        ]  # Mock object keys
        mock_init_client.return_value = MagicMock()  # Mock S3 client

        list_objects("mock_bucket", prefix="test_prefix")

        # Check error logging
        mock_logging.error.assert_called_once(
            f"S3 ClientError: {error.response['Error']['Message']} - {error.response['Error']['Code']}"
        )


@patch("amazon_seg_project.scripts.s3_utils.initialize_s3_client")
@patch("amazon_seg_project.scripts.s3_utils.paginate_s3_objects")
@patch("logging.warning")
def test_list_objects_no_objects(
    mock_logging: MagicMock,
    mock_paginate: MagicMock,
    mock_init_client: MagicMock,
) -> None:
    """
    Test response of list_objects() to an empty list of objects found at prefix path
    """
    mock_prefix = "test_prefix"
    with pytest.raises(
        ValueError, match=f"No objects found for prefix: '{mock_prefix}'"
    ):
        mock_pages = [{"NoContents": [{"Key": "file1.tif"}]}]

        # Mock return values
        mock_init_client.return_value = MagicMock()  # Mock S3 client
        mock_paginate.return_value = iter(mock_pages)  # Simulate pagination result

        list_objects("mock_bucket", prefix=mock_prefix)

        # Check warning log message
        mock_logging.warning.assert_called_once_with(
            f"No objects found for prefix: '{mock_prefix}'"
        )
