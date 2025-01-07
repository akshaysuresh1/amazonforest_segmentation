"""
Unit tests for modules defined in amazon_seg_project/ops/s3_utils.py
"""

from unittest.mock import patch, MagicMock
import pytest
from botocore.exceptions import ClientError
from amazon_seg_project.ops import (
    paginate_s3_objects,
    filter_object_keys,
    list_objects,
)


@patch("dagster_aws.s3.S3Resource")
def test_paginate_s3_objects(mock_s3_resource: MagicMock) -> None:
    """
    Test successful pagination of prefix path in S3 bucket across multiple non-empty pages.
    """
    # Define mock bucket and prefix path.
    mock_s3_bucket = "s3_bucket"
    mock_prefix = "prefix"

    # Define mock response pages.
    page1 = {"Contents": [{"Key": "file1.tif"}, {"Key": "file2.tif"}]}
    page2 = {"Contents": [{"Key": "file3.tif"}, {"Key": "file4.tif"}]}
    page3 = {"Contents": [{"Key": "file5.tif"}]}

    # Set up mock S3 client.
    mock_s3_client = MagicMock()
    mock_s3_resource.get_client.return_value = mock_s3_client

    # Set up mock paginator.
    mock_paginator = MagicMock()
    mock_s3_client.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = iter([page1, page2, page3])

    # Call test function.
    result = paginate_s3_objects(mock_s3_resource, mock_s3_bucket, mock_prefix)

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
    Test response of filter_object_keys() to an empty input list of pages.
    """
    mock_pages = []
    assert not filter_object_keys(mock_pages)


def test_filter_object_keys_nocontents() -> None:
    """
    Test the response of filter_object_keys() to a missing "Contents" key in pages.
    """
    pages_without_contents = [{"NoContents": [{"Key": "file1.tif"}]}]
    assert not filter_object_keys(pages_without_contents)


def test_filter_object_keys_contents_only() -> None:
    """
    Test the response of filter_object_keys() to multiple pages, each having a "Contents" keyword.
    Pages contain keys to objects with different extensions.
    """
    pages_with_contents = [
        {"Contents": [{"Key": "file1.tif"}, {"Key": "folder/"}, {"Key": "file2.txt"}]},
        {"Contents": [{"Key": "file3.tif"}, {"Key": "file4.txt"}]},
    ]
    expected_result = ["file1.tif", "file3.tif"]
    assert filter_object_keys(pages_with_contents, ".tif") == expected_result


def test_filter_object_keys_contents_mixed() -> None:
    """
    Test the response of filter_object_keys() to multiple pages.
    Only a subset of pages contain the "Contents" keyword.
    """
    pages_mixed = [
        {"Contents": [{"Key": "file1.tif"}, {"Key": "folder/"}]},
        {"NoContents": [{"Key": "file_bogus.tif"}]},
        {"Contents": [{"Key": "file2.tif"}, {"Key": "file3.txt"}]},
    ]
    expected_result = ["file1.tif", "file2.tif"]
    assert filter_object_keys(pages_mixed, ".tif") == expected_result


@patch("amazon_seg_project.ops.paginate_s3_objects")
def test_list_objects_success(mock_paginate: MagicMock) -> None:
    """
    Test for successful execution of list_objects()
    """
    # Define mock pages for paginator.
    mock_pages = [
        {"Contents": [{"Key": "file1.tif"}, {"Key": "folder/"}]},
        {"NoContents": [{"Key": "file_bogus.tif"}]},
        {"Contents": [{"Key": "file2.tif"}, {"Key": "file3.txt"}]},
    ]
    mock_paginate.return_value = iter(mock_pages)
    # Call internal function.
    object_keys = filter_object_keys(mock_paginate.return_value, file_extension=".tif")

    expected_result = ["file1.tif", "file2.tif"]

    assert object_keys == expected_result


@patch("amazon_seg_project.ops.s3_utils.paginate_s3_objects")
@patch("logging.error")
def test_list_objects_client_error(
    mock_logging: MagicMock, mock_paginate: MagicMock
) -> None:
    """
    Test response of list_objects() to client-end pagination failure
    """
    with pytest.raises(ClientError):
        # Mock ClientError for pagination
        error = ClientError(
            {"Error": {"Code": "code", "Message": "message"}}, "operation_name"
        )
        mock_paginate.side_effect = error

        # Mock S3 resource
        mock_s3_resource = MagicMock()

        # Call the test function.
        list_objects(mock_s3_resource, "mock_s3_bucket", prefix="test_prefix")

        # Check error logging
        mock_logging.error.assert_called_once(
            f"""S3 ClientError: {error.response['Error']['Message']} -
            {error.response['Error']['Code']}"""
        )


@patch("amazon_seg_project.ops.s3_utils.paginate_s3_objects")
@patch("logging.warning")
def test_list_objects_no_objects(
    mock_logging: MagicMock, mock_paginate: MagicMock
) -> None:
    """
    Test response of list_objects() to an empty list of objects found at prefix path
    """
    mock_prefix = "test_prefix"
    with pytest.raises(ValueError, match=f"No objects found for prefix: {mock_prefix}"):
        mock_pages = [{"NoContents": [{"Key": "file1.tif"}]}]

        # Mock S3 resource
        mock_s3_resource = MagicMock()

        # Set mock paginator return value.
        mock_paginate.return_value = iter(mock_pages)

        list_objects(mock_s3_resource, "mock_bucket", prefix=mock_prefix)

        # Check warning log message
        mock_logging.warning.assert_called_once_with(
            f"No objects found for prefix: {mock_prefix}"
        )
