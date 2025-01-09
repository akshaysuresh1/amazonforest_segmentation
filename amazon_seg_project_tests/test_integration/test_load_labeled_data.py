"""
Integration test for load_labeled_data()
"""

from unittest.mock import patch, MagicMock
import boto3
from moto import mock_aws
from amazon_seg_project.ops import load_labeled_data


@mock_aws
@patch("dagster_aws.s3.S3Resource")
def test_load_labeled_data_success(mock_s3_resource: MagicMock) -> None:
    """
    Test successful execution of load_labeled_data()
    """
    # Create a mock S3 client.
    s3_client = boto3.client("s3", region_name="us-east-1")
    bucket_name = "test-bucket"
    data_prefix = "data/"
    label_prefix = "labels/"

    # Create the bucket.
    s3_client.create_bucket(Bucket=bucket_name)

    # Upload test data files.
    data_files = ["data/file2.tif", "data/file1.tif", "data/file3.tif"]
    for file_key in data_files:
        s3_client.put_object(Bucket=bucket_name, Key=file_key, Body=b"content")

    # Upload corresponding label files.
    label_files = ["labels/file2.tif", "labels/file1.tif", "labels/file3.txt"]
    for file_key in label_files:
        s3_client.put_object(Bucket=bucket_name, Key=file_key, Body=b"label_content")

    # Create the mock Dagster AWS S3Resource to boto3 client.
    mock_s3_resource.get_client.return_value = s3_client

    # Call the test function.
    data_files_result, label_files_result = load_labeled_data(
        mock_s3_resource,
        bucket_name,
        data_prefix,
        label_prefix,
        ".tif",
        ".tif",
    )

    # Verify the results
    expected_data_files = ["data/file1.tif", "data/file2.tif"]
    expected_label_files = ["labels/file1.tif", "labels/file2.tif"]

    assert (
        data_files_result == expected_data_files
    ), f"Expected {expected_data_files}, got {data_files_result}"
    assert (
        label_files_result == expected_label_files
    ), f"Expected {expected_label_files}, got {label_files_result}"
