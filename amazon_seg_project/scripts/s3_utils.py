"""
Utilities for accessing and reading from S3 buckets
"""

import logging
import boto3
from typing import List, Any
from botocore.exceptions import ClientError
from .aws_credentials import (
    get_aws_region_name,
    get_s3_access_key_id,
    get_s3_secret_access_key,
)


def initialize_s3_client() -> boto3.client:
    """
    Initialize and return an S3 client with the required credentials and region.

    Returns: The initialized S3 client.
    """
    return boto3.client(
        service_name="s3",
        region_name=get_aws_region_name(),
        aws_access_key_id=get_s3_access_key_id(),
        aws_secret_access_key=get_s3_secret_access_key(),
    )


def paginate_s3_objects(
    s3: boto3.client, bucket: str, prefix: str
) -> List[dict[str, Any]]:
    """
    Paginate through S3 objects based on the given prefix and return pages of objects.

    Args:
        s3: The initialized S3 client.
        bucket: The name of the S3 bucket.
        prefix: The prefix path to list objects from.

    Returns: List of pages containing S3 object data.
    """
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    return page_iterator


def filter_object_keys(
    pages: List[dict[str, Any]], file_extension: str = ""
) -> List[str]:
    """
    Filter object keys with a specific file extension from the paginated S3 object data.

    Args:
        pages: List of pages containing S3 object data.
        file_extension: File format extension (e.g., ".tif", ".txt", ".img", etc.)

    Returns: List of filtered object keys.
    """
    object_keys = []
    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                if (not obj["Key"].endswith("/")) and (
                    obj["Key"].endswith(file_extension)
                ):
                    object_keys.append(obj["Key"])
    return object_keys


def list_objects(
    bucket: str, prefix: str = "", file_extension: str = ""
) -> List[str]:
    """
    List objects with a specific file extension that are stored in a prefix path of an S3 bucket.

    Args:
        bucket: Name of S3 bucket
        prefix: The prefix path within the S3 bucket to list objects from.
        file_extension: File format extension (e.g., ".tif", ".txt", ".img", etc.)

    Returns: List of object keys stored at the path identified by the "prefix" in S3 bucket.

    Raises:
        ClientError: If there's an error querying the S3 bucket.
        ValueError: If no objects are found for the given prefix.
    """
    s3 = initialize_s3_client()

    try:
        pages = paginate_s3_objects(s3, bucket, prefix)
        object_keys = filter_object_keys(pages, file_extension)

        if not object_keys:
            logging.warning(f"No objects found for prefix: '{prefix}'")
            raise ValueError(f"No objects found for prefix: '{prefix}'")

    except ClientError as e:
        logging.error(
            f"S3 ClientError: {e.response['Error']['Message']} - {e.response['Error']['Code']}"
        )
        raise  # Re-raise the ClientError exception after logging it

    return object_keys
