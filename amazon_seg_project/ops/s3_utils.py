"""
Modules to read and access specific files from S3 buckets
"""

import logging
from typing import List, Iterable, Dict, Any
from botocore.exceptions import ClientError
from dagster import op, In, Out
from dagster import Any as dg_Any
from dagster_aws.s3 import S3Resource


@op(out=Out(dg_Any))
def paginate_s3_objects(
    s3: S3Resource, s3_bucket: str, prefix: str
) -> Iterable[Dict[str, Any]]:
    """
    Paginate through objects in a S3 bucket based on a given prefix and return pages of objects.

    Args:
        s3: Dagster-AWS S3 resource
        s3_bucket: Name of S3 bucket
        prefix: Prefix path to search

    Returns: List of pages containing S3 object data
    """
    paginator = s3.get_client().get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=s3_bucket, Prefix=prefix)
    return page_iterator


@op(ins={"pages": In(dg_Any), "file_extension": In(str)}, out=Out(List[str]))
def filter_object_keys(
    pages: Iterable[Dict[str, Any]], file_extension: str = ""
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


@op(out=Out(List[str]))
def list_objects(
    s3: S3Resource, s3_bucket: str, prefix: str = "", file_extension: str = ""
) -> List[str]:
    """
    List objects with a specific file extension that are stored in a prefix path of an S3 bucket.

    Args:
        s3: Dagster-AWS S3 resource
        s3_bucket: Name of S3 bucket
        prefix: Prefix path to search
        file_extension: File format extension (e.g., ".tif", ".txt", ".img", etc.)

    Returns: List of object keys stored at the path identified by the "prefix" in S3 bucket.

    Raises:
        ClientError: If there is an error querying the S3 bucket.
        ValueError: If no objects are found for the given prefix.
    """
    try:
        pages = paginate_s3_objects(s3, s3_bucket, prefix)
    except ClientError as e:
        logging.error(
            "S3 ClientError: %s - %s",
            e.response["Error"]["Message"],
            e.response["Error"]["Message"],
        )
        # Re-raise the ClientError exception after logging it
        raise

    object_keys = filter_object_keys(pages, file_extension)
    if not object_keys:
        logging.warning("No objects found for prefix: %s", prefix)
        raise ValueError(f"No objects found for prefix: {prefix}")

    return object_keys
