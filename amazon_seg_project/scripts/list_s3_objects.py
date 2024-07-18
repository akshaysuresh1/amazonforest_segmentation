"""
Read data files from S3 bucket.
"""

import logging
import boto3
from typing import List
from botocore.exceptions import ClientError
from .aws_credentials import (
    get_aws_region_name,
    get_s3_access_key_id,
    get_s3_secret_access_key,
    get_s3_bucket,
)


def list_objects(prefix: str = "") -> List[str]:
    """
    List objects stored in a prefix path of an S3 bucket.

    Args:
        prefix (str): The prefix path within the S3 bucket to list objects from.

    Returns:
        List[str]: List of object keys stored at the path identified by the "prefix".

    Raises:
        ClientError: If there's an error querying the S3 bucket.
        ValueError: If no objects are found for the given prefix.
    """
    # Initialize a boto3 client to access S3.
    s3 = boto3.client(
        service_name="s3",
        region_name=get_aws_region_name(),
        aws_access_key_id=get_s3_access_key_id(),
        aws_secret_access_key=get_s3_secret_access_key(),
    )

    object_keys = []
    bucket = get_s3_bucket()

    try:
        # Initialize a paginator to handle multiple pages of results from s3.list_object_v2().
        paginator = s3.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

        # Flag to track if any objects are found
        objects_found = False

        # Iterate through each page of results.
        for page in page_iterator:
            if "Contents" in page:
                objects_found = True
                for obj in page["Contents"]:
                    # Ensure the object key is not a folder.
                    if not obj["Key"].endswith("/"):
                        object_keys.append(obj["Key"])
        
        if not objects_found:
            logging.warning(f"No objects found for prefix: '{prefix}'")
            raise ValueError(f"No objects found for prefix: '{prefix}'")
        
    except ClientError as e:
        logging.error(f"S3 ClientError: {e}")
        raise # Re-raise the exception after logging it

    return object_keys