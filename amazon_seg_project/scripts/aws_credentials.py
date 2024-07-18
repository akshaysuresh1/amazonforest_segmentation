"""
Access AWS credentials from .env file.
"""

import os
from .constants import AWS_REGIONS_LIST


def get_aws_region_name() -> str:
    """
    Reads AWS region name from environment variables and validates it against AWS_REGIONS_LIST.

    Returns:
    str: AWS region name

    Raises:
    AttributeError: If environment variable AWS_REGION_NAME is undefined
    ValueError: If environment variable AWS_REGION_NAME is empty or not found in AWS_REGIONS_LIST
    """

    aws_region_name = os.getenv("AWS_REGION_NAME")
    if aws_region_name is None:
        raise AttributeError("Environment variable AWS_REGION_NAME is undefined.")
    elif aws_region_name == "":
        raise ValueError("Environment variable AWS_REGION_NAME is empty.")
    elif aws_region_name not in AWS_REGIONS_LIST:
        raise ValueError(
            f"Region name '{aws_region_name}' is not valid or not supported by AWS."
        )

    return aws_region_name


def get_s3_access_key_id() -> str:
    """
    Reads in S3 access key ID from environment variables.

    Returns:
    str: S3 access key ID

    Raises:
    AttributeError: If environment variable S3_ACCESS_KEY_ID is undefined
    ValueError: If environment variable S3_ACCESS_KEY_ID is empty
    """

    s3_access_key_id = os.getenv("S3_ACCESS_KEY_ID")
    if s3_access_key_id is None:
        raise AttributeError("Environment variable S3_ACCESS_KEY_ID is undefined.")
    elif s3_access_key_id == "":
        raise ValueError("Environment variable S3_ACCESS_KEY_ID is empty.")

    return s3_access_key_id


def get_s3_secret_access_key() -> str:
    """
    Reads in S3 secret access key from environment variables.

    Returns:
    str: S3 secret access key

    Raises:
    AttributeError: If environment variable S3_SECRET_ACCESS_KEY is undefined
    ValueError: If environment variable S3_SECRET_ACCESS_KEY is empty
    """

    s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY")
    if s3_secret_access_key is None:
        raise AttributeError("Environment variable S3_SECRET_ACCESS_KEY is undefined.")
    elif s3_secret_access_key == "":
        raise ValueError("Environment variable S3_SECRET_ACCESS_KEY is empty.")

    return s3_secret_access_key

def get_s3_bucket() -> str:
    """
    Reads in S3 bucket name from environment variables.

    Returns:
    str: S3 bucket name

    Raises:
    AttributeError: If environment variable BUCKET is undefined
    ValueError: If environment variable BUCKET is empty
    """

    bucket = os.getenv("BUCKET")
    if bucket is None:
        raise AttributeError("Environment variable BUCKET is undefined.")
    elif bucket == "":
        raise ValueError("Environment variable BUCKET is empty.")
    
    return bucket
