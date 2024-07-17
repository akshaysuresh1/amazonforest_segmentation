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
    ValueError: If AWS region name from .env file is not found in AWS_REGIONS_LIST
    """

    aws_region_name = os.getenv("AWS_REGION_NAME")
    if aws_region_name is None:
        raise ValueError("AWS region name is not set.")
    elif aws_region_name == "":
        raise ValueError("AWS region name is empty.")
    elif aws_region_name not in AWS_REGIONS_LIST:
        raise ValueError(
            f"AWS region name '{aws_region_name}' is not valid or not supported."
        )

    return aws_region_name


def get_aws_access_key_id() -> str:
    """
    Reads in AWS access key ID from environment variables.

    Returns:
    str: AWS access key ID

    Raises:
    ValueError: If AWS access key ID is empty or not set
    """

    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    if aws_access_key_id is None:
        raise ValueError("AWS access key ID is not set.")
    elif aws_access_key_id == "":
        raise ValueError("AWS access key ID is empty.")

    return aws_access_key_id
