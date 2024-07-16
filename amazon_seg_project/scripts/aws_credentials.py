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
    if aws_region_name not in AWS_REGIONS_LIST:
        raise ValueError(
            f"AWS region name '{aws_region_name}' is not valid or not supported."
        )

    return aws_region_name
