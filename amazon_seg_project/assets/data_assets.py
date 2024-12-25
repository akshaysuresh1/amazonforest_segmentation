"""
Data assets: training, validation, and test datasets
"""

from typing import Any
from dagster import asset
from dagster_aws.s3 import S3Resource


@asset
def training_data_tif(s3: S3Resource) -> Any:
    """
    Placeholder
    """
    client = s3.get_client()
    return client
