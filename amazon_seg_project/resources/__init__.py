"""
Intialization of environment variables and Dagster resources
"""

from typing import TypeVar
import torch
import numpy as np
from dotenv import load_dotenv
from dagster import EnvVar
from dagster_aws.s3 import S3Resource


load_dotenv()

# Environment variables
AWS_REGION_NAME = EnvVar("AWS_REGION_NAME")
AWS_ACCESS_KEY_ID = EnvVar("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = EnvVar("AWS_SECRET_ACCESS_KEY")
AMAZON_TIF_BUCKET = EnvVar("AMAZON_TIF_BUCKET")

if AMAZON_TIF_BUCKET.get_value() is None:
    raise ValueError("Environment variable 'AMAZON_TIF_BUCKET' must be set.")

# Dagster-AWS S3 resource
s3_resource = S3Resource(
    region_name=AWS_REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# Set default device config for torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define ScalarType for numpy arrays.
ScalarTypeT = TypeVar("ScalarTypeT", np.int_, np.float64)
