"""
Dagster definitions
"""

from dagster import Definitions
from .resources import s3_resource

# Definitions object for Dagster
defs = Definitions(
    resources={
        "s3_resource": s3_resource,
    },
)
