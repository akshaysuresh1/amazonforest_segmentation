"""
Dagster definitions
"""

from dagster import Definitions, load_assets_from_modules
from .assets import data_products
from .resources import s3_resource

# Data assets
data_assets = load_assets_from_modules([data_products], group_name="data")

# Definitions object for Dagster
defs = Definitions(
    assets=data_assets,
    resources={
        "s3_resource": s3_resource,
    },
)
