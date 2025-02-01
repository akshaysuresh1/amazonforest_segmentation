"""
Dagster definitions
"""

from dagster import Definitions, load_assets_from_modules
from .assets import data_products, datasets, statistics
from .resources import s3_resource
from .jobs import compute_training_stats

# Assets
data_assets = load_assets_from_modules([data_products, datasets], group_name="data")
stats_assets = load_assets_from_modules([statistics], group_name="stats")

# Jobs
all_jobs = [compute_training_stats]

# Definitions object for Dagster
defs = Definitions(
    assets=data_assets + stats_assets,
    resources={
        "s3_resource": s3_resource,
    },
    jobs=all_jobs,
)
