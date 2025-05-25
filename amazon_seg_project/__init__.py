"""
Dagster definitions
"""

from dagster import Definitions, load_assets_from_modules
from .assets import data_products, datasets, model, val_dataset_metrics, statistics
from .resources import s3_resource
from .jobs import (
    compute_training_stats,
    compute_val_precision_recall_curve,
    run_wandb_sweep,
)

# Assets
data_assets = load_assets_from_modules([data_products, datasets], group_name="data")
stats_assets = load_assets_from_modules([statistics], group_name="stats")
model_assets = load_assets_from_modules([model], group_name="model")
result_assets = load_assets_from_modules(
    [val_dataset_metrics], group_name="results"
)

# Combine all assets into a list.
all_assets = [*data_assets, *stats_assets, *model_assets, *result_assets]

# Jobs
all_jobs = [compute_training_stats, compute_val_precision_recall_curve, run_wandb_sweep]

# Definitions object for Dagster
defs = Definitions(
    assets=all_assets,
    resources={"s3_resource": s3_resource},
    jobs=all_jobs,  # type: ignore
)
