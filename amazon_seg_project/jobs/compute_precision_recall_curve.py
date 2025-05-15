"""
Dagster job: Compute precision-recall curve on validation dataset.
"""

from dagster import define_asset_job, AssetSelection

# Select assets for job.
asset_selection = (
    AssetSelection.assets("precision_recall_curve")
    .upstream()
    .required_multi_asset_neighbors()
)
# pylint: disable=assignment-from-no-return

compute_val_precision_recall_curve = define_asset_job(
    name="compute_val_precision_recall_curve",
    selection=asset_selection,
    description="Compute precision-recall curve on validation dataset.",
)
