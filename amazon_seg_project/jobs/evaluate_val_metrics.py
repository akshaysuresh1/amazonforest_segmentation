"""
Dagster job: Evaluate metrics on validation dataset.
"""

from dagster import define_asset_job, AssetSelection

# Select assets for job.
asset_selection = (
    AssetSelection.assets("validation_metrics")
    .upstream()
    .required_multi_asset_neighbors()
)
# pylint: disable=assignment-from-no-return

compute_val_metrics = define_asset_job(
    name="compute_val_metrics",
    selection=asset_selection,
    description="""
        Evaluate validation metrics for a trained model using a specific binarization threshold.
    """,
)
