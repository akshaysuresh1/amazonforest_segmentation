"""
Dagster job: Evaluate model metrics on test dataset.
"""

from dagster import define_asset_job, AssetSelection

# Select assets for job.
asset_selection = (
    AssetSelection.assets("test_dataset_metrics")
    .upstream()
    .required_multi_asset_neighbors()
)
# pylint: disable=assignment-from-no-return

compute_test_dataset_metrics = define_asset_job(
    name="compute_test_dataset_metrics",
    selection=asset_selection,
    description="""
        Evaluate test dataset metrics for a trained model using a specific binarization threshold.
    """,
)
