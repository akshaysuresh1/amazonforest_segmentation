"""
Dagster job: Evaluate and store training data statistics in a local .csv file.
"""

from dagster import define_asset_job, AssetSelection

# Select assets for job.
asset_selection = (
    AssetSelection.assets("mean_training_data", "sigma_training_data")
    .upstream()
    .required_multi_asset_neighbors()
)
# pylint: disable=assignment-from-no-return

compute_training_stats = define_asset_job(
    name="compute_training_stats",
    selection=asset_selection,
    description="""
    Calculate channel-wise mean and standard deviation across images training dataset.
    Results get stored in a local .csv file.
    """,
)
