"""
Job: Evaluate and store statistics of training data distribution in a local .csv file.
"""

from dagster import define_asset_job, AssetSelection

# Select assets for jobb.
asset_selection = (
    AssetSelection.assets("mean_training_data", "sigma_training_data")
    .upstream()
    .required_multi_asset_neighbors()
)

compute_training_stats = define_asset_job(
    name="compute_training_stats",
    selection=asset_selection,
)
