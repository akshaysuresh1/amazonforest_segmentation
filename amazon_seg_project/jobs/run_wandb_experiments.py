"""
Job: Run an ML experiment using Weights & Biases.
"""

from dagster import job
from ..ops.wandb_utils import run_wandb_training


@job
def run_wandb_experiment() -> None:
    """
    Initiate a W&B experiment for ML model training
    """
    run_wandb_training()  # pylint: disable=E1120
