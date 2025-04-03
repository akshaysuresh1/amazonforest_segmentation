"""
Dagster job: Execute a hyperparameter sweep using Weights & Biases.
"""

from dagster import job
from ..ops.wandb_utils import run_sweep


@job
def run_wandb_sweep() -> None:
    """
    Dagster job wrapper around op "run_sweep()"
    """
    run_sweep()
