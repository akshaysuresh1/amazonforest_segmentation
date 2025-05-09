"""
Weights & Biases (W&B) ML utilities for runs and sweeps
"""

from typing import Dict, Any
import logging
import wandb
from dagster import op, materialize_to_memory, RunConfig
from .train_unet import train_unet
from .wandb_artifact_utils import promote_best_model_to_registry
from ..assets import (
    unet_model,
    data_training,
    data_validation,
    afs_training_dataset,
    afs_validation_dataset,
)
from ..config import (
    TrainingDatasetConfig,
    BasicUnetConfig,
    SweepConfig,
)


@op
def run_wandb_exp(wandb_config=None) -> None:
    """
    Run a W&B model training experiment.
    """
    with wandb.init(config=wandb_config) as run:
        # If called by wandb.agent, config will be set by Sweep Controller.
        wandb_config = run.config

        unet_config = BasicUnetConfig(
            encoder_name=wandb_config.get("encoder_name"),
            model_seed=wandb_config.get("seed"),
        )
        training_dataset_config = TrainingDatasetConfig(
            horizontal_flip_prob=wandb_config.get("horizontal_flip_prob"),
            vertical_flip_prob=wandb_config.get("vertical_flip_prob"),
            rotate90_prob=wandb_config.get("rotate90_prob"),
            augmentation_seed=wandb_config.get("seed"),
        )

        # Materialize data and model assets.
        assets = [
            unet_model,
            data_training,
            data_validation,
            afs_training_dataset,
            afs_validation_dataset,
        ]
        result = materialize_to_memory(
            assets,
            run_config=RunConfig(
                {
                    "basic_unet_model": unet_config,
                    "training_dataset": training_dataset_config,
                }
            ),
        )
        model = result.asset_value("basic_unet_model")
        training_dataset = result.asset_value("training_dataset")
        validation_dataset = result.asset_value("validation_dataset")

        # Call model training op.
        train_unet(run, training_dataset, validation_dataset, model)


@op
def make_sweep_config(config: SweepConfig) -> Dict[str, Any]:
    """
    Creates a W&B sweep config dictionary
    """
    sweep_config = {
        "method": config.method,  # Sampling strategy: grid, random, or bayes
        "metric": {
            "name": config.metric_name,
            "goal": config.metric_goal,
        },  # Metric to optimize
        "parameters": {
            "seed": config.seed,  # Seed for reproducibility
            "threshold": config.threshold,  # Mask binarization threshold
            "encoder_name": config.encoder_name,  # Encoder for U-net model
            "batch_size": config.batch_size,  # Batch size
            "lr_initial": config.lr_initial,  # Initial learning rate
            "max_epochs": config.max_epochs,  # Max no. of training epochs
            # Data augmentation probabilities
            "horizontal_flip_prob": config.horizontal_flip_prob,
            "vertical_flip_prob": config.vertical_flip_prob,
            "rotate90_prob": config.rotate90_prob,
        },
    }

    return sweep_config


@op
def run_sweep(config: SweepConfig) -> None:
    """
    Executes a W&B hyperparameter sweep
    """
    # Start sweep.
    wandb.login()
    sweep_config = make_sweep_config(config)
    # Set up sweep ID.
    sweep_id = wandb.sweep(sweep_config, project=config.project, entity=config.entity)
    logging.info("Sweep ID: %s", sweep_id)
    # Execute sweep.
    wandb.agent(sweep_id, function=run_wandb_exp)
    # Close sweep.
    wandb.finish()
    # Push model with lowest validation loss to registry.
    promote_best_model_to_registry(config.entity, config.project, sweep_id)
