"""
Weights & Biases (W&B) ML utilities
"""

from typing import Dict, Any
import logging
from tqdm import tqdm
import torch
import wandb
from wandb.sdk.wandb_run import Run
from dagster import op, In, materialize_to_memory, RunConfig
from dagster import Any as dg_Any
from segmentation_models_pytorch import Unet
from .metrics import dice_loss
from .torch_utils import (
    create_data_loaders,
    setup_adam_w,
    train_epoch,
    validate_epoch,
    save_model_weights,
)
from .write_files import write_loss_data_to_csv
from ..assets import (
    SegmentationDataset,
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
from ..data_paths import OUTPUT_PATH
from ..resources import device


@op(
    ins={
        "wb_run": In(dg_Any),
        "training_dset": In(dg_Any),
        "validation_dset": In(dg_Any),
        "model": In(dg_Any),
    }
)
def train_unet(
    wb_run: Run,
    training_dset: SegmentationDataset,
    validation_dset: SegmentationDataset,
    model: Unet,
) -> None:
    """
    Train a U-net using batch gradient descent and log run results to W&B.
    """
    config = wb_run.config
    seed = config["seed"]
    encoder = config["encoder_name"]
    batch_size = config["batch_size"]
    lr_initial = config["lr_initial"]

    # Set PyTorch seed.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Move model to device.
    model = model.to(device)

    # Create data loaders for training and validation.
    train_loader, val_loader = create_data_loaders(
        training_dset,
        validation_dset,
        batch_size=batch_size,
        seed=seed,
    )

    # Set up optimizer and loss criterion.
    optimizer = setup_adam_w(model, lr_initial=lr_initial)
    criterion = dice_loss

    # Track minimum validation loss observed across epochs.
    lowest_val_loss = float("inf")
    # Store batch-averaged training and validation loss at every epoch.
    train_loss = []
    val_loss = []
    # Track epochs since last improvement in validation loss.
    epochs_since_improvement = 0

    # Set file names reflecting training configuration.
    weights_file = f"{encoder}_batch{batch_size}_lr{lr_initial:.1e}_weights.pt"
    loss_curve_csv = f"{encoder}_batch{batch_size}_lr{lr_initial:.1e}_losscurve.csv"

    # Begin model training.
    wb_run.watch(model)
    for epoch in tqdm(range(1, config["max_epochs"] + 1)):
        # Training step
        current_train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        train_loss.append(current_train_loss)

        # Validation step
        current_val_loss = validate_epoch(model, val_loader, criterion, device)
        val_loss.append(current_val_loss)

        # W&B logging
        wb_run.log(
            {
                "epoch": epoch,
                "train_loss": current_train_loss,
                "val_loss": current_val_loss,
            }
        )

        # Check if validation loss improved.
        if val_loss[-1] < lowest_val_loss:
            lowest_val_loss = val_loss[-1]
            epochs_since_improvement = 0  # Reset counter
            save_model_weights(model, OUTPUT_PATH / weights_file)
        else:
            epochs_since_improvement += 1  # Increment counter if no improvement

        # Implement early stopping criterion.
        if epochs_since_improvement == 5:
            logging.info("Early stopping criterion triggered.")
            write_loss_data_to_csv(
                train_loss, val_loss, OUTPUT_PATH / "train" / loss_curve_csv
            )
            break

        # Save loss curve data to disk intermittently.
        if epoch % 10 == 0 or epoch == config["max_epochs"]:
            write_loss_data_to_csv(
                train_loss, val_loss, OUTPUT_PATH / "train" / loss_curve_csv
            )


@op
def run_wandb_exp(wandb_config = None) -> None:
    """
    Run a W&B model training experiment.
    """
    with wandb.init(config=wandb_config) as run:
        # If called by wandb.agent, config will be set by Sweep Controller.
        wandb_config = run.config

        unet_config = BasicUnetConfig(
            encoder_name=wandb_config["encoder_name"], model_seed=wandb_config["seed"]
        )
        training_dataset_config = TrainingDatasetConfig(
            horizontal_flip_prob=wandb_config["horizontal_flip_prob"],
            vertical_flip_prob=wandb_config["vertical_flip_prob"],
            rotate90_prob=wandb_config["rotate90_prob"],
            augmentation_seed=wandb_config["seed"],
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
def upload_best_model_to_wandb(project: str, sweep_id: str) -> None:
    """
    Pushes the model with lowest validation loss from a sweep to W&B registry
    """
    api = wandb.Api()
    # Fetch the sweep object using the sweep ID.
    sweep = api.sweep(sweep_id)
    # Access all the runs related to the sweep
    runs = sweep.runs
    # Identify the run with the lowest validation loss as the best run.
    best_run = min(runs, key=lambda run: run.summary.get("val_loss", float("inf")))

    # Weights file corresponding to best run
    encoder = best_run.config["encoder_name"]
    batch_size = best_run.config["batch_size"]
    lr_initial = best_run.config["lr_initial"]
    weights_file = (
        OUTPUT_PATH / f"{encoder}_batch{batch_size}_lr{lr_initial:.1e}_weights.pt"
    )

    # Log the weights file from the best run as an artifact.
    with wandb.init(project=project) as wandb_run:
        artifact = wandb.Artifact(
            name="unet_model",
            type="model",
            description=f"Best model from sweep {sweep_id} based on validation loss",
            metadata={
                "run_id": best_run.id,
                "val_loss": best_run.summary["val_loss"],
                "encoder": encoder,
            },
        )
        artifact.add_file(str(weights_file))
        # Publish the logged artifact to the registry.
        REGISTRY_NAME = project
        COLLECTION_NAME = "models"
        target_path = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
        wandb_run.link_artifact(artifact=artifact, target_path=target_path)


@op
def run_sweep(config: SweepConfig) -> None:
    """
    Executes a W&B hyperparameter sweep
    """
    wandb.login()
    sweep_config = make_sweep_config(config)
    sweep_id = wandb.sweep(sweep_config, project=config.project)
    logging.info("Sweep ID: %s", sweep_id)
    wandb.agent(sweep_id, function=run_wandb_exp)
    upload_best_model_to_wandb(config.project, sweep_id)
    wandb.finish()
