"""
Weights & Biases (W&B) ML utilities
"""

from typing import Dict, Any
from tqdm import tqdm
import torch
import wandb
from dagster import op, In, materialize_to_memory
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
from ..config import BasicUnetConfig, ModelTrainingConfig
from ..data_paths import OUTPUT_PATH
from ..resources import device


@op(
    ins={
        "project_name": In(str),
        "wandb_config": In(Dict[str, dg_Any]),
        "training_dset": In(dg_Any),
        "validation_dset": In(dg_Any),
        "model": In(dg_Any),
    }
)
def train_unet(
    project_name: str,
    wandb_config: Dict[str, Any],
    training_dset: SegmentationDataset,
    validation_dset: SegmentationDataset,
    model: Unet,
) -> None:
    """
    Train a U-net using batch gradient descent and log results to W&B.
    """
    with wandb.init(project=project_name, config=wandb_config) as run:
        # If called by wandb.agent, this config will be set by Sweep Controller.
        config = run.config
        encoder = config["encoder_name"]
        batch_size = config["batch_size"]
        lr_initial = config["lr_initial"]
        torch.manual_seed(config["seed"])

        # Move model to device.
        model = model.to(device)

        # Create data loaders for training and validation.
        train_loader, val_loader = create_data_loaders(
            training_dset, validation_dset, batch_size=batch_size
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
        weights_file = f"{encoder}_batch{batch_size}_lr{lr_initial}_weights.pt"
        loss_curve_csv = f"{encoder}_batch{batch_size}_lr{lr_initial}_losscurve.csv"

        # Begin model training.
        run.watch(model)
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
            run.log(
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
def run_wandb_training(config: ModelTrainingConfig) -> None:
    """
    Materialize data assets and conduct model training with W&B integration.
    """
    # Set up model.
    unet_config = BasicUnetConfig(encoder_name=config.encoder_name)
    model = unet_model(unet_config)

    # Materialize data assets.
    data_assets = [
        data_training,
        data_validation,
        afs_training_dataset,
        afs_validation_dataset,
    ]
    result = materialize_to_memory(data_assets)
    training_dataset = result.asset_value("training_dataset")
    validation_dataset = result.asset_value("validation_dataset")

    # Create W&B config dictionary.
    encoder = config.encoder_name
    batch_size = config.batch_size
    lr_initial = config.lr_initial
    wandb_config = {
        "name": f"{encoder}_batch{batch_size}_lr{lr_initial}",
        "seed": config.seed,
        "encoder_name": encoder,
        "batch_size": batch_size,
        "lr_initial": lr_initial,
        "max_epochs": config.max_epochs,
    }

    # Call model training op.
    train_unet(
        config.project, wandb_config, training_dataset, validation_dataset, model
    )
