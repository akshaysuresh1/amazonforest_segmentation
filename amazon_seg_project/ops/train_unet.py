"""
Op for U-Net training
"""

from tqdm import tqdm
import logging
import torch
from dagster import op, In
from dagster import Any as dg_Any
from wandb.sdk.wandb_run import Run
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import Unet
from .file_naming_conventions import name_weights_file, name_losscurve_csv_file
from .torch_utils import (
    create_data_loaders,
    setup_adam_w,
    train_epoch,
    validate_epoch,
    save_model_weights,
)
from .wandb_artifact_utils import create_and_log_wandb_artifact
from .write_files import write_loss_data_to_csv
from ..assets import SegmentationDataset
from ..data_paths import OUTPUT_PATH
from ..resources import device


@op(
    ins={
        "wandb_run": In(dg_Any),
        "training_dset": In(dg_Any),
        "validation_dset": In(dg_Any),
        "model": In(dg_Any),
    }
)
def train_unet(
    wandb_run: Run,
    training_dset: SegmentationDataset,
    validation_dset: SegmentationDataset,
    model: Unet,
) -> None:
    """
    Train a U-net using batch gradient descent and log run results to W&B.

    Args:
        wandb_run: Weights & Biases SDK Run object created with wandb.init()
        training_dset: Training dataset
        validation_dset: Validation dataset
        model: U-net model
        val_metrics: List of metrics to be evaluated on validation data
    """
    config = wandb_run.config
    seed = config.get("seed")
    encoder = config.get("encoder_name")
    batch_size = config.get("batch_size")
    lr_initial = config.get("lr_initial")
    threshold = config.get("threshold")
    max_epochs = config.get("max_epochs")

    if max_epochs < 1:
        logging.info("No. of training epochs < 1. Model training skipped.")
        return  # Exit the function

    # Set PyTorch seed.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Move model to device.
    model = model.to(device)
    # Utilize multiple GPUs if available.
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Create data loaders for training and validation.
    train_loader, val_loader = create_data_loaders(
        training_dset,
        validation_dset,
        batch_size=batch_size,
        seed=seed,
    )

    # Set up optimizer and loss criterion.
    optimizer = setup_adam_w(model, lr_initial=lr_initial)
    criterion = smp.losses.DiceLoss(
        mode="binary", from_logits=False, smooth=1.0e-6, eps=0.0
    )

    # Track minimum validation loss observed across epochs.
    lowest_val_loss = float("inf")
    # Store batch-averaged training and validation loss at every epoch.
    train_loss = []
    val_loss = []
    # Track epochs since last improvement in validation loss.
    epochs_since_improvement = 0

    # Set file names reflecting training configuration.
    weights_file = name_weights_file(encoder, batch_size, lr_initial)
    loss_curve_csv = name_losscurve_csv_file(encoder, batch_size, lr_initial)

    # Begin model training.
    wandb_run.watch(model)
    for epoch in tqdm(range(1, max_epochs + 1)):
        # Training step
        current_train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        train_loss.append(current_train_loss)

        # Validation step
        val_results = validate_epoch(model, val_loader, criterion, device, threshold)
        val_loss.append(val_results.get("val_loss"))

        # W&B logging
        wandb_run.log(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": current_train_loss,
                **val_results,
            }
        )

        # Check if validation loss improved.
        if val_loss[-1] < lowest_val_loss:
            lowest_val_loss = val_loss[-1]
            current_best_stats = val_results
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
        if epoch % 10 == 0 or epoch == config.get("max_epochs"):
            write_loss_data_to_csv(
                train_loss, val_loss, OUTPUT_PATH / "train" / loss_curve_csv
            )

    # Create and log a W&B artifact for the weights file.
    create_and_log_wandb_artifact(
        wandb_run, str(OUTPUT_PATH / weights_file), current_best_stats
    )
