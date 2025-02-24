"""
Utility functions for model training, validation, and inference
"""

import os
import logging
from typing import Any, Tuple, Union
import torch
from torch import optim
from torch.utils.data import DataLoader
from dagster import op, In, Out
import segmentation_models_pytorch as smp
from .write_files import create_directories
from ..resources import device


@op(
    ins={
        "training_dset": In(Any),
        "validation_dset": In(Any),
        "batch_size": In(int),
        "num_workers": In(int),
    },
    out=Out(Any),
)
def create_data_loaders(
    training_dset: Any,
    validation_dset: Any,
    batch_size: int = 8,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates data loaders for training and validation datasets.

    Args:
        training_det: Training dataset
        validation_dset: Validation dataset
        batch_size: Batch size for the data loaders
        num_workers: Number of workers for data loading (defaults to os.cpu_count())

    Returns: Training and validation data loaders
    """
    max_workers = os.cpu_count()
    if num_workers <= 0 or num_workers > max_workers:
        num_workers = max_workers

    loader_args = dict(
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    train_loader = DataLoader(
        training_dset, shuffle=True, drop_last=False, **loader_args
    )
    val_loader = DataLoader(
        validation_dset, shuffle=False, drop_last=False, **loader_args
    )
    return train_loader, val_loader


@op(ins={"model": In(Any), "lr_initial": In(float)}, out=Out(Any))
def setup_adam_w(model: smp.Unet, lr_initial: float = 1.0e-4) -> optim.AdamW:
    """
    Sets up the AdamW optimizer for training the given U-Net model.

    Args:
        model: PyTorch U-Net model to optimize.
        lr_initial: Initial learning rate.

    Returns:
        torch.optim.AdamW: The configured optimizer.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr_initial)
    return optimizer


@op(
    ins={
        "model": In(Any),
        "train_loader": In(Any),
        "optimizer": In(Any),
        "criterion": In(Any),
        "train_device": In(Any),
    },
    out=Out(float),
)
def train_epoch(
    model: smp.Unet,
    train_loader: DataLoader,
    optimizer: Any,
    criterion: Any,
    train_device: torch.device,
) -> float:
    """
    Trains a U-Net model for one epoch.

    Args:
        model: The PyTorch U-Net model to train
        train_loader: The training data loader
        optimizer: The optimizer
        criterion: The loss function
        train_device: The device to train on (CPU or CUDA).

    Returns: Batch-averaged training loss for the epoch
    """
    model.train()
    summed_train_loss = 0.0
    n_training_samples = 0  # No. of training samples
    for batch in train_loader:
        # Images shape = (batch size, color channels, height, width)
        # Masks shape = (batch size, 1, height, weight)
        images, true_masks = batch
        # Move data to device.
        images, true_masks = images.to(device), true_masks.to(train_device)
        # Reset gradients to zero before each batch.
        optimizer.zero_grad()
        # Obtain model predictions on data.
        pred_masks = model(images)
        # Loss computation
        loss = criterion(pred_masks, true_masks)
        summed_train_loss += loss.item() * true_masks.shape[0]
        n_training_samples += true_masks.shape[0]
        # Backpropagation
        loss.backward()
        optimizer.step()
    # Calculate batch-averaged training loss.
    if n_training_samples == 0:
        raise ValueError("No training data found.")
    avg_train_loss = summed_train_loss / n_training_samples
    return avg_train_loss


@op(
    ins={
        "model": In(Any),
        "val_loader": In(Any),
        "criterion": In(Any),
        "val_device": In(Any),
    },
    out=Out(float),
)
def validate_epoch(
    model: smp.Unet, val_loader: DataLoader, criterion: Any, val_device: torch.device
) -> float:
    """
    Perform one epoch of model validation

    Args:
        model: The PyTorch U-Net model to validate
        val_loader: The validation data loader
        criterion: The loss function
        val_device: The device to validate on (CPU or CUDA)

    Returns: Batch-averaged validation loss for the epoch
    """
    model.eval()
    summed_val_loss = 0.0
    n_val_samples = 0  # No. of validation samples
    with torch.no_grad():
        for batch in val_loader:
            # Images shape = (batch size, color channels, height, width)
            # Masks shape = (batch size, 1, height, weight)
            images, true_masks = batch
            # Move data to device.
            images, true_masks = images.to(val_device), true_masks.to(val_device)
            # Obtain model predictions and compute loss.
            pred_masks = model(images)
            loss = criterion(pred_masks, true_masks)
            summed_val_loss += loss.item() * true_masks.shape[0]
            n_val_samples += true_masks.shape[0]
    # Calculate batch-averaged validation loss.
    if n_val_samples == 0:
        raise ValueError("No validation data found.")
    avg_val_loss = summed_val_loss / n_val_samples
    return avg_val_loss


@op(ins={"model": In(Any), "filepath": In(Any)})
def save_model_weights(model: smp.Unet, filepath: Union[str, os.PathLike]) -> None:
    """
    Saves U-Net model weights to the specified filepath.

    Args:
        model: The PyTorch model to save.
        path: File name (including path) of output weights file
    """
    # Use a .pt extensions for model weights file.
    if not str(filepath).endswith(".pt"):
        filepath = str(filepath) + ".pt"
    create_directories(filepath)
    torch.save(model.state_dict(), str(filepath))
    logging.info("Saved model weights to %s", str(filepath))
