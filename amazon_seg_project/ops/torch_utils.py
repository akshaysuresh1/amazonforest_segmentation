"""
Utility functions for model training, validation, and inference
"""

import os
import random
import logging
from typing import Dict, Tuple, Union, Any
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from dagster import op, In, Out
from dagster import Any as dg_Any
from .metrics import smp_metrics
from .write_files import create_directories


def worker_seed_fn(worker_id: int) -> None:
    """
    Initializes a worker's random seed based on worker_id.

    Args:
        worker_id (int): The ID of the worker, which ensures a unique seed.
    """
    # Use the worker_id to ensure different seeds for each worker
    seed = (
        torch.initial_seed() % 2**32 + worker_id
    )  # Ensures a unique seed for each worker.
    np.random.seed(seed)  # Set the seed for NumPy's random generator.
    random.seed(seed)  # Set the seed for Python's random module.
    torch.manual_seed(seed)  # Set PyTorch's seed for the worker.


@op(
    ins={
        "training_dset": In(dg_Any),
        "validation_dset": In(dg_Any),
        "batch_size": In(int),
        "num_workers": In(int),
    },
    out=Out(dg_Any),
)
def create_data_loaders(
    training_dset: Any,
    validation_dset: Any,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 137,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates data loaders for training and validation datasets.

    Args:
        training_det: Training dataset
        validation_dset: Validation dataset
        batch_size: Batch size for the data loaders
        num_workers: Number of workers for data loading (defaults to os.cpu_count())
        seed: Seed for generator

    Returns: Training and validation data loaders
    """
    max_workers = os.cpu_count() or 1
    if num_workers <= 0 or num_workers > max_workers:
        num_workers = max_workers

    # Set manual seed for generator.
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        training_dset,
        shuffle=True,
        drop_last=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=worker_seed_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        validation_dset,
        shuffle=False,
        drop_last=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=worker_seed_fn,
        generator=generator,
    )
    return train_loader, val_loader


@op(ins={"model": In(dg_Any), "lr_initial": In(float)}, out=Out(dg_Any))
def setup_adam_w(model: torch.nn.Module, lr_initial: float = 1.0e-4) -> optim.AdamW:
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
        "model": In(dg_Any),
        "train_loader": In(dg_Any),
        "optimizer": In(dg_Any),
        "criterion": In(dg_Any),
        "train_device": In(dg_Any),
    },
    out=Out(float),
)
def train_epoch(
    model: torch.nn.Module,
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
    model.to(train_device)
    model.train()
    summed_train_loss = 0.0
    n_training_samples = 0  # No. of training samples
    for batch in train_loader:
        # Images shape = (batch size, color channels, height, width)
        # Masks shape = (batch size, 1, height, weight)
        images, true_masks = batch
        # Move data to device.
        images, true_masks = images.to(train_device), true_masks.to(train_device)
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
        "model": In(dg_Any),
        "val_loader": In(dg_Any),
        "criterion": In(dg_Any),
        "val_device": In(dg_Any),
        "threshold": In(float),
    },
    out=Out(dg_Any),
)
def validate_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: Any,
    val_device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    """
    Perform one epoch of model validation

    Args:
        model: The PyTorch U-Net model to validate
        val_loader: The validation data loader
        criterion: The loss function
        val_device: The device to validate on (CPU or CUDA)
        threshold: Binarization threshold in the range [0, 1]

    Returns: Dictionary comprising of validation metric values and batch-averaged loss
    """
    model.to(val_device)
    model.eval()
    summed_val_loss = 0.0
    n_val_samples = 0  # No. of validation samples
    # Store results of metric evaluations on validation data.
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1_score = 0.0
    iou_score = 0.0
    with torch.no_grad():
        for batch in val_loader:
            # Images shape = (batch size, color channels, height, width)
            # Masks shape = (batch size, 1, height, weight)
            images, true_masks = batch
            curr_batch_size = true_masks.shape[0]
            # Move data to device.
            images, true_masks = images.to(val_device), true_masks.to(val_device)
            # Obtain model predictions and compute loss.
            pred_masks = model(images)
            loss = criterion(pred_masks, true_masks)
            summed_val_loss += loss.item() * curr_batch_size
            n_val_samples += curr_batch_size
            # Compute metrics on batched validation data.
            metric_values = smp_metrics(
                pred_masks, true_masks.int(), threshold=threshold
            )
            accuracy += metric_values.get("Accuracy") * curr_batch_size
            precision += metric_values.get("Precision") * curr_batch_size
            recall += metric_values.get("Recall") * curr_batch_size
            f1_score += metric_values.get("F1 score") * curr_batch_size
            iou_score += metric_values.get("IoU") * curr_batch_size

    # Calculate batch-averaged values of validation loss and metrics.
    if n_val_samples == 0:
        raise ValueError("No validation data found.")
    avg_val_loss = summed_val_loss / n_val_samples
    accuracy /= n_val_samples
    precision /= n_val_samples
    recall /= n_val_samples
    f1_score /= n_val_samples
    iou_score /= n_val_samples

    return {
        "val_loss": avg_val_loss,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 score": f1_score,
        "IoU": iou_score,
    }


@op(ins={"model": In(dg_Any), "filepath": In(dg_Any)})
def save_model_weights(
    model: torch.nn.Module, filepath: Union[str, os.PathLike]
) -> None:
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
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), str(filepath))
    else:
        torch.save(model.state_dict(), str(filepath))
    logging.info("Saved model weights to %s", str(filepath))


@op(ins={"prob_tensor": In(dg_Any), "threshold_prob": In(float)}, out=Out(dg_Any))
def apply_prob_thresholding(
    prob_tensor: torch.Tensor, threshold_prob: float
) -> torch.Tensor:
    """
    Apply thresholding on a tensor of probabilities.
    Probability values <= threshold are mapped to 0.
    Probability values > threshold are mapped to 1.

    Args:
        prob_tensor: Probability tensor
        threshold_prob: Threshold probability

    Returns: Thresholded binary tensor
    """
    if prob_tensor.max() > 1.0:
        raise ValueError("Input tensor does not contain probabilities. Max value > 1.")

    if prob_tensor.min() < 0.0:
        raise ValueError("Input tensor does not contain probabilities. Min value < 0.")

    if not 0 <= threshold_prob <= 1.0:
        raise ValueError("Threshold value must be a probability.")

    binary_tensor = (prob_tensor > threshold_prob).int()
    return binary_tensor
