"""
Deep learning models for image segmentation
"""

import os
from typing import Generator
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
from dagster import asset, AssetIn, Output
from torch.utils.data import DataLoader
import torch.optim as optim
from . import SegmentationDataset
from ..config import PretrainedUnetConfig, FinetunedUnetConfig
from ..data_paths import OUTPUT_PATH
from ..ops import dice_loss
from ..resources import device


@asset(name="pretrained_unet")
def pretrained_unet_model(
    config: PretrainedUnetConfig,
) -> Generator[Output, None, None]:
    """
    A U-Net model whose encoder has been pretrained on imagenet data
    """
    model = smp.Unet(
        encoder=config.encoder,
        encoder_weights="imagenet",
        in_channels=config.in_channels,
        activation=config.activation,
    ).to(device)

    # Freeze pretrained encoder weights.
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Yield model and emit metadata to Dagster UI.
    total_param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    yield Output(
        model,
        metadata={
            "Total parameter count": total_param_count,
            "Trainable parameter count": trainable_param_count,
        },
    )


@asset(
    name="finetuned_unet",
    ins={
        "pretrained_unet": AssetIn(),
        "training_dataset": AssetIn(),
        "validation_dataset": AssetIn(),
    },
)
def finetuned_unet_model(
    config: FinetunedUnetConfig,
    pretrained_unet: smp.Unet,
    training_dataset: SegmentationDataset,
    validation_dataset: SegmentationDataset,
) -> smp.Unet:
    """
    U-Net model finetuned on Amazon forest satellite imagery
    """
    # Set up data loaders for training and validation.
    loader_args = dict(
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=os.cpu_count(),
    )
    train_loader = DataLoader(
        training_dataset, shuffle=True, drop_last=False, **loader_args
    )
    val_loader = DataLoader(
        validation_dataset, shuffle=False, drop_last=False, **loader_args
    )

    # Set up optimizer and loss criterion.
    optimizer = optim.AdamW(pretrained_unet.parameters(), lr=config.lr_initial)
    criterion = dice_loss
    # Track minimum validation loss observed across epochs.
    lowest_val_loss = float("inf")
    # Store batch-averaged training and validation loss at every epoch.
    train_loss = []
    val_loss = []

    # Begin model training.
    logging.info("Starting training:")
    print(f"Max no. of epochs = {config.max_epochs}")
    print(f"Batch size = {config.batch_size}")
    print(f"Initial learning rate = {config.lr_initial}")
    print(f"Training dataset size = {len(training_dataset)}")
    print(f"Validation dataset size = {len(validation_dataset)}")
    print(f"Device = {device.type}")

    for epoch in tqdm(range(1, config.max_epochs + 1)):
        # Training phase
        pretrained_unet.train()
        current_train_loss = 0  # Stores average training loss over batches
        for batch in train_loader:
            images, true_masks = batch
            # Set model parameter gradients to zero before every batch training.
            optimizer.zero_grad()
            # Compute predicted mask and loss per batch.
            pred_masks = pretrained_unet(images)
            loss = criterion(pred_masks, true_masks)
            current_train_loss += loss.item() * true_masks.shape[0]
            # Batch gradient descent
            loss.backward()
            optimizer.step()
        current_train_loss /= len(training_dataset)
        train_loss.append(current_train_loss)

        # Validation phase
        pretrained_unet.eval()
        current_val_loss = 0.0  # Stores average validation loss over batches
        with torch.no_grad():
            for batch in val_loader:
                images, true_masks = batch
                # Compute predicted mask and loss per batch.
                pred_masks = pretrained_unet(images)
                loss = criterion(pred_masks, true_masks)
                current_val_loss += loss.item() * true_masks.shape[0]
        current_val_loss /= len(validation_dataset)
        val_loss.append(current_val_loss)

        logging.info(
            "Statistics for epoch %d: \n Training loss: %.4f \n Validation loss: %.4f",
            epoch,
            train_loss[-1],
            val_loss[-1],
        )

        # Write model weights to disk.
        if val_loss[-1] < lowest_val_loss:
            lowest_val_loss = val_loss[-1]
            logging.info(
                "Achieved new minimum validation loss. Writing model weights to disk."
            )
            torch.save(
                pretrained_unet.state_dict(), str(OUTPUT_PATH / "model_weights.pth")
            )

        # Save loss curve data to disk intermittently.
        if epoch % 10 == 0 or epoch == config.max_epochs:
            loss_df = pd.DataFrame(
                {"train_loss": np.array(train_loss), "val_loss": np.array(val_loss)},
                index=np.arange(1, epoch + 1),
            )
            loss_df.to_csv(str(OUTPUT_PATH / "model_loss.csv"), index=False)

    return pretrained_unet
