"""
Deep learning models for image segmentation
"""

from typing import Generator
import logging
from tqdm import tqdm
from segmentation_models_pytorch import Unet
from dagster import asset, AssetIn, Output
from . import SegmentationDataset
from ..config import PretrainedUnetConfig, FinetunedUnetConfig
from ..data_paths import OUTPUT_PATH
from ..ops import (
    dice_loss,
    create_data_loaders,
    setup_adam_w,
    train_epoch,
    validate_epoch,
    save_model_weights,
    write_loss_data_to_csv,
)
from ..resources import device


@asset(name="pretrained_unet")
def pretrained_unet_model(
    config: PretrainedUnetConfig,
) -> Generator[Output, None, None]:
    """
    A U-Net model whose encoder has been pretrained on imagenet data
    """
    model = Unet(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=config.in_channels,
        activation=config.activation,
    )
    model = model.to(device)

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
    pretrained_unet: Unet,
    training_dataset: SegmentationDataset,
    validation_dataset: SegmentationDataset,
) -> Unet:
    """
    U-Net model finetuned on Amazon forest satellite imagery
    """
    # Move pretrained model to device of choice for training.
    model = pretrained_unet
    model = model.to(device)

    # Create data loaders for training and validation.
    train_loader, val_loader = create_data_loaders(
        training_dataset, validation_dataset, batch_size=config.batch_size
    )

    # Set up optimizer and loss criterion.
    optimizer = setup_adam_w(model, lr_initial=config.lr_initial)
    criterion = dice_loss

    # Track minimum validation loss observed across epochs.
    lowest_val_loss = float("inf")
    # Store batch-averaged training and validation loss at every epoch.
    train_loss = []
    val_loss = []
    # Track epochs since last improvement in validation loss.
    epochs_since_improvement = 0

    # Log training config parameters.
    logging.info(
        "Starting training:\n"
        "Max no. of epochs = %d\n"
        "Batch size = %d\n"
        "Initial learning rate = %.1g\n"
        "Training dataset size = %d\n"
        "Validation dataset size = %d\n"
        "Device = %s",
        config.max_epochs,
        config.batch_size,
        config.lr_initial,
        len(training_dataset),
        len(validation_dataset),
        device.type,
    )

    # Begin model training.
    for epoch in tqdm(range(1, config.max_epochs + 1)):
        # Training step
        current_train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        train_loss.append(current_train_loss)

        # Validation step
        current_val_loss = validate_epoch(model, val_loader, criterion, device)
        val_loss.append(current_val_loss)

        # Log epoch results.
        logging.info(
            "Statistics for epoch %d: \n Training loss: %.4f \n Validation loss: %.4f",
            epoch,
            train_loss[-1],
            val_loss[-1],
        )

        # Check if validation loss improved.
        if val_loss[-1] < lowest_val_loss:
            lowest_val_loss = val_loss[-1]
            epochs_since_improvement = 0  # Reset counter
            logging.info(
                "Achieved new minimum validation loss. Writing model weights to disk."
            )
            save_model_weights(model, OUTPUT_PATH / "model_weights.pt")
        else:
            epochs_since_improvement += 1  # Increment counter if no improvement
            logging.info(
                "Validation loss did not improve. Epochs since last improvement: %d",
                epochs_since_improvement,
            )

        # Implement early stopping criterion.
        if epochs_since_improvement == 5:
            logging.info(
                "Early stopping criterion met. Stopping training after epoch %d.",
                epoch,
            )
            write_loss_data_to_csv(
                train_loss, val_loss, OUTPUT_PATH / "train" / "loss_curve.csv"
            )
            break

        # Save loss curve data to disk intermittently.
        if epoch % 10 == 0 or epoch == config.max_epochs:
            write_loss_data_to_csv(
                train_loss, val_loss, OUTPUT_PATH / "train" / "loss_curve.csv"
            )

    return model
