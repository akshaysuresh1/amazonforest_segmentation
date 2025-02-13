"""
Deep learning models for image segmentation
"""

import os
from typing import Generator
import segmentation_models_pytorch as smp
from dagster import asset, AssetIn, Output
from torch.utils.data import DataLoader
from . import SegmentationDataset
from ..config import PretrainedUnetConfig, FinetunedUnetConfig
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
    _ = DataLoader(training_dataset, shuffle=True, drop_last=False, **loader_args)
    _ = DataLoader(validation_dataset, shuffle=False, drop_last=False, **loader_args)

    return pretrained_unet
