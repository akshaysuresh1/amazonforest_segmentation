"""
Deep learning models for image segmentation
"""

from typing import Generator
import torch
from dagster import asset, Output
from segmentation_models_pytorch import Unet
from ..config import BasicUnetConfig


@asset(name="basic_unet_model")
def unet_model(config: BasicUnetConfig) -> Generator[Output, None, None]:
    """
    A U-Net model with frozen encoder weights
    """
    torch.manual_seed(config.model_seed)
    model = Unet(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=config.in_channels,
        activation=config.activation,
    )

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
