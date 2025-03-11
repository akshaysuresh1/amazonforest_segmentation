"""
Deep learning models for image segmentation
"""

from typing import Generator
from segmentation_models_pytorch import Unet
from dagster import asset, Output
from ..config import PretrainedUnetConfig
from ..resources import device


@asset(name="basic_unet_model")
def unet_model(
    config: PretrainedUnetConfig,
) -> Generator[Output, None, None]:
    """
    A U-Net model with frozen encoder weights
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
