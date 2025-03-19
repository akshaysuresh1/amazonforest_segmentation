"""
Deep learning models for image segmentation
"""

import logging
from segmentation_models_pytorch import Unet
from ..config import BasicUnetConfig


def unet_model(model_config: BasicUnetConfig) -> Unet:
    """
    A U-Net model with frozen encoder weights
    """
    model = Unet(
        encoder_name=model_config.encoder_name,
        encoder_weights=model_config.encoder_weights,
        in_channels=model_config.in_channels,
        activation=model_config.activation,
    )

    # Freeze pretrained encoder weights.
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Log model parameter count.
    total_param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logging.info("Total parameter count = %d", total_param_count)
    logging.info("Trainable parameter count = %d", trainable_param_count)

    return model
