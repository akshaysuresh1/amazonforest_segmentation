"""
Deep learning models for image segmentation
"""

import os
from typing import Generator
import torch
import wandb
from dagster import asset, Output
from segmentation_models_pytorch import Unet
from ..config import BasicUnetConfig, TrainedUnetConfig
from ..ops.file_naming_conventions import name_weights_file
from ..ops.wandb_artifact_utils import check_artifact_exists
from ..resources import device


@asset(name="basic_unet_model")
def unet_model(config: BasicUnetConfig) -> Generator[Output, None, None]:
    """
    A U-Net model with the option to freeze encoder weights
    """
    torch.manual_seed(config.model_seed)
    model = Unet(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=config.in_channels,
        activation=config.activation,
    )

    # Freeze encoder weights if specified.
    if config.freeze_encoder_weights:
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


@asset(name="trained_unet_model")
def trained_unet_model(config: TrainedUnetConfig) -> Generator[Output, None, None]:
    """
    A U-Net model that has been trained on Sentinel-2 Amazon rainforest images.

    Model weights are downloaded from an online Weights & Biases artifact.
    """
    if not check_artifact_exists(
        config.wandb_artifact_path, config.wandb_artifact_version
    ):
        raise ValueError(
            "Model weights file not available for loading into U-net model."
        )

    api = wandb.Api()
    artifact = api.artifact(
        f"{config.wandb_artifact_path}:{config.wandb_artifact_version}"
    )
    # Extract required metadata.
    encoder = artifact.metadata.get("encoder")
    batch_size = artifact.metadata.get("batch_size")
    lr_initial = artifact.metadata.get("lr_initial")
    weights_file = name_weights_file(encoder, batch_size, lr_initial)
    # Download artifact.
    local_dir = artifact.download(path_prefix=weights_file)

    # Initialize U-Net model.
    model = Unet(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=config.in_channels,
        activation=config.activation,
    )

    # Load model weights.
    state_dict = torch.load(os.path.join(local_dir, weights_file), map_location=device)
    model.load_state_dict(state_dict)

    # Obtain total number of model parameters.
    total_param_count = sum(p.numel() for p in model.parameters())
    yield Output(model, metadata={"Total parameter count": total_param_count})
