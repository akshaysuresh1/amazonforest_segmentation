"""
Config schema for asset definitions
"""

from dagster import Config
from pydantic import Field


class TrainingDatasetConfig(Config):
    """
    Configurable parameters to asset "training_dataset"
    """

    horizontal_flip_prob: float = Field(default=0.5)
    vertical_flip_prob: float = Field(default=0.5)
    rotate90_prob: float = Field(default=0.5)


class PretrainedUnetConfig(Config):
    """
    Configurable parameters to asset "pretrained_unet"
    """

    encoder: str = Field(default="resnet50")
    in_channels: int = Field(default=4)
    activation: str = Field(default="sigmoid")


class FinetunedUnetConfig(Config):
    """
    Configurable parameters to asset "finetuned_unet
    """

    batch_size: int = Field(default=64)
