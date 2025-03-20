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
    augmentation_seed: int = Field(default=137)


class BasicUnetConfig(Config):
    """
    Configurable parameters to asset "basic_unet_model"
    """

    encoder_name: str = Field(default="resnet50")
    encoder_weights: str | None = Field(default="imagenet")
    in_channels: int = Field(default=4)
    activation: str = Field(default="sigmoid")


class ModelTrainingConfig(Config):
    """
    Configurable parameters for a W&B model training run
    """

    # W&B settings
    project: str = Field(default="amazonforest_segmentation")
    # ML parameters
    seed: int = Field(default=227)
    batch_size: int = Field(default=8)
    lr_initial: float = Field(default=1.0e-5)
    max_epochs: int = Field(default=10)
    encoder_name: str = Field(default="resnet50")
