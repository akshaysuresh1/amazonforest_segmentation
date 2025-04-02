"""
Config schema for asset definitions
"""

from typing import Dict, Any
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
    model_seed: int = Field(default=137)


class SweepConfig(Config):
    """
    Configurable parameters for a W&B sweep
    """

    entity: str = Field(default="akshaysuresh1")
    project: str = Field(default="amazonforest_segmentation")
    # Optimization method and metrics
    method: str = Field(default="grid")
    metric_name: str = Field(default="val_loss")
    metric_goal: str = Field(default="minimize")
    # Mask binarization threshold
    threshold: Dict[str, Any] = Field(default={"values": [0.5]})
    # Model training arameters
    seed: Dict[str, Any] = Field(default={"values": [43]})
    encoder_name: Dict[str, Any] = Field(default={"values": ["resnet50"]})
    batch_size: Dict[str, Any] = Field(default={"values": [4]})
    lr_initial: Dict[str, Any] = Field(default={"values": [1.0e-4]})
    max_epochs: Dict[str, Any] = Field(default={"values": [10]})
    # Data augmentation parameters
    horizontal_flip_prob: Dict[str, Any] = Field(default={"values": [0.5]})
    vertical_flip_prob: Dict[str, Any] = Field(default={"values": [0.5]})
    rotate90_prob: Dict[str, Any] = Field(default={"values": [0.5]})
