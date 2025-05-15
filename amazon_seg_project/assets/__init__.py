"""
Load data assets.
"""

from .data_products import (
    data_training,
    data_validation,
    data_test,
)
from .dataset_definition import SegmentationDataset
from .datasets import afs_training_dataset, afs_validation_dataset
from .model import unet_model, trained_unet_model
from .precision_recall_metrics import precision_recall_curve
from .statistics import channel_stats_training_data

__all__ = [
    "data_training",
    "data_validation",
    "data_test",
    "SegmentationDataset",
    "afs_training_dataset",
    "afs_validation_dataset",
    "unet_model",
    "trained_unet_model",
    "precision_recall_curve",
    "channel_stats_training_data",
]
