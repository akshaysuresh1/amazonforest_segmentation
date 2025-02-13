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
from .model import pretrained_unet_model, finetuned_unet_model
from .statistics import channel_stats_training_data
