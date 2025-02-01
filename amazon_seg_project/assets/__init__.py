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
from .statistics import channel_stats_training_data
