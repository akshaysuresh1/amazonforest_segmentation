"""
Assets for training, validation, and test datasets
"""

from typing import List, Generator
from dagster import asset, AssetIn, Output
from . import SegmentationDataset
from ..config import TrainingDatasetConfig
from ..ops.aug_utils import get_aug_pipeline
from ..ops.scaling_utils import robust_scaling
from ..resources import AMAZON_TIF_BUCKET


@asset(
    name="training_dataset",
    ins={"train_image_files": AssetIn(), "train_mask_files": AssetIn()},
)
def afs_training_dataset(
    config: TrainingDatasetConfig,
    train_image_files: List[str],
    train_mask_files: List[str],
) -> Generator[Output, None, None]:
    """
    Training dataset for Amazon forest segmentation
    """
    training_dataset = SegmentationDataset(
        images_list=train_image_files,
        masks_list=train_mask_files,
        s3_bucket=AMAZON_TIF_BUCKET.get_value() or "",
        scaling_func=robust_scaling,
        transform=get_aug_pipeline(
            horizontal_flip_prob=config.horizontal_flip_prob,
            vertical_flip_prob=config.vertical_flip_prob,
            rotate90_prob=config.rotate90_prob,
            augmentation_seed=config.augmentation_seed,
        ),
    )
    yield Output(
        training_dataset, metadata={"Training dataset length": len(training_dataset)}
    )


@asset(
    name="validation_dataset",
    ins={"val_image_files": AssetIn(), "val_mask_files": AssetIn()},
)
def afs_validation_dataset(
    val_image_files: List[str],
    val_mask_files: List[str],
) -> Generator[Output, None, None]:
    """
    Validation dataset for Amazon forest segmentation
    """
    validation_dataset = SegmentationDataset(
        images_list=val_image_files,
        masks_list=val_mask_files,
        s3_bucket=AMAZON_TIF_BUCKET.get_value() or "",
        scaling_func=robust_scaling,
        transform=None,
    )
    yield Output(
        validation_dataset,
        metadata={"Validation dataset length": len(validation_dataset)},
    )
