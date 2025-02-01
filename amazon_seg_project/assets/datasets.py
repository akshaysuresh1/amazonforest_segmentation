"""
Assets for training, validation, and test datasets
"""

from typing import List
from dagster import (
    asset,
    AssetIn,
    AssetExecutionContext,
    AssetMaterialization,
    Field,
    Float,
)
from . import SegmentationDataset
from ..resources import s3_resource, AMAZON_TIF_BUCKET


@asset(
    name="training_dataset",
    ins={"train_image_files": AssetIn(), "train_mask_files": AssetIn()},
    config_schema={
        "horizontal_flip_prob": Field(Float, default_value=0.5),
        "vertical_flip_prob": Field(Float, default_value=0.5),
        "rotate90_prob": Field(Float, default_value=0.5),
    },
)
def afs_training_dataset(
    context: AssetExecutionContext,
    train_image_files: List[str],
    train_mask_files: List[str],
) -> SegmentationDataset:
    """
    Training dataset for Amazon forest segmentation
    """
    training_dataset = SegmentationDataset(
        images_list=train_image_files,
        masks_list=train_mask_files,
        s3_resource=s3_resource,
        s3_bucket=AMAZON_TIF_BUCKET.get_value(),
        do_aug=True,
        horizontal_flip_prob=context.op_config["horizontal_flip_prob"],
        vertical_flip_prob=context.op_config["vertical_flip_prob"],
        rotate90_prob=context.op_config["rotate90_prob"],
    )
    # Emit metadata
    context.log_event(
        AssetMaterialization(
            asset_key="training_dataset",
            metadata={"Training dataset length": len(training_dataset)},
        )
    )
    return training_dataset


@asset(
    name="validation_dataset",
    ins={"val_image_files": AssetIn(), "val_mask_files": AssetIn()},
)
def afs_validation_dataset(
    context: AssetExecutionContext,
    val_image_files: List[str],
    val_mask_files: List[str],
) -> SegmentationDataset:
    """
    Validation dataset for Amazon forest segmentation
    """
    validation_dataset = SegmentationDataset(
        images_list=val_image_files,
        masks_list=val_mask_files,
        s3_resource=s3_resource,
        s3_bucket=AMAZON_TIF_BUCKET.get_value(),
    )
    # Emit metadata
    context.log_event(
        AssetMaterialization(
            asset_key="validation_dataset",
            metadata={"Validation dataset length": len(validation_dataset)},
        )
    )
    return validation_dataset
