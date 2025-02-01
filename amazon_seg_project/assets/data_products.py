"""
Data files:
1. Training data: Images, Masks
2. Validation data: Images, Masks
3. Test data
"""

from typing import List, Generator
from dagster import multi_asset, asset, AssetOut, Output
from ..data_paths import (
    TRAINING_IMAGES_PREFIX,
    TRAINING_MASKS_PREFIX,
    VALIDATION_IMAGES_PREFIX,
    VALIDATION_MASKS_PREFIX,
    TEST_IMAGES_PREFIX,
)
from ..resources import s3_resource, AMAZON_TIF_BUCKET
from ..ops import load_labeled_data
from ..ops.s3_utils import list_objects


@multi_asset(
    outs={
        "train_image_files": AssetOut(
            dagster_type=List[str], description="Training image files"
        ),
        "train_mask_files": AssetOut(
            dagster_type=List[str], description="Training mask files"
        ),
    }
)
def data_training() -> Generator[Output, None, None]:
    """
    Retrieve training data and masks from S3 bucket
    """
    train_image_files, train_mask_files = load_labeled_data(
        s3_resource,
        AMAZON_TIF_BUCKET.get_value(),
        str(TRAINING_IMAGES_PREFIX),
        str(TRAINING_MASKS_PREFIX),
        ".tif",
        ".tif",
    )

    # Raise ValueError if no ".tif" image files have been found.
    if not train_image_files:
        raise ValueError(
            f"Zero .tif images with masks found at {str(TRAINING_IMAGES_PREFIX)}"
        )

    yield Output(
        train_image_files,
        output_name="train_image_files",
        metadata={"description": f"Count of images = {len(train_image_files)}"},
    )
    yield Output(
        train_mask_files,
        output_name="train_mask_files",
        metadata={"description": f"Count of masks = {len(train_mask_files)}"},
    )


@multi_asset(
    outs={
        "val_image_files": AssetOut(
            dagster_type=List[str], description="Validation image files"
        ),
        "val_mask_files": AssetOut(
            dagster_type=List[str], description="Validation mask files"
        ),
    }
)
def data_validation() -> Generator[Output, None, None]:
    """
    Retrieve validation data and masks from S3 bucket
    """
    val_image_files, val_mask_files = load_labeled_data(
        s3_resource,
        AMAZON_TIF_BUCKET.get_value(),
        str(VALIDATION_IMAGES_PREFIX),
        str(VALIDATION_MASKS_PREFIX),
        ".tif",
        ".tif",
    )

    # Raise ValueError if no ".tif" image files have been found.
    if not val_image_files:
        raise ValueError(
            f"Zero .tif images with masks found at {str(VALIDATION_IMAGES_PREFIX)}"
        )

    yield Output(
        val_image_files,
        output_name="val_image_files",
        metadata={"description": f"Count of images = {len(val_image_files)}"},
    )
    yield Output(
        val_mask_files,
        output_name="val_mask_files",
        metadata={"description": f"Count of masks = {len(val_mask_files)}"},
    )


@asset(name="test_image_files")
def data_test() -> Generator[Output, None, None]:
    """
    Test image files
    """
    test_image_files = list_objects(
        s3_resource, AMAZON_TIF_BUCKET.get_value(), str(TEST_IMAGES_PREFIX), ".tif"
    )

    # Raise ValueError if no test images have been found.
    if not test_image_files:
        raise ValueError(f"Zero .tif images found at {str(TEST_IMAGES_PREFIX)}")

    yield Output(
        test_image_files,
        metadata={"description": f"Count of images = {len(test_image_files)}"},
    )
