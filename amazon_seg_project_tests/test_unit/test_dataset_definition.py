"""
Unit tests for the __init__() and __len__() methods of SegmentationDataset class
"""

from unittest.mock import MagicMock
import pytest
from amazon_seg_project.assets import SegmentationDataset
from amazon_seg_project.ops.aug_utils import get_aug_pipeline


def test_seg_dataset_creation_success() -> None:
    """
    Verify correct initialization of SegmentationDataset instance attributes.
    """
    # Define mock variables.
    images_list = ["image1.tif", "image2.tif"]
    masks_list = ["mask1.tif", "mask2.tif"]
    mock_s3_bucket = "test-bucket"
    mock_scaling_func = MagicMock()
    horizontal_flip_probability = 0.25
    vertical_flip_probability = 0.15
    rotate90_probability = 0.05

    # Create the SegmentationDataset object.
    dataset = SegmentationDataset(
        images_list=images_list,
        masks_list=masks_list,
        s3_bucket=mock_s3_bucket,
        scaling_func=mock_scaling_func,
        transform=get_aug_pipeline(
            horizontal_flip_prob=horizontal_flip_probability,
            vertical_flip_prob=vertical_flip_probability,
            rotate90_prob=rotate90_probability,
        ),
    )

    # Assertions for instance attributes
    assert dataset.images == images_list
    assert dataset.masks == masks_list
    assert dataset.s3_bucket == mock_s3_bucket
    assert dataset.scaling_func == mock_scaling_func
    assert (
        dataset.transform is not None
        and dataset.transform.transforms[0].p == horizontal_flip_probability
    )
    assert (
        dataset.transform is not None
        and dataset.transform.transforms[1].p == vertical_flip_probability
    )
    assert (
        dataset.transform is not None
        and dataset.transform.transforms[2].p == rotate90_probability
    )


def test_seg_dataset_creation_length_error() -> None:
    """
    Raise ValueError if unequal numbers of images and masks are input.
    """
    with pytest.raises(
        ValueError, match="Unequal numbers of images and masks supplied."
    ):
        # Define mock variables.
        images_list = ["image1.tif", "image2.tif", "image3.tid"]
        masks_list = ["mask1.tif", "mask2.tif"]
        mock_s3_bucket = "test-bucket"

        # Create the SegmentationDataset object.
        SegmentationDataset(images_list, masks_list, mock_s3_bucket)


def test_seg_dataset_len() -> None:
    """
    Verify that SegmentationDataset.__len__() works as expected.
    """
    # Define mock variables.
    images_list = ["image1.tif", "image2.tif"]
    masks_list = ["mask1.tif", "mask2.tif"]
    mock_s3_bucket = "test-bucket"

    # Create the SegmentationDataset object.
    dataset = SegmentationDataset(images_list, masks_list, mock_s3_bucket)

    assert len(dataset) == len(images_list)
