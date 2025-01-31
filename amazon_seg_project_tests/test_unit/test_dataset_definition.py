"""
Unit tests for the __init__() and __len__() methods of SegmentationDataset class
"""

from unittest.mock import MagicMock
import pytest
from amazon_seg_project.assets import SegmentationDataset


def test_seg_dataset_creation_success() -> None:
    """
    Verify correct initialization of SegmentationDataset attributes.
    """
    # Define mock variables.
    images_list = ["image1.tif", "image2.tif"]
    masks_list = ["mask1.tif", "mask2.tif"]
    mock_s3_resource = MagicMock()
    mock_s3_bucket = "test-bucket"
    do_aug = True
    horizontal_flip_probability = 0.25
    vertical_flip_probability = 0.15
    rotate90_probability = 0.05

    # Create the SegmentationDataset object.
    dataset = SegmentationDataset(
        images_list,
        masks_list,
        mock_s3_resource,
        mock_s3_bucket,
        do_aug,
        horizontal_flip_probability,
        vertical_flip_probability,
        rotate90_probability,
    )

    # Assertions for class attributes
    assert dataset.images == images_list
    assert dataset.masks == masks_list
    assert dataset.s3_resource == mock_s3_resource
    assert dataset.s3_bucket == mock_s3_bucket
    assert dataset.do_aug == do_aug
    assert dataset.horizontal_flip_prob == horizontal_flip_probability
    assert dataset.vertical_flip_prob == vertical_flip_probability
    assert dataset.rotate90_prob == rotate90_probability


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
        mock_s3_resource = MagicMock()
        mock_s3_bucket = "test-bucket"

        # Create the SegmentationDataset object.
        SegmentationDataset(images_list, masks_list, mock_s3_resource, mock_s3_bucket)


def test_seg_dataset_len() -> None:
    """
    Verify that SegmentationDataset.__len__() works as expected.
    """
    # Define mock variables.
    images_list = ["image1.tif", "image2.tif"]
    masks_list = ["mask1.tif", "mask2.tif"]
    mock_s3_resource = MagicMock()
    mock_s3_bucket = "test-bucket"

    # Create the SegmentationDataset object.
    dataset = SegmentationDataset(
        images_list, masks_list, mock_s3_resource, mock_s3_bucket
    )

    assert len(dataset) == len(images_list)
