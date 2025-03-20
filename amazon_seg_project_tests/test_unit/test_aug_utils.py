"""
Unit tests for augmentation utilities
"""

import re
import pytest
import albumentations as A
from amazon_seg_project.ops.aug_utils import get_aug_pipeline


def test_get_aug_pipeline_negative_horizontal_flip_prob() -> None:
    """
    Validate response of get_aug_pipeline() to a negative horizontal flip probability.
    """
    with pytest.raises(
        ValueError,
        match=re.escape("Horizontal flip probability must lie in the range [0, 1]."),
    ):
        get_aug_pipeline(horizontal_flip_prob=-0.2)


def test_get_aug_pipeline_horizontal_flip_prob_greater_than_1() -> None:
    """
    Validate response of get_aug_pipeline() to a horizontal flip probability exceeding 1.
    """
    with pytest.raises(
        ValueError,
        match=re.escape("Horizontal flip probability must lie in the range [0, 1]."),
    ):
        get_aug_pipeline(horizontal_flip_prob=1.2)


def test_get_aug_pipeline_negative_vertical_flip_prob() -> None:
    """
    Validate response of get_aug_pipeline() to a negative vertical flip probability.
    """
    with pytest.raises(
        ValueError,
        match=re.escape("Vertical flip probability must lie in the range [0, 1]."),
    ):
        get_aug_pipeline(vertical_flip_prob=-0.3)


def test_get_aug_pipeline_vertical_flip_prob_greater_than_1() -> None:
    """
    Validate response of get_aug_pipeline() to a vertical flip probability exceeding 1.
    """
    with pytest.raises(
        ValueError,
        match=re.escape("Vertical flip probability must lie in the range [0, 1]."),
    ):
        get_aug_pipeline(vertical_flip_prob=1.3)


def test_get_aug_pipeline_negative_rotate90_prob() -> None:
    """
    Validate response of get_aug_pipeline() to a negative image rotation probability.
    """
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Probability of random image rotation must lie in the range [0, 1]."
        ),
    ):
        get_aug_pipeline(rotate90_prob=-0.25)


def test_get_aug_pipeline_rotate90_prob_greater_than_1() -> None:
    """
    Validate response of get_aug_pipeline() to an image rotation probability exceeding 1.
    """
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Probability of random image rotation must lie in the range [0, 1]."
        ),
    ):
        get_aug_pipeline(rotate90_prob=1.25)


def test_get_aug_pipeline_success() -> None:
    """
    Test for successful execution of get_aug_pipeline().
    """
    horizontal_flip_prob = 0.2
    vertical_flip_prob = 0.3
    rotate90_prob = 0.25
    seed = 29

    # Call the test function.
    pipeline = get_aug_pipeline(
        horizontal_flip_prob=horizontal_flip_prob,
        vertical_flip_prob=vertical_flip_prob,
        rotate90_prob=rotate90_prob,
        augmentation_seed=seed,
    )

    # Assertions
    assert isinstance(pipeline, A.Compose)
    assert len(pipeline) == 3
    # First transform: Horizontal flip
    assert pipeline.transforms[0].p == horizontal_flip_prob
    # Second transform: Vertical flip
    assert pipeline.transforms[1].p == vertical_flip_prob
    # Thid transform: Image rotation by a multiple of 90 degrees
    assert pipeline.transforms[2].p == rotate90_prob
    # Verify seed setting.
    assert pipeline.seed == seed
