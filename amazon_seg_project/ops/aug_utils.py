"""
Data augmentation pipeline routine
"""

import albumentations as A
from dagster import op


@op
def get_aug_pipeline(
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.5,
    rotate90_prob: float = 0.5,
    augmentation_seed: int = 137,
) -> A.Compose:
    """
    Creates and returns a data augmentation pipeline

    Args:
        horizontal_flip_prob: Horizontal flip probability
        vertical_flip_prob: Vertical flip probability
        rotate90_prob: Probability for image rotation by a multiple of 90 deg.
        augmentation_seed: Random seed for Augmentation pipeline
    """
    if horizontal_flip_prob < 0 or horizontal_flip_prob > 1:
        raise ValueError("Horizontal flip probability must lie in the range [0, 1].")

    if vertical_flip_prob < 0 or vertical_flip_prob > 1:
        raise ValueError("Vertical flip probability must lie in the range [0, 1].")

    if rotate90_prob < 0 or rotate90_prob > 1:
        raise ValueError(
            "Probability of random image rotation must lie in the range [0, 1]."
        )

    # Create pipeline.
    pipeline = A.Compose(
        [
            A.HorizontalFlip(p=horizontal_flip_prob),
            A.VerticalFlip(p=vertical_flip_prob),
            A.RandomRotate90(p=rotate90_prob),
        ],
        seed=augmentation_seed,
    )

    return pipeline
