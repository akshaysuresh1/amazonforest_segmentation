"""
Values of global variables representing data paths
"""

from pathlib import Path

# Path to training data and labels
TRAINING_IMAGES_PREFIX = Path("train") / "images"
TRAINING_MASKS_PREFIX = Path("train") / "masks"

# Paths to validation data and labels
VALIDATION_IMAGES_PREFIX = Path("validation") / "images"
VALIDATION_MASKS_PREFIX = Path("validation") / "masks"

# Paths to test data
TEST_IMAGES_PREFIX = Path("test") / "images"
