"""
Values of global variables representing data paths
"""

from pathlib import Path

# Paths to training images and masks on S3 bucket
TRAINING_IMAGES_PREFIX = Path("train") / "images"
TRAINING_MASKS_PREFIX = Path("train") / "masks"

# Paths to validation images and masks on S3 bucket
VALIDATION_IMAGES_PREFIX = Path("validation") / "images"
VALIDATION_MASKS_PREFIX = Path("validation") / "masks"

# Paths to test images on S3 buckket
TEST_IMAGES_PREFIX = Path("test") / "images"

# Output products (to be saved locally)
OUTPUT_PATH = Path("outputs")
