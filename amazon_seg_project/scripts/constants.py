"""
Values of global variables
"""

import os

# Path to training data and labels
TRAINING_IMAGES_PREFIX = os.path.join("train", "images")
TRAINING_MASKS_PREFIX = os.path.join("train", "masks")

# Paths to validation data and labels
VALIDATION_IMAGES_PREFIX = os.path.join("validation", "images")
VALIDATION_MASKS_PREFIX = os.path.join("validation", "masks")

# Paths to test data
TEST_IMAGES_PREFIX = os.path.join("test", "images")
TEST_MASKS_PREFIX = os.path.join("test", "masks")

# Allowed AWS region names
AWS_REGIONS_LIST = [
    "us-east-1",  # US East (N. Virginia)
    "us-east-2",  # US East (Ohio)
    "us-west-1",  # US West (N. California)
    "us-west-2",  # US West (Oregon)
    "af-south-1",  # Africa (Cape Town)
    "ap-east-1",  # Asia Pacific (Hong Kong)
    "ap-south-1",  # Asia Pacific (Mumbai)
    "ap-northeast-3",  # Asia Pacific (Osaka-Local)
    "ap-northeast-2",  # Asia Pacific (Seoul)
    "ap-southeast-1",  # Asia Pacific (Singapore)
    "ap-southeast-2",  # Asia Pacific (Sydney)
    "ap-northeast-1",  # Asia Pacific (Tokyo)
    "ca-central-1",  # Canada (Central)
    "cn-north-1",  # China (Beijing) - Operated by Sinnet
    "cn-northwest-1",  # China (Ningxia) - Operated by NWCD
    "eu-central-1",  # Europe (Frankfurt)
    "eu-west-1",  # Europe (Ireland)
    "eu-west-2",  # Europe (London)
    "eu-south-1",  # Europe (Milan)
    "eu-west-3",  # Europe (Paris)
    "eu-north-1",  # Europe (Stockholm)
    "me-south-1",  # Middle East (Bahrain)
    "sa-east-1",  # South America (São Paulo)
]
