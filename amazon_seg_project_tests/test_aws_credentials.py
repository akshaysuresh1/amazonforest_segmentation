"""
Unit-test modules defined in scripts/aws_credentials.py
"""

import os
import random
import unittest.mock as mock
import pytest
from amazon_seg_project.scripts.constants import AWS_REGIONS_LIST
from amazon_seg_project.scripts.aws_credentials import (
    get_aws_region_name,
    get_s3_access_key_id,
    get_s3_secret_access_key,
)


def test_get_aws_region_name_success() -> None:
    """
    Tests whether get_aws_region_name() responds correctly to a region specified in AWS_REGIONS_LIST
    """
    mock_region = random.choice(AWS_REGIONS_LIST)
    with mock.patch.dict(os.environ, {"AWS_REGION_NAME": mock_region}):
        assert mock_region == get_aws_region_name()


def test_get_aws_region_name_nonexistent() -> None:
    """
    Tests the response of get_aws_region_name() to a non-existent AWS_REGION_NAME environment variable
    """
    with pytest.raises(ValueError) as e:
        with mock.patch.dict(os.environ, clear=True):
            get_aws_region_name()
    assert str(e.value) == "Environment variable AWS_REGION_NAME is undefined."


def test_get_aws_region_name_empty() -> None:
    """
    Tests the response of get_aws_region_name() to an empty AWS_REGION_NAME environment variable
    """
    with pytest.raises(ValueError) as e:
        with mock.patch.dict(os.environ, {"AWS_REGION_NAME": ""}):
            get_aws_region_name()
    assert str(e.value) == "Environment variable AWS_REGION_NAME is empty."


def test_get_aws_region_name_invalidname() -> None:
    """
    Tests the response of get_aws_region_name() to an invalid AWS_REGION_NAME environment variable
    """
    mock_region = "asia"
    with pytest.raises(ValueError) as e:
        with mock.patch.dict(os.environ, {"AWS_REGION_NAME": mock_region}):
            get_aws_region_name()
    assert (
        str(e.value)
        == f"Region name '{mock_region}' is not valid or not supported by AWS."
    )


def test_get_s3_access_key_id_success() -> None:
    """
    Tests correct execution of get_s3_access_key_id()
    """
    mock_access_key_id = "access_key"
    with mock.patch.dict(os.environ, {"S3_ACCESS_KEY_ID": mock_access_key_id}):
        assert mock_access_key_id == get_s3_access_key_id()


def test_get_s3_access_key_id_nonexistent() -> None:
    """
    Tests the response of get_s3_access_key_id() to a non-existent S3_ACCESS_KEY_ID environment variable
    """
    with pytest.raises(ValueError) as e:
        with mock.patch.dict(os.environ, clear=True):
            get_s3_access_key_id()
    assert str(e.value) == "Environment variable S3_ACCESS_KEY_ID is undefined."


def test_get_s3_access_key_id_empty() -> None:
    """
    Tests the response of get_s3_access_key_id() to an empty S3_ACCESS_KEY_ID environment variable
    """
    with pytest.raises(ValueError) as e:
        with mock.patch.dict(os.environ, {"S3_ACCESS_KEY_ID": ""}):
            get_s3_access_key_id()
    assert str(e.value) == "Environment variable S3_ACCESS_KEY_ID is empty."


def test_get_s3_secret_access_key_success() -> None:
    """
    Tests correct execution of get_s3_secret_access_key()
    """
    mock_secret_access_key = "secret"
    with mock.patch.dict(os.environ, {"S3_SECRET_ACCESS_KEY": mock_secret_access_key}):
        assert mock_secret_access_key == get_s3_secret_access_key()


def test_get_s3_secret_access_key_nonexistent() -> None:
    """
    Tests the response of get_s3_secret_access_key() to a non-existent S3_SECRET_ACCESS_KEY environment variable
    """
    with pytest.raises(ValueError) as e:
        with mock.patch.dict(os.environ, clear=True):
            get_s3_secret_access_key()
    assert str(e.value) == "Environment variable S3_SECRET_ACCESS_KEY is undefined."


def test_get_s3_secret_access_key_empty() -> None:
    """
    Tests the response of get_s3_secret_access_key() to an empty S3_SECRET_ACCESS_KEY environment variable
    """
    with pytest.raises(ValueError) as e:
        with mock.patch.dict(os.environ, {"S3_SECRET_ACCESS_KEY": ""}):
            get_s3_secret_access_key()
    assert str(e.value) == "Environment variable S3_SECRET_ACCESS_KEY is empty."    
