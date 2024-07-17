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
    get_aws_access_key_id,
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
    assert str(e.value) == "AWS region name is not set."


def test_get_aws_region_name_empty() -> None:
    """
    Tests the response of get_aws_region_name() to an empty AWS_REGION_NAME environment variable
    """
    with pytest.raises(ValueError) as e:
        with mock.patch.dict(os.environ, {"AWS_REGION_NAME": ""}):
            get_aws_region_name()
    assert str(e.value) == "AWS region name is empty."


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
        == f"AWS region name '{mock_region}' is not valid or not supported."
    )


def test_get_aws_access_key_id_success() -> None:
    """
    Tests correct execution of get_aws_access_key_id()
    """
    mock_access_key_id = "mock_access_key_id"
    with mock.patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": mock_access_key_id}):
        assert mock_access_key_id == get_aws_access_key_id()


def test_get_aws_access_key_id_nonexistent() -> None:
    """
    Tests the response of get_aws_access_key_id() to a non-existent AWS_ACCESS_KEY_ID environment variable
    """
    with pytest.raises(ValueError) as e:
        with mock.patch.dict(os.environ, clear=True):
            get_aws_access_key_id()
    assert str(e.value) == "AWS access key ID is not set."


def test_get_aws_access_key_id_empty() -> None:
    """
    Tests the response of get_aws_access_key_id() to an empty AWS_ACCESS_KEY_ID environment variable
    """
    with pytest.raises(ValueError) as e:
        with mock.patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": ""}):
            get_aws_access_key_id()
    assert str(e.value) == "AWS access key ID is empty."
