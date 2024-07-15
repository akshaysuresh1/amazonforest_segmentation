"""
Unit-test modules defined in scripts/aws_credentials.py
"""

import unittest.mock as mock
from amazon_seg_project.scripts.aws_credentials import get_aws_region_name


def test_get_aws_region_name_invalidname() -> None:
    """
    Tests the response of get_aws_region_name() to an invalid AWS region name.
    """
    mock_region = "asia"
    with mock.patch("os.getenv", return_value=mock_region):
        try:
            get_aws_region_name()
        except ValueError as e:
            assert (
                str(e)
                == f"AWS region name '{mock_region}' is not valid or not supported."
            )
