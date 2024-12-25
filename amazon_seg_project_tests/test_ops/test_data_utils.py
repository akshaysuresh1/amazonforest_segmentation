"""
Unit tests for modules defined in amazon_seg_project/ops/data_utils.py
"""

from unittest.mock import patch, MagicMock
from amazon_seg_project.ops import find_corresponding_files


def test_find_corresponding_files_matching_files() -> None:
    """
    Test execution of find_corresponding_files() to matched data and reference files
    """
    data_files = {"path/to/data/image_2.tif", "path/to/data/image_1.tif"}
    ref_files = {"path/to/ref/image_1.txt", "path/to/ref/image_2.txt"}

    matched_data, matched_ref = find_corresponding_files(data_files, ref_files)

    assert matched_data == ["path/to/data/image_1.tif", "path/to/data/image_2.tif"]
    assert matched_ref == ["path/to/ref/image_1.txt", "path/to/ref/image_2.txt"]


@patch("logging.warning")
def test_find_corresponding_files_partial_match(mock_warning: MagicMock) -> None:
    """
    Test response of find_corresponding_files() to partially matched input files
    """
    data_files = {
        "path/to/data/image_1.tif",
        "path/to/data/image_2.tif",
        "path/to/data/image_3.tif",
    }
    ref_files = {"path/to/ref/image_1.txt", "path/to/ref/image_2.txt"}

    matched_data, matched_ref = find_corresponding_files(data_files, ref_files)

    # Check warning log.
    mock_warning.assert_called_once_with(
        "One or more data files have no reference labels."
    )

    assert matched_data == ["path/to/data/image_1.tif", "path/to/data/image_2.tif"]
    assert matched_ref == ["path/to/ref/image_1.txt", "path/to/ref/image_2.txt"]


@patch("logging.warning")
def test_find_corresponding_files_no_match(mock_warning: MagicMock) -> None:
    """
    Test response of find_corresponding_files() to mismatched input files
    """
    data_files = {"path/to/data/image_1.tif", "path/to/data/image_2.tif"}
    ref_files = {"path/to/ref/image_3.txt"}

    matched_data, matched_ref = find_corresponding_files(data_files, ref_files)

    # Check warning log.
    mock_warning.assert_called_once_with(
        "One or more data files have no reference labels."
    )

    assert not matched_data
    assert not matched_ref
