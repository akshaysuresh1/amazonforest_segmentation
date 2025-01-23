"""
Unit tests for functions defined in amazon_seg_project/ops/write_files.py
"""

import os
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
from amazon_seg_project.ops import write_stats
from amazon_seg_project.data_paths import OUTPUT_PATH


@patch("os.makedirs")
@patch("pandas.DataFrame")
def test_write_stats_success(mock_df: MagicMock, mock_makedirs: MagicMock) -> None:
    """
    Test for successful execution of write_stats()
    """
    # Create mock inputs.
    means = np.array([1.0, 2.0, 3.0])
    sigma = np.array([0.3, 0.2, 0.1])
    bands = ["Red", "Green", "Blue"]

    # Mock DataFrame creation
    mock_df_instance = MagicMock()
    mock_df.return_value = mock_df_instance

    # Call the test function.
    write_stats(means, sigma, bands)

    # Assert that makedirs was called to create the output directory
    if not os.path.isdir(str(OUTPUT_PATH / "stats")):
        mock_makedirs.assert_called_once()

    # Verify that the correct dictionary was passed to pd.DataFrame().
    expected_contents = {"band": bands, "mean": means, "std": sigma}
    mock_df.assert_called_once_with(expected_contents)

    # Assert that to_csv() method of pd.DataFrame was called with the correct arguments.
    expected_file_path = str(OUTPUT_PATH / "stats" / "training_dataset_stats.csv")
    mock_df_instance.to_csv.assert_called_once_with(expected_file_path, index=False)


def test_write_stats_length_mismatch() -> None:
    """
    Check for correct response of write_stats() to inputs of unequal lengths
    """
    with pytest.raises(ValueError, match="Inputs do not have equal lengths."):
        means = np.zeros(4)
        sigma = np.ones(3)
        bands = ["Red", "Green", "Blue", "NIR", "UV"]

        write_stats(means, sigma, bands)
