"""
Unit tests for functions defined in amazon_seg_project/ops/write_files.py
"""

from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
from amazon_seg_project.ops.write_files import (
    create_directories,
    write_dict_to_csv,
    write_stats_to_csv,
    write_loss_data_to_csv,
    write_precision_recall_data,
)


@patch("os.makedirs")
@patch("logging.info")
def test_create_directories_success_str(
    mock_logging: MagicMock, mock_makedirs: MagicMock
) -> None:
    """
    Test successful execution of create_directories() with input string path.
    """
    filepath = "folder1/folder2/file.txt"
    dirpath = "folder1/folder2"

    # Call the test function.
    create_directories(filepath)

    # Verify that os.makedirs is called with the correct argument.
    mock_makedirs.assert_called_once_with(dirpath, exist_ok=True)
    # Check logging.
    mock_logging.assert_called_once_with(
        "Parent directories for '%s' created successfully.", str(filepath)
    )


@patch("os.makedirs")
@patch("logging.info")
def test_create_directories_success_path(
    mock_logging: MagicMock, mock_makedirs: MagicMock
) -> None:
    """
    Test successful execution of create_directories() with input Path object.
    """
    filepath = Path("folder1/folder2/file.txt")
    dirpath = "folder1/folder2"

    # Call the test function.
    create_directories(filepath)

    # Verify that os.makedirs is called with the correct argument.
    mock_makedirs.assert_called_once_with(dirpath, exist_ok=True)
    # Check logging.
    mock_logging.assert_called_once_with(
        "Parent directories for '%s' created successfully.", str(filepath)
    )


@patch("os.makedirs")
@patch("logging.error")
def test_create_directories_os_error(
    mock_logging: MagicMock, mock_makedirs: MagicMock
) -> None:
    """
    Test correct raise of OSError by create_directories().
    """
    with pytest.raises(OSError):
        filepath = Path("folder1/folder2/file.txt")
        e = "Permission denied."
        mock_makedirs.side_effect = OSError(e)

        # Call the test function.
        create_directories(filepath)

    # Verify logging.
    mock_logging.assert_called_once_with(
        "Error creating directories for %s: %s", str(filepath), e
    )


@patch("amazon_seg_project.ops.write_files.create_directories")
@patch("pandas.DataFrame")
def test_write_dict_to_csv_valid_filename(
    mock_df: MagicMock, mock_create_dirs: MagicMock
) -> None:
    """
    Test successful execution of write_stats_to_csv() with valid CSV extension.
    """
    # Create mock inputs.
    input_dict = {"Color": ["Red", "Green", "Blue"], "Version": 1.1}
    outcsv = Path("folder1/folder2/outfile.csv")

    # Mock DataFrame object
    mock_df_instance = MagicMock()
    mock_df.return_value = mock_df_instance

    # Call test function.
    write_dict_to_csv(input_dict, outcsv)

    # Assertions
    mock_create_dirs.assert_called_once_with(outcsv)
    mock_df.assert_called_once_with(input_dict)
    mock_df_instance.to_csv.assert_called_once_with(str(outcsv), index=False)


@patch("amazon_seg_project.ops.write_files.create_directories")
@patch("pandas.DataFrame")
def test_write_dict_to_csv_missing_file_extension(
    mock_df: MagicMock, mock_create_dirs: MagicMock
) -> None:
    """
    Test successful execution of write_stats_to_csv() with missing CSV extension.
    """
    # Create mock inputs.
    input_dict = {"Color": ["Red", "Green", "Blue"], "Version": 1.1}
    outcsv = Path("folder1/folder2/outfile")
    final_csv = str(outcsv) + ".csv"

    # Mock DataFrame object
    mock_df_instance = MagicMock()
    mock_df.return_value = mock_df_instance

    # Call test function.
    write_dict_to_csv(input_dict, outcsv)

    # Assertions
    mock_create_dirs.assert_called_once_with(final_csv)
    mock_df.assert_called_once_with(input_dict)
    mock_df_instance.to_csv.assert_called_once_with(final_csv, index=False)


@patch("amazon_seg_project.ops.write_files.create_directories")
@patch("pandas.DataFrame")
def test_write_stats_success_valid_filename(
    mock_df: MagicMock, mock_create_dirs: MagicMock
) -> None:
    """
    Test successful execution of write_stats_to_csv() with valid filename.
    """
    # Create mock inputs.
    means = np.array([1.0, 2.0, 3.0])
    sigma = np.array([0.3, 0.2, 0.1])
    bands = ["Red", "Green", "Blue"]
    outcsv = Path("folder1/folder2/stats.csv")

    # Mock DataFrame creation
    mock_df_instance = MagicMock()
    mock_df.return_value = mock_df_instance

    # Call the test function.
    write_stats_to_csv(means, sigma, bands, outcsv)

    # Assert that create_directories() was called to create the output directory.
    mock_create_dirs.assert_called_once_with(outcsv)

    # Verify that the correct dictionary was passed to pd.DataFrame().
    expected_contents = {"band": bands, "mean": means, "std": sigma}
    mock_df.assert_called_once_with(expected_contents)

    # Assert that to_csv() method of pd.DataFrame was called with the correct arguments.
    mock_df_instance.to_csv.assert_called_once_with(str(outcsv), index=False)


@patch("amazon_seg_project.ops.write_files.create_directories")
@patch("pandas.DataFrame")
def test_write_stats_success_incomplete_filename(
    mock_df: MagicMock, mock_create_dirs: MagicMock
) -> None:
    """
    Test successful execution of write_stats_to_csv() with incomplete filename.
    """
    # Create mock inputs.
    means = np.array([1.0, 2.0, 3.0])
    sigma = np.array([0.3, 0.2, 0.1])
    bands = ["Red", "Green", "Blue"]
    outcsv = Path("folder1/folder2/stats")
    final_csv = str(outcsv) + ".csv"

    # Mock DataFrame creation
    mock_df_instance = MagicMock()
    mock_df.return_value = mock_df_instance

    # Call the test function.
    write_stats_to_csv(means, sigma, bands, outcsv)

    # Assert that create_directories() was called to create the output directory.
    mock_create_dirs.assert_called_once_with(final_csv)

    # Verify that the correct dictionary was passed to pd.DataFrame().
    expected_contents = {"band": bands, "mean": means, "std": sigma}
    mock_df.assert_called_once_with(expected_contents)

    # Assert that to_csv() method of pd.DataFrame was called with the correct arguments.
    mock_df_instance.to_csv.assert_called_once_with(final_csv, index=False)


def test_write_stats_length_mismatch() -> None:
    """
    Check for correct response of write_stats_to_csv() to inputs of unequal lengths.
    """
    with pytest.raises(ValueError, match="Inputs do not have equal lengths."):
        means = np.zeros(4)
        sigma = np.ones(3)
        bands = ["Red", "Green", "Blue", "NIR", "UV"]
        outcsv = Path("folder1/folder2/stats.csv")

        write_stats_to_csv(means, sigma, bands, outcsv)


@patch("amazon_seg_project.ops.write_files.create_directories")
@patch("pandas.DataFrame")
def test_write_loss_data_valid_filename(
    mock_df: MagicMock, mock_create_dirs: MagicMock
) -> None:
    """
    Test successful execition of write_loss_data_to_csv().

    Assumption: Valid input filename supplied.
    """
    # Create mock inputs.
    mock_train_loss = [0.1, 0.2, 0.3]
    mock_val_loss = [0.15, 0.25, 0.35]
    outcsv = Path("root_folder/subfolder/loss_curve_data.csv")

    # Mock DataFrame creation
    mock_df_instance = MagicMock()
    mock_df.return_value = mock_df_instance

    # Call the test function.
    write_loss_data_to_csv(mock_train_loss, mock_val_loss, outcsv)

    # Assert that create_directories() was called to create the output directory.
    mock_create_dirs.assert_called_once_with(outcsv)

    # Verify that the correct dictionary was passed to pd.DataFrame().
    expected_contents = {
        "train_loss": np.array(mock_train_loss),
        "val_loss": np.array(mock_val_loss),
    }
    index_array = np.arange(1, len(mock_train_loss) + 1)

    args, kwargs = mock_df.call_args[0], mock_df.call_args[1]
    actual_contents = args[0]
    assert actual_contents.keys() == expected_contents.keys(), (
        "Dictionary keys do not match"
    )

    for key, expected_value in expected_contents.items():
        assert np.array_equal(expected_value, expected_contents[key]), (
            f"Array values for key '{key}' do not match."
        )

    # Compare the 'index' keyword argument.
    assert np.array_equal(kwargs.get("index"), index_array), (
        "Index array does not match."
    )

    # Assert that to_csv() method of pd.DataFrame was called with the correct arguments.
    mock_df_instance.to_csv.assert_called_once_with(str(outcsv), index=False)


@patch("amazon_seg_project.ops.write_files.create_directories")
@patch("pandas.DataFrame")
def test_write_loss_data_incomplete_filename(
    mock_df: MagicMock, mock_create_dirs: MagicMock
) -> None:
    """
    Test successful execition of write_loss_data_to_csv().

    Assumption: Input filename lacks .csv file extension.
    """
    # Create mock inputs.
    mock_train_loss = [0.1, 0.2, 0.3]
    mock_val_loss = [0.15, 0.25, 0.35]
    outcsv = Path("root_folder/subfolder/loss_curve_data")
    final_csv = str(outcsv) + ".csv"

    # Mock DataFrame creation
    mock_df_instance = MagicMock()
    mock_df.return_value = mock_df_instance

    # Call the test function.
    write_loss_data_to_csv(mock_train_loss, mock_val_loss, outcsv)

    # Assert that create_directories() was called to create the output directory.
    mock_create_dirs.assert_called_once_with(final_csv)

    # Verify that the correct dictionary was passed to pd.DataFrame().
    expected_contents = {
        "train_loss": np.array(mock_train_loss),
        "val_loss": np.array(mock_val_loss),
    }
    index_array = np.arange(1, len(mock_train_loss) + 1)

    args, kwargs = mock_df.call_args[0], mock_df.call_args[1]
    actual_contents = args[0]
    assert actual_contents.keys() == expected_contents.keys(), (
        "Dictionary keys do not match"
    )

    for key, expected_value in expected_contents.items():
        assert np.array_equal(expected_value, expected_contents[key]), (
            f"Array values for key '{key}' do not match."
        )

    # Compare the 'index' keyword argument.
    assert np.array_equal(kwargs["index"], index_array), "Index array does not match."

    # Assert that to_csv() method of pd.DataFrame was called with the correct arguments.
    mock_df_instance.to_csv.assert_called_once_with(final_csv, index=False)


def test_write_loss_data_length_mismatch() -> None:
    """
    Check for correct response of write_stats() to inputs of unequal lengths.
    """
    with pytest.raises(ValueError, match="Input lists have different lengths."):
        mock_train_loss = [0.1, 0.2, 0.3]
        mock_val_loss = [0.15, 0.25]
        outcsv = Path("root_folder/subfolder/loss_curve_data.csv")

        write_loss_data_to_csv(mock_train_loss, mock_val_loss, outcsv)


def test_write_precision_recall_data_unequal_input_shapes() -> None:
    """
    Test raise of ValueError() when input arrays have unequal lengths.
    """
    with pytest.raises(ValueError, match="Input arrays have unequal lengths."):
        # Simulate test inputs.
        precision_values = np.array([0.2, 0.6, 0.9])
        recall_values = np.array([0.8, 0.6, 0.4, 0.3])
        threshold_values = np.array([0.3, 0.5])
        iou_values = np.array([0.4, 0.6, 0.5])
        outcsv = Path("root_folder/subfolder/precision_recall_curve.csv")

        # Call the test function.
        write_precision_recall_data(
            precision_values, recall_values, threshold_values, iou_values, outcsv
        )


@patch("amazon_seg_project.ops.write_files.compute_f1_scores")
@patch("amazon_seg_project.ops.write_files.create_directories")
@patch("pandas.DataFrame")
def test_write_precision_recall_data_incomplete_filename(
    mock_df: MagicMock,
    mock_create_dirs: MagicMock,
    mock_compute_f1_scores: MagicMock,
) -> None:
    """
    Test write_precision_recall_data() with outcsv lacking ".csv" extension.
    """
    # Simulate inputs.
    precision_values = np.array([0.2, 0.6, 0.9])
    recall_values = np.array([0.8, 0.6, 0.4])
    threshold_values = np.array([0.3, 0.5, 0.7])
    iou_values = np.array([0.4, 0.6, 0.5])
    outcsv = Path("root_folder/subfolder/precision_recall_curve")
    final_csv = str(outcsv) + ".csv"

    # Set up mock dependencies.
    mock_df_object = MagicMock(name="mock-dataframe")
    mock_df.return_value = mock_df_object

    # Call the test function.
    write_precision_recall_data(
        precision_values, recall_values, threshold_values, iou_values, outcsv
    )

    # Assert for call to compute_f1_scores()
    f1_args, _ = mock_compute_f1_scores.call_args
    assert len(f1_args) == 2
    np.testing.assert_array_equal(f1_args[0], precision_values)
    np.testing.assert_array_equal(f1_args[1], recall_values)

    # Assertion for DataFrame object creation
    args, _ = mock_df.call_args
    assert len(args) == 1
    assert isinstance(args[0], dict)
    np.testing.assert_array_equal(args[0]["Binarization threshold"], threshold_values)
    np.testing.assert_array_equal(args[0]["Recall"], recall_values)
    np.testing.assert_array_equal(args[0]["Precision"], precision_values)
    np.testing.assert_array_equal(
        args[0]["F1 score"], mock_compute_f1_scores.return_value
    )
    np.testing.assert_array_equal(args[0]["IoU"], iou_values)

    # Assertion for parent directory creation.
    mock_create_dirs.assert_called_once_with(final_csv)
    # Assert that to_csv() method of pd.DataFrame was called with the correct arguments.
    mock_df_object.to_csv.assert_called_once_with(final_csv, index=False)


@patch("amazon_seg_project.ops.write_files.compute_f1_scores")
@patch("amazon_seg_project.ops.write_files.create_directories")
@patch("pandas.DataFrame")
def test_write_precision_recall_data_success(
    mock_df: MagicMock,
    mock_create_dirs: MagicMock,
    mock_compute_f1_scores: MagicMock,
) -> None:
    """
    Test write_precision_recall_data() for valid inputs.
    """
    # Simulate inputs.
    precision_values = np.array([0.2, 0.6, 0.9])
    recall_values = np.array([0.8, 0.6, 0.4])
    threshold_values = np.array([0.3, 0.5, 0.7])
    iou_values = np.array([0.4, 0.6, 0.5])
    outcsv = Path("root_folder/subfolder/precision_recall_curve.csv")

    # Set up mock dependencies.
    mock_df_object = MagicMock(name="mock-dataframe")
    mock_df.return_value = mock_df_object

    # Call the test function.
    write_precision_recall_data(
        precision_values, recall_values, threshold_values, iou_values, outcsv
    )

    # Assert for call to compute_f1_scores()
    f1_args, _ = mock_compute_f1_scores.call_args
    assert len(f1_args) == 2
    np.testing.assert_array_equal(f1_args[0], precision_values)
    np.testing.assert_array_equal(f1_args[1], recall_values)

    # Assertion for DataFrame object creation
    args, _ = mock_df.call_args
    assert len(args) == 1
    assert isinstance(args[0], dict)
    np.testing.assert_array_equal(args[0]["Binarization threshold"], threshold_values)
    np.testing.assert_array_equal(args[0]["Recall"], recall_values)
    np.testing.assert_array_equal(args[0]["Precision"], precision_values)
    np.testing.assert_array_equal(
        args[0]["F1 score"], mock_compute_f1_scores.return_value
    )
    np.testing.assert_array_equal(args[0]["IoU"], iou_values)

    # Assertion for parent directory creation.
    mock_create_dirs.assert_called_once_with(outcsv)
    # Assert that to_csv() method of pd.DataFrame was called with the correct arguments.
    mock_df_object.to_csv.assert_called_once_with(str(outcsv), index=False)
