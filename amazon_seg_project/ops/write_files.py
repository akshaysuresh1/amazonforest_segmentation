"""
Utility functions to write intermediate output files locally
"""

import os
import logging
from typing import Dict, List, Union, Any
import numpy as np
import numpy.typing as npt
import pandas as pd
from dagster import op, In
from dagster import Any as dg_Any
from .metrics import compute_f1_scores


@op(ins={"filepath": In(dg_Any)})
def create_directories(filepath: Union[str, os.PathLike]) -> None:
    """
    Recursively create directories for the given filepath.
    """
    # Extract the directory path from the filepath.
    directory = os.path.dirname(filepath)

    # Create directories recursively.
    try:
        os.makedirs(directory, exist_ok=True)
        logging.info("Parent directories for '%s' created successfully.", str(filepath))
    except OSError as e:
        logging.error("Error creating directories for %s: %s", str(filepath), str(e))
        # Re-raise OSError exception.
        raise


@op(ins={"dictionary": In(dg_Any), "outcsv": In(dg_Any)})
def write_dict_to_csv(
    dictionary: Dict[str, Any], outcsv: Union[str, os.PathLike]
) -> None:
    """
    Write the contents of a dictionary to a .csv file.

    Args:
        dictionary: Input dict object
        outcsv: Name (including path) of output .csv file
    """
    # Append .csv extension if not found at end of file name.
    if not str(outcsv).endswith(".csv"):
        outcsv = str(outcsv) + ".csv"

    # Create parent directories if non-existent.
    create_directories(outcsv)

    # Create pandas DataFrame object from input dictionary.
    df = pd.DataFrame(dictionary)
    # Write pandas DataFrame to "outcsv".
    df.to_csv(str(outcsv), index=False)


@op(
    ins={
        "means": In(dg_Any),
        "sigma": In(dg_Any),
        "bands": In(List[str]),
        "outcsv": In(dg_Any),
    }
)
def write_stats_to_csv(
    means: np.ndarray,
    sigma: np.ndarray,
    bands: List[str],
    outcsv: Union[str, os.PathLike],
) -> None:
    """
    Write channel-wise mean and standard deviation info to a .csv file.

    Args:
        means: Channel-wise mean of a multispectral dataset
        sigma: Channel-wise standard deviation of a multispectral dataset
        bands: Colors or wavelength regimes corresponding to different indices
        outfile: Output file name, including path
    """
    if not len(means) == len(sigma) == len(bands):
        raise ValueError("Inputs do not have equal lengths.")

    # Create pandas DataFrame object.
    df = pd.DataFrame({"band": bands, "mean": means, "std": sigma})

    # Append .csv extension if not found at end of file name.
    if not str(outcsv).endswith(".csv"):
        outcsv = str(outcsv) + ".csv"

    # Create parent directories if non-existent.
    create_directories(outcsv)

    # Write pandas DataFrame to "outcsv".
    df.to_csv(str(outcsv), index=False)


@op(
    ins={
        "train_loss": In(List[float]),
        "val_loss": In(List[float]),
        "outcsv": In(dg_Any),
    }
)
def write_loss_data_to_csv(
    train_loss: List[float], val_loss: List[float], outcsv: Union[str, os.PathLike]
) -> None:
    """
    Saves the training and validation loss curve data to a .csv file.

    Args:
        train_loss: List of training losses
        val_loss: List of validation losses
        outcsv: Name (including path) of .csv file to be saved
    """
    if len(train_loss) != len(val_loss):
        raise ValueError("Input lists have different lengths.")

    # Append .csv extension if not found at end of file name.
    if not str(outcsv).endswith(".csv"):
        outcsv = str(outcsv) + ".csv"

    loss_df = pd.DataFrame(
        {"train_loss": np.array(train_loss), "val_loss": np.array(val_loss)},
        index=np.arange(1, len(train_loss) + 1),
    )

    # Create parent directories if non-existent.
    create_directories(outcsv)

    # Write Pandas DataFrame to "outcsv".
    loss_df.to_csv(str(outcsv), index=False)


@op(
    ins={
        "precision": In(dg_Any),
        "recall": In(dg_Any),
        "threshold": In(dg_Any),
        "outcsv": In(dg_Any),
    }
)
def write_precision_recall_data(
    precision: npt.NDArray[np.float_],
    recall: npt.NDArray[np.float_],
    threshold: npt.NDArray[np.float_],
    outcsv: Union[str, os.PathLike],
) -> None:
    """
    Save data points from a precision-recall curve to a .csv file.

    Args:
        precision: Precision values at different binarization thresholds
        recall: Recall values at different binarization thresholds
        threshold: Binarization thresholds
        outcsv: Name (including path) of .csv file to be saved
    """
    if not len(precision) == len(recall) == len(threshold):
        raise ValueError("Input arrays have unequal lengths.")

    # Append .csv extension if not found at end of file name.
    if not str(outcsv).endswith(".csv"):
        outcsv = str(outcsv) + ".csv"

    f1_scores = compute_f1_scores(precision, recall)
    prec_recall_curve_df = pd.DataFrame(
        {
            "Binarization threshold": threshold,
            "Recall": recall,
            "Precision": precision,
            "F1 score": f1_scores,
        }
    )

    # Create parent directories if non-existent.
    create_directories(outcsv)

    # Write Pandas DataFrame to "outcsv".
    prec_recall_curve_df.to_csv(str(outcsv), index=False)
