"""
Utility functions to write intermediate output files locally
"""

import os
import logging
from typing import List, Any, Union
import numpy as np
import pandas as pd
from dagster import op, In


@op(ins={"filepath": In(Any)})
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


@op(ins={"means": In(Any), "sigma": In(Any), "bands": In(List[str]), "outcsv": In(Any)})
def write_stats_to_csv(
    means: np.ndarray,
    sigma: np.ndarray,
    bands: List[str],
    outcsv: Union[str, os.PathLike],
) -> None:
    """
    Write channel-wise mean and standard deviation info to a .csv file

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

    # Write Pandas DataFrame to "outcsv".
    df.to_csv(str(outcsv), index=False)


@op(
    ins={
        "train_loss": In(List[float]),
        "val_loss": In(List[float]),
        "outcsv": In(Any),
    }
)
def write_loss_data_to_csv(
    train_loss: List[float], val_loss: List[float], outcsv: Union[str, os.PathLike]
) -> None:
    """
    Saves the training and validation loss curve data to a CSV file.

    Args:
        train_loss: List of training losses.
        val_loss: List of validation losses.
        outcsv: Name (including path) of CSV file to be saved.
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
