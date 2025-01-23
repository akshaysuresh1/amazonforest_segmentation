"""
Utility functions to write intermediate output files locally
"""

import os
from typing import List
import numpy as np
import pandas as pd
from amazon_seg_project.data_paths import OUTPUT_PATH


def write_stats(means: np.ndarray, sigma: np.ndarray, bands: List[str]) -> None:
    """
    Write channel-wise mean and standard deviation info to a .csv file

    Args:
        means: Channel-wise mean of a multispectral dataset
        sigma: Channel-wise standard deviation of a multispectral dataset
        bands: Colors or wavelength regimes corresponding to different indices
    """
    if not len(means) == len(sigma) == len(bands):
        raise ValueError("Inputs do not have equal lengths.")

    # Create pandas DataFrame object.
    df = pd.DataFrame({"band": bands, "mean": means, "std": sigma})

    # Create output directory if non-existent.
    output_dir = OUTPUT_PATH / "stats"
    if not os.path.isdir(str(output_dir)):
        os.makedirs(str(output_dir))

    df.to_csv(str(output_dir / "training_dataset_stats.csv"), index=False)
