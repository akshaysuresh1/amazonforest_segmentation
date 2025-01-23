"""
Dataset statistics computation
"""

from typing import Generator, List
import numpy as np
from dagster import multi_asset, AssetIn, AssetOut, Output
from ..ops import load_tif_from_s3, write_stats
from ..resources import s3_resource, AMAZON_TIF_BUCKET


@multi_asset(
    ins={"train_image_files": AssetIn()},
    outs={
        "mean_training_data": AssetOut(
            dagster_type=np.ndarray, description="Channel-wise mean of training images"
        ),
        "sigma_training_data": AssetOut(
            dagster_type=np.ndarray,
            description="Channel-wise standard deviation of training images",
        ),
    },
)
def channel_stats_training_data(
    train_image_files: List[str],
) -> Generator[Output, None, None]:
    """
    Channel-wise statistics computed from training data

    Assumption:
    All .tif files have exactly four color channels (red, green, blue, NIR).

    Returns:
        mean_training_data: Channel-wise mean of training dataset
        sigma_training_data: Channel-wise standard deviation of training dataset
    """
    # Store sum and squared sums of mean image pixel values.
    channel_sums = None
    channel_squared_sums = None
    count = 0  # File count

    # Loop over images in dataset.
    for datafile in train_image_files:
        dataset = load_tif_from_s3(s3_resource, AMAZON_TIF_BUCKET.get_value(), datafile)
        # Typecast dataset to float64 for high precision computation.
        dataset = dataset.astype(np.float64)

        # Dataset dimensions = (band, y, x)
        if dataset.ndim == 3 and len(dataset["band"]) == 4:
            # Compute pixel mean and squared mean for each band.
            data_means = dataset.mean(dim=("y", "x")).values
            data_squared_means = (dataset**2).mean(dim=("y", "x")).values

            if not count:
                channel_sums = data_means
                channel_squared_sums = data_squared_means
            else:
                channel_sums += data_means
                channel_squared_sums += data_squared_means

            count += 1

    if count == 0:
        raise ValueError("No valid GeoTIFF images found in training data.")

    mean_training_data = channel_sums / count
    sigma_training_data = np.sqrt(
        (channel_squared_sums / count) - mean_training_data**2
    )

    # Write mean and standard deviation data to disk.
    bands = ["Red", "Green", "Blue", "NIR"]
    write_stats(mean_training_data, sigma_training_data, bands)

    yield Output(
        mean_training_data,
        output_name="mean_training_data",
        metadata={
            "description": f"""
                  {bands[0]}: {mean_training_data[0]:.2f},
                  {bands[1]}: {mean_training_data[1]:.2f},
                  {bands[2]}: {mean_training_data[2]:.2f},
                  {bands[3]}: {mean_training_data[3]:.2f}
            """
        },
    )
    yield Output(
        sigma_training_data,
        output_name="sigma_training_data",
        metadata={
            "description": f"""
                  {bands[0]}: {sigma_training_data[0]:.2f},
                  {bands[1]}: {sigma_training_data[1]:.2f},
                  {bands[2]}: {sigma_training_data[2]:.2f},
                  {bands[3]}: {sigma_training_data[3]:.2f}
            """
        },
    )
