"""
Definition of asset "precision_recall_curve"
"""

import logging
import numpy as np
import pandas as pd
import torch
from dagster import asset, AssetIn
from tqdm import tqdm
from . import SegmentationDataset
from ..config import PrecRecallCurveConfig
from ..data_paths import OUTPUT_PATH
from ..ops.metrics import smp_metrics, compute_f1_scores
from ..ops.plotting_utils import plot_precision_recall_curve
from ..ops.write_files import write_precision_recall_data


@asset(
    name="precision_recall_curve",
    ins={"validation_dataset": AssetIn(), "trained_unet_model": AssetIn()},
)
def precision_recall_curve(
    config: PrecRecallCurveConfig,
    validation_dataset: SegmentationDataset,
    trained_unet_model: torch.nn.Module,
) -> pd.DataFrame:
    """
    Precision-recall curve at different binarization thresholds for validation dataset
    """
    # Arrays to store precision and recall estimates at different thresholds
    threshold_values = np.array(config.thresholds_list)
    precision_values = np.zeros(len(threshold_values))
    recall_values = np.zeros(len(threshold_values))

    # Length of validation dataset
    len_val_dset = len(validation_dataset)
    logging.info(
        f"Computing precision-recall curve over validation dataset of length {len_val_dset}"
    )

    trained_unet_model.eval()
    # Loop over validation dataset.
    n_positive_samples = 0  # No. of pixels labeled as "forest" in ground truth mask
    n_samples = 0  # Total no. of pixels across all images
    for index in tqdm(range(len_val_dset)):
        image, ground_truth_mask = validation_dataset[index]
        _, img_height, img_width = image.shape
        n_samples += img_height * img_width

        # Cast ground truth mask as a torch tensor with int dtype.
        ground_truth_mask = ground_truth_mask.int()
        n_positive_samples += int(ground_truth_mask.sum().item())

        # Obtain predicted mask by passing image through model.
        with torch.inference_mode():
            predicted_mask = trained_unet_model(
                image.unsqueeze(0)
            )  # output shape = (1, 1, n_y, n_x)

        # Loop over different binarization thresholds.
        for thresh_idx, threshold in enumerate(threshold_values):
            results = smp_metrics(
                predicted_mask, ground_truth_mask.unsqueeze(0), threshold=threshold
            )
            precision_values[thresh_idx] += results.get("Precision")
            recall_values[thresh_idx] += results.get("Recall")

    # Obtain average precision and recall across validation dataset.
    precision_values /= len_val_dset
    recall_values /= len_val_dset

    # Write precision and recall arrays to disk.
    logging.info("Writing precision-recall curve data points to disk")
    write_precision_recall_data(
        precision_values,
        recall_values,
        threshold_values,
        OUTPUT_PATH / "val_precision_recall_curve.csv",
    )

    # Plot precision-recall curve.
    logging.info("Plotting precision-recall curve")
    plot_precision_recall_curve(
        precision_values,
        recall_values,
        threshold_values,
        n_positive_samples=n_positive_samples,
        n_samples=n_samples,
        basename=str(OUTPUT_PATH / "val"),
    )

    # Prepare output table for asset.
    df = pd.DataFrame(
        {
            "Binarization threshold": threshold_values,
            "Recall": recall_values,
            "Precision": precision_values,
            "F1 score": compute_f1_scores(precision_values, recall_values),
        }
    )
    logging.info("Precision-recall curve generated for validation dataset.")
    return df
