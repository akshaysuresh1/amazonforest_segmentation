"""
Assets pertaining to metrics computed on validation dataset
"""

from typing import Dict, Any
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from dagster import asset, AssetIn
from tqdm import tqdm
from . import SegmentationDataset
from ..config import PrecRecallCurveConfig, ModelEvaluationConfig
from ..data_paths import OUTPUT_PATH
from ..ops.metrics import smp_metrics, compute_f1_scores
from ..ops.plotting_utils import (
    plot_precision_recall_curve,
    visualize_and_save_model_predictions,
)
from ..ops.write_files import (
    create_directories,
    write_precision_recall_data,
    write_dict_to_csv,
)


@asset(
    name="precision_recall_curve",
    ins={"validation_dataset": AssetIn(), "trained_unet_model": AssetIn()},
)
def precision_recall_curve(
    config: PrecRecallCurveConfig,
    validation_dataset: SegmentationDataset,
    trained_unet_model: torch.nn.Module,
) -> Dict[str, Any]:
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
    n_positive_samples = 0  # No. of pixels labeled as "forest" in ground truth mask
    n_samples = 0  # Total no. of pixels across all images

    # Set up DataLoader object for looping over validation dataset.
    val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    # Loop over validation dataset.
    for image, ground_truth_mask in tqdm(val_loader):
        _, _, img_height, img_width = image.shape
        n_samples += img_height * img_width

        # Cast ground truth mask as a torch tensor with int dtype.
        ground_truth_mask = ground_truth_mask.int()
        n_positive_samples += int(ground_truth_mask.sum().item())

        # Obtain predicted mask by passing image through model.
        with torch.inference_mode():
            predicted_mask = trained_unet_model(
                image
            )  # output shape = (1, 1, n_y, n_x)

        # Loop over different binarization thresholds.
        for thresh_idx, threshold in enumerate(threshold_values):
            results = smp_metrics(
                predicted_mask, ground_truth_mask, threshold=threshold
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
    output = {
        "Binarization threshold": threshold_values,
        "Recall": recall_values,
        "Precision": precision_values,
        "F1 score": compute_f1_scores(precision_values, recall_values),
    }
    logging.info("Precision-recall curve generated for validation dataset.")
    return output


@asset(
    name="validation_metrics",
    ins={"validation_dataset": AssetIn(), "trained_unet_model": AssetIn()},
)
def validation_metrics(
    config: ModelEvaluationConfig,
    validation_dataset: SegmentationDataset,
    trained_unet_model: torch.nn.Module,
) -> Dict[str, Any]:
    """
    Validation metrics derived from model evaluation with a specific binarization threshold

    Included metrics (macro-imagewise):
    - Accuracy
    - Precision
    - Recall
    - F1 score
    - IoU

    Returns: Metrics for every image in validation dataset
    """
    len_val_dataset = len(validation_dataset)
    plot_basename = OUTPUT_PATH / "val_plots" / "val_index"
    # Create parent directories of plotting directory if non-existent.
    create_directories(plot_basename)

    # Metrics to be evaluated
    accuracy_values = np.zeros(len_val_dataset)
    precision_values = np.zeros(len_val_dataset)
    recall_values = np.zeros(len_val_dataset)
    f1_score_values = np.zeros(len_val_dataset)
    iou_score_values = np.zeros(len_val_dataset)

    # Create a DataLoader object for validation dataset.
    val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    # Loop over validation dataset.
    logging.info(f"Computing metrics over validation dataset of size {len_val_dataset}")
    for index, (image, ground_truth_mask) in tqdm(
        enumerate(val_loader), total=len(val_loader)
    ):
        # image shape = (1, n_channels, n_y, n_x)
        # mask shape = (1, 1, n_y, n_x)
        ground_truth_mask = ground_truth_mask.int()

        with torch.inference_mode():
            predicted_mask = trained_unet_model(
                image
            )  # output shape = (1, 1, n_y, n_x)

        # Sum metric evaluation over validation dataset images.
        results = smp_metrics(
            predicted_mask, ground_truth_mask, threshold=config.threshold
        )
        accuracy_values[index] = results.get("Accuracy")
        precision_values[index] = results.get("Precision")
        recall_values[index] = results.get("Recall")
        f1_score_values[index] = results.get("F1 score")
        iou_score_values[index] = results.get("IoU")

        # Produce a visualization of model predictions.
        img_plot = image.squeeze(0).permute((1, 2, 0)).numpy()
        ground_truth_mask_plot = ground_truth_mask.squeeze().numpy()
        predicted_mask_plot = predicted_mask.squeeze().numpy()
        visualize_and_save_model_predictions(
            img_plot,
            ground_truth_mask_plot,
            predicted_mask_plot,
            basename=str(plot_basename) + f"{index:03d}",
            accuracy=results.get("Accuracy"),
            precision=results.get("Precision"),
            recall=results.get("Recall"),
            iou_value=results.get("IoU"),
        )
        break

    # Build output dictionary.
    output = {
        "Accuracy": accuracy_values,
        "Precision": precision_values,
        "Recall": recall_values,
        "F1 score": f1_score_values,
        "IoU": iou_score_values,
    }
    write_dict_to_csv(
        output, str(OUTPUT_PATH / f"val_metrics_threshold_{config.threshold:.2f}.csv")
    )
    logging.info("Metrics computed for validation dataset.")
    return output
