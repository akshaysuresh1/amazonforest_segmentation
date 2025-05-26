"""
Assets for performance metrics computed on test data
"""

from typing import Dict, Any
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dagster import asset, AssetIn
from . import SegmentationDataset
from ..config import ModelEvaluationConfig
from ..data_paths import OUTPUT_PATH
from ..ops.metrics import smp_metrics
from ..ops.plotting_utils import visualize_and_save_model_predictions
from ..ops.write_files import create_directories, write_dict_to_csv


@asset(
    name="test_dataset_metrics",
    ins={"test_dataset": AssetIn(), "trained_unet_model": AssetIn()},
)
def afs_test_dataset_metrics(
    config: ModelEvaluationConfig,
    test_dataset: SegmentationDataset,
    trained_unet_model: torch.nn.Module,
) -> Dict[str, Any]:
    """
    Model performance metrics on test dataset, assuming a specific binarization threshold.

    Included metrics (macro-imagewise):
    - Accuracy
    - Precision
    - Recall
    - F1 score
    - IoU

    Returns: Metrics for every image in the test dataset
    """
    len_test_dataset = len(test_dataset)
    plot_basename = OUTPUT_PATH / "test_dataset_plots" / "test_data_index"
    # Create parent directories of plotting directory if non-existent.
    create_directories(plot_basename)

    # Metrics to be evaluated
    accuracy_values = np.zeros(len_test_dataset)
    precision_values = np.zeros(len_test_dataset)
    recall_values = np.zeros(len_test_dataset)
    f1_score_values = np.zeros(len_test_dataset)
    iou_score_values = np.zeros(len_test_dataset)

    trained_unet_model.eval()
    # Create a DataLoader object for test dataset.
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # Loop over test dataset.
    logging.info(f"Computing metrics over test dataset of size {len_test_dataset}")
    for index, (image, ground_truth_mask) in tqdm(
        enumerate(test_loader), total=len(test_loader)
    ):
        # image shape = (1, n_channels, n_y, n_x)
        # mask shape = (1, 1, n_y, n_x)
        ground_truth_mask = ground_truth_mask.int()

        with torch.inference_mode():
            predicted_mask = trained_unet_model(
                image
            )  # output shape = (1, 1, n_y, n_x)

        # Sum metric evaluation over test dataset images.
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

    # Build output dictionary.
    output = {
        "Accuracy": accuracy_values,
        "Precision": precision_values,
        "Recall": recall_values,
        "F1 score": f1_score_values,
        "IoU": iou_score_values,
    }
    write_dict_to_csv(
        output,
        str(OUTPUT_PATH / f"test_dataset_metrics_threshold_{config.threshold:.2f}.csv"),
    )
    logging.info("Metrics computed for test dataset.")
    return output
