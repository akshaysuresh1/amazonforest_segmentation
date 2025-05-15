"""
Metrics and loss functions for binary segmentation model performance evaluation
"""

from typing import Dict
import numpy as np
import numpy.typing as npt
import torch
import segmentation_models_pytorch as smp
from dagster import op, In, Out
from dagster import Any as dg_Any


@op(
    ins={"predicted": In(dg_Any), "target": In(dg_Any), "threshold": In(float)},
    out=Out(Dict[str, float]),
)
def smp_metrics(
    predicted: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluation metrics computed using routines defined in segmentation_models_pytorch

    Metrics included:
        Accuracy, Precision, Recall, F1 score, IoU

    Args:
        predicted: Predicted mask
        target: Expected or true mask
        threshold: Binarization threshold (d: 0.5)
    """
    if not 0 <= threshold <= 1:
        raise ValueError(
            f"Threshold must be in the range [0, 1]. Input threshold = {threshold}"
        )

    if predicted.shape != target.shape:
        raise ValueError("Predicted and target masks have different shapes.")

    # Compute confusion matrix elements.
    tp, fp, fn, tn = smp.metrics.get_stats(
        predicted, target, mode="binary", threshold=threshold
    )

    # Metric evaluation
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro-imagewise").item()
    precision = smp.metrics.precision(
        tp, fp, fn, tn, reduction="macro-imagewise"
    ).item()
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="macro-imagewise").item()
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise").item()
    iou_score = smp.metrics.iou_score(
        tp, fp, fn, tn, reduction="macro-imagewise"
    ).item()

    # Results dictionary
    results = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 score": f1_score,
        "IoU": iou_score,
    }
    return results


@op(ins={"precision": In(dg_Any), "recall": In(dg_Any)}, out=Out(dg_Any))
def compute_f1_scores(
    precision: npt.NDArray[np.float_], recall: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Compute an array of F1 scores from input precision and recall arrays.

    Args:
        precision: An array of precision values
        recall: An array of recall values
    """
    if precision.shape != recall.shape:
        raise ValueError("Precision and recall array have unequal shapes.")

    f1_scores = 2.0 * precision * recall / (precision + recall)
    return f1_scores


@op(ins={"predicted": In(dg_Any), "target": In(dg_Any)}, out=Out(dg_Any))
def iou_metric(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Intersection over Union (IoU) metric between predicted and target masks.

    Args:
        predicted: Predicted mask
        target: Expected or true mask
    """
    if predicted.shape != target.shape:
        raise ValueError("Predicted and target masks have different shapes.")

    predicted_flat = predicted.flatten()
    target_flat = target.flatten()
    intersection = (predicted_flat * target_flat).sum()
    union = predicted_flat.sum() + target_flat.sum() - intersection

    # Avoid division by zero.
    if union == 0:
        return torch.tensor(0.0)
    # Calculate IoU.
    iou_value = intersection / union
    return iou_value


@op(
    ins={"predicted": In(dg_Any), "target": In(dg_Any), "smooth": In(float)},
    out=Out(dg_Any),
)
def dice_coefficient(
    predicted: torch.Tensor, target: torch.Tensor, smooth: float = 1.0e-6
) -> torch.Tensor:
    """
    Calculate the Dice coefficient between the predicted and target masks.

    Args:
        predicted: Predicted mask
        target: Expected or true mask
        smooth: Smoothness parameter (d: 1.0e-6)
    """
    if predicted.shape != target.shape:
        raise ValueError("Predicted and target masks have different shapes.")

    predicted_flat = predicted.flatten()
    target_flat = target.flatten()

    intersection = (predicted_flat * target_flat).sum()
    coefficient = (2 * intersection + smooth) / (
        predicted_flat.sum() + target_flat.sum() + smooth
    )

    return coefficient


@op(
    ins={"predicted": In(dg_Any), "target": In(dg_Any), "smooth": In(float)},
    out=Out(dg_Any),
)
def dice_loss(
    predicted: torch.Tensor, target: torch.Tensor, smooth: float = 1.0e-6
) -> torch.Tensor:
    """
    Dice Loss computation between the predicted and target masks

    Args:
        predicted: Predicted mask
        target: Expected or true mask
        smooth: Smoothness parameter (d: 1.0e-6)
    """
    loss = 1 - dice_coefficient(predicted, target, smooth)
    return loss
