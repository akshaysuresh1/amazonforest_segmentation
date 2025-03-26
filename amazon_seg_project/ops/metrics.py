"""
Metrics and loss functions for model performance evaluation
"""

import torch
from dagster import op, In, Out, Any


@op(ins={"predicted": In(Any), "target": In(Any)}, out=Out(Any))
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


@op(ins={"predicted": In(Any), "target": In(Any), "smooth": In(float)}, out=Out(Any))
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


@op(ins={"predicted": In(Any), "target": In(Any), "smooth": In(float)}, out=Out(Any))
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
