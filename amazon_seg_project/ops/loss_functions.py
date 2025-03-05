"""
Loss function for model performance evaluation
"""

import torch
from dagster import op, In, Out, Any


@op(ins={"predicted": In(Any), "target": In(Any), "smooth": In(float)}, out=Out(Any))
def dice_loss(
    predicted: torch.Tensor, target: torch.Tensor, smooth: float = 1.0e-6
) -> torch.Tensor:
    """
    Dice Loss computation between predicted and target masks

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
    dice_coefficient = (2 * intersection + smooth) / (
        predicted_flat.sum() + target_flat.sum() + smooth
    )
    loss = 1 - dice_coefficient
    return loss
