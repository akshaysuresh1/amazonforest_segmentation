"""
Unit tests for modules defined in amazon_seg_project.ops.loss_functions
"""

import pytest
import torch
from amazon_seg_project.ops.metrics import iou_metric, dice_coefficient, dice_loss


def test_iou_unequal_arrayshapes() -> None:
    """
    Test for response of iou_metric() to inputs of unequal shapes
    """
    with pytest.raises(
        ValueError, match="Predicted and target masks have different shapes."
    ):
        predicted = torch.ones((2, 8, 8))
        target = torch.ones((4, 8, 8))

        _ = iou_metric(predicted, target)


def test_iou_zero_union() -> None:
    """
    Test for iou_metric() when the union of two masks is null
    """
    predicted = torch.zeros((2, 4, 5))
    target = torch.zeros((2, 4, 5))

    iou_value = iou_metric(predicted, target)
    assert torch.isclose(iou_value, torch.zeros(1))


def test_iou_success() -> None:
    """
    Test for successful computation of IoU metric
    """
    predicted = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    expected_iou = 1.0 / 3

    # Call the test function.
    iou_value = iou_metric(predicted=predicted, target=target)

    assert torch.isclose(iou_value, torch.tensor([expected_iou]))


def test_dice_coefficient_unequal_arrayshapes() -> None:
    """
    Test for response of dice_coefficient() to inputs of unequal shapes
    """
    with pytest.raises(
        ValueError, match="Predicted and target masks have different shapes."
    ):
        predicted = torch.ones((2, 8, 8))
        target = torch.ones((4, 8, 8))

        _ = dice_coefficient(predicted, target)


def test_dice_coefficient_success() -> None:
    """
    Test for successful computation of Dice coefficient
    """
    smooth = 1.0e-5
    predicted = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    expected_coeff = (2.0 + smooth) / (4.0 + smooth)

    # Call the test function.
    coeff_value = dice_coefficient(predicted=predicted, target=target, smooth=smooth)

    assert torch.isclose(coeff_value, torch.tensor([expected_coeff]))


def test_dice_loss_success() -> None:
    """
    Test for successful computation of Dice loss
    """
    smooth = 1.0e-5
    predicted = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    expected_loss = 1 - (2.0 + smooth) / (4.0 + smooth)

    # Call the test function.
    loss_value = dice_loss(predicted=predicted, target=target, smooth=smooth)

    assert torch.isclose(loss_value, torch.tensor([expected_loss]))
