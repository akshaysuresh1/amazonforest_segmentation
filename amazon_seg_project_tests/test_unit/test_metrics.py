"""
Unit tests for modules defined in amazon_seg_project.ops.loss_functions
"""

import pytest
import torch
from amazon_seg_project.ops.metrics import dice_coefficient, dice_loss


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
