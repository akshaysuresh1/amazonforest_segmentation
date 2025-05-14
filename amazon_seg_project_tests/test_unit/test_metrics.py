"""
Unit tests for modules defined in amazon_seg_project.ops.metrics
"""

import math
import re
import pytest
import numpy as np
import torch
from amazon_seg_project.ops.metrics import (
    smp_metrics,
    compute_f1_scores,
    iou_metric,
    dice_coefficient,
    dice_loss,
)


def test_smp_metrics_invalid_threshold() -> None:
    """
    Check response of smp_metrics() to a threshold value outside of [0, 1].
    """
    threshold = 1.3
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Threshold must be in the range [0, 1]. Input threshold = {threshold}"
        ),
    ):
        batch_size = 8
        img_height = 64
        img_width = 64

        predicted = torch.randn((batch_size, 1, img_height, img_width))
        target = torch.randn((batch_size, 1, img_height, img_width))
        smp_metrics(predicted, target, threshold=threshold)


def test_smp_metrics_unequal_tensor_shapes() -> None:
    """
    Check response of smp_metrics() to inputs of unequal shapes.
    """
    with pytest.raises(
        ValueError, match="Predicted and target masks have different shapes."
    ):
        batch_size = 8
        img_height = 64
        img_width = 64

        predicted = torch.randn((batch_size, 1, img_height, img_width))
        target = torch.randn((batch_size, 2, img_height, img_width))
        smp_metrics(predicted, target)


def test_smp_metrics_success() -> None:
    """
    Test for response of smp_metrics() to valid inputs
    """
    threshold = 0.4
    # Manually set predicted and target tensors
    predicted = (
        torch.tensor(
            [
                [0.5, 0.1, 0.4, 0.2],
                [0.7, 0.8, 0.9, 0.3],
                [0.0, 0.1, 0.6, 0.6],
                [0.5, 0.1, 0.2, 0.9],
            ],
            dtype=torch.float32,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )  # Shape: (1, 1, 4, 4)

    target = (
        torch.tensor(
            [[1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [1, 0, 1, 1]],
            dtype=torch.int,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )  # Shape: (1, 1, 4, 4)

    # Metric calculations
    binarized_predicted = torch.where(predicted >= threshold, 1.0, 0.0)
    tp = ((binarized_predicted == 1) & (target == 1)).sum().item()  # True positives
    fp = ((binarized_predicted == 1) & (target == 0)).sum().item()  # False positives
    fn = ((binarized_predicted == 0) & (target == 1)).sum().item()  # False negatives
    tn = ((binarized_predicted == 0) & (target == 0)).sum().item()  # True negatives
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    iou_score = tp / (tp + fp + fn)

    # Call the test function.
    result = smp_metrics(predicted, target, threshold=0.4)

    # Assertions
    rtol = 1.0e-6  # Relative tolerance for floating-point comparison
    atol = 1.0e-8  # Absolute tolerance for floating-point comparison
    assert "Accuracy" in result
    assert math.isclose(result.get("Accuracy"), accuracy, rel_tol=rtol, abs_tol=atol)
    assert "Precision" in result
    assert math.isclose(result.get("Precision"), precision, rel_tol=rtol, abs_tol=atol)
    assert "Recall" in result
    assert math.isclose(result.get("Recall"), recall, rel_tol=rtol, abs_tol=atol)
    assert "F1 score" in result
    assert math.isclose(result.get("F1 score"), f1_score, rel_tol=rtol, abs_tol=atol)
    assert "IoU" in result
    assert math.isclose(result.get("IoU"), iou_score, rel_tol=rtol, abs_tol=atol)


def test_compute_f1_scores_unequal_input_shapes() -> None:
    """
    Test for compute_f1_scores() with unequal input array shapes.
    """
    with pytest.raises(
        ValueError, match="Precision and recall array have unequal shapes."
    ):
        # Define test inputs.
        precision_values = np.random.randn(5)
        recall_values = np.random.randn(4)

        # Call the test function.
        compute_f1_scores(precision_values, recall_values)


def test_compute_f1_scores_success() -> None:
    """
    Test for successful execution of compute_f1_scores().
    """
    # Set up test inputs.
    precision_values = np.random.randn(5)
    recall_values = np.random.randn(5)
    expected_output = (
        2 * precision_values * recall_values / (precision_values + recall_values)
    )

    # Call the test function.
    result = compute_f1_scores(precision_values, recall_values)

    # Assertion
    np.testing.assert_array_almost_equal(expected_output, result)


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
