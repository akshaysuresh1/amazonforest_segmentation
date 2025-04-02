"""
Unit tests for validate_epoch() defined in amazon_seg_project.ops.torch_utils
"""

from unittest.mock import patch, MagicMock
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import Unet
from amazon_seg_project.ops.torch_utils import validate_epoch
from amazon_seg_project.resources import device


@patch("amazon_seg_project.ops.torch_utils.smp_metrics")
def test_val_epoch_moves_data_to_device(mock_metric_evaluation: MagicMock) -> None:
    """
    Check for transfer of images and masks to specified device.
    """
    threshold = 0.25
    # Create dummy data for validation.
    batch_size = 4
    in_channels = 3
    image_height = 64
    image_width = 64
    images = torch.randn(batch_size, in_channels, image_height, image_width)
    masks = torch.randint(
        low=0, high=2, size=(batch_size, 1, image_height, image_width)
    )
    val_dataset = TensorDataset(images, masks)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Set up mock model and loss criterion.
    mock_model = MagicMock(
        return_value=torch.randn(batch_size, 1, image_height, image_width)
    )
    mock_model.eval = MagicMock()
    mock_criterion = MagicMock(return_value=torch.tensor(0.5, requires_grad=True))

    # Call the test function.
    _ = validate_epoch(
        model=mock_model,
        val_loader=val_loader,
        criterion=mock_criterion,
        val_device=device,
        threshold=threshold,
    )

    # Assertions
    mock_model.assert_called()
    mock_model.to.assert_called_once_with(device)
    mock_criterion.assert_called_once()
    mock_metric_evaluation.assert_called_once()

    # Check if data are moved to the correct device during validation.
    for batch in val_loader:
        batched_images, batched_masks = batch
        batched_images, batched_masks = (
            batched_images.to(device),
            batched_masks.to(device),
        )
        # Check if the images and masks are on the correct device.
        assert batched_images.device.type == device.type, (
            "Images are not on the correct device."
        )
        assert batched_masks.device.type == device.type, (
            "Masks are not on the correct device."
        )
        break  # Only need to check the first batch


def test_validate_epoch_handles_empty_dataloader() -> None:
    """
    Test for successful handling of empty validation data loader
    """
    with pytest.raises(ValueError, match="No validation data found."):
        # Set up dummy data loader with image count < batch size and drop_last = True.
        batch_size = 4
        img_count = 2
        in_channels = 4
        image_height = 64
        image_width = 64
        images = torch.randn(img_count, in_channels, image_height, image_width)
        masks = torch.randint(
            low=0, high=2, size=(img_count, 1, image_height, image_width)
        )
        val_dataset = TensorDataset(images, masks)
        empty_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            drop_last=True,
        )

        # Define the model, optimizer, and loss criterion.
        model = Unet(
            encoder="resnet50",
            encoder_weights=None,
            in_channels=in_channels,
            activation="sigmoid",
        )
        criterion = smp.losses.DiceLoss(
            mode="binary", from_logits=False, smooth=1.0e-6, eps=0.0
        )

        # Call the test function.
        _ = validate_epoch(model, empty_loader, criterion, device, threshold=0.7)


def test_validate_epoch_success() -> None:
    """
    Test successful execution of validate_epoch() for multiple batches
    """
    threshold = 0.41
    # Set up data loader and model.
    batch_size = 4
    img_count = 4 * batch_size
    in_channels = 4
    image_height = 64
    image_width = 64

    images = torch.randn(img_count, in_channels, image_height, image_width)
    masks = torch.randint(low=0, high=2, size=(img_count, 1, image_height, image_width))
    val_dataset = TensorDataset(images, masks)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = Unet(
        encoder="resnet50",
        encoder_weights=None,
        in_channels=in_channels,
        activation="sigmoid",
    )

    # Validate for one epoch.
    criterion = smp.losses.DiceLoss(
        mode="binary", from_logits=False, smooth=1.0e-6, eps=0.0
    )
    validation_results = validate_epoch(
        model, val_loader, criterion, device, threshold=threshold
    )

    # Assert for float output type of batch-averaged loss.
    assert isinstance(validation_results.get("val_loss"), float)
    # Assert for float output type and [0, 1] range of various metrics returned.
    assert isinstance(validation_results.get("Accuracy"), float)
    assert 0 <= validation_results.get("Accuracy") <= 1

    assert isinstance(validation_results.get("Precision"), float)
    assert 0 <= validation_results.get("Precision") <= 1

    assert isinstance(validation_results.get("Recall"), float)
    assert 0 <= validation_results.get("Recall") <= 1

    assert isinstance(validation_results.get("F1 score"), float)
    assert 0 <= validation_results.get("F1 score") <= 1

    assert isinstance(validation_results.get("IoU"), float)
    assert 0 <= validation_results.get("IoU") <= 1
