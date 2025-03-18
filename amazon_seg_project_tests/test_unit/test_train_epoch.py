"""
Unit tests for train_epoch() defined in amazon_seg_project.ops.torch_utils
"""

from unittest.mock import MagicMock
import pytest
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from segmentation_models_pytorch import Unet
from amazon_seg_project.ops.torch_utils import train_epoch
from amazon_seg_project.ops.loss_functions import dice_loss
from amazon_seg_project.resources import device


def test_train_epoch_moves_data_to_device() -> None:
    """
    Check for transfer of images and masks to specified device
    """
    # Create dummy data for training.
    batch_size = 4
    in_channels = 3
    image_height = 64
    image_width = 64
    images = torch.randn(batch_size, in_channels, image_height, image_width)
    masks = torch.randint(
        low=0, high=2, size=(batch_size, 1, image_height, image_width)
    )
    train_dataset = TensorDataset(images, masks)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator(device=device.type),
    )

    # Set up mock model, optimizer, and loss criterion.
    mock_model = MagicMock(
        return_value=torch.randn(batch_size, 1, image_height, image_width)
    )
    mock_model.train = MagicMock()
    mock_optimizer = MagicMock(zero_grad=MagicMock(), step=MagicMock())
    mock_criterion = MagicMock(return_value=torch.tensor(0.5, requires_grad=True))

    # Call the test function.
    _ = train_epoch(
        model=mock_model,
        train_loader=train_loader,
        optimizer=mock_optimizer,
        criterion=mock_criterion,
        train_device=device,
    )

    # Assertions
    mock_model.assert_called()
    mock_model.to.assert_called_once_with(device)
    mock_criterion.assert_called()
    args, _ = mock_criterion.call_args
    # Check images.device
    assert args[0].device.type == device.type
    # Check masks.device
    assert args[1].device.type == device.type


def test_train_epoch_optimizer_called() -> None:
    """
    Test for verification of correct calls to optimizer
    """
    # Set up mock optimizer.
    mock_optimizer = MagicMock(zero_grad=MagicMock(), step=MagicMock())

    # Create dummy data and model for testing.
    batch_size = 4
    img_count = 2 * batch_size
    in_channels = 4
    image_height = 64
    image_width = 64
    images = torch.randn(img_count, in_channels, image_height, image_width)
    masks = torch.randint(low=0, high=2, size=(img_count, 1, image_height, image_width))
    train_dataset = TensorDataset(images, masks)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator(device=device.type),
    )

    model = Unet(
        encoder="resnet50",
        encoder_weights=None,
        in_channels=in_channels,
        activation="sigmoid",
    )
    criterion = dice_loss

    # Call the test function.
    _ = train_epoch(model, train_loader, mock_optimizer, criterion, device)

    # Assertions
    mock_optimizer.zero_grad.assert_called()
    mock_optimizer.step.assert_called()


def test_train_epoch_handles_empty_dataloader() -> None:
    """
    Test for successful handling of empty training data loader
    """
    with pytest.raises(ValueError, match="No training data found."):
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
        train_dataset = TensorDataset(images, masks)
        empty_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            drop_last=True,
            generator=torch.Generator(device=device.type),
        )

        # Define the model, optimizer, and loss criterion.
        model = Unet(
            encoder="resnet50",
            encoder_weights=None,
            in_channels=in_channels,
            activation="sigmoid",
        )
        optimizer = AdamW(model.parameters(), lr=1.0e-5)
        criterion = dice_loss

        # Call the test function.
        _ = train_epoch(model, empty_loader, optimizer, criterion, device)


def test_train_epoch_success() -> None:
    """
    Test successful execution of train_epoch() on mock data.
    """
    # Create dummy data and model for testing.
    batch_size = 4
    img_count = 4 * batch_size
    in_channels = 4
    image_height = 64
    image_width = 64

    images = torch.randn(img_count, in_channels, image_height, image_width)
    masks = torch.randint(low=0, high=2, size=(img_count, 1, image_height, image_width))
    train_dataset = TensorDataset(images, masks)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=torch.Generator(device=device.type),
    )

    model = Unet(
        encoder="resnet50",
        encoder_weights=None,
        in_channels=in_channels,
        activation="sigmoid",
    )

    # Set up optimizer and loss criterion.
    optimizer = AdamW(model.parameters(), lr=1.0e-5)
    criterion = dice_loss

    # Train for one epoch.
    loss_epoch1 = train_epoch(model, train_loader, optimizer, criterion, device)

    # Assert for float output type of batch-averaged loss.
    assert isinstance(loss_epoch1, float)

    # Verify that loss decreases after a few epochs.
    for _ in range(5):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
    assert loss <= loss_epoch1
