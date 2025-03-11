"""
Unit tests for short modules defined in amazon_seg_project.ops.torch_utils
"""

import os
from unittest.mock import patch, MagicMock
from pathlib import Path
import torch
from torch import optim
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from amazon_seg_project.ops.torch_utils import (
    create_data_loaders,
    setup_adam_w,
    save_model_weights,
)


class DummyDataset(Dataset):
    """
    DummyDataset class for testing purposes
    """

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> int:
        return idx


def test_create_data_loaders_default_inputs() -> None:
    """
    Test successful execution of create_data_loaders() with default input arguments.
    """
    train_dataset = DummyDataset(100)
    val_dataset = DummyDataset(50)

    # Call the test functionn.
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset)

    # Verify default batch size.
    assert train_loader.batch_size == 8
    assert val_loader.batch_size == 8

    # Verify shuffle = True only for train_loader.
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)

    # Verify drop_last = False for both loaders.
    assert not train_loader.drop_last
    assert not val_loader.drop_last

    # Assert pin_memory = True for both loaders.
    assert train_loader.pin_memory
    assert val_loader.pin_memory

    # Assert correct number of workers for both loaders.
    assert train_loader.num_workers == os.cpu_count()
    assert val_loader.num_workers == os.cpu_count()


def test_create_data_loaders_custom_inputs() -> None:
    """
    Test successful execution of create_data_loaders() with custom input arguments.
    """
    train_dataset = DummyDataset(100)
    val_dataset = DummyDataset(50)
    batch_size = 32
    num_workers = -8

    # Call the test function.
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, batch_size=batch_size, num_workers=num_workers
    )

    # Verify default batch size.
    assert train_loader.batch_size == batch_size
    assert val_loader.batch_size == batch_size

    # Verify shuffle = True only for train_loader.
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)

    # Verify drop_last = False for both loaders.
    assert not train_loader.drop_last
    assert not val_loader.drop_last

    # Assert pin_memory = True for both loaders.
    assert train_loader.pin_memory
    assert val_loader.pin_memory

    # Assert correct number of workers for both loaders.
    assert train_loader.num_workers == os.cpu_count()
    assert val_loader.num_workers == os.cpu_count()


def test_setup_adam_w_default_lr() -> None:
    """
    Test setup_adam_w() with default learning rate of 1.0e-4
    """
    # Set up a mock model with tensor parameters for testing.
    mock_model = MagicMock()
    param1 = torch.randn(10)
    param2 = torch.randn(20)

    mock_model.parameters.return_value = [
        param1.requires_grad_(),
        param2.requires_grad_(),
    ]

    # Call the test function.
    optimizer = setup_adam_w(mock_model)

    # Assertions
    assert isinstance(optimizer, optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 1.0e-4


def test_setup_adam_w_custom_lr() -> None:
    """
    Test setup_adam_w() with a custom learning rate
    """
    custom_lr = 2.0e-5
    # Set up a mock model with tensor parameters for testing.
    mock_model = MagicMock()
    param1 = torch.randn(10)
    param2 = torch.randn(20)
    mock_model.parameters.return_value = iter(
        [
            param1.requires_grad_(),
            param2.requires_grad_(),
        ]
    )

    # Call the test function.
    optimizer = setup_adam_w(mock_model, lr_initial=custom_lr)

    # Assertions
    assert isinstance(optimizer, optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == custom_lr


@patch("torch.save")
@patch("amazon_seg_project.ops.torch_utils.create_directories")
@patch("logging.info")
def test_save_model_weights_valid_file_extension(
    mock_logging: MagicMock,
    mock_create_directories: MagicMock,
    mock_torch_save: MagicMock,
) -> None:
    """
    Test save_model_weights() for a .pt file extension
    """
    mock_filepath = Path("main/subdir/weights.pt")
    # Create a mock model for testing purposes.
    mock_model = MagicMock()
    model_state_dict = {"param1": 2.75}
    mock_model.state_dict.return_value = model_state_dict

    # Call the test function.
    save_model_weights(mock_model, mock_filepath)

    # Assert that create_directories() was called once.
    mock_create_directories.assert_called_once_with(mock_filepath)

    # Verify that torch.save() was called once.
    mock_torch_save.assert_called_once_with(model_state_dict, str(mock_filepath))

    # Check for correct logging output.
    mock_logging.assert_called_once_with(
        "Saved model weights to %s", str(mock_filepath)
    )


@patch("torch.save")
@patch("amazon_seg_project.ops.torch_utils.create_directories")
@patch("logging.info")
def test_save_model_weights_incomplete_file_extension(
    mock_logging: MagicMock,
    mock_create_directories: MagicMock,
    mock_torch_save: MagicMock,
) -> None:
    """
    Test save_model_weights() for an incomplete file extension
    """
    mock_filepath = Path("main/subdir/weights")
    # Create a mock model for testing purposes.
    mock_model = MagicMock()
    model_state_dict = {"param1": 2.75}
    mock_model.state_dict.return_value = model_state_dict

    # Call the test function.
    save_model_weights(mock_model, mock_filepath)

    # Assert that create_directories() was called once.
    complete_filepath = str(mock_filepath) + ".pt"
    mock_create_directories.assert_called_once_with(complete_filepath)

    # Verify that torch.save() was called once.
    mock_torch_save.assert_called_once_with(model_state_dict, complete_filepath)

    # Check for correct logging output.
    mock_logging.assert_called_once_with("Saved model weights to %s", complete_filepath)
