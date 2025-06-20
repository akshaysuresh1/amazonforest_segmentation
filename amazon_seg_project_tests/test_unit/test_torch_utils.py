"""
Unit tests for short modules defined in amazon_seg_project.ops.torch_utils
"""

import os
from unittest.mock import patch, MagicMock
from pathlib import Path
import pytest
import torch
from torch import optim
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from amazon_seg_project.ops.torch_utils import (
    worker_seed_fn,
    create_data_loaders,
    setup_adam_w,
    save_model_weights,
    apply_prob_thresholding,
)
from amazon_seg_project.ops.model_checks import are_state_dicts_equal


class DummyModel(torch.nn.Module):
    """
    Dummy model for testing purposes
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)


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


@patch("random.seed")
@patch("numpy.random.seed")
@patch("torch.manual_seed")
def test_worker_seed_fn_setup(
    mock_torch_seed: MagicMock, mock_numpy_seed: MagicMock, mock_random_seed: MagicMock
) -> None:
    """
    Test correct setup of worker_seed_fn().
    """
    worker_id = 3
    seed = torch.initial_seed() % 2**32 + worker_id

    # Call the test function.
    worker_seed_fn(3)

    # Assert seed initializations.
    mock_numpy_seed.assert_called_once_with(seed)
    mock_random_seed.assert_called_once_with(seed)
    mock_torch_seed.assert_called_once_with(seed)


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

    # Assert for worker_init_fn.
    assert train_loader.worker_init_fn == worker_seed_fn
    assert val_loader.worker_init_fn == worker_seed_fn

    # Check generator seed.
    assert train_loader.generator.initial_seed() == 137
    assert val_loader.generator.initial_seed() == 137


def test_create_data_loaders_custom_inputs() -> None:
    """
    Test successful execution of create_data_loaders() with custom input arguments.
    """
    train_dataset = DummyDataset(100)
    val_dataset = DummyDataset(50)
    batch_size = 32
    num_workers = -8
    generator_seed = 28

    # Call the test function.
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=generator_seed,
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

    # Assert for worker_init_fn.
    assert train_loader.worker_init_fn == worker_seed_fn
    assert val_loader.worker_init_fn == worker_seed_fn

    # Check generator seed.
    assert train_loader.generator.initial_seed() == generator_seed
    assert val_loader.generator.initial_seed() == generator_seed


def test_setup_adam_w_default_lr() -> None:
    """
    Test setup_adam_w() with default learning rate of 1.0e-4.
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
    assert optimizer.param_groups[0].get("lr") == 1.0e-4


def test_setup_adam_w_custom_lr() -> None:
    """
    Test setup_adam_w() with a custom learning rate.
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
    assert optimizer.param_groups[0].get("lr") == custom_lr


@patch("torch.save")
@patch("amazon_seg_project.ops.torch_utils.create_directories")
@patch("logging.info")
def test_save_model_weights_valid_file_extension(
    mock_logging: MagicMock,
    mock_create_directories: MagicMock,
    mock_torch_save: MagicMock,
) -> None:
    """
    Test save_model_weights() for a .pt file extension.
    """
    mock_filepath = Path("main/subdir/weights.pt")
    # Create a dummy model for testing purposes.
    mock_model = DummyModel()

    # Call the test function.
    save_model_weights(mock_model, mock_filepath)

    # Assert that create_directories() was called once.
    mock_create_directories.assert_called_once_with(mock_filepath)

    # Verify that torch.save() was called once.
    mock_torch_save.assert_called_once
    args, _ = mock_torch_save.call_args
    assert are_state_dicts_equal(args[0], mock_model.state_dict())
    assert args[1] == str(mock_filepath)

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
    Test save_model_weights() for an incomplete file extension.
    """
    mock_filepath = Path("main/subdir/weights")
    # Create a dummy model for testing purposes.
    mock_model = DummyModel()

    # Call the test function.
    save_model_weights(mock_model, mock_filepath)

    # Assert that create_directories() was called once.
    complete_filepath = str(mock_filepath) + ".pt"
    mock_create_directories.assert_called_once_with(complete_filepath)

    # Verify that torch.save() was called once.
    mock_torch_save.assert_called_once
    args, _ = mock_torch_save.call_args
    assert are_state_dicts_equal(args[0], mock_model.state_dict())
    assert args[1] == complete_filepath

    # Check for correct logging output.
    mock_logging.assert_called_once_with("Saved model weights to %s", complete_filepath)


@patch("torch.save")
@patch("amazon_seg_project.ops.torch_utils.create_directories")
@patch("logging.info")
def test_save_model_weights_with_data_parallel(
    mock_logging: MagicMock,
    mock_create_directories: MagicMock,
    mock_torch_save: MagicMock,
) -> None:
    """
    Test save_model_weights() for a model wrapped in torch.nn.DataParallel.
    """
    mock_filepath = Path("main/subdir/weights")
    # Create a dummy model wrapped in torch.nn.DataParallel for testing purposes.
    mock_model = torch.nn.DataParallel(DummyModel())

    # Call the test function.
    save_model_weights(mock_model, mock_filepath)

    # Assert that create_directories() was called once.
    complete_filepath = str(mock_filepath) + ".pt"
    mock_create_directories.assert_called_once_with(complete_filepath)

    # Verify that torch.save() was called once.
    mock_torch_save.assert_called_once
    args, _ = mock_torch_save.call_args
    assert are_state_dicts_equal(args[0], mock_model.module.state_dict())
    assert args[1] == complete_filepath

    # Check for correct logging output.
    mock_logging.assert_called_once_with("Saved model weights to %s", complete_filepath)


def test_prob_thresholding_tensor_max_gtr_than_one() -> None:
    """
    Check for correct raise of ValueError when prob_tensor.max() > 1.
    """
    with pytest.raises(
        ValueError, match="Input tensor does not contain probabilities. Max value > 1."
    ):
        # Create inputs.
        prob_tensor = torch.tensor([[1.0, 0.0], [1.6, 0.1]])
        threshold_prob = 0.5

        # Call the test function.
        _ = apply_prob_thresholding(prob_tensor, threshold_prob=threshold_prob)


def test_prob_thresholding_tensor_min_negative() -> None:
    """
    Check for correct raise of ValueError when prob_tensor.min() < 0.
    """
    with pytest.raises(
        ValueError, match="Input tensor does not contain probabilities. Min value < 0."
    ):
        # Create inputs.
        prob_tensor = torch.tensor([[1.0, 0.0], [0.6, -0.1]])
        threshold_prob = 0.5

        # Call the test function.
        _ = apply_prob_thresholding(prob_tensor, threshold_prob=threshold_prob)


def test_prob_thresholding_invalid_threshold() -> None:
    """
    Check for correct raise of ValueError when threshold is not a probability.
    """
    with pytest.raises(ValueError, match="Threshold value must be a probability."):
        # Create inputs.
        prob_tensor = torch.tensor([[1.0, 0.0], [0.6, 0.1]])
        threshold_prob = -0.5

        # Call the test function.
        _ = apply_prob_thresholding(prob_tensor, threshold_prob=threshold_prob)


def test_prob_thresholding_success() -> None:
    """
    Test for successful execution of apply_prob_thresholding()
    """
    # Create inputs.
    prob_tensor = torch.tensor([[1.0, 0.1], [0.4, 0.7]])
    threshold_prob = 0.5
    expected_tensor = torch.tensor([[1, 0], [0, 1]]).int()

    # Call the test function.
    thresholded_binary_tensor = apply_prob_thresholding(
        prob_tensor, threshold_prob=threshold_prob
    )

    # Assertion
    assert torch.equal(expected_tensor, thresholded_binary_tensor), (
        "Thresholded tensor does not match expected output."
    )
