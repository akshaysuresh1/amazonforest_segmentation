"""
Unit tests for op "run_wandb_training"
"""

from typing import Dict, Any
from unittest.mock import patch, MagicMock
from dagster import RunConfig
from amazon_seg_project.assets import (
    unet_model,
    data_training,
    data_validation,
    afs_training_dataset,
    afs_validation_dataset,
)
from amazon_seg_project.config import (
    TrainingDatasetConfig,
    BasicUnetConfig,
)
from amazon_seg_project.ops.wandb_utils import run_wandb_exp


@patch("amazon_seg_project.ops.wandb_utils.materialize_to_memory")
@patch("amazon_seg_project.ops.wandb_utils.train_unet")
@patch("wandb.init")
def test_run_wandb_training(
    mock_wandb_init: MagicMock,
    mock_train_unet: MagicMock,
    mock_materialize_to_memory: MagicMock,
) -> None:
    """
    Test execution of run_wandb_training() using mocked internal functions
    """
    # Create mock W&B config.
    mock_wandb_config: Dict[str, Any] = {
        "seed": 67,
        "encoder_name": "resnet34",
        "batch_size": 8,
        "lr_initial": 1.0e-5,
        "max_epochs": 25,
        "horizontal_flip_prob": 0.8,
        "vertical_flip_prob": 0.1,
        "rotate90_prob": 0.42,
    }

    # Set up mock W&B run.
    mock_run = MagicMock(name="w&b run")
    mock_run.config = mock_wandb_config
    mock_wandb_init.return_value.__enter__.return_value = mock_run

    # Create mock datasets and mock model.
    mock_unet_model = MagicMock(name="mock_unet_model")
    mock_training_dataset = MagicMock(name="mock_training_dataset")
    mock_validation_dataset = MagicMock(name="mock_validation_dataset")
    mock_materialize_to_memory.return_value = MagicMock(
        asset_value=lambda name: {  # pylint: disable=W0108
            "training_dataset": mock_training_dataset,
            "validation_dataset": mock_validation_dataset,
            "basic_unet_model": mock_unet_model,
        }.get(name)
    )

    # Assemble assets and run config.
    assets = [
        unet_model,
        data_training,
        data_validation,
        afs_training_dataset,
        afs_validation_dataset,
    ]
    mock_unet_config = BasicUnetConfig(
        encoder_name=mock_wandb_config["encoder_name"],
        model_seed=mock_wandb_config["seed"],
    )
    mock_train_config = TrainingDatasetConfig(
        horizontal_flip_prob=mock_wandb_config["horizontal_flip_prob"],
        vertical_flip_prob=mock_wandb_config["vertical_flip_prob"],
        rotate90_prob=mock_wandb_config["rotate90_prob"],
        augmentation_seed=mock_wandb_config["seed"],
    )
    mock_run_config = RunConfig(
        {
            "basic_unet_model": mock_unet_config,
            "training_dataset": mock_train_config,
        }
    )

    # Call the test function.
    run_wandb_exp(mock_wandb_config)

    # Assertions
    mock_wandb_init.assert_called_once_with(config=mock_wandb_config)
    mock_materialize_to_memory.assert_called_once_with(
        assets, run_config=mock_run_config
    )
    mock_result = mock_materialize_to_memory.return_value
    assert mock_result.asset_value("basic_unet_model") == mock_unet_model
    assert mock_result.asset_value("training_dataset") == mock_training_dataset
    assert mock_result.asset_value("validation_dataset") == mock_validation_dataset

    mock_train_unet.assert_called_once_with(
        mock_run,
        mock_training_dataset,
        mock_validation_dataset,
        mock_unet_model,
    )
