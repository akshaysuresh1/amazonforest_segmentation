"""
Unit tests for op "run_wandb_exp" in wandb_utils.py
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
def test_run_wandb_exp(
    mock_wandb_init: MagicMock,
    mock_train_unet: MagicMock,
    mock_materialize_to_memory: MagicMock,
) -> None:
    """
    Test execution of run_wandb_exp() using mocked internal functions
    """
    # Create mock W&B config.
    config_dict: Dict[str, Any] = {
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
    mock_run.config = config_dict
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
        encoder_name=config_dict.get("encoder_name", ""),
        model_seed=config_dict.get("seed", ""),
    )
    mock_train_config = TrainingDatasetConfig(
        horizontal_flip_prob=config_dict.get("horizontal_flip_prob", ""),
        vertical_flip_prob=config_dict.get("vertical_flip_prob", ""),
        rotate90_prob=config_dict.get("rotate90_prob", ""),
        augmentation_seed=config_dict.get("seed", ""),
    )
    mock_run_config = RunConfig(
        {
            "basic_unet_model": mock_unet_config,
            "training_dataset": mock_train_config,
        }
    )

    # Call the test function.
    run_wandb_exp()

    # Assertions
    mock_wandb_init.assert_called_once_with(config=None)
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
