"""
Unit tests for op "run_wandb_training"
"""

from unittest.mock import patch, MagicMock
from dagster import RunConfig
from amazon_seg_project.assets import (
    unet_model,
    data_training,
    data_validation,
    afs_training_dataset,
    afs_validation_dataset,
)
from amazon_seg_project.config import BasicUnetConfig, ModelTrainingConfig
from amazon_seg_project.ops.wandb_utils import run_wandb_training


@patch("amazon_seg_project.ops.wandb_utils.materialize")
@patch("amazon_seg_project.ops.wandb_utils.train_unet")
def test_run_wandb_training(
    mock_train_unet: MagicMock, mock_materialize: MagicMock
) -> None:
    """
    Test execution of run_wandb_training() using mocked internal functions
    """
    # Define test config.
    test_config = ModelTrainingConfig(
        encoder_name="resnet34",
        batch_size=32,
        lr_initial=0.001,
        project="test_project",
        seed=47,
        max_epochs=16,
    )

    # Derivaties from test config
    assets = [
        data_training,
        data_validation,
        afs_training_dataset,
        afs_validation_dataset,
        unet_model,
    ]
    encoder = test_config.encoder_name
    batch_size = test_config.batch_size
    lr_initial = test_config.lr_initial
    mock_unet_config = BasicUnetConfig(encoder_name=test_config.encoder_name)
    mock_wandb_config = {
        "project": test_config.project,
        "name": f"{encoder}_batch{batch_size}_lr{lr_initial}",
        "seed": test_config.seed,
        "encoder_name": encoder,
        "batch_size": batch_size,
        "lr_initial": lr_initial,
        "max_epochs": test_config.max_epochs,
    }

    # Create mocks.
    mock_training_dataset = MagicMock(name="training_dataset")
    mock_validation_dataset = MagicMock(name="validation_dataset")
    mock_unet_model = MagicMock(name="unet_model")
    mock_materialize.return_value = MagicMock(
        asset_value=lambda name: {  # pylint: disable=W0108
            "training_dataset": mock_training_dataset,
            "validation_dataset": mock_validation_dataset,
            "basic_unet_model": mock_unet_model,
        }.get(name)
    )

    # Call the test function.
    run_wandb_training(test_config)

    # Assertions
    mock_materialize.assert_called_once_with(
        assets, run_config=RunConfig({"basic_unet_model": mock_unet_config})
    )
    mock_result = mock_materialize.return_value
    assert mock_result.asset_value("training_dataset") == mock_training_dataset
    assert mock_result.asset_value("validation_dataset") == mock_validation_dataset
    assert mock_result.asset_value("basic_unet_model") == mock_unet_model
    mock_train_unet.assert_called_once_with(
        mock_wandb_config,
        mock_training_dataset,
        mock_validation_dataset,
        mock_unet_model,
    )
