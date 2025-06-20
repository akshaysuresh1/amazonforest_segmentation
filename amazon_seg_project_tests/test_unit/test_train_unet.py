"""
Unit test for op "train_unet"
"""

from typing import List, Dict, Any
from unittest.mock import patch, MagicMock, call
from segmentation_models_pytorch import Unet
from amazon_seg_project.assets import SegmentationDataset
from amazon_seg_project.data_paths import OUTPUT_PATH
from amazon_seg_project.ops.aug_utils import get_aug_pipeline
from amazon_seg_project.ops.train_unet import train_unet
from amazon_seg_project.ops.file_naming_conventions import (
    name_weights_file,
    name_losscurve_csv_file,
)
from amazon_seg_project.resources import device


@patch("logging.info")
def test_skip_training_for_epochs_less_than_1(mock_logging: MagicMock) -> None:
    """
    Verify that model training is skipped if max number of training epochs < 1.
    """
    # Set up mock W&B run config.
    mock_wandb_config = {
        "seed": 56,
        "threshold": 0.45,
        "encoder_name": "resnet50",
        "batch_size": 4,
        "lr_initial": 0.001,
        "max_epochs": 0,
    }

    # Set up mock W&B run.
    mock_run = MagicMock(name="mock-wandb-run")
    mock_run.config = mock_wandb_config

    # Create mocks for model, training, and validation datasets.
    mock_training_dataset = MagicMock(name="training-dataset")
    mock_validation_dataset = MagicMock(name="validation-dataset")
    mock_model = MagicMock(name="model")

    # Call the test function.
    train_unet(mock_run, mock_training_dataset, mock_validation_dataset, mock_model)

    # Assertions
    mock_logging.assert_called_once_with(
        "No. of training epochs < 1. Model training skipped."
    )


@patch("amazon_seg_project.ops.train_unet.create_and_log_wandb_artifact")
@patch("amazon_seg_project.ops.train_unet.write_loss_data_to_csv")
@patch("amazon_seg_project.ops.train_unet.save_model_weights")
@patch("amazon_seg_project.ops.train_unet.validate_epoch")
@patch("amazon_seg_project.ops.train_unet.train_epoch")
@patch("amazon_seg_project.ops.train_unet.smp.losses.DiceLoss")
@patch("amazon_seg_project.ops.train_unet.setup_adam_w")
@patch("amazon_seg_project.ops.train_unet.create_data_loaders")
@patch("torch.nn.DataParallel")
@patch("torch.cuda.device_count", return_value=2)
@patch("torch.cuda.manual_seed_all")
@patch("torch.cuda.manual_seed")
@patch("torch.manual_seed")
def test_single_epoch_training(
    mock_torch_manual_seed: MagicMock,
    mock_torch_cuda_seed: MagicMock,
    mock_torch_cuda_all_seed: MagicMock,
    mock_torch_cuda_device_count: MagicMock,
    mock_torch_DataParallel: MagicMock,
    mock_create_data_loaders: MagicMock,
    mock_adamw_optimizer: MagicMock,
    mock_dice_loss: MagicMock,
    mock_train_epoch: MagicMock,
    mock_validate_epoch: MagicMock,
    mock_save_model_weights: MagicMock,
    mock_write_loss_data: MagicMock,
    mock_create_log_wandb_artifact: MagicMock,
) -> None:
    """
    Check for correct execution of a single epoch of U-net training.
    This test assumes a multi-GPU system.
    """
    # Set up mock W&B run config.
    mock_wandb_config = {
        "seed": 42,
        "threshold": 0.41,
        "encoder_name": "resnet50",
        "batch_size": 4,
        "lr_initial": 0.001,
        "max_epochs": 1,
    }
    mock_wandb_run = MagicMock(name="mock_wandb_run")
    mock_wandb_run.config = mock_wandb_config
    mock_wandb_run.id = "abcde"

    # Define datasets and model.
    seed = mock_wandb_config.get("seed")
    encoder = mock_wandb_config.get("encoder_name")
    batch_size = mock_wandb_config.get("batch_size")
    lr_initial = mock_wandb_config.get("lr_initial")
    weights_file = name_weights_file(encoder, batch_size, lr_initial)
    losscurve_csv = name_losscurve_csv_file(encoder, batch_size, lr_initial)
    training_dataset = SegmentationDataset(
        images_list=[],
        masks_list=[],
        s3_bucket="",
        transform=get_aug_pipeline(),
    )
    validation_dataset = SegmentationDataset(
        images_list=[],
        masks_list=[],
        s3_bucket="",
        transform=None,
    )
    model = Unet(
        encoder_name=mock_wandb_config.get("encoder_name"),
        encoder_weights=None,
        in_channels=4,
        activation="sigmoid",
    )
    mock_torch_DataParallel.return_value = model

    # Mock intermediate outputs.
    mock_train_loader = MagicMock(name="train_loader")
    mock_val_loader = MagicMock(name="val_loader")
    mock_create_data_loaders.return_value = (mock_train_loader, mock_val_loader)

    mock_optimizer = MagicMock(name="optimizer")
    mock_optimizer.param_groups = [{"lr": mock_wandb_config.get("lr_initial")}]
    mock_adamw_optimizer.return_value = mock_optimizer

    mock_train_epoch.return_value = 0.5
    mock_validate_epoch.return_value = {"val_loss": 0.4, "Accuracy": 0.6}

    # Call the test function.
    train_unet(mock_wandb_run, training_dataset, validation_dataset, model)

    # Assertions
    mock_torch_manual_seed.assert_called_once_with(seed)
    mock_torch_cuda_seed.assert_called_once_with(seed)
    mock_torch_cuda_all_seed.assert_called_once_with(seed)
    assert next(model.parameters()).device.type == device.type
    mock_torch_cuda_device_count.assert_called_once()
    mock_torch_DataParallel.assert_called_once_with(model)
    mock_create_data_loaders.assert_called_once_with(
        training_dataset,
        validation_dataset,
        batch_size=batch_size,
        seed=seed,
    )
    mock_adamw_optimizer.assert_called_once_with(model, lr_initial=lr_initial)
    mock_wandb_run.watch.assert_called_once_with(model)
    mock_train_epoch.assert_called_once_with(
        model, mock_train_loader, mock_optimizer, mock_dice_loss(), device
    )
    mock_validate_epoch.assert_called_once_with(
        model,
        mock_val_loader,
        mock_dice_loss(),
        device,
        mock_wandb_config.get("threshold"),
    )
    mock_wandb_run.log.assert_called_once_with(
        {
            "epoch": 1,
            "lr": mock_optimizer.param_groups[0]["lr"],
            "train_loss": mock_train_epoch.return_value,
            **mock_validate_epoch.return_value,
        }
    )
    mock_save_model_weights.assert_called_once_with(model, OUTPUT_PATH / weights_file)
    mock_write_loss_data.assert_called_once_with(
        [mock_train_epoch.return_value],
        [mock_validate_epoch.return_value["val_loss"]],
        OUTPUT_PATH / "train" / losscurve_csv,
    )
    mock_create_log_wandb_artifact(
        mock_wandb_run,
        str(OUTPUT_PATH / weights_file),
        mock_validate_epoch.return_value,
    )


@patch("amazon_seg_project.ops.train_unet.create_and_log_wandb_artifact")
@patch("logging.info")
@patch("amazon_seg_project.ops.train_unet.write_loss_data_to_csv")
@patch("amazon_seg_project.ops.train_unet.save_model_weights")
@patch("amazon_seg_project.ops.train_unet.validate_epoch")
@patch("amazon_seg_project.ops.train_unet.train_epoch")
@patch("amazon_seg_project.ops.train_unet.smp.losses.DiceLoss")
@patch("amazon_seg_project.ops.train_unet.setup_adam_w")
@patch("amazon_seg_project.ops.train_unet.create_data_loaders")
@patch("torch.cuda.device_count", return_value=1)
@patch("torch.cuda.manual_seed_all")
@patch("torch.cuda.manual_seed")
@patch("torch.manual_seed")
def test_early_stopping(
    mock_torch_manual_seed: MagicMock,
    mock_torch_cuda_seed: MagicMock,
    mock_torch_cuda_all_seed: MagicMock,
    mock_torch_cuda_device_count: MagicMock,
    mock_create_data_loaders: MagicMock,
    mock_adamw_optimizer: MagicMock,
    mock_dice_loss: MagicMock,
    mock_train_epoch: MagicMock,
    mock_validate_epoch: MagicMock,
    mock_save_model_weights: MagicMock,
    mock_write_loss_data: MagicMock,
    mock_logging: MagicMock,
    mock_create_log_wandb_artifact: MagicMock,
) -> None:
    """
    Test early stopping of U-net model training on a single GPU system
    """
    # Set up mock W&B run config.
    mock_wandb_config = {
        "seed": 47,
        "threshold": 0.39,
        "encoder_name": "resnet50",
        "batch_size": 4,
        "lr_initial": 0.001,
        "max_epochs": 8,
    }
    mock_wandb_run = MagicMock(name="mock_wandb_run")
    mock_wandb_run.config = mock_wandb_config

    # Define datasets and model.
    seed = mock_wandb_config.get("seed")
    encoder = mock_wandb_config.get("encoder_name")
    batch_size = mock_wandb_config.get("batch_size")
    lr_initial = mock_wandb_config.get("lr_initial")
    weights_file = name_weights_file(encoder, batch_size, lr_initial)
    losscurve_csv = name_losscurve_csv_file(encoder, batch_size, lr_initial)
    training_dataset = SegmentationDataset(
        images_list=[],
        masks_list=[],
        s3_bucket="",
        transform=get_aug_pipeline(),
    )
    validation_dataset = SegmentationDataset(
        images_list=[],
        masks_list=[],
        s3_bucket="",
        transform=None,
    )
    model = Unet(
        encoder_name=mock_wandb_config.get("encoder_name"),
        encoder_weights=None,
        in_channels=4,
        activation="sigmoid",
    )
    mock_train_loss = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.25, 0.25]
    mock_val_loss = [0.49, 0.48, 0.49, 0.49, 0.49, 0.49, 0.49, 0.48]
    mock_accuracy_values = [0.65, 0.67, 0.68, 0.70, 0.71, 0.72, 0.73, 0.74]

    # Mock intermediate outputs.
    mock_train_loader = MagicMock(name="train_loader")
    mock_val_loader = MagicMock(name="val_loader")
    mock_optimizer = MagicMock(name="optimizer")
    mock_create_data_loaders.return_value = (mock_train_loader, mock_val_loader)
    mock_adamw_optimizer.return_value = mock_optimizer
    mock_train_epoch.side_effect = mock_train_loss

    # Set up iterators over mock_val_loss and mock_metric_values.
    val_loss_iter = iter(mock_val_loss)
    accuracy_iter = iter(mock_accuracy_values)

    # Define a side_effect function for mocking
    def side_effect(*args, **kwargs) -> Dict[str, float | None]:
        """
        Custom side_effect function for mocking.

        It iterates over "mock_val_loss" and "metric_value_iter".
        """
        try:
            return {
                "val_loss": next(val_loss_iter),
                "Accuracy": next(accuracy_iter),
            }
        except StopIteration:
            # Handle case when either mock_val_loss or mock_accuracy_values is exhausted.
            return {"val_loss": None, "Accuracy": None}

    mock_validate_epoch.side_effect = side_effect

    # Call the test function.
    train_unet(mock_wandb_run, training_dataset, validation_dataset, model)

    # Assertions
    mock_torch_manual_seed.assert_called_once_with(seed)
    mock_torch_cuda_seed.assert_called_once_with(seed)
    mock_torch_cuda_all_seed.assert_called_once_with(seed)
    assert next(model.parameters()).device.type == device.type
    mock_torch_cuda_device_count.assert_called_once()
    mock_create_data_loaders.assert_called_once_with(
        training_dataset,
        validation_dataset,
        batch_size=batch_size,
        seed=seed,
    )
    mock_adamw_optimizer.assert_called_once_with(model, lr_initial=lr_initial)
    mock_wandb_run.watch.assert_called_once_with(model)
    mock_train_epoch.assert_called_with(
        model, mock_train_loader, mock_optimizer, mock_dice_loss(), device
    )
    mock_validate_epoch.assert_called_with(
        model,
        mock_val_loader,
        mock_dice_loss(),
        device,
        mock_wandb_config.get("threshold"),
    )
    # Logging assertions for epochs 1 – 7
    expected_calls: List[Any] = []
    for i in range(len(mock_train_loss) - 1):
        expected_calls.append(
            call(
                {
                    "epoch": i + 1,
                    "lr": mock_optimizer.param_groups[0]["lr"],
                    "train_loss": mock_train_loss[i],
                    "val_loss": mock_val_loss[i],
                    "Accuracy": mock_accuracy_values[i],
                }
            )
        )
    mock_wandb_run.log.assert_has_calls(expected_calls)
    # Assertions for save/write operations
    mock_save_model_weights.assert_called_with(model, OUTPUT_PATH / weights_file)
    mock_logging.assert_called_once_with("Early stopping criterion triggered.")
    mock_write_loss_data.assert_called_once_with(
        mock_train_loss[:-1],
        mock_val_loss[:-1],
        OUTPUT_PATH / "train" / losscurve_csv,
    )
    mock_create_log_wandb_artifact(
        mock_wandb_run,
        str(OUTPUT_PATH / weights_file),
        mock_validate_epoch.return_value,
    )
