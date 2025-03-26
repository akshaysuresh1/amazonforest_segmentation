"""
Unit test for op "train_unet"
"""

from typing import List, Any
from unittest.mock import patch, MagicMock, call
from segmentation_models_pytorch import Unet
from amazon_seg_project.assets import SegmentationDataset
from amazon_seg_project.data_paths import OUTPUT_PATH
from amazon_seg_project.ops.aug_utils import get_aug_pipeline
from amazon_seg_project.ops.wandb_utils import train_unet
from amazon_seg_project.resources import device


@patch("amazon_seg_project.ops.wandb_utils.write_loss_data_to_csv")
@patch("amazon_seg_project.ops.wandb_utils.save_model_weights")
@patch("amazon_seg_project.ops.wandb_utils.validate_epoch")
@patch("amazon_seg_project.ops.wandb_utils.train_epoch")
@patch("amazon_seg_project.ops.wandb_utils.dice_loss")
@patch("amazon_seg_project.ops.wandb_utils.setup_adam_w")
@patch("amazon_seg_project.ops.wandb_utils.create_data_loaders")
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
) -> None:
    """
    Check for correct execution of a single epoch of U-net training.
    This test assumes a multi-GPU system.
    """
    # Set up mock W&B run config.
    mock_wandb_config = {
        "seed": 42,
        "encoder_name": "resnet50",
        "batch_size": 4,
        "lr_initial": 0.001,
        "max_epochs": 1,
    }
    mock_wandb_run = MagicMock(name="mock_wandb_run")
    mock_wandb_run.config = mock_wandb_config

    # Define datasets and model.
    seed = mock_wandb_config.get("seed")
    encoder = mock_wandb_config.get("encoder_name")
    batch_size = mock_wandb_config.get("batch_size")
    lr_initial = mock_wandb_config.get("lr_initial")
    weights_file = f"{encoder}_batch{batch_size}_lr{lr_initial:.1e}_weights.pt"
    losscurve_csv = f"{encoder}_batch{batch_size}_lr{lr_initial:.1e}_losscurve.csv"
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
    mock_validate_epoch.return_value = 0.4

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
        model, mock_train_loader, mock_optimizer, mock_dice_loss, device
    )
    mock_validate_epoch.assert_called_once_with(
        model, mock_val_loader, mock_dice_loss, device
    )
    mock_wandb_run.log.assert_called_once_with(
        {
            "epoch": 1,
            "train_loss": mock_train_epoch.return_value,
            "val_loss": mock_validate_epoch.return_value,
            "lr": mock_optimizer.param_groups[0]["lr"],
        }
    )
    mock_save_model_weights.assert_called_once_with(model, OUTPUT_PATH / weights_file)
    mock_write_loss_data.assert_called_once_with(
        [mock_train_epoch.return_value],
        [mock_validate_epoch.return_value],
        OUTPUT_PATH / "train" / losscurve_csv,
    )


@patch("logging.info")
@patch("amazon_seg_project.ops.wandb_utils.write_loss_data_to_csv")
@patch("amazon_seg_project.ops.wandb_utils.save_model_weights")
@patch("amazon_seg_project.ops.wandb_utils.validate_epoch")
@patch("amazon_seg_project.ops.wandb_utils.train_epoch")
@patch("amazon_seg_project.ops.wandb_utils.dice_loss")
@patch("amazon_seg_project.ops.wandb_utils.setup_adam_w")
@patch("amazon_seg_project.ops.wandb_utils.create_data_loaders")
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
) -> None:
    """
    Test early stopping of U-net model training on a single GPU system
    """
    # Set up mock W&B run config.
    mock_wandb_config = {
        "seed": 47,
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
    weights_file = f"{encoder}_batch{batch_size}_lr{lr_initial:.1e}_weights.pt"
    losscurve_csv = f"{encoder}_batch{batch_size}_lr{lr_initial:.1e}_losscurve.csv"
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

    # Mock intermediate outputs.
    mock_train_loader = MagicMock(name="train_loader")
    mock_val_loader = MagicMock(name="val_loader")
    mock_optimizer = MagicMock(name="optimizer")
    mock_create_data_loaders.return_value = (mock_train_loader, mock_val_loader)
    mock_adamw_optimizer.return_value = mock_optimizer
    mock_train_epoch.side_effect = mock_train_loss
    mock_validate_epoch.side_effect = mock_val_loss

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
        model, mock_train_loader, mock_optimizer, mock_dice_loss, device
    )
    mock_validate_epoch.assert_called_with(
        model, mock_val_loader, mock_dice_loss, device
    )
    # Logging assertions for epochs 1 – 7
    expected_calls: List[Any] = []
    for i in range(len(mock_train_loss) - 1):
        expected_calls.append(
            call(
                {
                    "epoch": i + 1,
                    "train_loss": mock_train_loss[i],
                    "val_loss": mock_val_loss[i],
                     "lr": mock_optimizer.param_groups[0]["lr"],
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
