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
@patch("torch.cuda.manual_seed_all")
@patch("torch.cuda.manual_seed")
@patch("torch.manual_seed")
@patch("wandb.init")
def test_single_epoch_training(
    mock_wandb_init: MagicMock,
    mock_torch_manual_seed: MagicMock,
    mock_torch_cuda_seed: MagicMock,
    mock_torch_cuda_all_seed: MagicMock,
    mock_create_data_loaders: MagicMock,
    mock_adamw_optimizer: MagicMock,
    mock_dice_loss: MagicMock,
    mock_train_epoch: MagicMock,
    mock_validate_epoch: MagicMock,
    mock_save_model_weights: MagicMock,
    mock_write_loss_data: MagicMock,
) -> None:
    """
    Check for correct execution of a single epoch of U-net training
    """
    # Setup
    wandb_config = {
        "seed": 42,
        "encoder_name": "resnet50",
        "batch_size": 4,
        "lr_initial": 0.001,
        "max_epochs": 1,
    }
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
        encoder_name=wandb_config["encoder_name"],
        encoder_weights=None,
        in_channels=4,
        activation="sigmoid",
    )
    encoder = wandb_config["encoder_name"]
    batch_size = wandb_config["batch_size"]
    lr_initial = wandb_config["lr_initial"]
    weights_file = f"{encoder}_batch{batch_size}_lr{lr_initial}_weights.pt"
    losscurve_csv = f"{encoder}_batch{batch_size}_lr{lr_initial}_losscurve.csv"

    # Mock intermediate outputs.
    mock_run = MagicMock(name="run")
    mock_run.config = wandb_config
    mock_wandb_init.return_value.__enter__.return_value = mock_run

    mock_train_loader = MagicMock(name="train_loader")
    mock_val_loader = MagicMock(name="val_loader")
    mock_create_data_loaders.return_value = (mock_train_loader, mock_val_loader)

    mock_optimizer = MagicMock(name="optimizer")
    mock_adamw_optimizer.return_value = mock_optimizer

    mock_train_epoch.return_value = 0.5
    mock_validate_epoch.return_value = 0.4

    # Call the test function.
    train_unet(
        "mock_project", wandb_config, training_dataset, validation_dataset, model
    )

    # Assertions
    mock_wandb_init.assert_called_once_with(project="mock_project", config=wandb_config)
    mock_torch_manual_seed.assert_called_once_with(wandb_config["seed"])
    mock_torch_cuda_seed.assert_called_once_with(wandb_config["seed"])
    mock_torch_cuda_all_seed.assert_called_once_with(wandb_config["seed"])
    assert next(model.parameters()).device.type == device.type
    mock_create_data_loaders.assert_called_once_with(
        training_dataset,
        validation_dataset,
        batch_size=batch_size,
        seed=wandb_config["seed"],
    )
    mock_adamw_optimizer.assert_called_once_with(model, lr_initial=lr_initial)
    mock_run.watch.assert_called_once_with(model)
    mock_train_epoch.assert_called_once_with(
        model, mock_train_loader, mock_optimizer, mock_dice_loss, device
    )
    mock_validate_epoch.assert_called_once_with(
        model, mock_val_loader, mock_dice_loss, device
    )
    mock_run.log.assert_called_once_with(
        {
            "epoch": 1,
            "train_loss": mock_train_epoch.return_value,
            "val_loss": mock_validate_epoch.return_value,
        }
    )
    mock_save_model_weights.assert_called_once_with(model, OUTPUT_PATH / weights_file)
    mock_write_loss_data.assert_called_once_with(
        [mock_train_epoch.return_value],
        [mock_validate_epoch.return_value],
        OUTPUT_PATH / "train" / losscurve_csv,
    )


@patch("amazon_seg_project.ops.wandb_utils.write_loss_data_to_csv")
@patch("amazon_seg_project.ops.wandb_utils.save_model_weights")
@patch("amazon_seg_project.ops.wandb_utils.validate_epoch")
@patch("amazon_seg_project.ops.wandb_utils.train_epoch")
@patch("amazon_seg_project.ops.wandb_utils.dice_loss")
@patch("amazon_seg_project.ops.wandb_utils.setup_adam_w")
@patch("amazon_seg_project.ops.wandb_utils.create_data_loaders")
@patch("torch.cuda.manual_seed_all")
@patch("torch.cuda.manual_seed")
@patch("torch.manual_seed")
@patch("wandb.init")
def test_early_stopping(
    mock_wandb_init: MagicMock,
    mock_torch_manual_seed: MagicMock,
    mock_torch_cuda_seed: MagicMock,
    mock_torch_cuda_all_seed: MagicMock,
    mock_create_data_loaders: MagicMock,
    mock_adamw_optimizer: MagicMock,
    mock_dice_loss: MagicMock,
    mock_train_epoch: MagicMock,
    mock_validate_epoch: MagicMock,
    mock_save_model_weights: MagicMock,
    mock_write_loss_data: MagicMock,
) -> None:
    """
    Test early stopping of U-net model training
    """
    # Setup
    wandb_config = {
        "seed": 47,
        "encoder_name": "resnet50",
        "batch_size": 4,
        "lr_initial": 0.001,
        "max_epochs": 8,
    }
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
        encoder_name=wandb_config["encoder_name"],
        encoder_weights=None,
        in_channels=4,
        activation="sigmoid",
    )
    mock_train_loss = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.25, 0.25]
    mock_val_loss = [0.49, 0.48, 0.49, 0.49, 0.49, 0.49, 0.49, 0.48]

    encoder = wandb_config["encoder_name"]
    batch_size = wandb_config["batch_size"]
    lr_initial = wandb_config["lr_initial"]
    weights_file = f"{encoder}_batch{batch_size}_lr{lr_initial}_weights.pt"
    losscurve_csv = f"{encoder}_batch{batch_size}_lr{lr_initial}_losscurve.csv"

    # Mock intermediate outputs.
    mock_run = MagicMock(name="run")
    mock_run.config = wandb_config
    mock_wandb_init.return_value.__enter__.return_value = mock_run
    mock_train_loader = MagicMock(name="train_loader")
    mock_val_loader = MagicMock(name="val_loader")
    mock_optimizer = MagicMock(name="optimizer")
    mock_create_data_loaders.return_value = (mock_train_loader, mock_val_loader)
    mock_adamw_optimizer.return_value = mock_optimizer
    mock_train_epoch.side_effect = mock_train_loss
    mock_validate_epoch.side_effect = mock_val_loss

    # Call the test function.
    train_unet(
        "mock_project", wandb_config, training_dataset, validation_dataset, model
    )

    # Assertions
    mock_wandb_init.assert_called_once_with(project="mock_project", config=wandb_config)
    mock_torch_manual_seed.assert_called_once_with(wandb_config["seed"])
    mock_torch_cuda_seed.assert_called_once_with(wandb_config["seed"])
    mock_torch_cuda_all_seed.assert_called_once_with(wandb_config["seed"])
    assert next(model.parameters()).device.type == device.type
    mock_create_data_loaders.assert_called_once_with(
        training_dataset,
        validation_dataset,
        batch_size=batch_size,
        seed=wandb_config["seed"],
    )
    mock_adamw_optimizer.assert_called_once_with(model, lr_initial=lr_initial)
    mock_run.watch.assert_called_once_with(model)
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
                }
            )
        )
    mock_run.log.assert_has_calls(expected_calls)
    # Assertions for save/write operations
    mock_save_model_weights.assert_called_with(model, OUTPUT_PATH / weights_file)
    mock_write_loss_data.assert_called_once_with(
        mock_train_loss[:-1],
        mock_val_loss[:-1],
        OUTPUT_PATH / "train" / losscurve_csv,
    )
