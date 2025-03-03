"""
Unit tests for asset "finetuned_unet" 
"""

from unittest.mock import patch, MagicMock, call
import segmentation_models_pytorch as smp
from amazon_seg_project.assets import SegmentationDataset, finetuned_unet_model
from amazon_seg_project.data_paths import OUTPUT_PATH
from amazon_seg_project.config import FinetunedUnetConfig
from amazon_seg_project.resources import device


@patch("amazon_seg_project.assets.model.create_data_loaders")
@patch("amazon_seg_project.assets.model.setup_adam_w")
@patch("amazon_seg_project.assets.model.dice_loss")
@patch("amazon_seg_project.assets.model.train_epoch")
@patch("amazon_seg_project.assets.model.validate_epoch")
@patch("amazon_seg_project.assets.model.save_model_weights")
@patch("amazon_seg_project.assets.model.write_loss_data_to_csv")
@patch("logging.info")
def test_finetuned_unet_single_epoch_run(
    mock_logging: MagicMock,
    mock_write_loss_data: MagicMock,
    mock_save_model_weights: MagicMock,
    mock_validate_epoch: MagicMock,
    mock_train_epoch: MagicMock,
    mock_dice_loss: MagicMock,
    mock_adamw_optimizer: MagicMock,
    mock_setup_data_loaders: MagicMock,
) -> None:
    """
    Test for successful run of finetuned_unet_model() for one epoch
    """
    # Set up config, datasets, and pretrained model.
    config = FinetunedUnetConfig(max_epochs=1, batch_size=4, lr_initial=0.001)
    training_dataset = SegmentationDataset(
        images_list=[], masks_list=[], s3_bucket="", do_aug=True
    )
    validation_dataset = SegmentationDataset(
        images_list=[], masks_list=[], s3_bucket="", do_aug=False
    )
    pretrained_unet = smp.Unet(
        encoder="resnet50",
        encoder_weights=None,
        in_channels=4,
        activation="sigmoid",
    )
    model = pretrained_unet.to(device)

    # Mock intermediate outputs.
    mock_train_loader = MagicMock(name="train_loader")
    mock_val_loader = MagicMock(name="val_loader")
    mock_optimizer = MagicMock(name="optimizer")
    mock_setup_data_loaders.return_value = (mock_train_loader, mock_val_loader)
    mock_adamw_optimizer.return_value = mock_optimizer
    mock_train_epoch.return_value = 0.5
    mock_validate_epoch.return_value = 0.4

    # Call the test function.
    output_model = finetuned_unet_model(
        config, pretrained_unet, training_dataset, validation_dataset
    )

    # Assertions
    assert isinstance(output_model, smp.Unet)
    assert next(output_model.parameters()).device == device
    mock_setup_data_loaders.assert_called_once_with(
        training_dataset, validation_dataset, batch_size=config.batch_size
    )
    mock_adamw_optimizer.assert_called_once_with(model, lr_initial=config.lr_initial)
    mock_train_epoch.assert_called_once_with(
        model, mock_train_loader, mock_optimizer, mock_dice_loss, device
    )
    mock_validate_epoch.assert_called_once_with(
        model, mock_val_loader, mock_dice_loss, device
    )
    mock_save_model_weights.assert_called_once_with(
        model, OUTPUT_PATH / "model_weights.pt"
    )
    mock_write_loss_data.assert_called_once_with(
        [mock_train_epoch.return_value],
        [mock_validate_epoch.return_value],
        OUTPUT_PATH / "train" / "loss_curve.csv",
    )

    # Logging assertions
    expected_calls = [
        call(
            "Starting training:\n"
            "Max no. of epochs = %d\n"
            "Batch size = %d\n"
            "Initial learning rate = %.1g\n"
            "Training dataset size = %d\n"
            "Validation dataset size = %d\n"
            "Device = %s",
            config.max_epochs,
            config.batch_size,
            config.lr_initial,
            len(training_dataset),
            len(validation_dataset),
            device.type,
        ),
        call(
            "Statistics for epoch %d: \n Training loss: %.4f \n Validation loss: %.4f",
            1,
            mock_train_epoch.return_value,
            mock_validate_epoch.return_value,
        ),
        call("Achieved new minimum validation loss. Writing model weights to disk."),
    ]
    mock_logging.assert_has_calls(expected_calls)
