"""
Unit tests for unet_model()
"""

from unittest.mock import patch, MagicMock, call
from segmentation_models_pytorch import Unet
from amazon_seg_project.assets import unet_model
from amazon_seg_project.config import BasicUnetConfig


@patch("amazon_seg_project.assets.model.Unet")
def test_model_creation(mock_smp_unet: MagicMock) -> None:
    """
    Test if the model is created correctly.
    """
    # Define config for mock model.
    model_config = BasicUnetConfig(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        activation="sigmoid",
    )
    mock_model = MagicMock()
    mock_model.encoder = MagicMock()
    mock_smp_unet.return_value = mock_model

    # Call the test function.
    model = unet_model(model_config)

    # Assertion for model creation
    assert model == mock_model
    mock_smp_unet.assert_called_once_with(
        encoder_name=model_config.encoder_name,
        encoder_weights=model_config.encoder_weights,
        in_channels=model_config.in_channels,
        activation=model_config.activation,
    )

    # Assert that the loop to freeze encoder weights is initiated.
    mock_model.encoder.parameters.assert_called_once()


@patch("logging.info")
def test_model_parameter_settings(mock_logging: MagicMock) -> None:
    """
    Test to verify reported parameter counts in metadata.
    Also, check that the encoder weights are frozen.
    """
    # Set up config for model.
    model_config = BasicUnetConfig(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        activation="sigmoid",
    )

    # Call the test function.
    model = unet_model(model_config)

    # Assertions
    assert isinstance(model, Unet)

    # Check if the encoder weights are frozen.
    for param in model.encoder.parameters():
        assert not param.requires_grad

    # Check that the output logs report expected parameter counts.
    total_param_count = sum(p.numel() for p in model.parameters())
    encoder_param_count = sum(p.numel() for p in model.encoder.parameters())
    trainable_param_count = total_param_count - encoder_param_count

    expected_calls = [
        call("Total parameter count = %d", total_param_count),
        call("Trainable parameter count = %d", trainable_param_count),
    ]

    mock_logging.assert_has_calls(expected_calls)
