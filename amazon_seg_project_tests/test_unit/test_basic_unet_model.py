"""
Unit test for asset "basic_unet_model"
"""

from unittest.mock import patch, MagicMock
from segmentation_models_pytorch import Unet
from amazon_seg_project.assets import unet_model
from amazon_seg_project.config import BasicUnetConfig
from amazon_seg_project.resources import device


@patch("amazon_seg_project.assets.model.Unet")
def test_model_creation(mock_smp_unet: MagicMock) -> None:
    """
    Test if the model is created correctly and moved to the correct device.
    """
    # Define config for mock model.
    config = BasicUnetConfig(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        activation="sigmoid",
    )
    mock_model = MagicMock()
    mock_model.encoder = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_smp_unet.return_value = mock_model

    # Call the test function.
    generator = unet_model(config)
    # Ensure the function is fully executed.
    next(generator)  # type: ignore

    print(mock_smp_unet.call_args_list)

    # Assertion for model creation
    mock_smp_unet.assert_called_once_with(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=config.in_channels,
        activation=config.activation,
    )

    # Check if the model is moved to the correct device.
    mock_model.to.assert_called_once_with(device)

    # Assert that the loop to freeze encoder weights is initiated.
    mock_model.encoder.parameters.assert_called_once()


def test_model_parameter_settings() -> None:
    """
    Test to verify reported parameter counts in metadata.
    Also, check that the encoder weights are frozen.
    """
    # Set up config for model.
    config = BasicUnetConfig(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        activation="sigmoid",
    )

    # Call the test function.
    output_generator = unet_model(config)
    # Exhaust the generator into a list.
    output = list(output_generator)  # type: ignore

    # Assertions
    assert len(output) == 1
    assert isinstance(output[0].value, Unet)

    model = output[0].value
    # Check if the encoder weights are frozen.
    for param in model.encoder.parameters():
        assert not param.requires_grad

    # Check that the output metadata report expected parameter counts.
    total_param_count = sum(p.numel() for p in model.parameters())
    encoder_param_count = sum(p.numel() for p in model.encoder.parameters())
    trainable_param_count = total_param_count - encoder_param_count

    assert output[0].metadata["Total parameter count"].value == total_param_count
    assert (
        output[0].metadata["Trainable parameter count"].value == trainable_param_count
    )
