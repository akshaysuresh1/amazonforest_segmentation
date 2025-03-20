"""
Unit tests for unet_model()
"""

from unittest.mock import patch, MagicMock
from segmentation_models_pytorch import Unet
from amazon_seg_project.assets import unet_model
from amazon_seg_project.config import BasicUnetConfig


@patch("torch.manual_seed")
@patch("amazon_seg_project.assets.model.Unet")
def test_model_creation(mock_smp_unet: MagicMock, mock_torch_seed: MagicMock) -> None:
    """
    Test if the model is created correctly.
    """
    # Define config for mock model.
    model_config = BasicUnetConfig(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        activation="sigmoid",
        model_seed=28,
    )
    mock_model = MagicMock()
    mock_model.encoder = MagicMock()
    mock_smp_unet.return_value = mock_model

    # Call the test function.
    output_generator = unet_model(model_config)
    # Exhaust the generator into a list.
    output = list(output_generator)  # type: ignore

    # Initial assertions
    assert len(output) == 1
    model = output[0].value
    assert model == mock_model
    # Assert for seed initialization
    mock_torch_seed.assert_called_once_with(model_config.model_seed)
    # Assertion for model creation
    mock_smp_unet.assert_called_once_with(
        encoder_name=model_config.encoder_name,
        encoder_weights=model_config.encoder_weights,
        in_channels=model_config.in_channels,
        activation=model_config.activation,
    )
    # Assert that the loop to freeze encoder weights was initiated.
    mock_model.encoder.parameters.assert_called_once()


def test_model_parameter_settings() -> None:
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
        model_seed=28,
    )

    # Call the test function.
    output_generator = unet_model(model_config)
    # Exhaust the generator into a list.
    output = list(output_generator)  # type: ignore

    # Assertions
    model = output[0].value
    assert isinstance(model, Unet)

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
