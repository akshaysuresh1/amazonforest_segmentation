"""
Unit tests for asset "trained_unet_model"
"""

import os
from unittest.mock import patch, MagicMock
import pytest
from amazon_seg_project.assets import trained_unet_model
from amazon_seg_project.config import TrainedUnetConfig
from amazon_seg_project.resources import device


@patch("amazon_seg_project.assets.model.check_artifact_exists")
def test_trained_unet_wandb_artifact_not_found(mock_check_artifact: MagicMock) -> None:
    """
    Check for correct raise of ValueError in case of missing W&B artifact.
    """
    with pytest.raises(
        ValueError,
        match="Model weights file not available for loading into U-net model.",
    ):
        mock_check_artifact.return_value = False

        # Create mock inputs.
        mock_wandb_artifact_path = "entity/project/models"
        mock_wandb_artifact_version = "latest"
        test_config = TrainedUnetConfig(
            wandb_artifact_path=mock_wandb_artifact_path,
            wandb_artifact_version=mock_wandb_artifact_version,
        )

        # Call the test function.
        output_generator = trained_unet_model(test_config)
        # Exhaust the generator into a list.
        _ = list(output_generator)  # type: ignore

        # Assertions
        mock_check_artifact.assert_called_once_with(
            mock_wandb_artifact_path, mock_wandb_artifact_version
        )


@patch("torch.load")
@patch("amazon_seg_project.assets.model.Unet")
@patch("amazon_seg_project.assets.model.name_weights_file")
@patch("wandb.Api")
@patch("amazon_seg_project.assets.model.check_artifact_exists")
def test_trained_unet_model_success(
    mock_check_artifact: MagicMock,
    mock_wandb_api_function: MagicMock,
    mock_name_weights_file: MagicMock,
    mock_unet_function: MagicMock,
    mock_torch_load: MagicMock,
) -> None:
    """
    Test for successful materialization of asset "trained_unet_model()".
    """
    # Create mock inputs.
    mock_wandb_artifact_path = "entity/project/models"
    mock_wandb_artifact_version = "latest"
    test_config = TrainedUnetConfig(
        wandb_artifact_path=mock_wandb_artifact_path,
        wandb_artifact_version=mock_wandb_artifact_version,
    )

    # Set up mock dependencies.
    mock_check_artifact.return_value = True
    mock_name_weights_file.return_value = "mock_weights_file.pt"

    mock_api = MagicMock(name="mock-wandb-api")
    mock_wandb_api_function.return_value = mock_api

    # Create mock artifact.
    mock_artifact = MagicMock(name="mock-wandb-artifact")
    mock_artifact.metadata = {
        "encoder": "resnet34",
        "batch_size": 16,
        "lr_initial": 0.001,
    }
    mock_api.artifact.return_value = mock_artifact
    mock_artifact.download.return_value = "/home/entity/project"

    # Define mock model with 6 parameters across two layers.
    mock_params = [
        MagicMock(name="param_layer1", numel=MagicMock(side_effect=lambda: 2)),
        MagicMock(name="param_layer2", numel=MagicMock(side_effect=lambda: 4)),
    ]
    mock_model = MagicMock(
        name="mock-model", parameters=MagicMock(return_value=mock_params)
    )
    mock_unet_function.return_value = mock_model

    # Call the test function.
    output_generator = trained_unet_model(test_config)
    # Exhaust the generator into a list.
    output = list(output_generator)  # type: ignore

    # Assertions for outputs returned
    assert len(output) == 1
    model = output[0].value
    assert model == mock_model
    assert output[0].metadata.get("Total parameter count").value == 6

    # Assertions for mocked internal dependencies
    mock_check_artifact.assert_called_once_with(
        mock_wandb_artifact_path, mock_wandb_artifact_version
    )
    mock_wandb_api_function.assert_called_once()
    mock_api.artifact.assert_called_once_with(
        f"{mock_wandb_artifact_path}:{mock_wandb_artifact_version}"
    )
    mock_name_weights_file.assert_called_once_with(
        mock_artifact.metadata.get("encoder"),
        mock_artifact.metadata.get("batch_size"),
        mock_artifact.metadata.get("lr_initial"),
    )
    mock_artifact.download.assert_called_once_with(
        path_prefix=mock_name_weights_file.return_value
    )
    mock_unet_function.assert_called_once_with(
        encoder_name=mock_artifact.metadata.get("encoder"),
        encoder_weights=None,
        in_channels=test_config.in_channels,
        activation=test_config.activation,
    )
    mock_torch_load.assert_called_once_with(
        os.path.join(
            mock_artifact.download.return_value, mock_name_weights_file.return_value
        ),
        map_location=device,
    )
    mock_model.load_state_dict.assert_called_once_with(mock_torch_load.return_value)
