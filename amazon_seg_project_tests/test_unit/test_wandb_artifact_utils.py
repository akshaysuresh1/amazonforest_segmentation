"""
Unit tests for ops defined in wandb_artifact_utils.py
"""

from unittest.mock import patch, MagicMock
from amazon_seg_project.ops.wandb_artifact_utils import (
    create_and_log_wandb_artifact,
    promote_best_model_to_registry,
)


@patch("wandb.Artifact")
def test_create_log_artifact_success(mock_wandb_artifact: MagicMock) -> None:
    """
    Verify successful execution of create_and_log_wandb_artifact()
    """
    # Set up mock W&B run.
    mock_wandb_run = MagicMock(name="mock-wandb-run")
    mock_wandb_run.config = {
        "encoder_name": "resnet50",
        "lr_initial": 1.0e-3,
        "batch_size": 16,
    }
    mock_wandb_run.id = "abcde"

    # Set return value for mock_wandb_artifact.
    mock_artifact = MagicMock(name="mock-artifact")
    mock_wandb_artifact.return_value = mock_artifact

    # Define mock validation stats.
    mock_validation_stats = {"val_loss": 0.43, "Accuracy": 0.78}
    mock_weights_file = "mock_weights_file.pt"

    # Call the test function.
    create_and_log_wandb_artifact(
        mock_wandb_run, mock_weights_file, mock_validation_stats
    )

    # Assertions
    mock_wandb_artifact.assert_called_once_with(
        name=f"unet_with_{mock_wandb_run.config.get('encoder_name')}",
        type="model",
        metadata={
            "run_id": mock_wandb_run.id,
            "encoder": mock_wandb_run.config.get("encoder_name"),
            "lr_initial": mock_wandb_run.config.get("lr_initial"),
            "batch_size": mock_wandb_run.config.get("batch_size"),
            **mock_validation_stats,
        },
    )
    mock_artifact.add_file.assert_called_once_with(mock_weights_file)
    mock_wandb_run.log_artifact.assert_called_once_with(mock_artifact)


@patch("logging.error")
@patch("wandb.Api")
def test_promote_best_model_to_registry_except_handling(
    mock_wandb_api: MagicMock, mock_logging_error: MagicMock
) -> None:
    """
    Test for function response to failed W&B API call.
    """
    entity = "test-entity"
    project = "test-project"
    sweep_id = "mock-sweep-id"

    # Create mock API.
    api = MagicMock(name="api")
    mock_wandb_api.return_value = api

    # Simulate an exception when calling api.sweep
    api.sweep.side_effect = Exception("Test exception")

    # Call the test function.
    promote_best_model_to_registry(entity, project, sweep_id)

    # Assertions
    mock_wandb_api.assert_called_once_with(
        overrides={"entity": entity, "project": project}
    )
    api.sweep.assert_called_once_with(sweep_id)
    mock_logging_error.assert_called_with("An error occurred: %s", "Test exception")


@patch("wandb.init")
@patch("wandb.Api")
def test_promote_best_model_to_registry_success(
    mock_wandb_api: MagicMock, mock_wandb_init: MagicMock
) -> None:
    """
    Check for successful execution of promote_best_model_to_registry().
    """
    entity = "test-entity"
    project = "test-project"
    sweep_id = "mock-sweep-id"

    # Create mock API.
    api = MagicMock(name="api")
    mock_wandb_api.return_value = api

    # Set up mock for api.sweeps().
    sweep = MagicMock(name="wandb-sweep")
    api.sweep.return_value = sweep

    # Create mock runs.
    run1 = MagicMock(name="run 1")
    run1.summary = {"val_loss": 0.1}
    run1.id = "run1_id"

    run2 = MagicMock()
    run2.summary = {"val_loss": 0.05}
    run2.id = "run2_id"
    sweep.runs = [run1, run2]

    # Set up mock artifact for the best run, i.e., run 2.
    run2_artifact = MagicMock(name="run 2 artifact")
    run2_artifact.aliases = ["test"]
    run2.logged_artifacts.return_value = [run2_artifact]

    # Define mock run for linking run2_artifact to W&B registry.
    mock_run = MagicMock(name="mock-run")
    mock_wandb_init.return_value.__enter__.return_value = mock_run

    # Call the test function.
    promote_best_model_to_registry(entity, project, sweep_id)

    # Assertions
    mock_wandb_api.assert_called_once_with(
        overrides={"entity": entity, "project": project}
    )
    api.sweep.assert_called_once_with(sweep_id)
    run2.logged_artifacts.assert_called_once()
    assert run2_artifact.aliases[-1] == "best_model"
    run2_artifact.save.assert_called_once()
    mock_wandb_init.assert_called_once_with(
        entity=entity, project=project, job_type="artifact-upload"
    )
    mock_run.link_artifact.assert_called_once_with(
        artifact=run2_artifact,
        target_path=f"{entity}/model-registry/unet-models",
    )


@patch("logging.info")
@patch("wandb.Api")
def test_zero_runs_in_sweep(mock_wandb_api: MagicMock, mock_logging: MagicMock) -> None:
    """
    Test for module response to a W&B sweep that initiates zero runs.
    """
    entity = "test-entity"
    project = "test-project"
    sweep_id = "mock-sweep-id"

    # Create mock API.
    api = MagicMock(name="api")
    mock_wandb_api.return_value = api

    # Set up mock for api.sweeps().
    sweep = MagicMock(name="wandb-sweep")
    api.sweep.return_value = sweep
    sweep.runs = []  # No runs in sweep

    # Call the test function.
    promote_best_model_to_registry(entity, project, sweep_id)

    # Assertions
    mock_wandb_api.assert_called_once_with(
        overrides={"entity": entity, "project": project}
    )
    api.sweep.assert_called_once_with(sweep_id)
    mock_logging.assert_called_once_with(f"No runs found for sweep {sweep_id}.")


@patch("logging.info")
@patch("wandb.Api")
def test_no_unique_model_weights_file_for_best_run(
    mock_wandb_api: MagicMock, mock_logging: MagicMock
) -> None:
    """
    Test for correct raise of ValueError if missing artifact from best run
    """
    entity = "test-entity"
    project = "test-project"
    sweep_id = "mock-sweep-id"

    # Create mock API.
    api = MagicMock(name="api")
    mock_wandb_api.return_value = api

    # Set up mock for api.sweeps().
    sweep = MagicMock(name="wandb-sweep")
    api.sweep.return_value = sweep

    # Create mock run.
    mock_run = MagicMock(name="mock-run")
    mock_run.id = "mock-run-id"
    mock_run.summary = {"val_loss": 0.1}
    mock_run.logged_artifacts.return_value = []  # No artifacts logged
    sweep.runs = [mock_run]

    # Call the test function.
    promote_best_model_to_registry(entity, project, sweep_id)

    # Assertions
    mock_wandb_api.assert_called_once_with(
        overrides={"entity": entity, "project": project}
    )
    api.sweep.assert_called_once_with(sweep_id)
    mock_run.logged_artifacts.assert_called_once()
    mock_logging.assert_called_once_with(
        "There exists no unique model weights file associated with run %s.", mock_run.id
    )
