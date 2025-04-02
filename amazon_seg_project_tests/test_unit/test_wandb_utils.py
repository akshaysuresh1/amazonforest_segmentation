"""
Unit tests for short ops defined in wandb_utils.py
"""

from unittest.mock import patch, MagicMock
from amazon_seg_project.config import SweepConfig
from amazon_seg_project.ops.wandb_utils import (
    make_sweep_config,
    run_sweep,
    run_wandb_exp,
    upload_best_model_to_wandb,
)
from amazon_seg_project.data_paths import OUTPUT_PATH


def test_make_sweep_config_default_values() -> None:
    """
    Test successful creation of sweep config using default values.
    """
    test_config = SweepConfig()

    # Call the test function.
    output_sweep_config = make_sweep_config(test_config)

    # Assertions
    assert output_sweep_config.get("method") == test_config.method
    assert output_sweep_config.get("metric").get("name") == test_config.metric_name
    assert output_sweep_config.get("metric").get("goal") == test_config.metric_goal
    assert output_sweep_config.get("parameters").get("seed") == test_config.seed
    assert (
        output_sweep_config.get("parameters").get("threshold") == test_config.threshold
    )
    assert (
        output_sweep_config.get("parameters").get("encoder_name")
        == test_config.encoder_name
    )
    assert (
        output_sweep_config.get("parameters").get("batch_size")
        == test_config.batch_size
    )
    assert (
        output_sweep_config.get("parameters").get("lr_initial")
        == test_config.lr_initial
    )
    assert (
        output_sweep_config.get("parameters").get("max_epochs")
        == test_config.max_epochs
    )
    assert (
        output_sweep_config.get("parameters").get("horizontal_flip_prob")
        == test_config.horizontal_flip_prob
    )
    assert (
        output_sweep_config.get("parameters").get("vertical_flip_prob")
        == test_config.vertical_flip_prob
    )
    assert (
        output_sweep_config.get("parameters").get("rotate90_prob")
        == test_config.rotate90_prob
    )


def test_make_sweep_config_custom_values() -> None:
    """
    Test successful creation of sweep config using custom values.
    """
    test_config = SweepConfig(
        method="bayes",
        metric_name="val_dice_loss",
        metric_goal="minimize",
        seed={"values": [59]},
        threshold={"values": [0.4, 0.7]},
        encoder_name={"values": ["resnet50", "se_resnet50", "efficientnet-b6"]},
        batch_size={"values": [4, 8, 16, 32]},
        lr_initial={"distribution": "uniform", "min": 1.0e-5, "max": 1.0e-2},
        max_epochs={"values": [100]},
        horizontal_flip_prob={"distribution": "uniform", "min": 0.3, "max": 0.8},
        vertical_flip_prob={"distribution": "uniform", "min": 0.1, "max": 0.4},
        rotate90_prob={"distribution": "uniform", "min": 0.5, "max": 0.7},
    )

    # Call the test function.
    output_sweep_config = make_sweep_config(test_config)

    # Assertions
    assert output_sweep_config.get("method") == test_config.method
    assert output_sweep_config.get("metric").get("name") == test_config.metric_name
    assert output_sweep_config.get("metric").get("goal") == test_config.metric_goal
    assert output_sweep_config.get("parameters").get("seed") == test_config.seed
    assert (
        output_sweep_config.get("parameters").get("threshold") == test_config.threshold
    )
    assert (
        output_sweep_config.get("parameters").get("encoder_name")
        == test_config.encoder_name
    )
    assert (
        output_sweep_config.get("parameters").get("batch_size")
        == test_config.batch_size
    )
    assert (
        output_sweep_config.get("parameters").get("lr_initial")
        == test_config.lr_initial
    )
    assert (
        output_sweep_config.get("parameters").get("max_epochs")
        == test_config.max_epochs
    )
    assert (
        output_sweep_config.get("parameters").get("horizontal_flip_prob")
        == test_config.horizontal_flip_prob
    )
    assert (
        output_sweep_config.get("parameters").get("vertical_flip_prob")
        == test_config.vertical_flip_prob
    )
    assert (
        output_sweep_config.get("parameters").get("rotate90_prob")
        == test_config.rotate90_prob
    )


@patch("pathlib.Path.exists", return_value=True)
@patch("wandb.init")
@patch("wandb.Artifact")
@patch("wandb.Api")
def test_upload_best_model_to_wandb_success(
    mock_wandb_api: MagicMock,
    mock_wandb_artifact: MagicMock,
    mock_wandb_init: MagicMock,
    mock_weights_file_exists: MagicMock,
) -> None:
    """
    Test successful execution of upload_best_model_to_wandb().
    """
    # Set up mock W&B run.
    entity = "test-organization"
    project = "test_project"
    sweep_id = "test_id"
    mock_run = MagicMock(name="mock-run")
    mock_wandb_init.return_value.__enter__.return_value = mock_run

    # Set up mocks for wandb.Api() and wandb.Api().sweep().
    api = MagicMock(name="wandb-api")
    mock_wandb_api.return_value = api

    sweep = MagicMock(name="wandb-sweep")
    api.sweep.return_value = sweep

    # Mock runs
    run1 = MagicMock(name="run 1")
    run1.summary = {"val_loss": 0.1}
    run1.config = {"encoder_name": "resnet34", "batch_size": 32, "lr_initial": 1.0e-5}
    run1.id = "run1_id"

    run2 = MagicMock()
    run2.summary = {"val_loss": 0.05}
    run2.config = {"encoder_name": "resnet50", "batch_size": 4, "lr_initial": 1.0e-4}
    run2.id = "run2_id"
    sweep.runs = [run1, run2]

    # Mock artifact.
    artifact = MagicMock(name="artifact")
    mock_wandb_artifact.return_value = artifact

    # Call the test function.
    upload_best_model_to_wandb(entity, project, sweep_id)

    # Assertions
    mock_wandb_api.assert_called_once()
    api.sweep.assert_called_once_with(sweep_id)
    mock_weights_file_exists.assert_called_once()
    mock_wandb_init.assert_called_once_with(
        entity=entity, project=project, job_type="artifact-upload"
    )

    # Best run = run2
    encoder = run2.config.get("encoder_name")
    batch_size = run2.config.get("batch_size")
    lr_initial = run2.config.get("lr_initial")
    mock_wandb_artifact.assert_called_once_with(
        name="unet_model",
        type="model",
        description=f"Best model from sweep {sweep_id} based on validation loss",
        metadata={
            "run_id": run2.id,
            "val_loss": run2.summary.get("val_loss"),
            "encoder": encoder,
        },
    )

    artifact.add_file.assert_called_once_with(
        str(OUTPUT_PATH / f"{encoder}_batch{batch_size}_lr{lr_initial:.1e}_weights.pt")
    )
    mock_run.log_artifact.assert_called_once_with(artifact)
    mock_run.link_artifact.assert_called_once_with(
        artifact=mock_run.log_artifact.return_value,
        target_path=f"wandb-registry-{project}/models",
    )


@patch("wandb.finish")
@patch("amazon_seg_project.ops.wandb_utils.upload_best_model_to_wandb")
@patch("wandb.agent")
@patch("logging.info")
@patch("wandb.sweep")
@patch("wandb.login")
def test_run_sweep(
    mock_wandb_login: MagicMock,
    mock_wandb_sweep: MagicMock,
    mock_logging: MagicMock,
    mock_wandb_agent: MagicMock,
    mock_model_upload: MagicMock,
    mock_wandb_finish: MagicMock,
) -> None:
    """
    Test successful execution of run_sweep() using mocked dependencies.
    """
    config_object = SweepConfig()
    sweep_config_dict = make_sweep_config(config_object)
    mock_wandb_sweep.return_value = "mock_id"

    # Call the test function.
    run_sweep(config_object)

    # Assertions
    mock_wandb_login.assert_called_once()
    mock_wandb_sweep.assert_called_once_with(
        sweep_config_dict, project=config_object.project, entity=config_object.entity
    )
    mock_logging.info("Sweep ID: %s", mock_wandb_sweep.return_value)
    mock_wandb_agent.assert_called_once_with(
        mock_wandb_sweep.return_value, function=run_wandb_exp
    )
    # mock_model_upload.assert_called_once_with(
    #    config_object.entity, config_object.project, mock_wandb_sweep.return_value
    # )
    mock_wandb_finish.assert_called_once()
