"""
Unit tests for short ops defined in wandb_utils.py
"""

from unittest.mock import patch, MagicMock
from amazon_seg_project.config import SweepConfig
from amazon_seg_project.ops.wandb_utils import (
    make_sweep_config,
    run_sweep,
    run_wandb_exp,
)


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


@patch("amazon_seg_project.ops.wandb_utils.promote_best_model_to_registry")
@patch("wandb.finish")
@patch("wandb.agent")
@patch("logging.info")
@patch("wandb.sweep")
@patch("wandb.login")
def test_run_sweep(
    mock_wandb_login: MagicMock,
    mock_wandb_sweep: MagicMock,
    mock_logging: MagicMock,
    mock_wandb_agent: MagicMock,
    mock_wandb_finish: MagicMock,
    mock_promote_model_to_registry: MagicMock,
) -> None:
    """
    Test successful execution of run_sweep() using mocked dependencies.
    """
    config_object = SweepConfig()
    sweep_config_dict = make_sweep_config(config_object)
    mock_wandb_sweep.return_value = "mock_sweep_id"

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
    mock_wandb_finish.assert_called_once()
    mock_promote_model_to_registry.assert_called_once_with(
        config_object.entity, config_object.project, mock_wandb_sweep.return_value
    )
