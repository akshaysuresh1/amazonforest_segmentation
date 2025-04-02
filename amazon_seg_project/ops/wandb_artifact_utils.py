"""
Utility functions to create and parse W&B run artifacts
"""

from typing import Dict
import logging
from dagster import op, In
from dagster import Any as dg_Any
import wandb
from wandb.sdk.wandb_run import Run


@op(
    ins={
        "wandb_run": In(dg_Any),
        "model_weights_file": In(str),
        "validation_stats": In(dg_Any),
    }
)
def create_and_log_wandb_artifact(
    wandb_run: Run, model_weights_file: str, validation_stats: Dict[str, float]
) -> None:
    """
    Create and log a Weights & Biases artifact.

    Args:
        wandb_run: W&B Run instance
        model_weights_file: Model weights file to be linked to W&B artifact
        validation_stats: Dictionary of metric results on validation data
    """
    encoder_name = wandb_run.config.get("encoder_name", "")
    # Create W&B artifact.
    artifact = wandb.Artifact(
        name=f"unet_with_{encoder_name}",
        type="model",
        metadata={
            "run_id": wandb_run.id,
            "encoder": encoder_name,
            "lr_initial": wandb_run.config.get("lr_initial"),
            "batch_size": wandb_run.config.get("batch_size"),
            **validation_stats,
        },
    )
    # Add model weights file to artifact.
    artifact.add_file(model_weights_file)
    # Log artifact to W&B.
    wandb_run.log_artifact(artifact)


@op
def promote_best_model_to_registry(entity: str, project: str, sweep_id: str) -> None:
    """
    Scan through runs in a sweep and push the model from the best run to W&B registry.

    Args:
        entity: Organization or user name
        project: Project name
        sweep_id: String identifier for W&B sweep
    """
    # Initialize the W&B API
    api = wandb.Api(overrides={"entity": entity, "project": project})

    try:
        # Fetch the sweep object using the sweep ID.
        sweep = api.sweep(sweep_id)
    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        return None

    # Access all the runs related to the sweep
    runs = sweep.runs

    if not runs:
        logging.info(f"No runs found for sweep {sweep_id}.")
        return None

    # Identify the run with the lowest validation loss as the best run.
    best_run = min(runs, key=lambda run: run.summary.get("val_loss", float("inf")))

    # Retrieve artifact from best run.
    logged_artifacts = best_run.logged_artifacts()
    if len(logged_artifacts) != 1:
        logging.info(
            "There exists no unique model weights file associated with run %s.",
            best_run.id,
        )
        return None
    artifact = logged_artifacts[0]

    # Add alias "best_model" to artifact from best run/
    artifact.aliases.append("best_model")
    artifact.save()

    # Initiate a run to link artifact to W&B registry.
    with wandb.init(entity=entity, project=project, job_type="artifact-upload") as run:
        run.link_artifact(
            artifact=artifact, target_path=f"{entity}/model-registry/unet-models"
        )
