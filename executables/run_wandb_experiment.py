"""
Programmatic execution of segmentation model training

Execution syntax from main repo directory:
python executables/run_wandb_experiment.py
"""

import logging
from amazon_seg_project.config import ModelTrainingConfig
from amazon_seg_project.ops.wandb_utils import run_wandb_training

# Define custom config for run.
train_config = ModelTrainingConfig(
    seed=43, encoder_name="resnet50", batch_size=32, lr_initial=1.0e-5, max_epochs=2
)

# Execute the job programmatically.
if __name__ == "__main__":
    run_wandb_training(train_config)
    logging.info("Model training completed.")
