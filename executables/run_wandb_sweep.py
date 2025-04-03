"""
Programmatic execution of W&B sweep

Execution syntax from main repo directory:
python executables/run_wandb_sweep.py
"""

import logging
from amazon_seg_project.config import SweepConfig
from amazon_seg_project.ops.wandb_utils import run_sweep

# Define custom config for run.
sweep_config = SweepConfig(
    method="grid",
    metric_name="val_loss",
    metric_goal="minimize",
    seed={"values": [43]},
    threshold={"values": [0.5]},
    encoder_name={"values": ["resnet50", "se_resnet50", "efficientnet-b6"]},
    batch_size={"values": [4, 8, 16, 32]},
    lr_initial={"values": [1.0e-4]},
    max_epochs={"values": [60]},
    horizontal_flip_prob={"values": [0.5]},
    vertical_flip_prob={"values": [0.5]},
    rotate90_prob={"values": [0.5]},
)

# Execute the job programmatically.
if __name__ == "__main__":
    run_sweep(sweep_config)
    logging.info("W&B sweep completed.")
