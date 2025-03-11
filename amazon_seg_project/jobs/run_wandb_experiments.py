"""
Job: Run a ML experiment using Weights & Biases.
"""

"""
from ..ops.wandb_experiment import model_training_exp

setup_config = {
    "horizontal_flip_prob": 0.5,
    "vertical_flip_prob": 0.5,
    "rotate90_prob": 0.5,
}

# Turn the graph into a job.
run_wandb_exp = model_training_exp.to_job(input_values={"wandb_config": setup_config})
"""
