"""
Programmatic execution of job run_wandb_experiment()

Execution syntax from main repo directory:
python dagster_jobs/run_wandb_experiment.py
"""

from dagster import RunConfig
from amazon_seg_project import defs
from amazon_seg_project.config import ModelTrainingConfig


# Define custom config for run.
custom_config = ModelTrainingConfig(
    seed=43, encoder_name="resnet50", batch_size=4, lr_initial=1.0e-5, max_epochs=5
)

# Access the resolved job.
run_wandb_experiment = defs.get_job_def("run_wandb_experiment")

# Execute the job programmatically.
if __name__ == "__main__":
    result = run_wandb_experiment.execute_in_process(
        run_config=RunConfig({"run_wandb_training": custom_config})
    )
    print("Job execution success:", result.success)
