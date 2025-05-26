"""
Programmatic execution of Dagster job compute_val_metrics()

Execution syntax from main repo directory:
python executables/compute_val_metrics.py
"""

from dagster import RunConfig
from amazon_seg_project import defs
from amazon_seg_project.config import TrainedUnetConfig, ModelEvaluationConfig

# Specify configs for asset materializations. Modify as needed for custom configurations.
trained_unet_config = TrainedUnetConfig()
model_eval_config = ModelEvaluationConfig(threshold=0.35)

# Access the resolved job.
compute_val_metrics = defs.get_job_def("compute_val_metrics")

# Execute the job programmatically.
if __name__ == "__main__":
    result = compute_val_metrics.execute_in_process(
        run_config=RunConfig(
            {
                "trained_unet_model": trained_unet_config,
                "validation_metrics": model_eval_config,
            }
        )
    )
    print("Job execution success:", result.success)
