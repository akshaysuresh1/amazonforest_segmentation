"""
Programmatic execution of Dagster job compute_val_precision_recall_curve()

Execution syntax from main repo directory:
python executables/compute_val_precision_recall_curve.py
"""

from dagster import RunConfig
from amazon_seg_project import defs
from amazon_seg_project.config import TrainedUnetConfig, PrecRecallCurveConfig

# Specify configs for asset materializations. Modify as needed for custom configurations.
trained_unet_config = TrainedUnetConfig()
prec_recall_config = PrecRecallCurveConfig()

# Access the resolved job.
compute_val_precision_recall_curve = defs.get_job_def(
    "compute_val_precision_recall_curve"
)

# Execute the job programmatically.
if __name__ == "__main__":
    result = compute_val_precision_recall_curve.execute_in_process(
        run_config=RunConfig(
            {
                "trained_unet_model": trained_unet_config,
                "precision_recall_curve": prec_recall_config,
            }
        )
    )
    print("Job execution success:", result.success)
