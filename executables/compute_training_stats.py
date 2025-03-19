"""
Programmatic execution of Dagster job compute_training_stats()

Execution syntax from main repo directory:
python executables/compute_training_stats.py
"""

from amazon_seg_project import defs

# Access the resolved job.
compute_training_stats = defs.get_job_def("compute_training_stats")

# Execute the job programmatically.
if __name__ == "__main__":
    result = compute_training_stats.execute_in_process()
    print("Job execution success:", result.success)
