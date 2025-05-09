"""
Standalone functions for defining file name conventions
"""

from dagster import op


@op
def name_weights_file(
    encoder_name: str, batch_size: int, base_learning_rate: float
) -> str:
    """
    Defines a naming convention for model weights file
    """
    file_name = (
        f"{encoder_name}_batch{batch_size}_lr{base_learning_rate:.1e}_weights.pt"
    )
    return file_name


@op
def name_losscurve_csv_file(
    encoder_name: str, batch_size: int, base_learning_rate: float
) -> str:
    """
    Defines a naming convention for model weights file
    """
    file_name = (
        f"{encoder_name}_batch{batch_size}_lr{base_learning_rate:.1e}_losscurve.csv"
    )
    return file_name
