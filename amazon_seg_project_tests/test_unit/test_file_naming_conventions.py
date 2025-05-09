"""
Unit tests for file naming utilities
"""

from amazon_seg_project.ops.file_naming_conventions import (
    name_weights_file,
    name_losscurve_csv_file,
)


def test_weights_file_naming() -> None:
    """
    Basic test for correct execution of name_weights_file()
    """
    encoder = "mock_encoder"
    batch_size = 32
    learning_rate = 0.005

    # Call the test function.
    weights_file = name_weights_file(encoder, batch_size, learning_rate)

    assert weights_file == "mock_encoder_batch32_lr5.0e-03_weights.pt"


def test_losscurve_csv_file_naming() -> None:
    """
    Basic test for correct execution of name_losscurve_csv_file()
    """
    encoder = "mock_encoder"
    batch_size = 16
    learning_rate = 0.0057

    # Call the test function.
    weights_file = name_losscurve_csv_file(encoder, batch_size, learning_rate)

    assert weights_file == "mock_encoder_batch16_lr5.7e-03_losscurve.csv"