"""
Unit tests for modules defined in amazon_seg_project/ops/model_checks.py
"""

from typing import Dict
import torch
from amazon_seg_project.ops.model_checks import are_state_dicts_equal


def test_are_equal_state_dicts_equal() -> None:
    """
    Test are_state_dicts_equal() using two identical state dictionaries.
    """
    # Create two identical state dictionaries.
    state_dict1 = {
        "layer1.weight": torch.tensor([1.0, 2.0]),
        "layer1.bias": torch.tensor([0.5]),
    }
    state_dict2 = {
        "layer1.weight": torch.tensor([1.0, 2.0]),
        "layer1.bias": torch.tensor([0.5]),
    }

    # Call and assert the test function.
    assert are_state_dicts_equal(state_dict1, state_dict2)


def test_are_state_dicts_equal_for_different_keys() -> None:
    """
    Test are_state_dicts_equal() for dictionaries with different keys.
    """
    # Create dictionaries with different keys.
    state_dict1 = {
        "layer1.weight": torch.tensor([1.0, 2.0]),
        "layer1.bias": torch.tensor([0.5]),
    }
    state_dict2 = {
        "layer1.weight": torch.tensor([1.0, 2.0]),
        "layer2.bias": torch.tensor([0.5]),
    }

    # Call and assert the test function.
    assert not are_state_dicts_equal(state_dict1, state_dict2)


def test_are_state_dicts_equal_for_different_tensor_values() -> None:
    """
    Test are_state_dicts_equal() using dictionaries with different values.
    """
    # Create dictionaries with different values.
    state_dict1 = {
        "layer1.weight": torch.tensor([1.0, 2.0]),
        "layer1.bias": torch.tensor([0.5]),
    }
    state_dict2 = {
        "layer1.weight": torch.tensor([1.0, 2.0]),
        "layer1.bias": torch.tensor([0.6]),
    }

    # Call and assert the test function.
    assert not are_state_dicts_equal(state_dict1, state_dict2)


def test_are_state_dicts_equal_for_missing_key() -> None:
    """
    Test are_state_dicts_equal() using dictionaries with unequal number of keys.
    """
    # Create dictionaries with different sets of keys.
    state_dict1 = {
        "layer1.weight": torch.tensor([1.0, 2.0]),
        "layer1.bias": torch.tensor([0.5]),
    }
    state_dict2 = {
        "layer1.weight": torch.tensor([1.0, 2.0]),
    }

    # Call and assert the test function.
    assert not are_state_dicts_equal(state_dict1, state_dict2)


def test_are_state_dicts_equal_for_empty_dicts() -> None:
    """
    Test are_state_dicts_equal() for empty dictionaries.
    """
    # Create empty dictionaries.
    state_dict1: Dict[str, torch.Tensor] = {}
    state_dict2: Dict[str, torch.Tensor] = {}

    # Call and assert the test function.
    assert are_state_dicts_equal(state_dict1, state_dict2)
