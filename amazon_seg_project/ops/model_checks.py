"""
Handy functions to perform sanity checks on model during testing
"""

from typing import Dict
import torch


def are_state_dicts_equal(
    state_dict1: Dict[str, torch.Tensor], state_dict2: Dict[str, torch.Tensor]
) -> bool:
    """
    Check if two state dictionaries associated with PyTorch models are equal.
    """
    # Check if both dictionaries have the same keys
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    # Iterate over each key in the first state dictionary.
    for key, state_val in state_dict1.items():
        # Check if the tensors for this key are equal between the two dictionaries.
        if not torch.allclose(state_val, state_dict2[key]):
            return False

    return True
