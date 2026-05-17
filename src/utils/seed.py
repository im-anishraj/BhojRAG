"""
Reproducibility utilities.
Sets seeds across Python stdlib, NumPy, and PyTorch for deterministic results.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all frameworks.

    Also enables deterministic CuDNN behavior, which may
    slightly reduce GPU throughput but ensures bitwise
    reproducibility across runs.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic CuDNN for full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash seed for dict ordering
    os.environ["PYTHONHASHSEED"] = str(seed)
