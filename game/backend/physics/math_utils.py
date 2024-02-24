"""Simple utility functions to help with numpy computations"""

import numpy as np


def normalize(array: np.ndarray) -> np.ndarray:
    """Normalize the input array"""
    return array / np.linalg.norm(array)
