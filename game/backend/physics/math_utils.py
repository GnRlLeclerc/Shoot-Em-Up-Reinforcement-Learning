"""Simple utility functions to help with numpy computations"""

import numpy as np


def normalize(array: np.ndarray) -> np.ndarray:
    """Normalize the input array"""
    return array / np.linalg.norm(array)


def rotation_angle(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Get the rotation angle in radians from vec1 to vec2.
    This function assumes that the input vectors were normalized

    The output range is [- 2*pi, 2*pi]
    """
    return np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])


def rad_2_deg(rad: float) -> float:
    """Converts radians to degrees"""
    return rad * 180 / np.pi


def deg_2_rad(deg: float) -> float:
    """Converts degrees to radians"""
    return deg * np.pi / 180
