"""Simple utility functions to help with numpy computations"""

import numpy as np


def normalize(array: np.ndarray) -> np.ndarray:
    """Normalize the input array"""
    return array / np.linalg.norm(array)


def normalize_rad_angle(angle: float) -> float:
    """Normalize an angle in radians to be in the range [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def direction_angle(vec: np.ndarray) -> float:
    """Get the angle in radians from the x-axis to the input vector"""
    return np.arctan2(vec[1], vec[0])


def rotation_angle(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Get the rotation angle in radians from vec1 to vec2.
    This function assumes that the input vectors were normalized

    The output range is [- 2*pi, 2*pi]
    """
    return direction_angle(vec2) - direction_angle(vec1)


def rad_2_deg(rad: float) -> float:
    """Converts radians to degrees"""
    return rad * 180 / np.pi


def deg_2_rad(deg: float) -> float:
    """Converts degrees to radians"""
    return deg * np.pi / 180


def sigmoid(x: float) -> float:
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))


def interval_map(
    x: float | np.ndarray,
    in_min: float = 0,
    in_max: float = 1,
    out_min: float = 0,
    out_max: float = 1,
) -> float | np.ndarray:
    """Map the input values from [in_min, in_max] to [out_min, out_max]"""
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
