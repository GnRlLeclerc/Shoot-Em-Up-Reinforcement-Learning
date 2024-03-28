"""Manage and synchronize the device used for pytorch computations."""

import torch


def set_auto_global_device() -> None:
    """Automatically determine the best device for pytorch computations"""
    global DEVICE  # pylint: disable=global-statement
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"


def set_global_device(device: str) -> None:
    """Set the global device to the specified value"""
    global DEVICE  # pylint: disable=global-statement
    DEVICE = device


DEVICE = "cpu"
