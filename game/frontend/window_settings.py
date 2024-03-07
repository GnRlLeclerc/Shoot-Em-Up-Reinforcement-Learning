"""Define window settings for the game frontend.
"""

from dataclasses import dataclass


@dataclass
class WindowSettings:
    """Class to hold window settings used in order to display the game frontend."""

    width: int = 1280
    height: int = 960

    fps: int = 30
