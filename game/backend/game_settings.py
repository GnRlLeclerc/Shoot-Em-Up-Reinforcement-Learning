"""Simple class to hold game settings"""

from dataclasses import dataclass


@dataclass
class GameSettings:
    """Simple class to hold game settings"""

    # Enemy entity speed per step
    enemy_speed: float = 1.0
