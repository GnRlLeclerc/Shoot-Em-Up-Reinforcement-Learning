"""Simple class to hold game settings"""

from dataclasses import dataclass


@dataclass
class GameSettings:
    """Simple class to hold game settings"""

    # Enemy entity speed per step
    enemy_speed: float = 0.1

    # Player entity speed per step
    player_speed: float = 0.1

    # Map size
    map_size: float = 2
