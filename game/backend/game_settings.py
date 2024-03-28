"""Simple class to hold game settings"""

from dataclasses import dataclass


@dataclass
class GameSettings:
    """Simple class to hold game settings"""

    # pylint: disable=too-many-instance-attributes

    # Enemy entity speed per second & size
    enemy_speed: float = 0.4
    enemy_size: float = 0.05

    # Enemy Class specific settings
    # Skeleton
    skeleton_speed: float = 0.45
    skeleton_spawn_probability: float = 0.2
    skeleton_size = 0.08

    # Slime
    slime_speed: float = 0.3
    slime_size = 0.04

    # Player entity speed per second & size
    player_speed: float = 0.5
    player_size: float = 0.05
    # Player health. Setting a negative number will make the player invincible
    player_health: int = 1

    # Bullet entity speed per second & size
    bullet_speed: float = 1
    bullet_size: float = 0.02

    # Map size
    map_size: float = 2

    # Enemy spawn rate (per second)
    enemy_spawn_rate: float = 0.5

    # Delay between player shots (in seconds)
    player_shoot_delay: float = 0.3
