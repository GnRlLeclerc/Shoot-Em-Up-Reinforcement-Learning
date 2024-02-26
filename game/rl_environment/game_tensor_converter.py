"""
Convert game observations and states to scaled tensors, and back.
"""

import numpy as np
import torch

from game.backend.entities.enemy_entity import EnemyEntity
from game.backend.entities.player_entity import PlayerEntity
from game.backend.game_settings import GameSettings
from game.backend.physics.math_utils import normalize_rad_angle


class GameTensorConverter:
    """A converter class that helps transform game observations and states to tensors, and back."""

    game_settings: GameSettings

    # Store max positions, velocities, etc. in order to scale the tensors to [-1, 1]
    max_position: float
    max_velocity: float

    def __init__(self, game_settings: GameSettings | None = None):
        """Instantiate a new converter for a game environment"""

        if game_settings is None:
            game_settings = GameSettings()

        self.game_settings = game_settings

        self.max_position = game_settings.map_size
        self.max_velocity = max(game_settings.player_speed, game_settings.enemy_speed)

    def player_to_tensor(self, player: PlayerEntity) -> torch.Tensor:
        """Converts the player entity to a tensor observation.

        The conversion to a tensor is done in this format:
        * x position
        * y position
        * x speed
        * y speed
        * orientation: angle in [-pi, pi] scaled back to [-1, 1]

        All positions and velocities are scaled to [-1, 1] according to the game settings.

        :returns: A tensor observation of the player entity. Shape is (5,).
        """
        return torch.tensor(
            [
                *(player.object.position / self.max_position),
                *(player.object.velocity / self.max_velocity),
                normalize_rad_angle(player.rad_angle) / np.pi,
            ]
        )

    def enemy_to_tensor(self, enemy: EnemyEntity) -> torch.Tensor:
        """Converts the enemy entity to a tensor observation.

        The conversion to a tensor is done in this format:
        * x position
        * y position
        * x speed
        * y speed

        All positions and velocities are scaled to [-1, 1] according to the game settings.

        :returns: A tensor observation of the enemy entity. Shape is (4,).
        """
        return torch.tensor(
            [
                *(enemy.object.position / self.max_position),
                *(enemy.object.velocity / self.max_velocity),
            ]
        )
