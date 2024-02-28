"""
Convert game observations and states to scaled tensors, and back.
"""

import numpy as np
import torch

from game.backend.entities.bullet_entity import BulletEntity
from game.backend.entities.enemy_entity import EnemyEntity
from game.backend.entities.player_entity import PlayerEntity
from game.backend.game_settings import GameSettings
from game.backend.physics.math_utils import normalize_rad_angle
from game.backend.player_actions import PlayerAction


class GameTensorConverter:
    """A converter class that helps transform game observations and states to tensors, and back."""

    game_settings: GameSettings

    # Store max positions, velocities, etc. in order to scale the tensors to [-1, 1]
    max_position: float
    max_velocity: float

    def __init__(self, game_settings: GameSettings):
        """Instantiate a new converter for a game environment"""

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
        * presence: when sharing tensors in batched environments, we may have to pad. This value serves as an indicator.

        All positions and velocities are scaled to [-1, 1] according to the game settings.

        :returns: A tensor observation of the enemy entity. Shape is (5,).
        """
        return torch.tensor(
            [
                *(enemy.object.position / self.max_position),
                *(enemy.object.velocity / self.max_velocity),
                1.0,
            ]
        )

    def bullet_to_tensor(self, bullet: BulletEntity) -> torch.Tensor:
        """Converts a bullet entity to a tensor observation.

        The conversion to a tensor is done in this format:
        * x position
        * y position
        * x speed
        * y speed
        * presence: when sharing tensors in batched environments, we may have to pad. This value serves as an indicator.

        All positions and velocities are scaled to [-1, 1] according to the game settings.

        :returns: A tensor observation of the bullet entity. Shape is (5,).
        """
        return torch.tensor(
            [
                *(bullet.object.position / self.max_position),
                *(bullet.object.position / self.max_velocity),
                1.0,
            ]
        )

    def actions_from_tensor(self, tensor: torch.Tensor) -> list[PlayerAction]:
        """Converts a tensor to a list of player actions. This function is not batched!

        The tensor should be of shape (action_count,), where action_count is the number of possible actions.
        We currently have 5 actions: move up, move down, move left, move right, and shoot. They are ordered
        in the Enum order of PlayerAction.

        TODO: how to handle player orientation ? Use 1 angle, or x-y continuous coordinates (better for learning) ?

        :returns: A list of player actions.
        """

        actions: list[PlayerAction] = []

        for proba, action in zip(tensor, PlayerAction):
            if proba > 0.5:
                actions.append(action)

        return actions
