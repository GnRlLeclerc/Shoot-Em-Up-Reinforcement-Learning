"""
Convert game observations and states to scaled tensors, and back.
"""

import numpy as np
import torch

from game.backend.entities.bullet_entity import BulletEntity
from game.backend.entities.enemy_entity import EnemyEntity
from game.backend.entities.player_entity import PlayerEntity
from game.backend.environment import ActionDict
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
        * orientation x
        * orientation y

        All positions and velocities are scaled to [-1, 1] according to the game settings.

        :returns: A tensor observation of the player entity. Shape is (6,).
        """
        return torch.tensor(
            [
                *(player.object.position / self.max_position),
                *(player.object.velocity / self.max_velocity),
                player.direction[0],
                player.direction[1],
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

    def actions_from_tensor(self, tensor: torch.Tensor) -> ActionDict:
        """Converts a tensor to a list of player actions. This function is not batched!

        The tensor should be of shape (action_count,), where action_count is the number of possible actions.
        We currently have 5 actions: move up, move down, move left, move right, and shoot. They are ordered
        in the Enum order of PlayerAction.

        Note on orientation: in order to make reinforcement learning models understand that the orientation angle
        loops around from -pi to pi, we use 2 values to encode it: a sine and a cosine prediction.

        :param tensor: A tensor of shape (7,) representing the probabilities of each action.
            * move up
            * move down
            * move left
            * move right
            * shoot
            * orientation sine
            * orientation cosine

        :returns: A list of player actions.
        """
        assert (
            len(tensor.shape) == 1
        ), "This function does not take batched tensors as input."
        assert tensor.shape == (7,), "The tensor should have shape (7,)"

        actions: list[PlayerAction] = []

        # Loop through the 5 main actions
        for proba, action in zip(tensor, PlayerAction):
            if proba > 0.5:
                actions.append(action)

        # Handle the orientation predictions
        orientation_sine = tensor[5]
        orientation_cosine = tensor[6]

        # Compute the orientation angle from the sine and cosine predictions
        orientation = np.arctan2(orientation_sine, orientation_cosine)
        orientation = normalize_rad_angle(orientation)

        # Create a normalized orientation vector from the angle
        orientation = np.array([np.cos(orientation), np.sin(orientation)])

        return {
            "actions": actions,
            "orientation": orientation,
        }
