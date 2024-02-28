"""A default class for computing rewards. Feel free to modify it as you like.
"""

import numpy as np

from game.backend.entities.enemy_entity import EnemyEntity
from game.backend.environment import Environment
from game.backend.physics.math_utils import interval_map, sigmoid
from game.rl_environment.rewards.base_rewards import BaseRewards


class DefaultRewards(BaseRewards):
    """Default class for computing rewards.
    This class simply gives negative rewards when the player is too close.
    """

    def reward(self, environment: Environment) -> float:
        """Compute the reward value from a single environment using the distance to enemies"""

        # Compute the minimum distance to all enemies
        min_distance = self.game_settings.map_size

        for entity in environment.entities:
            if isinstance(entity, EnemyEntity):
                distance = np.sqrt(
                    (entity.object.position * environment.player.object.position).sum()
                )
                min_distance = min(min_distance, distance)

        # Scale the minimum distance for sigmoid input.
        # Basically, the reward starts going down when the player is 20% of the map size away from an enemy
        sigmoid_in = interval_map(
            min_distance, 0, self.game_settings.map_size * 0.2, -4, 4
        )

        return -sigmoid(-sigmoid_in)
