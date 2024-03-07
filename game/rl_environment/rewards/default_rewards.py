"""A default class for computing rewards. Feel free to modify it as you like.
"""

import numpy as np

from game.backend.environment import Environment, StepEvents
from game.backend.physics.math_utils import interval_map, sigmoid
from game.rl_environment.rewards.base_rewards import BaseRewards


class DefaultRewards(BaseRewards):
    """Default class for computing rewards.
    This class simply gives negative rewards when the player is too close.
    """

    def reward(self, environment: Environment, events: StepEvents) -> float:
        """Compute the reward value from a single environment using the distance to enemies"""

        if environment.done:
            return 0.0

        cum_reward = 0

        if events["enemy_shot_count"] > 0:
            cum_reward += 10  # Set a VERY high reward when enemies are shot down

        # Compute the minimum distance to all enemies
        min_distance = self.game_settings.map_size

        for entity in environment.enemy_entities:
            distance = np.sqrt(
                (entity.object.position * environment.player.object.position).sum()
            )
            min_distance = min(min_distance, distance)

        # Scale the minimum distance for sigmoid input.
        # Basically, the reward starts going down when the player is 20% of the map size away from an enemy
        sigmoid_in = interval_map(
            min_distance, 0, self.game_settings.map_size * 0.2, -4, 4
        )

        cum_reward -= sigmoid(-sigmoid_in)

        return cum_reward
