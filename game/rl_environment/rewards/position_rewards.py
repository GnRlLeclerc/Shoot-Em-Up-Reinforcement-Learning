"""Reward the player based on its position.
The idea is to reward:
* being far away from the enemies
* being close to the center (ie: avoid fleeing to the corners)
"""

import numpy as np

from game.backend.environment import Environment, StepEvents
from game.backend.physics.math_utils import interval_map, sigmoid
from game.rl_environment.rewards.base_rewards import BaseRewards


class PositionRewards(BaseRewards):
    """Compute rewards based on the player's position.
    The output is in the range [-1, 1].
    """

    def reward(
        self,
        environment: Environment,
        events: StepEvents,  # pylint: disable=unused-argument
    ) -> float:
        """Compute the reward value from a single environment using the distance to enemies and to the map center"""

        cum_reward = 0

        if environment.done:
            return cum_reward

        map_pos_prio = 0.7

        # Compute the minimum distance to all enemies
        if len(environment.enemy_entities) > 0:
            min_distance = self.game_settings.map_size

            for entity in environment.enemy_entities:
                distance = np.linalg.norm(
                    entity.object.position - environment.player.object.position
                )
                min_distance = min(min_distance, distance)

            # Scale the minimum distance for sigmoid input.
            # Basically, the reward starts going down when the player is 20% of the map size away from an enemy
            sigmoid_in = interval_map(
                min_distance, 0, self.game_settings.map_size * 0.2, -4, 4
            )

            # Negative reward within the range [-1, 0]
            cum_reward -= sigmoid(-sigmoid_in) * (1 - map_pos_prio)

        # Reward the player for staying close to the center of the map
        center_distance = (
            np.linalg.norm(
                environment.player.object.position - environment.game_map.center
            )
            / self.game_settings.map_size
        )

        # Positive reward within the range [0, 1]
        cum_reward += (1 - center_distance) * map_pos_prio

        return cum_reward * self.weight
