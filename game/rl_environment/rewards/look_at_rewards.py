"""
Rewards to train the agent to look at the closest enemy.
"""

import numpy as np

from game.backend.environment import Environment, StepEvents
from game.backend.physics.math_utils import normalize
from game.rl_environment.rewards.base_rewards import BaseRewards


class LookAtRewards(BaseRewards):
    """Rewards to train the agent to look at the closest enemy.
    The reward output is within the range [-1, 1].
    """

    def reward(
        self,
        environment: Environment,
        events: StepEvents,  # pylint: disable=unused-argument
    ) -> float:
        """Compute the reward value from a single environment using the orientation angle difference to the closest
        enemy."""

        cum_reward = 0

        if environment.done:
            return cum_reward

        # Reward the player for looking in the general direction of the closest enemy (force learning to shoot at it)
        closest_enemy = None
        min_distance = self.game_settings.map_size
        for entity in environment.enemy_entities:
            distance = np.linalg.norm(
                entity.object.position - environment.player.object.position
            )
            if distance < min_distance:
                min_distance = distance
                closest_enemy = entity

        if closest_enemy is not None:
            player_direction = environment.player.direction
            enemy_direction = normalize(
                closest_enemy.object.position - environment.player.object.position
            )
            cos = np.dot(player_direction, enemy_direction)
            cum_reward += 1.0 * cos  # Reward the player for looking at the enemy

        return cum_reward * self.weight
