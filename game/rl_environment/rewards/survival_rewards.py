"""Reward the agent for surviving, ie not getting hit by the enemies.
"""

from game.backend.environment import Environment, StepEvents
from game.rl_environment.rewards.base_rewards import BaseRewards


class KillingRewards(BaseRewards):
    """Reward the player for surviving.
    The output reward is within the range[-1, 0] in average
    """

    def reward(self, environment: Environment, events: StepEvents) -> float:
        """Compute the reward value the amount of collisions with enemies during this step"""

        cum_reward = 0

        if environment.done:
            return cum_reward

        # Traumatize the player for getting hit
        if events["enemy_contact_count"] > 0:
            cum_reward -= events["enemy_contact_count"]

        return cum_reward
