"""Reward the agent for surviving, the longer it lives the better.
"""

from game.backend.environment import Environment, StepEvents
from game.rl_environment.rewards.base_rewards import BaseRewards


class TimeRewards(BaseRewards):
    """Reward the player for surviving longer"""

    # pylint: disable=unused-argument
    def reward(self, environment: Environment, events: StepEvents) -> float:
        """Compute the reward value the amout of time the agent has survived."""

        cum_reward = 0

        if environment.done:
            return cum_reward

        cum_reward += environment.time

        return cum_reward * self.weight
