"""Reward the agent for killing enemies.
"""

from game.backend.environment import Environment, StepEvents
from game.rl_environment.rewards.base_rewards import BaseRewards


class KillingRewards(BaseRewards):
    """Reward the player for shooting and for killing ennemies.
    The output reward is within the range[0, 1.1] in average
    """

    def reward(self, environment: Environment, events: StepEvents) -> float:
        """Compute the reward value from the different shooting and killing events that occurred during this step"""

        cum_reward = 0

        if environment.done:
            return cum_reward

        if events["enemy_shot_count"] > 0:
            # Set a VERY high reward when enemies are shot down
            cum_reward += 1 * events["enemy_shot_count"]

        # Reward the player for shooting
        if events["player_did_shoot"]:
            cum_reward += 0.1

        return cum_reward
