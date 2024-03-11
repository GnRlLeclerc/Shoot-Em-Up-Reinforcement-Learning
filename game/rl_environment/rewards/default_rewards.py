"""A default class for computing rewards. Feel free to modify it as you like.
"""

from game.backend.environment import Environment, StepEvents
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
            # Set a VERY high reward when enemies are shot down
            cum_reward += 10 * events["enemy_shot_count"]

        # Reward the player for shooting
        if events["player_did_shoot"]:
            cum_reward += 1

        # Traumatize the player for getting hit
        if events["enemy_contact_count"] > 0:
            cum_reward -= 100 * events["enemy_contact_count"]

        return cum_reward
