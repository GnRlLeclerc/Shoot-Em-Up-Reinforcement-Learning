"""
Objective function class for evaluating a policy.
Useful for cma optimization.

Note that objective functions do not use batches !  They work with a single environment.
"""

from typing import Callable

import numpy as np
from torch import Tensor

from game.rl_agents.transformers.base_transformer import BaseTransformer
from game.rl_environment.game_env import GameEnv

# Policy function or model that can be called with external numpy weights
ExternalWeightPolicy = Callable[[Tensor, np.ndarray], Tensor]


class ObjectiveFunction:
    """Objective Function implementation"""

    # Parameters
    environment: GameEnv
    policy: ExternalWeightPolicy
    transformer: BaseTransformer
    num_episodes: int
    max_time_steps: int
    minimize: bool

    def __init__(
        self,
        environment: GameEnv,
        policy: ExternalWeightPolicy,
        transformer: BaseTransformer,
        num_episodes: int,
        max_time_steps: int,
        minimize: bool = False,
    ):
        """Initialize the objective function"""

        assert (
            len(environment.environments) == 1
        ), "Objective function only works with one environment"

        self.environment = environment
        self.policy = policy
        self.transformer = transformer
        self.num_episodes = num_episodes
        self.max_time_steps = max_time_steps
        self.minimize = minimize

    def __call__(self, params: np.ndarray) -> float:
        """Call the objective function with external parameters.

        :returns: The average total reward over the episodes.
        """
        average_rewards = 0.0

        for _ in range(self.num_episodes):
            # Reset the environment and get the initial state
            # Note that this is a batched game environment tensordict state
            tensordict_state = self.environment.reset()

            # Run the policy
            total_reward = 0.0
            for _ in range(self.max_time_steps):
                # Compute the action
                policy_state = self.transformer.transform_state(tensordict_state, 0)
                action_tensor = self.policy(policy_state, params)
                action_tensordict = self.transformer.action_to_dict([action_tensor])

                # Step the environment
                output = self.environment.step(action_tensordict)
                reward = output["reward"][0]

                # Accumulate the reward
                total_reward += reward

                # Check if the episode is done (as the batched environment has only one in this case, this will work
                if self.environment.done:
                    break

            # Accumulate the total reward
            average_rewards += total_reward / self.num_episodes

        # Invert the rewards if we are minimizing an output using optimizers
        return -average_rewards if self.minimize else average_rewards
