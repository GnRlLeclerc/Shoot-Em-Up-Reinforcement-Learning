"""
Objective function class for evaluating a policy.
Useful for cma optimization.

Note that objective functions do not use batches !  They work with a single environment.
"""

from typing import Callable

import numpy as np
import pygame as pg
import torch
from torch import Tensor

from game.frontend.window_settings import WindowSettings
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
    debug_window_settings: WindowSettings | None

    def __init__(  # pylint: disable=too-many-arguments
        self,
        environment: GameEnv,
        policy: ExternalWeightPolicy,
        transformer: BaseTransformer,
        num_episodes: int,
        max_time_steps: int,
        minimize: bool = False,
    ):
        """Initialize the objective function

        :param environment: The game environment to use for evaluation.
        :param policy: The policy to evaluate.
        :param transformer: The state transformer to use.
        :param num_episodes: The number of episodes to run.
        :param max_time_steps: The maximum number of time steps per episode.
        :param minimize: Whether to minimize the output of the objective function.
        """

        assert (
            len(environment.environments) == 1
        ), "Objective function only works with one environment"

        self.environment = environment
        self.policy = policy
        self.transformer = transformer
        self.num_episodes = num_episodes
        self.max_time_steps = max_time_steps
        self.minimize = minimize

    def __call__(
        self, params: np.ndarray, render_to: str | None = None, debug: bool = False
    ) -> float:
        """Call the objective function with external parameters.

        :param params: The parameters to evaluate.
        :param render_to: If not None, render the environment to the given file path.
        :param debug: If True, print debug information and try to render in real time (30 fps)

        :returns: The average total reward over the episodes.
        """
        average_rewards = 0.0

        if debug or render_to is not None:
            assert self.environment.support_rendering, (
                "Objective function called with debug=True or render_to != None, "
                "but rendering is not supported for the underlying environment."
            )

        clock = None
        if debug:

            # Clock for controlling the frame rate
            clock = pg.time.Clock()

        for _ in range(self.num_episodes):
            # Reset the environment and get the initial state
            # Note that this is a batched game environment tensordict state
            tensordict_state = self.environment.reset()

            # Run the policy
            total_reward = 0.0

            with torch.no_grad():  # The objective function evaluates and does not train
                for i in range(self.max_time_steps):
                    # Compute the action
                    policy_state = self.transformer.state_from_tensordict(
                        tensordict_state, 0
                    )
                    action_tensor = self.policy(policy_state, params)
                    action_tensordict = self.transformer.action_to_dict([action_tensor])

                    # Step the environment
                    output = self.environment.step(action_tensordict)
                    tensordict_state = output["next"]
                    reward = tensordict_state["reward"][0].item()

                    # Accumulate the reward
                    total_reward += reward

                    # Render if needed
                    if render_to is not None:
                        self.environment.render()

                    if debug:
                        # Handle quit signals via the GUI
                        for event in pg.event.get():
                            if event.type == pg.QUIT:
                                return average_rewards

                        # Print debug info
                        print(f"Input {i}:", policy_state)
                        print(f"Output {i}:", action_tensor)

                        # Render in real time
                        self.environment.render()
                        pg.display.flip()
                        clock.tick(30)

                    # Check if the episode is done (as the batched environment has only one in this case, this will work
                    if self.environment.done:
                        break

                # Accumulate the total reward
                average_rewards += total_reward / self.num_episodes

        # Save the rendering if needed
        if render_to is not None:
            self.environment.save_to_gif(render_to)

        # Invert the rewards if we are minimizing an output using optimizers
        return -average_rewards if self.minimize else average_rewards
