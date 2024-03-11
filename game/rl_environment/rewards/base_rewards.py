"""Base class for the computations of rewards from an environment state"""

from abc import ABC, abstractmethod

import torch

from game.backend.environment import Environment, StepEvents
from game.backend.game_settings import GameSettings


class BaseRewards(ABC):
    """The base reward computation class.
    Note that your rewards should be normalized between -1 and 1.
    """

    # Reference to the game settings
    game_settings: GameSettings

    # Inner weight multiplier for the reward
    weight: float

    def __init__(self, game_settings: GameSettings, weight: float = 1) -> None:
        """Instantiate the base reward computation class"""

        self.game_settings = game_settings
        self.weight = weight

    @abstractmethod
    def reward(self, environment: Environment, events: StepEvents) -> float:
        """Compute the reward value from a single environment.
        Note that the reward value should be normalized between -1 and 1.
        You should avoid computing rewards for environments that are `done`, leave them at 0.
        """

    def rewards(
        self, environments: list[Environment], events: list[StepEvents]
    ) -> torch.Tensor:
        """Compute a reward tensor from a list of environments"""
        return torch.tensor(
            [self.reward(env, event) for env, event in zip(environments, events)]
        )

    @staticmethod
    def all(
        rewards: list["BaseRewards"],
        environments: list[Environment],
        events: list[StepEvents],
    ):
        """Accumulate rewards across environments and rewards."""

        results = torch.zeros((len(environments), len(rewards)))

        for i, reward in enumerate(rewards):
            results[:, i] = reward.rewards(environments, events)

        # Sum the different reward modules
        return results.sum(dim=1)
