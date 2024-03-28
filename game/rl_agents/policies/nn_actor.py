"""A simple Neural Network Actor that outputs the action to take in a given state.
For use with a PPO reinforcement learning model.

Its structure is similar to that of the ValueFunction, but it outputs actions to take.
"""

import torch
import torch.nn.functional as F
from tensordict.nn import TensorDictModule
from torch import Tensor, nn

from game.rl_agents.device import DEVICE
from game.rl_agents.transformers.base_transformer import BaseTransformer


class Actor(nn.Module):
    """NN Actor class for a PPO model."""

    def __init__(
        self,
        transformer: BaseTransformer,
        input_size: int,
        hidden_size: int,
        device: str | None = None,
    ) -> None:
        """Initialize the PPO module."""
        super().__init__()

        self.transformer = transformer
        self.device = device

        if device is None:
            self.device = DEVICE

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, 7)

    def forward(
        self, player_obs: Tensor, enemy_obs: Tensor, bullet_obs: Tensor
    ) -> Tensor:
        """Forward pass of the PPO policy"""
        input_tensor = self.transformer.transform_state(
            player_obs, enemy_obs, bullet_obs
        )

        x = F.tanh(self.layer1(input_tensor))
        x = F.tanh(self.layer2(x))
        x = F.tanh(self.layer3(x))
        x = self.out_layer(x)

        # Build the output tensor
        probas = torch.sigmoid(x[:5])
        orientation = x[5:7]

        return torch.cat((probas, orientation), dim=0)


def build_policy_module(
    transformer: BaseTransformer,
    input_size: int,
    hidden_size: int,
    device: str | None = None,
) -> TensorDictModule:
    """Build a PPO module that takes tensordicts as input"""

    module = Actor(transformer, input_size, hidden_size, device)

    return TensorDictModule(
        module,
        in_keys=["player_obs", "enemy_obs", "bullet_obs"],
        out_keys=["actions"],
    )
