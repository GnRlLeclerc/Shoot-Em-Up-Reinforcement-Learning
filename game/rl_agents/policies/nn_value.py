"""A simple Neural Network Value function that outputs the value of the input state.
Remember that the value V(s) of a state is the total expected reward from that state onwards.

The forward method expects the following tensors in order :
* player_obs
* enemy_obs
* bullet_obs
"""

import torch.nn.functional as F
from torch import Tensor, nn
from torchrl.modules import ValueOperator

from game.rl_agents.device import DEVICE
from game.rl_agents.transformers.base_transformer import BaseTransformer


class ValueFunction(nn.Module):
    """Value function neural network.
    The input shape is automatically inferred from the first time the module is run.
    3 fully connected layers with tanh activation and a final linear layer with shape 1 are used.
    """

    # Transform the complex game observation tensors into a processable tensor
    transformer: BaseTransformer
    device: str

    def __init__(
        self, transformer: BaseTransformer, hidden_size: int, device: str | None = None
    ) -> None:
        """Initialize the value function module."""
        super().__init__()

        self.transformer = transformer
        self.device = device

        if device is None:
            self.device = DEVICE

        self.layer1 = nn.LazyLinear(hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, 1)

    def forward(
        self, player_obs: Tensor, enemy_obs: Tensor, bullet_obs: Tensor
    ) -> Tensor:
        """Forward pass of the value function. Computes a single value float tensor."""
        input_tensor = self.transformer.transform_state(
            player_obs, enemy_obs, bullet_obs
        )

        x = F.tanh(self.layer1(input_tensor))
        x = F.tanh(self.layer2(x))
        x = F.tanh(self.layer3(x))
        x = self.out_layer(x)

        return x


def build_value_module(
    transformer: BaseTransformer, hidden_size: int, device: str | None = None
) -> ValueOperator:
    """Build a value function module that takes tensordicts as input"""

    module = ValueFunction(transformer, hidden_size, device)

    return ValueOperator(
        module=module, in_keys=["player_obs", "enemy_obs", "bullet_obs"]
    )
