"""
A transformer implementation that puts together in a tensor the player state, and the states of the n closest
enemies to the player. The number of enemies to consider is fixed at construction. Data is padded with zeros.
"""

import torch
from tensordict import TensorDictBase
from torch import Tensor

from game.rl_agents.transformers.base_transformer import BaseTransformer


class FixedTransformer(BaseTransformer):
    """A transformer implementation that puts together in a tensor the player state, and the states of the n closest
    enemies to the player. The number of enemies to consider is fixed at construction. Data is padded with zeros.
    """

    max_enemies: int

    def __init__(self, max_enemies: int):
        """Instantiate a new fixed transformer that considers the n closest enemies to the player."""
        self.max_enemies = max_enemies

    def transform_state(self, batched_state: TensorDictBase, env_index: int) -> Tensor:
        """Transform the state tensordict to a model input tensor."""

        # Player state of size 6 + n enemies of size 5 each
        output = torch.zeros(6 + 5 * self.max_enemies)

        output[:6] = batched_state["player_obs"][env_index]

        enemy_count = batched_state["enemy_obs"].shape[1]

        if enemy_count == 0:
            return output

        player_position = batched_state["player_obs"][env_index][:2]

        # Compute the distances to the enemies (we will sort the indices by distance)
        indices = list(range(enemy_count))

        distances = torch.norm(
            player_position - batched_state["enemy_obs"][env_index, :, 0:2], dim=1
        )

        # Sort the indices by distance
        indices = [i for _, i in sorted(zip(distances, indices))]

        # Fill the output tensor with the closest enemies
        for i in range(min(self.max_enemies, enemy_count)):
            output[6 + i * 5 : 6 + (i + 1) * 5] = batched_state["enemy_obs"][
                env_index, indices[i]
            ]

        return output
