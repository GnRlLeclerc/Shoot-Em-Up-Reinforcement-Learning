"""
A transformer implementation that puts together in a tensor the player state, and the states of the n closest
enemies to the player. The number of enemies to consider is fixed at construction. Data is padded with zeros.
"""

import numpy as np
import torch
from torch import Tensor

from game.backend.physics.math_utils import sorted_indices
from game.rl_agents.transformers.base_transformer import BaseTransformer
from game.rl_environment.game_tensor_converter import ENEMY_SPAN, PLAYER_SPAN


class FixedTransformer(BaseTransformer):
    """A transformer implementation that puts together in a tensor the player state, and the states of the n closest
    enemies to the player. The number of enemies to consider is fixed at construction. Data is padded with zeros.
    """

    max_enemies: int

    def __init__(self, max_enemies: int):
        """Instantiate a new fixed transformer that considers the n closest enemies to the player."""
        self.max_enemies = max_enemies

    def transform_state(
        self,
        player_obs: Tensor,
        enemy_obs: Tensor,
        bullet_obs: Tensor,  # pylint: disable=unused-argument
    ) -> Tensor:
        """Transform the state tensordict to a model input tensor."""

        # Detect batches
        is_batched = player_obs.ndim > 1
        batch_size = player_obs.shape[0] if is_batched else 1

        if is_batched:
            output = torch.zeros(batch_size, 6 + 5 * self.max_enemies)
        else:
            # Player state of size 6 + n enemies of size 5 each
            output = torch.zeros(6 + 5 * self.max_enemies)

        output[..., :6] = player_obs

        # NOTE: in a batched tensor, the enemy count dimension is shared by all.
        enemy_count = enemy_obs.shape[-2]

        if enemy_count == 0:
            return output

        # Compute distances from player to ennemies
        player_position = player_obs[..., 2]
        # Ignore padding (count non-zero presence flags (index 4))
        actual_enemy_count = np.count_nonzero(enemy_obs[..., 4], axis=-1)

        # Compute the distances to the enemies (we will sort the indices by distance)
        distances = torch.norm(player_position - enemy_obs[..., 0:2], dim=-1)

        if is_batched:
            for batch in range(batch_size):
                enemy_count = actual_enemy_count[batch]
                indices = sorted_indices(distances[batch, :enemy_count].numpy())

                # Fill the output tensor with the closest enemies
                for i in range(min(self.max_enemies, enemy_count)):
                    output[
                        batch,
                        PLAYER_SPAN
                        + i * ENEMY_SPAN : PLAYER_SPAN
                        + (i + 1) * ENEMY_SPAN,
                    ] = enemy_obs[batch, indices[i]]

        else:
            indices = sorted_indices(distances[:actual_enemy_count].numpy())
            # Fill the output tensor with the closest enemies
            for i in range(min(self.max_enemies, actual_enemy_count)):
                output[
                    PLAYER_SPAN + i * ENEMY_SPAN : PLAYER_SPAN + (i + 1) * ENEMY_SPAN
                ] = enemy_obs[indices[i]]

        return output
