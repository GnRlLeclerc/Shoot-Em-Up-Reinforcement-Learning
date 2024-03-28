"""
Base class for helpers that transform tensor action outputs to batched tensordicts ready for the game environment,
and to convert state tensordicts to input tensors for models to interpret. You may implement many different strategies
to convert and preprocess environment states into model inputs
"""

from abc import ABC, abstractmethod

from tensordict import TensorDict, TensorDictBase
from torch import Tensor


class BaseTransformer(ABC):
    """The base class for transforming batched tensordicts states to singe model inputs."""

    def state_from_tensordict(
        self, batched_state: TensorDictBase, env_index: int | None = None
    ):
        """Extract observations from a tensordict to tensors for a specific environment

        :param batched_state: The batched state tensordict. Every one of its value tensor has a first dimension
            of batch size.
        :param env_index: The index of the environment in the batch. None if all batched states should be transformed.

        :returns: The separated player observation tensor, enemy observation tensor and bullet observation tensor in
        a tuple
        """

        if env_index is not None:
            player_obs = batched_state["player_obs"][env_index]
            enemy_obs = batched_state["enemy_obs"][env_index]
            bullet_obs = batched_state["bullet_obs"][env_index]
        else:
            player_obs = batched_state["player_obs"]
            enemy_obs = batched_state["enemy_obs"]
            bullet_obs = batched_state["bullet_obs"]

        return self.transform_state(player_obs, enemy_obs, bullet_obs)

    @abstractmethod
    def transform_state(
        self,
        player_obs: Tensor,
        enemy_obs: Tensor,
        bullet_obs: Tensor,
    ) -> Tensor:
        """Transform the state tensordict to a model input tensor.

        :param player_obs: The player observation tensor.
        :param enemy_obs: The enemy observation tensor.
        :param bullet_obs: The bullet observation tensor.

        :returns: A model input tensor, from the given environment state index, for a single forward pass of a model.
        """

    def action_to_dict(self, actions: list[Tensor] | Tensor) -> TensorDictBase:
        """Transform a list of state tensordicts to a batch of model input tensors"""
        if isinstance(actions, list):
            return TensorDict({"actions": actions}, batch_size=len(actions))

        # Else, this is a tensor
        assert (
            len(actions.shape) == 2
        ), "The actions tensor should be batched (1st dimension = batches, 2 dimension = actions)."

        return TensorDict({"actions": actions}, batch_size=actions.shape[0])
