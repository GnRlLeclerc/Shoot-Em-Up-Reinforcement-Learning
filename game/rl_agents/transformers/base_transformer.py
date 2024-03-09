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

    @abstractmethod
    def transform_state(self, batched_state: TensorDictBase, env_index: int) -> Tensor:
        """Transform the state tensordict to a model input tensor.

        :param batched_state: The batched state tensordict. Every one of its value tensor has a first dimension
            of batch size.
        :param env_index: The index of the environment in the batch.

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
