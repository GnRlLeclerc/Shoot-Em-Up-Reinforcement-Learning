"""A simple Neural Network Policy implementation that takes as an input
the current state, and that outputs an action tensor.

The action tensor is defined as follows (see the game tensor converter class):
A tensor of shape (7,) representing the probabilities of each action.
* move up
* move down
* move left
* move right
* shoot
* orientation sine
* orientation cosine

This class can be used with optimization strategies like cma.

"""

from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

DEVICE = Literal["cpu", "cuda"]


class NeuralPolicy(nn.Module):
    """A simple model that outputs a decision tensor compatible with our game environment"""

    # Model dimension information
    input_dim: int
    hidden_dim: int
    action_dim: int = 7

    device: DEVICE

    # Weight shapes
    shapes: list[torch.Size]
    param_count: int  # Total flattened parameter count

    def __init__(self, input_dim: int, hidden_dim: int, device: DEVICE = "cpu"):
        """Initialize the model"""
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # Neural network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.action_dim)

        # Build the weight shapes
        self.param_count = 0
        self.shapes = [param.shape for param in self.parameters()]
        for shape in self.shapes:
            self.param_count += np.prod(shape)

    def forward(self, x: Tensor, weights: np.ndarray | None = None) -> Tensor:
        """Forward pass of the neural network.

        :param x: The input tensor. Shape is (input_dim,).
        :param weights: The flattened weights of the neural network. If None, the current weights are used.
            If not None, the weights are loaded and used for the forward pass (useful for external optimization).

        Activation:
        * The first 5 output parameters are probabilities, so we use a sigmoid activation.
        * The last 2 output parameters are orientation predictions, so we use a tanh activation (-> [-1, 1]).
        """
        if weights is not None:
            self.from_numpy(weights)

        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)

        # Build the output tensor of size 7
        probas = torch.sigmoid(x[:5])
        orientation = torch.tanh(x[5:7])

        return torch.cat((probas, orientation), dim=0)

    def from_numpy(self, weights: np.ndarray) -> None:
        """Load tensor weights from a flat numpy array of shape (param_count,).
        (useful for interfacing with other optimization algorithms like cma).
        """
        offset = 0
        for param in self.parameters():
            current_count = np.prod(param.shape)
            new_param = weights[offset : offset + current_count].reshape(param.shape)
            param.data = torch.tensor(
                new_param, device=self.device, dtype=torch.float32
            )
            offset += current_count

    def to_numpy(self) -> np.ndarray:
        """Convert all weights to a flattened numpy array.
        The resulting array has shape (param_count,).
        """
        weights = np.zeros(self.param_count)

        offset = 0
        for param in self.parameters():
            current_count = np.prod(param.shape)
            weights[offset : offset + current_count] = (
                param.data.cpu().detach().numpy().ravel()
            )
            offset += current_count

        return weights

    def to_file(self, filename: str) -> None:
        """Save the weights to a file"""
        weights = self.to_numpy()
        np.savetxt(filename, weights)

    def from_file(self, filename: str) -> None:
        """Load the weights from a file"""
        weights = np.loadtxt(filename)
        self.from_numpy(weights)
