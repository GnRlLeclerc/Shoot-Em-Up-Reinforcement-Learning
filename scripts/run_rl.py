"""Run reinforcement learning models on the game environment"""

import setup  # pylint: disable=unused-import
import torch
from tensordict import TensorDict

from game.rl_environment.game_env import GameEnv

if __name__ == "__main__":

    game_env = GameEnv()

    actions = torch.zeros(1, 5)
    dummy = TensorDict({"actions": actions}, batch_size=1, device="cpu")
    result = game_env.step(dummy)
    print(result)
