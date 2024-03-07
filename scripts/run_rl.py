"""Run reinforcement learning models on the game environment"""

import setup  # pylint: disable=unused-import
import torch
from tensordict import TensorDict
from tqdm import tqdm

from game.rl_environment.game_env import GameEnv

if __name__ == "__main__":

    game_env = GameEnv(support_rendering=True)

    actions = torch.zeros(1, 7)
    dummy = TensorDict({"actions": actions}, batch_size=1, device="cpu")

    # 10 seconds at 30 fps
    for _ in tqdm(range(30 * 10)):
        game_env.step(dummy)
        game_env.render()

        if game_env.done:
            break

    print(f"Game finished at frame {len(game_env.frames) - 1}")
    game_env.save_to_gif("result.gif")
