"""Run a PPO reinforcement learning model on the game environment.
"""

import setup  # pylint: disable=unused-import

from game.rl_agents.device import set_auto_global_device
from game.rl_agents.policies.nn_value import build_value_module
from game.rl_agents.transformers.fixed_transformer import FixedTransformer

if __name__ == "__main__":

    # Configure the models to use gpu if available
    set_auto_global_device()

    MAX_ENEMIES_SEEN = 2  # The model will be aware of the 2 closest enemies

    transformer = FixedTransformer(MAX_ENEMIES_SEEN)
    value_module = build_value_module(transformer, hidden_size=64)
