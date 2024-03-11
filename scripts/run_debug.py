"""Debug a model by running games and displaying them in real time."""

import pygame as pg
import setup  # pylint: disable=unused-import

from game.backend.game_settings import GameSettings
from game.frontend.window_settings import WindowSettings
from game.rl_agents.evaluation.objective_function import ObjectiveFunction
from game.rl_agents.policies.nn_policy import NeuralPolicy
from game.rl_agents.transformers.fixed_transformer import FixedTransformer
from game.rl_environment.game_env import GameEnv
from game.rl_environment.rewards.default_rewards import DefaultRewards

if __name__ == "__main__":
    # DEBUG SETTINGS
    window_settings = WindowSettings()

    # GAME SETTINGS
    MAX_ENEMIES_SEEN = 2  # The model will be aware of the 2 closest enemies

    # Train with infinite health and high enemy spawn rate (average 2 by second)
    game_settings = GameSettings(player_health=10, enemy_spawn_rate=2)
    rewards = DefaultRewards(game_settings)

    # Build the game environment with debug settings
    environment = GameEnv(
        game_settings,
        rewards,
        support_rendering=True,
        debug_window_settings=window_settings,
        batch_size=1,
    )
    policy = NeuralPolicy(input_dim=6 + MAX_ENEMIES_SEEN * 5, hidden_dim=64)
    transformer = FixedTransformer(MAX_ENEMIES_SEEN)
    objective_function = ObjectiveFunction(
        environment,
        policy,
        transformer,
        num_episodes=1,
        max_time_steps=900,  # 30 * 30 = 900 (30 seconds at 30 fps)
        minimize=True,
    )

    # Get the model's initial weights
    policy.from_file("weights.txt")  # Load existing weights
    weights = policy.to_numpy()

    pg.display.set_caption("Shoot'Em Up DEBUG")

    # Test the model and render to gif, but this time with player health at 1
    # A reference to this object is shared with the environment
    game_settings.player_health = 10
    # Render in real time, but do not save the gif result
    objective_function(weights, debug=True, render_to=None)
