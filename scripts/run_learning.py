"""Run reinforcement learning models on the game environment"""

import cma
import setup  # pylint: disable=unused-import

from game.backend.game_settings import GameSettings
from game.rl_agents.evaluation.objective_function import ObjectiveFunction
from game.rl_agents.policies.nn_policy import NeuralPolicy
from game.rl_agents.transformers.fixed_transformer import FixedTransformer
from game.rl_environment.game_env import GameEnv
from game.rl_environment.rewards.orientation_rewards import OrientationRewards
from game.rl_environment.rewards.position_rewards import PositionRewards

# from game.rl_environment.rewards.default_rewards import DefaultRewards

if __name__ == "__main__":

    MAX_ENEMIES_SEEN = 2  # The model will be aware of the 2 closest enemies
    game_settings = GameSettings(player_health=-1, enemy_spawn_rate=0.5)

    # Prepare rewards
    rewards = [
        OrientationRewards(game_settings, weight=1),
        PositionRewards(game_settings, weight=2),
    ]

    # Train with infinite health and high enemy spawn rate (average 2 by second)
    # rewards = DefaultRewards(game_settings)
    environment = GameEnv(game_settings, rewards, support_rendering=True, batch_size=1)
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

    # Use cma for minimization
    initial_weights = policy.to_numpy()

    # The objective function will be called with the weights as input
    x_optimal, es = cma.fmin2(
        objective_function, x0=initial_weights, sigma0=10.0, options={"maxfevals": 1500}
    )

    # Save the weights to a file
    weights = policy.to_numpy()
    policy.to_file("weights.txt")

    # Test the model and render to gif, but this time with player health at 1
    # A reference to this object is shared with the environment
    game_settings.player_health = 10
    objective_function(weights, "result.gif")
