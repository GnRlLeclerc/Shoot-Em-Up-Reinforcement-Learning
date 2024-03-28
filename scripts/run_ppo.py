"""Run a PPO reinforcement learning model on the game environment.
"""

from dataclasses import dataclass

import setup  # pylint: disable=unused-import

# Split output into scale and loc (basically, normal distros.) où est-ce qu'on convertit ça nous-mêmes ?
# from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
import torch

from game.backend.game_settings import GameSettings
from game.rl_agents.device import DEVICE, set_auto_global_device
from game.rl_agents.policies.nn_value import build_value_module
from game.rl_agents.transformers.fixed_transformer import FixedTransformer
from game.rl_environment.game_env import GameEnv
from game.rl_environment.rewards.orientation_rewards import OrientationRewards
from game.rl_environment.rewards.position_rewards import PositionRewards


@dataclass
class HyperParameters:
    """Hyperparameters for the PPO model."""

    # Batch and frames
    frames_per_batch: int = 1
    total_frames: int = 1000

    # Advantage function hyperparameters
    gamma: float = 0.99
    lmbda: float = 0.95

    # Loss function hyperparameters
    clip_epsilon: float = 0.2
    entropy_eps: float = 0.01
    critic_coef: float = 1.0

    # Optimizer hyperparameters
    lr: float = 1e-3


if __name__ == "__main__":

    # Configure the models to use gpu if available
    set_auto_global_device()

    # Configure PPO hyperparameters
    params = HyperParameters()

    MAX_ENEMIES_SEEN = 2  # The model will be aware of the 2 closest enemies
    game_settings = GameSettings(player_health=-1, enemy_spawn_rate=1)

    # Prepare rewards
    rewards = [
        OrientationRewards(game_settings, weight=1),
        PositionRewards(game_settings, weight=2),
        # SurvivalRewards(game_settings, weight=100),
        # KillingRewards(game_settings, weight=10),
    ]

    # Train with infinite health and high enemy spawn rate (average 2 by second)
    environment = GameEnv(game_settings, rewards, support_rendering=True, batch_size=1)

    transformer = FixedTransformer(MAX_ENEMIES_SEEN)
    value_module = build_value_module(transformer, hidden_size=64)

    # TODO
    policy_module = None

    # Step collector in order to fill the replay buffer
    collector = SyncDataCollector(
        environment,
        policy_module,
        frames_per_batch=params.frames_per_batch,
        total_frames=params.total_frames,
        split_trajs=False,
        device=DEVICE,
    )

    # Replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=params.frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    # Advantage function module. Takes the value module as input.
    advantage_module = GAE(
        gamma=params.gamma,
        lmbda=params.lmbda,
        value_network=value_module,
        average_gae=True,
    )

    # Loss function module. Takes the policy and value modules as input.
    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=params.clip_epsilon,
        entropy_bonus=bool(params.entropy_eps),
        entropy_coef=params.entropy_eps,
        critic_coef=params.critic_coef,
        loss_critic_type="smooth_l1",
    )

    # Optimizer and learning rate scheduler
    optim = torch.optim.Adam(loss_module.parameters(), params.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, params.total_frames // params.frames_per_batch, 0.0
    )

    # TODO: training loop (see how the policy must be conceived in order to have correct outputs)
