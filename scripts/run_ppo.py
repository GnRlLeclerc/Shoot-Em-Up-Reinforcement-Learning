"""Run a PPO reinforcement learning model on the game environment.
"""

from collections import defaultdict
from dataclasses import dataclass

import setup  # pylint: disable=unused-import
import torch

# Split output into scale and loc (basically, normal distros.) où est-ce qu'on convertit ça nous-mêmes ?
# from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tensordict.nn import ProbabilisticTensorDictSequential
from tqdm import tqdm

from game.backend.game_settings import GameSettings
from game.rl_agents.device import DEVICE, set_auto_global_device
from game.rl_agents.policies.nn_actor import build_policy_module
from game.rl_agents.policies.nn_value import build_value_module
from game.rl_agents.transformers.fixed_transformer import FixedTransformer
from game.rl_environment.game_env import GameEnv
from game.rl_environment.rewards.orientation_rewards import OrientationRewards
from game.rl_environment.rewards.position_rewards import PositionRewards


@dataclass
class HyperParameters:
    """Hyperparameters for the PPO model."""

    # pylint: disable=too-many-instance-attributes

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

    # Training parameters
    num_epochs: int = 10
    sub_batch_size: int = 64  # Number of samples to take from the replay buffer
    max_grad_norm: float = 1.0  # Keep the gradient bounded (clipping)


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

    # Build the policy module
    policy_module = build_policy_module(transformer, hidden_size=64)

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
        actor_network=ProbabilisticTensorDictSequential(
            policy_module
        ),  # TODO: il lui faut aussi un probabilistic actor ?
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

    # Training loop
    logs = defaultdict(list)  # A dict that returns an empty list when a key is missing
    pbar = tqdm(total=params.total_frames)
    eval_str = ""  # pylint: disable=invalid-name

    # We iterate over the collector until it reaches its total number of frames.
    for i, tensordict_data in enumerate(collector):
        # tensordict_data contains one batch of data.
        for _ in range(params.num_epochs):
            # Update the advantage module
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())

            # Sample data from the replay buffer
            for _ in range(params.frames_per_batch // params.sub_batch_size):
                subdata = replay_buffer.sample(params.sub_batch_size)
                loss_vals = loss_module(subdata.to(DEVICE))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), params.max_grad_norm
                )
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        if i % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                # Evaluate the policy for max 1000 steps
                eval_rollout = environment.rollout(1000, policy_module)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout
        pbar.set_description(
            ", ".join([eval_str, cum_reward_str, stepcount_str, lr_str])
        )

        # Step the scheduler to update the learning rate
        scheduler.step()

    # Save the model
    torch.save(policy_module.state_dict(), "ppo_policy.pth")
