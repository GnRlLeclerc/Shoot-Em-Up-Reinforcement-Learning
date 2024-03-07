"""TorchRL environment wrapper for the game environment.
"""

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs import EnvBase

from game.backend.entities.base_entity import EntityType
from game.backend.entities.bullet_entity import BulletEntity
from game.backend.entities.enemy_entity import EnemyEntity
from game.backend.environment import Environment
from game.backend.game_settings import GameSettings
from game.rl_environment.game_tensor_converter import GameTensorConverter
from game.rl_environment.rewards.base_rewards import BaseRewards
from game.rl_environment.rewards.default_rewards import DefaultRewards


class GameEnv(EnvBase):
    """Environment wrapper for reinforcement learning with torchrl"""

    environments: list[Environment]
    converter: GameTensorConverter
    rewards: BaseRewards

    def __init__(
        self,
        game_settings: GameSettings | None = None,
        rewards: BaseRewards | None = None,
        device: DEVICE_TYPING = "cpu",
        batch_size: int = 1,
    ) -> None:
        """Instantiates a torchrl environment wrapper for the game environment.

        :param game_settings: Game settings shared by all environments.
        :param rewards: Reward module to use for the training reward computations.
        :param device: Device to use for the training (cpu or gpu).
        :param batch_size: Number of parallel environments.
        """
        super().__init__(batch_size=torch.Size((batch_size,)), device=device)

        # Initialize game settings if not provided in order to avoid having all environments build their own
        if game_settings is None:
            game_settings = GameSettings()

        # Initialize all game environments
        self.environments = [Environment(game_settings) for _ in range(batch_size)]
        self.converter = GameTensorConverter(game_settings)

        # Initialize rewards
        if rewards is None:
            rewards = DefaultRewards(game_settings)
        self.rewards = rewards

    def _reset(
        self,
        tensordict: TensorDictBase | None = None,  # pylint: disable=unused-argument
        **kwargs,  # pylint: disable=unused-argument
    ) -> TensorDictBase:
        """Reset environment and return the initial state.
        This method redefines the abstract method _reset() from EnvBase
        """

        # Reset all environments
        for env in self.environments:
            env.reset()

        return self.get_state()

    def _set_seed(self, seed: int | None):
        """Set seed for the environment."""
        raise NotImplementedError

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Do a step in the environment and return the next state and reward."""

        # Step all environments
        for i, env in enumerate(self.environments):
            torch_actions = tensordict["actions"][i]
            env_actions = self.converter.actions_from_tensor(torch_actions)
            env.step(env_actions)

        step_output = self.get_state()
        step_output["reward"] = self.rewards.rewards(self.environments)

        return step_output.to(self.device)

    def get_state(self) -> TensorDictBase:
        """Get the current state of all environments in a batched tensordict"""

        # Count the max bullets and ennemies for this batch among all environments
        counts = Environment.get_max_entity_count(self.environments)

        player_observations = torch.zeros(self.batch_size + (5,))
        bullet_observations = torch.zeros(
            self.batch_size + (counts[EntityType.BULLET], 5)
        )
        enemy_observations = torch.zeros(
            self.batch_size + (counts[EntityType.ENEMY], 5)
        )

        # NOTE: use this mask tensor in order to avoid learning from environments that have ended.
        dones = torch.zeros(self.batch_size, dtype=torch.bool)

        # Step all environments
        for i, env in enumerate(self.environments):

            # Convert player observations to tensors
            player_observations[i] = self.converter.player_to_tensor(env.player)
            dones[i] = env.done

            # Build the entity tensors
            enemy_count = 0
            bullet_count = 0

            for entity in env.entities:
                if isinstance(entity, EnemyEntity):
                    enemy_observations[i, enemy_count] = self.converter.enemy_to_tensor(
                        entity
                    )
                    enemy_count += 1
                elif isinstance(entity, BulletEntity):
                    bullet_observations[i, bullet_count] = (
                        self.converter.bullet_to_tensor(entity)
                    )
                    bullet_count += 1

        states = TensorDict(
            {
                "player_obs": player_observations,
                "enemy_obs": enemy_observations,
                "bullet_obs": bullet_observations,
                "done": dones,
            },
            batch_size=self.batch_size,
        )

        return states.to(self.device)
