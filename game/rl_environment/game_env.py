"""TorchRL environment wrapper for the game environment.
"""

import pygame
import torch
from PIL import Image
from tensordict import TensorDict, TensorDictBase
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs import EnvBase

from game.backend.entities.base_entity import EntityType
from game.backend.environment import Environment, StepEvents
from game.backend.game_settings import GameSettings
from game.frontend.display.coordinates_converter import CoordinatesConverter
from game.frontend.display.renderer import Renderer
from game.frontend.window_settings import WindowSettings
from game.rl_environment.game_tensor_converter import GameTensorConverter
from game.rl_environment.rewards.base_rewards import BaseRewards
from game.rl_environment.rewards.default_rewards import DefaultRewards


class GameEnv(EnvBase):
    """Environment wrapper for reinforcement learning with torchrl"""

    # Game utils
    environments: list[Environment]
    converter: GameTensorConverter
    rewards: BaseRewards
    done: bool  # Evaluate whether all environments are done

    # Rendering utils
    support_rendering: bool
    renderer: Renderer
    frames: list[Image]  # Store generated frames
    surface: pygame.Surface

    def __init__(
        self,
        game_settings: GameSettings | None = None,
        rewards: BaseRewards | None = None,
        support_rendering: bool = False,
        debug_window_settings: WindowSettings | None = None,
        device: DEVICE_TYPING = "cpu",
        batch_size: int = 1,
    ) -> None:
        """Instantiates a torchrl environment wrapper for the game environment.

        :param game_settings: Game settings shared by all environments.
        :param rewards: Reward module to use for the training reward computations.
        :param support_rendering: Whether to support rendering or not.
        :param debug_window_settings: Settings in order to render the game in real time for debugging.
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

        # Initialize renderer
        self.support_rendering = support_rendering
        if support_rendering:
            assert (
                len(self.environments) == 1
            ), "Rendering only works with one environment"

            # Initialize Pygame for off-screen rendering
            pygame.init()

            if debug_window_settings is not None:
                self.surface = pygame.display.set_mode(
                    (debug_window_settings.width, debug_window_settings.height)
                )
            else:
                pygame.display.set_mode(
                    (0, 0), pygame.NOFRAME
                )  # Tiny window, not shown
                debug_window_settings = WindowSettings()
                self.surface = pygame.Surface(
                    (debug_window_settings.width, debug_window_settings.height)
                )

            self.frames = []

            converter = CoordinatesConverter(
                self.environments[0], debug_window_settings
            )
            self.renderer = Renderer(converter, game_settings, self.surface)

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

        self.done = False
        self.frames = []

        return self.get_state()

    def _set_seed(self, seed: int | None):
        """Set seed for the environment."""
        raise NotImplementedError

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Do a step in the environment and return the next state and reward."""

        events: list[StepEvents] = []

        # Step all environments
        for i, env in enumerate(self.environments):
            torch_actions = tensordict["actions"][i]
            env_actions = self.converter.actions_from_tensor(torch_actions)
            events.append(env.step(env_actions))

        step_output = self.get_state()
        step_output["reward"] = self.rewards.rewards(self.environments, events)

        self.done = all(env.done for env in self.environments)

        return step_output.to(self.device)

    def get_state(self) -> TensorDictBase:
        """Get the current state of all environments in a batched tensordict"""

        # Count the max bullets and ennemies for this batch among all environments
        counts = Environment.get_max_entity_count(self.environments)

        player_observations = torch.zeros(self.batch_size + (6,))
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
            for index, enemy in enumerate(env.enemy_entities):
                enemy_observations[i, index] = self.converter.enemy_to_tensor(enemy)

            for index, bullet in enumerate(env.bullet_entities):
                bullet_observations[i, index] = self.converter.bullet_to_tensor(bullet)

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

    def render(self) -> None:
        """Render the current state of the environment into a frame list.
        Call save_to_gif to save the generated images to a gif file.
        """
        assert self.support_rendering, "Rendering is not supported"

        self.renderer.render_all(self.environments[0])

        # Save the frame as an image
        img_str = pygame.image.tostring(self.surface, "RGB")
        img = Image.frombytes("RGB", self.surface.get_size(), img_str)
        self.frames.append(img)

    def save_to_gif(self, filename: str) -> None:
        """Save the generated images to a gif image."""
        assert self.support_rendering, "Rendering is not supported"

        print("Saving to gif (this may take a while)...")
        self.frames[0].save(
            filename,
            save_all=True,
            append_images=self.frames[1:],
            duration=1 / self.environments[0].step_seconds,
            optimize=False,
            loop=0,
        )
