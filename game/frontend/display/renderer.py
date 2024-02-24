"""Various rendering functions in order to display the different entities on screen and the map.
"""

from typing import Callable

import numpy as np
import numpy.typing as npt
import pygame
from pygame import Surface, SurfaceType

from game.backend.entities.base_entity import EntityBase, EntityType
from game.backend.entities.bullet_entity import BulletEntity
from game.backend.entities.enemy_entity import EnemyEntity
from game.backend.entities.player_entity import PlayerEntity
from game.backend.environment import Environment
from game.frontend.display.colors import Color
from game.frontend.window_settings import WindowSettings


class Renderer:  # pylint: disable=too-many-instance-attributes
    """Utility class that helps render the different entities on screen with the correct position and size.

    This class initializes scaling and positioning values upon creation.
    It stores internally a reference to the screen it must render to.
    """

    # Screen to render to
    screen: Surface | SurfaceType

    # Rendering function lookup
    render_lookup: dict[EntityType, Callable[[any], None] | None]

    # Map rectangle
    map_rect: pygame.Rect

    # Scaling factors and offsets
    scale: float
    # The offsets represent an amount of pixels.
    map_offsets: npt.NDArray[np.int64]
    entity_offsets: npt.NDArray[np.int64]

    # Store window settings (access player size, etc.)
    window_settings: WindowSettings

    def __init__(
        self,
        environment: Environment,
        window_settings: WindowSettings,
        screen: Surface | SurfaceType,
    ) -> None:
        """Initializes the entity renderer with default settings."""
        self.screen = screen
        self.window_settings = window_settings

        # Compute game map bounds
        x_factor = window_settings.width / environment.game_map.width
        y_factor = window_settings.height / environment.game_map.height
        # Take the minimum in order to make the map fit in the screen
        self.scale = min(x_factor, y_factor)

        # Compute map offsets
        self.map_offsets = np.zeros(2, dtype=np.int64)

        if x_factor < y_factor:
            # There is extra space on the y-axis
            self.map_offsets[1] = (
                window_settings.height - environment.game_map.height * self.scale
            ) * 0.5
        else:
            # There is extra space on the x-axis
            self.map_offsets[0] = (
                window_settings.width - environment.game_map.width * self.scale
            ) * 0.5

        # Compute the entity display offsets
        self.entity_offsets = np.copy(self.map_offsets)
        lower_map_corner = environment.game_map.center - environment.game_map.half_size
        self.entity_offsets -= (lower_map_corner * self.scale).astype(np.int64)

        # Compute map rectangle
        self.map_rect = pygame.Rect(
            float(self.map_offsets[0]),
            float(self.map_offsets[1]),
            environment.game_map.width * self.scale,
            environment.game_map.height * self.scale,
        )

        # Register rendering functions
        self.render_lookup = {
            EntityType.PLAYER: self.render_player,
            EntityType.BULLET: self.render_bullet,
            EntityType.ENEMY: self.render_enemy,
        }

    def render_entity(self, entity: EntityBase) -> None:
        """Render an entity on screen.
        This method uses a dict lookup to find the correct rendering function for the entity in O(1) time.
        """
        render_fn = self.render_lookup[entity.type]

        if render_fn is not None:
            render_fn(entity)

    def render_player(self, player: PlayerEntity) -> None:
        """Render the player entity on screen."""
        # Render player
        pygame.draw.rect(
            self.screen,
            Color.RED,
            (
                *self._convert_position(
                    player.object.position, self.window_settings.player_size
                ),
                self.window_settings.player_size,
                self.window_settings.player_size,
            ),
        )

    def render_bullet(self, bullet: BulletEntity) -> None:
        """Render a bullet entity on screen."""
        # TODO

    def render_enemy(self, enemy: EnemyEntity) -> None:
        """Render an enemy entity on screen."""
        # TODO

    def render_map(self) -> None:
        """Render the map background on screen."""
        pygame.draw.rect(
            self.screen,
            Color.GREEN,
            self.map_rect,
        )

    def _convert_position(
        self, position: npt.NDArray[np.float64], size: float = 0
    ) -> npt.NDArray[np.int64]:
        """Convert a position from the game map to the screen.

        Params:
        ------
        * position: npt.NDArray[np.float64] - The position on the game map to be converted.
        * size: float - The size of the entity to be displayed, the half of which will be subtracted from the position.
        """
        position = position * self.scale + self.entity_offsets
        position[1] = self.window_settings.height - position[1]
        position -= size * 0.5
        return position.astype(np.int64)
