"""Various rendering functions in order to display the different entities on screen and the map.
"""

from typing import Callable

import pygame
from pygame import Rect, Surface, SurfaceType

from game.backend.entities.base_entity import EntityBase, EntityType
from game.backend.entities.bullet_entity import BulletEntity
from game.backend.entities.enemy_entity import EnemyEntity
from game.backend.entities.player_entity import PlayerEntity
from game.frontend.display.colors import Color
from game.frontend.display.coordinates_converter import CoordinatesConverter
from game.frontend.display.render_utils import draw_rotated_rect


class Renderer:  # pylint: disable=too-many-instance-attributes
    """Utility class that helps render the different entities on screen with the correct position and size.

    This class initializes scaling and positioning values upon creation.
    It stores internally a reference to the screen it must render to.
    """

    # Screen to render to
    screen: Surface | SurfaceType

    # Rendering function lookup
    render_lookup: dict[EntityType, Callable[[any], None] | None]

    # Converter helper for the coordinates
    converter: CoordinatesConverter

    def __init__(
        self,
        converter: CoordinatesConverter,
        screen: Surface | SurfaceType,
    ) -> None:
        """Initializes the entity renderer with default settings."""
        self.screen = screen
        self.converter = converter

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
        player_size = self.converter.window_settings.player_size

        draw_rotated_rect(
            self.screen,
            Color.RED,
            Rect(
                *self.converter.to_screen_coords(player.object.position, player_size),
                player_size,
                player_size,
            ),
            player.angle,
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
            self.converter.map_rect,
        )
