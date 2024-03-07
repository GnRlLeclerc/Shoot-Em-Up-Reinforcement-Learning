"""Various rendering functions in order to display the different entities on screen and the map.
"""

from typing import Callable

import pygame
from pygame import Rect, Surface, SurfaceType

from game.backend.entities.base_entity import EntityBase, EntityType
from game.backend.entities.bullet_entity import BulletEntity
from game.backend.entities.enemy_entity import EnemyEntity
from game.backend.entities.player_entity import PlayerEntity
from game.backend.environment import Environment
from game.backend.game_settings import GameSettings
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
    game_settings: GameSettings

    # Precompute sizes
    player_size: float
    bullet_size: float
    enemy_size: float

    def __init__(
        self,
        converter: CoordinatesConverter,
        game_settings: GameSettings,
        screen: Surface | SurfaceType,
    ) -> None:
        """Initializes the entity renderer with default settings."""
        self.converter = converter
        self.game_settings = game_settings
        self.screen = screen

        # Register rendering functions
        self.render_lookup = {
            EntityType.PLAYER: self.render_player,
            EntityType.BULLET: self.render_bullet,
            EntityType.ENEMY: self.render_enemy,
        }

        self.player_size = self.converter.to_screen_size(game_settings.player_size)
        self.enemy_size = self.converter.to_screen_size(game_settings.enemy_size)
        self.bullet_size = self.converter.to_screen_size(game_settings.bullet_size)

    def render_entity(self, entity: EntityBase) -> None:
        """Render an entity on screen.
        This method uses a dict lookup to find the correct rendering function for the entity in O(1) time.
        """
        render_fn = self.render_lookup[entity.type]

        if render_fn is not None:
            render_fn(entity)

    def render_player(self, player: PlayerEntity) -> None:
        """Render the player entity on screen."""
        actual_size = self.converter.square_same_area(self.player_size)

        draw_rotated_rect(
            self.screen,
            Color.BLUE,
            Rect(
                *self.converter.to_screen_coords(player.object.position, actual_size),
                actual_size,
                actual_size,
            ),
            player.deg_angle,
        )

    def render_bullet(self, bullet: BulletEntity) -> None:
        """Render a bullet entity on screen."""
        # Draw a magenta circle
        pygame.draw.circle(
            self.screen,
            Color.MAGENTA,
            self.converter.to_screen_coords(bullet.object.position, self.bullet_size),
            self.bullet_size,
        )

    def render_enemy(self, enemy: EnemyEntity) -> None:
        """Render an enemy entity on screen."""
        # Draw a red circle
        pygame.draw.circle(
            self.screen,
            Color.RED,
            self.converter.to_screen_coords(enemy.object.position, self.enemy_size),
            self.enemy_size,
        )

    def render_map(self) -> None:
        """Render the map background on screen."""
        pygame.draw.rect(
            self.screen,
            Color.GREEN,
            self.converter.map_rect,
        )

    def render_all(self, environment: Environment) -> None:
        """Render all entities on screen."""
        # Fill the screen black
        self.screen.fill(Color.BLACK)

        self.render_map()

        # Render all bullets
        for bullet in environment.bullet_entities:
            self.render_bullet(bullet)

        # Render all ennemies
        for enemy in environment.enemy_entities:
            self.render_enemy(enemy)

        # Render the player
        self.render_player(environment.player)

        # Update the display
        pygame.display.flip()
