"""Various rendering functions in order to display the different entities on screen and the map.
"""

from typing import Callable

import pygame
from pygame import Surface, SurfaceType

from game.backend.entities.base_entity import EntityBase, EntityType
from game.backend.entities.bullet_entity import BulletEntity
from game.backend.entities.enemy_entity import EnemyEntity, EnemyType
from game.backend.entities.player_entity import PlayerEntity
from game.backend.environment import Environment
from game.backend.game_settings import GameSettings
from game.frontend.display.colors import Color
from game.frontend.display.coordinates_converter import CoordinatesConverter


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

    skeleton_sprites: list[Surface] = [
        pygame.image.load(f"game/frontend/assets/skeleton/skeleton_animation{i+1}.png")
        for i in range(12)
    ]
    slime_sprites: list[Surface] = [
        pygame.image.load(f"game/frontend/assets/slime/slime_animation{i+1}.png")
        for i in range(4)
    ]

    player_moving_sprites: list[Surface] = [
        pygame.image.load(f"game/frontend/assets/player/player_animation{i+1}.png")
        for i in range(8)
    ]

    player_idle_sprites: list[Surface] = [
        pygame.image.load(f"game/frontend/assets/player/player_idle{i+1}.png")
        for i in range(2)
    ]

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
        self.skeleton_size = self.converter.to_screen_size(game_settings.skeleton_size)
        self.slime_size = self.converter.to_screen_size(game_settings.slime_size)
        self.bullet_size = self.converter.to_screen_size(game_settings.bullet_size)

        # Update the size of the sprites to match the enemy size
        self.skeleton_sprites = [
            pygame.transform.scale(
                sprite, (int(self.skeleton_size * 1.5), int(self.skeleton_size * 1.5))
            )
            for sprite in self.skeleton_sprites
        ]
        self.slime_sprites = [
            pygame.transform.scale(
                sprite, (int(self.slime_size * 1.5), int(self.slime_size * 1.5))
            )
            for sprite in self.slime_sprites
        ]

        self.player_moving_sprites = [
            pygame.transform.scale(
                sprite, (int(self.player_size * 1.5), int(self.player_size * 1.5))
            )
            for sprite in self.player_moving_sprites
        ]

        self.player_idle_sprites = [
            pygame.transform.scale(
                sprite, (int(self.player_size * 1.5), int(self.player_size * 1.5))
            )
            for sprite in self.player_idle_sprites
        ]

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

        player_image = self.player_moving_sprites[player.movement_frame]
        if player.object.velocity[0] < 0:
            player_image = pygame.transform.flip(player_image, True, False)
        if player.object.velocity[0] == 0 and player.object.velocity[1] == 0:
            # Short idle animation
            player_idle_frame = (
                1 if player.movement_frame < len(self.player_moving_sprites) // 2 else 0
            )
            player_image = self.player_idle_sprites[player_idle_frame]
            # flip the image if the cursor is on the left side to create an "aiming" effect
            if player.deg_angle > 90 or player.deg_angle < -90:
                player_image = pygame.transform.flip(player_image, True, False)

        position = self.converter.to_screen_coords(player.object.position, actual_size)
        self.screen.blit(player_image, position)
        player.movement_frame = (player.movement_frame + 1) % len(
            self.player_moving_sprites
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

        # Default values
        enemy_image: Surface = pygame.Surface((0, 0))
        enemy_size: float = self.enemy_size
        total_frames: int = 0
        image_length: int = 0
        image_height: int = 0

        if enemy.class_type == EnemyType.SKELETON:
            enemy_image = self.skeleton_sprites[enemy.movement_frame]
            total_frames = len(self.skeleton_sprites)
            # Images have a specific aspect ratio so we need to adjust the position to center them
            image_length = self.skeleton_sprites[0].get_width()
            image_height = self.skeleton_sprites[0].get_height()
            enemy_size = self.skeleton_size

        else:
            # elif enemy.class_type == EnemyType.SLIME:
            enemy_image = self.slime_sprites[enemy.movement_frame]
            total_frames = len(self.slime_sprites)
            image_length = self.slime_sprites[0].get_width()
            # I should have cropped the image better but this works as well
            image_height = self.slime_sprites[0].get_height() + 10
            enemy_size = self.slime_size

        # Need to flip the image if the enemy is moving left
        if enemy.object.velocity[0] < 0:
            enemy_image = pygame.transform.flip(enemy_image, True, False)

        position = self.converter.to_screen_coords(
            enemy.object.position, enemy_size
        ) - (image_length / 2, image_height / 2)

        self.screen.blit(enemy_image, position)

        # Animate multiple frames
        enemy.movement_frame = (enemy.movement_frame + 1) % total_frames

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
