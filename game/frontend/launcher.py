"""Game launcher class.
This launcher enables the user to play the game themselves.
"""

import pygame

from game.backend.environment import Environment
from game.backend.game_settings import GameSettings
from game.backend.physics.math_utils import normalize
from game.backend.player_actions import PlayerAction
from game.frontend.display.colors import Color
from game.frontend.display.coordinates_converter import CoordinatesConverter
from game.frontend.display.renderer import Renderer
from game.frontend.display.screens import Screen
from game.frontend.window_settings import WindowSettings


class Launcher:
    """Game frontend launcher.
    This class is used to launch the game in order to play it yourself.
    """

    window_settings: WindowSettings
    environment: Environment

    def __init__(
        self,
        window_settings: WindowSettings | None = None,
        game_settings: GameSettings | None = None,
    ) -> None:

        # Initialize window settings
        if window_settings is None:
            window_settings = WindowSettings()
        self.window_settings = window_settings

        # Initialize game environment
        self.environment = Environment(
            game_settings, step_seconds=1 / window_settings.fps
        )

    def launch(  # pylint: disable=too-many-branches, too-many-locals
        self,
    ) -> None:
        """Launch the game"""
        pygame.init()

        # Screen dimensions
        window = pygame.display.set_mode(
            (self.window_settings.width, self.window_settings.height)
        )
        pygame.display.set_caption("Shoot'Em Up")

        # Initialize renderer
        converter = CoordinatesConverter(self.environment, self.window_settings)
        renderer = Renderer(converter, self.environment.game_settings, window)

        # Initialize game state
        screen = Screen.TITLE
        running = True

        # Clock for controlling the frame rate
        clock = pygame.time.Clock()

        # List of actions that the player can perform at each frame
        actions: list[PlayerAction] = []

        while running:

            # Clear user actions from the previous frame
            actions.clear()

            # Handle user input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                    if screen == Screen.TITLE:
                        screen = Screen.GAME

            if screen == Screen.GAME:
                # Handle pressed keys
                keys = pygame.key.get_pressed()

                if keys[pygame.K_LEFT]:
                    actions.append(PlayerAction.MOVE_LEFT)
                if keys[pygame.K_RIGHT]:
                    actions.append(PlayerAction.MOVE_RIGHT)
                if keys[pygame.K_UP]:
                    actions.append(PlayerAction.MOVE_UP)
                if keys[pygame.K_DOWN]:
                    actions.append(PlayerAction.MOVE_DOWN)

                mouse_buttons = pygame.mouse.get_pressed()
                if mouse_buttons[0]:
                    actions.append(PlayerAction.SHOOT)

                # Orient the player in the direction of the mouse
                mouse_pos = pygame.mouse.get_pos()
                game_pos = converter.to_game_coords(mouse_pos)
                orientation = normalize(
                    game_pos - self.environment.player.object.position
                )

                # Update game state
                self.environment.step(
                    {
                        "actions": actions,
                        "orientation": orientation,
                    }
                )

                # Check if the game is over
                if self.environment.done:
                    screen = Screen.TITLE
                    self.environment.reset()

            # Render the game
            window.fill(Color.BLACK)

            if screen == Screen.TITLE:
                # Render the title screen
                font = pygame.font.Font(None, 74)
                text = font.render("Press any key", True, Color.WHITE)
                text_rect = text.get_rect(
                    center=(
                        self.window_settings.width / 2,
                        self.window_settings.height / 2,
                    )
                )
                window.blit(text, text_rect)

            elif screen == Screen.GAME:
                renderer.render_map()
                # Render the player as a square
                renderer.render_player(self.environment.player)

                # Render all the enemies as red circles
                for enemy in self.environment.enemy_entities:
                    renderer.render_enemy(enemy)

                # Render all bullets
                for bullet in self.environment.bullet_entities:
                    renderer.render_bullet(bullet)

            pygame.display.flip()

            # Control the frame rate to cap the fps
            clock.tick(self.window_settings.fps)

        pygame.quit()
