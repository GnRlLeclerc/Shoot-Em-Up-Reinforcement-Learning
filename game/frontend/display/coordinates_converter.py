"""Utility class to convert coordinates between the screen pixel display and the game map"""

import numpy as np
import numpy.typing as npt
import pygame

from game.backend.environment import Environment
from game.frontend.window_settings import WindowSettings


class CoordinatesConverter:
    """Convert coordinates between the screen pixel display and the game map"""

    # The offsets represent an amount of pixels.

    # Offset to apply to the game map to center it in the screen
    map_offsets: npt.NDArray[np.int64]
    # Offsets to apply to the entities after conversion in order to align screen and game map origins
    entity_offsets: npt.NDArray[np.int64]

    scale: float

    # Store window settings (access player size, etc.)
    window_settings: WindowSettings

    # Precomputed map Rect object for faster display
    map_rect: pygame.Rect

    def __init__(self, environment: Environment, window_settings: WindowSettings):
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

    def to_screen_coords(
        self, coords: np.ndarray, size: float = 0
    ) -> npt.NDArray[np.int64]:
        """Convert game map coordinates to screen pixel coordinates

        Params:
        ------
        * coords: npt.NDArray[np.float64] - The position on the game map to be converted.
        * size: float - The size of the entity to be displayed in screen pixels,
          the half of which will be subtracted from the position.
        """
        coords = coords * self.scale + self.entity_offsets
        coords[1] = self.window_settings.height - coords[1]
        coords -= size * 0.5
        return coords.astype(np.int64)

    def to_game_coords(
        self, coords: tuple[int, int], size: float = 0
    ) -> npt.NDArray[np.float64]:
        """Convert screen pixel coordinates to game map coordinates

        Params:
        ------
        * coords: tuple[int, int] - The position on the screen to be converted.
        * size: float - The size of the entity to be displayed in screen pixels,
          the half of which will be subtracted from the position.
        """

        coords = np.array(coords, dtype=np.float64)
        coords += size * 0.5
        coords[1] = self.window_settings.height - coords[1]
        coords -= self.entity_offsets
        return coords / self.scale

    def to_game_size(self, size: float) -> float:
        """Convert a size in screen pixels to game map units"""
        return size / self.scale

    def to_screen_size(self, size: float) -> float:
        """Convert a size in game map units to screen pixels"""
        return size * self.scale

    def square_same_area(self, size: float) -> float:
        """Return a square width such that the resulting square has the same area as a circle with the input size as
        a radius"""
        return np.sqrt(np.pi) * size
