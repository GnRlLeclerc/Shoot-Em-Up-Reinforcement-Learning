"""Utility functions for rendering things"""

import pygame
from pygame import Rect, Surface, SurfaceType


def draw_rotated_rect(
    surface: Surface | SurfaceType,
    color: tuple[int, int, int],
    rect: Rect,
    angle: float,
):
    """Draws a rotated rectangle on the given surface.

    Params:
    ------
    * surface: the surface on which to draw
    * color: the rectangle color
    * rect: the rectangle to be rotated
    * angle: the rotation angle in degrees (°). Rotation direction is trigonometric
    """
    # BUG: ya un truc bizarre

    # Create a new surface with the same size as the rectangle
    rect_surf = Surface((rect.width, rect.height), pygame.SRCALPHA)
    # Draw the rectangle onto the newly created surface
    pygame.draw.rect(rect_surf, color, rect_surf.get_rect())
    # Rotate the surface
    rotated_surf = pygame.transform.rotate(rect_surf, angle)
    # Get the new rect after rotation to keep the rectangle centered
    rotated_rect = rotated_surf.get_rect(center=rect.center)
    # Blit the rotated surface onto the original surface
    surface.blit(rotated_surf, rotated_rect.topleft)
