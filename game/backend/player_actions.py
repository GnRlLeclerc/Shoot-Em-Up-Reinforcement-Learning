"""Define all actions that the player can perform in an iteration of the game.
They may be stored in a list for the backend to process.
"""

from enum import Enum


class PlayerAction(Enum):
    """Simple Enum that identifies each action that the player can perform"""

    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    SHOOT = 4
