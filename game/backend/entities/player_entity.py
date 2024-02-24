"""The player entity"""

import numpy as np

from game.backend.entities.base_entity import EntityBase, EntityType
from game.backend.physics.math_utils import normalize, rad_2_deg, rotation_angle


class PlayerEntity(EntityBase):
    """The player entity base class"""

    # Direction at which the player looks
    direction: np.ndarray

    def __init__(self) -> None:
        """Instantiate a player entity"""
        super().__init__()
        self.direction = np.zeros(2)

    @property
    def type(self) -> EntityType:
        """Returns the entity type (readonly)"""
        return EntityType.PLAYER

    def look_at(self, position: np.ndarray):
        """Move the player orientation in order to make it look at another position"""
        self.direction = normalize(position - self.object.position)

    @property
    def angle(self) -> float:
        """Rotation angle in degrees, in the trigonometric direction.
        An angle of 0° means that the player is looking to the top
        """

        angle = rad_2_deg(rotation_angle(np.array([0, 1]), self.direction))
        return angle
