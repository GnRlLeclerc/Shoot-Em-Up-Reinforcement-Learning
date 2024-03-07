"""The player entity"""

import numpy as np

from game.backend.entities.base_entity import EntityBase, EntityType
from game.backend.physics.math_utils import normalize, rad_2_deg, rotation_angle
from game.backend.physics.physical_object import Object2D


class PlayerEntity(EntityBase):
    """The player entity base class"""

    # Direction at which the player looks
    direction: np.ndarray

    def __init__(self, obj: Object2D | None = None) -> None:
        """Instantiate a player entity"""
        super().__init__(obj)
        self.direction = np.zeros(2)

    @property
    def type(self) -> EntityType:
        """Returns the entity type (readonly)"""
        return EntityType.PLAYER

    def look_at(self, position: np.ndarray):
        """Move the player orientation in order to make it look at another position"""
        self.direction = normalize(position - self.object.position)

    @property
    def deg_angle(self) -> float:
        """Rotation angle in degrees, in the trigonometric direction.
        An angle of 0° means that the player is looking to the right
        """

        return rad_2_deg(rotation_angle(np.array([1, 0]), self.direction))

    @property
    def rad_angle(self) -> float:
        """Orientation of the player in radians, within [-2pi, 2pi]"""
        return rotation_angle(np.array([1, 0]), self.direction)
