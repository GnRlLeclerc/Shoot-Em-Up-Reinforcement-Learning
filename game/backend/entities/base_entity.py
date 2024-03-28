"""Base entity types and classes.
The base entity is an abstract class and cannot be instanced."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from game.backend.physics.physical_object import Object2D

if TYPE_CHECKING:
    # Avoid circular import which is only needed for type hints
    from game.backend.environment import Environment


class EntityType(Enum):
    """Entity type enumeration"""

    PLAYER = 0
    BULLET = 1
    ENEMY = 2


class EntityBase(ABC):
    """The base entity class.
    All derived entity classes should inherit from this class,
    and implement the abstract properties and methods."""

    # Position and velocity
    object: Object2D

    # Active flag (when iterating through sets and deleting entities
    active: bool = True

    def __init__(self, obj: Object2D | None = None) -> None:
        """Instantiates a base entity"""

        if obj is None:
            self.object = Object2D()
        else:
            self.object = obj

        self.movement_frame: int = 0

    @property
    @abstractmethod
    def type(self) -> EntityType:
        """Returns the entity type (readonly)"""

    def type_code(self) -> int:
        """Returns the entity type code"""
        return self.type.value

    def step(self, env: "Environment") -> None:
        """Updates the entity state based on the current environment state"""
        self.object.step(env.step_seconds)
