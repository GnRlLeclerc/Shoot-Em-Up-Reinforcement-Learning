"""The basic player bullet entity"""

from game.backend.entities.base_entity import EntityBase, EntityType
from game.backend.environment import Environment


class BulletEntity(EntityBase):
    """The enemy entity base class"""

    @property
    def type(self) -> EntityType:
        """Returns the entity type (readonly)"""
        return EntityType.BULLET

    def step(self, env: "Environment") -> None:
        """Simulates one step of the bullet entity movement"""
        self.object.step(env.step_seconds)
