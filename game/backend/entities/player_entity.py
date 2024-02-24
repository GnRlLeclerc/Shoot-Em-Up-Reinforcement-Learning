"""The player entity"""

from game.backend.entities.base_entity import EntityBase, EntityType


class PlayerEntity(EntityBase):
    """The player entity base class"""

    @property
    def type(self) -> EntityType:
        """Returns the entity type (readonly)"""
        return EntityType.PLAYER

    # TODO: handle player inputs from pygame (make it compatible with RL)
