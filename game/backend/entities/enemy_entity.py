"""The basic enemy entity"""

from typing import TYPE_CHECKING
from enum import Enum

from game.backend.entities.base_entity import EntityBase, EntityType
from game.backend.physics.math_utils import normalize

if TYPE_CHECKING:
    # Avoid circular import which is only needed for type hints
    from game.backend.environment import Environment


class EnemyType(Enum):
    """The different types of enemies"""

    SKELETON = 0
    SLIME = 1


class EnemyEntity(EntityBase):
    """The enemy entity base class"""

    def __init__(self, obj, class_type: EnemyType | None = None) -> None:
        """Instantiates an enemy entity"""
        super().__init__(obj)
        if class_type is None:
            self.class_type = EnemyType.SKELETON
        else:
            self.class_type = class_type

    @property
    def type(self) -> EntityType:
        """Returns the entity type (readonly)"""
        return EntityType.ENEMY

    def step(self, env: "Environment") -> None:
        """Simulates one step of the enemy entity movement"""
        # Compute the velocity vector in order to make the enemy go nearer the player
        player_direction = normalize(env.player.object.position - self.object.position)

        enemy_speed = env.game_settings.enemy_speed
        if self.class_type == EnemyType.SKELETON:
            enemy_speed = env.game_settings.skeleton_speed
        elif self.class_type == EnemyType.SLIME:
            enemy_speed = env.game_settings.slime_speed

        self.object.velocity = player_direction * enemy_speed

        # Update the enemy position
        super().step(env)
