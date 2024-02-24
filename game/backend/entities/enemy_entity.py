"""The basic enemy entity"""

from game.backend.entities.base_entity import EntityBase, EntityType
from game.backend.environment import Environment
from game.backend.physics.math_utils import normalize


class EnemyEntity(EntityBase):
    """The enemy entity base class"""

    @property
    def type(self) -> EntityType:
        """Returns the entity type (readonly)"""
        return EntityType.ENEMY

    def step(self, env: "Environment") -> None:
        """Simulates one step of the enemy entity movement"""
        # Compute the velocity vector in order to make the enemy go nearer the player
        player_direction = normalize(env.player.object.position - self.object.position)
        self.object.velocity = player_direction * env.game_settings.enemy_speed

        # Update the enemy position
        self.object.step(env.step_seconds)
