"""This module defines the environment class, that holds all information about the current game state."""

from game.backend.entities.base_entity import EntityBase, EntityType
from game.backend.entities.player_entity import PlayerEntity
from game.backend.game_settings import GameSettings
from game.backend.physics.bounding_box import BoundingBox2D


class Environment:
    """Holds the current game state"""

    # Reference to the main player entity
    player: PlayerEntity

    # All other entities (note that we use a set for easy removal
    entities: set[EntityBase]
    delete_entities: list[EntityBase]  # Entities to be removed at the end of the step

    # Game mechanism values
    step_seconds: float  # Simulation duration between steps

    # Game map
    game_map: BoundingBox2D

    # Game settings
    game_settings: GameSettings

    def __init__(self) -> None:
        """Instantiates the environment"""
        self.player = PlayerEntity()
        self.entities = set()
        self.delete_entities = []
        self.step_seconds = 0.1

        # Create an empty game map
        self.game_map = BoundingBox2D(2, 2, 0, 0)
        self.game_settings = GameSettings()

    def step(self) -> None:
        """Updates the environment state"""

        # Update all entities
        for entity in self.entities:
            entity.step(self)

        # Update the player position
        # TODO
        self.player.object.position = self.game_map.clip_inside(
            self.player.object.position
        )

        # Filter out bullet entities that are out of the game map
        self.delete_entities.clear()
        # Identify entities to be removed in O(n) time
        for entity in self.entities:
            if entity.type == EntityType.BULLET and not self.game_map.collide_point(
                entity.object.position
            ):
                self.delete_entities.append(entity)

        # Remove the entities in O(n) time
        for entity in self.delete_entities:
            self.entities.remove(entity)
