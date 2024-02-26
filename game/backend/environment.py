"""This module defines the environment class, that holds all information about the current game state."""

import numpy as np

from game.backend.entities.base_entity import EntityBase, EntityType
from game.backend.entities.player_entity import PlayerEntity
from game.backend.game_settings import GameSettings
from game.backend.physics.bounding_box import BoundingBox2D
from game.backend.physics.math_utils import normalize
from game.backend.player_actions import PlayerAction


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

    def __init__(self, game_settings: GameSettings | None = None) -> None:
        """Instantiates the environment"""
        self.player = PlayerEntity()
        self.entities = set()
        self.delete_entities = []
        self.step_seconds = 0.1

        if game_settings is None:
            game_settings = GameSettings()
        self.game_settings = game_settings

        # Create an empty game map
        self.game_map = BoundingBox2D(
            game_settings.map_size, game_settings.map_size, 0, 0
        )

    def step(self, actions: list[PlayerAction]) -> None:
        """Updates the environment state"""

        # Update all entities
        for entity in self.entities:
            entity.step(self)

        # Update the player position
        self.handle_player_actions(actions)
        self.player.step(self)
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

    def handle_player_actions(self, actions: list[PlayerAction]) -> None:
        """Handles the player actions and compute the new player state.

        Movement actions are combined into a single direction vector that is normalized before being
        multiplied by the player speed.
        """
        velocity = np.array([0, 0])

        for action in actions:
            if action == PlayerAction.MOVE_UP:
                velocity[1] += 1
            elif action == PlayerAction.MOVE_DOWN:
                velocity[1] -= 1
            elif action == PlayerAction.MOVE_LEFT:
                velocity[0] -= 1
            elif action == PlayerAction.MOVE_RIGHT:
                velocity[0] += 1
            elif action == PlayerAction.SHOOT:
                pass  # TODO: handle shooting and cooldown. Start with one weapon, but we can add more later

        if np.linalg.norm(velocity) > 0:
            self.player.object.velocity = (
                normalize(velocity) * self.game_settings.player_speed
            )
        else:
            self.player.object.velocity = velocity
