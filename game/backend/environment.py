"""This module defines the environment class, that holds all information about the current game state."""

from typing import TypedDict

import numpy as np

from game.backend.entities.base_entity import EntityType
from game.backend.entities.bullet_entity import BulletEntity
from game.backend.entities.enemy_entity import EnemyEntity
from game.backend.entities.player_entity import PlayerEntity
from game.backend.game_settings import GameSettings
from game.backend.physics.bounding_box import BoundingBox2D
from game.backend.physics.math_utils import normalize
from game.backend.physics.physical_object import Object2D
from game.backend.player_actions import PlayerAction
from game.utils.lazy_remove import LazyRemove


class StepEvents(TypedDict):
    """Store specific events that occurred during a game step, and that cannot be determined by just observing
    the new environment state.
    """

    enemy_contact_count: int
    done: bool


class Environment:  # pylint: disable=too-many-instance-attributes
    """Holds the current game state"""

    # Reference to the main player entity
    player: PlayerEntity

    # All other entities (note that we use a set for easy removal)
    enemy_entities: set[EnemyEntity]
    bullet_entities: set[BulletEntity]
    enemy_deleter: LazyRemove
    bullet_deleter: LazyRemove

    # Shooting cooldown in steps
    player_shoot_cooldown: int
    current_shot_cooldown: int

    # Game mechanism values
    step_seconds: float  # Simulation duration between steps

    # Game map
    game_map: BoundingBox2D

    # Game settings
    game_settings: GameSettings

    # Game flags and temporary values
    # Done flag: identify when the game is over
    done: bool

    def __init__(
        self, game_settings: GameSettings | None = None, step_seconds: float = 1 / 30
    ) -> None:
        """Instantiates the environment"""
        if game_settings is None:
            game_settings = GameSettings()

        # Instantiate entity holders
        self.enemy_entities = set()
        self.enemy_deleter = LazyRemove(self.enemy_entities)
        self.bullet_entities = set()
        self.bullet_deleter = LazyRemove(self.bullet_entities)

        self.step_seconds = step_seconds
        self.done = False

        self.game_settings = game_settings

        # Create an empty game map
        self.game_map = BoundingBox2D(
            game_settings.map_size, game_settings.map_size, 0, 0
        )

        self.player_shoot_cooldown = int(
            game_settings.player_shoot_delay / step_seconds
        )
        self.current_shot_cooldown = 0

        self.init_player()

    def step(self, actions: list[PlayerAction]) -> StepEvents:
        """Updates the environment state"""

        # Initialize the events for this step
        events: StepEvents = {
            "enemy_contact_count": 0,
            "done": self.done,
        }

        if self.done:
            return events

        # Update the player position
        self.handle_player_actions(actions)
        self.player.step(self)
        self.player.object.position = self.game_map.clip_inside(
            self.player.object.position
        )

        # Update all entity positions
        for bullet in self.bullet_entities:
            bullet.step(self)
        for enemy in self.enemy_entities:
            enemy.step(self)

        # Handle entity collisions
        for bullet in self.bullet_entities:
            collide = False
            for enemy in self.enemy_entities:
                if (
                    enemy.active
                    and bullet.active
                    and bullet.object.collide(enemy.object)
                ):
                    self.bullet_deleter.schedule_remove(bullet)
                    self.enemy_deleter.schedule_remove(enemy)
                    bullet.active = False
                    enemy.active = False
                    collide = True
                    break

            # Remove bullets that are out of the game map
            if not collide and not self.game_map.collide_point(bullet.object.position):
                self.bullet_deleter.schedule_remove(bullet)

        self.bullet_deleter.apply_remove()
        self.enemy_deleter.apply_remove()

        for enemy in self.enemy_entities:
            # Detect collisions with the player
            if self.player.object.collide(enemy.object):
                events["enemy_contact_count"] += 1
                self.done = True
                break

        self.enemy_deleter.apply_remove()

        # Try to spawn an enemy
        self.try_spawn_enemy()

        # Update shot cooldown
        if self.current_shot_cooldown > 0:
            self.current_shot_cooldown -= 1

        return events

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
                self.shoot_bullet()

        if np.linalg.norm(velocity) > 0:
            self.player.object.velocity = (
                normalize(velocity) * self.game_settings.player_speed
            )
        else:
            self.player.object.velocity = velocity

    def reset(self) -> None:
        """Resets the environment to its initial state"""
        self.init_player()
        self.enemy_entities.clear()
        self.enemy_deleter.clear()
        self.bullet_entities.clear()
        self.bullet_deleter.clear()
        self.done = False
        self.current_shot_cooldown = 0

    def get_entity_counts(self) -> dict[EntityType, int]:
        """Returns the count of each entity type in the environment"""
        counts = {
            EntityType.ENEMY: len(self.enemy_entities),
            EntityType.BULLET: len(self.bullet_entities),
        }

        return counts

    @staticmethod
    def get_max_entity_count(environments: list["Environment"]):
        """Get the maximum entity count among all environments for tensor batching"""

        max_counts = {entity_type: 0 for entity_type in EntityType}
        for env in environments:
            counts = env.get_entity_counts()
            for entity_type in EntityType:
                max_counts[entity_type] = max(
                    max_counts[entity_type], counts[entity_type]
                )
        return max_counts

    def try_spawn_enemy(self) -> None:
        """Try to spawn an enemy at a pseudo random position.
        This method should be called at every step.
        It has a probability of enemy_spawn_rate * step_seconds to spawn an enemy.
        """
        probability = self.game_settings.enemy_spawn_rate * self.step_seconds
        if np.random.rand() < probability:

            # Generate a random angle to place the enemy
            angle = np.random.rand() * 2 * np.pi
            position = self.game_map.edge_position_from_center_angle(angle)
            obj = Object2D.from_position(position)
            obj.size = self.game_settings.enemy_size
            self.enemy_entities.add(EnemyEntity(obj))

    def init_player(self) -> None:
        """Initialize the player position"""
        obj = Object2D.from_position(self.game_map.center)
        obj.size = self.game_settings.player_size
        self.player = PlayerEntity(obj)

    def shoot_bullet(self) -> None:
        """Shoot a bullet from the player position.
        This function checks and updates the shoot cooldown.
        """
        if self.current_shot_cooldown > 0:
            return

        self.current_shot_cooldown = self.player_shoot_cooldown

        obj = Object2D()
        obj.position = self.player.object.position + self.player.direction * (
            self.game_settings.player_size + self.game_settings.bullet_size
        )
        obj.velocity = self.player.direction * self.game_settings.bullet_speed
        obj.size = self.game_settings.bullet_size
        bullet = BulletEntity(obj)
        self.bullet_entities.add(bullet)
