"""
Example script.
Every script must import "common" at the top level in order to update the base directory,
enabling imports of "game".
"""

import setup  # pylint: disable=unused-import

from game.backend.entities.bullet_entity import BulletEntity
from game.backend.entities.player_entity import PlayerEntity
from game.backend.environment import Environment

if __name__ == "__main__":
    # Some random allocations and function calls to check that everything is working
    env = Environment()
    env.step()

    player = PlayerEntity()
    player.step(env)

    bullet = BulletEntity()
    print("Game is running !")
