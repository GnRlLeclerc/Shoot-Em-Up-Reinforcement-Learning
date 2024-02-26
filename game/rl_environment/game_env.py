"""TorchRL environment wrapper for the game environment.

TODO: voir comment transformer le format de l'environnement en tensordict
+ pour le DQN qui n'a pas besoin de ça à priori ?

TODO: définir notre manière de convertir to & from environments.
on ne va pas forcément embed tout de la même manière ?

=> ennemis : créer une "liste de tenseurs" (ajouter une dimension au bon endroit, le dernier) pour que ça puisse
être processed facilement
=> mettre des utility functions dans un truc à côté.
[batch size, max enemies, enemy features]

padding strategy:
* use zeros
* maybe add a 5th value that is 1 if enemy is present, 0 else (to avoid confusion with padding)

=> normalize input values around 0 so that padding with 0 makes sense.

normalization: use the game settings ! normalize speed based on the max speed available (player x bullet x enemy)
same for space, use the game map for position normalization.

mask tensor ? ça va dépendre de ce dont on a besoin pour les calculs qu'on fait sur les ennemis.

---
calculs ennemis

-> si on les process 1 par 1 pour les sommer : à la fin, on peut multiplier par le masque
-> si on fait un "fixed-size" pour calculer ça: un masque c'est pas idéal ? Il faudrait plutôt un indicateur de padding,
sinon il va croire qu'il y a plein d'ennemis immobiles en 0.

=> à voir si c'est à l'env wrapper d'émettre ce masque ? Via un tensordict c'est pas gênant.

---
séparer en fichiers:
=> des helpers pour convertir tensor <-> env et back
=> composition : différentes manières de représenter ça dans l'env, utiliser des sous classes composables ? (à la torch)
"""

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs import EnvBase

from game.backend.environment import Environment
from game.backend.game_settings import GameSettings
from game.rl_environment.game_tensor_converter import GameTensorConverter


class GameEnv(EnvBase):
    """Environment wrapper for reinforcement learning with torchrl"""

    environments: list[Environment]
    converter: GameTensorConverter

    def __init__(
        self,
        game_settings: GameSettings | None = None,
        device: DEVICE_TYPING = "cpu",
        batch_size: int = 1,
    ) -> None:
        """Instantiates a torchrl environment wrapper for the game environment.

        :param game_settings: Game settings shared by all environments.
        :param device: Device to use for the training (cpu or gpu).
        :param batch_size: Number of parallel environments.
        """
        super().__init__(batch_size=torch.Size((batch_size,)), device=device)

        # Initialize game settings if not provided in order to avoid having all environments build their own
        if game_settings is None:
            game_settings = GameSettings()

        # Initialize all game environments
        self.environments = [Environment(game_settings) for _ in range(batch_size)]
        self.converter = GameTensorConverter(game_settings)

    def _reset(
        self,
        tensordict: TensorDictBase | None = None,
        **kwargs,
    ) -> TensorDictBase:
        """Reset environment and return the initial state.
        This method redefines the abstract method _reset() from EnvBase
        """
        raise NotImplementedError

    def _set_seed(self, seed: int | None):
        """Set seed for the environment."""
        raise NotImplementedError

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Do a step in the environment and return the next state and reward."""

        # TODO: get actions from the input tensordict. Specify format in the class docstring.
        player_observations = torch.zeros(self.batch_size + (5,))

        # Step all environments
        for i, env in enumerate(self.environments):
            env.step([])

            # Convert player observations to tensors
            player_observations[i] = self.converter.player_to_tensor(env.player)

            # TODO: same for ennemies and bullets

        step_output = TensorDict(
            {
                "player_obs": player_observations,
                "enemy_obs": torch.zeros(
                    1, 1
                ),  # TODO: replace with actual enemy observations
                "bullet_obs": torch.zeros(
                    1, 1
                ),  # TODO: replace with actual bullet observations
                "reward": torch.zeros(1, 1),  # TODO: replace with actual reward
                "done": torch.zeros(1, 1),  # TODO: replace with actual done
            },
            batch_size=self.batch_size,
        )

        return step_output.to(self.device)
