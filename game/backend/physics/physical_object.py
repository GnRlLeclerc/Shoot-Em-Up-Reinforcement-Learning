"""Physics related classes and functions"""

import numpy as np
import numpy.typing as npt


class Object2D:
    """A 2D physical object with a velocity"""

    position: npt.NDArray[np.float64]
    velocity: npt.NDArray[np.float64]

    def __init__(self, x: float = 0, y: float = 0, vx: float = 0, vy: float = 0):
        self.position = np.array([x, y], dtype=np.float64)
        self.velocity = np.array([vx, vy], dtype=np.float64)

    def step(self, dt: float) -> None:
        """Do one simulation step of duration dt."""
        self.position += self.velocity * dt
