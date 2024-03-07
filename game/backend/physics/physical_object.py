"""Physics related classes and functions"""

import numpy as np
import numpy.typing as npt


class Object2D:
    """A 2D physical object with a velocity"""

    position: npt.NDArray[np.float64]
    velocity: npt.NDArray[np.float64]
    size: float  # Radius

    def __init__(  # pylint: disable=too-many-arguments
        self, x: float = 0, y: float = 0, vx: float = 0, vy: float = 0, size: float = 1
    ):
        self.position = np.array([x, y], dtype=np.float64)
        self.velocity = np.array([vx, vy], dtype=np.float64)
        self.size = size

    @classmethod
    def from_position(cls, position: np.ndarray) -> "Object2D":
        """Creates an object from a position"""
        return cls(float(position[0]), float(position[1]))

    def step(self, dt: float) -> None:
        """Do one simulation step of duration dt."""
        self.position += self.velocity * dt

    def collide(self, other: "Object2D") -> bool:
        """Returns True if the two objects collide."""
        return np.linalg.norm(self.position - other.position) < self.size + other.size
