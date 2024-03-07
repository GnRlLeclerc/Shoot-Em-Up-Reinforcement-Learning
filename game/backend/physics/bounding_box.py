"""Bounding box utilities"""

import numpy as np
import numpy.typing as npt


class BoundingBox2D:
    """A simple 2D bounding box class implementation"""

    # Bounding box size (width, height)
    size: npt.NDArray[np.float64]
    half_size: npt.NDArray[np.float64]  # Utility (because we use center position)

    # Bounding box center position (center_x, center_y)
    center: npt.NDArray[np.float64]

    def __init__(
        self, width: float, height: float, center_x: float = 0, center_y: float = 0
    ):
        self.size = np.array([width, height], dtype=np.float64)
        self.half_size = self.size * 0.5
        self.center = np.array([center_x, center_y], dtype=np.float64)

    @property
    def width(self) -> float:
        """Returns the box width"""
        return float(self.size[0])

    @property
    def height(self) -> float:
        """Returns the box height"""
        return float(self.size[1])

    @property
    def center_x(self) -> float:
        """Returns the x coordinate of the box center"""
        return float(self.center[0])

    @property
    def center_y(self) -> float:
        """Returns the y coordinate of the box center"""
        return float(self.center[1])

    def collide_box(self, box: "BoundingBox2D") -> bool:
        """Check collision with another box.

        Returns:
        -------
        * True if the boxes collide
        * False if not
        """
        center_delta = np.abs(self.center - box.center)
        total_size = self.half_size + box.half_size

        return np.all(center_delta < total_size)

    def collide_point(self, point: np.ndarray) -> bool:
        """Check if a point is inside the box

        Returns:
        -------
        * True if the point is inside the box
        * False if not
        """
        center_delta = np.abs(self.center - point)
        return np.all(center_delta < self.half_size)

    def clip_inside(self, point: np.ndarray) -> np.ndarray:
        """Clips the point position in order to make it stay inside the box"""
        return np.clip(
            point, self.center - self.half_size, self.center + self.half_size
        )

    def edge_position_from_center_angle(self, angle: float) -> np.ndarray:
        """Returns a point belonging to the box edge with the given angle from the center.
        An angle of 0 means looking to the right, and the angle increases counter-clockwise (trigonometric direction).
        The angle is expected to be in radians.
        """

        # Vector with the correct direction
        vector = self.center + np.array(
            [np.cos(angle), np.sin(angle)], dtype=np.float64
        )

        # Evaluate the minimum factor that makes the vector clip with the boundaries
        multiplier = (self.half_size / np.abs(vector - self.center)).min()

        return vector * multiplier
