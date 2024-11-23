import numpy as np

from numpy.typing import NDArray
from typing import Protocol

class Shape(Protocol):
    vertices: NDArray[np.float32]
    indices: NDArray[np.float32]

class Cube(Shape):
    def __init__(
        self, 
        translation: NDArray[np.float32] = np.zeros(3, dtype=np.float32),
        rotation: NDArray[np.float32] = np.eye(3, dtype=np.float32),
        alpha: float = 0.5
    ):
        positions = np.array([
            [x, y, z] 
            for x in {-0.5, 0.5}
            for y in {-0.5, 0.5}
            for z in {-0.5, 0.5}
        ], dtype=np.float32)

        positions = (positions @ rotation.T) + translation


        color = np.array([
            [1 - 0.05 * i, 1 - 0.05 * i, 0.05, alpha] for i in range(len(positions))
        ], dtype=np.float32)

        self.vertices = np.hstack((positions, color))

        self.indices = np.array([
            # Front Face
            [0, 1, 3],
            [0, 3, 2],
            # Back Face
            [4, 6, 7],
            [4, 7, 5],
            # Left Face
            [0, 2, 6],
            [0, 6, 4],
            # Right Face
            [1, 5, 7],
            [1, 7, 3],
            # Bottom Face
            [0, 4, 5],
            [0, 5, 1],
            # Top Face
            [2, 3, 7],
            [2, 7, 6]
        ], dtype=np.uint32)

        self.veretex_count = len(self.vertices)
        self.index_count = len(self.indices.flatten())

class Plane:
    def __init__(
        self,
        origin: NDArray[np.float32],
        u: NDArray[np.float32],
        v: NDArray[np.float32],
        dim_u: float = 1,
        dim_v: float = 1
    ):
        pass