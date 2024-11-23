import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field

@dataclass
class Camera:
    position: NDArray[np.float32] = field(default_factory=lambda: np.array([0.0, 0.0, 3.0], dtype=np.float32))
    front: NDArray[np.float32] = field(default_factory=lambda: np.array([0.0, 0.0, -1.0], dtype=np.float32))
    up: NDArray[np.float32] = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32))
    right: NDArray[np.float32] = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32))
    yaw: float = -90.0
    pitch: float = 0.0
    sensitivity: float = 0.1
    zoom_speed: float = 0.1
    min_zoom: float = 0.1
    max_zoom: float = 10.0
    fov: float = np.radians(45.0)
    near: float = 0.1
    far: float = 100.0

    def rotate(self, delta_x: float, delta_y: float) -> None:
        self.yaw += delta_x * self.sensitivity
        self.pitch += delta_y * self.sensitivity
        self.pitch = np.clip(self.pitch, -89.0, 89.0)
        self._update_vectors()

    def zoom(self, delta: float) -> None:
        zoom_amount = delta * self.zoom_speed
        new_z = self.position[2] - zoom_amount
        self.position[2] = np.clip(new_z, self.min_zoom, self.max_zoom)

    def _update_vectors(self) -> None:
        front = np.array([
            np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)),
            np.sin(np.radians(self.pitch)),
            np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        ], dtype=np.float32)
        
        self.front = front / np.linalg.norm(front)
        self.right = np.cross(self.front, np.array([0.0, 1.0, 0.0]))
        self.right = self.right / np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.front)
        self.up = self.up / np.linalg.norm(self.up)

    @property
    def view_matrix(self) -> NDArray[np.float32]:
        z_axis = -self.front / np.linalg.norm(self.front)
        x_axis = np.cross(self.up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        rotation = np.array([
            [x_axis[0], y_axis[0], z_axis[0], 0],
            [x_axis[1], y_axis[1], z_axis[1], 0],
            [x_axis[2], y_axis[2], z_axis[2], 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        translation = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-self.position[0], -self.position[1], -self.position[2], 1]
        ], dtype=np.float32)
        
        return rotation @ translation

    def get_projection_matrix(self, width: float, height: float) -> NDArray[np.float32]:
        aspect_ratio = width / height if height != 0 else 1.0
        
        fov = self.fov
        near = self.near
        far = self.far
        
        f = 1.0 / np.tan(fov / 2.0)
        projection = np.array([
            [f/aspect_ratio, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, -(far+near)/(far-near), -1.0],
            [0.0, 0.0, -(2*far*near)/(far-near), 0.0]
        ], dtype=np.float32)

        return projection
