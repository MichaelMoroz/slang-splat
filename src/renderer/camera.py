from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import slangpy as spy


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v), 1e-8)


@dataclass(slots=True)
class Camera:
    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_y_degrees: float = 60.0
    near: float = 0.1
    far: float = 100.0

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float32)
        self.target = np.asarray(self.target, dtype=np.float32)
        self.up = _normalize(self.up)

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = _normalize(self.target - self.position)
        right = _normalize(np.cross(self.up, forward))
        up = _normalize(np.cross(forward, right))
        return right, up, forward

    def focal_pixels(self, height: int) -> float:
        fov = np.deg2rad(self.fov_y_degrees)
        return float(0.5 * float(height) / np.tan(0.5 * fov))

    def gpu_params(self, width: int, height: int) -> dict[str, object]:
        right, up, forward = self.basis()
        basis = np.stack((right, up, forward), axis=0).astype(np.float32)
        return {
            "viewport": spy.float2(float(width), float(height)),
            "camPos": spy.float3(*self.position.tolist()),
            "camBasis": spy.float3x3(basis),
            "focalPixels": float(self.focal_pixels(height)),
            "nearDepth": float(self.near),
            "farDepth": float(self.far),
        }

    @staticmethod
    def look_at(
        position: tuple[float, float, float] | np.ndarray,
        target: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
        up: tuple[float, float, float] | np.ndarray = (0.0, 1.0, 0.0),
        fov_y_degrees: float = 60.0,
        near: float = 0.1,
        far: float = 100.0,
    ) -> "Camera":
        return Camera(
            position=np.asarray(position, dtype=np.float32),
            target=np.asarray(target, dtype=np.float32),
            up=np.asarray(up, dtype=np.float32),
            fov_y_degrees=fov_y_degrees,
            near=near,
            far=far,
        )
