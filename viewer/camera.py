from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import slangpy as spy

_EPS = 1e-8


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return (v / max(norm, _EPS)).astype(np.float32)


@dataclass(slots=True)
class Camera:
    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_y_degrees: float = 60.0
    near: float = 0.0
    far: float = 1000.0
    distortion_k1: float = 0.0
    distortion_k2: float = 0.0

    @staticmethod
    def look_at(
        position: tuple[float, float, float] | np.ndarray,
        target: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
        up: tuple[float, float, float] | np.ndarray = (0.0, 1.0, 0.0),
        fov_y_degrees: float = 60.0,
        near: float = 0.0,
        far: float = 1000.0,
        distortion_k1: float = 0.0,
        distortion_k2: float = 0.0,
    ) -> "Camera":
        return Camera(
            position=np.asarray(position, dtype=np.float32),
            target=np.asarray(target, dtype=np.float32),
            up=np.asarray(up, dtype=np.float32),
            fov_y_degrees=float(fov_y_degrees),
            near=float(near),
            far=float(far),
            distortion_k1=float(distortion_k1),
            distortion_k2=float(distortion_k2),
        )

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = _normalize(self.target - self.position)
        right = _normalize(np.cross(self.up, forward))
        up = _normalize(np.cross(forward, right))
        return right, up, forward

    def gpu_params(self, width: int, height: int) -> dict[str, object]:
        right, up, forward = self.basis()
        basis = np.stack((right, up, forward), axis=0).astype(np.float32)
        focal = 0.5 * float(height) / max(np.tan(0.5 * np.deg2rad(self.fov_y_degrees)), _EPS)
        return {
            "viewport": spy.float2(float(width), float(height)),
            "camPos": spy.float3(*self.position.tolist()),
            "camBasis": spy.float3x3(basis),
            "focalPixels": spy.float2(float(focal), float(focal)),
            "principalPoint": spy.float2(0.5 * float(width), 0.5 * float(height)),
            "nearDepth": float(self.near),
            "farDepth": float(self.far),
            "k1": float(self.distortion_k1),
            "k2": float(self.distortion_k2),
        }
