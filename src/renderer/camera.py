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
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    basis_override: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float32)
        self.target = np.asarray(self.target, dtype=np.float32)
        self.up = _normalize(self.up)
        if self.basis_override is not None:
            basis = np.asarray(self.basis_override, dtype=np.float32).reshape(3, 3)
            self.basis_override = basis

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.basis_override is not None:
            basis = np.asarray(self.basis_override, dtype=np.float32).reshape(3, 3)
            return _normalize(basis[0]), _normalize(basis[1]), _normalize(basis[2])
        forward = _normalize(self.target - self.position)
        right = _normalize(np.cross(self.up, forward))
        up = _normalize(np.cross(forward, right))
        return right, up, forward

    def focal_pixels(self, height: int) -> float:
        if self.fy is not None:
            return float(self.fy)
        fov = np.deg2rad(self.fov_y_degrees)
        return float(0.5 * float(height) / np.tan(0.5 * fov))

    def focal_pixels_xy(self, width: int, height: int) -> tuple[float, float]:
        fx = float(self.fx) if self.fx is not None else float(self.focal_pixels(height))
        fy = float(self.fy) if self.fy is not None else float(self.focal_pixels(height))
        return fx, fy

    def principal_point(self, width: int, height: int) -> tuple[float, float]:
        cx = float(self.cx) if self.cx is not None else 0.5 * float(width)
        cy = float(self.cy) if self.cy is not None else 0.5 * float(height)
        return cx, cy

    def gpu_params(self, width: int, height: int) -> dict[str, object]:
        right, up, forward = self.basis()
        basis = np.stack((right, up, forward), axis=0).astype(np.float32)
        fx, fy = self.focal_pixels_xy(width, height)
        cx, cy = self.principal_point(width, height)
        return {
            "viewport": spy.float2(float(width), float(height)),
            "camPos": spy.float3(*self.position.tolist()),
            "camBasis": spy.float3x3(basis),
            "focalPixels": spy.float2(fx, fy),
            "principalPoint": spy.float2(cx, cy),
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

    @staticmethod
    def _rotation_matrix_from_quaternion_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
        q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
        q = q / np.maximum(np.linalg.norm(q), 1e-12)
        w = float(q[0])
        x = float(q[1])
        y = float(q[2])
        z = float(q[3])
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
                [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def from_colmap(
        q_wxyz: tuple[float, float, float, float] | np.ndarray,
        t_xyz: tuple[float, float, float] | np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        near: float = 0.1,
        far: float = 100.0,
    ) -> "Camera":
        rot = Camera._rotation_matrix_from_quaternion_wxyz(np.asarray(q_wxyz, dtype=np.float32))
        t = np.asarray(t_xyz, dtype=np.float32).reshape(3)
        cam_pos = (-rot.T @ t.astype(np.float64)).astype(np.float32)
        forward = rot[2]
        up = rot[1]
        target = cam_pos + _normalize(forward)
        return Camera(
            position=cam_pos,
            target=target,
            up=up,
            near=near,
            far=far,
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            basis_override=rot,
        )
