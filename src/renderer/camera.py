from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import slangpy as spy
from slangpy import math as smath

from ..common import VEC_EPS, as_float3, normalize3

SPLAT_PIXEL_CLAMP_PX = 0.75


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
    distortion_k1: float | None = None
    distortion_k2: float | None = None
    basis_override: np.ndarray | None = None

    def focal_pixels(self, height: int) -> float:
        return float(self.fy) if self.fy is not None else float(0.5 * float(height) / np.tan(0.5 * np.deg2rad(self.fov_y_degrees)))

    def focal_pixels_xy(self, width: int, height: int) -> tuple[float, float]:
        focal_y = self.focal_pixels(height)
        return float(self.fx) if self.fx is not None else float(focal_y), float(self.fy) if self.fy is not None else float(focal_y)

    def principal_point(self, width: int, height: int) -> tuple[float, float]:
        return float(self.cx) if self.cx is not None else 0.5 * float(width), float(self.cy) if self.cy is not None else 0.5 * float(height)

    def pixel_world_size_max(self, depth: float, width: int, height: int) -> float:
        return float(SPLAT_PIXEL_CLAMP_PX * max(float(depth), 1e-8) / max(min(self.focal_pixels_xy(width, height)), 1e-8))

    @staticmethod
    def look_at(
        position,
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov_y_degrees=60.0,
        near=0.1,
        far=100.0,
        distortion_k1: float | None = None,
        distortion_k2: float | None = None,
    ) -> "Camera":
        return Camera(
            position=np.asarray(position, dtype=np.float32),
            target=np.asarray(target, dtype=np.float32),
            up=np.asarray(up, dtype=np.float32),
            fov_y_degrees=fov_y_degrees,
            near=near,
            far=far,
            distortion_k1=distortion_k1,
            distortion_k2=distortion_k2,
        )

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float32).reshape(3)
        self.target = np.asarray(self.target, dtype=np.float32).reshape(3)
        self.up = np.asarray(normalize3(self.up, eps=VEC_EPS), dtype=np.float32)
        if self.distortion_k1 is not None:
            self.distortion_k1 = float(self.distortion_k1)
        if self.distortion_k2 is not None:
            self.distortion_k2 = float(self.distortion_k2)
        if self.basis_override is not None:
            basis = np.asarray(self.basis_override, dtype=np.float32).reshape(3, 3)
            self.basis_override = basis

    def distortion_coeffs(self, default_k1: float = 0.0, default_k2: float = 0.0) -> tuple[float, float]:
        return (
            float(default_k1 if self.distortion_k1 is None else self.distortion_k1),
            float(default_k2 if self.distortion_k2 is None else self.distortion_k2),
        )

    @staticmethod
    def _distort_normalized(uv: np.ndarray, k1: float, k2: float) -> np.ndarray:
        uv_arr = np.asarray(uv, dtype=np.float32).reshape(2)
        r2 = float(np.dot(uv_arr, uv_arr))
        return uv_arr * np.float32(1.0 + k1 * r2 + k2 * r2 * r2)

    @staticmethod
    def _undistort_normalized(uv_distorted: np.ndarray, k1: float, k2: float, iters: int = 6) -> np.ndarray:
        uv_d = np.asarray(uv_distorted, dtype=np.float32).reshape(2)
        if (not np.isfinite(uv_d).all()) or (abs(k1) <= 1e-12 and abs(k2) <= 1e-12):
            return uv_d
        radius_d = float(np.linalg.norm(uv_d))
        if radius_d <= 1e-12:
            return uv_d
        radius = radius_d
        for _ in range(max(int(iters), 0)):
            r2 = radius * radius
            r4 = r2 * r2
            deriv = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r4
            if abs(deriv) <= 1e-12:
                break
            step = (radius * (1.0 + k1 * r2 + k2 * r4) - radius_d) / deriv
            next_radius = radius - step
            if not np.isfinite(next_radius) or next_radius < 0.0:
                break
            radius = next_radius
        return uv_d * np.float32(radius / max(radius_d, 1e-12))

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.basis_override is not None:
            basis = np.asarray(self.basis_override, dtype=np.float32).reshape(3, 3)
            return tuple(np.asarray(normalize3(axis, eps=VEC_EPS), dtype=np.float32) for axis in basis)
        forward = normalize3(self.target - self.position, eps=VEC_EPS)
        right = normalize3(smath.cross(as_float3(self.up), forward), eps=VEC_EPS)
        up = normalize3(smath.cross(forward, right), eps=VEC_EPS)
        return (
            np.asarray(right, dtype=np.float32),
            np.asarray(up, dtype=np.float32),
            np.asarray(forward, dtype=np.float32),
        )

    def world_to_camera(self, world_vector: np.ndarray) -> np.ndarray:
        return np.asarray(self.basis(), dtype=np.float32) @ np.asarray(world_vector, dtype=np.float32).reshape(3)

    def camera_to_world(self, camera_vector: np.ndarray) -> np.ndarray:
        return np.asarray(self.basis(), dtype=np.float32).T @ np.asarray(camera_vector, dtype=np.float32).reshape(3)

    def world_point_to_camera(self, world_pos: np.ndarray) -> np.ndarray:
        return self.world_to_camera(np.asarray(world_pos, dtype=np.float32).reshape(3) - self.position)

    def camera_point_to_world(self, camera_pos: np.ndarray) -> np.ndarray:
        return self.position + self.camera_to_world(camera_pos)

    def project_camera_to_screen(self, camera_pos: np.ndarray, width: int, height: int, default_k1: float = 0.0, default_k2: float = 0.0) -> tuple[np.ndarray, bool]:
        cam = np.asarray(camera_pos, dtype=np.float32).reshape(3)
        depth = float(cam[2])
        if not np.isfinite(depth) or depth <= 1e-12:
            return np.zeros((2,), dtype=np.float32), False
        fx, fy = self.focal_pixels_xy(width, height)
        cx, cy = self.principal_point(width, height)
        k1, k2 = self.distortion_coeffs(default_k1, default_k2)
        uv = self._distort_normalized(cam[:2] / np.float32(depth), k1, k2)
        screen = uv * np.asarray((fx, fy), dtype=np.float32) + np.asarray((cx, cy), dtype=np.float32)
        return screen.astype(np.float32, copy=False), bool(np.isfinite(screen).all())

    def project_world_to_screen(self, world_pos: np.ndarray, width: int, height: int, default_k1: float = 0.0, default_k2: float = 0.0) -> tuple[np.ndarray, bool]:
        return self.project_camera_to_screen(self.world_point_to_camera(world_pos), width, height, default_k1, default_k2)

    def screen_to_world(self, screen_pos: np.ndarray, depth: float, width: int, height: int, default_k1: float = 0.0, default_k2: float = 0.0) -> np.ndarray:
        fx, fy = self.focal_pixels_xy(width, height)
        cx, cy = self.principal_point(width, height)
        uv = (np.asarray(screen_pos, dtype=np.float32).reshape(2) - np.asarray((cx, cy), dtype=np.float32)) / np.maximum(
            np.asarray((fx, fy), dtype=np.float32),
            np.float32(1e-12),
        )
        undistorted = self._undistort_normalized(uv, *self.distortion_coeffs(default_k1, default_k2))
        depth_safe = max(float(depth), 1e-12)
        return self.camera_point_to_world(np.array([undistorted[0] * depth_safe, undistorted[1] * depth_safe, depth_safe], dtype=np.float32))

    def screen_to_world_ray(self, screen_pos: np.ndarray, width: int, height: int, default_k1: float = 0.0, default_k2: float = 0.0) -> np.ndarray:
        world = self.screen_to_world(screen_pos, 1.0, width, height, default_k1, default_k2)
        return np.asarray(normalize3(world - self.position, eps=VEC_EPS), dtype=np.float32)

    def gpu_params(self, width: int, height: int) -> dict[str, object]:
        right, up, forward = self.basis()
        basis = np.stack((right, up, forward), axis=0).astype(np.float32)
        fx, fy = self.focal_pixels_xy(width, height)
        cx, cy = self.principal_point(width, height)
        return {
            "viewport": spy.float2(float(width), float(height)),
            "camPos": as_float3(self.position),
            "camBasis": spy.float3x3(basis),
            "focalPixels": spy.float2(fx, fy),
            "principalPoint": spy.float2(cx, cy),
            "nearDepth": float(self.near),
            "farDepth": float(self.far),
        }


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
        distortion_k1: float | None = None,
        distortion_k2: float | None = None,
        near: float = 0.1,
        far: float = 100.0,
    ) -> "Camera":
        rot = Camera._rotation_matrix_from_quaternion_wxyz(np.asarray(q_wxyz, dtype=np.float32))
        t = np.asarray(t_xyz, dtype=np.float32).reshape(3)
        cam_pos = (-rot.T @ t.astype(np.float64)).astype(np.float32)
        forward = np.asarray(rot[2], dtype=np.float32)
        up = np.asarray(rot[1], dtype=np.float32)
        target = cam_pos + np.asarray(normalize3(forward, eps=VEC_EPS), dtype=np.float32)
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
            distortion_k1=distortion_k1,
            distortion_k2=distortion_k2,
            basis_override=rot,
        )
