from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import slangpy as spy
from slangpy import math as smath

from ..utility import VEC_EPS, as_float3, normalize3

SPLAT_PIXEL_CLAMP_PX = 0.75
_DISTORTION_COEFF_COUNT = 8
_DISTORTION_EPS = 1e-12
_DISTORTION_NEWTON_ITERS = 8


@dataclass(slots=True)
class Camera:
    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_y_degrees: float = 60.0
    near: float = 0.1
    far: float = 100.0
    min_camera_distance: float = 0.0
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    distortion_k1: float | None = None
    distortion_k2: float | None = None
    distortion_p1: float | None = None
    distortion_p2: float | None = None
    distortion_k3: float | None = None
    distortion_k4: float | None = None
    distortion_k5: float | None = None
    distortion_k6: float | None = None
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
        distortion_p1: float | None = None,
        distortion_p2: float | None = None,
        distortion_k3: float | None = None,
        distortion_k4: float | None = None,
        distortion_k5: float | None = None,
        distortion_k6: float | None = None,
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
            distortion_p1=distortion_p1,
            distortion_p2=distortion_p2,
            distortion_k3=distortion_k3,
            distortion_k4=distortion_k4,
            distortion_k5=distortion_k5,
            distortion_k6=distortion_k6,
        )

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float32).reshape(3)
        self.target = np.asarray(self.target, dtype=np.float32).reshape(3)
        self.up = np.asarray(normalize3(self.up, eps=VEC_EPS), dtype=np.float32)
        for attr in ("distortion_k1", "distortion_k2", "distortion_p1", "distortion_p2", "distortion_k3", "distortion_k4", "distortion_k5", "distortion_k6"):
            value = getattr(self, attr)
            if value is not None:
                setattr(self, attr, float(value))
        self.min_camera_distance = max(float(self.min_camera_distance), 0.0)
        if self.basis_override is not None:
            basis = np.asarray(self.basis_override, dtype=np.float32).reshape(3, 3)
            self.basis_override = basis

    @staticmethod
    def _resolve_distortion_defaults(defaults: tuple[float, ...] | None = None) -> tuple[float, float, float, float, float, float, float, float]:
        if defaults is None:
            return (0.0,) * _DISTORTION_COEFF_COUNT
        values = tuple(float(value) for value in defaults)
        if len(values) != _DISTORTION_COEFF_COUNT:
            raise ValueError(f"Expected {_DISTORTION_COEFF_COUNT} distortion coefficients, got {len(values)}.")
        return values

    def distortion_params(self, defaults: tuple[float, ...] | None = None) -> tuple[float, float, float, float, float, float, float, float]:
        resolved = self._resolve_distortion_defaults(defaults)
        return (
            float(resolved[0] if self.distortion_k1 is None else self.distortion_k1),
            float(resolved[1] if self.distortion_k2 is None else self.distortion_k2),
            float(resolved[2] if self.distortion_p1 is None else self.distortion_p1),
            float(resolved[3] if self.distortion_p2 is None else self.distortion_p2),
            float(resolved[4] if self.distortion_k3 is None else self.distortion_k3),
            float(resolved[5] if self.distortion_k4 is None else self.distortion_k4),
            float(resolved[6] if self.distortion_k5 is None else self.distortion_k5),
            float(resolved[7] if self.distortion_k6 is None else self.distortion_k6),
        )

    def distortion_coeffs(self, default_k1: float = 0.0, default_k2: float = 0.0) -> tuple[float, float]:
        k1, k2, *_ = self.distortion_params((default_k1, default_k2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        return k1, k2

    @staticmethod
    def _safe_denominator(value: float) -> float:
        return value if abs(value) > _DISTORTION_EPS else float(np.copysign(_DISTORTION_EPS, value if value != 0.0 else 1.0))

    @staticmethod
    def _radial_distortion(r2: float, k1: float, k2: float, k3: float, k4: float, k5: float, k6: float) -> tuple[float, float]:
        r4 = r2 * r2
        r6 = r4 * r2
        numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        denominator = Camera._safe_denominator(1.0 + k4 * r2 + k5 * r4 + k6 * r6)
        d_numerator = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4
        d_denominator = k4 + 2.0 * k5 * r2 + 3.0 * k6 * r4
        radial = numerator / denominator
        d_radial_dr2 = (d_numerator * denominator - numerator * d_denominator) / (denominator * denominator)
        return radial, d_radial_dr2

    @staticmethod
    def _tangential_distortion(x: float, y: float, r2: float, p1: float, p2: float) -> tuple[float, float]:
        return 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x), p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    @staticmethod
    def _distort_normalized_with_params(uv: np.ndarray, params: tuple[float, float, float, float, float, float, float, float]) -> np.ndarray:
        x, y = np.asarray(uv, dtype=np.float64).reshape(2)
        k1, k2, p1, p2, k3, k4, k5, k6 = (float(value) for value in params)
        r2 = x * x + y * y
        radial, _ = Camera._radial_distortion(r2, k1, k2, k3, k4, k5, k6)
        tx, ty = Camera._tangential_distortion(x, y, r2, p1, p2)
        return np.array((x * radial + tx, y * radial + ty), dtype=np.float64)

    @staticmethod
    def _distort_normalized(uv: np.ndarray, k1: float, k2: float, p1: float = 0.0, p2: float = 0.0, k3: float = 0.0, k4: float = 0.0, k5: float = 0.0, k6: float = 0.0) -> np.ndarray:
        return Camera._distort_normalized_with_params(uv, (k1, k2, p1, p2, k3, k4, k5, k6)).astype(np.float32, copy=False)

    @staticmethod
    def _distortion_jacobian(uv: np.ndarray, params: tuple[float, float, float, float, float, float, float, float]) -> np.ndarray:
        x, y = np.asarray(uv, dtype=np.float64).reshape(2)
        k1, k2, p1, p2, k3, k4, k5, k6 = (float(value) for value in params)
        r2 = x * x + y * y
        radial, d_radial_dr2 = Camera._radial_distortion(r2, k1, k2, k3, k4, k5, k6)
        d_radial_dx = d_radial_dr2 * 2.0 * x
        d_radial_dy = d_radial_dr2 * 2.0 * y
        return np.array(
            (
                (radial + x * d_radial_dx + 2.0 * p1 * y + 6.0 * p2 * x, x * d_radial_dy + 2.0 * p1 * x + 2.0 * p2 * y),
                (y * d_radial_dx + 2.0 * p1 * x + 2.0 * p2 * y, radial + y * d_radial_dy + 6.0 * p1 * y + 2.0 * p2 * x),
            ),
            dtype=np.float64,
        )

    @staticmethod
    def _undistort_normalized(uv_distorted: np.ndarray, k1: float, k2: float, p1: float = 0.0, p2: float = 0.0, k3: float = 0.0, k4: float = 0.0, k5: float = 0.0, k6: float = 0.0, iters: int = _DISTORTION_NEWTON_ITERS) -> np.ndarray:
        uv_d = np.asarray(uv_distorted, dtype=np.float64).reshape(2)
        params = (float(k1), float(k2), float(p1), float(p2), float(k3), float(k4), float(k5), float(k6))
        if (not np.isfinite(uv_d).all()) or all(abs(value) <= _DISTORTION_EPS for value in params):
            return uv_d.astype(np.float32, copy=False)
        estimate = np.array(uv_d, dtype=np.float64, copy=True)
        for _ in range(max(int(iters), 0)):
            error = Camera._distort_normalized_with_params(estimate, params) - uv_d
            if float(np.dot(error, error)) <= _DISTORTION_EPS * _DISTORTION_EPS:
                break
            jacobian = Camera._distortion_jacobian(estimate, params)
            det = float(jacobian[0, 0] * jacobian[1, 1] - jacobian[0, 1] * jacobian[1, 0])
            if (not np.isfinite(det)) or abs(det) <= _DISTORTION_EPS:
                break
            step = np.array(
                (
                    (jacobian[1, 1] * error[0] - jacobian[0, 1] * error[1]) / det,
                    (-jacobian[1, 0] * error[0] + jacobian[0, 0] * error[1]) / det,
                ),
                dtype=np.float64,
            )
            next_estimate = estimate - step
            if not np.isfinite(next_estimate).all():
                break
            estimate = next_estimate
        return estimate.astype(np.float32, copy=False)

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
        uv = self._distort_normalized(cam[:2] / np.float32(depth), *self.distortion_params((default_k1, default_k2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
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
        undistorted = self._undistort_normalized(uv, *self.distortion_params((default_k1, default_k2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
        depth_safe = max(float(depth), 1e-12)
        return self.camera_point_to_world(np.array([undistorted[0] * depth_safe, undistorted[1] * depth_safe, depth_safe], dtype=np.float32))

    def screen_to_world_ray(self, screen_pos: np.ndarray, width: int, height: int, default_k1: float = 0.0, default_k2: float = 0.0) -> np.ndarray:
        world = self.screen_to_world(screen_pos, 1.0, width, height, default_k1, default_k2)
        return np.asarray(normalize3(world - self.position, eps=VEC_EPS), dtype=np.float32)

    def gpu_params(self, width: int, height: int, default_distortion: tuple[float, ...] | None = None) -> dict[str, object]:
        right, up, forward = self.basis()
        basis = np.stack((right, up, forward), axis=0).astype(np.float32)
        fx, fy = self.focal_pixels_xy(width, height)
        cx, cy = self.principal_point(width, height)
        distortion = self.distortion_params(default_distortion)
        return {
            "viewport": spy.float2(float(width), float(height)),
            "camPos": as_float3(self.position),
            "camBasis": spy.float3x3(basis),
            "focalPixels": spy.float2(fx, fy),
            "principalPoint": spy.float2(cx, cy),
            "nearDepth": float(self.near),
            "farDepth": float(self.far),
            "projDistortionK1K2P1P2": spy.float4(*distortion[:4]),
            "projDistortionK3K4K5K6": spy.float4(*distortion[4:]),
            "minCameraDistance": float(self.min_camera_distance),
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
        distortion_p1: float | None = None,
        distortion_p2: float | None = None,
        distortion_k3: float | None = None,
        distortion_k4: float | None = None,
        distortion_k5: float | None = None,
        distortion_k6: float | None = None,
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
            distortion_p1=distortion_p1,
            distortion_p2=distortion_p2,
            distortion_k3=distortion_k3,
            distortion_k4=distortion_k4,
            distortion_k5=distortion_k5,
            distortion_k6=distortion_k6,
            basis_override=rot,
        )
