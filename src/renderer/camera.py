from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import slangpy as spy
from slangpy import math as smath

from ..utility import VEC_EPS, as_float3, normalize3

SPLAT_PIXEL_CLAMP_PX = 0.75
PROJECTION_MODEL_PINHOLE = 0
PROJECTION_MODEL_EQUIRECTANGULAR = 1
PROJECTION_MODEL_FISHEYE = 2
PROJECTION_MODEL_VALUES = (PROJECTION_MODEL_PINHOLE, PROJECTION_MODEL_EQUIRECTANGULAR, PROJECTION_MODEL_FISHEYE)
_DISTORTION_COEFF_COUNT = 8
_DISTORTION_EPS = 1e-12
_DISTORTION_NEWTON_ITERS = 8
_EQUIRECTANGULAR_EPS = 1e-12
_FISHEYE_EPS = 1e-12


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
    projection_model: int = PROJECTION_MODEL_PINHOLE

    def focal_pixels(self, height: int) -> float:
        return float(self.fy) if self.fy is not None else float(0.5 * float(height) / np.tan(0.5 * np.deg2rad(self.fov_y_degrees)))

    def focal_pixels_xy(self, width: int, height: int) -> tuple[float, float]:
        focal_y = self.focal_pixels(height)
        return float(self.fx) if self.fx is not None else float(focal_y), float(self.fy) if self.fy is not None else float(focal_y)

    def principal_point(self, width: int, height: int) -> tuple[float, float]:
        return float(self.cx) if self.cx is not None else 0.5 * float(width), float(self.cy) if self.cy is not None else 0.5 * float(height)

    def cropped(self, crop_x: float, crop_y: float, native_width: int, native_height: int) -> "Camera":
        """Camera for a native-resolution crop window: same pose and focal, shifted
        principal point. Exact for every projection model — pinhole and distorted
        cameras keep distortion valid because it is defined around the principal point,
        and equirect keeps its longitude period at fx*2pi (the native width) so seam
        wrapping still resolves correctly inside the crop viewport.
        """
        from dataclasses import replace

        if self.is_equirectangular:
            fx, fy, cx, cy = self._equirect_intrinsics(native_width, native_height)
        else:
            fx, fy = self.focal_pixels_xy(native_width, native_height)
            cx, cy = self.principal_point(native_width, native_height)
        return replace(self, fx=float(fx), fy=float(fy), cx=float(cx) - float(crop_x), cy=float(cy) - float(crop_y))

    def pixel_world_size_max(self, depth: float, width: int, height: int) -> float:
        if self.is_angular:
            fx, fy = self._equirect_intrinsics(width, height)[:2] if self.is_equirectangular else self.focal_pixels_xy(width, height)
            angular_pixel_size = max(1.0 / max(fx, 1e-8), 1.0 / max(fy, 1e-8))
            return float(SPLAT_PIXEL_CLAMP_PX * max(float(depth), 1e-8) * angular_pixel_size)
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
        projection_model: int = PROJECTION_MODEL_PINHOLE,
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
            projection_model=projection_model,
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
        self.projection_model = int(self.projection_model)
        if self.projection_model not in PROJECTION_MODEL_VALUES:
            raise ValueError(f"Unsupported projection model {self.projection_model}.")
        if self.basis_override is not None:
            basis = np.asarray(self.basis_override, dtype=np.float32).reshape(3, 3)
            self.basis_override = basis

    @property
    def is_equirectangular(self) -> bool:
        return int(self.projection_model) == PROJECTION_MODEL_EQUIRECTANGULAR

    @property
    def is_fisheye(self) -> bool:
        return int(self.projection_model) == PROJECTION_MODEL_FISHEYE

    @property
    def is_angular(self) -> bool:
        """Equirectangular and fisheye share angular-projection semantics: depth is the
        radial camera distance and visibility is not bounded by pinhole frustum planes."""
        return self.is_equirectangular or self.is_fisheye

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

    def _fisheye_coefficients(self) -> tuple[float, float, float, float]:
        """KB4 theta-polynomial coefficients (COLMAP OPENCV_FISHEYE), riding in the
        standard k1/k2/k3/k4 distortion fields; p1/p2/k5/k6 are unused for fisheye."""
        k1, k2, _, _, k3, k4, _, _ = self.distortion_params()
        return k1, k2, k3, k4

    @staticmethod
    def _fisheye_distort_theta(theta: float, coeffs: tuple[float, float, float, float]) -> tuple[float, float]:
        k1, k2, k3, k4 = (float(value) for value in coeffs)
        t2 = float(theta) * float(theta)
        theta_d = float(theta) * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        derivative = 1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * (9.0 * k4))))
        return theta_d, derivative

    @staticmethod
    def _fisheye_theta_from_distorted(theta_d: float, coeffs: tuple[float, float, float, float], iters: int = _DISTORTION_NEWTON_ITERS) -> float:
        estimate = float(np.clip(theta_d, 0.0, np.pi))
        if all(abs(float(value)) <= _FISHEYE_EPS for value in coeffs):
            return estimate
        for _ in range(max(int(iters), 0)):
            distorted, derivative = Camera._fisheye_distort_theta(estimate, coeffs)
            error = distorted - float(theta_d)
            if abs(error) <= _FISHEYE_EPS:
                break
            if (not np.isfinite(derivative)) or abs(derivative) <= _FISHEYE_EPS:
                break
            next_estimate = float(np.clip(estimate - error / derivative, 0.0, np.pi))
            if not np.isfinite(next_estimate):
                break
            estimate = next_estimate
        return estimate

    def _project_fisheye_camera_to_screen(self, camera_pos: np.ndarray, width: int, height: int) -> tuple[np.ndarray, bool]:
        cam = np.asarray(camera_pos, dtype=np.float64).reshape(3)
        axial = float(np.linalg.norm(cam[:2]))
        if (not np.isfinite(cam).all()) or float(np.linalg.norm(cam)) <= _FISHEYE_EPS:
            return np.zeros((2,), dtype=np.float32), False
        # Directly behind the optical axis the azimuth is undefined (the fisheye analog
        # of the equirect pole).
        if axial <= _FISHEYE_EPS and float(cam[2]) < 0.0:
            return np.zeros((2,), dtype=np.float32), False
        theta = float(np.arctan2(axial, float(cam[2])))
        theta_d, _ = self._fisheye_distort_theta(theta, self._fisheye_coefficients())
        direction = cam[:2] / axial if axial > _FISHEYE_EPS else np.zeros((2,), dtype=np.float64)
        fx, fy = self.focal_pixels_xy(width, height)
        cx, cy = self.principal_point(width, height)
        screen = direction * theta_d * np.array((fx, fy), dtype=np.float64) + np.array((cx, cy), dtype=np.float64)
        return screen.astype(np.float32, copy=False), bool(np.isfinite(screen).all())

    def _fisheye_screen_to_world_ray(self, screen_pos: np.ndarray, width: int, height: int) -> np.ndarray:
        screen = np.asarray(screen_pos, dtype=np.float64).reshape(2)
        fx, fy = self.focal_pixels_xy(width, height)
        cx, cy = self.principal_point(width, height)
        uv = (screen - np.array((cx, cy), dtype=np.float64)) / np.maximum(np.array((fx, fy), dtype=np.float64), _FISHEYE_EPS)
        radial = float(np.linalg.norm(uv))
        theta = self._fisheye_theta_from_distorted(radial, self._fisheye_coefficients())
        direction = uv / radial if radial > _FISHEYE_EPS else np.zeros((2,), dtype=np.float64)
        ray_camera = np.array((direction[0] * np.sin(theta), direction[1] * np.sin(theta), np.cos(theta)), dtype=np.float32)
        return np.asarray(normalize3(self.camera_to_world(ray_camera), eps=VEC_EPS), dtype=np.float32)

    def project_camera_to_screen(self, camera_pos: np.ndarray, width: int, height: int, default_k1: float = 0.0, default_k2: float = 0.0) -> tuple[np.ndarray, bool]:
        cam = np.asarray(camera_pos, dtype=np.float32).reshape(3)
        if self.is_equirectangular:
            return self._project_equirectangular_camera_to_screen(cam, width, height)
        if self.is_fisheye:
            return self._project_fisheye_camera_to_screen(cam, width, height)
        depth = float(cam[2])
        if not np.isfinite(depth) or depth <= 1e-12:
            return np.zeros((2,), dtype=np.float32), False
        fx, fy = self.focal_pixels_xy(width, height)
        cx, cy = self.principal_point(width, height)
        uv = self._distort_normalized(cam[:2] / np.float32(depth), *self.distortion_params((default_k1, default_k2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
        screen = uv * np.asarray((fx, fy), dtype=np.float32) + np.asarray((cx, cy), dtype=np.float32)
        return screen.astype(np.float32, copy=False), bool(np.isfinite(screen).all())

    def _equirect_intrinsics(self, width: int, height: int) -> tuple[float, float, float, float]:
        """COLMAP model-17 intrinsics with full-sphere fallbacks (mirrors camera.slang).

        Legacy equirect cameras carry zeroed intrinsics and map the viewport to the full
        sphere; crop cameras set explicit native-scale intrinsics with a shifted
        principal point, making the longitude period fx*2pi instead of the viewport.
        """
        width_safe = max(float(width), 1.0)
        height_safe = max(float(height), 1.0)
        fx_set = self.fx is not None and float(self.fx) > 1e-8
        fy_set = self.fy is not None and float(self.fy) > 1e-8
        fx = float(self.fx) if fx_set else width_safe / (2.0 * np.pi)
        fy = float(self.fy) if fy_set else height_safe / np.pi
        has_intrinsics = fx_set or fy_set
        cx = float(self.cx) if has_intrinsics and self.cx is not None else 0.5 * width_safe
        cy = float(self.cy) if has_intrinsics and self.cy is not None else 0.5 * height_safe
        return fx, fy, cx, cy

    def _project_equirectangular_camera_to_screen(self, camera_pos: np.ndarray, width: int, height: int) -> tuple[np.ndarray, bool]:
        cam = np.asarray(camera_pos, dtype=np.float64).reshape(3)
        horizontal = float(np.linalg.norm(cam[[0, 2]]))
        if (not np.isfinite(cam).all()) or horizontal + abs(float(cam[1])) <= _EQUIRECTANGULAR_EPS:
            return np.zeros((2,), dtype=np.float32), False
        theta = float(np.arctan2(float(cam[0]), float(cam[2])))
        phi = float(np.arctan2(-float(cam[1]), horizontal))
        fx, fy, cx, cy = self._equirect_intrinsics(width, height)
        screen = np.array((theta * fx + cx, -phi * fy + cy), dtype=np.float64)
        period = max(fx * 2.0 * np.pi, 1e-8)
        viewport_center = 0.5 * max(float(width), 1.0)
        screen[0] -= np.floor((screen[0] - viewport_center) / period + 0.5) * period
        return screen.astype(np.float32, copy=False), bool(np.isfinite(screen).all())

    def project_world_to_screen(self, world_pos: np.ndarray, width: int, height: int, default_k1: float = 0.0, default_k2: float = 0.0) -> tuple[np.ndarray, bool]:
        return self.project_camera_to_screen(self.world_point_to_camera(world_pos), width, height, default_k1, default_k2)

    def screen_to_world(self, screen_pos: np.ndarray, depth: float, width: int, height: int, default_k1: float = 0.0, default_k2: float = 0.0) -> np.ndarray:
        if self.is_angular:
            return self.position + self.screen_to_world_ray(screen_pos, width, height) * np.float32(max(float(depth), 0.0))
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
        if self.is_equirectangular:
            return self._equirectangular_screen_to_world_ray(screen_pos, width, height)
        if self.is_fisheye:
            return self._fisheye_screen_to_world_ray(screen_pos, width, height)
        world = self.screen_to_world(screen_pos, 1.0, width, height, default_k1, default_k2)
        return np.asarray(normalize3(world - self.position, eps=VEC_EPS), dtype=np.float32)

    def _equirectangular_screen_to_world_ray(self, screen_pos: np.ndarray, width: int, height: int) -> np.ndarray:
        screen = np.asarray(screen_pos, dtype=np.float64).reshape(2)
        fx, fy, cx, cy = self._equirect_intrinsics(width, height)
        theta = (float(screen[0]) - cx) / max(fx, 1e-12)
        phi = -(float(screen[1]) - cy) / max(fy, 1e-12)
        cos_phi = float(np.cos(phi))
        ray_camera = np.array((cos_phi * np.sin(theta), -np.sin(phi), cos_phi * np.cos(theta)), dtype=np.float32)
        return np.asarray(normalize3(self.camera_to_world(ray_camera), eps=VEC_EPS), dtype=np.float32)

    def gpu_params(self, width: int, height: int, default_distortion: tuple[float, ...] | None = None) -> dict[str, object]:
        right, up, forward = self.basis()
        basis = np.stack((right, up, forward), axis=0).astype(np.float32)
        if self.is_equirectangular:
            # Equirect cameras must ship model-17 intrinsics (x = fx*theta + cx): legacy
            # cameras without explicit intrinsics get the full-sphere mapping, crop
            # cameras keep their native-scale focal and shifted principal point. The
            # pinhole fov-derived focal fallback would be meaningless here.
            fx, fy, cx, cy = self._equirect_intrinsics(width, height)
        else:
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
            "projectionModel": np.uint32(self.projection_model),
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
        projection_model: int = PROJECTION_MODEL_PINHOLE,
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
            projection_model=projection_model,
        )
