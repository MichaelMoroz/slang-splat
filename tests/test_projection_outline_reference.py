from __future__ import annotations

import math

import numpy as np

from reference_impls.reference_cpu import project_splats
from src.renderer import Camera
from src.scene import GaussianScene

_ALPHA_CUTOFF = 1.0 / 255.0


def make_scene(count: int, seed: int = 123) -> GaussianScene:
    rng = np.random.default_rng(seed)
    positions = np.zeros((count, 3), dtype=np.float32)
    positions[:, 0] = rng.uniform(-1.5, 1.5, size=count).astype(np.float32)
    positions[:, 1] = rng.uniform(-1.0, 1.0, size=count).astype(np.float32)
    positions[:, 2] = rng.uniform(-1.5, 1.5, size=count).astype(np.float32)
    scales = rng.uniform(-3.0, -0.6, size=(count, 3)).astype(np.float32)
    rotations = rng.normal(size=(count, 4)).astype(np.float32)
    rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-8)
    opacities = rng.uniform(0.1, 0.9, size=count).astype(np.float32)
    colors = rng.uniform(0.0, 1.0, size=(count, 3)).astype(np.float32)
    sh_coeffs = np.zeros((count, 1, 3), dtype=np.float32)
    return GaussianScene(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
        sh_coeffs=sh_coeffs,
    )


def _quat_rotate(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    qv = q[1:]
    cross1 = np.cross(v, qv)
    cross2 = np.cross(cross1 + q[0] * v, qv)
    return v + 2.0 * cross2


def _outline_screen_point(center: np.ndarray, conic: np.ndarray, theta: float) -> np.ndarray:
    direction = np.array((math.cos(theta), math.sin(theta)), dtype=np.float32)
    denom = conic[0] * direction[0] * direction[0] + 2.0 * conic[1] * direction[0] * direction[1] + conic[2] * direction[1] * direction[1]
    return center + direction / np.float32(math.sqrt(max(float(denom), 1e-12)))


def _pick_tangent_axis(direction: np.ndarray) -> np.ndarray:
    direction_arr = np.asarray(direction, dtype=np.float32).reshape(3)
    return np.array((0.0, 0.0, 1.0), dtype=np.float32) if abs(float(direction_arr[2])) < 0.999 else np.array((0.0, 1.0, 0.0), dtype=np.float32)


def _build_direction_frame(center_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center_dir_arr = np.asarray(center_dir, dtype=np.float32).reshape(3)
    tangent_x = np.cross(_pick_tangent_axis(center_dir_arr), center_dir_arr).astype(np.float32, copy=False)
    tangent_x /= np.float32(max(float(np.linalg.norm(tangent_x)), 1e-12))
    tangent_y = np.cross(center_dir_arr, tangent_x).astype(np.float32, copy=False)
    return tangent_x, tangent_y


def _directional_alpha(camera: Camera, ray_direction: np.ndarray, splat: np.ndarray, radius_scale: float, alpha_cutoff: float) -> float:
    opacity = float(np.clip(splat[13], 0.0, 1.0))
    if opacity < alpha_cutoff:
        return 0.0

    camera_pos = camera.world_point_to_camera(splat[0:3])
    center_distance = float(np.linalg.norm(camera_pos))
    if center_distance <= 1e-12:
        return 0.0

    center_dir = camera_pos / np.float32(center_distance)
    tangent_x, tangent_y = _build_direction_frame(center_dir)
    sigma = np.exp(splat[3:6]).astype(np.float32) * np.float32(radius_scale)
    variance = sigma * sigma
    local_tangent_x = _quat_rotate(camera.camera_to_world(tangent_x), splat[6:10])
    local_tangent_y = _quat_rotate(camera.camera_to_world(tangent_y), splat[6:10])
    sigma_ortho = np.array(
        (
            float(np.dot(local_tangent_x * variance, local_tangent_x)),
            float(np.dot(local_tangent_x * variance, local_tangent_y)),
            float(np.dot(local_tangent_y * variance, local_tangent_y)),
        ),
        dtype=np.float32,
    ) / np.float32(max(center_distance * center_distance, 1e-12))
    det = float(sigma_ortho[0] * sigma_ortho[2] - sigma_ortho[1] * sigma_ortho[1])
    if det <= 1e-12 or not np.isfinite(det):
        return 0.0

    ray_dir_camera = camera.world_to_camera(ray_direction)
    ray_dir_camera = np.asarray(ray_dir_camera, dtype=np.float32)
    denom = float(np.dot(center_dir, ray_dir_camera))
    if denom <= 1e-10:
        return 0.0
    eta = np.array(
        (
            float(np.dot(tangent_x, ray_dir_camera)),
            float(np.dot(tangent_y, ray_dir_camera)),
        ),
        dtype=np.float32,
    ) / np.float32(denom)
    quad_vec = np.array(
        (
            sigma_ortho[2] * eta[0] - sigma_ortho[1] * eta[1],
            -sigma_ortho[1] * eta[0] + sigma_ortho[0] * eta[1],
        ),
        dtype=np.float32,
    ) / np.float32(det)
    return float(opacity * math.exp(-0.5 * float(np.dot(eta, quad_vec))))


def _directional_outline_screen_point(camera: Camera, splat: np.ndarray, radius_scale: float, alpha_cutoff: float, width: int, height: int, sample_index: int) -> np.ndarray | None:
    opacity = float(np.clip(splat[13], 0.0, 1.0))
    if opacity < alpha_cutoff:
        return None

    camera_pos = camera.world_point_to_camera(splat[0:3])
    center_distance = float(np.linalg.norm(camera_pos))
    if center_distance <= 1e-12:
        return None

    center_dir = camera_pos / np.float32(center_distance)
    tangent_x, tangent_y = _build_direction_frame(center_dir)
    sigma = np.exp(splat[3:6]).astype(np.float32) * np.float32(radius_scale)
    variance = sigma * sigma
    local_tangent_x = _quat_rotate(camera.camera_to_world(tangent_x), splat[6:10])
    local_tangent_y = _quat_rotate(camera.camera_to_world(tangent_y), splat[6:10])
    sigma_ortho = np.array(
        (
            float(np.dot(local_tangent_x * variance, local_tangent_x)),
            float(np.dot(local_tangent_x * variance, local_tangent_y)),
            float(np.dot(local_tangent_y * variance, local_tangent_y)),
        ),
        dtype=np.float32,
    ) / np.float32(max(center_distance * center_distance, 1e-12))
    det = float(sigma_ortho[0] * sigma_ortho[2] - sigma_ortho[1] * sigma_ortho[1])
    if det <= 1e-12 or not np.isfinite(det):
        return None

    l00 = math.sqrt(max(float(sigma_ortho[0]), 0.0))
    if l00 <= 1e-6 or not np.isfinite(l00):
        return None
    l10 = float(sigma_ortho[1]) / l00
    l11_sq = float(sigma_ortho[2]) - l10 * l10
    l11 = math.sqrt(max(l11_sq, 0.0))
    if l11 <= 1e-6 or not np.isfinite(l11):
        return None

    cutoff_radius = math.sqrt(max(-2.0 * math.log(alpha_cutoff / max(opacity, alpha_cutoff)), 0.0))
    theta = float(2.0 * math.pi * sample_index / 5.0)
    eta = np.array((l00 * math.cos(theta), l10 * math.cos(theta) + l11 * math.sin(theta)), dtype=np.float32) * np.float32(cutoff_radius)
    ray_dir_camera = center_dir + tangent_x * eta[0] + tangent_y * eta[1]
    ray_dir_camera /= np.float32(max(float(np.linalg.norm(ray_dir_camera)), 1e-12))
    screen_point, ok = camera.project_camera_to_screen(ray_dir_camera, width, height)
    return screen_point if ok else None


def test_projection_outline_reference_is_finite() -> None:
    scene = make_scene(128, seed=7)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    projected = project_splats(scene, camera, width=160, height=96, radius_scale=1.8)

    assert np.all(np.isfinite(projected.center_radius_depth))
    assert np.all(np.isfinite(projected.pos_local))
    assert np.all(np.isfinite(projected.inv_scale))
    assert np.all(np.isfinite(projected.color_alpha))
    assert np.all(projected.center_radius_depth[:, 2] >= 0.0)


def test_projection_outline_reference_is_deterministic() -> None:
    scene = make_scene(64, seed=21)
    camera = Camera.look_at(position=(0.0, 0.0, 3.0), target=(0.0, 0.0, 0.0), near=0.1, far=12.0)

    p0 = project_splats(scene, camera, width=128, height=128, radius_scale=1.6)
    p1 = project_splats(scene, camera, width=128, height=128, radius_scale=1.6)

    np.testing.assert_allclose(p0.center_radius_depth, p1.center_radius_depth)
    np.testing.assert_array_equal(p0.valid, p1.valid)
    np.testing.assert_allclose(p0.ellipse_conic, p1.ellipse_conic)


def test_projection_outline_hits_alpha_cutoff() -> None:
    scene = make_scene(96, seed=19)
    camera = Camera.look_at(position=(0.0, 0.0, 3.5), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    projected = project_splats(scene, camera, width=192, height=128, radius_scale=1.6)

    visible = np.flatnonzero(projected.valid != 0)
    assert visible.size > 0
    rng = np.random.default_rng(123)
    sampled = rng.choice(visible, size=min(12, visible.size), replace=False)
    packed = np.concatenate(
        [scene.positions, scene.scales, scene.rotations, scene.colors, scene.opacities[:, None]],
        axis=1,
    ).astype(np.float32)
    for splat_index in sampled.tolist():
        center = projected.center_radius_depth[splat_index, :2]
        conic = projected.ellipse_conic[splat_index]
        for sample_index in range(5):
            outline_point = _directional_outline_screen_point(camera, packed[splat_index], 1.6, _ALPHA_CUTOFF, 192, 128, sample_index)
            assert outline_point is not None
            ray_direction = camera.screen_to_world_ray(outline_point, 192, 128)
            alpha = _directional_alpha(camera, ray_direction, packed[splat_index], 1.6, _ALPHA_CUTOFF)
            delta = outline_point - center
            quad = float(conic[0] * delta[0] * delta[0] + 2.0 * conic[1] * delta[0] * delta[1] + conic[2] * delta[1] * delta[1])
            assert abs(alpha - _ALPHA_CUTOFF) <= 5e-4
            assert abs(quad - 1.0) <= 1e-3