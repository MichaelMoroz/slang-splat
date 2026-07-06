from __future__ import annotations

import math

import numpy as np

from reference_impls.reference_cpu import _compute_scanline_tile_span_universal, project_splats
from src.renderer import Camera, PROJECTION_MODEL_EQUIRECTANGULAR
from src.scene import GaussianScene

_ALPHA_CUTOFF = 1.0 / 255.0
_PROJECTION_ALPHA_TOL = 5e-4
_EQUIRECT_DENSE_WIDTH = 384
_EQUIRECT_DENSE_HEIGHT = 192
_EQUIRECT_DENSE_LONGITUDES = 48
_EQUIRECT_DENSE_LATITUDES = 23
_EQUIRECT_SHELL_RADIUS = np.float32(3.0)
_EQUIRECT_SHELL_SIGMA = np.float32(0.006)
_EQUIRECT_SHELL_OPACITY = np.float32(0.8)
_EQUIRECT_RADIUS_SCALE = 1.6
_EQUIRECT_LARGE_CENTER_WIDTH = 1536
_EQUIRECT_LARGE_CENTER_HEIGHT = 768
_EQUIRECT_LARGE_CENTER_SIGMA = np.float32(0.08)
_EDGE_SAMPLE_COUNT = 8


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


def _equirectangular_dense_shell_scene() -> GaussianScene:
    lon_ids, lat_ids = np.meshgrid(
        np.arange(_EQUIRECT_DENSE_LONGITUDES, dtype=np.float32),
        np.arange(_EQUIRECT_DENSE_LATITUDES, dtype=np.float32),
    )
    theta = np.float32(2.0 * np.pi) * ((lon_ids + np.float32(0.5)) / np.float32(_EQUIRECT_DENSE_LONGITUDES)) - np.float32(np.pi)
    phi = np.float32(np.pi) * (np.float32(0.5) - (lat_ids + np.float32(0.5)) / np.float32(_EQUIRECT_DENSE_LATITUDES))
    cos_phi = np.cos(phi)
    positions = np.stack((cos_phi * np.sin(theta), -np.sin(phi), cos_phi * np.cos(theta)), axis=-1).reshape(-1, 3).astype(np.float32) * _EQUIRECT_SHELL_RADIUS
    count = int(positions.shape[0])
    return GaussianScene(
        positions=positions,
        scales=np.full((count, 3), np.log(_EQUIRECT_SHELL_SIGMA), dtype=np.float32),
        rotations=np.repeat(np.array(((1.0, 0.0, 0.0, 0.0),), dtype=np.float32), count, axis=0),
        opacities=np.full((count,), _EQUIRECT_SHELL_OPACITY, dtype=np.float32),
        colors=np.repeat(np.array(((0.8, 0.7, 0.6),), dtype=np.float32), count, axis=0),
        sh_coeffs=np.zeros((count, 1, 3), dtype=np.float32),
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


def _ray_splat_intersection_alpha(ray_origin: np.ndarray, ray_direction: np.ndarray, splat: np.ndarray, radius_scale: float) -> float:
    opacity = float(np.clip(splat[13], 0.0, 1.0))
    if opacity < _ALPHA_CUTOFF:
        return 0.0
    support_sigma_radius = math.sqrt(max(-2.0 * math.log(_ALPHA_CUTOFF / max(opacity, _ALPHA_CUTOFF)), 0.0))
    scale = np.maximum(np.exp(splat[3:6]).astype(np.float32) * np.float32(radius_scale * support_sigma_radius), np.float32(1e-6))
    ro_local = _quat_rotate(ray_origin - splat[0:3], splat[6:10]) / scale
    ray_local = _quat_rotate(ray_direction, splat[6:10]) / scale
    denom = float(np.dot(ray_local, ray_local))
    if denom <= 1e-10:
        return 0.0
    t_closest = -float(np.dot(ray_local, ro_local)) / denom
    if t_closest <= 0.0:
        return 0.0
    closest = ro_local + ray_local * np.float32(t_closest)
    rho2 = max(float(np.dot(closest, closest)), 0.0)
    return float(opacity * math.exp(-0.5 * support_sigma_radius * support_sigma_radius * rho2))


def _eval_conic(conic: np.ndarray, x: float, y: float) -> float:
    return float(conic[0]) * x * x + 2.0 * float(conic[1]) * x * y + float(conic[2]) * y * y


def _min_conic_over_tile_box(conic: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> float:
    values = [_eval_conic(conic, x, y) for x in (x0, x1) for y in (y0, y1)]
    if x0 <= 0.0 <= x1 and y0 <= 0.0 <= y1:
        return 0.0
    a, b, c = map(float, conic)
    if c > 1e-12:
        values.extend(_eval_conic(conic, x, float(np.clip(-b * x / c, y0, y1))) for x in (x0, x1))
    if a > 1e-12:
        values.extend(_eval_conic(conic, float(np.clip(-b * y / a, x0, x1)), y) for y in (y0, y1))
    return float(min(values))


def _tile_box(center: tuple[float, float], scan_along_x: bool, tile_size: int, line_tile: int, minor_tile: int) -> tuple[float, float, float, float]:
    major = np.array([minor_tile, line_tile], dtype=np.float32) if scan_along_x else np.array([line_tile, minor_tile], dtype=np.float32)
    lo = major * float(tile_size) - np.asarray(center, dtype=np.float32)
    hi = (major + 1.0) * float(tile_size) - np.asarray(center, dtype=np.float32)
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def _compute_scanline_tile_span_bruteforce(center: tuple[float, float], conic: np.ndarray, scan_along_x: bool, tile_size: int, line_coord_tile: int, min_minor_tile: int, max_minor_tile: int) -> tuple[bool, int, int]:
    hits = []
    for minor in range(max(min_minor_tile, 0), max_minor_tile + 1):
        if _min_conic_over_tile_box(conic, *_tile_box(center, scan_along_x, tile_size, line_coord_tile, minor)) <= 1.0 + 1e-6:
            hits.append(minor)
    return (False, 0, 0) if not hits else (True, hits[0], hits[-1] - hits[0] + 1)


def _is_fullscreen_fallback_ellipse(center_radius_depth: np.ndarray, conic: np.ndarray, width: int, height: int) -> bool:
    radius = float(center_radius_depth[2])
    expected_radius = float(max(width, height))
    inv_radius_sq = 1.0 / max(expected_radius * expected_radius, 1e-12)
    return (
        np.allclose(center_radius_depth[:2], np.array((0.5 * width, 0.5 * height), dtype=np.float32), atol=1e-4)
        and radius >= expected_radius
        and np.allclose(conic, np.array((inv_radius_sq, 0.0, inv_radius_sq), dtype=np.float32), atol=1e-7)
    )


def _is_cap_rect_conic(conic: np.ndarray) -> bool:
    return float(conic[0]) < 0.0 and float(conic[2]) < 0.0


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


def test_equirectangular_reference_keeps_side_splat_visible() -> None:
    scene = GaussianScene(
        positions=np.array(((3.0, 0.0, 0.0),), dtype=np.float32),
        scales=np.full((1, 3), -4.5, dtype=np.float32),
        rotations=np.array(((1.0, 0.0, 0.0, 0.0),), dtype=np.float32),
        opacities=np.array((0.8,), dtype=np.float32),
        colors=np.array(((0.8, 0.7, 0.6),), dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )
    camera = Camera.look_at(
        position=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, 1.0),
        near=0.1,
        far=20.0,
        projection_model=PROJECTION_MODEL_EQUIRECTANGULAR,
    )

    projected = project_splats(scene, camera, width=256, height=128, radius_scale=1.6)
    screen, ok = camera.project_world_to_screen(scene.positions[0], 256, 128)

    assert ok
    np.testing.assert_allclose(screen, np.array((192.0, 64.0), dtype=np.float32), rtol=0.0, atol=1e-5)
    assert int(projected.valid[0]) == 1


def test_equirectangular_reference_keeps_large_centered_fit_visible() -> None:
    scene = GaussianScene(
        positions=np.array(((0.0, 0.0, _EQUIRECT_SHELL_RADIUS),), dtype=np.float32),
        scales=np.full((1, 3), np.log(_EQUIRECT_LARGE_CENTER_SIGMA), dtype=np.float32),
        rotations=np.array(((1.0, 0.0, 0.0, 0.0),), dtype=np.float32),
        opacities=np.array((_EQUIRECT_SHELL_OPACITY,), dtype=np.float32),
        colors=np.array(((0.8, 0.7, 0.6),), dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )
    camera = Camera.look_at(
        position=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, 1.0),
        near=0.1,
        far=20.0,
        projection_model=PROJECTION_MODEL_EQUIRECTANGULAR,
    )

    projected = project_splats(scene, camera, width=_EQUIRECT_LARGE_CENTER_WIDTH, height=_EQUIRECT_LARGE_CENTER_HEIGHT, radius_scale=_EQUIRECT_RADIUS_SCALE)
    conic = projected.ellipse_conic[0]
    det = float(conic[0] * conic[2] - conic[1] * conic[1])

    assert int(projected.valid[0]) == 1
    assert float(projected.center_radius_depth[0, 2]) > 32.0
    assert 0.0 < det < 1e-6
    assert not _is_fullscreen_fallback_ellipse(projected.center_radius_depth[0], conic, _EQUIRECT_LARGE_CENTER_WIDTH, _EQUIRECT_LARGE_CENTER_HEIGHT)


def test_projection_outline_hits_alpha_cutoff() -> None:
    scene = make_scene(96, seed=19)
    camera = Camera.look_at(position=(0.0, 0.0, 3.5), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    projected = project_splats(scene, camera, width=192, height=128, radius_scale=1.6)

    visible = np.array(
        [
            splat_index
            for splat_index in np.flatnonzero(projected.valid != 0)
            if not _is_fullscreen_fallback_ellipse(projected.center_radius_depth[splat_index], projected.ellipse_conic[splat_index], 192, 128)
        ],
        dtype=np.int32,
    )
    assert visible.size > 0
    rng = np.random.default_rng(123)
    sampled = rng.choice(visible, size=min(12, visible.size), replace=False)
    packed = np.concatenate(
        [scene.positions, scene.scales, scene.rotations, scene.colors, scene.opacities[:, None]],
        axis=1,
    ).astype(np.float32)
    for splat_index in sampled.tolist():
        outline_point = _outline_screen_point(projected.center_radius_depth[splat_index, :2], projected.ellipse_conic[splat_index], float(rng.uniform(0.0, 2.0 * math.pi)))
        ray_direction = camera.screen_to_world_ray(outline_point, 192, 128)
        alpha = _ray_splat_intersection_alpha(camera.position, ray_direction, packed[splat_index], 1.6)
        assert abs(alpha - _ALPHA_CUTOFF) <= _PROJECTION_ALPHA_TOL


def test_equirectangular_dense_shell_outline_hits_alpha_cutoff() -> None:
    scene = _equirectangular_dense_shell_scene()
    camera = Camera.look_at(
        position=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, 1.0),
        near=0.1,
        far=20.0,
        projection_model=PROJECTION_MODEL_EQUIRECTANGULAR,
    )
    projected = project_splats(scene, camera, width=_EQUIRECT_DENSE_WIDTH, height=_EQUIRECT_DENSE_HEIGHT, radius_scale=_EQUIRECT_RADIUS_SCALE)
    packed = np.concatenate(
        [scene.positions, scene.scales, scene.rotations, scene.colors, scene.opacities[:, None]],
        axis=1,
    ).astype(np.float32)
    non_fallback_count = 0
    cap_rect_count = 0
    edge_angles = np.linspace(0.0, 2.0 * np.pi, num=_EDGE_SAMPLE_COUNT, endpoint=False)

    assert int(np.count_nonzero(projected.valid)) == scene.count
    for splat_index in np.flatnonzero(projected.valid != 0).tolist():
        assert not _is_fullscreen_fallback_ellipse(projected.center_radius_depth[splat_index], projected.ellipse_conic[splat_index], _EQUIRECT_DENSE_WIDTH, _EQUIRECT_DENSE_HEIGHT)
        if _is_cap_rect_conic(projected.ellipse_conic[splat_index]):
            cap_rect_count += 1
            continue
        non_fallback_count += 1
        for theta in edge_angles:
            outline_point = _outline_screen_point(projected.center_radius_depth[splat_index, :2], projected.ellipse_conic[splat_index], float(theta))
            ray_direction = camera.screen_to_world_ray(outline_point, _EQUIRECT_DENSE_WIDTH, _EQUIRECT_DENSE_HEIGHT)
            alpha = _ray_splat_intersection_alpha(camera.position, ray_direction, packed[splat_index], _EQUIRECT_RADIUS_SCALE)
            assert abs(alpha - _ALPHA_CUTOFF) <= _PROJECTION_ALPHA_TOL

    assert non_fallback_count > scene.count - 16
    assert 0 < cap_rect_count <= 16


def test_scanline_span_analytic_solver_matches_bruteforce() -> None:
    rng = np.random.default_rng(7)
    for _ in range(2000):
        theta = float(rng.uniform(0.0, np.pi))
        c = math.cos(theta)
        s = math.sin(theta)
        eig0 = float(10.0 ** rng.uniform(-3.0, 1.0))
        eig1 = float(10.0 ** rng.uniform(-3.0, 1.0))
        conic = np.array(
            (
                c * c * eig0 + s * s * eig1,
                c * -s * eig0 + s * c * eig1,
                s * s * eig0 + c * c * eig1,
            ),
            dtype=np.float32,
        )
        center = (float(rng.uniform(-128.0, 128.0)), float(rng.uniform(-128.0, 128.0)))
        scan_along_x = bool(rng.integers(0, 2))
        tile_size = int(rng.choice(np.array([8, 16, 32], dtype=np.int32)))
        line_coord = int(rng.integers(0, 32))
        min_minor = int(rng.integers(-4, 16))
        max_minor = min_minor + int(rng.integers(0, 24))

        brute = _compute_scanline_tile_span_bruteforce(center, conic, scan_along_x, tile_size, line_coord, min_minor, max_minor)
        analytic = _compute_scanline_tile_span_universal(center, conic, scan_along_x, tile_size, line_coord, min_minor, max_minor)
        assert analytic == brute
