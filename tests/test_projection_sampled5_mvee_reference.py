from __future__ import annotations

import numpy as np

from reference_impls.projection_sampled5_mvee_reference import project_splats_sampled5_mvee
from src.renderer import Camera
from src.scene import GaussianScene

_log_sigma = lambda sigma: np.log(np.asarray(sigma, dtype=np.float32))


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


def ellipse_values(points: np.ndarray, center: np.ndarray, axes: np.ndarray, angle: float) -> np.ndarray:
    c = np.cos(float(angle))
    s = np.sin(float(angle))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    local = (points - center[None, :]) @ rot
    denom = np.maximum(axes, 1e-6)
    return (local[:, 0] / denom[0]) ** 2 + (local[:, 1] / denom[1]) ** 2


def test_sampled5_mvee_reference_is_finite() -> None:
    scene = make_scene(128, seed=7)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    projected = project_splats_sampled5_mvee(scene, camera, width=160, height=96, radius_scale=1.8)

    assert np.all(np.isfinite(projected.center_radius_depth))
    assert np.all(np.isfinite(projected.pos_local))
    assert np.all(np.isfinite(projected.inv_scale))
    assert np.all(np.isfinite(projected.color_alpha))
    assert np.all(projected.center_radius_depth[:, 2] >= 1.0)


def test_sampled5_mvee_contains_sample_points() -> None:
    scene = make_scene(96, seed=19)
    camera = Camera.look_at(position=(0.0, 0.0, 3.5), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    projected = project_splats_sampled5_mvee(scene, camera, width=192, height=128, radius_scale=1.6)

    checked = 0
    for i in range(scene.count):
        pts = projected.sample_points_screen[i]
        if not np.all(np.isfinite(pts)):
            continue
        ell = projected.ellipse_center_axes[i]
        axes = ell[2:4]
        if np.min(axes) <= 0.0:
            continue
        vals = ellipse_values(pts, ell[:2], axes, float(ell[4]))
        assert float(np.max(vals)) <= 1.0005
        checked += 1
    assert checked > 0


def test_sampled5_mvee_reference_is_deterministic() -> None:
    scene = make_scene(64, seed=21)
    camera = Camera.look_at(position=(0.0, 0.0, 3.0), target=(0.0, 0.0, 0.0), near=0.1, far=12.0)

    p0 = project_splats_sampled5_mvee(scene, camera, width=128, height=128, radius_scale=1.6)
    p1 = project_splats_sampled5_mvee(scene, camera, width=128, height=128, radius_scale=1.6)

    np.testing.assert_allclose(p0.center_radius_depth, p1.center_radius_depth)
    np.testing.assert_array_equal(p0.valid, p1.valid)
    np.testing.assert_array_equal(p0.status_bits, p1.status_bits)


def test_sampled5_mvee_handles_degenerate_thin_splats() -> None:
    scene = make_scene(32, seed=3)
    scene.scales[:] = _log_sigma(np.array([1e-5, 0.3, 1e-5], dtype=np.float32))
    camera = Camera.look_at(position=(0.0, 0.0, 2.5), target=(0.0, 0.0, 0.0), near=0.05, far=10.0)

    projected = project_splats_sampled5_mvee(scene, camera, width=96, height=96, radius_scale=2.0)
    assert np.all(np.isfinite(projected.center_radius_depth))
    assert np.all(projected.center_radius_depth[:, 2] <= 96.0)


def test_sampled5_mvee_clamps_loaded_scale_to_configured_pixel_world_size() -> None:
    scene = GaussianScene(
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        scales=_log_sigma(np.array([[1e-6, 1e-6, 1e-6]], dtype=np.float32)),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.5], dtype=np.float32),
        colors=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)

    projected = project_splats_sampled5_mvee(scene, camera, width=128, height=128, radius_scale=1.0)
    expected_scale = camera.pixel_world_size_max(4.0, 128, 128)
    effective_scale = 1.0 / projected.inv_scale[0]
    assert np.all(effective_scale >= expected_scale)
    assert np.all(effective_scale <= 1.25 * expected_scale)
