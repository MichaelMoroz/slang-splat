from __future__ import annotations

from pathlib import Path

import numpy as np

from reference_impls.reference_cpu import (
    build_tile_key_value_pairs,
    build_tile_ranges,
    project_splats,
    rasterize,
    sort_key_values,
)
from src.app.shared import RendererParams, renderer_kwargs
from src.utility import SHADER_ROOT
from src.renderer import Camera, GaussianRenderer
from src.scene import GaussianScene, SH_C0, SUPPORTED_SH_COEFF_COUNT
from src.training import contribution_fixed_count_from_value

_GAUSSIAN_SUPPORT_SIGMA_RADIUS = 3.0
_TYPES_SHADER_PATH = Path(SHADER_ROOT / "renderer" / "gaussian_types.slang")
_PROJECTION_ALPHA_TOL = 5e-4
_PROJECTION_SAMPLE_COUNT = 12

_log_sigma = lambda sigma: np.log(np.asarray(sigma, dtype=np.float32))
_stored_from_support_scale = lambda support_scale: np.log(np.asarray(support_scale, dtype=np.float32) / _GAUSSIAN_SUPPORT_SIGMA_RADIUS)


def _quat_rotate(v: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    q = q / max(np.linalg.norm(q), 1e-12)
    qv = q[1:4]
    vec = np.asarray(v, dtype=np.float64).reshape(3)
    return np.asarray(vec + 2.0 * np.cross(np.cross(vec, qv) + q[0] * vec, qv), dtype=np.float32)


def _outline_screen_point(center: np.ndarray, conic: np.ndarray, theta: float) -> np.ndarray:
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    denom = conic[0] * direction[0] * direction[0] + 2.0 * conic[1] * direction[0] * direction[1] + conic[2] * direction[1] * direction[1]
    return np.asarray(center, dtype=np.float32) + direction / np.float32(np.sqrt(max(float(denom), 1e-12)))


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

    ray_dir_camera = np.asarray(camera.world_to_camera(ray_direction), dtype=np.float32)
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
    return float(opacity * np.exp(-0.5 * float(np.dot(eta, quad_vec))))


def _directional_outline_screen_point(camera: Camera, splat: np.ndarray, radius_scale: float, alpha_cutoff: float, width: int, height: int, sample_index: int, default_k1: float = 0.0, default_k2: float = 0.0) -> np.ndarray | None:
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

    l00 = float(np.sqrt(max(float(sigma_ortho[0]), 0.0)))
    if l00 <= 1e-6 or not np.isfinite(l00):
        return None
    l10 = float(sigma_ortho[1]) / l00
    l11_sq = float(sigma_ortho[2]) - l10 * l10
    l11 = float(np.sqrt(max(l11_sq, 0.0)))
    if l11 <= 1e-6 or not np.isfinite(l11):
        return None

    cutoff_radius = float(np.sqrt(max(-2.0 * np.log(alpha_cutoff / max(opacity, alpha_cutoff)), 0.0)))
    theta = float(2.0 * np.pi * sample_index / 5.0)
    eta = np.array((l00 * np.cos(theta), l10 * np.cos(theta) + l11 * np.sin(theta)), dtype=np.float32) * np.float32(cutoff_radius)
    ray_dir_camera = center_dir + tangent_x * eta[0] + tangent_y * eta[1]
    ray_dir_camera /= np.float32(max(float(np.linalg.norm(ray_dir_camera)), 1e-12))
    screen_point, ok = camera.project_camera_to_screen(ray_dir_camera, width, height, default_k1, default_k2)
    return screen_point if ok else None


def make_scene(count: int, seed: int = 0) -> GaussianScene:
    rng = np.random.default_rng(seed)
    positions = np.zeros((count, 3), dtype=np.float32)
    positions[:, 0] = rng.uniform(-1.2, 1.2, size=count).astype(np.float32)
    positions[:, 1] = rng.uniform(-0.9, 0.9, size=count).astype(np.float32)
    positions[:, 2] = np.linspace(-1.0, 1.0, count, dtype=np.float32)
    scales = _log_sigma(np.full((count, 3), 0.04, dtype=np.float32))
    rotations = np.zeros((count, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    opacities = np.linspace(0.25, 0.75, count, dtype=np.float32)
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


def _with_sh_debug_coeffs(scene: GaussianScene, coeff_index: int, scale: float = 0.75) -> GaussianScene:
    sh_coeffs = np.zeros((scene.count, SUPPORTED_SH_COEFF_COUNT, 3), dtype=np.float32)
    values = np.linspace(-scale, scale, scene.count, dtype=np.float32)
    sh_coeffs[:, coeff_index, 0] = values
    sh_coeffs[:, coeff_index, 1] = values[::-1]
    sh_coeffs[:, coeff_index, 2] = np.sin(np.linspace(0.0, np.pi, scene.count, dtype=np.float32)) * np.float32(scale)
    return GaussianScene(
        positions=scene.positions,
        scales=scene.scales,
        rotations=scene.rotations,
        opacities=scene.opacities,
        colors=scene.colors,
        sh_coeffs=sh_coeffs,
    )


def _make_layered_depth_scene(depths: tuple[float, ...], sigma: float = 0.04, opacity: float = 0.75) -> GaussianScene:
    count = len(depths)
    positions = np.zeros((count, 3), dtype=np.float32)
    positions[:, 2] = np.asarray(depths, dtype=np.float32)
    scales = _log_sigma(np.full((count, 3), sigma, dtype=np.float32))
    rotations = np.zeros((count, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    opacities = np.full((count,), opacity, dtype=np.float32)
    colors = np.full((count, 3), 0.5, dtype=np.float32)
    sh_coeffs = np.zeros((count, 1, 3), dtype=np.float32)
    return GaussianScene(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
        sh_coeffs=sh_coeffs,
    )


def _make_longitudinal_depth_scene(depth: float, transverse_sigma: float, longitudinal_sigma: float, opacity: float = 0.75) -> GaussianScene:
    return GaussianScene(
        positions=np.array([[0.0, 0.0, depth]], dtype=np.float32),
        scales=np.array([_log_sigma((transverse_sigma, transverse_sigma, longitudinal_sigma))], dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([opacity], dtype=np.float32),
        colors=np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )


def test_renderer_loads_raster_constants_from_shader(device):
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, list_capacity_multiplier=32)
    config = GaussianRenderer._load_raster_config(Path(SHADER_ROOT / "renderer" / "gaussian_types.slang"))

    assert renderer.tile_size == config.tile_size
    assert renderer._raster_config.thread_tile_dim == config.thread_tile_dim
    assert renderer._raster_config.tile_size == config.tile_size
    assert renderer._raster_config.batch == config.batch


def test_renderer_params_default_to_fixed_cached_grad_atomics():
    params = RendererParams()
    kwargs = renderer_kwargs(params)

    assert params.cached_raster_grad_atomic_mode == "fixed"
    assert kwargs["cached_raster_grad_atomic_mode"] == "fixed"
    assert params.cached_raster_grad_fixed_ro_local_range == 0.01
    assert kwargs["cached_raster_grad_fixed_ro_local_range"] == 0.01
    assert params.cached_raster_grad_fixed_scale_range == 0.01
    assert kwargs["cached_raster_grad_fixed_scale_range"] == 0.01
    assert params.cached_raster_grad_fixed_color_range == 0.2
    assert kwargs["cached_raster_grad_fixed_color_range"] == 0.2
    assert params.cached_raster_grad_fixed_opacity_range == 0.2
    assert kwargs["cached_raster_grad_fixed_opacity_range"] == 0.2
    assert params.max_anisotropy == 32.0
    assert kwargs["max_anisotropy"] == 32.0
    assert params.debug_gaussian_scale_multiplier == 1.0
    assert kwargs["debug_gaussian_scale_multiplier"] == 1.0
    assert params.debug_min_opacity == 0.0
    assert kwargs["debug_min_opacity"] == 0.0
    assert params.debug_opacity_multiplier == 1.0
    assert kwargs["debug_opacity_multiplier"] == 1.0
    assert params.debug_ellipse_scale_multiplier == 1.0
    assert kwargs["debug_ellipse_scale_multiplier"] == 1.0


def test_render_ignores_max_splat_steps_cap(device):
    count = 96
    positions = np.zeros((count, 3), dtype=np.float32)
    positions[:, 0] = np.linspace(-0.015, 0.015, count, dtype=np.float32)
    scales = np.full((count, 3), _log_sigma(0.05), dtype=np.float32)
    rotations = np.zeros((count, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    opacities = np.full((count,), 0.04, dtype=np.float32)
    colors = np.stack(
        (
            np.linspace(0.2, 0.9, count, dtype=np.float32),
            np.linspace(0.1, 0.8, count, dtype=np.float32),
            np.linspace(0.05, 0.6, count, dtype=np.float32),
        ),
        axis=1,
    )
    scene = GaussianScene(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
        sh_coeffs=np.zeros((count, 1, 3), dtype=np.float32),
    )
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    background = np.array([0.02, 0.03, 0.05], dtype=np.float32)
    limited = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, max_splat_steps=1, list_capacity_multiplier=128)
    unlimited = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, max_splat_steps=32768, list_capacity_multiplier=128)

    limited_image = limited.render(scene, camera, background=background).image
    unlimited_image = unlimited.render(scene, camera, background=background).image

    mean_abs_error = float(np.mean(np.abs(limited_image - unlimited_image)))
    max_abs_error = float(np.max(np.abs(limited_image - unlimited_image)))
    assert mean_abs_error < 2e-4
    assert max_abs_error < 3e-3


def test_max_anisotropy_clamps_cached_raster_scale(device):
    scene = GaussianScene(
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        scales=np.array([_log_sigma((0.2, 0.02, 0.02))], dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.85], dtype=np.float32),
        colors=np.array([[0.7, 0.6, 0.4]], dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    unconstrained = GaussianRenderer(device, width=64, height=64, radius_scale=1.0, max_anisotropy=100.0)
    constrained = GaussianRenderer(device, width=64, height=64, radius_scale=1.0, max_anisotropy=2.0)

    unconstrained_cache = np.asarray(unconstrained.debug_pipeline_data(scene, camera)["raster_cache"], dtype=np.float32)
    constrained_cache = np.asarray(constrained.debug_pipeline_data(scene, camera)["raster_cache"], dtype=np.float32)

    np.testing.assert_allclose(unconstrained_cache[0, 3:6], np.array([0.0025, 0.0, 0.000025], dtype=np.float32), rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(constrained_cache[0, 3:6], np.array([0.0025, 0.0, 0.000625], dtype=np.float32), rtol=2e-4, atol=2e-4)


def test_projection_keeps_partially_visible_splat_when_center_is_behind_camera(device):
    scene = GaussianScene(
        positions=np.array([[0.0, 0.0, 5.0]], dtype=np.float32),
        scales=np.array([_log_sigma((1.2, 1.2, 1.2))], dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.9], dtype=np.float32),
        colors=np.array([[0.8, 0.6, 0.2]], dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.0, list_capacity_multiplier=64)

    debug = renderer.debug_pipeline_data(scene, camera)
    center_radius_depth = np.asarray(debug["screen_center_radius_depth"], dtype=np.float32)

    assert center_radius_depth[0, 2] >= 64.0
    assert int(debug["generated_entries"]) > 0


def test_tile_keys_and_ranges_match_reference(device):
    scene = make_scene(32, seed=1)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=96, height=96, radius_scale=1.8, list_capacity_multiplier=32)
    debug = renderer.debug_pipeline_data(scene, camera)

    projected = project_splats(scene, camera, renderer.width, renderer.height, renderer.radius_scale)
    keys, values, generated = build_tile_key_value_pairs(
        projected=projected,
        tile_width=renderer.tile_width,
        tile_height=renderer.tile_height,
        tile_size=renderer.tile_size,
        max_list_entries=renderer._max_list_entries,
    )
    sorted_count = min(generated, renderer._max_list_entries)
    ref_keys, ref_values = sort_key_values(keys, values, sorted_count)
    ref_ranges = build_tile_ranges(ref_keys, sorted_count, renderer.tile_count)

    assert int(debug["generated_entries"]) == generated
    assert int(debug["sorted_count"]) == sorted_count
    np.testing.assert_array_equal(debug["keys"], ref_keys)
    np.testing.assert_array_equal(debug["values"], ref_values)
    np.testing.assert_array_equal(debug["tile_ranges"], ref_ranges)


def test_large_rotated_splat_tile_coverage_matches_reference(device):
    scene = GaussianScene(
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        scales=np.array([_log_sigma((0.28, 0.035, 0.035))], dtype=np.float32),
        rotations=np.array([[0.8660254, 0.0, 0.0, 0.5]], dtype=np.float32),
        opacities=np.array([0.9], dtype=np.float32),
        colors=np.array([[0.8, 0.6, 0.3]], dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=96, height=96, radius_scale=1.8, list_capacity_multiplier=64)
    debug = renderer.debug_pipeline_data(scene, camera)

    projected = project_splats(scene, camera, renderer.width, renderer.height, renderer.radius_scale)
    keys, values, generated = build_tile_key_value_pairs(
        projected=projected,
        tile_width=renderer.tile_width,
        tile_height=renderer.tile_height,
        tile_size=renderer.tile_size,
        max_list_entries=renderer._max_list_entries,
    )
    sorted_count = min(generated, renderer._max_list_entries)
    ref_keys, ref_values = sort_key_values(keys, values, sorted_count)
    ref_ranges = build_tile_ranges(ref_keys, sorted_count, renderer.tile_count)

    assert int(projected.valid[0]) == 1
    assert generated > renderer.tile_width
    assert int(debug["generated_entries"]) == generated
    np.testing.assert_array_equal(debug["keys"], ref_keys)
    np.testing.assert_array_equal(debug["values"], ref_values)
    np.testing.assert_array_equal(debug["tile_ranges"], ref_ranges)


def test_projection_culls_splats_inside_camera_min_distance(device):
    scene = GaussianScene(
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        scales=np.full((1, 3), _log_sigma(0.05), dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.8], dtype=np.float32),
        colors=np.array([[0.9, 0.7, 0.5]], dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    camera.min_camera_distance = 4.5
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, list_capacity_multiplier=8)

    debug = renderer.debug_pipeline_data(scene, camera)

    assert float(debug["screen_center_radius_depth"][0, 2]) == 0.0
    assert int(debug["generated_entries"]) == 0


def test_renderer_tile_sort_keeps_stable_order_for_equal_keys(device):
    count = 6
    positions = np.zeros((count, 3), dtype=np.float32)
    positions[:, 2] = 0.0
    scales = _log_sigma(np.full((count, 3), 0.04, dtype=np.float32))
    rotations = np.zeros((count, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    opacities = np.full((count,), 0.6, dtype=np.float32)
    colors = np.stack(
        (
            np.linspace(0.1, 0.6, count, dtype=np.float32),
            np.linspace(0.2, 0.7, count, dtype=np.float32),
            np.linspace(0.3, 0.8, count, dtype=np.float32),
        ),
        axis=1,
    )
    scene = GaussianScene(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
        sh_coeffs=np.zeros((count, 1, 3), dtype=np.float32),
    )
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, list_capacity_multiplier=8)

    debug = renderer.debug_pipeline_data(scene, camera)
    sorted_values = np.asarray(debug["values"], dtype=np.uint32)
    sorted_keys = np.asarray(debug["keys"], dtype=np.uint32)
    matching = sorted_values[sorted_keys == sorted_keys[0]]

    assert matching.size > 1
    np.testing.assert_array_equal(matching, np.sort(matching, kind="stable"))


def test_prepass_sort_camera_position_changes_sort_order_without_reprojecting(device):
    scene = _make_layered_depth_scene((0.0, 1.0), sigma=0.04, opacity=0.8)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, list_capacity_multiplier=16)

    real_sort = renderer.debug_pipeline_data(scene, camera)
    dither_sort = renderer.debug_pipeline_data(scene, camera, sort_camera_position=np.array([0.0, 0.0, -4.0], dtype=np.float32))
    center_tile = (renderer.tile_height // 2) * renderer.tile_width + renderer.tile_width // 2
    real_start, real_end = np.asarray(real_sort["tile_ranges"], dtype=np.uint32)[center_tile]
    dither_start, dither_end = np.asarray(dither_sort["tile_ranges"], dtype=np.uint32)[center_tile]
    real_values = np.asarray(real_sort["values"], dtype=np.uint32)[int(real_start): int(real_end)]
    dither_values = np.asarray(dither_sort["values"], dtype=np.uint32)[int(dither_start): int(dither_end)]

    assert int(real_sort["generated_entries"]) == int(dither_sort["generated_entries"])
    assert int(real_sort["sorted_count"]) == int(dither_sort["sorted_count"])
    np.testing.assert_allclose(real_sort["screen_center_radius_depth"], dither_sort["screen_center_radius_depth"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(real_sort["raster_cache"], dither_sort["raster_cache"], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(real_values[:2], np.array([1, 0], dtype=np.uint32))
    np.testing.assert_array_equal(dither_values[:2], np.array([0, 1], dtype=np.uint32))


def test_prepass_sort_camera_dither_changes_sort_order_per_splat_without_reprojecting(device):
    scene = _make_layered_depth_scene((0.0, 1.0), sigma=0.04, opacity=0.8)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, list_capacity_multiplier=16)

    real_sort = renderer.debug_pipeline_data(scene, camera)
    center_tile = (renderer.tile_height // 2) * renderer.tile_width + renderer.tile_width // 2
    real_start, real_end = np.asarray(real_sort["tile_ranges"], dtype=np.uint32)[center_tile]
    real_values = np.asarray(real_sort["values"], dtype=np.uint32)[int(real_start): int(real_end)]
    dither_sort = None
    dither_values = None
    for seed in range(1, 33):
        candidate = renderer.debug_pipeline_data(scene, camera, sort_camera_dither_sigma=8.0, sort_camera_dither_seed=seed)
        start, end = np.asarray(candidate["tile_ranges"], dtype=np.uint32)[center_tile]
        values = np.asarray(candidate["values"], dtype=np.uint32)[int(start): int(end)]
        if values.size >= 2 and not np.array_equal(values[:2], real_values[:2]):
            dither_sort = candidate
            dither_values = values
            break

    assert dither_sort is not None
    assert int(real_sort["generated_entries"]) == int(dither_sort["generated_entries"])
    assert int(real_sort["sorted_count"]) == int(dither_sort["sorted_count"])
    np.testing.assert_allclose(real_sort["screen_center_radius_depth"], dither_sort["screen_center_radius_depth"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(real_sort["raster_cache"], dither_sort["raster_cache"], rtol=0.0, atol=0.0)
    assert dither_values is not None
    assert sorted(int(value) for value in dither_values[:2]) == [0, 1]


def test_tiny_render_matches_cpu_reference(device):
    scene = make_scene(18, seed=5)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    background = np.array([0.1, 0.15, 0.2], dtype=np.float32)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, list_capacity_multiplier=32)
    gpu_image = renderer.render(scene, camera, background=background).image

    projected = project_splats(scene, camera, renderer.width, renderer.height, renderer.radius_scale)
    keys, values, generated = build_tile_key_value_pairs(
        projected=projected,
        tile_width=renderer.tile_width,
        tile_height=renderer.tile_height,
        tile_size=renderer.tile_size,
        max_list_entries=renderer._max_list_entries,
    )
    sorted_count = min(generated, renderer._max_list_entries)
    ref_keys, ref_values = sort_key_values(keys, values, sorted_count)
    ref_ranges = build_tile_ranges(ref_keys, sorted_count, renderer.tile_count)
    cpu_image = rasterize(
        projected=projected,
        sorted_values=ref_values,
        tile_ranges=ref_ranges,
        camera=camera,
        width=renderer.width,
        height=renderer.height,
        tile_size=renderer.tile_size,
        tile_width=renderer.tile_width,
        background=background,
        alpha_cutoff=renderer.alpha_cutoff,
        max_splat_steps=renderer.max_splat_steps,
        transmittance_threshold=renderer.transmittance_threshold,
    )

    mean_abs_error = float(np.mean(np.abs(gpu_image - cpu_image)))
    assert mean_abs_error < 5e-3


def test_projection_outline_hits_alpha_cutoff(device):
    scene = make_scene(28, seed=8)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=96, height=96, radius_scale=1.7, list_capacity_multiplier=32)
    debug = renderer.debug_pipeline_data(scene, camera)
    visible = np.flatnonzero(np.asarray(debug["screen_center_radius_depth"], dtype=np.float32)[:, 2] > 0.0)

    assert visible.size > 0
    rng = np.random.default_rng(123)
    sample_count = min(_PROJECTION_SAMPLE_COUNT, visible.size)
    sampled = rng.choice(visible, size=sample_count, replace=False)
    for splat_index in sampled.tolist():
        center = debug["screen_center_radius_depth"][splat_index, :2]
        conic = debug["screen_ellipse_conic"][splat_index, :3]
        packed = np.concatenate((scene.positions[splat_index], scene.scales[splat_index], scene.rotations[splat_index], scene.colors[splat_index], scene.opacities[splat_index:splat_index + 1]))
        for sample_index in range(5):
            outline_point = _directional_outline_screen_point(camera, packed, renderer.radius_scale, renderer.alpha_cutoff, renderer.width, renderer.height, sample_index, renderer.proj_distortion_k1, renderer.proj_distortion_k2)
            assert outline_point is not None
            ray_direction = camera.screen_to_world_ray(outline_point, renderer.width, renderer.height, renderer.proj_distortion_k1, renderer.proj_distortion_k2)
            alpha = _directional_alpha(camera, ray_direction, packed, renderer.radius_scale, renderer.alpha_cutoff)
            delta = outline_point - center
            quad = float(conic[0] * delta[0] * delta[0] + 2.0 * conic[1] * delta[0] * delta[1] + conic[2] * delta[1] * delta[1])
            assert abs(alpha - renderer.alpha_cutoff) <= _PROJECTION_ALPHA_TOL
            assert abs(quad - 1.0) <= 1e-3


def test_distorted_render_matches_cpu_reference(device):
    scene = make_scene(18, seed=17)
    camera = Camera.look_at(
        position=(0.0, 0.0, 4.0),
        target=(0.0, 0.0, 0.0),
        near=0.1,
        far=20.0,
        distortion_k1=0.08,
        distortion_k2=-0.02,
    )
    background = np.array([0.08, 0.12, 0.18], dtype=np.float32)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, list_capacity_multiplier=32)
    gpu_image = renderer.render(scene, camera, background=background).image

    projected = project_splats(scene, camera, renderer.width, renderer.height, renderer.radius_scale)
    keys, values, generated = build_tile_key_value_pairs(
        projected=projected,
        tile_width=renderer.tile_width,
        tile_height=renderer.tile_height,
        tile_size=renderer.tile_size,
        max_list_entries=renderer._max_list_entries,
    )
    sorted_count = min(generated, renderer._max_list_entries)
    ref_keys, ref_values = sort_key_values(keys, values, sorted_count)
    ref_ranges = build_tile_ranges(ref_keys, sorted_count, renderer.tile_count)
    cpu_image = rasterize(
        projected=projected,
        sorted_values=ref_values,
        tile_ranges=ref_ranges,
        camera=camera,
        width=renderer.width,
        height=renderer.height,
        tile_size=renderer.tile_size,
        tile_width=renderer.tile_width,
        background=background,
        alpha_cutoff=renderer.alpha_cutoff,
        max_splat_steps=renderer.max_splat_steps,
        transmittance_threshold=renderer.transmittance_threshold,
    )

    mean_abs_error = float(np.mean(np.abs(gpu_image - cpu_image)))
    assert mean_abs_error < 7e-3


def test_prepass_populates_raster_cache(device):
    scene = make_scene(12, seed=9)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, list_capacity_multiplier=32)
    debug = renderer.debug_pipeline_data(scene, camera)

    raster_cache = np.asarray(debug["raster_cache"], dtype=np.float32)
    screen_center_radius_depth = np.asarray(debug["screen_center_radius_depth"], dtype=np.float32)
    screen_color_alpha = np.asarray(debug["screen_color_alpha"], dtype=np.float32)
    expected_center_dir = np.asarray(
        [
            camera.world_point_to_camera(scene.positions[index]) / max(np.linalg.norm(camera.world_point_to_camera(scene.positions[index])), 1e-12)
            for index in range(scene.count)
        ],
        dtype=np.float32,
    )

    assert raster_cache.shape == (scene.count, 11)
    assert np.all(np.isfinite(raster_cache))
    assert float(np.max(np.abs(raster_cache[:, :10]))) > 0.0
    np.testing.assert_allclose(raster_cache[:, :3], expected_center_dir, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(np.linalg.norm(raster_cache[:, :3], axis=1), np.ones((scene.count,), dtype=np.float32), rtol=2e-4, atol=2e-4)
    assert np.all(raster_cache[:, 3] > 0.0)
    assert np.all(raster_cache[:, 5] > 0.0)
    assert np.all(raster_cache[:, 3] * raster_cache[:, 5] - raster_cache[:, 4] * raster_cache[:, 4] > 0.0)
    np.testing.assert_allclose(raster_cache[:, 6], screen_center_radius_depth[:, 3], rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(raster_cache[:, 7:10], screen_color_alpha[:, :3], rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(raster_cache[:, 10], scene.opacities, rtol=0.0, atol=1e-6)


def test_projected_color_prepass_keeps_out_of_range_values(device):
    sh_coeffs = np.zeros((1, 1, 3), dtype=np.float32)
    sh_coeffs[0, 0, :] = (np.array([1.35, -0.2, 0.6], dtype=np.float32) - 0.5) / np.float32(SH_C0)
    scene = GaussianScene(
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        scales=_log_sigma(np.full((1, 3), 0.04, dtype=np.float32)),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.75], dtype=np.float32),
        colors=np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        sh_coeffs=sh_coeffs,
    )
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, list_capacity_multiplier=32)

    render_debug = renderer.debug_pipeline_data(scene, camera)
    render_cache = np.asarray(render_debug["raster_cache"], dtype=np.float32)

    renderer.set_scene(scene)
    np.testing.assert_allclose(render_cache[0, 7:10], np.array([1.35, -0.2, 0.6], dtype=np.float32), rtol=0.0, atol=1e-5)
    renderer.execute_prepass_for_current_scene(camera, sync_counts=True)
    np.testing.assert_allclose(renderer.read_raster_cache(scene.count)[0, 7:10], np.array([1.35, -0.2, 0.6], dtype=np.float32), rtol=0.0, atol=1e-5)


def test_projection_render_smoke(device):
    scene = make_scene(20, seed=13)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    background = np.array([0.05, 0.1, 0.15], dtype=np.float32)
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        radius_scale=1.6,
        list_capacity_multiplier=32,
    )
    out = renderer.render(scene, camera, background=background)
    assert out.image.shape == (64, 64, 4)
    assert np.all(np.isfinite(out.image))


def test_radius_scale_scales_true_3dgs_size_without_hidden_fudge(device):
    scene = GaussianScene(
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        scales=np.full((1, 3), _log_sigma(0.2), dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.75], dtype=np.float32),
        colors=np.array([[0.8, 0.6, 0.2]], dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    projected_1x = project_splats(scene, camera, width=512, height=512, radius_scale=1.0)
    projected_2x = project_splats(scene, camera, width=512, height=512, radius_scale=2.0)

    support_1x = 1.0 / projected_1x.inv_scale[0]
    support_2x = 1.0 / projected_2x.inv_scale[0]

    np.testing.assert_allclose(support_1x, np.full((3,), 0.2 * _GAUSSIAN_SUPPORT_SIGMA_RADIUS, dtype=np.float32), rtol=0.0, atol=5e-4)
    np.testing.assert_allclose(support_2x / support_1x, np.full((3,), 2.0, dtype=np.float32), rtol=5e-4, atol=1e-6)

    far_camera = Camera.look_at(position=(0.0, 0.0, 20.0), target=(0.0, 0.0, 0.0), near=0.1, far=100.0)
    projected_footprint_1x = project_splats(scene, far_camera, width=512, height=512, radius_scale=1.0)
    projected_footprint_2x = project_splats(scene, far_camera, width=512, height=512, radius_scale=2.0)
    adjusted_1x = projected_footprint_1x.center_radius_depth[0, 2] - 1.0
    adjusted_2x = projected_footprint_2x.center_radius_depth[0, 2] - 1.0
    assert np.isclose(adjusted_2x / adjusted_1x, 2.0, rtol=4e-3, atol=1e-4)


def test_debug_ellipse_overlay_render_smoke(device):
    scene = make_scene(24, seed=31)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        radius_scale=1.6,
        list_capacity_multiplier=32,
        debug_show_ellipses=True,
    )
    out = renderer.render(scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert out.image.shape == (64, 64, 4)
    assert np.all(np.isfinite(out.image))


def test_debug_ellipse_overlay_defaults_to_4px_thickness(device):
    renderer = GaussianRenderer(device, width=64, height=64)
    assert renderer.debug_ellipse_thickness_px == 4.0
    assert renderer.debug_gaussian_scale_multiplier == 1.0
    assert renderer.debug_min_opacity == 0.0
    assert renderer.debug_opacity_multiplier == 1.0
    assert renderer.debug_ellipse_scale_multiplier == 1.0


def test_debug_ellipse_overlay_antialiases_across_boundary(device):
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    scene = GaussianScene(
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        scales=np.full((1, 3), _log_sigma(0.08), dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.85], dtype=np.float32),
        colors=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )
    renderer = GaussianRenderer(
        device,
        width=65,
        height=65,
        radius_scale=1.6,
        list_capacity_multiplier=16,
        debug_show_ellipses=True,
    )

    out = renderer.render(scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    debug = renderer.debug_pipeline_data(scene, camera)
    center = debug["screen_center_radius_depth"][0, :2]
    conic = debug["screen_ellipse_conic"][0, :3]
    ys, xs = np.mgrid[0 : renderer.height, 0 : renderer.width]
    delta = np.stack((xs + 0.5 - center[0], ys + 0.5 - center[1]), axis=-1).astype(np.float32)
    quad = (
        conic[0] * delta[..., 0] * delta[..., 0]
        + 2.0 * conic[1] * delta[..., 0] * delta[..., 1]
        + conic[2] * delta[..., 1] * delta[..., 1]
    )
    boundary_ring = (quad >= 0.85) & (quad <= 1.25)

    assert np.any(boundary_ring)
    assert float(np.max(out.image[..., 3][boundary_ring])) > 1e-4


def test_debug_ellipse_overlay_keeps_half_opacity_fill(device):
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    scene = GaussianScene(
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        scales=np.full((1, 3), _log_sigma(0.08), dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.85], dtype=np.float32),
        colors=np.array([[1.0, 0.8, 0.4]], dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )
    normal_renderer = GaussianRenderer(
        device,
        width=65,
        height=65,
        radius_scale=1.6,
        list_capacity_multiplier=16,
        debug_show_ellipses=False,
    )
    outline_renderer = GaussianRenderer(
        device,
        width=65,
        height=65,
        radius_scale=1.6,
        list_capacity_multiplier=16,
        debug_show_ellipses=True,
    )

    normal_out = normal_renderer.render(scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    outline_out = outline_renderer.render(scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    debug = outline_renderer.debug_pipeline_data(scene, camera)
    center = debug["screen_center_radius_depth"][0, :2]
    center_px = tuple(int(v) for v in np.floor(center).astype(np.int32))

    assert 0 <= center_px[0] < outline_renderer.width
    assert 0 <= center_px[1] < outline_renderer.height
    assert float(outline_out.image[center_px[1], center_px[0], 3]) > 1e-4
    assert float(outline_out.image[center_px[1], center_px[0], 3]) < float(normal_out.image[center_px[1], center_px[0], 3])


def test_debug_processed_count_render_smoke(device):
    scene = make_scene(24, seed=37)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        radius_scale=1.6,
        list_capacity_multiplier=32,
        debug_show_processed_count=True,
    )
    out = renderer.render(scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert out.image.shape == (64, 64, 4)
    assert np.all(np.isfinite(out.image))
    channel_spread = np.max(out.image[..., :3], axis=-1) - np.min(out.image[..., :3], axis=-1)
    assert float(np.max(channel_spread)) > 1e-4


def test_debug_grad_norm_render_smoke(device):
    scene = make_scene(24, seed=43)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        radius_scale=1.6,
        list_capacity_multiplier=32,
        debug_show_grad_norm=True,
    )
    renderer.upload_debug_grad_norm(np.geomspace(1e-8, 1e-2, scene.count, dtype=np.float32))
    out = renderer.render(scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert out.image.shape == (64, 64, 4)
    assert np.all(np.isfinite(out.image))


def test_debug_grad_variance_render_smoke(device):
    scene = make_scene(24, seed=44)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        radius_scale=1.6,
        list_capacity_multiplier=32,
        debug_mode=GaussianRenderer.DEBUG_MODE_GRAD_VARIANCE,
    )
    samples = np.geomspace(1e-8, 1e-2, scene.count, dtype=np.float32)
    stats = np.zeros((scene.count, 4), dtype=np.float32)
    stats[:, 0] = samples + samples * 3.0
    stats[:, 1] = samples * samples + (samples * 3.0) * (samples * 3.0)
    stats[:, 2] = 2.0
    renderer.upload_debug_grad_stats(stats)
    out = renderer.render(scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert out.image.shape == (64, 64, 4)
    assert np.all(np.isfinite(out.image))


def test_debug_splat_age_render_smoke(device):
    scene = make_scene(24, seed=47)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        radius_scale=1.6,
        list_capacity_multiplier=32,
        debug_mode=GaussianRenderer.DEBUG_MODE_SPLAT_AGE,
    )
    renderer.upload_debug_splat_age(np.linspace(0.0, 1.0, scene.count, dtype=np.float32))
    out = renderer.render(scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert out.image.shape == (64, 64, 4)
    assert np.all(np.isfinite(out.image))
    channel_spread = np.max(out.image[..., :3], axis=-1) - np.min(out.image[..., :3], axis=-1)
    assert float(np.max(channel_spread)) > 1e-4


def test_debug_splat_density_render_smoke(device):
    scene = make_scene(24, seed=53)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        radius_scale=1.6,
        list_capacity_multiplier=32,
        debug_mode=GaussianRenderer.DEBUG_MODE_SPLAT_DENSITY,
    )
    out = renderer.render(scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert out.image.shape == (64, 64, 4)
    assert np.all(np.isfinite(out.image))
    channel_spread = np.max(out.image[..., :3], axis=-1) - np.min(out.image[..., :3], axis=-1)
    assert float(np.max(channel_spread)) > 1e-4


def test_debug_contribution_amount_render_smoke(device):
    scene = make_scene(24, seed=59)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        radius_scale=1.6,
        list_capacity_multiplier=32,
        debug_mode=GaussianRenderer.DEBUG_MODE_CONTRIBUTION_AMOUNT,
    )
    observed_pixels = renderer.width * renderer.height
    renderer.set_debug_contribution_observed_pixel_count(observed_pixels)
    renderer.upload_debug_splat_contribution(
        np.array(
            [contribution_fixed_count_from_value(value, observed_pixels) for value in np.geomspace(0.001, 1.0, scene.count, dtype=np.float32)],
            dtype=np.uint32,
        )
    )
    out = renderer.render(scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert out.image.shape == (64, 64, 4)
    assert np.all(np.isfinite(out.image))
    channel_spread = np.max(out.image[..., :3], axis=-1) - np.min(out.image[..., :3], axis=-1)
    assert float(np.max(channel_spread)) > 1e-4


def test_debug_adam_momentum_render_smoke(device):
    scene = make_scene(24, seed=67)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    for mode, channel in ((GaussianRenderer.DEBUG_MODE_ADAM_MOMENTUM, 0), (GaussianRenderer.DEBUG_MODE_ADAM_SECOND_MOMENT, 1)):
        renderer = GaussianRenderer(
            device,
            width=64,
            height=64,
            radius_scale=1.6,
            list_capacity_multiplier=32,
            debug_mode=mode,
        )
        moments = np.zeros((renderer.TRAINABLE_PARAM_COUNT, scene.count, 2), dtype=np.float32)
        moments[:, :, channel] = np.linspace(0.0, 0.02, scene.count, dtype=np.float32)[None, :]
        adam_buffer = device.create_buffer(size=moments.size * 4, usage=renderer._RW_BUFFER_USAGE)
        adam_buffer.copy_from_numpy(moments.reshape(-1, 2))
        renderer.set_debug_adam_moments_buffer(adam_buffer)
        out = renderer.render(scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert out.image.shape == (64, 64, 4)
        assert np.all(np.isfinite(out.image))
        channel_spread = np.max(out.image[..., :3], axis=-1) - np.min(out.image[..., :3], axis=-1)
        assert float(np.max(channel_spread)) > 1e-4


def test_debug_sh_view_dependent_render_smoke(device):
    scene = _with_sh_debug_coeffs(make_scene(24, seed=71), 1, scale=0.5)
    camera = Camera.look_at(position=(0.6, 0.2, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        radius_scale=1.6,
        list_capacity_multiplier=32,
        debug_mode=GaussianRenderer.DEBUG_MODE_SH_VIEW_DEPENDENT,
    )
    out = renderer.render(scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert out.image.shape == (64, 64, 4)
    assert np.all(np.isfinite(out.image))
    channel_spread = np.max(out.image[..., :3], axis=-1) - np.min(out.image[..., :3], axis=-1)
    assert float(np.max(channel_spread)) > 1e-4


def test_debug_sh_coefficient_render_smoke(device):
    scene = _with_sh_debug_coeffs(make_scene(24, seed=73), 9, scale=0.75)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        radius_scale=1.6,
        list_capacity_multiplier=32,
        debug_mode=GaussianRenderer.DEBUG_MODE_SH_COEFFICIENT,
        debug_sh_coeff_index=9,
    )
    out = renderer.render(scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert out.image.shape == (64, 64, 4)
    assert np.all(np.isfinite(out.image))
    channel_spread = np.max(out.image[..., :3], axis=-1) - np.min(out.image[..., :3], axis=-1)
    assert float(np.max(channel_spread)) > 1e-4


def test_debug_black_negative_render_smoke(device):
    scene = _with_sh_debug_coeffs(make_scene(24, seed=73), 9, scale=1.5)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        radius_scale=1.6,
        list_capacity_multiplier=32,
        debug_mode=GaussianRenderer.DEBUG_MODE_BLACK_NEGATIVE,
    )
    out = renderer.render(scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert out.image.shape == (64, 64, 4)
    assert np.all(np.isfinite(out.image))


def test_debug_depth_local_mismatch_render_smoke(device):
    scene = make_scene(24, seed=61)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        radius_scale=1.6,
        list_capacity_multiplier=32,
        debug_mode=GaussianRenderer.DEBUG_MODE_DEPTH_LOCAL_MISMATCH,
    )
    out = renderer.render(scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert out.image.shape == (64, 64, 4)
    assert np.all(np.isfinite(out.image))
    channel_spread = np.max(out.image[..., :3], axis=-1) - np.min(out.image[..., :3], axis=-1)
    assert float(np.max(channel_spread)) > 1e-4


def test_debug_depth_local_mismatch_highlights_close_layered_splats_more_than_far_layers(device):
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    near_scene = _make_layered_depth_scene((0.0, 0.03, 0.06))
    far_scene = _make_layered_depth_scene((0.0, 0.5, 1.0))
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        radius_scale=1.6,
        list_capacity_multiplier=32,
        debug_mode=GaussianRenderer.DEBUG_MODE_DEPTH_LOCAL_MISMATCH,
        debug_depth_local_mismatch_range=(0.0, 0.1),
        debug_depth_local_mismatch_smooth_radius=2.0,
        debug_depth_local_mismatch_reject_radius=4.0,
    )

    near_out = renderer.render(near_scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32)).image
    far_out = renderer.render(far_scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32)).image

    assert float(np.max(near_out[..., :3])) > float(np.max(far_out[..., :3])) + 1e-3


def test_debug_depth_mean_uses_center_distance_not_longitudinal_intersection(device):
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    compact_scene = _make_longitudinal_depth_scene(depth=0.0, transverse_sigma=0.04, longitudinal_sigma=0.04, opacity=0.9)
    stretched_scene = _make_longitudinal_depth_scene(depth=0.0, transverse_sigma=0.04, longitudinal_sigma=1.2, opacity=0.9)
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        radius_scale=1.6,
        list_capacity_multiplier=32,
        debug_mode=GaussianRenderer.DEBUG_MODE_DEPTH_MEAN,
        debug_depth_mean_range=(3.0, 5.0),
    )

    compact_out = renderer.render(compact_scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32)).image
    stretched_out = renderer.render(stretched_scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32)).image
    center_y = renderer.height // 2
    center_x = renderer.width // 2

    assert float(np.max(compact_out[center_y, center_x, :3])) > 1e-3
    np.testing.assert_allclose(stretched_out[center_y, center_x, :3], compact_out[center_y, center_x, :3], rtol=0.0, atol=2e-4)


def test_render_stats_are_one_frame_delayed(device):
    scene = make_scene(16, seed=41)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, list_capacity_multiplier=32)
    renderer.set_scene(scene)
    _, stats0 = renderer.render_to_texture(camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert stats0["stats_valid"] is False
    assert int(stats0["stats_latency_frames"]) == 1
    _, stats1 = renderer.render_to_texture(camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert stats1["stats_valid"] is True
    assert int(stats1["generated_entries"]) >= 0
    assert int(stats1["written_entries"]) >= 0


def test_raster_backward_smoke_and_determinism(device):
    scene = make_scene(20, seed=73)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        radius_scale=1.6,
        list_capacity_multiplier=32,
    )

    grads0 = renderer.debug_raster_backward_grads(scene, camera, background=np.array([0.1, 0.15, 0.2], dtype=np.float32))
    grads1 = renderer.debug_raster_backward_grads(scene, camera, background=np.array([0.1, 0.15, 0.2], dtype=np.float32))

    assert renderer.cached_raster_grad_atomic_mode == "fixed"
    assert grads0["cached_raster_grads_mode"] == "fixed"
    np.testing.assert_allclose(
        np.asarray(grads0["cached_raster_grads_active"], dtype=np.int32),
        np.asarray(grads1["cached_raster_grads_active"], dtype=np.int32),
        rtol=0.0,
        atol=0.0,
    )
    for name in ("grad_positions", "grad_scales", "grad_rotations", "grad_color_alpha"):
        assert grads0[name].shape == (scene.count, 4)
        assert np.all(np.isfinite(grads0[name]))
        np.testing.assert_allclose(grads0[name], grads1[name], rtol=2e-5, atol=3e-5)


def test_raster_backward_decodes_fixed_grad_grid(device):
    scene = make_scene(20, seed=79)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, list_capacity_multiplier=32, cached_raster_grad_atomic_mode="fixed")
    grads = renderer.debug_raster_backward_grads(scene, camera, background=np.array([0.05, 0.1, 0.15], dtype=np.float32))

    assert grads["cached_raster_grads_mode"] == "fixed"
    values = np.asarray(grads["cached_raster_grads_fixed"], dtype=np.int32)
    assert values.shape == (scene.count, 10)
    nonzero = values[values != 0]
    assert nonzero.size > 0
    decoded = np.asarray(renderer.read_cached_raster_grads_fixed_decoded(scene.count), dtype=np.float32)
    requantized = np.rint(decoded / renderer.cached_raster_grad_fixed_decode_scale_table(scene.count)).astype(np.int32)
    assert int(np.max(np.abs(requantized[values != 0] - values[values != 0]))) <= 128


def test_raster_backward_float_mode_produces_float_intermediate_and_final_grads(device):
    scene = make_scene(18, seed=83)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, list_capacity_multiplier=32, cached_raster_grad_atomic_mode="float")
    grads = renderer.debug_raster_backward_grads(scene, camera, background=np.array([0.02, 0.04, 0.06], dtype=np.float32))

    float_nonzero = np.asarray(grads["cached_raster_grads_float"], dtype=np.float32)
    assert grads["cached_raster_grads_mode"] == "float"
    assert np.any(float_nonzero != 0.0)
    active_nonzero = float_nonzero[np.abs(float_nonzero) > 0.0]
    assert active_nonzero.size > 0
    requantized = float_nonzero / renderer.cached_raster_grad_fixed_decode_scale_table(scene.count)
    assert np.any(np.abs(requantized[np.abs(float_nonzero) > 0.0] - np.rint(requantized[np.abs(float_nonzero) > 0.0])) > 1e-4)
    assert np.count_nonzero(float_nonzero[:, 6:9]) > 0

    final_values = np.concatenate(
        [
            np.asarray(grads["grad_positions"], dtype=np.float32).reshape(-1),
            np.asarray(grads["grad_scales"], dtype=np.float32).reshape(-1),
            np.asarray(grads["grad_rotations"], dtype=np.float32).reshape(-1),
            np.asarray(grads["grad_color_alpha"], dtype=np.float32).reshape(-1),
        ]
    )
    final_nonzero = final_values[np.abs(final_values) > 0.0]
    assert final_nonzero.size > 0
    assert np.any(np.abs(final_nonzero * 65536.0 - np.rint(final_nonzero * 65536.0)) > 1e-4)
    assert np.count_nonzero(np.asarray(grads["grad_color_alpha"], dtype=np.float32)[:, :3]) > 0


def test_raster_backward_fixed_mode_tracks_float_mode_for_anisotropic_scales(device):
    scene = GaussianScene(
        positions=np.array([[-0.18, 0.0, 0.0], [0.0, 0.0, 0.05], [0.18, 0.0, -0.05]], dtype=np.float32),
        scales=np.array(
            [
                _log_sigma((0.04, 0.04, 1.2)),
                _log_sigma((0.2, 0.02, 0.02)),
                _log_sigma((0.2, 0.02, 1.2)),
            ],
            dtype=np.float32,
        ),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.full((3,), 0.8, dtype=np.float32),
        colors=np.array([[0.75, 0.2, 0.1], [0.1, 0.75, 0.2], [0.2, 0.1, 0.75]], dtype=np.float32),
        sh_coeffs=np.zeros((3, 1, 3), dtype=np.float32),
    )
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    background = np.array([0.03, 0.05, 0.07], dtype=np.float32)
    fixed_renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, list_capacity_multiplier=32, cached_raster_grad_atomic_mode="fixed")
    float_renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, list_capacity_multiplier=32, cached_raster_grad_atomic_mode="float")

    fixed_grads = fixed_renderer.debug_raster_backward_grads(scene, camera, background=background)
    float_grads = float_renderer.debug_raster_backward_grads(scene, camera, background=background)

    for name in ("grad_positions", "grad_scales", "grad_rotations", "grad_color_alpha"):
        fixed_values = np.asarray(fixed_grads[name], dtype=np.float32)
        float_values = np.asarray(float_grads[name], dtype=np.float32)
        assert np.any(np.abs(float_values) > 0.0)
        np.testing.assert_allclose(fixed_values, float_values, rtol=0.05, atol=5e-4)


def test_active_cached_raster_grad_metrics_tensor_matches_float_mode_buffer(device):
    scene = make_scene(10, seed=89)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, list_capacity_multiplier=16, cached_raster_grad_atomic_mode="float")
    grads = renderer.debug_raster_backward_grads(scene, camera, background=np.array([0.03, 0.05, 0.07], dtype=np.float32))

    prepared = renderer.read_active_cached_raster_grads_float_tensor(scene.count)

    np.testing.assert_allclose(prepared, np.asarray(grads["cached_raster_grads_float"], dtype=np.float32), rtol=0.0, atol=0.0)


def test_active_cached_raster_grad_metrics_tensor_decodes_fixed_mode(device):
    scene = make_scene(10, seed=97)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, list_capacity_multiplier=16, cached_raster_grad_atomic_mode="fixed")
    renderer.debug_raster_backward_grads(scene, camera, background=np.array([0.01, 0.03, 0.05], dtype=np.float32))

    prepared = renderer.read_active_cached_raster_grads_float_tensor(scene.count)
    decoded = np.asarray(renderer.read_cached_raster_grads_fixed_decoded(scene.count), dtype=np.float32)

    np.testing.assert_allclose(prepared, decoded, rtol=0.0, atol=1e-7)


def test_cached_raster_grad_fixed_range_updates_decode_scales(device):
    renderer = GaussianRenderer(device, width=32, height=32, cached_raster_grad_fixed_ro_local_range=2.5)

    np.testing.assert_allclose(
        renderer.cached_raster_grad_fixed_decode_scales[:3],
        np.full((3,), np.float32(2.5 / GaussianRenderer._RASTER_GRAD_FIXED_INT_MAX), dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )



def test_prepass_capacity_budget_caps_growth(device):
    scene = make_scene(256, seed=97)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(
        device,
        width=96,
        height=96,
        radius_scale=1.6,
        list_capacity_multiplier=1024,
        max_prepass_memory_mb=1,
    )
    renderer.set_scene(scene)

    entry_cap = max(
        (renderer.max_prepass_memory_mb * renderer._MEBIBYTE_BYTES) // renderer._PREPASS_ENTRY_BYTES,
        1,
    )
    assert renderer._max_list_entries <= entry_cap
    assert renderer._max_scanline_entries <= entry_cap

    _, stats0 = renderer.render_to_texture(camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    _, stats1 = renderer.render_to_texture(camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert stats0["stats_valid"] is False
    assert stats1["stats_valid"] is True
    assert int(stats1["prepass_entry_cap"]) == int(entry_cap)
    assert int(stats1["prepass_memory_mb"]) == 1


def test_prepass_overflow_refresh_grows_capacity_and_converges(device):
    scene = make_scene(192, seed=109)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(
        device,
        width=128,
        height=128,
        radius_scale=1.6,
        list_capacity_multiplier=1,
        max_prepass_memory_mb=64,
    )
    renderer.set_scene(scene)
    initial_capacity = int(renderer._max_list_entries)

    _, _ = renderer.render_to_texture(camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    _, stats1 = renderer.render_to_texture(camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    follow_up = [
        renderer.render_to_texture(camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))[1]
        for _ in range(3)
    ]

    assert stats1["stats_valid"] is True
    assert int(renderer._max_list_entries) >= initial_capacity
    if bool(stats1["overflow"]):
        assert int(renderer._max_list_entries) > initial_capacity
        grown_capacities = [int(renderer._max_list_entries)]
        grown_capacities.extend(int(stats["max_list_entries"]) for stats in follow_up)
        assert max(grown_capacities) > initial_capacity


def test_partial_tile_render_matches_cpu_reference(device):
    scene = make_scene(22, seed=101)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    background = np.array([0.12, 0.08, 0.18], dtype=np.float32)
    renderer = GaussianRenderer(device, width=71, height=53, radius_scale=1.6, list_capacity_multiplier=32)
    gpu_image = renderer.render(scene, camera, background=background).image

    projected = project_splats(scene, camera, renderer.width, renderer.height, renderer.radius_scale)
    keys, values, generated = build_tile_key_value_pairs(
        projected=projected,
        tile_width=renderer.tile_width,
        tile_height=renderer.tile_height,
        tile_size=renderer.tile_size,
        max_list_entries=renderer._max_list_entries,
    )
    sorted_count = min(generated, renderer._max_list_entries)
    ref_keys, ref_values = sort_key_values(keys, values, sorted_count)
    ref_ranges = build_tile_ranges(ref_keys, sorted_count, renderer.tile_count)
    cpu_image = rasterize(
        projected=projected,
        sorted_values=ref_values,
        tile_ranges=ref_ranges,
        camera=camera,
        width=renderer.width,
        height=renderer.height,
        tile_size=renderer.tile_size,
        tile_width=renderer.tile_width,
        background=background,
        alpha_cutoff=renderer.alpha_cutoff,
        max_splat_steps=renderer.max_splat_steps,
        transmittance_threshold=renderer.transmittance_threshold,
    )

    mean_abs_error = float(np.mean(np.abs(gpu_image - cpu_image)))
    assert mean_abs_error < 5e-3
