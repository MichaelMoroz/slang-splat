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


def _expected_raster_sigma(camera: Camera, support_scale: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.asarray(support_scale, dtype=np.float64).reshape(3), 1e-12)
    rotation = np.asarray(Camera._rotation_matrix_from_quaternion_wxyz(quat_wxyz), dtype=np.float64).reshape(3, 3)
    camera_to_world = np.asarray(camera.basis(), dtype=np.float64).reshape(3, 3).T
    transform = np.diag(1.0 / scale) @ rotation @ camera_to_world
    sigma = transform.T @ transform
    return np.array(
        [
            sigma[0, 0],
            sigma[1, 1],
            sigma[2, 2],
            sigma[0, 1],
            sigma[0, 2],
            sigma[1, 2],
        ],
        dtype=np.float32,
    )


_DEPTH_RASTER_GRAD_COMPONENT_IDS = np.array([2, 5, 7, 8], dtype=np.int64)


def _outline_screen_point(center: np.ndarray, conic: np.ndarray, theta: float) -> np.ndarray:
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    denom = conic[0] * direction[0] * direction[0] + 2.0 * conic[1] * direction[0] * direction[1] + conic[2] * direction[1] * direction[1]
    return np.asarray(center, dtype=np.float32) + direction / np.float32(np.sqrt(max(float(denom), 1e-12)))


def _ray_splat_intersection_alpha(ray_origin: np.ndarray, ray_direction: np.ndarray, splat: np.ndarray, radius_scale: float, alpha_cutoff: float) -> float:
    opacity = float(np.clip(splat[13], 0.0, 1.0))
    if opacity < alpha_cutoff:
        return 0.0
    support_sigma_radius = float(np.sqrt(max(-2.0 * np.log(alpha_cutoff / max(opacity, alpha_cutoff)), 0.0)))
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
    return float(opacity * np.exp(-0.5 * support_sigma_radius * support_sigma_radius * rho2))


def _is_fullscreen_fallback_ellipse(center_radius_depth: np.ndarray, conic: np.ndarray, width: int, height: int) -> bool:
    radius = float(center_radius_depth[2])
    expected_radius = float(max(width, height))
    inv_radius_sq = 1.0 / max(expected_radius * expected_radius, 1e-12)
    return (
        np.allclose(center_radius_depth[:2], np.array((0.5 * width, 0.5 * height), dtype=np.float32), atol=1e-4)
        and radius >= expected_radius
        and np.allclose(conic[:3], np.array((inv_radius_sq, 0.0, inv_radius_sq), dtype=np.float32), atol=1e-7)
    )


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
    support_sigma_radius = float(np.sqrt(max(-2.0 * np.log(alpha_cutoff / max(opacity, alpha_cutoff)), 0.0)))
    scale = np.maximum(np.exp(splat[3:6]).astype(np.float32) * np.float32(radius_scale * support_sigma_radius), np.float32(1e-6))
    ro_local = _quat_rotate(ray_origin - splat[0:3], splat[6:10]) / scale
    ray_local = _quat_rotate(ray_direction, splat[6:10]) / scale
    denom = float(np.dot(ray_local, ray_local))
    assert kwargs["cached_raster_grad_fixed_color_range"] == 8.0
    assert params.cached_raster_grad_fixed_opacity_range == 8.0
    assert kwargs["cached_raster_grad_fixed_opacity_range"] == 8.0
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
    t_closest = -float(np.dot(ray_local, ro_local)) / denom
    if t_closest <= 0.0:
        return 0.0
    closest = ro_local + ray_local * np.float32(t_closest)
    rho2 = max(float(np.dot(closest, closest)), 0.0)
    return float(opacity * np.exp(-0.5 * support_sigma_radius * support_sigma_radius * rho2))

    unconstrained_cache = np.asarray(unconstrained.debug_pipeline_data(scene, camera)["raster_cache"], dtype=np.float32)
    constrained_cache = np.asarray(constrained.debug_pipeline_data(scene, camera)["raster_cache"], dtype=np.float32)

    np.testing.assert_allclose(unconstrained_cache[0, 3:6], np.array([1.0 / (0.6 * 0.6), 1.0 / (0.06 * 0.06), 1.0 / (0.06 * 0.06)], dtype=np.float32), rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(constrained_cache[0, 3:6], np.array([1.0 / (0.6 * 0.6), 1.0 / (0.3 * 0.3), 1.0 / (0.3 * 0.3)], dtype=np.float32), rtol=2e-4, atol=2e-4)


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


def test_projection_does_not_cull_splats_beyond_camera_far(device):
    scene = GaussianScene(
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        scales=np.full((1, 3), _log_sigma(0.05), dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.8], dtype=np.float32),
        colors=np.array([[0.7, 0.6, 0.5]], dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=1.0)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, list_capacity_multiplier=16)

    debug = renderer.debug_pipeline_data(scene, camera)

    assert int(debug["generated_entries"]) > 0
    assert int(np.asarray(debug["splat_visible"], dtype=np.uint32)[0]) == 1
    assert float(np.asarray(debug["screen_center_radius_depth"], dtype=np.float32)[0, 2]) > 0.0


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
    visible = np.array(
        [
            splat_index
            for splat_index in np.flatnonzero(np.asarray(debug["screen_center_radius_depth"], dtype=np.float32)[:, 2] > 0.0)
            if not _is_fullscreen_fallback_ellipse(
                np.asarray(debug["screen_center_radius_depth"], dtype=np.float32)[splat_index],
                np.asarray(debug["screen_ellipse_conic"], dtype=np.float32)[splat_index],
                renderer.width,
                renderer.height,
            )
        ],
        dtype=np.int32,
    )

    assert visible.size > 0
    rng = np.random.default_rng(123)
    sample_count = min(_PROJECTION_SAMPLE_COUNT, visible.size)
    sampled = rng.choice(visible, size=sample_count, replace=False)
    for splat_index in sampled.tolist():
        outline_point = _outline_screen_point(
            debug["screen_center_radius_depth"][splat_index, :2],
            debug["screen_ellipse_conic"][splat_index, :3],
            float(rng.uniform(0.0, 2.0 * np.pi)),
        )
        ray_direction = camera.screen_to_world_ray(outline_point, renderer.width, renderer.height, renderer.proj_distortion_k1, renderer.proj_distortion_k2)
        alpha = _ray_splat_intersection_alpha(
            camera.position,
            ray_direction,
            np.concatenate((scene.positions[splat_index], scene.scales[splat_index], scene.rotations[splat_index], scene.colors[splat_index], scene.opacities[splat_index:splat_index + 1])),
            renderer.radius_scale,
            renderer.alpha_cutoff,
        )
        assert abs(alpha - renderer.alpha_cutoff) <= _PROJECTION_ALPHA_TOL


def test_projection_culls_offscreen_near_plane_grazer(device):
    scene = GaussianScene(
        positions=np.array([[5.0, 0.0, 0.05]], dtype=np.float32),
        scales=_log_sigma(np.array([[0.5, 0.5, 0.5]], dtype=np.float32)),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.9], dtype=np.float32),
        colors=np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )
    camera = Camera.look_at(position=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=96, height=96, radius_scale=1.6, list_capacity_multiplier=32)

    debug = renderer.debug_pipeline_data(scene, camera)
    projected = project_splats(scene, camera, renderer.width, renderer.height, renderer.radius_scale)
    _, _, generated = build_tile_key_value_pairs(
        projected=projected,
        tile_width=renderer.tile_width,
        tile_height=renderer.tile_height,
        tile_size=renderer.tile_size,
        max_list_entries=renderer._max_list_entries,
    )

    assert int(debug["generated_entries"]) == 0
    assert int(np.asarray(debug["splat_visible"], dtype=np.uint32)[0]) == 0
    assert int(generated) == 0


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
    projected = project_splats(scene, camera, renderer.width, renderer.height, renderer.radius_scale)

    raster_cache = np.asarray(debug["raster_cache"], dtype=np.float32)
    assert raster_cache.shape == (scene.count, 13)
    assert np.all(np.isfinite(raster_cache))
    assert float(np.max(np.abs(raster_cache[:, :9]))) > 0.0
    expected_ro = np.asarray([camera.world_to_camera(camera.position - position) for position in scene.positions], dtype=np.float32)
    expected_scale = 1.0 / np.maximum(np.asarray(projected.inv_scale, dtype=np.float32), 1e-12)
    expected_quat = np.asarray(projected.quat, dtype=np.float32)
    expected_sigma = np.asarray([_expected_raster_sigma(camera, scale, quat) for scale, quat in zip(expected_scale, expected_quat, strict=False)], dtype=np.float32)
    np.testing.assert_allclose(raster_cache[:, :3], expected_ro, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(raster_cache[:, 3:9], expected_sigma, rtol=3e-4, atol=3e-4)


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
    np.testing.assert_allclose(render_cache[0, 9:12], np.array([1.35, -0.2, 0.6], dtype=np.float32), rtol=0.0, atol=1e-5)
    renderer.execute_prepass_for_current_scene(camera, sync_counts=True)
    np.testing.assert_allclose(renderer.read_raster_cache(scene.count)[0, 9:12], np.array([1.35, -0.2, 0.6], dtype=np.float32), rtol=0.0, atol=1e-5)


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
    stats = np.zeros((scene.count, 2), dtype=np.float32)
    stats[:, 0] = samples + samples * 3.0
    stats[:, 1] = samples * samples + (samples * 3.0) * (samples * 3.0)
    renderer.upload_debug_grad_stats(stats)
    renderer.set_debug_contribution_observed_pixel_count(2.0)
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


def test_debug_refinement_distribution_render_smoke(device):
    scene = make_scene(24, seed=61)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        radius_scale=1.6,
        list_capacity_multiplier=32,
        debug_mode=GaussianRenderer.DEBUG_MODE_REFINEMENT_DISTRIBUTION,
    )
    observed_pixels = renderer.width * renderer.height
    samples = np.geomspace(1e-8, 1e-2, scene.count, dtype=np.float32)
    stats = np.zeros((scene.count, 2), dtype=np.float32)
    stats[:, 0] = samples + samples * 3.0
    stats[:, 1] = samples * samples + (samples * 3.0) * (samples * 3.0)
    renderer.upload_debug_grad_stats(stats)
    renderer.set_debug_contribution_observed_pixel_count(observed_pixels)
    renderer.upload_debug_splat_contribution(
        np.array(
            [contribution_fixed_count_from_value(value, observed_pixels) for value in np.geomspace(0.001, 1.0, scene.count, dtype=np.float32)],
            dtype=np.uint32,
        )
    )
    renderer.debug_refinement_grad_variance_weight_exponent = 0.1
    renderer.debug_refinement_contribution_weight_exponent = 0.1
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
    near_scene = _make_layered_depth_scene((0.0, 0.03))
    far_scene = _make_layered_depth_scene((0.0, 0.5))
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
    assert values.shape == (scene.count, 13)
    nonzero = values[values != 0]
    assert nonzero.size > 0
    assert np.count_nonzero(values[:, _DEPTH_RASTER_GRAD_COMPONENT_IDS]) == 0
    decoded = np.asarray(renderer.read_cached_raster_grads_fixed_decoded(scene.count), dtype=np.float32)
    assert np.count_nonzero(decoded[:, _DEPTH_RASTER_GRAD_COMPONENT_IDS]) == 0
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
    assert np.count_nonzero(float_nonzero[:, _DEPTH_RASTER_GRAD_COMPONENT_IDS]) == 0
    requantized = float_nonzero / renderer.cached_raster_grad_fixed_decode_scale_table(scene.count)
    assert np.any(np.abs(requantized[np.abs(float_nonzero) > 0.0] - np.rint(requantized[np.abs(float_nonzero) > 0.0])) > 1e-4)
    assert np.count_nonzero(float_nonzero[:, 9:12]) > 0

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


def test_raster_backward_can_include_depth_cached_grad_channels(device):
    scene = make_scene(18, seed=83)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        radius_scale=1.6,
        list_capacity_multiplier=32,
        cached_raster_grad_atomic_mode="float",
        cached_raster_grad_include_depth=True,
    )
    grads = renderer.debug_raster_backward_grads(scene, camera, background=np.array([0.02, 0.04, 0.06], dtype=np.float32))

    float_values = np.asarray(grads["cached_raster_grads_float"], dtype=np.float32)

    assert np.count_nonzero(float_values[:, _DEPTH_RASTER_GRAD_COMPONENT_IDS]) > 0


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


def test_cached_raster_grad_fixed_decode_scale_table_uses_precision_diagonal_only(device):
    renderer = GaussianRenderer(device, width=32, height=32)
    raster_cache = np.zeros((2, renderer._RASTER_CACHE_PARAM_COUNT), dtype=np.float32)
    raster_cache[0, 3:6] = np.array([16.0, 1.0, 1.0], dtype=np.float32)
    raster_cache[0, 6:9] = np.array([9.0, 9.0, 9.0], dtype=np.float32)
    raster_cache[1, 3:6] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    raster_cache[1, 6:9] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    decode_scales = renderer.cached_raster_grad_fixed_decode_scale_table(2, raster_cache=raster_cache)
    expected_avg_inv_scale = np.power(np.array([16.0, 1.0], dtype=np.float32), np.float32(1.0 / 6.0))
    expected_scale_sq = 1.0 / (expected_avg_inv_scale * expected_avg_inv_scale)

    np.testing.assert_allclose(
        decode_scales[:, :3],
        expected_avg_inv_scale[:, None] * renderer.cached_raster_grad_fixed_decode_scales[None, :3],
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        decode_scales[:, 3:9],
        expected_scale_sq[:, None] * renderer.cached_raster_grad_fixed_decode_scales[None, 3:9],
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        decode_scales[:, 9:],
        np.broadcast_to(renderer.cached_raster_grad_fixed_decode_scales[9:], (2, renderer._RASTER_CACHE_PARAM_COUNT - 9)),
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
