from __future__ import annotations

from pathlib import Path

import numpy as np

from reference_impls.projection_sampled5_mvee_reference import project_splats_sampled5_mvee
from reference_impls.reference_cpu import (
    build_tile_key_value_pairs,
    build_tile_ranges,
    project_splats,
    rasterize,
    sort_key_values,
)
from src.app.shared import RendererParams, renderer_kwargs
from src.common import SHADER_ROOT
from src.renderer import Camera, GaussianRenderer
from src.scene import GaussianScene

_GAUSSIAN_SUPPORT_SIGMA_RADIUS = 3.0
_TYPES_SHADER_PATH = Path(SHADER_ROOT / "renderer" / "gaussian_types.slang")
_SAMPLED5_MVEE_ITERS = GaussianRenderer._load_uint_shader_constant(_TYPES_SHADER_PATH, "SAMPLED5_MVEE_ITERS")
_SAMPLED5_SAFETY_SCALE = GaussianRenderer._load_float_shader_constant(_TYPES_SHADER_PATH, "SAMPLED5_SAFETY_SCALE")
_SAMPLED5_RADIUS_PAD_PX = GaussianRenderer._load_float_shader_constant(_TYPES_SHADER_PATH, "SAMPLED5_RADIUS_PAD_PX")
_SAMPLED5_EPS = GaussianRenderer._load_float_shader_constant(_TYPES_SHADER_PATH, "SAMPLED5_EPS")

_log_sigma = lambda sigma: np.log(np.asarray(sigma, dtype=np.float32))
_stored_from_support_scale = lambda support_scale: np.log(np.asarray(support_scale, dtype=np.float32) / _GAUSSIAN_SUPPORT_SIGMA_RADIUS)


def _quat_rotate(v: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    q = q / max(np.linalg.norm(q), 1e-12)
    qv = q[1:4]
    vec = np.asarray(v, dtype=np.float64).reshape(3)
    return np.asarray(vec + 2.0 * np.cross(np.cross(vec, qv) + q[0] * vec, qv), dtype=np.float32)


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
    assert params.cached_raster_grad_fixed_quat_range == 0.01
    assert kwargs["cached_raster_grad_fixed_quat_range"] == 0.01
    assert params.cached_raster_grad_fixed_color_range == 0.2
    assert kwargs["cached_raster_grad_fixed_color_range"] == 0.2
    assert params.cached_raster_grad_fixed_opacity_range == 0.2
    assert kwargs["cached_raster_grad_fixed_opacity_range"] == 0.2


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
        depth_bits=renderer.depth_bits,
        near_depth=camera.near,
        far_depth=camera.far,
        max_list_entries=renderer._max_list_entries,
    )
    sorted_count = min(generated, renderer._max_list_entries)
    ref_keys, ref_values = sort_key_values(keys, values, sorted_count)
    ref_ranges = build_tile_ranges(ref_keys, sorted_count, renderer.tile_count, renderer.depth_bits)

    assert int(debug["generated_entries"]) == generated
    assert int(debug["sorted_count"]) == sorted_count
    gpu_tile_ids = debug["keys"] >> np.uint32(renderer.depth_bits)
    ref_tile_ids = ref_keys >> np.uint32(renderer.depth_bits)
    np.testing.assert_array_equal(gpu_tile_ids, ref_tile_ids)
    gpu_depth_codes = debug["keys"] & np.uint32((1 << renderer.depth_bits) - 1)
    ref_depth_codes = ref_keys & np.uint32((1 << renderer.depth_bits) - 1)
    assert np.max(np.abs(gpu_depth_codes.astype(np.int64) - ref_depth_codes.astype(np.int64))) <= 8
    np.testing.assert_array_equal(debug["tile_ranges"], ref_ranges)


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
        depth_bits=renderer.depth_bits,
        near_depth=camera.near,
        far_depth=camera.far,
        max_list_entries=renderer._max_list_entries,
    )
    sorted_count = min(generated, renderer._max_list_entries)
    ref_keys, ref_values = sort_key_values(keys, values, sorted_count)
    ref_ranges = build_tile_ranges(ref_keys, sorted_count, renderer.tile_count, renderer.depth_bits)
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


def test_sampled5_mvee_projection_matches_cpu_reference(device):
    scene = make_scene(28, seed=8)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=96, height=96, radius_scale=1.7, list_capacity_multiplier=32)
    debug = renderer.debug_pipeline_data(scene, camera)
    cpu_projected = project_splats_sampled5_mvee(
        scene=scene,
        camera=camera,
        width=renderer.width,
        height=renderer.height,
        radius_scale=renderer.radius_scale,
        mvee_iters=_SAMPLED5_MVEE_ITERS,
        safety_scale=_SAMPLED5_SAFETY_SCALE,
        radius_pad_px=_SAMPLED5_RADIUS_PAD_PX,
        mvee_eps=_SAMPLED5_EPS,
        distortion_k1=renderer.proj_distortion_k1,
        distortion_k2=renderer.proj_distortion_k2,
    )

    gpu_center = debug["screen_center_radius_depth"][:, :2]
    cpu_center = cpu_projected.center_radius_depth[:, :2]
    gpu_radius = debug["screen_center_radius_depth"][:, 2]
    cpu_radius = cpu_projected.center_radius_depth[:, 2]
    center_err = np.linalg.norm(gpu_center - cpu_center, axis=1)
    radius_err = np.abs(gpu_radius - cpu_radius)
    assert float(np.max(center_err)) < 0.5
    assert float(np.max(radius_err)) < 0.8


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
        depth_bits=renderer.depth_bits,
        near_depth=camera.near,
        far_depth=camera.far,
        max_list_entries=renderer._max_list_entries,
    )
    sorted_count = min(generated, renderer._max_list_entries)
    ref_keys, ref_values = sort_key_values(keys, values, sorted_count)
    ref_ranges = build_tile_ranges(ref_keys, sorted_count, renderer.tile_count, renderer.depth_bits)
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
    assert raster_cache.shape == (scene.count, 14)
    assert np.all(np.isfinite(raster_cache))
    assert float(np.max(np.abs(raster_cache[:, :10]))) > 0.0
    expected_ro = np.asarray(projected.pos_local, dtype=np.float32)
    expected_scale = 1.0 / np.maximum(np.asarray(projected.inv_scale, dtype=np.float32), 1e-12)
    expected_quat = np.asarray(projected.quat, dtype=np.float32)
    np.testing.assert_allclose(raster_cache[:, :3], expected_ro, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(raster_cache[:, 3:6], expected_scale, rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(raster_cache[:, 6:10], expected_quat, rtol=3e-4, atol=3e-4)


def test_sampled5_mvee_render_smoke(device):
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
    projected_footprint_1x = project_splats_sampled5_mvee(
        scene=scene,
        camera=far_camera,
        width=512,
        height=512,
        radius_scale=1.0,
        mvee_iters=6,
        safety_scale=1.0,
        radius_pad_px=0.0,
        mvee_eps=1e-6,
    )
    projected_footprint_2x = project_splats_sampled5_mvee(
        scene=scene,
        camera=far_camera,
        width=512,
        height=512,
        radius_scale=2.0,
        mvee_iters=6,
        safety_scale=1.0,
        radius_pad_px=0.0,
        mvee_eps=1e-6,
    )
    assert np.isclose(projected_footprint_2x.center_radius_depth[0, 2] / projected_footprint_1x.center_radius_depth[0, 2], 2.0, rtol=2e-3, atol=1e-4)


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


def test_debug_ellipse_overlay_defaults_to_2px_thickness(device):
    renderer = GaussianRenderer(device, width=64, height=64)
    assert renderer.debug_ellipse_thickness_px == 2.0


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
    outside_ring = (quad >= 1.0) & (quad <= 1.25)

    assert np.any(outside_ring)
    assert float(np.max(out.image[..., 3][outside_ring])) > 1e-4


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
    assert values.shape == (scene.count, 14)
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
    assert np.count_nonzero(float_nonzero[:, 10:13]) > 0

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
    grads = renderer.debug_raster_backward_grads(scene, camera, background=np.array([0.01, 0.03, 0.05], dtype=np.float32))

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
        depth_bits=renderer.depth_bits,
        near_depth=camera.near,
        far_depth=camera.far,
        max_list_entries=renderer._max_list_entries,
    )
    sorted_count = min(generated, renderer._max_list_entries)
    ref_keys, ref_values = sort_key_values(keys, values, sorted_count)
    ref_ranges = build_tile_ranges(ref_keys, sorted_count, renderer.tile_count, renderer.depth_bits)
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
