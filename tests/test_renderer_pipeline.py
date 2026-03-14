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
from src.common import SHADER_ROOT
from src.renderer import Camera, GaussianRenderer
from src.scene import GaussianScene

_RASTER_GRAD_FIXED_SCALE = 65536.0


def make_scene(count: int, seed: int = 0) -> GaussianScene:
    rng = np.random.default_rng(seed)
    positions = np.zeros((count, 3), dtype=np.float32)
    positions[:, 0] = rng.uniform(-1.2, 1.2, size=count).astype(np.float32)
    positions[:, 1] = rng.uniform(-0.9, 0.9, size=count).astype(np.float32)
    positions[:, 2] = np.linspace(-1.0, 1.0, count, dtype=np.float32)
    scales = np.full((count, 3), 0.04, dtype=np.float32)
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
    renderer = GaussianRenderer(
        device,
        width=96,
        height=96,
        radius_scale=1.7,
        list_capacity_multiplier=32,
        sampled5_mvee_iters=6,
        sampled5_safety_scale=1.05,
        sampled5_radius_pad_px=1.0,
        sampled5_eps=1e-6,
    )
    debug = renderer.debug_pipeline_data(scene, camera)
    cpu_projected = project_splats_sampled5_mvee(
        scene=scene,
        camera=camera,
        width=renderer.width,
        height=renderer.height,
        radius_scale=renderer.radius_scale,
        mvee_iters=renderer.sampled5_mvee_iters,
        safety_scale=renderer.sampled5_safety_scale,
        radius_pad_px=renderer.sampled5_radius_pad_px,
        mvee_eps=renderer.sampled5_eps,
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


def test_prepass_populates_raster_cache(device):
    scene = make_scene(12, seed=9)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, list_capacity_multiplier=32)
    debug = renderer.debug_pipeline_data(scene, camera)

    raster_cache = np.asarray(debug["raster_cache"], dtype=np.float32)
    assert raster_cache.shape == (scene.count, 14)
    assert np.all(np.isfinite(raster_cache))
    assert float(np.max(np.abs(raster_cache[:, :10]))) > 0.0


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


def test_subpixel_gaussian_uses_pixel_floor_in_projection_and_raster(device):
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    background = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    renderer = GaussianRenderer(device, width=65, height=65, radius_scale=1.0, list_capacity_multiplier=16)
    expected_scale = camera.pixel_world_size_max(4.0, renderer.width, renderer.height)
    raw_scale = 0.5 * expected_scale
    scene = GaussianScene(
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        scales=np.full((1, 3), raw_scale, dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.75], dtype=np.float32),
        colors=np.array([[0.8, 0.6, 0.2]], dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )

    gpu_image = renderer.render(scene, camera, background=background).image
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

    effective_scale = float(np.mean(1.0 / projected.inv_scale[0]))
    expected_alpha = scene.opacities[0] * (raw_scale / effective_scale) ** 2
    center_pixel = renderer.width // 2
    assert expected_scale < effective_scale < 1.5 * expected_scale
    assert float(debug["screen_center_radius_depth"][0, 2]) >= 0.75
    assert 0.1 <= float(debug["screen_ellipse_conic"][0, 3]) <= 0.25
    assert gpu_image[center_pixel, center_pixel, 3] < scene.opacities[0]
    assert np.isclose(float(cpu_image[center_pixel, center_pixel, 3]), expected_alpha, atol=2e-3)
    np.testing.assert_allclose(gpu_image, cpu_image, rtol=0.0, atol=5e-3)


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
        scales=np.full((1, 3), 0.08, dtype=np.float32),
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

    for name in (
        "grad_positions",
        "grad_scales",
        "grad_rotations",
        "grad_color_alpha",
    ):
        assert grads0[name].shape == (scene.count, 4)
        assert np.all(np.isfinite(grads0[name]))
        np.testing.assert_allclose(grads0[name], grads1[name], rtol=2e-5, atol=3e-5)


def test_raster_backward_decodes_q16_16_grad_grid(device):
    scene = make_scene(20, seed=79)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, list_capacity_multiplier=32)
    grads = renderer.debug_raster_backward_grads(scene, camera, background=np.array([0.05, 0.1, 0.15], dtype=np.float32))

    values = np.asarray(grads["cached_raster_grads_fixed"], dtype=np.int32)
    assert values.shape == (scene.count, 14)
    nonzero = values[values != 0]
    assert nonzero.size > 0
    decoded = nonzero.astype(np.float32) / _RASTER_GRAD_FIXED_SCALE
    np.testing.assert_allclose(decoded * _RASTER_GRAD_FIXED_SCALE, np.rint(decoded * _RASTER_GRAD_FIXED_SCALE), rtol=0.0, atol=1e-4)


def test_raster_backward_produces_float_final_grads_and_fixed_intermediate(device):
    scene = make_scene(18, seed=83)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=64, height=64, radius_scale=1.6, list_capacity_multiplier=32)
    grads = renderer.debug_raster_backward_grads(scene, camera, background=np.array([0.02, 0.04, 0.06], dtype=np.float32))

    fixed_nonzero = np.asarray(grads["cached_raster_grads_fixed"], dtype=np.int32)
    assert np.any(fixed_nonzero != 0)

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
    assert np.any(np.abs(final_nonzero * _RASTER_GRAD_FIXED_SCALE - np.rint(final_nonzero * _RASTER_GRAD_FIXED_SCALE)) > 1e-4)


def test_clamped_subpixel_raster_backward_has_scale_gradient(device):
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=65, height=65, radius_scale=1.0, list_capacity_multiplier=16)
    raw_scale = 0.5 * camera.pixel_world_size_max(4.0, renderer.width, renderer.height)
    scene = GaussianScene(
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        scales=np.full((1, 3), raw_scale, dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.75], dtype=np.float32),
        colors=np.array([[0.8, 0.6, 0.2]], dtype=np.float32),
        sh_coeffs=np.zeros((1, 1, 3), dtype=np.float32),
    )

    grads = renderer.debug_raster_backward_grads(scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))

    assert np.all(np.isfinite(grads["grad_scales"]))
    assert float(np.max(np.abs(grads["grad_scales"][0, :3]))) > 1e-7


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
