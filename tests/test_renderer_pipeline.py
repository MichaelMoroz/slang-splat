from __future__ import annotations

import numpy as np

from src.renderer import Camera, GaussianRenderer
from src.renderer.projection_sampled5_mvee_reference import project_splats_sampled5_mvee
from src.renderer.reference_cpu import (
    build_tile_key_value_pairs,
    build_tile_ranges,
    project_splats,
    rasterize,
    sort_key_values,
)
from src.scene import GaussianScene


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


def test_tile_keys_and_ranges_match_reference(device):
    scene = make_scene(32, seed=1)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=96, height=96, tile_size=16, radius_scale=1.8, list_capacity_multiplier=32)
    debug = renderer.debug_pipeline_data(scene, camera)

    projected = project_splats(
        scene, camera, renderer.width, renderer.height, renderer.radius_scale, renderer.max_splat_radius_px
    )
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
    assert np.max(np.abs(gpu_depth_codes.astype(np.int64) - ref_depth_codes.astype(np.int64))) <= 4
    np.testing.assert_array_equal(debug["tile_ranges"], ref_ranges)


def test_tiny_render_matches_cpu_reference(device):
    scene = make_scene(18, seed=5)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    background = np.array([0.1, 0.15, 0.2], dtype=np.float32)
    renderer = GaussianRenderer(device, width=64, height=64, tile_size=16, radius_scale=1.6, list_capacity_multiplier=32)
    gpu_image = renderer.render(scene, camera, background=background).image

    projected = project_splats(
        scene, camera, renderer.width, renderer.height, renderer.radius_scale, renderer.max_splat_radius_px
    )
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
        tile_size=16,
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
        max_splat_radius_px=renderer.max_splat_radius_px,
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


def test_sampled5_mvee_render_smoke(device):
    scene = make_scene(20, seed=13)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    background = np.array([0.05, 0.1, 0.15], dtype=np.float32)
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        tile_size=16,
        radius_scale=1.6,
        list_capacity_multiplier=32,
    )
    out = renderer.render(scene, camera, background=background)
    assert out.image.shape == (64, 64, 4)
    assert np.all(np.isfinite(out.image))


def test_debug_ellipse_overlay_render_smoke(device):
    scene = make_scene(24, seed=31)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        tile_size=16,
        radius_scale=1.6,
        list_capacity_multiplier=32,
        debug_show_ellipses=True,
    )
    out = renderer.render(scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert out.image.shape == (64, 64, 4)
    assert np.all(np.isfinite(out.image))


def test_debug_processed_count_render_smoke(device):
    scene = make_scene(24, seed=37)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(
        device,
        width=64,
        height=64,
        tile_size=16,
        radius_scale=1.6,
        list_capacity_multiplier=32,
        debug_show_processed_count=True,
    )
    out = renderer.render(scene, camera, background=np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert out.image.shape == (64, 64, 4)
    assert np.all(np.isfinite(out.image))


def test_render_stats_are_one_frame_delayed(device):
    scene = make_scene(16, seed=41)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=64, height=64, tile_size=16, radius_scale=1.6, list_capacity_multiplier=32)
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
        tile_size=16,
        radius_scale=1.6,
        list_capacity_multiplier=32,
    )

    grads0 = renderer.debug_raster_backward_grads(scene, camera, background=np.array([0.1, 0.15, 0.2], dtype=np.float32))
    grads1 = renderer.debug_raster_backward_grads(scene, camera, background=np.array([0.1, 0.15, 0.2], dtype=np.float32))

    for name in (
        "grad_splat_pos_local",
        "grad_splat_inv_scale",
        "grad_splat_quat",
        "grad_screen_color_alpha",
    ):
        assert grads0[name].shape == (scene.count, 4)
        assert np.all(np.isfinite(grads0[name]))
        np.testing.assert_allclose(grads0[name], grads1[name], rtol=1e-5, atol=1e-6)


def test_prepass_capacity_budget_caps_growth(device):
    scene = make_scene(256, seed=97)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(
        device,
        width=96,
        height=96,
        tile_size=16,
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
