from __future__ import annotations

import numpy as np

from src.renderer import Camera, GaussianRenderer
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
    )

    mean_abs_error = float(np.mean(np.abs(gpu_image - cpu_image)))
    assert mean_abs_error < 5e-3
