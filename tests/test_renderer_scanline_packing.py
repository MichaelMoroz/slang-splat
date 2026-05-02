from __future__ import annotations

import numpy as np

from reference_impls.reference_cpu import build_tile_ranges, sort_key_values
from src.renderer import Camera, GaussianRenderer
from src.scene import GaussianScene


def _make_scene(count: int, seed: int = 0) -> GaussianScene:
    rng = np.random.default_rng(seed)
    positions = np.zeros((count, 3), dtype=np.float32)
    positions[:, 0] = rng.uniform(-1.2, 1.2, size=count).astype(np.float32)
    positions[:, 1] = rng.uniform(-0.9, 0.9, size=count).astype(np.float32)
    positions[:, 2] = np.linspace(-1.0, 1.0, count, dtype=np.float32)
    scales = np.log(np.full((count, 3), 0.04, dtype=np.float32))
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


def test_packed_scanline_work_items_reconstruct_gpu_tile_lists(device):
    scene = _make_scene(32, seed=1)
    camera = Camera.look_at(position=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0), near=0.1, far=20.0)
    renderer = GaussianRenderer(device, width=96, height=96, radius_scale=1.8, list_capacity_multiplier=32)
    debug = renderer.debug_pipeline_data(scene, camera)

    scanline_counter = int(renderer._read_array(renderer._work_buffers["scanline_counter"], np.uint32, 1)[0])
    scanline_items = renderer._read_array(renderer._work_buffers["scanline_work_items"], np.uint32, scanline_counter, width=2)
    scanline_offsets = renderer._read_array(renderer._work_buffers["scanline_tile_offsets"], np.uint32, scanline_counter)
    generated_entries = int(debug["generated_entries"])
    start_tile_mask = (1 << GaussianRenderer._SCANLINE_WORK_ITEM_START_TILE_BITS) - 1
    line_count_mask = (1 << GaussianRenderer._SCANLINE_WORK_ITEM_LINE_COUNT_BITS) - 1

    reconstructed_keys = np.zeros((generated_entries,), dtype=np.uint32)
    reconstructed_values = np.zeros((generated_entries,), dtype=np.uint32)
    for raw_item, base_offset in zip(scanline_items, scanline_offsets, strict=False):
        splat_id = int(raw_item[0])
        packed = int(raw_item[1])
        start_tile_id = packed & start_tile_mask
        line_count = (packed >> GaussianRenderer._SCANLINE_WORK_ITEM_START_TILE_BITS) & line_count_mask
        stride = 1 if (packed & 0x80000000) else renderer.tile_width
        for item_index in range(line_count):
            reconstructed_keys[base_offset + item_index] = start_tile_id + item_index * stride
            reconstructed_values[base_offset + item_index] = splat_id

    sorted_keys, sorted_values = sort_key_values(reconstructed_keys, reconstructed_values, generated_entries)
    reconstructed_ranges = build_tile_ranges(sorted_keys, generated_entries, renderer.tile_count)

    np.testing.assert_array_equal(debug["keys"], sorted_keys)
    np.testing.assert_array_equal(debug["values"], sorted_values)
    np.testing.assert_array_equal(debug["tile_ranges"], reconstructed_ranges)


def test_packed_scanline_work_item_limits_apply_to_capacity_growth(device):
    renderer = GaussianRenderer(device, width=64, height=64)
    over_limit_width = GaussianRenderer._SCANLINE_WORK_ITEM_MAX_LINE_COUNT * renderer.tile_size + 1

    try:
        renderer.ensure_render_capacity(over_limit_width, renderer.tile_size)
    except ValueError as exc:
        assert "Packed scanline work items" in str(exc)
    else:
        raise AssertionError("Expected packed scanline limit error.")


def test_scanline_capacity_is_sized_independently_from_list_capacity(device):
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=1024)

    renderer.bind_scene_count(1)

    assert renderer._max_list_entries == 1024
    assert renderer._max_scanline_entries == 32


def test_scanline_overflow_updates_pending_growth_independently(device):
    renderer = GaussianRenderer(device, width=64, height=64, list_capacity_multiplier=512)

    renderer.bind_scene_count(1)
    renderer._ensure_counter_readback_ring()
    renderer._counter_readback_frame_id = 2
    scanline_capacity = renderer._max_scanline_entries
    overflow_scanlines = scanline_capacity + 48
    renderer._counter_readback_capacity[0] = (renderer._max_list_entries, scanline_capacity)
    renderer._counter_readback_ring[0].copy_from_numpy(np.array([400, overflow_scanlines], dtype=np.uint32))

    renderer._update_delayed_counter_stats()

    assert renderer._delayed_list_overflow is False
    assert renderer._delayed_scanline_overflow is True
    assert renderer._pending_min_list_entries == overflow_scanlines
    assert renderer._pending_min_scanline_entries == overflow_scanlines

    renderer._ensure_work_buffers(1, renderer._pending_min_list_entries, renderer._pending_min_scanline_entries)

    assert renderer._max_scanline_entries >= overflow_scanlines
    assert renderer._max_scanline_entries < renderer._max_list_entries