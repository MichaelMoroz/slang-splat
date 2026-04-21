from __future__ import annotations

from types import SimpleNamespace

from src.utility import clear_debug_resource_allocations, register_debug_resource
from src.viewer.buffer_debug import collect_resource_debug_snapshot, format_resource_bytes, format_resource_debug_log


def test_resource_debug_snapshot_deduplicates_and_sorts_largest_first() -> None:
    clear_debug_resource_allocations()
    large_buffer = object()
    small_buffer = object()
    texture = object()
    register_debug_resource(small_buffer, kind="Buffer", name="small", byte_size=32, usage="rw")
    register_debug_resource(large_buffer, kind="Buffer", name="large", byte_size=128, usage="rw")
    register_debug_resource(texture, kind="Texture", name="target", byte_size=64, usage="srv")
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            renderer=SimpleNamespace(small=small_buffer, nested={"large": large_buffer}),
            training_renderer=None,
            debug_renderer=None,
            trainer=SimpleNamespace(shared=large_buffer),
            viewport_texture=texture,
        )
    )

    snapshot = collect_resource_debug_snapshot(viewer)

    assert [row.name for row in snapshot.rows] == ["large", "target", "small"]
    assert [row.details for row in snapshot.rows] == ["32 x 4 B units", "", "8 x 4 B units"]
    assert snapshot.total_consumption == 224
    assert snapshot.buffer_count == 2
    assert snapshot.buffer_total == 160
    assert snapshot.buffer_mean == 80.0
    assert snapshot.buffer_median == 80.0
    assert snapshot.texture_count == 1
    assert snapshot.texture_total == 64
    assert format_resource_bytes(1536) == "1.50 KiB"
    log = format_resource_debug_log(snapshot)
    assert "Total Consumption: 224 B (224 bytes)" in log
    assert f"128 B (128 bytes)\tBuffer\t32 x 4 B units\tlarge\t{snapshot.rows[0].owner}\trw" in log
    clear_debug_resource_allocations()
