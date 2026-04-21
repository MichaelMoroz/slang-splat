from __future__ import annotations

from types import SimpleNamespace

from src.utility import clear_debug_resource_allocations, register_debug_resource
from src.viewer.buffer_debug import collect_resource_debug_snapshot, format_resource_bytes, format_resource_debug_log


def test_resource_debug_snapshot_deduplicates_and_sorts_largest_first() -> None:
    clear_debug_resource_allocations()
    class WeakResource:
        pass

    large_buffer = object()
    small_buffer = object()
    texture = object()
    unowned = WeakResource()
    non_weakref_unowned = object()
    register_debug_resource(small_buffer, kind="Buffer", name="small", byte_size=32, usage="rw")
    register_debug_resource(large_buffer, kind="Buffer", name="large", byte_size=128, usage="rw")
    register_debug_resource(texture, kind="Texture", name="target", byte_size=64, usage="srv")
    register_debug_resource(unowned, kind="Buffer", name="unowned", byte_size=16, usage="rw")
    register_debug_resource(non_weakref_unowned, kind="Buffer", name="non_weakref_unowned", byte_size=8, usage="rw")
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

    assert [row.name for row in snapshot.rows] == ["large", "target", "small", "unowned"]
    assert [row.details for row in snapshot.rows] == ["32 x 4 B units", "", "8 x 4 B units", "4 x 4 B units"]
    assert snapshot.rows[-1].owner == "debug_registry.unowned"
    assert "non_weakref_unowned" not in {row.name for row in snapshot.rows}
    assert snapshot.total_consumption == 240
    assert snapshot.buffer_count == 3
    assert snapshot.buffer_total == 176
    assert snapshot.buffer_mean == 176.0 / 3.0
    assert snapshot.buffer_median == 32.0
    assert snapshot.texture_count == 1
    assert snapshot.texture_total == 64
    assert format_resource_bytes(1536) == "1.50 KiB"
    log = format_resource_debug_log(snapshot)
    assert "Total Consumption: 240 B (240 bytes)" in log
    assert f"128 B (128 bytes)\tBuffer\t32 x 4 B units\tlarge\t{snapshot.rows[0].owner}\trw" in log
    clear_debug_resource_allocations()


def test_resource_debug_log_includes_process_vram_delta() -> None:
    snapshot = collect_resource_debug_snapshot(SimpleNamespace(s=SimpleNamespace()), include_process_vram=False)
    snapshot = type(snapshot)(
        rows=snapshot.rows,
        total_consumption=128,
        buffer_count=1,
        buffer_total=128,
        buffer_mean=128.0,
        buffer_median=128.0,
        texture_count=0,
        texture_total=0,
        process_vram=512,
        process_vram_delta=384,
        process_vram_source="test",
    )

    log = format_resource_debug_log(snapshot)

    assert "Process Dedicated VRAM: 512 B (512 bytes) [test]" in log
    assert "Untracked / Driver Reserved: 384 B (384 bytes)" in log
