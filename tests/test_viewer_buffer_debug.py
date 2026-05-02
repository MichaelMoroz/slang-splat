from __future__ import annotations

from types import SimpleNamespace

from src.viewer import buffer_debug
from src.utility import clear_debug_resource_allocations, debug_resource_allocations, register_debug_resource
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
    assert [allocation.name for _, allocation in debug_resource_allocations()] == ["unowned"]
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


def test_resource_debug_snapshot_includes_unregistered_device_backed_objects() -> None:
    clear_debug_resource_allocations()

    class FakeMemoryUsage:
        def __init__(self, device: int) -> None:
            self.device = device

    class FakeTextureUsage:
        def __str__(self) -> str:
            return "TextureUsage.shader_resource"

    class FakeTexture:
        def __init__(self) -> None:
            self.memory_usage = FakeMemoryUsage(256)
            self.usage = FakeTextureUsage()
            self.width = 8
            self.height = 4
            self.depth = 1
            self.array_length = 1
            self.mip_count = 1
            self.format = "Format.rgba8_unorm_srgb"

    viewer = SimpleNamespace(
        device=SimpleNamespace(cached_texture=FakeTexture()),
        s=SimpleNamespace(renderer=None, training_renderer=None, debug_renderer=None, trainer=None),
    )

    snapshot = collect_resource_debug_snapshot(viewer)

    assert len(snapshot.rows) == 1
    row = snapshot.rows[0]
    assert row.kind == "Texture"
    assert row.name == "unregistered.FakeTexture"
    assert row.owner == "viewer.device.cached_texture"
    assert row.byte_size == 256
    assert row.details == "8x4 rgba8_unorm_srgb"
    assert row.usage == "shader_resource"


def test_counter_prefix_from_luid_matches_windows_counter_format() -> None:
    prefix = buffer_debug._counter_prefix_from_luid([0x6C, 0xDF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

    assert prefix == "luid_0x00000000_0x0000df6c"


def test_query_total_device_vram_used_uses_windows_adapter_counter(monkeypatch) -> None:
    buffer_debug._DEVICE_VRAM_CACHE.clear()
    buffer_debug._DEVICE_VRAM_IN_FLIGHT.clear()
    monkeypatch.setattr(buffer_debug.sys, "platform", "win32")
    commands: list[list[str]] = []

    def _run(args, capture_output, text, timeout, check):
        commands.append(list(args))
        return SimpleNamespace(returncode=0, stdout="2147483648\n")

    monkeypatch.setattr(buffer_debug.subprocess, "run", _run)
    device = SimpleNamespace(
        info=SimpleNamespace(adapter_name="GPU X", api_name="Vulkan"),
        desc=SimpleNamespace(adapter_luid=[0x6C, 0xDF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
        report_heaps=lambda: [],
    )

    used, source = buffer_debug.query_total_device_vram_used(device)

    assert used == 2147483648
    assert source == "Windows GPU Adapter Memory"
    assert commands and "luid_0x00000000_0x0000df6c" in commands[0][-1]


def test_query_total_device_vram_used_cached_starts_background_refresh(monkeypatch) -> None:
    buffer_debug._DEVICE_VRAM_CACHE.clear()
    buffer_debug._DEVICE_VRAM_IN_FLIGHT.clear()
    monkeypatch.setattr(buffer_debug, "_query_device_heap_usage_bytes", lambda _device: None)
    launches: list[buffer_debug.DeviceVramQueryContext] = []
    monkeypatch.setattr(buffer_debug, "_start_device_vram_refresh", lambda query_context: launches.append(query_context))
    device = SimpleNamespace(
        info=SimpleNamespace(adapter_name="GPU X", api_name="Vulkan"),
        desc=SimpleNamespace(adapter_luid=[0x6C, 0xDF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
        report_heaps=lambda: [],
    )

    used, source = buffer_debug.query_total_device_vram_used_cached(device)

    assert used is None
    assert source == ""
    assert launches == [buffer_debug.DeviceVramQueryContext(cache_key="Vulkan|GPU X|luid_0x00000000_0x0000df6c", adapter_prefixes=("luid_0x00000000_0x0000df6c",))]


def test_split_resource_usage_separates_dataset_textures_from_app_resources() -> None:
    snapshot = buffer_debug.ResourceDebugSnapshot(
        rows=(
            buffer_debug.ResourceDebugRow("Texture", "viewer.dataset_texture", "viewer.trainer.frame_targets_native[0]", 256, "rgba8", "srv", 1),
            buffer_debug.ResourceDebugRow("Texture", "viewer.dataset_texture_bc7", "viewer.state.colmap_import_textures[1]", 512, "bc7", "srv", 2),
            buffer_debug.ResourceDebugRow("Buffer", "renderer.buf", "viewer.main_renderer.buf", 1024, "buf", "rw", 3),
            buffer_debug.ResourceDebugRow("Texture", "viewer.viewport_texture", "viewer.state.viewport_texture", 2048, "rgba32", "srv", 4),
        ),
        total_consumption=3840,
        buffer_count=1,
        buffer_total=1024,
        buffer_mean=1024.0,
        buffer_median=1024.0,
        texture_count=3,
        texture_total=2816,
    )

    usage = buffer_debug.split_resource_usage(snapshot)

    assert usage.dataset_bytes == 768
    assert usage.app_bytes == 3072
    assert usage.total_bytes == 3840


def test_query_total_device_vram_capacity_uses_cached_result(monkeypatch) -> None:
    buffer_debug._DEVICE_VRAM_CAPACITY_CACHE.clear()
    calls: list[object] = []
    monkeypatch.setattr(buffer_debug, "_query_total_device_vram_capacity", lambda device: calls.append(device) or (8 * 1024**3, "DXGI Adapter Desc"))
    device = SimpleNamespace(info=SimpleNamespace(adapter_name="GPU X", api_name="Vulkan"), desc=SimpleNamespace(adapter_luid=None))

    total_a, source_a = buffer_debug.query_total_device_vram_capacity(device)
    total_b, source_b = buffer_debug.query_total_device_vram_capacity(device)

    assert total_a == 8 * 1024**3
    assert total_b == 8 * 1024**3
    assert source_a == "DXGI Adapter Desc"
    assert source_b == "DXGI Adapter Desc"
    assert calls == [device]


def test_device_adapter_counter_prefixes_fall_back_to_enumerated_adapter_name(monkeypatch) -> None:
    monkeypatch.setattr(
        buffer_debug.spy.Device,
        "enumerate_adapters",
        lambda: [
            SimpleNamespace(name="Other GPU", luid=[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
            SimpleNamespace(name="GPU X", luid=[0x6C, 0xDF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
        ],
    )
    device = SimpleNamespace(
        info=SimpleNamespace(adapter_name="GPU X"),
        desc=SimpleNamespace(adapter_luid=None),
    )

    prefixes = buffer_debug._device_adapter_counter_prefixes(device)

    assert prefixes == ("luid_0x00000000_0x0000df6c",)
