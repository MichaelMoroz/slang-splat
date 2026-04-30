from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from statistics import median
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import slangpy as spy

from ..utility import ResourceAllocation, debug_resource_allocations, resource_allocation

_MAX_WALK_DEPTH = 12
_MAX_WALK_NODES = 20000
_PROCESS_VRAM_CACHE_SECONDS = 5.0
_DEVICE_VRAM_CACHE_SECONDS = 2.0
_DEVICE_VRAM_CAPACITY_CACHE_SECONDS = 300.0
_STATE_RESOURCE_ATTRS = (
    "viewport_texture",
    "loss_debug_texture",
    "debug_target_texture",
    "debug_present_texture",
    "debug_dssim_blur",
    "debug_dssim_moments",
    "debug_dssim_blurred_moments",
)
_SKIP_ATTRS = {
    "app",
    "callbacks",
    "ctx",
    "device",
    "frames",
    "loss_debug_view_options",
    "renderer",
    "toolkit",
    "ui",
}


@dataclass(frozen=True, slots=True)
class ResourceDebugRow:
    kind: str
    name: str
    owner: str
    byte_size: int
    details: str
    usage: str
    order: int


@dataclass(frozen=True, slots=True)
class ResourceDebugSnapshot:
    rows: tuple[ResourceDebugRow, ...]
    total_consumption: int
    buffer_count: int
    buffer_total: int
    buffer_mean: float
    buffer_median: float
    texture_count: int
    texture_total: int
    process_vram: int | None = None
    process_vram_delta: int | None = None
    process_vram_source: str = ""


@dataclass(frozen=True, slots=True)
class ResourceUsageSplit:
    dataset_bytes: int
    app_bytes: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class DeviceVramQueryContext:
    cache_key: str
    adapter_prefixes: tuple[str, ...] = ()


_PROCESS_VRAM_CACHE: tuple[float, int | None, str] = (0.0, None, "")
_DEVICE_VRAM_CACHE: dict[str, tuple[float, int | None, str]] = {}
_DEVICE_VRAM_CAPACITY_CACHE: dict[str, tuple[float, int | None, str]] = {}
_DEVICE_VRAM_CACHE_LOCK = threading.Lock()
_DEVICE_VRAM_IN_FLIGHT: set[str] = set()
_PROCESS_VRAM_CACHE_LOCK = threading.Lock()


def format_resource_bytes(byte_count: float | int) -> str:
    value = float(max(byte_count, 0.0))
    for suffix in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or suffix == "GiB":
            return f"{value:.0f} {suffix}" if suffix == "B" else f"{value:.2f} {suffix}"
        value /= 1024.0
    return f"{value:.2f} GiB"


def query_total_device_vram_used(device: object) -> tuple[int | None, str]:
    cache_key = _device_vram_cache_key(device)
    if not cache_key:
        return None, ""
    now = time.monotonic()
    cached = _read_cached_device_vram(cache_key)
    if cached is not None and now - cached[0] < _DEVICE_VRAM_CACHE_SECONDS:
        return cached[1], cached[2]
    value, source = _query_total_device_vram_used(device)
    value = None if value is None else max(int(value), 0)
    _store_cached_device_vram(cache_key, value, source, now=now)
    return value, source


def query_total_device_vram_capacity(device: object) -> tuple[int | None, str]:
    cache_key = _device_vram_cache_key(device)
    if not cache_key:
        return None, ""
    now = time.monotonic()
    with _DEVICE_VRAM_CACHE_LOCK:
        cached = _DEVICE_VRAM_CAPACITY_CACHE.get(cache_key)
    if cached is not None and now - cached[0] < _DEVICE_VRAM_CAPACITY_CACHE_SECONDS:
        return cached[1], cached[2]
    value, source = _query_total_device_vram_capacity(device)
    value = None if value is None else max(int(value), 0)
    with _DEVICE_VRAM_CACHE_LOCK:
        _DEVICE_VRAM_CAPACITY_CACHE[cache_key] = (now, value, source)
    return value, source


def split_resource_usage(snapshot: ResourceDebugSnapshot) -> ResourceUsageSplit:
    dataset_bytes = int(
        sum(
            max(int(row.byte_size), 0)
            for row in snapshot.rows
            if row.kind == "Texture" and str(row.name).startswith("viewer.dataset_texture")
        )
    )
    total_bytes = max(int(snapshot.total_consumption), 0)
    return ResourceUsageSplit(dataset_bytes=dataset_bytes, app_bytes=max(total_bytes - dataset_bytes, 0), total_bytes=total_bytes)


def query_total_device_vram_used_cached(device: object) -> tuple[int | None, str]:
    cache_key = _device_vram_cache_key(device)
    if not cache_key:
        return None, ""
    now = time.monotonic()
    cached = _read_cached_device_vram(cache_key)
    if cached is not None and now - cached[0] < _DEVICE_VRAM_CACHE_SECONDS:
        return cached[1], cached[2]
    heap_usage = _query_device_heap_usage_bytes(device)
    if heap_usage is not None:
        source = "Slangpy Device Heaps"
        _store_cached_device_vram(cache_key, heap_usage, source, now=now)
        return int(heap_usage), source
    query_context = _device_vram_query_context(device, cache_key)
    if query_context is None:
        return (cached[1], cached[2]) if cached is not None else (None, "")
    should_launch = False
    with _DEVICE_VRAM_CACHE_LOCK:
        if cache_key not in _DEVICE_VRAM_IN_FLIGHT:
            _DEVICE_VRAM_IN_FLIGHT.add(cache_key)
            should_launch = True
    if should_launch:
        _start_device_vram_refresh(query_context)
    return (cached[1], cached[2]) if cached is not None else (None, "")


def write_resource_debug_log(snapshot: ResourceDebugSnapshot, directory: Path | str | None = None) -> Path:
    root = Path(__file__).resolve().parents[2]
    log_dir = root / "temp" / "resource_logs" if directory is None else Path(directory)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"resource_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path.write_text(format_resource_debug_log(snapshot), encoding="utf-8")
    return path


def format_resource_debug_log(snapshot: ResourceDebugSnapshot) -> str:
    rows = tuple(sorted(snapshot.rows, key=lambda item: (-item.byte_size, item.order)))
    lines = [
        "Slang Splat Resource Debug Log",
        "",
        f"Total Consumption: {format_resource_bytes(snapshot.total_consumption)} ({snapshot.total_consumption:,} bytes)",
        (
            f"Buffers: {snapshot.buffer_count:,} | total={format_resource_bytes(snapshot.buffer_total)} ({snapshot.buffer_total:,} bytes) | "
            f"mean={format_resource_bytes(snapshot.buffer_mean)} | median={format_resource_bytes(snapshot.buffer_median)}"
        ),
        f"Textures: {snapshot.texture_count:,} | total={format_resource_bytes(snapshot.texture_total)} ({snapshot.texture_total:,} bytes)",
    ]
    if snapshot.process_vram is not None:
        source = f" [{snapshot.process_vram_source}]" if snapshot.process_vram_source else ""
        delta = 0 if snapshot.process_vram_delta is None else snapshot.process_vram_delta
        lines.extend(
            (
                f"Process Dedicated VRAM: {format_resource_bytes(snapshot.process_vram)} ({snapshot.process_vram:,} bytes){source}",
                f"Untracked / Driver Reserved: {format_resource_bytes(delta)} ({delta:,} bytes)",
            )
        )
    lines.extend(("", "Duplicate-Looking Groups", "Count\tTotal Size\tSingle Size\tType\tDetails\tName"))
    duplicate_lines = _duplicate_group_lines(rows)
    lines.extend(duplicate_lines if duplicate_lines else ("<none>",))
    lines.extend(
        (
            "",
            "Resource Table",
            "Size\tType\tDetails\tName\tOwner\tUsage",
        )
    )
    for row in rows:
        lines.append(
            "\t".join(
                (
                    f"{format_resource_bytes(row.byte_size)} ({row.byte_size:,} bytes)",
                    row.kind,
                    row.details,
                    row.name,
                    row.owner,
                    row.usage,
                )
            )
        )
    return "\n".join(lines) + "\n"


def _duplicate_group_lines(rows: tuple[ResourceDebugRow, ...]) -> tuple[str, ...]:
    groups: dict[tuple[str, str, int, str], list[ResourceDebugRow]] = defaultdict(list)
    for row in rows:
        groups[(row.kind, row.name, row.byte_size, row.details)].append(row)
    duplicate_groups = sorted(
        (group for group in groups.values() if len(group) > 1),
        key=lambda group: (-len(group) * group[0].byte_size, -len(group), group[0].name),
    )
    return tuple(
        "\t".join(
            (
                f"{len(group):,}",
                f"{format_resource_bytes(len(group) * group[0].byte_size)} ({len(group) * group[0].byte_size:,} bytes)",
                f"{format_resource_bytes(group[0].byte_size)} ({group[0].byte_size:,} bytes)",
                group[0].kind,
                group[0].details,
                group[0].name,
            )
        )
        for group in duplicate_groups
    )


def collect_resource_debug_snapshot(viewer: object, *, include_process_vram: bool = False) -> ResourceDebugSnapshot:
    found: dict[int, tuple[ResourceAllocation, list[str]]] = {}
    visited: set[int] = set()
    node_count = [0]
    for root_name, root in _viewer_resource_roots(viewer):
        _walk_resource_graph(root, root_name, found, visited, node_count, _MAX_WALK_DEPTH)
    for resource, allocation in debug_resource_allocations():
        found.setdefault(id(resource), (allocation, ["debug_registry.unowned"]))
    rows = tuple(
        sorted(
            (
                ResourceDebugRow(
                    kind=allocation.kind,
                    name=allocation.name,
                    owner=min(paths, key=len),
                    byte_size=allocation.byte_size,
                    details=allocation.details,
                    usage=allocation.usage,
                    order=allocation.order,
                )
                for allocation, paths in found.values()
            ),
            key=lambda row: (-row.byte_size, row.order),
        )
    )
    process_vram, source = _process_vram_snapshot() if include_process_vram else (None, "")
    return _snapshot_from_rows(rows, process_vram=process_vram, process_vram_source=source)


def _viewer_resource_roots(viewer: object) -> tuple[tuple[str, object], ...]:
    state = getattr(viewer, "s", None)
    if state is None:
        return ()
    roots: list[tuple[str, object]] = []
    for attr, label in (
        ("renderer", "main_renderer"),
        ("training_renderer", "training_renderer"),
        ("debug_renderer", "debug_renderer"),
        ("trainer", "trainer"),
    ):
        value = getattr(state, attr, None)
        if value is not None:
            roots.append((f"viewer.{label}", value))
    state_resources = {attr: getattr(state, attr, None) for attr in _STATE_RESOURCE_ATTRS if getattr(state, attr, None) is not None}
    progress = getattr(state, "colmap_import_progress", None)
    if progress is not None:
        state_resources["colmap_import_textures"] = getattr(progress, "native_textures", ())
    if state_resources:
        roots.append(("viewer.state", SimpleNamespace(**state_resources)))
    return tuple(roots)


def _snapshot_from_rows(rows: tuple[ResourceDebugRow, ...], *, process_vram: int | None = None, process_vram_source: str = "") -> ResourceDebugSnapshot:
    buffer_sizes = [row.byte_size for row in rows if row.kind == "Buffer"]
    texture_sizes = [row.byte_size for row in rows if row.kind == "Texture"]
    buffer_total = int(sum(buffer_sizes))
    texture_total = int(sum(texture_sizes))
    total = buffer_total + texture_total
    delta = None if process_vram is None else max(int(process_vram) - total, 0)
    return ResourceDebugSnapshot(
        rows=rows,
        total_consumption=total,
        buffer_count=len(buffer_sizes),
        buffer_total=buffer_total,
        buffer_mean=float(buffer_total / len(buffer_sizes)) if buffer_sizes else 0.0,
        buffer_median=float(median(buffer_sizes)) if buffer_sizes else 0.0,
        texture_count=len(texture_sizes),
        texture_total=texture_total,
        process_vram=process_vram,
        process_vram_delta=delta,
        process_vram_source=process_vram_source,
    )


def _process_vram_snapshot() -> tuple[int | None, str]:
    now = time.monotonic()
    cached_at, cached_value, cached_source = _read_cached_process_vram() or (0.0, None, "")
    if cached_at > 0.0 and now - cached_at < _PROCESS_VRAM_CACHE_SECONDS:
        return cached_value, cached_source
    value, source = _query_process_vram()
    _store_process_vram_cache(value, source, now=now)
    return value, source


def _read_cached_process_vram() -> tuple[float, int | None, str] | None:
    with _PROCESS_VRAM_CACHE_LOCK:
        return _PROCESS_VRAM_CACHE


def _store_process_vram_cache(value: int | None, source: str, *, now: float | None = None) -> None:
    timestamp = time.monotonic() if now is None else float(now)
    with _PROCESS_VRAM_CACHE_LOCK:
        global _PROCESS_VRAM_CACHE
        _PROCESS_VRAM_CACHE = (timestamp, None if value is None else int(value), str(source))


def _read_cached_device_vram(cache_key: str) -> tuple[float, int | None, str] | None:
    with _DEVICE_VRAM_CACHE_LOCK:
        return _DEVICE_VRAM_CACHE.get(cache_key)


def _store_cached_device_vram(cache_key: str, value: int | None, source: str, *, now: float | None = None) -> None:
    timestamp = time.monotonic() if now is None else float(now)
    with _DEVICE_VRAM_CACHE_LOCK:
        _DEVICE_VRAM_CACHE[cache_key] = (timestamp, None if value is None else int(value), str(source))


def _device_vram_query_context(device: object, cache_key: str) -> DeviceVramQueryContext | None:
    if sys.platform == "win32":
        prefixes = _device_adapter_counter_prefixes(device)
        if len(prefixes) > 0:
            return DeviceVramQueryContext(cache_key=cache_key, adapter_prefixes=prefixes)
    return None


def _start_device_vram_refresh(query_context: DeviceVramQueryContext) -> None:
    thread = threading.Thread(target=_device_vram_refresh_worker, args=(query_context,), name=f"device-vram-{query_context.cache_key}", daemon=True)
    thread.start()


def _device_vram_refresh_worker(query_context: DeviceVramQueryContext) -> None:
    try:
        value, source = _query_total_device_vram_used_from_context(query_context)
        value = None if value is None else max(int(value), 0)
        _store_cached_device_vram(query_context.cache_key, value, source)
    finally:
        with _DEVICE_VRAM_CACHE_LOCK:
            _DEVICE_VRAM_IN_FLIGHT.discard(query_context.cache_key)


def _device_vram_cache_key(device: object) -> str:
    info = getattr(device, "info", None)
    desc = getattr(device, "desc", None)
    adapter_name = str(getattr(info, "adapter_name", "") or "").strip()
    api_name = str(getattr(info, "api_name", "") or "").strip()
    luid_prefix = _counter_prefix_from_luid(getattr(desc, "adapter_luid", None))
    parts = tuple(part for part in (api_name, adapter_name, luid_prefix) if part)
    return "|".join(parts)


def _query_total_device_vram_used(device: object) -> tuple[int | None, str]:
    heap_usage = _query_device_heap_usage_bytes(device)
    if heap_usage is not None:
        return heap_usage, "Slangpy Device Heaps"
    query_context = _device_vram_query_context(device, _device_vram_cache_key(device))
    if query_context is None:
        return None, ""
    return _query_total_device_vram_used_from_context(query_context)


def _query_total_device_vram_used_from_context(query_context: DeviceVramQueryContext) -> tuple[int | None, str]:
    if len(query_context.adapter_prefixes) > 0:
        value = _query_windows_device_dedicated_vram_from_prefixes(query_context.adapter_prefixes)
        return value, "Windows GPU Adapter Memory" if value is not None else ""
    return None, ""


def _query_total_device_vram_capacity(device: object) -> tuple[int | None, str]:
    if sys.platform == "win32":
        value = _query_windows_device_total_vram_capacity(device)
        return value, "DXGI Adapter Desc" if value is not None else ""
    return None, ""


def _query_device_heap_usage_bytes(device: object) -> int | None:
    try:
        heaps = tuple(device.report_heaps())
    except Exception:
        return None
    if len(heaps) == 0:
        return None
    total = 0
    found = False
    for heap in heaps:
        used_bytes = _heap_used_bytes(heap)
        if used_bytes is None:
            continue
        total += max(int(used_bytes), 0)
        found = True
    return total if found else None


def _heap_used_bytes(heap: object) -> int | None:
    if isinstance(heap, dict):
        for key in ("used_bytes", "used_size", "bytes_used"):
            if key in heap:
                try:
                    return int(heap[key])
                except Exception:
                    return None
    for attr in ("used_bytes", "used_size", "bytes_used"):
        try:
            value = getattr(heap, attr)
        except Exception:
            continue
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def _query_windows_device_dedicated_vram(device: object) -> int | None:
    prefixes = _device_adapter_counter_prefixes(device)
    return _query_windows_device_dedicated_vram_from_prefixes(prefixes)


def _query_windows_device_total_vram_capacity(device: object) -> int | None:
    target_luid = _device_adapter_luid_bytes(device)
    if target_luid is None:
        return None
    return _query_windows_dxgi_dedicated_video_memory(target_luid)


def _query_windows_device_dedicated_vram_from_prefixes(prefixes: tuple[str, ...]) -> int | None:
    if len(prefixes) == 0:
        return None
    pattern = "|".join(re.escape(prefix) for prefix in prefixes)
    command = (
        f"$pattern='^({pattern})_phys_[0-9]+$'; "
        "$samples=(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' -ErrorAction Stop).CounterSamples; "
        "[int64](($samples | Where-Object { $_.InstanceName -match $pattern } | Measure-Object -Property CookedValue -Sum).Sum)"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(lines) == 0:
            return None
        value = int(float(lines[-1]))
    except Exception:
        return None
    return max(value, 0)


def _device_adapter_counter_prefixes(device: object) -> tuple[str, ...]:
    luid = _device_adapter_luid_bytes(device)
    if luid is None:
        return ()
    prefix = _counter_prefix_from_luid(luid)
    return (prefix,) if prefix else ()


def _device_adapter_luid_bytes(device: object) -> bytes | None:
    direct_luid = _luid_bytes(getattr(getattr(device, "desc", None), "adapter_luid", None))
    if direct_luid is not None:
        return direct_luid
    adapter_name = str(getattr(getattr(device, "info", None), "adapter_name", "") or "").strip().casefold()
    if not adapter_name:
        return None
    try:
        adapters = tuple(spy.Device.enumerate_adapters())
    except Exception:
        return None
    for adapter in adapters:
        if str(getattr(adapter, "name", "") or "").strip().casefold() != adapter_name:
            continue
        luid = _luid_bytes(getattr(adapter, "luid", None))
        if luid is not None:
            return luid
    return None


def _counter_prefix_from_luid(luid: object) -> str:
    raw = _luid_bytes(luid)
    if raw is None:
        return ""
    low = int.from_bytes(raw[:4], "little", signed=False)
    high = int.from_bytes(raw[4:8], "little", signed=False)
    return f"luid_0x{high:08x}_0x{low:08x}"


def _luid_bytes(luid: object) -> bytes | None:
    if luid is None:
        return None
    try:
        raw = bytes(int(value) & 0xFF for value in luid)
    except Exception:
        return None
    if len(raw) < 8:
        return None
    return raw[:8]


def _query_windows_dxgi_dedicated_video_memory(target_luid: bytes) -> int | None:
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return None

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    class DXGI_ADAPTER_DESC1(ctypes.Structure):
        _fields_ = [
            ("Description", ctypes.c_wchar * 128),
            ("VendorId", ctypes.c_uint),
            ("DeviceId", ctypes.c_uint),
            ("SubSysId", ctypes.c_uint),
            ("Revision", ctypes.c_uint),
            ("DedicatedVideoMemory", ctypes.c_size_t),
            ("DedicatedSystemMemory", ctypes.c_size_t),
            ("SharedSystemMemory", ctypes.c_size_t),
            ("AdapterLuid", LUID),
            ("Flags", ctypes.c_uint),
        ]

    def _hresult_failed(result: int) -> bool:
        return int(result) < 0

    def _hresult_code(result: int) -> int:
        return int(result) & 0xFFFFFFFF

    def _com_method(ptr: ctypes.c_void_p, index: int, restype, *argtypes):
        vtable = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        func_type = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
        return func_type(vtable[index])

    def _release(ptr: ctypes.c_void_p) -> None:
        if not ptr:
            return
        _com_method(ptr, 2, ctypes.c_ulong)(ptr)

    create_factory = ctypes.windll.dxgi.CreateDXGIFactory1
    create_factory.argtypes = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
    create_factory.restype = ctypes.c_long
    factory_iid = GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
    factory_ptr = ctypes.c_void_p()
    result = create_factory(ctypes.byref(factory_iid), ctypes.byref(factory_ptr))
    if _hresult_failed(result):
        return None
    dxgi_error_not_found = 0x887A0002
    try:
        enum_adapters1 = _com_method(factory_ptr, 12, ctypes.c_long, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))
        adapter_index = 0
        while True:
            adapter_ptr = ctypes.c_void_p()
            result = enum_adapters1(factory_ptr, adapter_index, ctypes.byref(adapter_ptr))
            if _hresult_code(result) == dxgi_error_not_found:
                break
            if _hresult_failed(result):
                return None
            try:
                get_desc1 = _com_method(adapter_ptr, 10, ctypes.c_long, ctypes.POINTER(DXGI_ADAPTER_DESC1))
                desc = DXGI_ADAPTER_DESC1()
                if _hresult_failed(get_desc1(adapter_ptr, ctypes.byref(desc))):
                    continue
                low = int(desc.AdapterLuid.LowPart) & 0xFFFFFFFF
                high = int(desc.AdapterLuid.HighPart) & 0xFFFFFFFF
                adapter_luid = low.to_bytes(4, "little") + high.to_bytes(4, "little")
                if adapter_luid == target_luid:
                    return int(desc.DedicatedVideoMemory)
            finally:
                _release(adapter_ptr)
            adapter_index += 1
    finally:
        _release(factory_ptr)
    return None


def _query_process_vram() -> tuple[int | None, str]:
    if sys.platform == "win32":
        value = _query_windows_process_dedicated_vram()
        return value, "Windows GPU Process Memory" if value is not None else ""
    value = _query_nvidia_smi_process_vram()
    return value, "nvidia-smi" if value is not None else ""


def _query_windows_process_dedicated_vram() -> int | None:
    pid = int(os.getpid())
    command = (
        f"$pidValue={pid}; "
        "$samples=(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' -ErrorAction Stop).CounterSamples; "
        "[int64](($samples | Where-Object { $_.InstanceName -like \"pid_${pidValue}_*\" } | "
        "Measure-Object -Property CookedValue -Sum).Sum)"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        value = int(float(result.stdout.strip().splitlines()[-1]))
    except Exception:
        return None
    return max(value, 0)


def _query_nvidia_smi_process_vram() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    total_mib = 0
    pid = str(os.getpid())
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2 or parts[0] != pid:
            continue
        if parts[1].upper() == "N/A":
            return None
        try:
            total_mib += int(parts[1])
        except ValueError:
            return None
    return total_mib * 1024 * 1024 if total_mib > 0 else None


def _walk_resource_graph(value: object, path: str, found: dict[int, tuple[ResourceAllocation, list[str]]], visited: set[int], node_count: list[int], depth: int) -> None:
    allocation = resource_allocation(value)
    if allocation is not None:
        entry = found.setdefault(id(value), (allocation, []))
        entry[1].append(path)
        return
    if depth <= 0 or _is_terminal(value):
        return
    value_id = id(value)
    if value_id in visited:
        return
    visited.add(value_id)
    node_count[0] += 1
    if node_count[0] > _MAX_WALK_NODES:
        return
    for child_name, child in _iter_children(value):
        _walk_resource_graph(child, f"{path}.{child_name}", found, visited, node_count, depth - 1)


def _is_terminal(value: object) -> bool:
    return value is None or isinstance(value, (str, bytes, bytearray, bool, int, float, complex, np.ndarray))


def _iter_children(value: object) -> tuple[tuple[str, object], ...]:
    if isinstance(value, dict):
        return tuple((str(key), child) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return tuple((f"[{idx}]", child) for idx, child in enumerate(value))
    if isinstance(value, (set, frozenset)):
        return tuple((f"[{idx}]", child) for idx, child in enumerate(value))
    attrs = _object_attrs(value)
    return tuple((name, child) for name, child in attrs.items() if name not in _SKIP_ATTRS)


def _object_attrs(value: object) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    attrs: dict[str, Any] = {}
    for cls in type(value).__mro__:
        slots = getattr(cls, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot.startswith("__"):
                continue
            try:
                attrs[slot] = getattr(value, slot)
            except AttributeError:
                pass
    return attrs
