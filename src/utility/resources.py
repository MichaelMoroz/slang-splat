from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import weakref
from typing import Any, Callable, TypeVar

import numpy as np
import slangpy as spy

RW_BUFFER_USAGE = (
    spy.BufferUsage.shader_resource
    | spy.BufferUsage.unordered_access
    | spy.BufferUsage.copy_source
    | spy.BufferUsage.copy_destination
)
RO_BUFFER_USAGE = spy.BufferUsage.shader_resource | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
INDIRECT_BUFFER_USAGE = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.indirect_argument
SRV_TEXTURE_USAGE = spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination
UAV_TEXTURE_USAGE = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access


@dataclass(frozen=True, slots=True)
class ResourceAllocation:
    kind: str
    name: str
    byte_size: int
    details: str
    usage: str
    order: int


_RESOURCE_ALLOCATIONS: dict[int, ResourceAllocation] = {}
_RESOURCE_REFS: dict[int, object | None] = {}
_RESOURCE_TYPES: dict[int, type] = {}
_RESOURCE_ALLOCATION_ORDER = 0
_DEFERRED_RELEASE_MIN_AGE = 3
_DEFERRED_RELEASE_DRAIN_BYTES = 256 * 1024 * 1024
_DEFERRED_RELEASE_HARD_LIMIT_BYTES = 4 * 1024 * 1024 * 1024
_DEFERRED_RELEASE_GENERATION = 0
_DEFERRED_RELEASE_BYTES = 0


@dataclass(slots=True)
class _DeferredResourceRelease:
    resource: object
    byte_size: int
    generation: int


_DEFERRED_RELEASES: deque[_DeferredResourceRelease] = deque()
_TResourceSet = TypeVar("_TResourceSet")


def _next_resource_order() -> int:
    global _RESOURCE_ALLOCATION_ORDER
    _RESOURCE_ALLOCATION_ORDER += 1
    return _RESOURCE_ALLOCATION_ORDER


def _usage_text(usage: object) -> str:
    return str(usage).replace("BufferUsage.", "").replace("TextureUsage.", "")


def _register_resource(resource: object, *, kind: str, name: str, byte_size: int, usage: object) -> object:
    resource_id = id(resource)
    _RESOURCE_TYPES[resource_id] = type(resource)
    _RESOURCE_ALLOCATIONS[resource_id] = ResourceAllocation(
        kind=str(kind),
        name=str(name),
        byte_size=_native_device_bytes(resource, max(int(byte_size), 0)),
        details=_resource_details(resource, str(kind), max(int(byte_size), 0)),
        usage=_usage_text(usage),
        order=_next_resource_order(),
    )
    ref = _make_resource_ref(resource)
    if ref is None:
        _RESOURCE_REFS.pop(resource_id, None)
    else:
        _RESOURCE_REFS[resource_id] = ref
    return resource


def register_debug_resource(resource: object, *, kind: str, name: str, byte_size: int, usage: object = "") -> object:
    return _register_resource(resource, kind=kind, name=name, byte_size=byte_size, usage=usage)


def resource_allocation(resource: object) -> ResourceAllocation | None:
    resource_id = id(resource)
    if resource_id not in _RESOURCE_ALLOCATIONS:
        return None
    expected_type = _RESOURCE_TYPES.get(resource_id)
    if expected_type is not None and not isinstance(resource, expected_type):
        return None
    ref = _RESOURCE_REFS.get(resource_id)
    if ref is None:
        return _RESOURCE_ALLOCATIONS.get(resource_id)
    target = ref()
    if target is None:
        _RESOURCE_REFS.pop(resource_id, None)
        _RESOURCE_ALLOCATIONS.pop(resource_id, None)
        _RESOURCE_TYPES.pop(resource_id, None)
        return None
    return _RESOURCE_ALLOCATIONS.get(resource_id) if target is resource else None


def debug_resource_allocations() -> tuple[tuple[object, ResourceAllocation], ...]:
    resources: list[tuple[object, ResourceAllocation]] = []
    for resource_id, ref in tuple(_RESOURCE_REFS.items()):
        if ref is None:
            continue
        target = ref()
        if target is None:
            _RESOURCE_REFS.pop(resource_id, None)
            _RESOURCE_ALLOCATIONS.pop(resource_id, None)
            _RESOURCE_TYPES.pop(resource_id, None)
            continue
        allocation = _RESOURCE_ALLOCATIONS.get(resource_id)
        if allocation is not None:
            resources.append((target, allocation))
    return tuple(resources)


def defer_resource_release(resource: object | None) -> None:
    if resource is None:
        return
    global _DEFERRED_RELEASE_BYTES
    allocation = resource_allocation(resource)
    byte_size = 0 if allocation is None else max(int(allocation.byte_size), 0)
    if byte_size <= 0:
        byte_size = _native_device_bytes(resource, 0)
    _DEFERRED_RELEASES.append(_DeferredResourceRelease(resource=resource, byte_size=max(byte_size, 0), generation=_DEFERRED_RELEASE_GENERATION))
    _DEFERRED_RELEASE_BYTES += max(byte_size, 0)
    if _DEFERRED_RELEASE_BYTES > _DEFERRED_RELEASE_HARD_LIMIT_BYTES:
        drain_deferred_resource_releases(max_bytes=_DEFERRED_RELEASE_BYTES - _DEFERRED_RELEASE_HARD_LIMIT_BYTES, min_age=0, advance_generation=False)


def defer_resource_releases(resources: Any) -> None:
    seen: set[int] = set()
    for resource in tuple(resources):
        resource_id = id(resource)
        if resource_id in seen:
            continue
        seen.add(resource_id)
        defer_resource_release(resource)


def drain_deferred_resource_releases(
    *,
    max_bytes: int = _DEFERRED_RELEASE_DRAIN_BYTES,
    min_age: int = _DEFERRED_RELEASE_MIN_AGE,
    advance_generation: bool = True,
) -> tuple[int, int]:
    global _DEFERRED_RELEASE_BYTES, _DEFERRED_RELEASE_GENERATION
    if advance_generation:
        _DEFERRED_RELEASE_GENERATION += 1
    released_count = 0
    released_bytes = 0
    byte_budget = max(int(max_bytes), 0)
    while _DEFERRED_RELEASES and (released_bytes < byte_budget or _DEFERRED_RELEASE_BYTES > _DEFERRED_RELEASE_HARD_LIMIT_BYTES):
        item = _DEFERRED_RELEASES[0]
        if _DEFERRED_RELEASE_GENERATION - item.generation < int(min_age) and _DEFERRED_RELEASE_BYTES <= _DEFERRED_RELEASE_HARD_LIMIT_BYTES:
            break
        item = _DEFERRED_RELEASES.popleft()
        released_count += 1
        released_bytes += item.byte_size
        _DEFERRED_RELEASE_BYTES = max(_DEFERRED_RELEASE_BYTES - item.byte_size, 0)
        del item
    return released_count, released_bytes


def clear_debug_resource_allocations() -> None:
    global _RESOURCE_ALLOCATION_ORDER, _DEFERRED_RELEASE_BYTES, _DEFERRED_RELEASE_GENERATION
    _RESOURCE_ALLOCATIONS.clear()
    _RESOURCE_REFS.clear()
    _RESOURCE_TYPES.clear()
    _DEFERRED_RELEASES.clear()
    _DEFERRED_RELEASE_BYTES = 0
    _DEFERRED_RELEASE_GENERATION = 0
    _RESOURCE_ALLOCATION_ORDER = 0


def _make_resource_ref(resource: object) -> object | None:
    resource_id = id(resource)
    try:
        return weakref.ref(resource, lambda _ref: (_RESOURCE_REFS.pop(resource_id, None), _RESOURCE_ALLOCATIONS.pop(resource_id, None), _RESOURCE_TYPES.pop(resource_id, None)))
    except TypeError:
        return None


def _texture_bytes_per_pixel(format: spy.Format) -> int:
    text = str(format).lower()
    if "rgba32_float" in text:
        return 16
    if "rgba8" in text:
        return 4
    return 0


def _native_device_bytes(resource: object, fallback: int) -> int:
    try:
        device_bytes = int(resource.memory_usage.device)
    except Exception:
        return int(fallback)
    return device_bytes if device_bytes > 0 else int(fallback)


def _format_format(format: object) -> str:
    return str(format).replace("Format.", "")


def _buffer_details(resource: object, byte_size: int) -> str:
    try:
        struct_size = int(resource.struct_size)
    except Exception:
        struct_size = 0
    if struct_size > 0:
        element_count = max(int(byte_size), 0) // struct_size
        return f"{element_count:,} elements x {struct_size} B"
    try:
        reported_size = int(resource.size)
    except Exception:
        reported_size = int(byte_size)
    size = max(reported_size, 0)
    if size > 0 and size % 4 == 0:
        return f"{size // 4:,} x 4 B units"
    return f"{size:,} bytes"


def _texture_details(resource: object) -> str:
    try:
        width, height, depth = int(resource.width), int(resource.height), int(resource.depth)
        array_length = int(resource.array_length)
        mip_count = int(resource.mip_count)
        format_text = _format_format(resource.format)
    except Exception:
        return ""
    depth_text = f"x{depth}" if depth > 1 else ""
    array_text = f" array={array_length}" if array_length > 1 else ""
    mip_text = f" mips={mip_count}" if mip_count > 1 else ""
    return f"{width}x{height}{depth_text} {format_text}{array_text}{mip_text}"


def _resource_details(resource: object, kind: str, byte_size: int) -> str:
    if kind == "Buffer":
        return _buffer_details(resource, byte_size)
    if kind == "Texture":
        return _texture_details(resource)
    return ""


def grow_capacity(required: int, current: int) -> int:
    base = max(int(current), 1)
    return max(int(required), base + base // 2)


def ensure_capacity_resources(
    required: int,
    current: int,
    resources: _TResourceSet | None,
    *,
    create: Callable[[int], _TResourceSet],
    release: Callable[[_TResourceSet], None] | None = None,
) -> tuple[int, _TResourceSet]:
    needed = int(required)
    if resources is not None and needed <= int(current):
        return int(current), resources
    capacity = grow_capacity(needed, current)
    if release is not None and resources is not None:
        release(resources)
    return capacity, create(capacity)


def alloc_buffer(
    device: spy.Device,
    *,
    name: str,
    size: int,
    usage: spy.BufferUsage,
    min_size: int = 4,
) -> spy.Buffer:
    byte_size = max(int(size), int(min_size))
    return _register_resource(device.create_buffer(size=byte_size, usage=usage, label=str(name)), kind="Buffer", name=name, byte_size=byte_size, usage=usage)


def alloc_texture_2d(
    device: spy.Device,
    *,
    name: str,
    format: spy.Format,
    width: int,
    height: int,
    usage: spy.TextureUsage,
) -> spy.Texture:
    texture_width = max(int(width), 1)
    texture_height = max(int(height), 1)
    byte_size = texture_width * texture_height * _texture_bytes_per_pixel(format)
    return _register_resource(device.create_texture(
        format=format,
        width=texture_width,
        height=texture_height,
        usage=usage,
        label=str(name),
    ), kind="Texture", name=name, byte_size=byte_size, usage=usage)


def buffer_to_numpy(buffer: spy.Buffer, dtype: np.dtype) -> np.ndarray:
    return np.frombuffer(buffer.to_numpy().tobytes(), dtype=dtype)


def remap_named_buffers(mapping: dict[str, str], source: dict[str, object]) -> dict[str, object]:
    return {shader_name: source[name] for name, shader_name in mapping.items()}
