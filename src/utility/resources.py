from __future__ import annotations

from dataclasses import dataclass
import weakref
from typing import Any

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
_RESOURCE_REFS: dict[int, object] = {}
_RESOURCE_ALLOCATION_ORDER = 0


def _next_resource_order() -> int:
    global _RESOURCE_ALLOCATION_ORDER
    _RESOURCE_ALLOCATION_ORDER += 1
    return _RESOURCE_ALLOCATION_ORDER


def _usage_text(usage: object) -> str:
    return str(usage).replace("BufferUsage.", "").replace("TextureUsage.", "")


def _register_resource(resource: object, *, kind: str, name: str, byte_size: int, usage: object) -> object:
    resource_id = id(resource)
    _RESOURCE_ALLOCATIONS[resource_id] = ResourceAllocation(
        kind=str(kind),
        name=str(name),
        byte_size=_native_device_bytes(resource, max(int(byte_size), 0)),
        details=_resource_details(resource, str(kind), max(int(byte_size), 0)),
        usage=_usage_text(usage),
        order=_next_resource_order(),
    )
    _RESOURCE_REFS[resource_id] = _make_resource_ref(resource)
    return resource


def register_debug_resource(resource: object, *, kind: str, name: str, byte_size: int, usage: object = "") -> object:
    return _register_resource(resource, kind=kind, name=name, byte_size=byte_size, usage=usage)


def resource_allocation(resource: object) -> ResourceAllocation | None:
    resource_id = id(resource)
    ref = _RESOURCE_REFS.get(resource_id)
    if ref is None:
        return None
    target = ref()
    if target is None:
        _RESOURCE_REFS.pop(resource_id, None)
        _RESOURCE_ALLOCATIONS.pop(resource_id, None)
        return None
    return _RESOURCE_ALLOCATIONS.get(resource_id) if target is resource else None


def clear_debug_resource_allocations() -> None:
    global _RESOURCE_ALLOCATION_ORDER
    _RESOURCE_ALLOCATIONS.clear()
    _RESOURCE_REFS.clear()
    _RESOURCE_ALLOCATION_ORDER = 0


class _StrongResourceRef:
    __slots__ = ("_resource",)

    def __init__(self, resource: object) -> None:
        self._resource = resource

    def __call__(self) -> object:
        return self._resource


def _make_resource_ref(resource: object) -> object:
    resource_id = id(resource)
    try:
        return weakref.ref(resource, lambda _ref: (_RESOURCE_REFS.pop(resource_id, None), _RESOURCE_ALLOCATIONS.pop(resource_id, None)))
    except TypeError:
        return _StrongResourceRef(resource)


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
