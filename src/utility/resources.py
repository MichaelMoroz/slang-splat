from __future__ import annotations

from dataclasses import dataclass
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
    usage: str
    order: int


_RESOURCE_ALLOCATIONS: dict[int, ResourceAllocation] = {}
_RESOURCE_ALLOCATION_ORDER = 0


def _next_resource_order() -> int:
    global _RESOURCE_ALLOCATION_ORDER
    _RESOURCE_ALLOCATION_ORDER += 1
    return _RESOURCE_ALLOCATION_ORDER


def _usage_text(usage: object) -> str:
    return str(usage).replace("BufferUsage.", "").replace("TextureUsage.", "")


def _register_resource(resource: object, *, kind: str, name: str, byte_size: int, usage: object) -> object:
    _RESOURCE_ALLOCATIONS[id(resource)] = ResourceAllocation(
        kind=str(kind),
        name=str(name),
        byte_size=_native_device_bytes(resource, max(int(byte_size), 0)),
        usage=_usage_text(usage),
        order=_next_resource_order(),
    )
    return resource


def register_debug_resource(resource: object, *, kind: str, name: str, byte_size: int, usage: object = "") -> object:
    return _register_resource(resource, kind=kind, name=name, byte_size=byte_size, usage=usage)


def resource_allocation(resource: object) -> ResourceAllocation | None:
    return _RESOURCE_ALLOCATIONS.get(id(resource))


def clear_debug_resource_allocations() -> None:
    global _RESOURCE_ALLOCATION_ORDER
    _RESOURCE_ALLOCATIONS.clear()
    _RESOURCE_ALLOCATION_ORDER = 0


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
