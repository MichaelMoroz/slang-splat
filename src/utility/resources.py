from __future__ import annotations

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


def grow_capacity(required: int, current: int) -> int:
    base = max(int(current), 1)
    return max(int(required), base + base // 2)


def alloc_buffer(
    device: spy.Device,
    *,
    size: int,
    usage: spy.BufferUsage,
    min_size: int = 4,
) -> spy.Buffer:
    return device.create_buffer(size=max(int(size), int(min_size)), usage=usage)


def alloc_texture_2d(
    device: spy.Device,
    *,
    format: spy.Format,
    width: int,
    height: int,
    usage: spy.TextureUsage,
) -> spy.Texture:
    return device.create_texture(
        format=format,
        width=max(int(width), 1),
        height=max(int(height), 1),
        usage=usage,
    )


def buffer_to_numpy(buffer: spy.Buffer, dtype: np.dtype) -> np.ndarray:
    return np.frombuffer(buffer.to_numpy().tobytes(), dtype=dtype)


def remap_named_buffers(mapping: dict[str, str], source: dict[str, object]) -> dict[str, object]:
    return {shader_name: source[name] for name, shader_name in mapping.items()}
