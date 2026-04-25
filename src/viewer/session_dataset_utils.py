from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
import slangpy as spy

from ..scene._internal.colmap_types import ColmapFrame
from ..utility import alloc_texture_2d, register_debug_resource

_DDS_MAGIC = 0x20534444
_DDS_HEADER_SIZE = 124
_DDS_FOURCC_FLAG = 0x4
_DDS_FOURCC_DXT1 = 0x31545844
_DDS_FOURCC_DX10 = 0x30315844
_DXGI_BC1_UNORM = 71
_DXGI_BC1_UNORM_SRGB = 72
_DXGI_BC7_UNORM = 98
_DXGI_BC7_UNORM_SRGB = 99


@dataclass(frozen=True, slots=True)
class _CompressedDatasetTexture:
    width: int
    height: int
    format: spy.Format
    payload: np.ndarray


def _dataset_cache_root(images_root: Path) -> Path:
    return Path(images_root).resolve() / "cache"


def _dataset_bc7_cache_dir(images_root: Path, width: int, height: int) -> Path:
    return _dataset_cache_root(images_root) / "bc7" / f"{max(int(width), 1)}x{max(int(height), 1)}"


def _dataset_bc7_cache_path(images_root: Path, frame: ColmapFrame) -> Path:
    relative = Path(frame.image_path).resolve().relative_to(Path(images_root).resolve())
    return (_dataset_bc7_cache_dir(images_root, frame.width, frame.height) / relative).with_suffix(".dds")


def _bc_payload_byte_count(width: int, height: int, format: spy.Format) -> int:
    block_bytes = 8 if format in (spy.Format.bc1_unorm, spy.Format.bc1_unorm_srgb) else 16 if format in (spy.Format.bc7_unorm, spy.Format.bc7_unorm_srgb) else 0
    if block_bytes <= 0:
        raise ValueError(f"Unsupported BC format: {format}")
    blocks_x = (max(int(width), 1) + 3) // 4
    blocks_y = (max(int(height), 1) + 3) // 4
    return int(blocks_x * blocks_y * block_bytes)


def _parse_compressed_dataset_texture(dds_path: Path) -> _CompressedDatasetTexture:
    blob = dds_path.read_bytes()
    if len(blob) < 128:
        raise RuntimeError(f"DDS file is too small: {dds_path}")
    magic, header_size = struct.unpack_from("<II", blob, 0)
    if magic != _DDS_MAGIC or header_size != _DDS_HEADER_SIZE:
        raise RuntimeError(f"Invalid DDS header in {dds_path}")
    height, width = struct.unpack_from("<II", blob, 12)
    pf_flags, fourcc = struct.unpack_from("<II", blob, 80)
    if (pf_flags & _DDS_FOURCC_FLAG) == 0:
        raise RuntimeError(f"DDS cache is not block-compressed: {dds_path}")
    payload_offset = 128
    if fourcc == _DDS_FOURCC_DXT1:
        texture_format = spy.Format.bc1_unorm
    elif fourcc == _DDS_FOURCC_DX10:
        if len(blob) < 148:
            raise RuntimeError(f"DDS DX10 header is truncated: {dds_path}")
        dxgi_format = struct.unpack_from("<I", blob, 128)[0]
        payload_offset = 148
        if dxgi_format == _DXGI_BC1_UNORM:
            texture_format = spy.Format.bc1_unorm
        elif dxgi_format == _DXGI_BC1_UNORM_SRGB:
            texture_format = spy.Format.bc1_unorm_srgb
        elif dxgi_format == _DXGI_BC7_UNORM:
            texture_format = spy.Format.bc7_unorm
        elif dxgi_format == _DXGI_BC7_UNORM_SRGB:
            texture_format = spy.Format.bc7_unorm_srgb
        else:
            raise RuntimeError(f"Unsupported DDS DXGI format {dxgi_format} in {dds_path}")
    else:
        raise RuntimeError(f"Unsupported DDS FOURCC 0x{fourcc:08x} in {dds_path}")
    payload = np.frombuffer(blob[payload_offset:], dtype=np.uint8).copy()
    expected_bytes = _bc_payload_byte_count(int(width), int(height), texture_format)
    if int(payload.size) != expected_bytes:
        raise RuntimeError(f"DDS payload size mismatch in {dds_path}: expected {expected_bytes} bytes, got {int(payload.size)}")
    return _CompressedDatasetTexture(width=int(width), height=int(height), format=texture_format, payload=payload)


def _create_native_dataset_texture_from_bc_payload(viewer: object, payload: _CompressedDatasetTexture) -> spy.Texture:
    texture = viewer.device.create_texture(
        format=payload.format,
        width=int(payload.width),
        height=int(payload.height),
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
        label="viewer.dataset_texture_bc7",
    )
    texture.copy_from_numpy(np.ascontiguousarray(payload.payload, dtype=np.uint8))
    return register_debug_resource(
        texture,
        kind="Texture",
        name="viewer.dataset_texture_bc7",
        byte_size=_bc_payload_byte_count(payload.width, payload.height, payload.format),
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
    )


def _create_native_dataset_texture_from_rgba8(viewer: object, rgba8: np.ndarray) -> spy.Texture:
    texture = alloc_texture_2d(
        viewer.device,
        name="viewer.dataset_texture",
        format=spy.Format.rgba8_unorm_srgb,
        width=int(rgba8.shape[1]),
        height=int(rgba8.shape[0]),
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
    )
    texture.copy_from_numpy(np.ascontiguousarray(rgba8, dtype=np.uint8))
    return texture