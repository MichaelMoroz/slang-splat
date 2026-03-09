from __future__ import annotations

from pathlib import Path

import numpy as np
import slangpy as spy
from slangpy import math as smath

ROOT = Path(__file__).resolve().parent.parent
SHADER_ROOT = ROOT / "shaders"
VEC_EPS = 1e-8


def device_type_from_name(name: str) -> spy.DeviceType:
    normalized = str(name).strip().lower()
    if normalized in {"vulkan", "vk"}:
        return spy.DeviceType.vulkan
    raise ValueError(f"Unsupported device type '{name}'. Use 'vulkan'.")


def clamp_float(value: float, lo: float, hi: float) -> float:
    return float(np.clip(float(value), float(lo), float(hi)))


def clamp_int(value: int, lo: int, hi: int) -> int:
    return int(np.clip(int(value), int(lo), int(hi)))


def clamp_index(index: int, size: int) -> int:
    return clamp_int(int(index), 0, max(int(size) - 1, 0))


def thread_count_1d(count: int) -> spy.uint3:
    return spy.uint3(int(count), 1, 1)


def thread_count_2d(width: int, height: int, depth: int = 1) -> spy.uint3:
    return spy.uint3(int(width), int(height), int(depth))


def require_not_none[T](value: T | None, message: str) -> T:
    if value is None:
        raise RuntimeError(message)
    return value


def buffer_to_numpy(buffer: spy.Buffer, dtype: np.dtype) -> np.ndarray:
    return np.frombuffer(buffer.to_numpy().tobytes(), dtype=dtype)


def remap_named_buffers(mapping: dict[str, str], source: dict[str, object]) -> dict[str, object]:
    return {shader_name: source[name] for name, shader_name in mapping.items()}


def debug_color(index: int, weight: float = 0.05) -> spy.float3:
    phase = float(int(index)) * 1.7320508075688772
    return spy.float3(
        smath.lerp(1.0, smath.sin(phase), weight),
        smath.lerp(1.0, smath.sin(phase + 2.0943951023931953), weight),
        smath.lerp(1.0, smath.sin(phase + 4.1887902047863905), weight),
    )


def as_float3(value: object) -> spy.float3:
    xyz = np.asarray(value, dtype=np.float32).reshape(3)
    return spy.float3(float(xyz[0]), float(xyz[1]), float(xyz[2]))


def normalize3(value: object, eps: float = VEC_EPS) -> spy.float3:
    vec = as_float3(value)
    return smath.normalize(vec) if float(smath.length(vec)) > float(eps) else spy.float3(0.0, 0.0, 0.0)


def create_default_device(device_type: spy.DeviceType = spy.DeviceType.vulkan, enable_debug_layers: bool = False) -> spy.Device:
    return spy.create_device(
        device_type,
        include_paths=(SHADER_ROOT, SHADER_ROOT / "renderer", SHADER_ROOT / "utility"),
        enable_debug_layers=enable_debug_layers,
    )
