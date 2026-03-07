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


def debug_color(index: int, weight: float = 0.05) -> spy.float3:
    t = float(int(index)) * 1.7320508075688772
    return spy.float3(
        smath.lerp(1.0, smath.sin(t), weight),
        smath.lerp(1.0, smath.sin(t + 2.0943951023931953), weight),
        smath.lerp(1.0, smath.sin(t + 4.1887902047863905), weight),
    )


def as_float3(value: tuple[float, float, float] | np.ndarray | spy.float3) -> spy.float3:
    x, y, z = np.asarray(value, dtype=np.float32).reshape(3).tolist()
    return spy.float3(float(x), float(y), float(z))


def normalize3(value: tuple[float, float, float] | np.ndarray | spy.float3, eps: float = VEC_EPS) -> spy.float3:
    vec = as_float3(value)
    return smath.normalize(vec) if float(smath.length(vec)) > float(eps) else spy.float3(0.0, 0.0, 0.0)


def create_default_device(
    device_type: spy.DeviceType = spy.DeviceType.vulkan, enable_debug_layers: bool = False
) -> spy.Device:
    include_paths = (
        SHADER_ROOT,
        SHADER_ROOT / "radix_sort",
        SHADER_ROOT / "renderer",
    )
    return spy.create_device(
        device_type,
        include_paths=include_paths,
        enable_debug_layers=enable_debug_layers,
    )
