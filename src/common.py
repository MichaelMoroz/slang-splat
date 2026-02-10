from __future__ import annotations

from pathlib import Path

import slangpy as spy
from slangpy import math as smath

ROOT = Path(__file__).resolve().parent.parent
SHADER_ROOT = ROOT / "shaders"


def device_type_from_name(name: str) -> spy.DeviceType:
    normalized = str(name).strip().lower()
    if normalized in {"d3d12", "direct3d12", "dx12"}:
        return spy.DeviceType.d3d12
    if normalized in {"vulkan", "vk"}:
        return spy.DeviceType.vulkan
    raise ValueError(f"Unsupported device type '{name}'. Use 'd3d12' or 'vulkan'.")


def debug_color(index: int, weight: float = 0.05) -> spy.float3:
    t = float(int(index)) * 1.7320508075688772
    return spy.float3(
        smath.lerp(1.0, smath.sin(t), weight),
        smath.lerp(1.0, smath.sin(t + 2.0943951023931953), weight),
        smath.lerp(1.0, smath.sin(t + 4.1887902047863905), weight),
    )


def create_default_device(
    device_type: spy.DeviceType = spy.DeviceType.d3d12, enable_debug_layers: bool = False
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
