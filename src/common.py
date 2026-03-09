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
debug_color = lambda index, weight=0.05: (lambda t: spy.float3(smath.lerp(1.0, smath.sin(t), weight), smath.lerp(1.0, smath.sin(t + 2.0943951023931953), weight), smath.lerp(1.0, smath.sin(t + 4.1887902047863905), weight)))(float(int(index)) * 1.7320508075688772)
as_float3 = lambda value: (lambda xyz: spy.float3(float(xyz[0]), float(xyz[1]), float(xyz[2])))(np.asarray(value, dtype=np.float32).reshape(3))
normalize3 = lambda value, eps=VEC_EPS: (lambda vec: smath.normalize(vec) if float(smath.length(vec)) > float(eps) else spy.float3(0.0, 0.0, 0.0))(as_float3(value))
create_default_device = lambda device_type=spy.DeviceType.vulkan, enable_debug_layers=False: spy.create_device(device_type, include_paths=(SHADER_ROOT, SHADER_ROOT / "renderer", SHADER_ROOT / "utility"), enable_debug_layers=enable_debug_layers)
