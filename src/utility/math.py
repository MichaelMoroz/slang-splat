from __future__ import annotations

import numpy as np
import slangpy as spy
from slangpy import math as smath

VEC_EPS = 1e-8


def require_not_none[T](value: T | None, message: str) -> T:
    if value is None:
        raise RuntimeError(message)
    return value


def clamp_float(value: float, lo: float, hi: float) -> float:
    return float(np.clip(float(value), float(lo), float(hi)))


def clamp_int(value: int, lo: int, hi: int) -> int:
    return int(np.clip(int(value), int(lo), int(hi)))


def clamp_index(index: int, size: int) -> int:
    return clamp_int(int(index), 0, max(int(size) - 1, 0))


def as_float3(value: object) -> spy.float3:
    xyz = np.asarray(value, dtype=np.float32).reshape(3)
    return spy.float3(float(xyz[0]), float(xyz[1]), float(xyz[2]))


def normalize3(value: object, eps: float = VEC_EPS) -> spy.float3:
    vec = as_float3(value)
    return smath.normalize(vec) if float(smath.length(vec)) > float(eps) else spy.float3(0.0, 0.0, 0.0)
