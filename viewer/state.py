from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

import slangpy as spy

DEBUG_MODE_NORMAL = 0
DEBUG_MODE_PROCESSED_COUNT = 1
DEBUG_MODE_DEPTH_MEAN = 2
DEBUG_MODE_DEPTH_STD = 3
DEBUG_MODE_ELLIPSE_OUTLINES = 4
DEBUG_MODE_SPLAT_SPATIAL_DENSITY = 5
DEBUG_MODE_SPLAT_SCREEN_DENSITY = 6


def _float2() -> spy.float2:
    return spy.float2(0.0, 0.0)


def _float3() -> spy.float3:
    return spy.float3(0.0, 0.0, 0.0)


@dataclass(slots=True)
class ViewerState:
    scene_path: Path | None = None
    splats: object | None = None
    packed_splats: object | None = None
    splat_count: int = 0
    scene_dirty: bool = False
    camera_pos: spy.float3 = field(default_factory=lambda: spy.float3(0.0, 0.0, -3.0))
    up: spy.float3 = field(default_factory=lambda: spy.float3(0.0, 1.0, 0.0))
    yaw: float = 0.0
    pitch: float = 0.0
    move_speed: float = 2.0
    look_speed: float = 0.003
    fov_y: float = 60.0
    near: float = 0.0
    far: float = 1000.0
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius_scale: float = 1.0
    max_anisotropy: float = 12.0
    alpha_cutoff: float = 0.02
    trans_threshold: float = 0.005
    debug_mode: int = DEBUG_MODE_NORMAL
    keys: dict[spy.KeyCode, bool] = field(default_factory=dict)
    mouse_left: bool = False
    mouse_delta: spy.float2 = field(default_factory=_float2)
    move_vel: spy.float3 = field(default_factory=_float3)
    rot_vel: spy.float2 = field(default_factory=_float2)
    mx: float | None = None
    my: float | None = None
    scroll_delta: float = 0.0
    last_time: float = field(default_factory=time.perf_counter)
    fps_smooth: float = 60.0
    last_error: str = ""
