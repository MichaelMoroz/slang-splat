from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

import slangpy as spy

from ..scene import ColmapReconstruction, GaussianInitHyperParams, GaussianScene
from ..training import GaussianTrainer
from ..renderer import GaussianRenderer

DEFAULT_LIST_CAPACITY_MULTIPLIER = 16
DEFAULT_MAX_PREPASS_MEMORY_MB = 4096
DEFAULT_VIEWER_BACKGROUND = (0.0, 0.0, 0.0)

@dataclass(slots=True)
class SceneCountProxy:
    count: int


def _default_camera_pos() -> spy.float3:
    return spy.float3(0.0, 0.0, -3.0)


def _default_up() -> spy.float3:
    return spy.float3(0.0, 1.0, 0.0)


def _default_background() -> spy.float3:
    return spy.float3(*DEFAULT_VIEWER_BACKGROUND)


def _default_mouse_delta() -> spy.float2:
    return spy.float2(0.0, 0.0)


def _default_move_vel() -> spy.float3:
    return spy.float3(0.0, 0.0, 0.0)


def _default_rot_vel() -> spy.float2:
    return spy.float2(0.0, 0.0)


@dataclass(slots=True)
class ViewerState:
    list_capacity_multiplier: int = DEFAULT_LIST_CAPACITY_MULTIPLIER; max_prepass_memory_mb: int = DEFAULT_MAX_PREPASS_MEMORY_MB
    renderer: GaussianRenderer | None = None; training_renderer: GaussianRenderer | None = None; debug_renderer: GaussianRenderer | None = None
    scene: GaussianScene | SceneCountProxy | None = None; scene_path: Path | None = None; stats: dict[str, int | bool | float] = field(default_factory=dict)
    colmap_root: Path | None = None; colmap_recon: ColmapReconstruction | None = None; training_frames: list = field(default_factory=list)
    colmap_point_positions_buffer: spy.Buffer | None = None; colmap_point_colors_buffer: spy.Buffer | None = None; colmap_point_count: int = 0
    trainer: GaussianTrainer | None = None; training_active: bool = False; loss_debug_texture: spy.Texture | None = None
    debug_abs_diff_kernel: spy.ComputeKernel | None = None; debug_letterbox_kernel: spy.ComputeKernel | None = None; debug_present_texture: spy.Texture | None = None
    synced_step_main: int = -1; synced_step_debug: int = -1; scene_init_signature: tuple[object, ...] | None = None
    last_training_batch_steps: int = 0
    suggested_init_hparams: GaussianInitHyperParams | None = None; suggested_init_count: int | None = None
    camera_pos: spy.float3 = field(default_factory=_default_camera_pos); yaw: float = 0.0; pitch: float = 0.0
    up: spy.float3 = field(default_factory=_default_up); fov_y: float = 60.0; near: float = 0.1; far: float = 120.0
    move_speed: float = 2.0; look_speed: float = 0.003; background: spy.float3 = field(default_factory=_default_background)
    keys: dict[spy.KeyCode, bool] = field(default_factory=dict); mouse_left: bool = False; mouse_delta: spy.float2 = field(default_factory=_default_mouse_delta)
    scroll_delta: float = 0.0; move_vel: spy.float3 = field(default_factory=_default_move_vel); rot_vel: spy.float2 = field(default_factory=_default_rot_vel)
    mx: float | None = None; my: float | None = None; last_time: float = field(default_factory=time.perf_counter); fps_smooth: float = 60.0
    last_error: str = ""; last_resize_exception: str = ""; last_render_exception: str = ""


LOSS_DEBUG_OPTIONS = (("rendered", "Rendered"), ("target", "Target"), ("abs_diff", "Abs Diff"))
IMAGE_SUBDIR_OPTIONS = ("images_8", "images_4", "images_2", "images")
DEFAULT_IMAGE_SUBDIR_INDEX = 1
