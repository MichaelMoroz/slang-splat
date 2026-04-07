from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
import slangpy as spy

from ..metrics import ParamLog10Histograms, ParamTensorRanges
from ..scene import ColmapFrame, ColmapReconstruction, GaussianInitHyperParams, GaussianScene
from ..training import GaussianTrainer
from ..renderer import GaussianRenderer

DEFAULT_LIST_CAPACITY_MULTIPLIER = 16
DEFAULT_MAX_PREPASS_MEMORY_MB = 4096
DEFAULT_VIEWER_BACKGROUND = (0.0, 0.0, 0.0)

@dataclass(slots=True)
class SceneCountProxy:
    count: int


@dataclass(slots=True)
class ColmapImportSettings:
    database_path: Path | None = None
    images_root: Path | None = None
    depth_root: Path | None = None
    depth_value_mode: str = "z_depth"
    init_mode: str = "pointcloud"
    custom_ply_path: Path | None = None
    image_downscale_mode: str = "original"
    image_downscale_max_size: int = 2048
    image_downscale_scale: float = 1.0
    nn_radius_scale_coef: float = 0.5
    depth_point_count: int = 100000
    diffused_point_count: int = 100000
    diffusion_radius: float = 1.0


@dataclass(slots=True)
class ColmapImportProgress:
    dataset_root: Path
    colmap_root: Path
    database_path: Path | None
    images_root: Path
    init_mode: str
    custom_ply_path: Path | None
    image_downscale_mode: str
    image_downscale_max_size: int
    image_downscale_scale: float
    nn_radius_scale_coef: float
    depth_point_count: int = 100000
    diffused_point_count: int = 100000
    diffusion_radius: float = 1.0
    depth_value_mode: str = "z_depth"
    depth_root: Path | None = None
    phase: str = "prepare"
    current: int = 0
    total: int = 1
    current_name: str = ""
    recon: ColmapReconstruction | None = None
    image_items: list[tuple[int, object]] = field(default_factory=list)
    frames: list[ColmapFrame] = field(default_factory=list)
    frame_images: list[object] = field(default_factory=list)
    depth_paths: list[Path] = field(default_factory=list)
    depth_index: dict[str, Path] | None = None
    depth_init_payloads: list[object] = field(default_factory=list)
    native_textures: list[spy.Texture] = field(default_factory=list)
    native_rgba8_loader: object | None = None
    native_rgba8_iter: object | None = None

    @property
    def fraction(self) -> float:
        return 1.0 if self.total <= 0 else min(max(float(self.current) / float(self.total), 0.0), 1.0)


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
    colmap_import: ColmapImportSettings = field(default_factory=ColmapImportSettings)
    colmap_import_progress: ColmapImportProgress | None = None
    colmap_point_positions_buffer: spy.Buffer | None = None; colmap_point_colors_buffer: spy.Buffer | None = None; colmap_point_count: int = 0
    cached_init_point_positions: np.ndarray | None = None; cached_init_point_colors: np.ndarray | None = None
    cached_init_scene: GaussianScene | None = None; cached_init_signature: tuple[object, ...] | None = None
    trainer: GaussianTrainer | None = None; training_active: bool = False; viewport_texture: spy.Texture | None = None; loss_debug_texture: spy.Texture | None = None
    debug_abs_diff_kernel: spy.ComputeKernel | None = None; debug_edge_kernel: spy.ComputeKernel | None = None; debug_letterbox_kernel: spy.ComputeKernel | None = None; debug_present_texture: spy.Texture | None = None
    synced_step_main: int = -1; synced_step_debug: int = -1; scene_init_signature: tuple[object, ...] | None = None
    applied_renderer_params_main: tuple[object, ...] | None = None; applied_renderer_params_training: tuple[object, ...] | None = None; applied_renderer_params_debug: tuple[object, ...] | None = None
    applied_training_signature: tuple[object, ...] | None = None; applied_training_runtime_signature: tuple[object, ...] | None = None; applied_training_runtime_factor: int | None = None
    cached_training_setup_signature: tuple[object, ...] | None = None; cached_training_setup: tuple[object, object, object, object] | None = None
    training_runtime_factor_changed: bool = False; pending_training_runtime_resize: bool = False; pending_training_reinitialize: bool = False
    last_training_batch_steps: int = 0
    training_elapsed_s: float = 0.0; training_resume_time: float | None = None
    suggested_init_hparams: GaussianInitHyperParams | None = None; suggested_init_count: int | None = None
    cached_raster_grad_histograms: ParamLog10Histograms | None = None
    cached_raster_grad_ranges: ParamTensorRanges | None = None
    cached_raster_grad_histogram_mode: str = ""
    cached_raster_grad_histogram_step: int = -1
    cached_raster_grad_histogram_scene_count: int = -1
    cached_raster_grad_histogram_signature: tuple[object, ...] | None = None
    cached_raster_grad_histogram_status: str = ""
    camera_overlay_world_segments: np.ndarray | None = None
    camera_overlay_frame_indices: np.ndarray | None = None
    camera_overlay_signature: tuple[object, ...] | None = None
    camera_pos: spy.float3 = field(default_factory=_default_camera_pos); yaw: float = 0.0; pitch: float = 0.0
    up: spy.float3 = field(default_factory=_default_up); fov_y: float = 60.0; near: float = 0.1; far: float = 120.0
    move_speed: float = 2.0; look_speed: float = 0.003; background: spy.float3 = field(default_factory=_default_background)
    keys: dict[spy.KeyCode, bool] = field(default_factory=dict); mouse_left: bool = False; mouse_delta: spy.float2 = field(default_factory=_default_mouse_delta)
    scroll_delta: float = 0.0; move_vel: spy.float3 = field(default_factory=_default_move_vel); rot_vel: spy.float2 = field(default_factory=_default_rot_vel)
    mx: float | None = None; my: float | None = None; last_time: float = field(default_factory=time.perf_counter); fps_smooth: float = 60.0
    last_error: str = ""; last_resize_exception: str = ""; last_render_exception: str = ""
LOSS_DEBUG_OPTIONS = (
    ("rendered", "Rendered"),
    ("target", "Target"),
    ("abs_diff", "Abs Diff"),
    ("rendered_edges", "Rendered Edges"),
    ("target_edges", "Target Edges"),
)
