from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
import slangpy as spy

from ..repo_defaults import viewer_defaults
from ..metrics import ParamLog10Histograms, ParamTensorRanges
from ..scene import ColmapFrame, ColmapReconstruction, GaussianScene
from ..training.alpha_modes import TARGET_ALPHA_MODE_OFF, resolve_target_alpha_mode, target_alpha_skip_mask_enabled
from ..training.defaults import TRAINING_BUILD_ARG_DEFAULTS
from ..training import GaussianTrainer, PhotometricCompensationTrainer
from ..renderer import GaussianRenderer

_VIEWER_DEFAULTS = viewer_defaults()
_VIEWER_STATE_DEFAULTS = _VIEWER_DEFAULTS["state"]
_VIEWER_IMPORT_DEFAULTS = _VIEWER_DEFAULTS["import"]

DEFAULT_LIST_CAPACITY_MULTIPLIER = int(_VIEWER_STATE_DEFAULTS["list_capacity_multiplier"])
DEFAULT_MAX_PREPASS_MEMORY_MB = int(_VIEWER_STATE_DEFAULTS["max_prepass_memory_mb"])
DEFAULT_VIEWER_BACKGROUND = tuple(float(v) for v in _VIEWER_STATE_DEFAULTS["background"])
DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH = int(_VIEWER_IMPORT_DEFAULTS["colmap_min_track_length"])
DEFAULT_COLMAP_INIT_NEIGHBOR_COUNT = int(_VIEWER_IMPORT_DEFAULTS.get("colmap_init_neighbor_count", 8))
DEFAULT_COLMAP_INIT_ANISOTROPY_STRENGTH = float(_VIEWER_IMPORT_DEFAULTS.get("colmap_init_anisotropy_strength", 1.0))
DEFAULT_TARGET_ALPHA_THRESHOLD = float(TRAINING_BUILD_ARG_DEFAULTS["target_alpha_threshold"])
COLMAP_ROTATION_MODE_NONE = 0
COLMAP_ROTATION_MODE_CUSTOM = 1
COLMAP_ROTATION_MODE_AUTO = 2
DEFAULT_COLMAP_ROTATION_MODE = int(
    _VIEWER_IMPORT_DEFAULTS.get(
        "colmap_rotation_mode",
        COLMAP_ROTATION_MODE_AUTO if bool(_VIEWER_IMPORT_DEFAULTS.get("colmap_auto_rotate_scene", True)) else COLMAP_ROTATION_MODE_NONE,
    )
)
DEFAULT_COLMAP_CUSTOM_ROTATION_DEG = tuple(float(v) for v in _VIEWER_IMPORT_DEFAULTS.get("colmap_custom_rotation_deg", (0.0, 0.0, 0.0)))

@dataclass(slots=True)
class SceneCountProxy:
    count: int


@dataclass(slots=True)
class ColmapImportSettings:
    database_path: Path | None = None
    images_root: Path | None = None
    depth_root: Path | None = None
    selected_camera_ids: tuple[int, ...] = ()
    depth_value_mode: str = "z_depth"
    init_mode: str = "pointcloud"
    rotation_mode: int = DEFAULT_COLMAP_ROTATION_MODE
    custom_rotation_deg: tuple[float, float, float] = DEFAULT_COLMAP_CUSTOM_ROTATION_DEG
    compress_dataset_using_bc7: bool = bool(_VIEWER_IMPORT_DEFAULTS.get("compress_dataset_using_bc7", False))
    training_image_color_init: bool = bool(_VIEWER_IMPORT_DEFAULTS.get("colmap_training_image_color_init", False))
    photometric_compensation_enabled: bool = bool(_VIEWER_IMPORT_DEFAULTS.get("colmap_photometric_compensation_enabled", False))
    custom_ply_path: Path | None = None
    image_downscale_mode: str = "original"
    image_downscale_max_size: int = int(_VIEWER_IMPORT_DEFAULTS["colmap_image_max_size"])
    image_downscale_scale: float = float(_VIEWER_IMPORT_DEFAULTS["colmap_image_scale"])
    nn_radius_scale_coef: float = 0.5
    min_track_length: int = DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH
    init_neighbor_count: int = DEFAULT_COLMAP_INIT_NEIGHBOR_COUNT
    init_anisotropy_strength: float = DEFAULT_COLMAP_INIT_ANISOTROPY_STRENGTH
    depth_point_count: int = int(_VIEWER_IMPORT_DEFAULTS["colmap_depth_point_count"])
    diffused_point_count: int = int(_VIEWER_IMPORT_DEFAULTS["colmap_diffused_point_count"])
    fibonacci_sphere_point_count: int = 0
    fibonacci_sphere_radius_multiplier: float = float(_VIEWER_IMPORT_DEFAULTS.get("colmap_fibonacci_sphere_radius_multiplier", _VIEWER_IMPORT_DEFAULTS.get("colmap_fibonacci_sphere_radius", 2.0)))
    fibonacci_sphere_color: tuple[float, float, float] = tuple(float(v) for v in _VIEWER_IMPORT_DEFAULTS.get("colmap_fibonacci_sphere_color", (0.8, 0.8, 0.8)))
    fibonacci_sphere_upper_hemisphere_only: bool = bool(_VIEWER_IMPORT_DEFAULTS.get("colmap_fibonacci_sphere_upper_hemisphere_only", False))
    target_alpha_mode: int | None = None
    target_alpha_threshold: float = DEFAULT_TARGET_ALPHA_THRESHOLD
    use_target_alpha_mask: bool = False
    pointcloud_enabled: bool = bool(_VIEWER_IMPORT_DEFAULTS.get("colmap_pointcloud_enabled", False))
    pointcloud_nn_radius_scale_coef: float = float(_VIEWER_IMPORT_DEFAULTS.get("colmap_pointcloud_nn_radius_scale_coef", _VIEWER_IMPORT_DEFAULTS.get("colmap_nn_radius_scale_coef", 0.5)))
    diffused_enabled: bool = bool(_VIEWER_IMPORT_DEFAULTS.get("colmap_diffused_enabled", False))
    diffused_diffusion_radius: float = float(_VIEWER_IMPORT_DEFAULTS.get("colmap_diffused_diffusion_radius", 1.0))
    diffused_nn_radius_scale_coef: float = float(_VIEWER_IMPORT_DEFAULTS.get("colmap_diffused_nn_radius_scale_coef", _VIEWER_IMPORT_DEFAULTS.get("colmap_nn_radius_scale_coef", 0.5)))
    custom_ply_enabled: bool = bool(_VIEWER_IMPORT_DEFAULTS.get("colmap_custom_ply_enabled", False))
    custom_ply_nn_radius_scale_coef: float = float(_VIEWER_IMPORT_DEFAULTS.get("colmap_custom_ply_nn_radius_scale_coef", 1.0))
    custom_mesh_enabled: bool = bool(_VIEWER_IMPORT_DEFAULTS.get("colmap_custom_mesh_enabled", False))
    custom_mesh_path: Path | None = None
    custom_mesh_point_count: int = int(_VIEWER_IMPORT_DEFAULTS.get("colmap_custom_mesh_point_count", _VIEWER_IMPORT_DEFAULTS.get("colmap_diffused_point_count", 500000)))
    custom_mesh_nn_radius_scale_coef: float = float(_VIEWER_IMPORT_DEFAULTS.get("colmap_custom_mesh_nn_radius_scale_coef", _VIEWER_IMPORT_DEFAULTS.get("colmap_nn_radius_scale_coef", 0.5)))
    fibonacci_sphere_enabled: bool = bool(_VIEWER_IMPORT_DEFAULTS.get("colmap_fibonacci_sphere_enabled", int(_VIEWER_IMPORT_DEFAULTS.get("colmap_fibonacci_sphere_point_count", 0)) > 0))
    fibonacci_sphere_nn_radius_scale_coef: float = float(_VIEWER_IMPORT_DEFAULTS.get("colmap_fibonacci_sphere_nn_radius_scale_coef", 1.0))

    def __post_init__(self) -> None:
        self.target_alpha_mode = resolve_target_alpha_mode(self.target_alpha_mode, legacy_use_target_alpha_mask=self.use_target_alpha_mask)
        self.target_alpha_threshold = float(np.clip(self.target_alpha_threshold, 0.0, 1.0))
        self.use_target_alpha_mask = target_alpha_skip_mask_enabled(self.target_alpha_mode)


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
    rotation_mode: int = DEFAULT_COLMAP_ROTATION_MODE
    custom_rotation_deg: tuple[float, float, float] = DEFAULT_COLMAP_CUSTOM_ROTATION_DEG
    compress_dataset_using_bc7: bool = False
    training_image_color_init: bool = False
    photometric_compensation_enabled: bool = bool(_VIEWER_IMPORT_DEFAULTS.get("colmap_photometric_compensation_enabled", False))
    selected_camera_ids: tuple[int, ...] = ()
    min_track_length: int = DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH
    init_neighbor_count: int = DEFAULT_COLMAP_INIT_NEIGHBOR_COUNT
    init_anisotropy_strength: float = DEFAULT_COLMAP_INIT_ANISOTROPY_STRENGTH
    depth_point_count: int = 100000
    diffused_point_count: int = 100000
    fibonacci_sphere_point_count: int = 0
    fibonacci_sphere_radius_multiplier: float = float(_VIEWER_IMPORT_DEFAULTS.get("colmap_fibonacci_sphere_radius_multiplier", _VIEWER_IMPORT_DEFAULTS.get("colmap_fibonacci_sphere_radius", 2.0)))
    fibonacci_sphere_color: tuple[float, float, float] = tuple(float(v) for v in _VIEWER_IMPORT_DEFAULTS.get("colmap_fibonacci_sphere_color", (0.8, 0.8, 0.8)))
    fibonacci_sphere_upper_hemisphere_only: bool = bool(_VIEWER_IMPORT_DEFAULTS.get("colmap_fibonacci_sphere_upper_hemisphere_only", False))
    target_alpha_mode: int | None = None
    target_alpha_threshold: float = DEFAULT_TARGET_ALPHA_THRESHOLD
    use_target_alpha_mask: bool = False
    depth_value_mode: str = "z_depth"
    depth_root: Path | None = None
    pointcloud_enabled: bool = False
    pointcloud_nn_radius_scale_coef: float = 0.5
    diffused_enabled: bool = False
    diffused_diffusion_radius: float = 1.0
    diffused_nn_radius_scale_coef: float = 0.5
    custom_ply_enabled: bool = False
    custom_ply_nn_radius_scale_coef: float = 1.0
    custom_mesh_enabled: bool = False
    custom_mesh_path: Path | None = None
    custom_mesh_point_count: int = 500000
    custom_mesh_nn_radius_scale_coef: float = 0.5
    fibonacci_sphere_enabled: bool = False
    fibonacci_sphere_nn_radius_scale_coef: float = 1.0
    phase: str = "prepare"
    current: int = 0
    total: int = 1
    current_name: str = ""
    recon: ColmapReconstruction | None = None
    image_items: list[tuple[int, object]] = field(default_factory=list)
    frames: list[ColmapFrame] = field(default_factory=list)
    frame_images: list[object] = field(default_factory=list)
    depth_paths: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.target_alpha_mode = resolve_target_alpha_mode(self.target_alpha_mode, legacy_use_target_alpha_mask=self.use_target_alpha_mask)
        self.target_alpha_threshold = float(np.clip(self.target_alpha_threshold, 0.0, 1.0))
        self.use_target_alpha_mask = target_alpha_skip_mask_enabled(self.target_alpha_mode)
    depth_index: dict[str, Path] | None = None
    depth_init_payloads: list[object] = field(default_factory=list)
    native_textures: list[spy.Texture] = field(default_factory=list)
    native_rgba8_loader: object | None = None
    native_rgba8_iter: object | None = None
    photometric_trainer: PhotometricCompensationTrainer | None = None

    @property
    def fraction(self) -> float:
        return 1.0 if self.total <= 0 else min(max(float(self.current) / float(self.total), 0.0), 1.0)


@dataclass(frozen=True, slots=True)
class DatasetMetricsReport:
    generated_at: str
    report_path: Path | None
    dataset_root: Path | None
    requested_frame_count: int
    splat_count: int
    total_elapsed_s: float
    rows: tuple[object, ...] = ()


@dataclass(slots=True)
class DatasetMetricsTask:
    trainer_id: int
    requested_frame_count: int
    splat_count: int
    dataset_root: Path | None = None
    previous_force_prepass_count_readback: bool = False
    started_at: float = field(default_factory=time.perf_counter)
    next_frame_index: int = 0
    rows: list[object] = field(default_factory=list)
    previous_renderer_size: tuple[int, int] | None = None
    previous_renderer_capacity: tuple[int, int] | None = None

    @property
    def fraction(self) -> float:
        total = max(int(self.requested_frame_count), 0)
        current = min(max(int(self.next_frame_index), 0), total)
        return 1.0 if total <= 0 else min(max(float(current) / float(total), 0.0), 1.0)


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
    cached_init_point_positions: np.ndarray | None = None; cached_init_point_colors: np.ndarray | None = None
    cached_init_signature: tuple[object, ...] | None = None
    cached_init_pointcloud_positions: np.ndarray | None = None; cached_init_pointcloud_colors: np.ndarray | None = None
    cached_init_diffused_positions: np.ndarray | None = None; cached_init_diffused_colors: np.ndarray | None = None
    cached_init_custom_ply_scene: GaussianScene | None = None
    cached_init_custom_mesh_positions: np.ndarray | None = None; cached_init_custom_mesh_colors: np.ndarray | None = None
    cached_init_fibonacci_positions: np.ndarray | None = None; cached_init_fibonacci_colors: np.ndarray | None = None
    trainer: GaussianTrainer | None = None; photometric_trainer: PhotometricCompensationTrainer | None = None
    dataset_metrics_task: DatasetMetricsTask | None = None; dataset_metrics_report: DatasetMetricsReport | None = None; dataset_metrics_status: str = ""
    training_active: bool = False; photometric_active: bool = False; photometric_prepare_pending_active: bool = False
    viewport_texture: spy.Texture | None = None; loss_debug_texture: spy.Texture | None = None; debug_target_texture: spy.Texture | None = None
    debug_abs_diff_kernel: spy.ComputeKernel | None = None; debug_edge_kernel: spy.ComputeKernel | None = None; debug_dssim_features_kernel: spy.ComputeKernel | None = None; debug_dssim_compose_kernel: spy.ComputeKernel | None = None; debug_letterbox_kernel: spy.ComputeKernel | None = None; debug_target_sample_kernel: spy.ComputeKernel | None = None; debug_present_texture: spy.Texture | None = None
    debug_dssim_blur: object | None = None; debug_dssim_resolution: tuple[int, int] | None = None
    debug_dssim_moments: spy.Buffer | None = None; debug_dssim_blurred_moments: spy.Buffer | None = None
    applied_renderer_params_main: tuple[object, ...] | None = None; applied_renderer_params_training: tuple[object, ...] | None = None; applied_renderer_params_debug: tuple[object, ...] | None = None
    applied_training_signature: tuple[object, ...] | None = None; applied_training_runtime_signature: tuple[object, ...] | None = None; applied_training_runtime_factor: int | None = None
    cached_training_setup_signature: tuple[object, ...] | None = None; cached_training_setup: tuple[object, object, object, object] | None = None
    training_runtime_factor_changed: bool = False; pending_training_runtime_resize: bool = False; pending_training_reinitialize: bool = False
    pending_python_frame_capture: bool = False; pending_renderdoc_frame_capture: bool = False
    last_training_batch_steps: int = 0; render_frame_index: int = 0
    last_periodic_renderer_reallocation_time: float | None = None
    training_elapsed_s: float = 0.0; training_resume_time: float | None = None
    photometric_elapsed_s: float = 0.0; photometric_resume_time: float | None = None
    last_interaction_time: float = 0.0
    cached_raster_grad_histograms: ParamLog10Histograms | None = None
    cached_raster_grad_ranges: ParamTensorRanges | None = None
    cached_raster_grad_histogram_mode: str = ""
    cached_raster_grad_histogram_step: int = -1
    cached_raster_grad_histogram_scene_count: int = -1
    cached_raster_grad_histogram_signature: tuple[object, ...] | None = None
    cached_raster_grad_histogram_status: str = ""
    camera_overlay_world_segments: np.ndarray | None = None
    camera_overlay_frame_indices: np.ndarray | None = None
    camera_overlay_world_positions: np.ndarray | None = None
    camera_overlay_signature: tuple[object, ...] | None = None
    training_camera_colmap_observation_index: dict[int, tuple[tuple[int, str], ...]] | None = None
    training_camera_colmap_observation_signature: tuple[object, ...] | None = None
    training_camera_colmap_payload: dict[str, object] | None = None
    training_camera_colmap_payload_signature: tuple[object, ...] | None = None
    training_camera_colmap_payload_cache: dict[tuple[object, ...], dict[str, object]] | None = None
    training_camera_colmap_payload_cache_signature: tuple[object, ...] | None = None
    camera_reset_position: tuple[float, float, float] | None = None
    camera_reset_up: tuple[float, float, float] | None = None
    camera_reset_yaw: float | None = None
    camera_reset_pitch: float | None = None
    camera_reset_near: float | None = None
    camera_reset_far: float | None = None
    camera_reset_move_speed: float | None = None
    camera_pos: spy.float3 = field(default_factory=_default_camera_pos); yaw: float = 0.0; pitch: float = 0.0
    up: spy.float3 = field(default_factory=_default_up); fov_y: float = 60.0; near: float = 0.1; far: float = 120.0
    move_speed: float = 2.0; look_speed: float = 0.003; background: spy.float3 = field(default_factory=_default_background)
    keys: dict[spy.KeyCode, bool] = field(default_factory=dict); mouse_left: bool = False; mouse_right: bool = False; mouse_delta: spy.float2 = field(default_factory=_default_mouse_delta)
    scroll_delta: float = 0.0; move_vel: spy.float3 = field(default_factory=_default_move_vel); rot_vel: spy.float2 = field(default_factory=_default_rot_vel)
    mx: float | None = None; my: float | None = None; last_time: float = field(default_factory=time.perf_counter); fps_smooth: float = 60.0
    last_error: str = ""; last_resize_exception: str = ""; last_render_exception: str = ""
LOSS_DEBUG_OPTIONS = (
    ("rendered", "Rendered"),
    ("target", "Target"),
    ("abs_diff", "Abs Diff"),
    ("dssim", "DSSIM"),
    ("rendered_edges", "Rendered Edges"),
    ("target_edges", "Target Edges"),
)
