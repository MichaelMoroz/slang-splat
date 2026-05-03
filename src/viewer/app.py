from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import math
from pathlib import Path
import time

import numpy as np
import slangpy as spy
from slangpy import math as smath

from .. import create_default_device
from ..repo_defaults import defaults_path, load_defaults, write_defaults
from ..app.training_controls import TRAINING_BUILD_ARG_UI_KEYS
from ..app.shared import RendererParams, build_init_params, build_training_params, fit_camera
from ..filter import SeparableGaussianBlur
from ..metrics import Metrics
from ..scan.prefix_sum import GPUPrefixSum
from ..sort.radix_sort import GPURadixSort
from ..training import AdamOptimizer, GaussianOptimizer, GaussianTrainer, resolve_sh_band
from ..utility import SHADER_ROOT, drain_deferred_resource_releases, load_compute_items, load_compute_kernels, normalize3
from ..renderer import Camera, GaussianRenderSettings, GaussianRenderer
from ..scene import GaussianScene, save_gaussian_ply
from . import presenter, session
from .constants import _WINDOW_TITLE
from .state import (
    DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH, DEFAULT_MAX_PREPASS_MEMORY_MB,
    LOSS_DEBUG_OPTIONS, ViewerState,
)
from .ui import _threshold_band_range, build_ui, create_toolkit_window, default_control_values, export_repo_defaults_from_ui_values

_VIEW_VEC_EPS = 1e-6
_SCROLL_SPEED_BASE = 1.1
_LOOK_SMOOTH = 12.0
_MOVE_SMOOTH = 10.0
_PITCH_LIMIT = math.radians(89.0)
_TRAINING_PARAM_KEYS = TRAINING_BUILD_ARG_UI_KEYS
_TRAIN_SETUP_DEFAULTS = default_control_values("Train Setup")
_TRAINING_DEFAULTS = default_control_values("Train Optimizer", "Train Stability")
_METRICS_KERNEL_ENTRIES = {
    "_k_clear_uint": "csClearUIntBuffer",
    "_k_clear_float": "csClearFloatBuffer",
    "_k_scale_hist": "csHistogramScaleLog10",
    "_k_anisotropy_hist": "csHistogramAnisotropyLog10",
    "_k_param_tensor_hist": "csHistogramParamTensorLog10",
    "_k_param_tensor_hist_linear": "csHistogramParamTensorLinear",
    "_k_scene_param_hist_linear": "csHistogramSceneParamsLinear",
    "_k_refinement_distribution_hist": "csHistogramRefinementDistributionsLog10",
    "_k_init_param_ranges": "csInitParamTensorRanges",
    "_k_param_tensor_range": "csRangeParamTensor",
    "_k_scene_param_range": "csRangeSceneParams",
    "_k_refinement_distribution_range": "csRangeRefinementDistributions",
    "_k_image_mse": "csAccumulateImageMSE",
}
_VIEWER_DEBUG_KERNEL_ENTRIES = {
    "debug_abs_diff_kernel": "csComposeAbsDiffDebug",
    "debug_edge_kernel": "csComposeEdgeDebug",
    "debug_dssim_features_kernel": "csComputeSSIMFeaturesDebug",
    "debug_dssim_compose_kernel": "csComposeDSSIMDebug",
    "debug_letterbox_kernel": "csComposeLetterboxDebug",
    "debug_target_sample_kernel": "csSampleTrainingDebugTarget",
}
_ADAM_KERNEL_ENTRIES = {
    "compute_grad_norms": "csComputePackedElementGradNorms",
    "clip_grads": "csClipPackedParamGrads",
    "adam_step": "csAdamStepPacked",
    "regularize": "csRegularizePacked",
}
_GAUSSIAN_OPTIMIZER_KERNEL_ENTRIES = {"project_params": "csProjectGaussianParams"}
_PREFIX_SUM_ITEM_SPECS = {
    "scan_blocks": ("kernel", SHADER_ROOT / "utility" / "prefix_sum" / "prefix_sum.slang", "csPrefixScanBlocks"),
    "add_offsets": ("kernel", SHADER_ROOT / "utility" / "prefix_sum" / "prefix_sum.slang", "csPrefixAddOffsets"),
    "write_total_kernel": ("kernel", SHADER_ROOT / "utility" / "prefix_sum" / "prefix_sum.slang", "csPrefixWriteTotal"),
    "scan_blocks_float": ("kernel", SHADER_ROOT / "utility" / "prefix_sum" / "prefix_sum.slang", "csPrefixScanBlocksFloat"),
    "add_offsets_float": ("kernel", SHADER_ROOT / "utility" / "prefix_sum" / "prefix_sum.slang", "csPrefixAddOffsetsFloat"),
    "write_total_kernel_float": ("kernel", SHADER_ROOT / "utility" / "prefix_sum" / "prefix_sum.slang", "csPrefixWriteTotalFloat"),
    "compute_dispatch_args_from_buffer_kernel": ("kernel", SHADER_ROOT / "utility" / "prefix_sum" / "prefix_sum.slang", "csComputeDispatchArgsFromBuffer"),
    "compute_prefix_args_from_buffer_kernel": ("kernel", SHADER_ROOT / "utility" / "prefix_sum" / "prefix_sum.slang", "csComputePrefixIndirectArgsFromBuffer"),
    "scan_blocks_pipeline": ("pipeline", SHADER_ROOT / "utility" / "prefix_sum" / "prefix_sum.slang", "csPrefixScanBlocks"),
    "add_offsets_pipeline": ("pipeline", SHADER_ROOT / "utility" / "prefix_sum" / "prefix_sum.slang", "csPrefixAddOffsets"),
}
_RADIX_SORT_ITEM_SPECS = {
    "compute_args": ("kernel", SHADER_ROOT / "utility" / "radix_sort" / "compute_indirect_args.slang", "csComputeIndirectArgs"),
    "compute_args_from_buffer": ("kernel", SHADER_ROOT / "utility" / "radix_sort" / "compute_indirect_args_from_buffer.slang", "csComputeIndirectArgsFromBuffer"),
    "histogram": ("pipeline", SHADER_ROOT / "utility" / "radix_sort" / "histogram.slang", "csRadixHistogram"),
    "prefix_level": ("pipeline", SHADER_ROOT / "utility" / "radix_sort" / "prefix_block.slang", "csRadixPrefixLevel"),
    "prefix_add": ("pipeline", SHADER_ROOT / "utility" / "prefix_sum" / "prefix_sum.slang", "csPrefixAddOffsets"),
    "scatter": ("pipeline", SHADER_ROOT / "utility" / "radix_sort" / "scatter.slang", "csRadixScatter"),
}
_DEBUG_MODE_VALUES = (
    GaussianRenderer.DEBUG_MODE_NORMAL,
    GaussianRenderer.DEBUG_MODE_PROCESSED_COUNT,
    GaussianRenderer.DEBUG_MODE_SPLAT_AGE,
    GaussianRenderer.DEBUG_MODE_ELLIPSE_OUTLINES,
    GaussianRenderer.DEBUG_MODE_SPLAT_DENSITY,
    GaussianRenderer.DEBUG_MODE_SPLAT_SPATIAL_DENSITY,
    GaussianRenderer.DEBUG_MODE_SPLAT_SCREEN_DENSITY,
    GaussianRenderer.DEBUG_MODE_CONTRIBUTION_AMOUNT,
    GaussianRenderer.DEBUG_MODE_ADAM_MOMENTUM,
    GaussianRenderer.DEBUG_MODE_ADAM_SECOND_MOMENT,
    GaussianRenderer.DEBUG_MODE_GRAD_VARIANCE,
    GaussianRenderer.DEBUG_MODE_REFINEMENT_DISTRIBUTION,
    GaussianRenderer.DEBUG_MODE_DEPTH_MEAN,
    GaussianRenderer.DEBUG_MODE_DEPTH_STD,
    GaussianRenderer.DEBUG_MODE_DEPTH_LOCAL_MISMATCH,
    GaussianRenderer.DEBUG_MODE_GRAD_NORM,
    GaussianRenderer.DEBUG_MODE_SH_VIEW_DEPENDENT,
    GaussianRenderer.DEBUG_MODE_SH_COEFFICIENT,
    GaussianRenderer.DEBUG_MODE_BLACK_NEGATIVE,
)



def _raster_grad_kernel_entries(entry_suffix: str) -> dict[str, str]:
    return {
        "training_forward": f"csRasterizeTrainingForward{entry_suffix}",
        "clear": f"csClearRasterGrads{entry_suffix}",
        "backward": f"csRasterizeBackward{entry_suffix}",
        "resolve_stats": f"csResolveGradientStats{entry_suffix}",
        "backprop": f"csBackpropCachedRasterGrads{entry_suffix}",
    }


def _precompile_runtime_shaders(device: spy.Device) -> None:
    load_compute_items(
        device,
        {
            attr: (kind, SHADER_ROOT / "renderer" / shader_name, entry)
            for attr, kind, shader_name, entry in GaussianRenderer._SHADERS
        },
    )
    load_compute_kernels(device, SHADER_ROOT / "renderer" / "gaussian_raster_stage.slang", _raster_grad_kernel_entries("Fixed"))
    load_compute_kernels(device, SHADER_ROOT / "renderer" / "gaussian_raster_stage.slang", _raster_grad_kernel_entries("Float"))
    load_compute_kernels(device, SHADER_ROOT / "utility" / "metrics" / "metrics.slang", _METRICS_KERNEL_ENTRIES)
    load_compute_kernels(
        device,
        SHADER_ROOT / "renderer" / "gaussian_training_stage.slang",
        {name: entry for name, (shader_path, entry) in GaussianTrainer._KERNEL_ENTRIES.items() if Path(shader_path) == SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"},
    )
    load_compute_kernels(device, SHADER_ROOT / "renderer" / "gaussian_training_stage.slang", _VIEWER_DEBUG_KERNEL_ENTRIES)
    load_compute_kernels(device, SHADER_ROOT / "utility" / "blur" / "separable_gaussian_blur.slang", SeparableGaussianBlur._KERNEL_ENTRIES)
    load_compute_kernels(device, SHADER_ROOT / "utility" / "optimizer" / "optimizer.slang", _ADAM_KERNEL_ENTRIES)
    load_compute_kernels(device, SHADER_ROOT / "utility" / "optimizer" / "gaussian_optimizer_stage.slang", _GAUSSIAN_OPTIMIZER_KERNEL_ENTRIES)
    load_compute_items(device, _PREFIX_SUM_ITEM_SPECS)
    load_compute_items(device, _RADIX_SORT_ITEM_SPECS)


def _mark_recent_interaction(viewer: object, timestamp: float | None = None) -> None:
    resolved = float(timestamp) if timestamp is not None else time.perf_counter()
    viewer.s.last_interaction_time = resolved


def _viewer_ui_values(viewer: object) -> dict[str, object]:
    ui = viewer.ui
    try:
        values = ui._values
    except AttributeError:
        values = None
    if isinstance(values, dict):
        return values
    controls = ui.controls
    return {str(key): control.value for key, control in controls.items()}


def _canonical_viewer_up() -> spy.float3:
    return spy.float3(0.0, 1.0, 0.0)


@dataclass(slots=True)
class _ViewerRenderContext:
    surface_texture: spy.Texture
    command_encoder: spy.CommandEncoder


class _ViewerWindowHost:
    def __init__(
        self,
        app: spy.App,
        *,
        width: int,
        height: int,
        title: str,
        resizable: bool,
        enable_vsync: bool,
        surface_format: spy.Format = spy.Format.undefined,
    ) -> None:
        self._app = app
        self._device = app.device
        self._window_width = max(int(width), 1)
        self._window_height = max(int(height), 1)
        self._window_title = str(title)
        self._window_resizable = bool(resizable)
        self._surface_format = surface_format
        self._enable_vsync = bool(enable_vsync)
        self._window: spy.Window | None = None
        self._surface: spy.Surface | None = None
        self._window_position: spy.int2 | None = None
        self._terminated = False
        self._exit_confirmed = False
        self._recreate_window(open_exit_confirmation=False)

    @property
    def device(self) -> spy.Device:
        return self._device

    def _bind_window_events(self) -> None:
        if self._window is None:
            return
        self._window.on_resize = self._on_window_resize
        self._window.on_keyboard_event = self.on_keyboard_event
        self._window.on_mouse_event = self.on_mouse_event

    def _configure_surface(self) -> None:
        if self._surface is None:
            return
        self._surface.configure(
            self._window_width,
            self._window_height,
            format=self._surface_format,
            vsync=self._enable_vsync,
        )

    def _recreate_window(self, *, open_exit_confirmation: bool) -> None:
        previous_window = self._window
        if previous_window is not None:
            try:
                self._window_position = previous_window.position
            except Exception:
                self._window_position = None
            try:
                self._window_width = max(int(previous_window.width), 1)
                self._window_height = max(int(previous_window.height), 1)
            except Exception:
                pass
        if self._surface is not None:
            try:
                self._surface.unconfigure()
            except Exception:
                pass
        self._surface = None
        self._window = spy.Window(
            width=self._window_width,
            height=self._window_height,
            title=self._window_title,
            resizable=self._window_resizable,
        )
        if self._window_position is not None:
            try:
                self._window.position = self._window_position
            except Exception:
                pass
        self._bind_window_events()
        self._surface = self.device.create_surface(self._window)
        self._configure_surface()
        if open_exit_confirmation:
            _request_exit_confirmation(self)

    def _on_window_resize(self, width: int, height: int) -> None:
        self._window_width = max(int(width), 1)
        self._window_height = max(int(height), 1)
        self._configure_surface()
        self.on_resize(width, height)

    def close(self) -> None:
        self._terminated = True
        window = self._window
        if window is not None and not window.should_close():
            window.close()

    def run(self) -> None:
        while not self._terminated:
            window = self._window
            surface = self._surface
            if window is None or surface is None:
                raise RuntimeError("Viewer window host is not initialized")
            window.process_events()
            if self._terminated:
                break
            if window.should_close():
                if self._exit_confirmed:
                    break
                self._recreate_window(open_exit_confirmation=True)
                continue
            try:
                surface_texture = surface.acquire_next_image()
            except Exception:
                if window.should_close():
                    if self._exit_confirmed:
                        break
                    self._recreate_window(open_exit_confirmation=True)
                else:
                    self._configure_surface()
                continue
            command_encoder = self.device.create_command_encoder()
            self.render(_ViewerRenderContext(surface_texture=surface_texture, command_encoder=command_encoder))
            self.device.submit_command_buffer(command_encoder.finish())
            surface.present()

    def shutdown(self) -> None:
        if self._surface is not None:
            try:
                self._surface.unconfigure()
            except Exception:
                pass
            self._surface = None
        self._window = None


def _set_viewer_ui_value(viewer: object, key: str, value: object) -> None:
    ui = viewer.ui
    try:
        values = ui._values
    except AttributeError:
        values = None
    if not isinstance(values, dict):
        values = {}
        ui._values = values
    values[str(key)] = value


def _set_exit_confirmation_open(viewer: object, value: bool) -> None:
    _set_viewer_ui_value(viewer, "_exit_confirmation_open", bool(value))


def _request_exit_confirmation(viewer: object) -> None:
    viewer._exit_confirmed = False
    _set_exit_confirmation_open(viewer, True)


def _cancel_exit_confirmation(viewer: object) -> None:
    _set_exit_confirmation_open(viewer, False)


def _confirm_exit(viewer: object) -> None:
    _set_exit_confirmation_open(viewer, False)
    viewer._exit_confirmed = True
    close = getattr(viewer, "close", None)
    if callable(close):
        close()
        return
    app = getattr(viewer, "_app", None)
    if app is not None and hasattr(app, "terminate"):
        app.terminate()


def _yaw_pitch_from_forward(forward: np.ndarray) -> tuple[float, float]:
    direction = np.asarray(normalize3(forward, eps=_VIEW_VEC_EPS), dtype=np.float32).reshape(3)
    yaw = math.atan2(float(direction[0]), float(direction[2]))
    pitch = math.asin(max(min(float(direction[1]), 1.0), -1.0))
    return float(yaw), float(pitch)

def _training_param_value(name: str, value_for) -> object:
    value = value_for(_TRAINING_PARAM_KEYS[name])
    return int(value) if name == "train_subsample_factor" else value


def _training_kwargs(value_for) -> dict[str, object]:
    return {name: _training_param_value(name, value_for) for name in _TRAINING_PARAM_KEYS}


def _default_training_control_value(control: str) -> object:
    return _TRAIN_SETUP_DEFAULTS[control] if control in _TRAIN_SETUP_DEFAULTS else _TRAINING_DEFAULTS[control]


def _training_background_value(value_for) -> tuple[float, float, float]:
    return tuple(float(v) for v in np.asarray(value_for("train_background_color"), dtype=np.float32).reshape(3))


def _viewer_background_value(value_for) -> tuple[float, float, float]:
    if int(value_for("render_background_mode")) == 0:
        return _training_background_value(value_for)
    return tuple(float(v) for v in np.asarray(value_for("render_background_color"), dtype=np.float32).reshape(3))


def default_training_params(background=(1.0, 1.0, 1.0)):
    return build_training_params(background=background, **_training_kwargs(_default_training_control_value))


def _initial_renderer_params(state: object) -> RendererParams:
    return replace(
        RendererParams(),
        list_capacity_multiplier=int(getattr(state, "list_capacity_multiplier")),
        max_prepass_memory_mb=int(getattr(state, "max_prepass_memory_mb")),
    )


class SplatViewer(_ViewerWindowHost):

    def c(self, key: str):
        return self.ui.control(key)

    def t(self, key: str):
        return self.ui.text(key)

    def renderer_params(self, allow_debug_overlays: bool) -> RendererParams:
        ui_values = _viewer_ui_values(self)
        params = RendererParams.from_ui_values(ui_values, _DEBUG_MODE_VALUES, _threshold_band_range)
        debug_mode = params.debug_mode if allow_debug_overlays else GaussianRenderer.DEBUG_MODE_NORMAL
        return replace(
            params,
            list_capacity_multiplier=self.s.list_capacity_multiplier,
            max_prepass_memory_mb=self.s.max_prepass_memory_mb,
            debug_mode=debug_mode,
            debug_show_ellipses=debug_mode == GaussianRenderer.DEBUG_MODE_ELLIPSE_OUTLINES,
            debug_show_processed_count=debug_mode == GaussianRenderer.DEBUG_MODE_PROCESSED_COUNT,
            debug_show_grad_norm=debug_mode == GaussianRenderer.DEBUG_MODE_GRAD_NORM,
        )

    def init_params(self):
        return build_init_params(None, None, None, self.c("init_opacity").value, self.c("seed").value)

    def _forward(self) -> spy.float3:
        cy, sy = math.cos(self.s.yaw), math.sin(self.s.yaw)
        cp, sp = math.cos(self.s.pitch), math.sin(self.s.pitch)
        return normalize3(spy.float3(cp * sy, sp, cp * cy), eps=_VIEW_VEC_EPS)

    def camera(self) -> Camera:
        forward = self._forward()
        return Camera.look_at(position=self.s.camera_pos, target=self.s.camera_pos + forward, up=self.s.up, fov_y_degrees=float(self.s.fov_y), near=float(self.s.near), far=float(self.s.far))

    def apply_camera_fit(self, bounds) -> None:
        fit = fit_camera(bounds, self.s.fov_y)
        self.s.camera_pos = fit.position
        self.s.up = _canonical_viewer_up()
        self.s.near = fit.near
        self.s.far = fit.far
        self.s.move_speed = fit.move_speed
        self.c("move_speed").value = float(fit.move_speed)
        self.s.yaw = 0.0
        self.s.pitch = 0.0
        self.s.move_vel = spy.float3(0.0, 0.0, 0.0)
        self.s.rot_vel = spy.float2(0.0, 0.0)

    def apply_camera_position(self, camera_or_position: object, *, near: float | None = None, far: float | None = None, move_speed: float | None = None) -> None:
        position = np.asarray(getattr(camera_or_position, "position", camera_or_position), dtype=np.float32).reshape(-1)
        self.s.camera_pos = spy.float3(*position[:3].tolist())
        if near is not None:
            self.s.near = float(near)
        if far is not None:
            self.s.far = float(far)
        if move_speed is not None:
            self.s.move_speed = float(move_speed)
            self.c("move_speed").value = float(move_speed)
        self.s.move_vel = spy.float3(0.0, 0.0, 0.0)
        self.s.rot_vel = spy.float2(0.0, 0.0)

    def apply_camera_pose(self, camera: Camera, *, near: float | None = None, far: float | None = None, move_speed: float | None = None) -> None:
        resolved_camera = camera if isinstance(camera, Camera) else Camera(
            position=np.asarray(getattr(camera, "position", (0.0, 0.0, -3.0)), dtype=np.float32),
            target=np.asarray(getattr(camera, "target", (0.0, 0.0, 0.0)), dtype=np.float32),
            up=np.asarray(getattr(camera, "up", (0.0, 1.0, 0.0)), dtype=np.float32),
            near=float(getattr(camera, "near", 0.1)),
            far=float(getattr(camera, "far", 120.0)),
        )
        yaw, pitch = _yaw_pitch_from_forward(np.asarray(resolved_camera.target - resolved_camera.position, dtype=np.float32))
        self.s.camera_pos = spy.float3(*np.asarray(resolved_camera.position, dtype=np.float32).tolist())
        self.s.up = _canonical_viewer_up()
        self.s.near = float(resolved_camera.near if near is None else near)
        self.s.far = float(resolved_camera.far if far is None else far)
        if move_speed is not None:
            self.s.move_speed = float(move_speed)
            self.c("move_speed").value = float(move_speed)
        self.s.yaw = yaw
        self.s.pitch = pitch
        self.s.move_vel = spy.float3(0.0, 0.0, 0.0)
        self.s.rot_vel = spy.float2(0.0, 0.0)

    def on_keyboard_event(self, event) -> None:
        if event.type in (spy.KeyboardEventType.key_press, spy.KeyboardEventType.key_release):
            _mark_recent_interaction(self)
        if self.toolkit.handle_keyboard_event(event):
            if event.type == spy.KeyboardEventType.key_release:
                self.s.keys[event.key] = False
            return
        if event.type in (spy.KeyboardEventType.key_press, spy.KeyboardEventType.key_release):
            self.s.keys[event.key] = event.type == spy.KeyboardEventType.key_press

    def training_params(self):
        return build_training_params(background=_training_background_value(self._training_control_value), **_training_kwargs(self._training_control_value))

    def render_background(self) -> spy.float3:
        return spy.float3(*_viewer_background_value(self._training_control_value))

    def _training_control_value(self, control: str) -> object:
        return self.c(control).value

    def _apply_resize(self, width: int, height: int) -> None:
        target_width, target_height = int(width), int(height)
        toolkit = getattr(self, "toolkit", None)
        viewport_size = None if toolkit is None else getattr(toolkit, "viewport_size", None)
        if callable(viewport_size):
            viewport_width, viewport_height = viewport_size()
            if int(viewport_width) > 0 and int(viewport_height) > 0:
                target_width, target_height = int(viewport_width), int(viewport_height)
        self.device.wait()
        if target_width > 0 and target_height > 0 and (self.s.renderer.width, self.s.renderer.height) != (target_width, target_height):
            session.recreate_renderer(self, target_width, target_height)
        self.s.last_resize_exception = ""
        self.s.last_error = ""

    def on_resize(self, width: int, height: int) -> None:
        try:
            self._apply_resize(width, height)
        except Exception as exc:
            self.s.last_resize_exception = str(exc)
            self.s.last_error = self.s.last_resize_exception

    def on_mouse_event(self, event) -> None:
        if self.toolkit.handle_mouse_event(event):
            if event.type in (spy.MouseEventType.button_down, spy.MouseEventType.button_up, spy.MouseEventType.move, spy.MouseEventType.scroll):
                _mark_recent_interaction(self)
            if event.type == spy.MouseEventType.move:
                self.s.mx = event.pos.x
                self.s.my = event.pos.y
                self.s.mouse_delta = spy.float2(0.0, 0.0)
            elif event.type in (spy.MouseEventType.button_down, spy.MouseEventType.button_up):
                if event.button == spy.MouseButton.left:
                    self.s.mouse_left = False
                elif event.button == spy.MouseButton.right:
                    self.s.mouse_right = False
            return
        if event.type in (spy.MouseEventType.button_down, spy.MouseEventType.button_up, spy.MouseEventType.scroll):
            _mark_recent_interaction(self)
        if event.type in (spy.MouseEventType.button_down, spy.MouseEventType.button_up) and event.button == spy.MouseButton.left:
            self.s.mouse_left = event.type == spy.MouseEventType.button_down
        if event.type in (spy.MouseEventType.button_down, spy.MouseEventType.button_up) and event.button == spy.MouseButton.right:
            self.s.mouse_right = event.type == spy.MouseEventType.button_down
        if event.type == spy.MouseEventType.move:
            if self.s.mouse_left or self.s.mouse_right:
                _mark_recent_interaction(self)
            if self.s.mx is not None and self.s.my is not None:
                self.s.mouse_delta += spy.float2(event.pos.x - self.s.mx, event.pos.y - self.s.my)
            self.s.mx = event.pos.x
            self.s.my = event.pos.y
        if event.type == spy.MouseEventType.scroll:
            self.s.scroll_delta += float(event.scroll.y)

    def __init__(self, app: spy.App, width: int = 1280, height: int = 720, title: str = _WINDOW_TITLE, max_prepass_memory_mb: int = 4096) -> None:
        super().__init__(app, width=width, height=height, title=title, resizable=True, enable_vsync=False)
        self._exit_confirmed = False
        self.loss_debug_view_options = LOSS_DEBUG_OPTIONS
        self.s = ViewerState(max_prepass_memory_mb=max(int(max_prepass_memory_mb), 1))
        _precompile_runtime_shaders(self.device)
        self.s.renderer = GaussianRenderSettings.from_renderer_params(width, height, _initial_renderer_params(self.s)).create_renderer(self.device)
        self.ui = build_ui(self.s.renderer)
        self.toolkit = create_toolkit_window(self.device, width, height)
        self._bind_toolkit_callbacks()
        session.create_debug_shaders(self)

    def _bind_toolkit_callbacks(self) -> None:
        cb = self.toolkit.callbacks
        cb.load_ply = self._load_ply_callback
        cb.export_ply = self._export_ply_callback
        cb.browse_colmap_root = self._browse_colmap_root_callback
        cb.browse_colmap_images = self._browse_colmap_images_callback
        cb.browse_colmap_depth = self._browse_colmap_depth_callback
        cb.browse_colmap_ply = self._browse_colmap_ply_callback
        cb.browse_colmap_mesh = self._browse_colmap_mesh_callback
        cb.import_colmap = self._import_colmap_callback
        cb.reload = self._reload_callback
        cb.reinitialize = self._reinitialize_callback
        cb.request_exit = self._request_exit_callback
        cb.confirm_exit = self._confirm_exit_callback
        cb.cancel_exit = self._cancel_exit_callback
        cb.start_training = self._start_training_callback
        cb.stop_training = self._stop_training_callback
        cb.move_to_training_camera = self._move_to_training_camera_callback
        cb.reset_camera = self._reset_camera_callback
        cb.save_defaults = self._save_defaults_callback

    def _request_exit_callback(self) -> None:
        _request_exit_confirmation(self)

    def _confirm_exit_callback(self) -> None:
        _confirm_exit(self)

    def _cancel_exit_callback(self) -> None:
        _cancel_exit_confirmation(self)

    def _run_action(self, action, *, close_colmap_import: bool = False) -> None:
        try:
            action()
        except Exception as exc:
            self.s.last_error = str(exc)
        else:
            self.s.last_error = ""
            if close_colmap_import:
                self.toolkit.close_colmap_import_window()

    def _load_ply_callback(self) -> None:
        path = spy.platform.open_file_dialog([spy.platform.FileDialogFilter("PLY Files", "*.ply")])
        if path:
            self._run_action(lambda: session.load_scene(self, Path(path)))

    def _export_source_scene(self) -> GaussianScene:
        if self.s.trainer is not None:
            return self.s.trainer.read_live_scene()
        if isinstance(self.s.scene, GaussianScene):
            return self.s.scene
        raise RuntimeError("No gaussian scene is available to export.")

    def _export_should_include_sh(self) -> bool:
        if self.s.trainer is not None:
            return resolve_sh_band(self.s.trainer.training, self.s.trainer.state.step) > 0
        training = self.training_params().training
        return int(getattr(training, "sh_band", 3 if bool(getattr(training, "use_sh", False)) else 0)) > 0

    def _export_ply_callback(self) -> None:
        path = spy.platform.save_file_dialog([spy.platform.FileDialogFilter("PLY Files", "*.ply")])
        if path is None:
            return
        def _export() -> None:
            scene = SplatViewer._export_source_scene(self)
            include_sh = SplatViewer._export_should_include_sh(self)
            export_path = Path(path)
            export_path = export_path.with_suffix(".ply") if export_path.suffix.lower() != ".ply" else export_path
            saved_path = save_gaussian_ply(export_path, scene, include_sh=include_sh)
            print(f"Exported scene: {saved_path} ({scene.count:,} splats)")
        self._run_action(_export)

    def _browse_colmap_root_callback(self) -> None:
        path = spy.platform.choose_folder_dialog()
        if path:
            self._run_action(lambda: session.choose_colmap_root(self, Path(path)))

    def _browse_colmap_images_callback(self) -> None:
        path = spy.platform.choose_folder_dialog()
        if path:
            self._run_action(lambda: session.choose_colmap_images_root(self, Path(path)))

    def _browse_colmap_depth_callback(self) -> None:
        path = spy.platform.choose_folder_dialog()
        if path:
            self._run_action(lambda: session.choose_colmap_depth_root(self, Path(path)))

    def _browse_colmap_ply_callback(self) -> None:
        path = spy.platform.open_file_dialog([spy.platform.FileDialogFilter("PLY Files", "*.ply")])
        if path:
            self._run_action(lambda: session.choose_colmap_custom_ply(self, Path(path)))

    def _browse_colmap_mesh_callback(self) -> None:
        path = spy.platform.open_file_dialog([
            spy.platform.FileDialogFilter("Mesh Files", "*.obj;*.gltf;*.glb;*.ply;*.stl;*.off"),
            spy.platform.FileDialogFilter("All Files", "*.*"),
        ])
        if path:
            self._run_action(lambda: session.choose_colmap_custom_mesh(self, Path(path)))

    def _import_colmap_callback(self) -> None:
        self._run_action(lambda: session.import_colmap_from_ui(self))

    def _reload_callback(self) -> None:
        if self.s.scene_path is not None:
            self._run_action(lambda: session.load_scene(self, self.s.scene_path))
        elif self.s.colmap_root is not None:
            import_cfg = self.s.colmap_import
            if import_cfg.images_root is None:
                self.s.last_error = "COLMAP reload requires a stored image folder."
                return
            self._run_action(
                lambda: session.import_colmap_dataset(
                    self,
                    colmap_root=self.s.colmap_root,
                    database_path=import_cfg.database_path,
                    images_root=import_cfg.images_root,
                    depth_root=import_cfg.depth_root,
                    init_mode=import_cfg.init_mode,
                    auto_rotate_scene=import_cfg.auto_rotate_scene,
                    custom_ply_path=import_cfg.custom_ply_path,
                    image_downscale_mode=import_cfg.image_downscale_mode,
                    image_downscale_max_size=import_cfg.image_downscale_max_size,
                    image_downscale_scale=import_cfg.image_downscale_scale,
                    nn_radius_scale_coef=import_cfg.nn_radius_scale_coef,
                    selected_camera_ids=tuple(int(camera_id) for camera_id in getattr(import_cfg, "selected_camera_ids", ())),
                    min_track_length=int(getattr(import_cfg, "min_track_length", DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH)),
                    depth_point_count=import_cfg.depth_point_count,
                    diffused_point_count=import_cfg.diffused_point_count,
                    fibonacci_sphere_point_count=import_cfg.fibonacci_sphere_point_count,
                    fibonacci_sphere_radius=import_cfg.fibonacci_sphere_radius,
                    use_target_alpha_mask=import_cfg.use_target_alpha_mask,
                    pointcloud_enabled=bool(getattr(import_cfg, "pointcloud_enabled", False)),
                    pointcloud_nn_radius_scale_coef=getattr(import_cfg, "pointcloud_nn_radius_scale_coef", None),
                    diffused_enabled=bool(getattr(import_cfg, "diffused_enabled", False)),
                    diffused_diffusion_radius=getattr(import_cfg, "diffused_diffusion_radius", None),
                    diffused_nn_radius_scale_coef=getattr(import_cfg, "diffused_nn_radius_scale_coef", None),
                    custom_ply_enabled=bool(getattr(import_cfg, "custom_ply_enabled", False)),
                    custom_ply_nn_radius_scale_coef=getattr(import_cfg, "custom_ply_nn_radius_scale_coef", None),
                    custom_mesh_enabled=bool(getattr(import_cfg, "custom_mesh_enabled", False)),
                    custom_mesh_path=getattr(import_cfg, "custom_mesh_path", None),
                    custom_mesh_point_count=getattr(import_cfg, "custom_mesh_point_count", None),
                    custom_mesh_nn_radius_scale_coef=getattr(import_cfg, "custom_mesh_nn_radius_scale_coef", None),
                    fibonacci_sphere_enabled=bool(getattr(import_cfg, "fibonacci_sphere_enabled", False)),
                    fibonacci_sphere_nn_radius_scale_coef=getattr(import_cfg, "fibonacci_sphere_nn_radius_scale_coef", None),
                )
            )

    def _reinitialize_callback(self) -> None:
        self.s.training_active = False
        self.s.pending_training_reinitialize = True

    def _start_training_callback(self) -> None:
        session.set_training_active(self, True)

    def _stop_training_callback(self) -> None:
        session.set_training_active(self, False)

    def _move_to_training_camera_callback(self) -> None:
        self._run_action(lambda: session.move_main_camera_to_selected_training_frame(self))

    def _reset_camera_callback(self) -> None:
        self._run_action(lambda: session.reset_main_camera(self))

    def _save_defaults_callback(self) -> None:
        try:
            ui_values = _viewer_ui_values(self)
            defaults = load_defaults()
            defaults["training_build_args"] = {
                **defaults["training_build_args"],
                **{
                    build_arg: ui_values[control_key]
                    for build_arg, control_key in _TRAINING_PARAM_KEYS.items()
                    if control_key in ui_values
                },
            }
            exported = export_repo_defaults_from_ui_values(ui_values)
            defaults["renderer"] = exported["renderer"]
            defaults["cli"]["common_render"] = exported["cli"]["common_render"]
            defaults["viewer"]["controls"] = exported["viewer"]["controls"]
            defaults["viewer"]["import"] = exported["viewer"]["import"]
            defaults["viewer"]["ui"] = exported["viewer"]["ui"]
            write_defaults(defaults)
        except Exception as exc:
            if hasattr(self, "t"):
                self.t("defaults_status").text = ""
            if hasattr(self, "s"):
                self.s.last_error = str(exc)
            return
        if hasattr(self, "t"):
            self.t("defaults_status").text = f"Saved {defaults_path().relative_to(defaults_path().parents[1])}"
        if hasattr(self, "s"):
            self.s.last_error = ""

    def _render_frame(self, render_context) -> None:
        presenter.render_frame(self, render_context)
        self.toolkit.render(self.ui, render_context.surface_texture, render_context.command_encoder, viewport_texture=self.s.viewport_texture)
        drain_deferred_resource_releases()

    def render(self, render_context) -> None:
        try:
            self._render_frame(render_context)
        except Exception as exc:
            self.s.training_active = False
            self.s.last_error = str(exc)
            self.s.last_render_exception = self.s.last_error

    def update_camera(self, dt: float) -> None:
        self.s.move_speed, self.s.fov_y = float(self.c("move_speed").value), float(self.c("fov").value)
        scroll_active = abs(self.s.scroll_delta) > 1e-5
        if scroll_active:
            self.s.move_speed = max(self.s.move_speed * (_SCROLL_SPEED_BASE ** self.s.scroll_delta), 0.0)
            self.c("move_speed").value, self.s.scroll_delta = self.s.move_speed, 0.0
        mouse_delta = spy.float2(float(self.s.mouse_delta.x), float(self.s.mouse_delta.y))
        target_rot = mouse_delta * self.s.look_speed if self.s.mouse_left else spy.float2(0.0, 0.0)
        self.s.rot_vel += (target_rot - self.s.rot_vel) * min(1.0, _LOOK_SMOOTH * dt)
        self.s.mouse_delta = spy.float2(0.0, 0.0)
        if float(smath.length(self.s.rot_vel)) > _VIEW_VEC_EPS:
            self.s.yaw += float(self.s.rot_vel.x)
            self.s.pitch = min(max(self.s.pitch + float(self.s.rot_vel.y), -_PITCH_LIMIT), _PITCH_LIMIT)
        forward = self._forward()
        right, up = normalize3(smath.cross(self.s.up, forward), eps=_VIEW_VEC_EPS), normalize3(smath.cross(forward, normalize3(smath.cross(self.s.up, forward), eps=_VIEW_VEC_EPS)), eps=_VIEW_VEC_EPS)
        move = spy.float3(float(self.s.keys.get(spy.KeyCode.e, False)) - float(self.s.keys.get(spy.KeyCode.q, False)), float(self.s.keys.get(spy.KeyCode.d, False)) - float(self.s.keys.get(spy.KeyCode.a, False)), float(self.s.keys.get(spy.KeyCode.w, False)) - float(self.s.keys.get(spy.KeyCode.s, False)))
        move_length = float(smath.length(move))
        if scroll_active or self.s.mouse_left or self.s.mouse_right or move_length > _VIEW_VEC_EPS:
            _mark_recent_interaction(self, float(getattr(self.s, "last_time", time.perf_counter())))
        target_move = move * (self.s.move_speed / max(move_length, _VIEW_VEC_EPS)) if move_length > _VIEW_VEC_EPS else spy.float3(0.0, 0.0, 0.0)
        if self.s.mouse_right:
            drag_speed = self.s.move_speed * self.s.look_speed / max(float(dt), _VIEW_VEC_EPS)
            target_move += spy.float3(-float(mouse_delta.y) * drag_speed, -float(mouse_delta.x) * drag_speed, 0.0)
        self.s.move_vel += (target_move - self.s.move_vel) * min(1.0, _MOVE_SMOOTH * dt)
        self.s.camera_pos += (up * self.s.move_vel.x + right * self.s.move_vel.y + forward * self.s.move_vel.z) * dt

    def shutdown(self) -> None:
        self.toolkit.shutdown()
        super().shutdown()


def _compute_view_geometry() -> tuple[int, int]:
    """Pick a single-window viewer size based on the current desktop size."""
    import sys
    if sys.platform == "win32":
        import ctypes

        user32 = ctypes.windll.user32
        screen_width = int(user32.GetSystemMetrics(0))
        screen_height = int(user32.GetSystemMetrics(1))
        return max(min(int(screen_width * 0.9), 1920), 1280), max(min(int(screen_height * 0.9), 1200), 720)
    return 1600, 900

def main() -> int:
    view_w, view_h = _compute_view_geometry()
    device = create_default_device(enable_debug_layers=False)
    app = spy.App(device=device)
    viewer = SplatViewer(app, width=view_w, height=view_h, title=_WINDOW_TITLE, max_prepass_memory_mb=DEFAULT_MAX_PREPASS_MEMORY_MB)
    try:
        viewer.run()
    finally:
        viewer.shutdown()
    return 0
