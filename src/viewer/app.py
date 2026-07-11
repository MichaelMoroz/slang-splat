from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import replace
import math
from pathlib import Path
import time
import traceback

import numpy as np
import slangpy as spy
from slangpy import math as smath

from .. import create_default_device, device_type_from_name
from ..repo_defaults import defaults_path, load_defaults, write_defaults
from ..app.training_controls import TRAINING_BUILD_ARG_UI_KEYS
from ..app.shared import RendererParams, build_init_params, build_training_params, fit_camera
from ..filter import SeparableGaussianBlur
from ..metrics import Metrics
from ..scan.prefix_sum import GPUPrefixSum
from ..sort.radix_sort import GPURadixSort
from ..training import AdamOptimizer, GaussianOptimizer, GaussianTrainer, resolve_sh_band
from ..training.defaults import TRAINING_BUILD_ARG_DEFAULTS
from ..training.image_color_init import TrainingImageColorInitializer
from ..utility import SHADER_ROOT, device_type_name, drain_deferred_resource_releases, load_compute_items, load_compute_kernels, normalize3
from ..renderer import Camera, GaussianRenderSettings, GaussianRenderer
from ..scene import GaussianScene, save_gaussian_ply
from . import frame_capture, presenter, project, session, splat_editor
from .constants import _WINDOW_TITLE
from .state import (
    COLMAP_ROTATION_MODE_AUTO,
    COLMAP_ROTATION_MODE_NONE,
    DEFAULT_COLMAP_INIT_ANISOTROPY_STRENGTH,
    DEFAULT_COLMAP_INIT_NEIGHBOR_COUNT,
    DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH, DEFAULT_MAX_PREPASS_MEMORY_MB,
    LOSS_DEBUG_OPTIONS, ViewerState,
)
from .ui import _threshold_band_range, build_ui, create_toolkit_window, default_control_values, export_repo_defaults_from_ui_values
from .ui_schema import _DEBUG_MODE_VALUES, _RENDERER_DEBUG_MODE_VALUES

_VIEW_VEC_EPS = 1e-6
_SCROLL_SPEED_BASE = 1.1
_LOOK_SMOOTH = 12.0
_MOVE_SMOOTH = 10.0
_PITCH_LIMIT = math.radians(89.0)
_TRAINING_PARAM_KEYS = TRAINING_BUILD_ARG_UI_KEYS
_TRAIN_SETUP_DEFAULTS = default_control_values("Train Setup")
_TRAINING_DEFAULTS = default_control_values("Train Optimizer", "Train Stability")
_DEFAULT_TARGET_ALPHA_THRESHOLD = float(TRAINING_BUILD_ARG_DEFAULTS["target_alpha_threshold"])
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
    "adam_step": "csAdamStepPacked",
}
_GAUSSIAN_OPTIMIZER_KERNEL_ENTRIES = {
    "project_params": "csProjectGaussianParams",
}
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


def _preferred_graphics_api_name(defaults: dict[str, object] | None = None) -> str:
    resolved_defaults = load_defaults() if defaults is None else defaults
    viewer_defaults = resolved_defaults.get("viewer", {}) if isinstance(resolved_defaults, dict) else {}
    ui_defaults = viewer_defaults.get("ui", {}) if isinstance(viewer_defaults, dict) else {}
    raw_value = ui_defaults.get("graphics_api", "vulkan") if isinstance(ui_defaults, dict) else "vulkan"
    try:
        return device_type_name(device_type_from_name(str(raw_value)))
    except ValueError:
        return "vulkan"


def _graphics_api_label(value: str) -> str:
    return "DX12" if str(value) == "dx12" else "Vulkan"


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
    load_compute_kernels(device, TrainingImageColorInitializer.SHADER_PATH, TrainingImageColorInitializer.KERNEL_ENTRIES)
    load_compute_kernels(device, SHADER_ROOT / "utility" / "optimizer" / "optimizer.slang", _ADAM_KERNEL_ENTRIES)
    load_compute_kernels(device, SHADER_ROOT / "utility" / "optimizer" / "gaussian_optimizer_stage.slang", _GAUSSIAN_OPTIMIZER_KERNEL_ENTRIES)
    load_compute_items(device, _PREFIX_SUM_ITEM_SPECS)
    load_compute_items(device, _RADIX_SORT_ITEM_SPECS)


def _mark_recent_interaction(viewer: object, timestamp: float | None = None) -> None:
    state = getattr(viewer, "s", None)
    if state is None:
        return
    resolved = float(timestamp) if timestamp is not None else time.perf_counter()
    state.last_interaction_time = resolved


def _project_dir_from_dialog_path(path: str | Path) -> Path:
    target = Path(path)
    return target if target.suffix.lower() == ".splatproj" else target.with_suffix(".splatproj")


def _viewer_ui_values(viewer: object) -> dict[str, object]:
    values = getattr(getattr(viewer, "ui", None), "_values", None)
    if isinstance(values, dict):
        return values
    controls = getattr(getattr(viewer, "ui", None), "controls", {})
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
        self._surface_suspended = False
        self._window_position: spy.int2 | None = None
        self._terminated = False
        self._exit_confirmed = False
        self._ignore_close_until_present = False
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
        if self._surface is None or self._surface_suspended:
            return
        self._surface.configure(
            self._window_width,
            self._window_height,
            format=self._surface_format,
            vsync=self._enable_vsync,
        )

    def _current_window_size(self) -> tuple[int, int]:
        window = self._window
        if window is not None:
            try:
                return int(window.width), int(window.height)
            except Exception:
                pass
        return int(self._window_width), int(self._window_height)

    def _suspend_surface(self) -> None:
        if self._surface_suspended:
            return
        self._surface_suspended = True
        if self._surface is None:
            return
        try:
            self._surface.unconfigure()
        except Exception:
            pass

    def _surface_renderable(self) -> bool:
        width, height = self._current_window_size()
        return width > 0 and height > 0 and not self._surface_suspended

    def _recover_surface_failure(self, window: spy.Window, *, open_exit_confirmation: bool = True) -> None:
        if self._window_should_close(window):
            if bool(getattr(self, "_exit_confirmed", False)):
                self._terminated = True
                return
            self._recreate_window(open_exit_confirmation=open_exit_confirmation)
            return
        if not self._surface_renderable():
            self._suspend_surface()
            return
        self._surface_suspended = False
        self._configure_surface()

    def _begin_renderdoc_capture(self, window: spy.Window) -> frame_capture.RenderDocCaptureSession | None:
        if not bool(getattr(getattr(self, "s", None), "pending_renderdoc_frame_capture", False)):
            return None
        self.s.pending_renderdoc_frame_capture = False
        try:
            return frame_capture.begin_renderdoc_frame_capture(
                device=self.device,
                window=window,
                frame_index=int(getattr(self.s, "render_frame_index", 0)),
            )
        except Exception as exc:
            self.s.last_error = str(exc)
            self.s.last_render_exception = self.s.last_error
            return None

    def _finish_renderdoc_capture(self, capture_session: frame_capture.RenderDocCaptureSession | None) -> None:
        if capture_session is None:
            return
        try:
            frame_capture.end_renderdoc_frame_capture(capture_session)
        except Exception as exc:
            self.s.last_error = str(exc)
            self.s.last_render_exception = self.s.last_error

    def _recreate_window(self, *, open_exit_confirmation: bool) -> None:
        previous_window = getattr(self, "_window", None)
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
        self._surface_suspended = False
        self._ignore_close_until_present = bool(open_exit_confirmation)
        self._configure_surface()
        if open_exit_confirmation:
            _request_exit_confirmation(self)

    def _window_should_close(self, window: spy.Window) -> bool:
        return bool(window.should_close()) and not self._ignore_close_until_present

    def _on_window_resize(self, width: int, height: int) -> None:
        resized_width = int(width)
        resized_height = int(height)
        if resized_width <= 0 or resized_height <= 0:
            self._suspend_surface()
            return
        self._window_width = resized_width
        self._window_height = resized_height
        self._surface_suspended = False
        self._configure_surface()
        self.on_resize(resized_width, resized_height)

    def close(self) -> None:
        self._terminated = True
        window = self._window
        if window is None:
            return
        try:
            window.close()
        except Exception:
            pass

    def run(self) -> None:
        while not self._terminated:
            window = self._window
            surface = self._surface
            if window is None or surface is None:
                raise RuntimeError("Viewer window host is not initialized")
            window.process_events()
            if self._terminated:
                break
            if self._window_should_close(window):
                if bool(getattr(self, "_exit_confirmed", False)):
                    break
                self._recreate_window(open_exit_confirmation=True)
                continue
            if not self._surface_renderable():
                continue
            capture_session = self._begin_renderdoc_capture(window)
            try:
                surface_texture = surface.acquire_next_image()
            except Exception:
                self._finish_renderdoc_capture(capture_session)
                self._recover_surface_failure(window)
                if self._terminated:
                    break
                continue
            command_encoder = self.device.create_command_encoder()
            self.render(_ViewerRenderContext(surface_texture=surface_texture, command_encoder=command_encoder))
            self.device.submit_command_buffer(command_encoder.finish())
            try:
                surface.present()
            except Exception:
                self._finish_renderdoc_capture(capture_session)
                self._recover_surface_failure(window)
                if self._terminated:
                    break
                continue
            self._finish_renderdoc_capture(capture_session)
            self._ignore_close_until_present = False

    def shutdown(self) -> None:
        if self._surface is not None:
            try:
                self._surface.unconfigure()
            except Exception:
                pass
            self._surface = None
        self._window = None


def _set_viewer_ui_value(viewer: object, key: str, value: object) -> None:
    ui = getattr(viewer, "ui", None)
    if ui is None:
        return
    values = getattr(ui, "_values", None)
    if not isinstance(values, dict):
        values = {}
        setattr(ui, "_values", values)
    values[str(key)] = value


def _set_exit_confirmation_open(viewer: object, value: bool) -> None:
    _set_viewer_ui_value(viewer, "_exit_confirmation_open", bool(value))


def _request_exit_confirmation(viewer: object) -> None:
    setattr(viewer, "_exit_confirmed", False)
    _set_exit_confirmation_open(viewer, True)


def _cancel_exit_confirmation(viewer: object) -> None:
    _set_exit_confirmation_open(viewer, False)


def _confirm_exit(viewer: object) -> None:
    _set_exit_confirmation_open(viewer, False)
    setattr(viewer, "_exit_confirmed", True)
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


# Free-camera quaternions (wxyz, camera-to-world): right = q*x, up = q*y, forward = q*z.
_QUAT_IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    quat = np.asarray(q, dtype=np.float64).reshape(4)
    return (quat / max(float(np.linalg.norm(quat)), _VIEW_VEC_EPS)).astype(np.float32)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = (float(v) for v in np.asarray(a, dtype=np.float64).reshape(4))
    bw, bx, by, bz = (float(v) for v in np.asarray(b, dtype=np.float64).reshape(4))
    return np.array(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dtype=np.float32,
    )


def _quat_from_axis_angle(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    half = 0.5 * float(angle)
    sin_half = math.sin(half)
    return np.array((math.cos(half), axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half), dtype=np.float32)


def _quat_rotate_vec(q: np.ndarray, v: tuple[float, float, float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in np.asarray(q, dtype=np.float64).reshape(4))
    qv = np.array((x, y, z), dtype=np.float64)
    vec = np.asarray(v, dtype=np.float64).reshape(3)
    return (vec + 2.0 * np.cross(qv, np.cross(qv, vec) + w * vec)).astype(np.float32)


def _quat_from_basis(right: np.ndarray, up: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """Quaternion whose columns (right, up, forward) form the camera-to-world rotation."""
    rotation = np.stack(
        (
            np.asarray(right, dtype=np.float64).reshape(3),
            np.asarray(up, dtype=np.float64).reshape(3),
            np.asarray(forward, dtype=np.float64).reshape(3),
        ),
        axis=1,
    )
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = (0.25 * s, (rotation[2, 1] - rotation[1, 2]) / s, (rotation[0, 2] - rotation[2, 0]) / s, (rotation[1, 0] - rotation[0, 1]) / s)
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        quat = ((rotation[2, 1] - rotation[1, 2]) / s, 0.25 * s, (rotation[0, 1] + rotation[1, 0]) / s, (rotation[0, 2] + rotation[2, 0]) / s)
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        quat = ((rotation[0, 2] - rotation[2, 0]) / s, (rotation[0, 1] + rotation[1, 0]) / s, 0.25 * s, (rotation[1, 2] + rotation[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        quat = ((rotation[1, 0] - rotation[0, 1]) / s, (rotation[0, 2] + rotation[2, 0]) / s, (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s)
    return _quat_normalize(np.array(quat, dtype=np.float64))


def _quat_from_yaw_pitch(yaw: float, pitch: float, world_up: np.ndarray) -> np.ndarray:
    """Zero-roll orientation matching the horizon camera's yaw/pitch basis."""
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
    forward = np.array((cp * sy, sp, cp * cy), dtype=np.float64)
    up_ref = np.asarray(world_up, dtype=np.float64).reshape(3)
    right = np.cross(up_ref, forward)
    right /= max(float(np.linalg.norm(right)), _VIEW_VEC_EPS)
    up = np.cross(forward, right)
    up /= max(float(np.linalg.norm(up)), _VIEW_VEC_EPS)
    return _quat_from_basis(right, up, forward)


def _camera_state_rotation(state: object) -> np.ndarray:
    rotation = getattr(state, "camera_rot", None)
    return np.array(_QUAT_IDENTITY, dtype=np.float32) if rotation is None else np.asarray(rotation, dtype=np.float32).reshape(4)


def _camera_basis_from_state(state: object) -> tuple[spy.float3, spy.float3, spy.float3]:
    """(right, up, forward): horizon mode derives them from yaw/pitch around the locked
    world up; free mode reads them from the free-rotation quaternion."""
    if bool(getattr(state, "camera_free_mode", False)):
        rotation = _camera_state_rotation(state)
        right = spy.float3(*(float(v) for v in _quat_rotate_vec(rotation, (1.0, 0.0, 0.0))))
        up = spy.float3(*(float(v) for v in _quat_rotate_vec(rotation, (0.0, 1.0, 0.0))))
        forward = spy.float3(*(float(v) for v in _quat_rotate_vec(rotation, (0.0, 0.0, 1.0))))
        return right, up, forward
    cy, sy = math.cos(float(state.yaw)), math.sin(float(state.yaw))
    cp, sp = math.cos(float(state.pitch)), math.sin(float(state.pitch))
    forward = normalize3(spy.float3(cp * sy, sp, cp * cy), eps=_VIEW_VEC_EPS)
    right = normalize3(smath.cross(state.up, forward), eps=_VIEW_VEC_EPS)
    up = normalize3(smath.cross(forward, right), eps=_VIEW_VEC_EPS)
    return right, up, forward


def _sync_camera_mode(viewer: object) -> None:
    """Reconcile the UI toggle with the camera state: entering free mode adopts the
    current yaw/pitch as a zero-roll quaternion; leaving it re-levels the horizon by
    keeping only the forward direction."""
    state = viewer.s
    free = bool(getattr(getattr(viewer, "ui", None), "_values", {}).get("camera_free_mode", False))
    if free == bool(getattr(state, "camera_free_mode", False)):
        return
    if free:
        state.camera_rot = _quat_from_yaw_pitch(state.yaw, state.pitch, np.asarray(state.up, dtype=np.float32))
    else:
        forward = _quat_rotate_vec(_camera_state_rotation(state), (0.0, 0.0, 1.0))
        yaw, pitch = _yaw_pitch_from_forward(np.asarray(forward, dtype=np.float32))
        state.yaw = yaw
        state.pitch = min(max(pitch, -_PITCH_LIMIT), _PITCH_LIMIT)
    state.camera_free_mode = free

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


class ViewerCore:
    def c(self, key: str):
        return self.ui.control(key)

    def t(self, key: str):
        return self.ui.text(key)

    def renderer_params(self, allow_debug_overlays: bool) -> RendererParams:
        ui_values = _viewer_ui_values(self)
        params = RendererParams.from_ui_values(ui_values, _RENDERER_DEBUG_MODE_VALUES, _threshold_band_range)
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
        _, up, forward = _camera_basis_from_state(self.s)
        camera_up = up if bool(getattr(self.s, "camera_free_mode", False)) else self.s.up
        return Camera.look_at(position=self.s.camera_pos, target=self.s.camera_pos + forward, up=camera_up, fov_y_degrees=float(self.s.fov_y), near=float(self.s.near), far=float(self.s.far))

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
        self.s.camera_rot = np.array(_QUAT_IDENTITY, dtype=np.float32)
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
        # Free mode adopts the full source orientation including roll, so jumping to a
        # training camera reproduces its exact framing; the horizon representation above
        # keeps only the forward direction (re-leveled).
        try:
            basis_right, basis_up, basis_forward = resolved_camera.basis()
            self.s.camera_rot = _quat_from_basis(basis_right, basis_up, basis_forward)
        except Exception:
            self.s.camera_rot = _quat_from_yaw_pitch(yaw, pitch, np.asarray(self.s.up, dtype=np.float32))
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

    def _export_source_scene(self) -> GaussianScene:
        if self.s.trainer is not None:
            return self.s.trainer.read_live_scene()
        # Read the live GPU buffer so splat-editor edits are exported, not the loaded PLY.
        renderer = getattr(self.s, "renderer", None)
        if renderer is not None and int(getattr(renderer, "_scene_count", 0)) > 0 and hasattr(renderer, "read_live_scene"):
            return renderer.read_live_scene()
        if isinstance(self.s.scene, GaussianScene):
            return self.s.scene
        raise RuntimeError("No gaussian scene is available to export.")

    def _export_should_include_sh(self) -> bool:
        if self.s.trainer is not None:
            return resolve_sh_band(self.s.trainer.training, self.s.trainer.state.step) > 0
        training = self.training_params().training
        return int(getattr(training, "sh_band", 3 if bool(getattr(training, "use_sh", False)) else 0)) > 0

    def _run_action(self, action, *, close_colmap_import: bool = False) -> None:
        try:
            action()
        except Exception as exc:
            self.s.last_error = str(exc)
            traceback.print_exc()
        else:
            self.s.last_error = ""
            if close_colmap_import:
                self.toolkit.close_colmap_import_window()


class SplatViewer(_ViewerWindowHost, ViewerCore):

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
        cb.open_project = self._open_project_callback
        cb.save_project = self._save_project_callback
        cb.save_project_as = self._save_project_as_callback
        cb.project_collision_choice = self._project_collision_choice_callback
        cb.project_relink_choice = self._project_relink_choice_callback
        cb.browse_colmap_root = self._browse_colmap_root_callback
        cb.browse_colmap_images = self._browse_colmap_images_callback
        cb.browse_colmap_alpha_mask = self._browse_colmap_alpha_mask_callback
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
        cb.log_dataset_metrics = self._log_dataset_metrics_callback
        cb.start_photometric = self._start_photometric_callback
        cb.stop_photometric = self._stop_photometric_callback
        cb.reset_photometric = self._reset_photometric_callback
        cb.move_to_training_camera = self._move_to_training_camera_callback
        cb.reset_camera = self._reset_camera_callback
        cb.capture_python_frame = self._capture_python_frame_callback
        cb.capture_renderdoc_frame = self._capture_renderdoc_frame_callback
        cb.save_defaults = self._save_defaults_callback
        cb.set_graphics_api = self._set_graphics_api_callback
        cb.editor_select_box = self._editor_select_box_callback
        cb.editor_select_range = self._editor_select_range_callback
        cb.editor_invert_selection = self._editor_invert_selection_callback
        cb.editor_clear_selection = self._editor_clear_selection_callback
        cb.editor_reset_box = self._editor_reset_box_callback
        cb.editor_resample = self._editor_resample_callback
        cb.editor_edit_properties = self._editor_edit_properties_callback

    def _request_exit_callback(self) -> None:
        _request_exit_confirmation(self)

    def _confirm_exit_callback(self) -> None:
        _confirm_exit(self)

    def _cancel_exit_callback(self) -> None:
        _cancel_exit_confirmation(self)

    def _load_ply_callback(self) -> None:
        path = spy.platform.open_file_dialog([spy.platform.FileDialogFilter("PLY Files", "*.ply")])
        if path:
            self._run_action(lambda: session.load_scene(self, Path(path)))

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

    def _browse_colmap_alpha_mask_callback(self) -> None:
        path = spy.platform.choose_folder_dialog()
        if path:
            self._run_action(lambda: session.choose_colmap_alpha_mask_root(self, Path(path)))

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
        def _request() -> None:
            if project.request_colmap_import(self) != "unwritable":
                return
            # Mandatory creation: the default project dir is unwritable, so a
            # location must be chosen before the import may proceed.
            path = spy.platform.save_file_dialog([spy.platform.FileDialogFilter("Splat Project", "*.splatproj")])
            if path is None:
                raise RuntimeError("COLMAP import cancelled: no writable project location was chosen.")
            project.start_colmap_import(self, _project_dir_from_dialog_path(path))

        self._run_action(_request)

    def set_project_window_title(self, project_name: str | None) -> None:
        title = _WINDOW_TITLE if not project_name else f"{_WINDOW_TITLE} — {project_name}"
        self._window_title = title
        window = getattr(self, "_window", None)
        if window is not None:
            try:
                window.title = title
            except Exception:
                pass

    def _open_project_callback(self) -> None:
        path = spy.platform.open_file_dialog([
            spy.platform.FileDialogFilter("Splat Project", "project.json;*.splatproj"),
            spy.platform.FileDialogFilter("All Files", "*.*"),
        ])
        if path:
            self._run_action(lambda: project.open_project(self, Path(path)))

    def _project_save_completed(self, saved: Path, error: Exception | None) -> None:
        # Runs on the writer thread; a plain dict text update is GIL-safe.
        self.t("defaults_status").text = f"Project save failed: {error}" if error else f"Saved project {saved}"

    def _save_project_callback(self) -> None:
        if self.s.project.dir is None:
            self._save_project_as_callback()
            return
        self._run_action(lambda: self._start_project_save(None))

    def _start_project_save(self, target: Path | None) -> None:
        # Resolve the display path before spawning the writer: on_complete may fire
        # before save_project's return value is assigned.
        resolved = Path(target) if target is not None else self.s.project.dir
        project.save_project(self, target, on_complete=lambda error: self._project_save_completed(resolved, error))
        self.t("defaults_status").text = f"Saving project {resolved}…"

    def _save_project_as_callback(self) -> None:
        path = spy.platform.save_file_dialog([spy.platform.FileDialogFilter("Splat Project", "*.splatproj")])
        if path is None:
            return
        target = _project_dir_from_dialog_path(path)
        self._run_action(lambda: self._start_project_save(target))

    def _project_collision_choice_callback(self, choice: str) -> None:
        chosen: Path | None = None
        if choice == "choose":
            path = spy.platform.save_file_dialog([spy.platform.FileDialogFilter("Splat Project", "*.splatproj")])
            if path is None:
                choice = "cancel"
            else:
                chosen = _project_dir_from_dialog_path(path)
        self._run_action(lambda: project.collision_choice(self, choice, chosen_dir=chosen))

    def _project_relink_choice_callback(self, choice: str) -> None:
        located: Path | None = None
        if choice == "locate":
            path = spy.platform.choose_folder_dialog()
            if path is None:
                # No folder picked: keep the pending open (and its modal) alive.
                self._run_action(lambda: project.relink_choice(self, "none"))
                return
            located = Path(path)
        self._run_action(lambda: project.relink_choice(self, choice, located_root=located))

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
                    alpha_mask_root=getattr(import_cfg, "alpha_mask_root", None),
                    use_alpha_masks=bool(getattr(import_cfg, "use_alpha_masks", False)),
                    fisheye_mask_fov_degrees=float(getattr(import_cfg, "fisheye_mask_fov_degrees", 0.0)),
                    depth_root=import_cfg.depth_root,
                    init_mode=import_cfg.init_mode,
                    rotation_mode=getattr(
                        import_cfg,
                        "rotation_mode",
                        COLMAP_ROTATION_MODE_AUTO if bool(getattr(import_cfg, "auto_rotate_scene", True)) else COLMAP_ROTATION_MODE_NONE,
                    ),
                    custom_rotation_deg=getattr(import_cfg, "custom_rotation_deg", (0.0, 0.0, 0.0)),
                    custom_ply_path=import_cfg.custom_ply_path,
                    image_downscale_mode=import_cfg.image_downscale_mode,
                    image_downscale_max_size=import_cfg.image_downscale_max_size,
                    image_downscale_scale=import_cfg.image_downscale_scale,
                    nn_radius_scale_coef=import_cfg.nn_radius_scale_coef,
                    max_pose_subset=int(getattr(import_cfg, "max_pose_subset", 0)),
                    selected_camera_ids=tuple(int(camera_id) for camera_id in getattr(import_cfg, "selected_camera_ids", ())),
                    min_track_length=int(getattr(import_cfg, "min_track_length", DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH)),
                    init_neighbor_count=int(getattr(import_cfg, "init_neighbor_count", DEFAULT_COLMAP_INIT_NEIGHBOR_COUNT)),
                    init_anisotropy_strength=float(getattr(import_cfg, "init_anisotropy_strength", DEFAULT_COLMAP_INIT_ANISOTROPY_STRENGTH)),
                    depth_point_count=import_cfg.depth_point_count,
                    diffused_point_count=import_cfg.diffused_point_count,
                    fibonacci_sphere_point_count=import_cfg.fibonacci_sphere_point_count,
                    fibonacci_sphere_radius_multiplier=getattr(import_cfg, "fibonacci_sphere_radius_multiplier", getattr(import_cfg, "fibonacci_sphere_radius", 2.0)),
                    fibonacci_sphere_color=getattr(import_cfg, "fibonacci_sphere_color", (0.8, 0.8, 0.8)),
                    fibonacci_sphere_upper_hemisphere_only=bool(getattr(import_cfg, "fibonacci_sphere_upper_hemisphere_only", False)),
                    target_alpha_mode=getattr(import_cfg, "target_alpha_mode", None),
                    target_alpha_threshold=getattr(import_cfg, "target_alpha_threshold", _DEFAULT_TARGET_ALPHA_THRESHOLD),
                    use_target_alpha_mask=bool(getattr(import_cfg, "use_target_alpha_mask", False)),
                    training_image_color_init=bool(getattr(import_cfg, "training_image_color_init", False)),
                    photometric_compensation_enabled=bool(getattr(import_cfg, "photometric_compensation_enabled", False)),
                    pointcloud_enabled=bool(getattr(import_cfg, "pointcloud_enabled", False)),
                    pointcloud_nn_radius_scale_coef=getattr(import_cfg, "pointcloud_nn_radius_scale_coef", None),
                    diffused_enabled=bool(getattr(import_cfg, "diffused_enabled", False)),
                    diffused_diffusion_radius=getattr(import_cfg, "diffused_diffusion_radius", None),
                    diffused_visibility_strength=float(getattr(import_cfg, "diffused_visibility_strength", 0.5)),
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
        self.s.pending_training_reinitialize = True

    def _start_training_callback(self) -> None:
        self._run_action(lambda: session.set_training_active(self, True))

    def _stop_training_callback(self) -> None:
        self._run_action(lambda: session.set_training_active(self, False))

    def _log_dataset_metrics_callback(self) -> None:
        self._run_action(lambda: session.start_dataset_metrics_logging(self))

    def _start_photometric_callback(self) -> None:
        self._run_action(lambda: session.set_photometric_active(self, True))

    def _stop_photometric_callback(self) -> None:
        self._run_action(lambda: session.set_photometric_active(self, False))

    def _reset_photometric_callback(self) -> None:
        self._run_action(lambda: (session.reset_photometric_compensation(self), session.initialize_photometric_compensation(self)))

    def _move_to_training_camera_callback(self) -> None:
        self._run_action(lambda: session.move_main_camera_to_selected_training_frame(self))

    def _reset_camera_callback(self) -> None:
        self._run_action(lambda: session.reset_main_camera(self))

    def _editor_select_box_callback(self) -> None:
        self._run_action(lambda: splat_editor.select_box(self))

    def _editor_select_range_callback(self, kind: str) -> None:
        self._run_action(lambda: splat_editor.select_range(self, kind))

    def _editor_invert_selection_callback(self) -> None:
        self._run_action(lambda: splat_editor.invert_selection(self))

    def _editor_clear_selection_callback(self) -> None:
        self._run_action(lambda: splat_editor.clear_selection(self))

    def _editor_reset_box_callback(self) -> None:
        self._run_action(lambda: splat_editor.init_box_to_scene(self, force=True))

    def _editor_resample_callback(self) -> None:
        self._run_action(lambda: splat_editor.apply_resample(self))

    def _editor_edit_properties_callback(self) -> None:
        self._run_action(lambda: splat_editor.apply_edit_properties(self))

    def _capture_python_frame_callback(self) -> None:
        self._run_action(
            lambda: (
                setattr(self.s, "pending_renderdoc_frame_capture", False),
                setattr(self.s, "pending_python_frame_capture", True),
            )
        )

    def _capture_renderdoc_frame_callback(self) -> None:
        self._run_action(
            lambda: (
                setattr(self.s, "pending_python_frame_capture", False),
                setattr(self.s, "pending_renderdoc_frame_capture", True),
            )
        )

    def _save_defaults_callback(self) -> None:
        try:
            ui_values = _viewer_ui_values(self)
            defaults = load_defaults()
            training_build_args = defaults.setdefault("training_build_args", {})
            viewer_defaults = defaults.setdefault("viewer", {})
            defaults["training_build_args"] = {
                **training_build_args,
                **{
                    build_arg: ui_values[control_key]
                    for build_arg, control_key in _TRAINING_PARAM_KEYS.items()
                    if control_key in ui_values
                },
            }
            exported = export_repo_defaults_from_ui_values(ui_values)
            defaults["renderer"] = exported.get("renderer", {})
            viewer_export = exported.get("viewer", {})
            viewer_defaults["controls"] = viewer_export.get("controls", {})
            viewer_defaults["import"] = viewer_export.get("import", {})
            viewer_defaults["ui"] = viewer_export.get("ui", {})
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

    def _set_graphics_api_callback(self, value: str) -> None:
        try:
            graphics_api = device_type_name(device_type_from_name(value))
            defaults = load_defaults()
            viewer_defaults = defaults.setdefault("viewer", {})
            ui_defaults = viewer_defaults.setdefault("ui", {})
            ui_defaults["graphics_api"] = graphics_api
            write_defaults(defaults)
            if hasattr(self, "ui") and hasattr(self.ui, "_values"):
                self.ui._values["graphics_api"] = graphics_api
        except Exception as exc:
            if hasattr(self, "t"):
                self.t("defaults_status").text = ""
            if hasattr(self, "s"):
                self.s.last_error = str(exc)
            return
        active_api = _preferred_graphics_api_name({"viewer": {"ui": {"graphics_api": getattr(getattr(getattr(self, "device", None), "info", None), "api_name", graphics_api)}}})
        status_suffix = " (restart required)" if graphics_api != active_api else ""
        if hasattr(self, "t"):
            self.t("defaults_status").text = f"Preferred graphics API: {_graphics_api_label(graphics_api)}{status_suffix}"
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
            traceback.print_exc()

    def update_camera(self, dt: float) -> None:
        _sync_camera_mode(self)
        self.s.move_speed, self.s.fov_y = float(self.c("move_speed").value), float(self.c("fov").value)
        scroll_active = abs(self.s.scroll_delta) > 1e-5
        if scroll_active:
            self.s.move_speed = max(self.s.move_speed * (_SCROLL_SPEED_BASE ** self.s.scroll_delta), 0.0)
            self.c("move_speed").value, self.s.scroll_delta = self.s.move_speed, 0.0
        mouse_delta = spy.float2(float(self.s.mouse_delta.x), float(self.s.mouse_delta.y))
        gizmo_busy = bool(getattr(getattr(self, "ui", None), "_values", {}).get("_splat_editor_gizmo_capturing", False))
        # Assign each drag to exactly one consumer (camera vs gizmo/UI) on the first frame the
        # mouse actually moves, then hold it until release. Deferring to first motion lets the
        # gizmo register its grab from the press, so a gizmo drag never also moves the camera,
        # and a camera drag begun in empty space is not stolen when it sweeps over the gizmo.
        any_mouse_down = bool(self.s.mouse_left) or bool(self.s.mouse_right)
        mouse_moved = abs(float(mouse_delta.x)) > 0.0 or abs(float(mouse_delta.y)) > 0.0
        if not any_mouse_down:
            self.s.drag_owner = ""
        elif self.s.drag_owner == "" and mouse_moved:
            self.s.drag_owner = "gizmo" if gizmo_busy else "camera"
        camera_owns_drag = self.s.drag_owner == "camera"
        mouse_left = bool(self.s.mouse_left) and camera_owns_drag
        mouse_right = bool(self.s.mouse_right) and camera_owns_drag
        target_rot = mouse_delta * self.s.look_speed if mouse_left else spy.float2(0.0, 0.0)
        self.s.rot_vel += (target_rot - self.s.rot_vel) * min(1.0, _LOOK_SMOOTH * dt)
        self.s.mouse_delta = spy.float2(0.0, 0.0)
        if float(smath.length(self.s.rot_vel)) > _VIEW_VEC_EPS:
            if bool(getattr(self.s, "camera_free_mode", False)):
                # Free rotation: yaw/pitch spin around the camera's own axes (no world-up
                # lock, no pitch clamp); their composition rolls naturally, flight-style.
                delta = _quat_multiply(
                    _quat_from_axis_angle((0.0, 1.0, 0.0), float(self.s.rot_vel.x)),
                    _quat_from_axis_angle((1.0, 0.0, 0.0), -float(self.s.rot_vel.y)),
                )
                self.s.camera_rot = _quat_normalize(_quat_multiply(_camera_state_rotation(self.s), delta))
            else:
                self.s.yaw += float(self.s.rot_vel.x)
                self.s.pitch = min(max(self.s.pitch + float(self.s.rot_vel.y), -_PITCH_LIMIT), _PITCH_LIMIT)
        right, up, forward = _camera_basis_from_state(self.s)
        move = spy.float3(float(self.s.keys.get(spy.KeyCode.e, False)) - float(self.s.keys.get(spy.KeyCode.q, False)), float(self.s.keys.get(spy.KeyCode.d, False)) - float(self.s.keys.get(spy.KeyCode.a, False)), float(self.s.keys.get(spy.KeyCode.w, False)) - float(self.s.keys.get(spy.KeyCode.s, False)))
        move_length = float(smath.length(move))
        if scroll_active or self.s.mouse_left or self.s.mouse_right or move_length > _VIEW_VEC_EPS:
            _mark_recent_interaction(self, float(getattr(self.s, "last_time", time.perf_counter())))
        target_move = move * (self.s.move_speed / max(move_length, _VIEW_VEC_EPS)) if move_length > _VIEW_VEC_EPS else spy.float3(0.0, 0.0, 0.0)
        if mouse_right:
            drag_speed = self.s.move_speed * self.s.look_speed / max(float(dt), _VIEW_VEC_EPS)
            target_move += spy.float3(-float(mouse_delta.y) * drag_speed, -float(mouse_delta.x) * drag_speed, 0.0)
        self.s.move_vel += (target_move - self.s.move_vel) * min(1.0, _MOVE_SMOOTH * dt)
        self.s.camera_pos += (up * self.s.move_vel.x + right * self.s.move_vel.y + forward * self.s.move_vel.z) * dt

    def shutdown(self) -> None:
        project.join_pending_writes(self)
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


def _build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Slang Splat viewer.", add_help=True)
    parser.add_argument("--headless", action="store_true", help="Run the viewer automation pipeline without creating a window.")
    parser.add_argument("--config", type=Path, default=None, help="JSON config overlay. Required with --headless; in GUI mode this is reserved for config preselection.")
    parser.add_argument("--graphics-api", type=str, default=None, choices=("vulkan", "dx12"), help="Override the graphics API selected from config/defaults.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args, _unknown = _build_main_parser().parse_known_args(argv)
    if bool(args.headless):
        if args.config is None:
            print("--headless requires --config")
            return 2
        from .headless import run_headless_from_config

        return run_headless_from_config(args.config, graphics_api=args.graphics_api)
    view_w, view_h = _compute_view_geometry()
    graphics_api = _preferred_graphics_api_name() if args.graphics_api is None else str(args.graphics_api)
    try:
        frame_capture.prepare_renderdoc_startup()
    except Exception as exc:
        print(f"RenderDoc startup prepare failed: {exc}")
    device = create_default_device(device_type=device_type_from_name(graphics_api), enable_debug_layers=False)
    app = spy.App(device=device)
    viewer = SplatViewer(app, width=view_w, height=view_h, title=_WINDOW_TITLE, max_prepass_memory_mb=DEFAULT_MAX_PREPASS_MEMORY_MB)
    try:
        viewer.run()
    finally:
        viewer.shutdown()
    return 0
