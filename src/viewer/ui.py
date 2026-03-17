from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache, partial
import importlib
import math
from pathlib import Path
import re
from types import SimpleNamespace
import time

import numpy as np
import slangpy as spy
import slangpy.ui.imgui_bundle as simgui
from imgui_bundle import imgui, implot

from .state import LOSS_DEBUG_OPTIONS

TOOLKIT_WIDTH_FRACTION = 0.15
VIEW_WIDTH_FRACTION = 0.85
LOSS_HISTORY_SIZE = 512
FPS_HISTORY_SIZE = 128
_DOC_MAX_WIDTH = 104
_SHORTCUTS_TEXT = "Controls: LMB drag look | WASDQE move | wheel speed"
_LOSS_DEBUG_ABS_SCALE_DEFAULT = 1.0
_LOSS_DEBUG_ABS_SCALE_MIN = 0.125
_LOSS_DEBUG_ABS_SCALE_MAX = 64.0
_LOSS_DEBUG_ABS_SCALE_KEY = "loss_debug_abs_scale"
_INTERFACE_SCALE_KEY = "interface_scale"
_INTERFACE_SCALE_OPTIONS = (
    ("75%", 0.75),
    ("100%", 1.0),
    ("125%", 1.25),
    ("150%", 1.5),
    ("175%", 1.75),
    ("200%", 2.0),
)
_DEFAULT_INTERFACE_SCALE_INDEX = 3
_BASE_FONT_SIZE_PX = 16.0
_FONT_ATLAS_SIZE_PX = _BASE_FONT_SIZE_PX * _INTERFACE_SCALE_OPTIONS[-1][1]
_COLMAP_INIT_MODE_LABELS = ("COLMAP Pointcloud", "Custom PLY")
_TRAIN_DOWNSCALE_MODE_LABELS = ("Auto",) + tuple(f"{i}x" for i in range(1, 17))
_DEBUG_GRAD_NORM_THRESHOLD_DEFAULT = 2e-4
_DEBUG_COLORBAR_WIDTH = 18.0
_DEBUG_COLORBAR_HEIGHT = 240.0
_DEBUG_COLORBAR_MARGIN = 22.0
_DEBUG_COLORBAR_TICKS = 5
_DEBUG_COLORBAR_STEPS = 96
_DEBUG_COLORBAR_LEFT_PAD = 56.0
_DEBUG_COLORBAR_RIGHT_PAD = 64.0
_DEBUG_COLORBAR_TOP_PAD = 28.0
_DEBUG_COLORBAR_BOTTOM_PAD = 12.0
_HISTOGRAM_BIN_COUNT_DEFAULT = 64
_HISTOGRAM_MIN_LOG10_DEFAULT = -8.0
_HISTOGRAM_MAX_LOG10_DEFAULT = 2.0
_HISTOGRAM_Y_LIMIT_DEFAULT = 1.0
_HISTOGRAM_GROUPS = (
    ("roLocal", (0, 1, 2)),
    ("logLDiag", (3, 4, 5)),
    ("lOffDiag", (6, 7, 8)),
    ("color", (9, 10, 11)),
    ("opacity", (12,)),
)
_CACHED_RASTER_GRAD_ATOMIC_MODE_LABELS = ("Float Atomics", "Fixed Point")


@lru_cache(maxsize=1)
def _shader_debug_constants() -> dict[str, float]:
    source = (Path(__file__).resolve().parents[2] / "shaders" / "utility" / "math" / "constants.slang").read_text(encoding="utf-8")
    names = ("DEBUG_GRAD_NORM_FLOOR", "DEBUG_GRAD_THRESHOLD_MIN_SCALE", "DEBUG_GRAD_THRESHOLD_MAX_SCALE")
    constants: dict[str, float] = {}
    for name in names:
        match = re.search(rf"static\s+const\s+float\s+{name}\s*=\s*([^;]+);", source)
        if match is None:
            raise RuntimeError(f"Missing shader debug constant: {name}")
        constants[name] = float(match.group(1).strip())
    return constants


def _read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _build_about_text() -> str:
    return "\n".join(
        (
            "Slang Splat Viewer",
            "",
            "Single-window Gaussian splat viewer and trainer built on Slangpy.",
            "The scene and imgui_bundle UI are rendered into the same swapchain image.",
            "",
            _SHORTCUTS_TEXT,
        )
    )


def _build_documentation_text() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    parts = [
        "Viewer Documentation",
        "",
        _read_text_if_exists(repo_root / "doc" / "Viewer.md").strip(),
    ]
    text = "\n\n".join(part for part in parts if part)
    return text if text else "Documentation is unavailable."


def _panel_rect(width: int, height: int, menu_bar_height: float) -> tuple[float, float, float, float]:
    panel_width = float(max(int(width * TOOLKIT_WIDTH_FRACTION), 280))
    return 0.0, float(menu_bar_height), panel_width, max(float(height) - float(menu_bar_height), 1.0)


@lru_cache(maxsize=1)
def _default_font_path() -> Path | None:
    package = importlib.import_module("imgui_bundle")
    path = Path(package.__file__).resolve().parent / "assets" / "fonts" / "DroidSans.ttf"
    return path if path.exists() else None


def _menu_item(label: str, shortcut: str = "", selected: bool = False, enabled: bool = True) -> bool:
    return bool(imgui.menu_item(label, shortcut, selected, enabled)[0])


def _saturate(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _jet_colormap(value: float) -> tuple[float, float, float]:
    t = _saturate(value)
    return (
        _saturate(1.5 - abs(4.0 * t - 3.0)),
        _saturate(1.5 - abs(4.0 * t - 2.0)),
        _saturate(1.5 - abs(4.0 * t - 1.0)),
    )


def _color_u32(r: float, g: float, b: float, a: float = 1.0) -> int:
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(float(r), float(g), float(b), float(a)))


def _debug_colorbar_mode(ui: "ViewerUI") -> str | None:
    if bool(ui._values.get("debug_processed_count", False)):
        return "processed_count"
    if bool(ui._values.get("debug_grad_norm", False)):
        return "grad_norm"
    return None


def _processed_count_tick_value(t: float, max_splat_steps: int) -> float:
    return math.pow(2.0, _saturate(t) * math.log2(max(max_splat_steps, 0) + 1.0)) - 1.0


def _grad_norm_tick_value(t: float, threshold: float) -> float:
    constants = _shader_debug_constants()
    grad_norm_floor = constants["DEBUG_GRAD_NORM_FLOOR"]
    grad_threshold = max(float(threshold), grad_norm_floor)
    lo = math.log10(max(grad_threshold * constants["DEBUG_GRAD_THRESHOLD_MIN_SCALE"], grad_norm_floor))
    hi = math.log10(max(grad_threshold * constants["DEBUG_GRAD_THRESHOLD_MAX_SCALE"], grad_norm_floor))
    return math.pow(10.0, lo + _saturate(t) * (hi - lo))


def _histogram_log_range_from_ranges(payload: object) -> tuple[float, float] | None:
    if payload is None:
        return None
    min_values = np.asarray(getattr(payload, "min_values", np.zeros((0,), dtype=np.float32)), dtype=np.float64)
    max_values = np.asarray(getattr(payload, "max_values", np.zeros((0,), dtype=np.float32)), dtype=np.float64)
    if min_values.ndim != 1 or max_values.ndim != 1 or min_values.size != max_values.size or min_values.size == 0:
        return None
    extrema = np.maximum(np.abs(min_values), np.abs(max_values))
    finite_nonzero = extrema[np.isfinite(extrema) & (extrema > 0.0)]
    if finite_nonzero.size == 0:
        return None
    log_values = np.log10(finite_nonzero)
    return float(np.min(log_values)), float(np.max(log_values))


@dataclass(frozen=True, slots=True)
class ControlSpec:
    key: str
    kind: str
    label: str
    kwargs: dict[str, object]


GROUP_SPECS = {
    "View": (
        ControlSpec(_INTERFACE_SCALE_KEY, "combo", "Interface Scale", {"value": _DEFAULT_INTERFACE_SCALE_INDEX, "options": tuple(label for label, _ in _INTERFACE_SCALE_OPTIONS)}),
    ),
    "Main": (
        ControlSpec("loss_debug", "checkbox", "Visual Loss Debug", {"value": False}),
        ControlSpec("loss_debug_view", "slider_int", "Debug View", {"value": 2, "min": 0, "max": len(LOSS_DEBUG_OPTIONS) - 1}),
        ControlSpec("loss_debug_frame", "slider_int", "Debug Frame", {"value": 0, "min": 0, "max": 10000}),
        ControlSpec(_LOSS_DEBUG_ABS_SCALE_KEY, "slider_float", "Abs Diff Scale", {"value": _LOSS_DEBUG_ABS_SCALE_DEFAULT, "min": _LOSS_DEBUG_ABS_SCALE_MIN, "max": _LOSS_DEBUG_ABS_SCALE_MAX, "format": "%.3gx", "logarithmic": True}),
    ),
    "Camera": (
        ControlSpec("move_speed", "slider_float", "Move Speed", {"value": 2.0, "min": 0.1, "max": 20.0}),
        ControlSpec("fov", "slider_float", "FOV", {"value": 60.0, "min": 25.0, "max": 100.0}),
    ),
    "Train Setup": (
        ControlSpec("max_gaussians", "slider_int", "Max Gaussians", {"value": 5900000, "min": 1000, "max": 10000000}),
        ControlSpec("training_steps_per_frame", "slider_int", "Steps / Frame", {"value": 1, "min": 1, "max": 8}),
        ControlSpec("train_downscale_mode", "combo", "Downscale Mode", {"value": 1, "options": _TRAIN_DOWNSCALE_MODE_LABELS}),
        ControlSpec("train_auto_start_downscale", "slider_int", "Auto Start Downscale", {"value": 16, "min": 1, "max": 16}),
        ControlSpec("train_downscale_base_iters", "input_int", "Downscale Base Iters", {"value": 200, "step": 25, "step_fast": 100}),
        ControlSpec("train_downscale_iter_step", "input_int", "Downscale Iter Step", {"value": 50, "step": 10, "step_fast": 50}),
        ControlSpec("train_downscale_max_iters", "input_int", "Downscale Max Iters", {"value": 30000, "step": 1000, "step_fast": 5000}),
        ControlSpec("seed", "slider_int", "Shuffle Seed", {"value": 1234, "min": 0, "max": 1000000}),
        ControlSpec("init_opacity", "input_float", "Init Opacity", {"value": 0.5, "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}),
    ),
    "Train Optimizer": (
        ControlSpec("lr_base", "input_float", "Base LR", {"value": 1e-3, "step": 1e-5, "step_fast": 1e-4, "format": "%.8f"}),
        ControlSpec("lr_pos_mul", "input_float", "LR Mul Position", {"value": 1.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ControlSpec("lr_scale_mul", "input_float", "LR Mul Scale", {"value": 5.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ControlSpec("lr_rot_mul", "input_float", "LR Mul Rotation", {"value": 1.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ControlSpec("lr_color_mul", "input_float", "LR Mul Color", {"value": 1.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ControlSpec("lr_opacity_mul", "input_float", "LR Mul Opacity", {"value": 1.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ControlSpec("beta1", "input_float", "Beta1", {"value": 0.9, "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ControlSpec("beta2", "input_float", "Beta2", {"value": 0.999, "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}),
        ControlSpec("scale_l2", "input_float", "Scale Log Reg", {"value": 0.0, "step": 1e-5, "step_fast": 1e-4, "format": "%.8f"}),
        ControlSpec("scale_abs_reg", "input_float", "Scale Abs Reg", {"value": 0.01, "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}),
        ControlSpec("opacity_reg", "input_float", "Opacity Reg", {"value": 0.01, "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}),
        ControlSpec("max_anisotropy", "input_float", "Max Anisotropy", {"value": 10.0, "step": 0.1, "step_fast": 0.5, "format": "%.6f"}),
        ControlSpec("grad_clip", "input_float", "Grad Clip", {"value": 10.0, "step": 0.1, "step_fast": 1.0, "format": "%.4f"}),
        ControlSpec("grad_norm_clip", "input_float", "Grad Norm Clip", {"value": 10.0, "step": 0.1, "step_fast": 1.0, "format": "%.4f"}),
        ControlSpec("max_update", "input_float", "Max Update", {"value": 0.05, "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}),
    ),
    "Train Stability": (
        ControlSpec("min_scale", "input_float", "Min Scale", {"value": 1e-3, "step": 1e-5, "step_fast": 1e-4, "format": "%.8f"}),
        ControlSpec("max_scale", "input_float", "Max Scale", {"value": 3.0, "step": 1e-2, "step_fast": 0.1, "format": "%.5f"}),
        ControlSpec("min_opacity", "input_float", "Min Opacity", {"value": 1e-4, "step": 1e-5, "step_fast": 1e-4, "format": "%.8f"}),
        ControlSpec("max_opacity", "input_float", "Max Opacity", {"value": 0.9999, "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}),
        ControlSpec("position_abs_max", "input_float", "Pos Abs Max", {"value": 1e4, "step": 10.0, "step_fast": 100.0, "format": "%.3f"}),
        ControlSpec("train_near", "input_float", "Train Near", {"value": 0.1, "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ControlSpec("train_far", "input_float", "Train Far", {"value": 120.0, "step": 1.0, "step_fast": 10.0, "format": "%.3f"}),
    ),
}


def default_control_values(*group_names: str) -> dict[str, object]:
    groups = GROUP_SPECS.values() if not group_names else (GROUP_SPECS[name] for name in group_names)
    return {spec.key: spec.kwargs["value"] for specs in groups for spec in specs if "value" in spec.kwargs}


RENDER_PARAM_SPECS = (
    ControlSpec("radius_scale", "slider_float", "Radius Scale", {"value": 1.0, "min": 0.25, "max": 4.0, "format": "%.3g"}),
    ControlSpec("alpha_cutoff", "slider_float", "Alpha Cutoff", {"value": 0.0039, "min": 0.0001, "max": 0.1, "format": "%.2e"}),
    ControlSpec("max_splat_steps", "slider_int", "Max Splat Steps", {"value": 32768, "min": 16, "max": 32768}),
    ControlSpec("trans_threshold", "slider_float", "Trans Threshold", {"value": 0.005, "min": 0.001, "max": 0.2, "format": "%.2e"}),
    ControlSpec("sampled5_safety", "slider_float", "MVEE Safety", {"value": 1.0, "min": 1.0, "max": 1.2}),
    ControlSpec("cached_raster_grad_atomic_mode", "combo", "Cached Grad Atomics", {"value": 1, "options": _CACHED_RASTER_GRAD_ATOMIC_MODE_LABELS}),
    ControlSpec("cached_raster_grad_fixed_ro_local_range", "slider_float", "Cached Grad Pos Range", {"value": 10.0, "min": 0.25, "max": 1024.0, "format": "%.4g", "logarithmic": True}),
    ControlSpec("cached_raster_grad_fixed_log_l_diag_range", "slider_float", "Cached Grad Scale Range", {"value": 10.0, "min": 0.25, "max": 1024.0, "format": "%.4g", "logarithmic": True}),
    ControlSpec("cached_raster_grad_fixed_l_offdiag_range", "slider_float", "Cached Grad Rot Range", {"value": 10.0, "min": 0.25, "max": 1024.0, "format": "%.4g", "logarithmic": True}),
    ControlSpec("cached_raster_grad_fixed_color_range", "slider_float", "Cached Grad Color Range", {"value": 200.0, "min": 0.25, "max": 2048.0, "format": "%.4g", "logarithmic": True}),
    ControlSpec("cached_raster_grad_fixed_opacity_range", "slider_float", "Cached Grad Opacity Range", {"value": 200.0, "min": 0.25, "max": 2048.0, "format": "%.4g", "logarithmic": True}),
    ControlSpec("debug_ellipse", "checkbox", "Debug Ellipse Outlines", {"value": False}),
    ControlSpec("debug_processed_count", "checkbox", "Debug Processed Count", {"value": False}),
    ControlSpec("debug_grad_norm", "checkbox", "Debug Grad Norm", {"value": False}),
)

_ALL_DEFAULTS = {spec.key: spec.kwargs["value"] for group in GROUP_SPECS.values() for spec in group if "value" in spec.kwargs}
_ALL_DEFAULTS.update({spec.key: spec.kwargs["value"] for spec in RENDER_PARAM_SPECS if "value" in spec.kwargs})


class _ControlProxy:
    """Minimal proxy giving .value attribute for backward compatibility."""
    __slots__ = ("_store", "_key")

    def __init__(self, store: dict, key: str) -> None:
        self._store = store
        self._key = key

    @property
    def value(self):
        return self._store[self._key]

    @value.setter
    def value(self, v):
        self._store[self._key] = v


class _TextProxy:
    """Minimal proxy giving .text attribute for backward compatibility."""
    __slots__ = ("_store", "_key")

    def __init__(self, store: dict, key: str) -> None:
        self._store = store
        self._key = key

    @property
    def text(self):
        return self._store.get(self._key, "")

    @text.setter
    def text(self, v):
        self._store[self._key] = v


@dataclass(slots=True)
class ViewerUI:
    """Backward-compatible wrapper over plain dicts — provides .controls[key].value and .texts[key].text."""
    _values: dict[str, object] = field(default_factory=dict)
    _texts: dict[str, str] = field(default_factory=dict)

    @property
    def controls(self):
        return {k: _ControlProxy(self._values, k) for k in self._values}

    @property
    def texts(self):
        return {k: _TextProxy(self._texts, k) for k in self._texts}

    def control(self, key: str) -> _ControlProxy:
        return _ControlProxy(self._values, key)

    def text(self, key: str) -> _TextProxy:
        return _TextProxy(self._texts, key)


@dataclass(slots=True)
class ToolkitState:
    loss_history: deque = field(default_factory=partial(deque, maxlen=LOSS_HISTORY_SIZE))
    fps_history: deque = field(default_factory=partial(deque, maxlen=FPS_HISTORY_SIZE))
    psnr_history: deque = field(default_factory=partial(deque, maxlen=LOSS_HISTORY_SIZE))
    step_history: deque = field(default_factory=partial(deque, maxlen=LOSS_HISTORY_SIZE))
    step_time_history: deque = field(default_factory=partial(deque, maxlen=LOSS_HISTORY_SIZE))


def _noop() -> None:
    return None


class ToolkitWindow:
    """Dear ImGui overlay rendered into the active Slangpy AppWindow surface."""

    @staticmethod
    def _iters_per_second(step_history: deque, step_time_history: deque) -> float:
        if len(step_history) < 2 or len(step_time_history) < 2:
            return 0.0
        sample_count = min(len(step_history), len(step_time_history), 16)
        steps = np.array(list(step_history)[-sample_count:], dtype=np.float64)
        times = np.array(list(step_time_history)[-sample_count:], dtype=np.float64)
        dt = float(times[-1] - times[0])
        if dt <= 1e-6:
            return 0.0
        return float(max(steps[-1] - steps[0], 0.0) / dt)

    def __init__(self, device: spy.Device, width: int, height: int):
        self.device = device
        self.ctx = simgui.create_imgui_context(width, height)
        imgui.set_current_context(self.ctx)
        implot.create_context()
        imgui.get_io().config_flags |= imgui.ConfigFlags_.docking_enable.value
        self.renderer = spy.ui.Context(device)
        self._configure_default_font()
        self._apply_theme()
        self.callbacks = SimpleNamespace(
            load_ply=_noop,
            browse_colmap_root=_noop,
            browse_colmap_images=_noop,
            browse_colmap_ply=_noop,
            import_colmap=_noop,
            reload=_noop,
            reinitialize=_noop,
            start_training=_noop,
            stop_training=_noop,
        )
        self.tk = ToolkitState()
        self._alive = True
        self._frame_textures: list[spy.Texture] = []
        self._last_frame_time = time.perf_counter()
        self._show_about = False
        self._show_docs = False
        self._show_colmap_import = False
        self._about_text = _build_about_text()
        self._documentation_text = _build_documentation_text()
        self._menu_bar_height = 0.0
        self._applied_interface_scale = -1.0
        self._dockspace_id = 0
        self._toolkit_window_open = True
        self._toolkit_rect = (
            0.0,
            0.0,
            float(max(int(width * TOOLKIT_WIDTH_FRACTION), 280)),
            max(float(height), 1.0),
        )
        self._set_interface_scale(_INTERFACE_SCALE_OPTIONS[_DEFAULT_INTERFACE_SCALE_INDEX][1])

    def _set_current_context(self) -> None:
        imgui.set_current_context(self.ctx)

    def _apply_theme(self):
        self._set_current_context()
        imgui.style_colors_dark()
        style = imgui.get_style()
        style.window_rounding = 4.0
        style.frame_rounding = 3.0
        style.grab_rounding = 2.0
        style.scrollbar_rounding = 4.0
        style.tab_rounding = 3.0
        style.popup_rounding = 3.0
        style.child_rounding = 3.0
        style.window_border_size = 1.0
        style.frame_border_size = 0.0
        style.item_spacing = imgui.ImVec2(8, 5)
        style.item_inner_spacing = imgui.ImVec2(6, 4)
        style.frame_padding = imgui.ImVec2(6, 4)
        style.indent_spacing = 18.0
        style.scrollbar_size = 12.0
        _c = style.set_color_
        _c(imgui.Col_.title_bg_active, imgui.ImVec4(0.18, 0.35, 0.58, 1.0))
        _c(imgui.Col_.header, imgui.ImVec4(0.22, 0.40, 0.62, 0.55))
        _c(imgui.Col_.header_hovered, imgui.ImVec4(0.26, 0.50, 0.74, 0.80))
        _c(imgui.Col_.header_active, imgui.ImVec4(0.26, 0.50, 0.74, 1.00))
        _c(imgui.Col_.button, imgui.ImVec4(0.20, 0.40, 0.60, 0.62))
        _c(imgui.Col_.button_hovered, imgui.ImVec4(0.26, 0.50, 0.74, 0.80))
        _c(imgui.Col_.button_active, imgui.ImVec4(0.20, 0.40, 0.60, 1.00))
        _c(imgui.Col_.tab, imgui.ImVec4(0.14, 0.28, 0.45, 0.86))
        _c(imgui.Col_.tab_hovered, imgui.ImVec4(0.26, 0.50, 0.74, 0.80))
        _c(imgui.Col_.tab_selected, imgui.ImVec4(0.20, 0.42, 0.68, 1.00))
        _c(imgui.Col_.separator, imgui.ImVec4(0.35, 0.35, 0.45, 0.50))
        _c(imgui.Col_.plot_lines, imgui.ImVec4(0.40, 0.75, 1.00, 1.00))
        _c(imgui.Col_.frame_bg, imgui.ImVec4(0.12, 0.16, 0.22, 0.90))
        _c(imgui.Col_.frame_bg_hovered, imgui.ImVec4(0.18, 0.28, 0.42, 1.00))
        _c(imgui.Col_.frame_bg_active, imgui.ImVec4(0.20, 0.34, 0.52, 1.00))
        _c(imgui.Col_.slider_grab, imgui.ImVec4(0.30, 0.55, 0.82, 0.78))
        _c(imgui.Col_.slider_grab_active, imgui.ImVec4(0.36, 0.62, 0.90, 1.00))
        _c(imgui.Col_.check_mark, imgui.ImVec4(0.40, 0.75, 1.00, 1.00))
        _c(imgui.Col_.scrollbar_grab, imgui.ImVec4(0.24, 0.36, 0.52, 0.60))
        _c(imgui.Col_.scrollbar_grab_hovered, imgui.ImVec4(0.30, 0.46, 0.64, 0.80))
        _c(imgui.Col_.scrollbar_grab_active, imgui.ImVec4(0.30, 0.46, 0.64, 1.00))
        _c(imgui.Col_.separator_hovered, imgui.ImVec4(0.30, 0.55, 0.82, 0.78))
        _c(imgui.Col_.resize_grip, imgui.ImVec4(0.26, 0.50, 0.74, 0.25))
        _c(imgui.Col_.resize_grip_hovered, imgui.ImVec4(0.26, 0.50, 0.74, 0.67))
        _c(imgui.Col_.resize_grip_active, imgui.ImVec4(0.26, 0.50, 0.74, 0.95))

    def _configure_default_font(self) -> None:
        io = imgui.get_io()
        atlas = io.fonts
        atlas.clear()
        font_path = _default_font_path()
        io.font_default = atlas.add_font_from_file_ttf(str(font_path), _FONT_ATLAS_SIZE_PX) if font_path is not None else atlas.add_font_default()
        if atlas.tex_data is not None:
            atlas.tex_data.get_pixels_array()

    def _interface_scale_factor(self, ui: ViewerUI) -> float:
        idx = min(max(int(ui._values.get(_INTERFACE_SCALE_KEY, _DEFAULT_INTERFACE_SCALE_INDEX)), 0), len(_INTERFACE_SCALE_OPTIONS) - 1)
        return float(_INTERFACE_SCALE_OPTIONS[idx][1])

    def _set_interface_scale(self, scale: float) -> None:
        clamped_scale = max(float(scale), 0.5)
        if abs(clamped_scale - self._applied_interface_scale) <= 1e-6:
            return
        self._set_current_context()
        self._apply_theme()
        style = imgui.get_style()
        style.scale_all_sizes(clamped_scale)
        style.font_scale_main = clamped_scale * (_BASE_FONT_SIZE_PX / _FONT_ATLAS_SIZE_PX)
        self._applied_interface_scale = clamped_scale

    def _sync_interface_scale(self, ui: ViewerUI) -> None:
        self._set_interface_scale(self._interface_scale_factor(ui))

    @property
    def alive(self) -> bool:
        return self._alive

    def handle_keyboard_event(self, event) -> bool:
        if not self._alive:
            return False
        self._set_current_context()
        return bool(simgui.handle_keyboard_event(event))

    def handle_mouse_event(self, event) -> bool:
        if not self._alive:
            return False
        self._set_current_context()
        return bool(simgui.handle_mouse_event(event))

    def render(self, ui: ViewerUI, surface_texture: spy.Texture, command_encoder: spy.CommandEncoder) -> None:
        if not self._alive:
            return
        width = int(surface_texture.width)
        height = int(surface_texture.height)
        now = time.perf_counter()
        dt = max(now - self._last_frame_time, 1e-5)
        self._last_frame_time = now
        self._set_current_context()
        simgui.begin_frame(width, height, dt)
        self._sync_interface_scale(ui)
        self._menu_bar_height = self._draw_main_menu_bar(ui)
        self._draw_dockspace()
        self._draw_panel(ui, width, height)
        self._draw_debug_colorbar(ui, width, height)
        self._draw_histogram_window(ui)
        imgui.render()
        draw_data = imgui.get_draw_data()
        self._frame_textures = simgui.sync_draw_data_textures(self.device, self.renderer, draw_data)
        simgui.render_imgui_draw_data(self.renderer, draw_data, surface_texture, command_encoder)

    def _draw_dockspace(self) -> None:
        viewport = imgui.get_main_viewport()
        self._dockspace_id = int(
            imgui.dock_space_over_viewport(
                viewport=viewport,
                flags=imgui.DockNodeFlags_.passthru_central_node.value,
            )
        )

    def _draw_panel(self, ui: ViewerUI, width: int, height: int) -> None:
        panel_x, panel_y, panel_width, panel_height = _panel_rect(width, height, self._menu_bar_height)
        imgui.set_next_window_pos(imgui.ImVec2(panel_x, panel_y), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(panel_width, panel_height), imgui.Cond_.first_use_ever.value)
        if self._dockspace_id != 0:
            imgui.set_next_window_dock_id(self._dockspace_id, imgui.Cond_.first_use_ever.value)
        flags = imgui.WindowFlags_.no_collapse.value
        opened, self._toolkit_window_open = imgui.begin("Toolkit", self._toolkit_window_open, flags=flags)
        if opened:
            pos = imgui.get_window_pos()
            size = imgui.get_window_size()
            self._toolkit_rect = (float(pos.x), float(pos.y), float(size.x), float(size.y))
            self._section_performance(ui)
            self._section_status(ui)
            self._section_scene_io(ui)
            self._section_camera(ui)
            self._section_training_control(ui)
            self._section_training_setup(ui)
            self._section_optimizer(ui)
            self._section_stability(ui)
            self._section_render_params(ui)
        imgui.end()
        self._draw_about_window()
        self._draw_documentation_window()
        self._draw_colmap_import_window(ui)

    def _draw_debug_colorbar(self, ui: ViewerUI, width: int, height: int) -> None:
        mode = _debug_colorbar_mode(ui)
        if mode is None:
            return
        draw_list = imgui.get_foreground_draw_list()
        bar_height = min(_DEBUG_COLORBAR_HEIGHT, max(float(height) - self._menu_bar_height - 2.0 * _DEBUG_COLORBAR_MARGIN, 80.0))
        panel_x, _, panel_width, _ = self._toolkit_rect
        box_width = _DEBUG_COLORBAR_LEFT_PAD + _DEBUG_COLORBAR_WIDTH + _DEBUG_COLORBAR_RIGHT_PAD
        box_height = _DEBUG_COLORBAR_TOP_PAD + bar_height + _DEBUG_COLORBAR_BOTTOM_PAD
        box_min_x = panel_x + panel_width + 8.0
        box_max_x = float(width) - _DEBUG_COLORBAR_MARGIN - box_width
        box_x = max(min(float(width) - _DEBUG_COLORBAR_MARGIN - box_width, box_max_x), box_min_x)
        if box_x + box_width > float(width) - 4.0:
            box_x = max(float(width) - 4.0 - box_width, 0.0)
        box_y = min(max(self._menu_bar_height + _DEBUG_COLORBAR_MARGIN, 0.0), max(float(height) - box_height - 4.0, self._menu_bar_height))
        x0 = box_x + _DEBUG_COLORBAR_LEFT_PAD
        y0 = box_y + _DEBUG_COLORBAR_TOP_PAD
        x1 = x0 + _DEBUG_COLORBAR_WIDTH
        y1 = y0 + bar_height
        draw_list.add_rect_filled(
            imgui.ImVec2(box_x, box_y),
            imgui.ImVec2(box_x + box_width, box_y + box_height),
            _color_u32(0.05, 0.06, 0.08, 0.58),
            8.0,
        )
        draw_list.add_text(imgui.ImVec2(box_x, box_y + 6.0), _color_u32(0.85, 0.88, 0.92), self._debug_colorbar_title(mode))
        self._draw_debug_colorbar_gradient(draw_list, x0, y0, x1, y1)
        draw_list.add_rect(imgui.ImVec2(x0, y0), imgui.ImVec2(x1, y1), _color_u32(0.95, 0.97, 1.0, 0.95), 2.0, 0, 1.0)
        self._draw_debug_colorbar_ticks(draw_list, mode, x0, y0, x1, y1, ui)

    def _draw_debug_colorbar_gradient(self, draw_list: object, x0: float, y0: float, x1: float, y1: float) -> None:
        height = max(y1 - y0, 1.0)
        for idx in range(_DEBUG_COLORBAR_STEPS):
            t0 = idx / _DEBUG_COLORBAR_STEPS
            t1 = (idx + 1) / _DEBUG_COLORBAR_STEPS
            rgb = _jet_colormap(1.0 - 0.5 * (t0 + t1))
            draw_list.add_rect_filled(
                imgui.ImVec2(x0, y0 + t0 * height),
                imgui.ImVec2(x1, y0 + t1 * height),
                _color_u32(*rgb),
            )

    def _draw_debug_colorbar_ticks(self, draw_list: object, mode: str, x0: float, y0: float, x1: float, y1: float, ui: ViewerUI) -> None:
        for idx in range(_DEBUG_COLORBAR_TICKS):
            t = 1.0 - idx / max(_DEBUG_COLORBAR_TICKS - 1, 1)
            y = y0 + (1.0 - t) * (y1 - y0)
            draw_list.add_line(imgui.ImVec2(x1 + 2.0, y), imgui.ImVec2(x1 + 8.0, y), _color_u32(0.85, 0.88, 0.92, 0.9), 1.0)
            draw_list.add_text(imgui.ImVec2(x1 + 12.0, y - 6.0), _color_u32(0.85, 0.88, 0.92, 0.95), self._debug_colorbar_tick_label(mode, t, ui))

    def _debug_colorbar_title(self, mode: str) -> str:
        return "Processed Count" if mode == "processed_count" else "Grad Norm"

    def _debug_colorbar_tick_label(self, mode: str, t: float, ui: ViewerUI) -> str:
        if mode == "processed_count":
            max_splat_steps = int(ui._values.get("max_splat_steps", 32768))
            value = _processed_count_tick_value(t, max_splat_steps)
            return f"{int(round(value)):,}"
        threshold = float(ui._values.get("debug_grad_norm_threshold", _DEBUG_GRAD_NORM_THRESHOLD_DEFAULT))
        return f"{_grad_norm_tick_value(t, threshold):.1e}"

    def _draw_main_menu_bar(self, ui: ViewerUI) -> float:
        if not imgui.begin_main_menu_bar():
            return 0.0
        if imgui.begin_menu("File"):
            if _menu_item("Load PLY..."):
                self.callbacks.load_ply()
            if _menu_item("Load COLMAP..."):
                self._show_colmap_import = True
            if _menu_item("Reload"):
                self.callbacks.reload()
            imgui.separator()
            if _menu_item("Reinitialize Gaussians"):
                self.callbacks.reinitialize()
            imgui.end_menu()
        if imgui.begin_menu("View"):
            active_idx = min(max(int(ui._values.get(_INTERFACE_SCALE_KEY, _DEFAULT_INTERFACE_SCALE_INDEX)), 0), len(_INTERFACE_SCALE_OPTIONS) - 1)
            for idx, (label, _) in enumerate(_INTERFACE_SCALE_OPTIONS):
                if _menu_item(label, selected=idx == active_idx):
                    ui._values[_INTERFACE_SCALE_KEY] = idx
            imgui.separator()
            if _menu_item("Reset Interface Scale"):
                ui._values[_INTERFACE_SCALE_KEY] = _DEFAULT_INTERFACE_SCALE_INDEX
            imgui.end_menu()
        if imgui.begin_menu("Debug"):
            selected = bool(ui._values.get("show_histograms", False))
            if _menu_item("Histograms", selected=selected):
                ui._values["show_histograms"] = not selected
            imgui.end_menu()
        if imgui.begin_menu("Help"):
            if _menu_item("Documentation"):
                self._show_docs = True
            if _menu_item("About"):
                self._show_about = True
            imgui.end_menu()
        menu_bar_height = float(imgui.get_window_height())
        imgui.end_main_menu_bar()
        return menu_bar_height

    def _draw_about_window(self) -> None:
        if not self._show_about:
            return
        imgui.set_next_window_pos(imgui.ImVec2(24.0, self._menu_bar_height + 24.0), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(420.0, 220.0), imgui.Cond_.first_use_ever.value)
        opened, self._show_about = imgui.begin("About", True)
        if opened:
            imgui.text_wrapped(self._about_text)
        imgui.end()

    def _draw_documentation_window(self) -> None:
        if not self._show_docs:
            return
        imgui.set_next_window_pos(imgui.ImVec2(40.0, self._menu_bar_height + 32.0), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(760.0, 620.0), imgui.Cond_.first_use_ever.value)
        opened, self._show_docs = imgui.begin("Documentation", True)
        if opened:
            imgui.text_disabled("Local viewer documentation")
            imgui.separator()
            if imgui.begin_child("##docs_scroll", imgui.ImVec2(0.0, 0.0), imgui.ChildFlags_.borders.value):
                imgui.push_text_wrap_pos(_DOC_MAX_WIDTH * imgui.get_font_size() * 0.5)
                imgui.text_unformatted(self._documentation_text)
                imgui.pop_text_wrap_pos()
                imgui.end_child()
        imgui.end()

    def close_colmap_import_window(self) -> None:
        self._show_colmap_import = False

    @staticmethod
    def _path_text(ui: ViewerUI, key: str, empty_text: str = "<not selected>") -> str:
        value = str(ui._values.get(key, "")).strip()
        return value if value else empty_text

    def _draw_import_path_selector(self, ui: ViewerUI, *, label: str, key: str, button_label: str, callback) -> None:
        imgui.text_disabled(label)
        imgui.text_wrapped(self._path_text(ui, key))
        if imgui.button(button_label):
            callback()

    def _draw_colmap_import_window(self, ui: ViewerUI) -> None:
        if not self._show_colmap_import:
            return
        imgui.set_next_window_pos(imgui.ImVec2(56.0, self._menu_bar_height + 40.0), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(540.0, 0.0), imgui.Cond_.first_use_ever.value)
        opened, self._show_colmap_import = imgui.begin("COLMAP Import", True)
        import_active = bool(ui._values.get("_colmap_import_active", False))
        if import_active and not self._show_colmap_import:
            self._show_colmap_import = True
        if opened:
            imgui.text_wrapped("Select the dataset root, verify the auto-detected image folder, choose the gaussian initialization source, then import the dataset.")
            imgui.separator()
            if import_active:
                status = ui._texts.get("colmap_import_status", "")
                current_name = ui._texts.get("colmap_import_current", "")
                progress = max(0.0, min(float(ui._values.get("_colmap_import_fraction", 0.0)), 1.0))
                if status:
                    imgui.text_wrapped(status)
                imgui.progress_bar(progress, imgui.ImVec2(-1.0, 0.0))
                if current_name:
                    imgui.text_disabled(current_name)
                imgui.spacing()
            imgui.begin_disabled(import_active)
            self._draw_import_path_selector(ui, label="COLMAP Root", key="colmap_root_path", button_label="Browse Root...", callback=self.callbacks.browse_colmap_root)
            imgui.spacing()
            self._draw_import_path_selector(ui, label="Image Folder", key="colmap_images_root", button_label="Browse Image Folder...", callback=self.callbacks.browse_colmap_images)
            imgui.spacing()
            mode_idx = max(0, min(int(ui._values.get("colmap_init_mode", 0)), len(_COLMAP_INIT_MODE_LABELS) - 1))
            if imgui.begin_combo("Initialization", _COLMAP_INIT_MODE_LABELS[mode_idx]):
                for idx, label in enumerate(_COLMAP_INIT_MODE_LABELS):
                    selected = idx == mode_idx
                    if imgui.selectable(label, selected)[0]:
                        ui._values["colmap_init_mode"] = idx
                        mode_idx = idx
                    if selected:
                        imgui.set_item_default_focus()
                imgui.end_combo()
            if mode_idx == 0:
                changed, value = imgui.drag_float(
                    "NN Radius Scale Coef",
                    float(ui._values.get("colmap_nn_radius_scale_coef", 0.25)),
                    0.01,
                    1e-4,
                    16.0,
                    "%.4f",
                    imgui.SliderFlags_.logarithmic.value,
                )
                if changed:
                    ui._values["colmap_nn_radius_scale_coef"] = max(float(value), 1e-4)
                if imgui.is_item_hovered():
                    imgui.set_item_tooltip("Multiplier applied to the median COLMAP nearest-neighbor radius when initializing gaussian scales.")
            else:
                imgui.spacing()
                self._draw_import_path_selector(ui, label="Custom PLY", key="colmap_custom_ply_path", button_label="Browse PLY...", callback=self.callbacks.browse_colmap_ply)
            imgui.spacing()
            if imgui.button("Importing..." if import_active else "Import", imgui.ImVec2(imgui.get_content_region_avail().x, 0.0)):
                self.callbacks.import_colmap()
            imgui.end_disabled()
        imgui.end()

    def _draw_histogram_window(self, ui: ViewerUI) -> None:
        if not bool(ui._values.get("show_histograms", False)):
            return
        imgui.set_next_window_pos(imgui.ImVec2(72.0, self._menu_bar_height + 56.0), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(980.0, 720.0), imgui.Cond_.first_use_ever.value)
        opened, show = imgui.begin("Histograms", True)
        ui._values["show_histograms"] = bool(show)
        if opened:
            self._draw_histogram_controls(ui)
            status = str(ui._texts.get("histogram_status", "")).strip()
            payload = ui._values.get("_histogram_payload")
            range_payload = ui._values.get("_histogram_range_payload")
            self._update_histogram_log_range(ui, range_payload)
            if status:
                imgui.text_disabled(status)
                imgui.separator()
            if payload is None or getattr(payload, "counts", np.zeros((0, 0), dtype=np.int64)).size == 0 or int(np.sum(payload.counts)) == 0:
                imgui.text_wrapped("No cached ellipse gradient histogram data is available yet.")
            else:
                self._update_histogram_y_limit(ui, payload)
                self._draw_histogram_groups(ui, payload)
                imgui.separator()
                self._draw_histogram_range_debug(range_payload)
        imgui.end()

    def _draw_histogram_controls(self, ui: ViewerUI) -> None:
        changed, value = imgui.checkbox("Auto Refresh", bool(ui._values.get("hist_auto_refresh", True)))
        if changed:
            ui._values["hist_auto_refresh"] = bool(value)
        imgui.same_line()
        if imgui.button("Refresh"):
            ui._values["_histograms_refresh_requested"] = True
        imgui.same_line()
        if imgui.button("Update Y Scale"):
            ui._values["_histogram_update_y_limit"] = True
        imgui.same_line()
        if imgui.button("Update Log Range"):
            ui._values["_histogram_update_log_range"] = True
        imgui.push_item_width(160.0)
        changed, value = imgui.input_int("Bin Count", int(ui._values.get("hist_bin_count", _HISTOGRAM_BIN_COUNT_DEFAULT)), 8, 32)
        if changed:
            ui._values["hist_bin_count"] = max(int(value), 1)
        imgui.pop_item_width()
        imgui.same_line()
        imgui.push_item_width(160.0)
        changed, value = imgui.input_float("Min Log10", float(ui._values.get("hist_min_log10", _HISTOGRAM_MIN_LOG10_DEFAULT)), 0.25, 1.0, "%.3f")
        if changed:
            ui._values["hist_min_log10"] = float(value)
        imgui.pop_item_width()
        imgui.same_line()
        imgui.push_item_width(160.0)
        changed, value = imgui.input_float("Max Log10", float(ui._values.get("hist_max_log10", _HISTOGRAM_MAX_LOG10_DEFAULT)), 0.25, 1.0, "%.3f")
        if changed:
            ui._values["hist_max_log10"] = float(value)
        imgui.pop_item_width()
        imgui.same_line()
        imgui.push_item_width(160.0)
        changed, value = imgui.input_float("Y Limit", float(ui._values.get("hist_y_limit", _HISTOGRAM_Y_LIMIT_DEFAULT)), 1.0, 10.0, "%.1f")
        if changed:
            ui._values["hist_y_limit"] = max(float(value), 1.0)
        imgui.pop_item_width()
        imgui.separator()

    def _update_histogram_y_limit(self, ui: ViewerUI, payload: object) -> None:
        if not bool(ui._values.get("_histogram_update_y_limit", False)):
            return
        counts = np.asarray(getattr(payload, "counts", np.zeros((0, 0), dtype=np.int64)), dtype=np.float64)
        ui._values["hist_y_limit"] = max(1.3 * float(np.max(counts) if counts.size > 0 else 0.0), 1.0)
        ui._values["_histogram_update_y_limit"] = False

    def _update_histogram_log_range(self, ui: ViewerUI, range_payload: object) -> None:
        if not bool(ui._values.get("_histogram_update_log_range", False)):
            return
        log_range = _histogram_log_range_from_ranges(range_payload)
        if log_range is not None:
            ui._values["hist_min_log10"] = float(log_range[0])
            ui._values["hist_max_log10"] = float(log_range[1])
            ui._values["_histograms_refresh_requested"] = True
        ui._values["_histogram_update_log_range"] = False

    def _draw_histogram_groups(self, ui: ViewerUI, payload: object) -> None:
        labels = tuple(str(label) for label in getattr(payload, "param_labels", ()))
        centers = np.asarray(getattr(payload, "bin_centers_log10", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
        counts = np.asarray(getattr(payload, "counts", np.zeros((0, 0), dtype=np.int64)), dtype=np.float64)
        if counts.ndim != 2 or centers.size == 0:
            imgui.text_wrapped("Histogram payload is malformed.")
            return
        y_limit = float(ui._values.get("hist_y_limit", _HISTOGRAM_Y_LIMIT_DEFAULT))
        for group_name, indices in _HISTOGRAM_GROUPS:
            valid = tuple(index for index in indices if 0 <= int(index) < counts.shape[0])
            if not valid:
                continue
            imgui.separator_text(group_name)
            if imgui.begin_table(f"##hist_{group_name}", 2, imgui.TableFlags_.sizing_stretch_same.value):
                for index in valid:
                    imgui.table_next_column()
                    self._draw_histogram_plot(labels[index] if index < len(labels) else f"param {index}", centers, counts[index], y_limit)
                imgui.end_table()

    def _draw_histogram_plot(self, label: str, centers: np.ndarray, counts: np.ndarray, y_limit: float) -> None:
        imgui.text_disabled(label)
        plot_id = f"##plot_{label}"
        if implot.begin_plot(plot_id, imgui.ImVec2(-1, 180)):
            implot.setup_axes("log10(abs(grad))", "count", 0, 0)
            if centers.size > 0:
                implot.setup_axis_limits(implot.ImAxis_.x1.value, float(centers[0]), float(centers[-1]), implot.Cond_.always.value)
            implot.setup_axis_limits(implot.ImAxis_.y1.value, 0.0, max(float(y_limit), 1.0), implot.Cond_.always.value)
            implot.plot_line(label, centers, np.asarray(counts, dtype=np.float64))
            implot.end_plot()

    def _draw_histogram_range_debug(self, payload: object) -> None:
        imgui.separator_text("Range Debug")
        if payload is None:
            imgui.text_wrapped("Range debug data is unavailable.")
            return
        labels = tuple(str(label) for label in getattr(payload, "param_labels", ()))
        min_values = np.asarray(getattr(payload, "min_values", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        max_values = np.asarray(getattr(payload, "max_values", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        if min_values.ndim != 1 or max_values.ndim != 1 or min_values.size != max_values.size:
            imgui.text_wrapped("Range payload is malformed.")
            return
        flags = imgui.TableFlags_.row_bg.value | imgui.TableFlags_.borders.value | imgui.TableFlags_.sizing_stretch_same.value
        if not imgui.begin_table("##hist_range_debug", 4, flags):
            return
        imgui.table_setup_column("Component")
        imgui.table_setup_column("Min")
        imgui.table_setup_column("Max")
        imgui.table_setup_column("Max Abs")
        imgui.table_headers_row()
        for index in range(min_values.size):
            label = labels[index] if index < len(labels) else f"param {index}"
            min_value = float(min_values[index])
            max_value = float(max_values[index])
            max_abs = max(abs(min_value), abs(max_value))
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_unformatted(label)
            imgui.table_next_column()
            imgui.text_unformatted(self._format_histogram_range_value(min_value))
            imgui.table_next_column()
            imgui.text_unformatted(self._format_histogram_range_value(max_value))
            imgui.table_next_column()
            imgui.text_unformatted(self._format_histogram_range_value(max_abs))
        imgui.end_table()

    @staticmethod
    def _format_histogram_range_value(value: float) -> str:
        return "n/a" if not np.isfinite(value) else f"{value:.3e}"

    # -- Sections --

    def _section_status(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Status", imgui.TreeNodeFlags_.default_open.value):
            return
        # Training state indicator
        training_text = ui._texts.get("training", "Training: not initialized")
        if "running" in training_text:
            imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(0.2, 0.9, 0.3, 1.0))
            imgui.bullet_text("Training active")
            imgui.pop_style_color()
        elif "paused" in training_text:
            imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(1.0, 0.85, 0.2, 1.0))
            imgui.bullet_text("Training paused")
            imgui.pop_style_color()
        else:
            imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(0.5, 0.5, 0.5, 1.0))
            imgui.bullet_text("Idle")
            imgui.pop_style_color()

        # FPS with color gradient (green=good, yellow=ok, red=bad)
        fps = self.tk.fps_history[-1] if self.tk.fps_history else 0.0
        fps_color = imgui.ImVec4(0.2, 0.9, 0.3, 1.0) if fps >= 30 else imgui.ImVec4(1.0, 0.85, 0.2, 1.0) if fps >= 15 else imgui.ImVec4(1.0, 0.3, 0.3, 1.0)
        imgui.text("FPS: ")
        imgui.same_line(0, 0)
        imgui.push_style_color(imgui.Col_.text.value, fps_color)
        imgui.text(f"{fps:.1f}")
        imgui.pop_style_color()

        iters_per_second = self._iters_per_second(self.tk.step_history, self.tk.step_time_history)
        imgui.same_line()
        imgui.text_disabled("|")
        imgui.same_line()
        imgui.text("Iter/s: ")
        imgui.same_line(0, 0)
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(0.3, 0.85, 0.5, 1.0) if iters_per_second > 0.0 else imgui.ImVec4(0.5, 0.5, 0.5, 1.0))
        imgui.text(f"{iters_per_second:.1f}")
        imgui.pop_style_color()

        # Stats table
        tbl_flags = imgui.TableFlags_.sizing_stretch_same.value | imgui.TableFlags_.no_borders_in_body.value
        if imgui.begin_table("##status_tbl", 2, tbl_flags):
            imgui.table_setup_column("Label", imgui.TableColumnFlags_.width_fixed.value, 60)
            imgui.table_setup_column("Value")
            for label, key, fallback in (
                ("Scene", "path", "Scene: <none>"),
                ("Splats", "scene_stats", "Splats: 0"),
            ):
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_disabled(label)
                imgui.table_next_column()
                raw = ui._texts.get(key, fallback)
                imgui.text(raw.split(": ", 1)[-1] if ": " in raw else raw)
            imgui.end_table()

        # Render stats with tooltip
        render_stats = ui._texts.get("render_stats", "")
        if render_stats:
            imgui.text_disabled("Render:")
            imgui.same_line()
            imgui.text(render_stats.split(": ", 1)[-1] if "Generated:" in render_stats else render_stats)
            if imgui.is_item_hovered():
                imgui.set_item_tooltip("Generated = splat entries created, Written = pixels rasterized")

        error = ui._texts.get("error", "")
        if error:
            imgui.spacing()
            imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(1.0, 0.3, 0.3, 1.0))
            imgui.text_wrapped(error)
            imgui.pop_style_color()
        imgui.separator()

    def _section_scene_io(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Scene I/O", imgui.TreeNodeFlags_.default_open.value):
            return
        imgui.text_disabled("Use File for scene load and reload actions.")
        imgui.text_disabled("Load COLMAP opens a dedicated import window.")
        if imgui.button("Open COLMAP Import", imgui.ImVec2(imgui.get_content_region_avail().x, 0.0)):
            self._show_colmap_import = True
        root_text = self._path_text(ui, "colmap_root_path", "<none>")
        images_text = self._path_text(ui, "colmap_images_root", "<none>")
        mode_idx = max(0, min(int(ui._values.get("colmap_init_mode", 0)), len(_COLMAP_INIT_MODE_LABELS) - 1))
        imgui.spacing()
        imgui.text_disabled(f"Root: {Path(root_text).name if root_text != '<none>' else root_text}")
        imgui.text_disabled(f"Images: {Path(images_text).name if images_text != '<none>' else images_text}")
        imgui.text_disabled(f"Init: {_COLMAP_INIT_MODE_LABELS[mode_idx]}")
        imgui.separator()

    def _section_camera(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Camera", imgui.TreeNodeFlags_.default_open.value):
            return
        changed, val = imgui.drag_float(
            "Move Speed", float(ui._values["move_speed"]),
            0.05, 0.1, 20.0, "%.2f", imgui.SliderFlags_.logarithmic.value
        )
        if changed:
            ui._values["move_speed"] = max(0.1, min(val, 20.0))
        if imgui.is_item_hovered():
            imgui.set_item_tooltip("Camera movement speed (scroll wheel also adjusts)")
        changed, val = imgui.slider_float("FOV", float(ui._values["fov"]), 25.0, 100.0, "%.1f\u00b0")
        if changed:
            ui._values["fov"] = val
        if imgui.is_item_hovered():
            imgui.set_item_tooltip("Vertical field of view in degrees")
        imgui.spacing()
        imgui.text_disabled("LMB drag=look | WASDQE=move | Wheel=speed")
        self._ctx_reset("camera_ctx", ui, ("move_speed", "fov"))
        imgui.separator()

    def _section_training_control(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Training", imgui.TreeNodeFlags_.default_open.value):
            return
        # Training info in compact table
        training_text = ui._texts.get("training", "Training: not initialized")
        time_text = ui._texts.get("training_time", "Time: n/a")
        avg_iters_text = ui._texts.get("training_iters_avg", "Avg it/s: n/a")
        loss_text = ui._texts.get("training_loss", "Loss Avg: n/a")
        mse_text = ui._texts.get("training_mse", "MSE: n/a")
        psnr_text = ui._texts.get("training_psnr", "PSNR: n/a")
        if imgui.begin_table("##train_info", 2, imgui.TableFlags_.sizing_stretch_same.value):
            imgui.table_setup_column("L", imgui.TableColumnFlags_.width_fixed.value, 50)
            imgui.table_setup_column("V")
            for label, text in (("Step", training_text), ("Time", time_text), ("Avg", avg_iters_text), ("Loss", loss_text), ("MSE", mse_text), ("PSNR", psnr_text)):
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_disabled(label)
                imgui.table_next_column()
                imgui.text(text.split(": ", 1)[-1] if ": " in text else text)
            imgui.end_table()

        instability = ui._texts.get("training_instability", "")
        if instability:
            imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(1.0, 0.85, 0.2, 1.0))
            imgui.text_wrapped(instability)
            imgui.pop_style_color()

        # Colored Start/Stop buttons
        imgui.spacing()
        half_w = imgui.get_content_region_avail().x * 0.48
        imgui.push_style_color(imgui.Col_.button.value, imgui.ImVec4(0.15, 0.55, 0.22, 0.80))
        imgui.push_style_color(imgui.Col_.button_hovered.value, imgui.ImVec4(0.20, 0.68, 0.28, 0.90))
        imgui.push_style_color(imgui.Col_.button_active.value, imgui.ImVec4(0.15, 0.55, 0.22, 1.00))
        if imgui.button("Start", imgui.ImVec2(half_w, 0)):
            self.callbacks.start_training()
        imgui.pop_style_color(3)
        imgui.same_line()
        imgui.push_style_color(imgui.Col_.button.value, imgui.ImVec4(0.65, 0.15, 0.15, 0.80))
        imgui.push_style_color(imgui.Col_.button_hovered.value, imgui.ImVec4(0.80, 0.20, 0.20, 0.90))
        imgui.push_style_color(imgui.Col_.button_active.value, imgui.ImVec4(0.65, 0.15, 0.15, 1.00))
        if imgui.button("Stop", imgui.ImVec2(half_w, 0)):
            self.callbacks.stop_training()
        imgui.pop_style_color(3)

        if imgui.button("Reinitialize Gaussians", imgui.ImVec2(imgui.get_content_region_avail().x, 0)):
            self.callbacks.reinitialize()
        if imgui.is_item_hovered():
            imgui.set_item_tooltip("Re-initialize all gaussians from the point cloud")

        # Debug section with indent
        imgui.spacing()
        changed, val = imgui.checkbox("Visual Loss Debug", bool(ui._values["loss_debug"]))
        if changed:
            ui._values["loss_debug"] = val

        if bool(ui._values["loss_debug"]):
            imgui.indent()
            # Combo for debug view selection
            view_idx = min(max(int(ui._values["loss_debug_view"]), 0), len(LOSS_DEBUG_OPTIONS) - 1)
            if imgui.begin_combo("##debugview", LOSS_DEBUG_OPTIONS[view_idx][1]):
                for i, (key, name) in enumerate(LOSS_DEBUG_OPTIONS):
                    if imgui.selectable(name, i == view_idx)[0]:
                        ui._values["loss_debug_view"] = i
                    if i == view_idx:
                        imgui.set_item_default_focus()
                imgui.end_combo()
            if LOSS_DEBUG_OPTIONS[view_idx][0] == "abs_diff":
                self._draw_control(ui, next(spec for spec in GROUP_SPECS["Main"] if spec.key == _LOSS_DEBUG_ABS_SCALE_KEY))

            frame_max = max(int(ui._values.get("_loss_debug_frame_max", 0)), 0)
            changed, val = imgui.slider_int(
                "Frame", int(ui._values["loss_debug_frame"]), 0, frame_max
            )
            if changed:
                ui._values["loss_debug_frame"] = val
            frame_text = ui._texts.get("loss_debug_frame", "")
            if frame_text:
                imgui.text_disabled(frame_text.split(": ", 1)[-1] if ": " in frame_text else frame_text)
            imgui.unindent()
        imgui.separator()

    def _section_training_setup(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Train Setup"):
            return
        for key in ("max_gaussians", "training_steps_per_frame", "train_downscale_mode"):
            self._draw_control(ui, next(spec for spec in GROUP_SPECS["Train Setup"] if spec.key == key))
        if int(ui._values.get("train_downscale_mode", 0)) == 0:
            for key in ("train_auto_start_downscale", "train_downscale_base_iters", "train_downscale_iter_step", "train_downscale_max_iters"):
                self._draw_control(ui, next(spec for spec in GROUP_SPECS["Train Setup"] if spec.key == key))
        for key in ("seed", "init_opacity"):
            self._draw_control(ui, next(spec for spec in GROUP_SPECS["Train Setup"] if spec.key == key))
        train_resolution = ui._texts.get("training_resolution", "")
        if train_resolution:
            imgui.text_disabled(train_resolution.split(": ", 1)[-1] if ": " in train_resolution else train_resolution)
        downscale_status = ui._texts.get("training_downscale", "")
        if downscale_status:
            imgui.text_disabled(downscale_status.split(": ", 1)[-1] if ": " in downscale_status else downscale_status)
        imgui.text_disabled("COLMAP import chooses pointcloud NN-scale init or a custom PLY scene.")
        self._ctx_reset("train_setup_ctx", ui, [s.key for s in GROUP_SPECS["Train Setup"]])
        imgui.separator()

    def _section_optimizer(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Optimizer"):
            return
        if imgui.begin_tab_bar("##optim_tabs"):
            if imgui.begin_tab_item("Learning Rates")[0]:
                imgui.spacing()
                for spec in GROUP_SPECS["Train Optimizer"][:6]:
                    self._draw_control(ui, spec)
                imgui.end_tab_item()
            if imgui.begin_tab_item("Adam")[0]:
                imgui.spacing()
                for spec in GROUP_SPECS["Train Optimizer"][6:8]:
                    self._draw_control(ui, spec)
                imgui.end_tab_item()
            if imgui.begin_tab_item("Regularization")[0]:
                imgui.spacing()
                for spec in GROUP_SPECS["Train Optimizer"][8:]:
                    self._draw_control(ui, spec)
                imgui.end_tab_item()
            imgui.end_tab_bar()
        self._ctx_reset("optimizer_ctx", ui, [s.key for s in GROUP_SPECS["Train Optimizer"]])
        imgui.separator()

    def _section_stability(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Stability"):
            return
        # Paired min/max controls in a two-column table
        pairs = (
            ("min_scale", "max_scale", "Scale"),
            ("min_opacity", "max_opacity", "Opacity"),
        )
        if imgui.begin_table("##stab_pairs", 2, imgui.TableFlags_.sizing_stretch_same.value | imgui.TableFlags_.no_borders_in_body.value):
            imgui.table_setup_column("Min")
            imgui.table_setup_column("Max")
            for min_key, max_key, label in pairs:
                min_spec = next(s for s in GROUP_SPECS["Train Stability"] if s.key == min_key)
                max_spec = next(s for s in GROUP_SPECS["Train Stability"] if s.key == max_key)
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.push_item_width(-1)
                self._draw_control(ui, min_spec)
                imgui.pop_item_width()
                imgui.table_next_column()
                imgui.push_item_width(-1)
                self._draw_control(ui, max_spec)
                imgui.pop_item_width()
            imgui.end_table()

        # Remaining non-paired controls
        paired_keys = {k for min_k, max_k, _ in pairs for k in (min_k, max_k)}
        for spec in GROUP_SPECS["Train Stability"]:
            if spec.key not in paired_keys:
                self._draw_control(ui, spec)
        imgui.text_disabled("Bounds and anisotropy are clamped after each ADAM step")
        self._ctx_reset("stability_ctx", ui, [s.key for s in GROUP_SPECS["Train Stability"]])
        imgui.separator()

    def _section_render_params(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Render Params"):
            return
        # Main render parameters
        for spec in RENDER_PARAM_SPECS[:6]:
            self._draw_control(ui, spec)

        # Debug overlay group
        imgui.separator_text("Debug Overlays")
        for spec in RENDER_PARAM_SPECS[6:]:
            self._draw_control(ui, spec)
        self._ctx_reset("render_ctx", ui, [s.key for s in RENDER_PARAM_SPECS])
        imgui.separator()

    def _section_performance(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Plots", imgui.TreeNodeFlags_.default_open.value):
            return

        iters_per_second = self._iters_per_second(self.tk.step_history, self.tk.step_time_history)
        if iters_per_second > 0.0:
            imgui.text_disabled(f"iters/s {iters_per_second:.1f}")

        fps_arr = np.array(self.tk.fps_history, dtype=np.float64)
        if len(fps_arr) >= 2:
            imgui.text_disabled(f"avg {np.mean(fps_arr):.1f}  min {np.min(fps_arr):.1f}  max {np.max(fps_arr):.1f}")
            if implot.begin_plot("##FPS", imgui.ImVec2(-1, 110)):
                implot.setup_axes("", "FPS", implot.AxisFlags_.no_tick_labels.value, implot.AxisFlags_.auto_fit.value)
                implot.setup_axis_limits(implot.ImAxis_.x1.value, 0, len(fps_arr) - 1, implot.Cond_.always.value)
                shade_spec = implot.Spec()
                shade_spec.fill_alpha = 0.25
                implot.plot_shaded("FPS", fps_arr, 0.0, spec=shade_spec)
                implot.plot_line("FPS", fps_arr)
                implot.annotation(float(len(fps_arr) - 1), float(fps_arr[-1]), imgui.ImVec4(0.4, 0.75, 1.0, 1.0), imgui.ImVec2(-10, -10), True, f"{fps_arr[-1]:.0f}")
                implot.end_plot()
        else:
            imgui.text_disabled("Waiting for FPS data...")

        loss_arr = np.array(self.tk.loss_history, dtype=np.float64)
        step_arr = np.array(self.tk.step_history, dtype=np.float64)
        if len(loss_arr) >= 2 and len(step_arr) >= 2:
            min_len = min(len(loss_arr), len(step_arr))
            s, l = step_arr[:min_len], loss_arr[:min_len]
            imgui.separator_text("Loss")
            loss_spec = implot.Spec()
            loss_spec.line_color = imgui.ImVec4(1.0, 0.6, 0.2, 1.0)
            if implot.begin_plot("##Loss", imgui.ImVec2(-1, 180)):
                implot.setup_axes("step", "loss", 0, implot.AxisFlags_.auto_fit.value)
                implot.setup_axis_limits(implot.ImAxis_.x1.value, float(s[0]), float(s[-1]), implot.Cond_.always.value)
                implot.setup_axis_scale(implot.ImAxis_.y1.value, implot.Scale_.log10.value)
                implot.plot_line("Avg Loss", s, l, spec=loss_spec)
                implot.annotation(float(s[-1]), float(l[-1]), imgui.ImVec4(1.0, 0.6, 0.2, 1.0), imgui.ImVec2(-10, -10), True, f"{l[-1]:.2e}")
                implot.end_plot()

        psnr_arr = np.array(self.tk.psnr_history, dtype=np.float64)
        if len(psnr_arr) >= 2 and len(step_arr) >= 2:
            min_len = min(len(psnr_arr), len(step_arr))
            s, p = step_arr[:min_len], psnr_arr[:min_len]
            imgui.separator_text("PSNR")
            psnr_spec = implot.Spec()
            psnr_spec.line_color = imgui.ImVec4(0.3, 0.85, 0.5, 1.0)
            if implot.begin_plot("##PSNR", imgui.ImVec2(-1, 180)):
                implot.setup_axes("step", "PSNR (dB)", 0, implot.AxisFlags_.auto_fit.value)
                implot.setup_axis_limits(implot.ImAxis_.x1.value, float(s[0]), float(s[-1]), implot.Cond_.always.value)
                implot.plot_line("PSNR", s, p, spec=psnr_spec)
                implot.annotation(float(s[-1]), float(p[-1]), imgui.ImVec4(0.3, 0.85, 0.5, 1.0), imgui.ImVec2(-10, -10), True, f"{p[-1]:.2f}")
                implot.end_plot()

    # -- Helpers --

    _TOOLTIPS = {
        "radius_scale": "Multiplier on top of true 3DGS gaussian size for rendering",
        "alpha_cutoff": "Minimum alpha threshold — splats below this are skipped",
        "max_splat_steps": "Maximum rasterization steps per pixel ray",
        "trans_threshold": "Transmittance threshold for early ray termination",
        "sampled5_safety": "Safety margin for MVEE bounding ellipsoid",
        "cached_raster_grad_atomic_mode": "Choose float atomics or fixed-point atomics for cached ellipsoid gradient accumulation during raster backward",
        "cached_raster_grad_fixed_ro_local_range": "Symmetric [-X, X] range for shapeAlpha-normalized cached position gradients",
        "cached_raster_grad_fixed_log_l_diag_range": "Symmetric [-X, X] range for shapeAlpha-normalized cached scale gradients",
        "cached_raster_grad_fixed_l_offdiag_range": "Symmetric [-X, X] range for shapeAlpha-normalized cached rotation gradients",
        "cached_raster_grad_fixed_color_range": "Symmetric [-X, X] range for cached color gradients",
        "cached_raster_grad_fixed_opacity_range": "Symmetric [-X, X] range for cached opacity gradients",
        "debug_ellipse": "Show ellipse outlines around each gaussian",
        "debug_processed_count": "Heatmap of processed splats per pixel",
        "debug_grad_norm": "Heatmap of gradient norms per pixel",
        "lr_base": "Base learning rate for all parameters",
        "lr_pos_mul": "Learning rate multiplier for position",
        "lr_scale_mul": "Learning rate multiplier for scale",
        "lr_rot_mul": "Learning rate multiplier for rotation",
        "lr_color_mul": "Learning rate multiplier for color/SH",
        "lr_opacity_mul": "Learning rate multiplier for opacity",
        "beta1": "Adam first moment decay (momentum)",
        "beta2": "Adam second moment decay (RMSprop)",
        "grad_clip": "Per-parameter gradient clipping threshold",
        "grad_norm_clip": "Global gradient norm clipping threshold",
        "max_update": "Maximum per-step parameter update magnitude",
        "scale_l2": "L2 regularization on log-scale",
        "scale_abs_reg": "Absolute scale regularization weight",
        "opacity_reg": "Opacity regularization weight (pushes toward 0 or 1)",
        "max_anisotropy": "Maximum ratio between largest and smallest scale axes",
        "min_scale": "Floor for decoded gaussian sigma",
        "max_scale": "Ceiling for decoded gaussian sigma",
        "min_opacity": "Floor for opacity",
        "max_opacity": "Ceiling for opacity",
        "position_abs_max": "Absolute position bounding box (per axis)",
        "train_near": "Near clip plane for training camera",
        "train_far": "Far clip plane for training camera",
        "max_gaussians": "Maximum number of gaussians in the scene",
        "training_steps_per_frame": "Number of training optimizer steps to run before each viewer redraw; higher improves training throughput but reduces UI refresh rate",
        "train_downscale_mode": "Use Auto for scheduled downscale descent or choose a fixed manual override from 1x to 16x",
        "train_auto_start_downscale": "Initial downscale factor used at step 0 when Downscale Mode is Auto",
        "train_downscale_base_iters": "Number of iterations spent at the auto start factor before descending",
        "train_downscale_iter_step": "Additional iterations added to each lower auto downscale phase",
        "train_downscale_max_iters": "Displayed training schedule budget for the auto downscale progression; training does not stop automatically",
        _LOSS_DEBUG_ABS_SCALE_KEY: "Multiplier applied to absolute RGB difference before presenting the debug texture",
        "seed": "Random seed for training frame shuffle order",
        "init_opacity": "Initial opacity for new gaussians",
    }

    def _draw_control(self, ui: ViewerUI, spec: ControlSpec) -> None:
        key = spec.key
        if key not in ui._values:
            ui._values[key] = spec.kwargs.get("value", 0)
        if spec.kind == "slider_float":
            changed, val = imgui.slider_float(
                spec.label, float(ui._values[key]),
                float(spec.kwargs.get("min", 0.0)), float(spec.kwargs.get("max", 1.0)),
                spec.kwargs.get("format", "%.3f"),
                imgui.SliderFlags_.logarithmic.value if spec.kwargs.get("logarithmic") else 0
            )
            if changed:
                ui._values[key] = val
        elif spec.kind == "slider_int":
            changed, val = imgui.slider_int(
                spec.label, int(ui._values[key]),
                int(spec.kwargs.get("min", 0)), int(spec.kwargs.get("max", 100))
            )
            if changed:
                ui._values[key] = val
        elif spec.kind == "input_int":
            changed, val = imgui.input_int(
                spec.label,
                int(ui._values[key]),
                int(spec.kwargs.get("step", 1)),
                int(spec.kwargs.get("step_fast", 10)),
            )
            if changed:
                ui._values[key] = val
        elif spec.kind == "combo":
            options = tuple(spec.kwargs.get("options", ()))
            option_count = max(len(options), 1)
            current = min(max(int(ui._values[key]), 0), option_count - 1)
            preview = str(options[current]) if options else str(current)
            if imgui.begin_combo(spec.label, preview):
                for idx, option in enumerate(options):
                    selected = idx == current
                    if imgui.selectable(str(option), selected)[0]:
                        ui._values[key] = idx
                    if selected:
                        imgui.set_item_default_focus()
                imgui.end_combo()
        elif spec.kind == "input_float":
            changed, val = imgui.input_float(
                spec.label, float(ui._values[key]),
                float(spec.kwargs.get("step", 0.0)),
                float(spec.kwargs.get("step_fast", 0.0)),
                spec.kwargs.get("format", "%.3f")
            )
            if changed:
                ui._values[key] = val
        elif spec.kind == "checkbox":
            changed, val = imgui.checkbox(spec.label, bool(ui._values[key]))
            if changed:
                ui._values[key] = val
        tip = self._TOOLTIPS.get(key)
        if tip and imgui.is_item_hovered():
            imgui.set_item_tooltip(tip)

    def _ctx_reset(self, ctx_id: str, ui: ViewerUI, keys) -> None:
        if imgui.begin_popup_context_item(ctx_id):
            if imgui.selectable("Reset to Defaults")[0]:
                for key in keys:
                    if key in _ALL_DEFAULTS:
                        ui._values[key] = _ALL_DEFAULTS[key]
            imgui.separator()
            if imgui.selectable("Copy Values to Clipboard")[0]:
                lines = [f"{k}={ui._values.get(k, '?')}" for k in keys if k in ui._values]
                imgui.set_clipboard_text("\n".join(lines))
            imgui.end_popup()

    def shutdown(self) -> None:
        if not self._alive:
            return
        self._alive = False
        self._frame_textures.clear()
        self._set_current_context()
        implot.destroy_context()
        imgui.destroy_context(self.ctx)


def build_ui(renderer) -> ViewerUI:
    """Build a ViewerUI with default control values populated from renderer."""
    values: dict[str, object] = {}
    for specs in GROUP_SPECS.values():
        for spec in specs:
            if "value" in spec.kwargs:
                values[spec.key] = spec.kwargs["value"]
    for spec in RENDER_PARAM_SPECS:
        values[spec.key] = spec.kwargs.get("value", 0)
    values["radius_scale"] = float(renderer.radius_scale)
    values["alpha_cutoff"] = float(renderer.alpha_cutoff)
    values["max_splat_steps"] = int(renderer.max_splat_steps)
    values["trans_threshold"] = float(renderer.transmittance_threshold)
    values["sampled5_safety"] = float(renderer.sampled5_safety_scale)
    values["cached_raster_grad_atomic_mode"] = 0 if getattr(renderer, "cached_raster_grad_atomic_mode", "fixed") == "float" else 1
    values["cached_raster_grad_fixed_ro_local_range"] = float(getattr(renderer, "cached_raster_grad_fixed_ro_local_range", 10.0))
    values["cached_raster_grad_fixed_log_l_diag_range"] = float(getattr(renderer, "cached_raster_grad_fixed_log_l_diag_range", 10.0))
    values["cached_raster_grad_fixed_l_offdiag_range"] = float(getattr(renderer, "cached_raster_grad_fixed_l_offdiag_range", 10.0))
    values["cached_raster_grad_fixed_color_range"] = float(getattr(renderer, "cached_raster_grad_fixed_color_range", 200.0))
    values["cached_raster_grad_fixed_opacity_range"] = float(getattr(renderer, "cached_raster_grad_fixed_opacity_range", 200.0))
    values["debug_ellipse"] = bool(renderer.debug_show_ellipses)
    values["debug_processed_count"] = bool(renderer.debug_show_processed_count)
    values["debug_grad_norm"] = bool(renderer.debug_show_grad_norm)
    values["debug_grad_norm_threshold"] = float(getattr(renderer, "debug_grad_norm_threshold", _DEBUG_GRAD_NORM_THRESHOLD_DEFAULT))
    values["colmap_root_path"] = ""
    values["colmap_database_path"] = ""
    values["colmap_images_root"] = ""
    values["colmap_init_mode"] = 0
    values["colmap_custom_ply_path"] = ""
    values["colmap_nn_radius_scale_coef"] = 1.0
    values["show_histograms"] = False
    values["hist_auto_refresh"] = True
    values["hist_bin_count"] = _HISTOGRAM_BIN_COUNT_DEFAULT
    values["hist_min_log10"] = _HISTOGRAM_MIN_LOG10_DEFAULT
    values["hist_max_log10"] = _HISTOGRAM_MAX_LOG10_DEFAULT
    values["hist_y_limit"] = _HISTOGRAM_Y_LIMIT_DEFAULT
    values["_histograms_refresh_requested"] = False
    values["_histogram_update_y_limit"] = True
    values["_histogram_update_log_range"] = False
    values["_histogram_payload"] = None
    values["_histogram_range_payload"] = None
    values["_loss_debug_frame_max"] = 0
    values["_colmap_import_active"] = False
    values["_colmap_import_fraction"] = 0.0

    texts: dict[str, str] = {
        key: "" for key in (
            "fps", "path", "scene_stats", "render_stats", "training",
            "training_time", "training_iters_avg", "training_loss", "training_mse", "training_psnr", "training_instability", "error",
            "loss_debug_view", "loss_debug_frame",
            "colmap_import_status", "colmap_import_current",
            "training_resolution", "training_downscale",
            "histogram_status",
            "setup_hint", "stability_hint",
        )
    }
    return ViewerUI(_values=values, _texts=texts)


def create_toolkit_window(device: spy.Device, width: int, height: int) -> ToolkitWindow:
    return ToolkitWindow(device=device, width=width, height=height)
