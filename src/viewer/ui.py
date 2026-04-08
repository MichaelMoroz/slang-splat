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
from imgui_bundle import hello_imgui, imgui, imgui_md, implot

from .constants import _WINDOW_TITLE
from .state import DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH, LOSS_DEBUG_OPTIONS

TOOLKIT_WIDTH_FRACTION = 0.1875
VIEW_WIDTH_FRACTION = 0.8125
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
_COLMAP_INIT_MODE_BASE_LABELS = ("COLMAP Pointcloud", "Diffused Pointcloud", "Custom PLY")
_COLMAP_INIT_MODE_DEPTH_LABEL = "From Depth"
_COLMAP_INIT_MODE_LABELS = _COLMAP_INIT_MODE_BASE_LABELS
_COLMAP_DEPTH_VALUE_MODE_LABELS = ("Depth Is Distance", "Depth Is Z-Depth")
_COLMAP_IMAGE_DOWNSCALE_LABELS = ("Original", "Max Size", "Scale Factor")
_TRAIN_BACKGROUND_MODE_LABELS = ("Custom", "Random")
_VIEWER_BACKGROUND_MODE_LABELS = ("Train Background", "Custom")
_TRAIN_DOWNSCALE_MODE_LABELS = ("Auto",) + tuple(f"{i}x" for i in range(1, 17))
_TRAIN_SUBSAMPLE_LABELS = ("Auto", "Off", "1/2", "1/3", "1/4")
_SH_BAND_LABELS = ("SH0", "SH1", "SH2", "SH3")
_DEBUG_GRAD_NORM_THRESHOLD_DEFAULT = 2e-4
_DEBUG_ADAM_MOMENTUM_THRESHOLD_DEFAULT = 1e-2
_DEBUG_CONTRIBUTION_AMOUNT_FLOOR = 1e-6
_DEBUG_COLORBAR_HEIGHT = 28.0
_DEBUG_COLORBAR_MIN_WIDTH = 320.0
_DEBUG_COLORBAR_MAX_WIDTH = 640.0
_DEBUG_COLORBAR_MARGIN = 18.0
_DEBUG_COLORBAR_TICKS = 5
_DEBUG_COLORBAR_STEPS = 96
_DEBUG_COLORBAR_SIDE_PAD = 18.0
_DEBUG_COLORBAR_TOP_PAD = 30.0
_DEBUG_COLORBAR_BOTTOM_PAD = 30.0
_VIEWPORT_OVERLAY_MARGIN = 8.0
_VIEWPORT_OVERLAY_WIDTH = 320.0
_VIEWPORT_OVERLAY_MIN_WIDTH = 220.0
_VIEWPORT_OVERLAY_PADDING = 10.0
_VIEWPORT_OVERLAY_MIN_HEIGHT = 44.0
_HISTOGRAM_AUTO_RANGE_KEEP_FRACTION = 0.99
_DOCKSPACE_FLAGS = int(imgui.DockNodeFlags_.none)
_TOOLKIT_WINDOW_NAME = "Toolkit"
_VIEWPORT_WINDOW_NAME = "Viewport###Viewport"
_HISTOGRAM_BIN_COUNT_DEFAULT = 64
_HISTOGRAM_MIN_LOG10_DEFAULT = -8.0
_HISTOGRAM_MAX_LOG10_DEFAULT = 2.0
_HISTOGRAM_Y_LIMIT_DEFAULT = 1.0
_HISTOGRAM_WINDOW_WIDTH = 1200.0
_HISTOGRAM_WINDOW_HEIGHT = 860.0
_HISTOGRAM_CONTROL_LABEL_WIDTH = 150.0
_HISTOGRAM_PLOT_HEIGHT = 230.0
_HISTOGRAM_PLOT_MIN_COLUMN_WIDTH = 460.0
_HISTOGRAM_GROUPS = (
    ("roLocal", (0, 1, 2)),
    ("scale", (3, 4, 5)),
    ("quat", (6, 7, 8, 9)),
    ("color", (10, 11, 12)),
    ("opacity", (13,)),
)
_CACHED_RASTER_GRAD_ATOMIC_MODE_LABELS = ("Float Atomics", "Fixed Point")
_DEBUG_MODE_VALUES = (
    "normal",
    "processed_count",
    "clone_count",
    "ellipse_outlines",
    "splat_density",
    "splat_spatial_density",
    "splat_screen_density",
    "contribution_amount",
    "adam_momentum",
    "depth_mean",
    "depth_std",
    "depth_local_mismatch",
    "grad_norm",
    "sh_view_dependent",
    "sh_coefficient",
)
_DEBUG_MODE_LABELS = (
    "Normal",
    "Processed Count",
    "Clone Count",
    "Ellipse Outlines",
    "Splat Density",
    "Spatial Density",
    "Screen Density",
    "Contribution Amount",
    "Adam Momentum",
    "Depth Mean",
    "Depth Std",
    "Depth Local Mismatch",
    "Grad Norm",
    "SH View-Dependent",
    "SH Coefficient",
)
_DEBUG_SH_COEFF_LABELS = ("SH0 DC", "SH1 X", "SH1 Y", "SH1 Z", "SH2 0", "SH2 1", "SH2 2", "SH2 3", "SH2 4", "SH3 0", "SH3 1", "SH3 2", "SH3 3", "SH3 4", "SH3 5", "SH3 6")
def _valid_depth_root_text(value: object) -> bool:
    text = str(value or "").strip()
    return bool(text) and Path(text).expanduser().is_dir()


def _colmap_init_mode_labels(depth_available: bool) -> tuple[str, ...]:
    return _COLMAP_INIT_MODE_BASE_LABELS + ((_COLMAP_INIT_MODE_DEPTH_LABEL,) if depth_available else ())


def _colmap_init_mode_label(ui: "ViewerUI", index: int | None = None) -> str:
    labels = _colmap_init_mode_labels(_valid_depth_root_text(ui._values.get("colmap_depth_root", "")))
    mode_idx = max(0, min(int(ui._values.get("colmap_init_mode", 0) if index is None else index), len(labels) - 1))
    return labels[mode_idx]


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


def _imgui_bundle_assets_path() -> Path:
    package = importlib.import_module("imgui_bundle")
    return Path(package.__file__).resolve().parent / "assets"


def _markdown_font_base_path() -> Path | None:
    path = _imgui_bundle_assets_path() / "fonts" / "Roboto" / "Roboto"
    return path if path.with_name(path.name + "-Regular.ttf").exists() else None


def _status_suffix(text: str) -> str:
    value = str(text).strip()
    return value.split(": ", 1)[-1] if ": " in value else value


def _draw_disabled_wrapped_text(text: str) -> None:
    value = _status_suffix(text)
    if not value:
        return
    imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x)
    imgui.begin_disabled()
    imgui.text_unformatted(value)
    imgui.end_disabled()
    imgui.pop_text_wrap_pos()


def _draw_markdown_text(text: str) -> None:
    value = str(text).strip()
    if not value:
        return
    try:
        imgui_md.render_unindented(value)
    except Exception:
        imgui.push_text_wrap_pos(_DOC_MAX_WIDTH * imgui.get_font_size() * 0.5)
        imgui.text_unformatted(value)
        imgui.pop_text_wrap_pos()


def _build_about_text() -> str:
    return "\n".join(
        (
            f"# {_WINDOW_TITLE}",
            "",
            "Single-window Gaussian splat viewer and trainer built on **Slangpy**.",
            "",
            "The scene is presented inside a docked viewport window with the **imgui-bundle** UI around it.",
            "",
            f"**Controls:** {_SHORTCUTS_TEXT.split(': ', 1)[-1]}",
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
    return max(float(width) - panel_width, 0.0), float(menu_bar_height), panel_width, max(float(height) - float(menu_bar_height), 1.0)


def _clamp_viewport_size(width: float, height: float) -> tuple[int, int]:
    return max(int(round(float(width))), 1), max(int(round(float(height))), 1)


def _rect_contains(rect: tuple[float, float, float, float], point: tuple[float, float] | None) -> bool:
    if point is None:
        return False
    x, y, width, height = rect
    px, py = point
    return px >= x and py >= y and px < x + width and py < y + height


def _should_capture_keyboard_for_ui(handled: bool, viewport_input_active: bool, want_text_input: bool) -> bool:
    return bool(handled) and not (bool(viewport_input_active) and not bool(want_text_input))


@lru_cache(maxsize=1)
def _default_font_path() -> Path | None:
    package = importlib.import_module("imgui_bundle")
    path = Path(package.__file__).resolve().parent / "assets" / "fonts" / "DroidSans.ttf"
    return path if path.exists() else None


def _menu_item(label: str, shortcut: str = "", selected: bool = False, enabled: bool = True) -> bool:
    return bool(imgui.menu_item(label, shortcut, selected, enabled)[0])


def _imgui_opened(value: object) -> bool:
    return bool(value[0] if isinstance(value, tuple) else value)


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
    index = min(max(int(ui._values.get("debug_mode", 0)), 0), len(_DEBUG_MODE_VALUES) - 1)
    mode = _DEBUG_MODE_VALUES[index]
    return None if mode in ("normal", "ellipse_outlines", "sh_view_dependent", "sh_coefficient") else mode


def _renderer_debug_control_keys(mode: str) -> tuple[str, ...]:
    if mode == "ellipse_outlines": return ("debug_mode", "debug_ellipse_thickness_px")
    if mode == "grad_norm": return ("debug_mode", "debug_grad_norm_threshold")
    if mode == "sh_coefficient": return ("debug_mode", "debug_sh_coeff_index")
    if mode == "clone_count": return ("debug_mode", "debug_clone_count_min", "debug_clone_count_max")
    if mode in ("splat_density", "splat_spatial_density", "splat_screen_density"): return ("debug_mode", "debug_density_min", "debug_density_max")
    if mode == "contribution_amount": return ("debug_mode", "debug_contribution_min", "debug_contribution_max")
    if mode == "adam_momentum": return ("debug_mode", "debug_grad_norm_threshold")
    if mode == "depth_mean": return ("debug_mode", "debug_depth_mean_min", "debug_depth_mean_max")
    if mode == "depth_std": return ("debug_mode", "debug_depth_std_min", "debug_depth_std_max")
    if mode == "depth_local_mismatch": return ("debug_mode", "debug_depth_local_mismatch_min", "debug_depth_local_mismatch_max", "debug_depth_local_mismatch_smooth_radius", "debug_depth_local_mismatch_reject_radius")
    return ("debug_mode",)


def _processed_count_tick_value(t: float, max_splat_steps: int) -> float:
    return math.pow(2.0, _saturate(t) * math.log2(max(max_splat_steps, 0) + 1.0)) - 1.0


def _threshold_band_range(threshold: float) -> tuple[float, float]:
    constants = _shader_debug_constants()
    floor = constants["DEBUG_GRAD_NORM_FLOOR"]
    value = max(float(threshold), floor)
    return (
        max(value * constants["DEBUG_GRAD_THRESHOLD_MIN_SCALE"], floor),
        max(value * constants["DEBUG_GRAD_THRESHOLD_MAX_SCALE"], floor),
    )


def _threshold_band_tick_value(t: float, threshold: float) -> float:
    lo_value, hi_value = _threshold_band_range(threshold)
    lo = math.log10(lo_value)
    hi = math.log10(hi_value)
    return math.pow(10.0, lo + _saturate(t) * (hi - lo))


def _threshold_from_band_range(value_min: float, value_max: float, default: float) -> float:
    constants = _shader_debug_constants()
    floor = constants["DEBUG_GRAD_NORM_FLOOR"]
    lo = max(float(min(value_min, value_max)), 0.0)
    hi = max(float(max(value_min, value_max)), 0.0)
    candidates: list[float] = []
    if lo > 0.0: candidates.append(lo / constants["DEBUG_GRAD_THRESHOLD_MIN_SCALE"])
    if hi > 0.0: candidates.append(hi / constants["DEBUG_GRAD_THRESHOLD_MAX_SCALE"])
    if len(candidates) == 0:
        return max(float(default), floor)
    if len(candidates) == 1:
        return max(candidates[0], floor)
    return max(math.sqrt(candidates[0] * candidates[1]), floor)


def _contribution_amount_tick_value(t: float, value_min: float, value_max: float) -> float:
    lo = math.log10(max(float(min(value_min, value_max)), _DEBUG_CONTRIBUTION_AMOUNT_FLOOR))
    hi = math.log10(max(float(max(value_min, value_max)), max(float(min(value_min, value_max)), _DEBUG_CONTRIBUTION_AMOUNT_FLOOR) * (1.0 + _DEBUG_CONTRIBUTION_AMOUNT_FLOOR)))
    return math.pow(10.0, lo + _saturate(t) * (hi - lo))


def _debug_range_tick_value(t: float, value_min: float, value_max: float) -> float:
    lo = float(min(value_min, value_max))
    hi = float(max(value_min, value_max))
    return lo + _saturate(t) * (hi - lo)


def _histogram_log_range_from_histogram(payload: object) -> tuple[float, float] | None:
    if payload is None:
        return None
    centers = np.asarray(getattr(payload, "bin_centers_log10", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
    counts = np.asarray(getattr(payload, "counts", np.zeros((0, 0), dtype=np.int64)), dtype=np.float64)
    if counts.ndim != 2 or centers.ndim != 1 or counts.shape[1] != centers.size or centers.size == 0:
        return None
    summed = np.sum(counts, axis=0)
    finite = np.isfinite(centers) & np.isfinite(summed) & (summed > 0.0)
    if not np.any(finite):
        return None
    filtered_centers = centers[finite]
    filtered_counts = summed[finite]
    total = float(np.sum(filtered_counts))
    if total <= 0.0:
        return None
    if filtered_centers.size == 1:
        value = float(filtered_centers[0])
        return value, value
    trim_fraction = max(0.0, 0.5 * (1.0 - _HISTOGRAM_AUTO_RANGE_KEEP_FRACTION))
    cumulative = np.cumsum(filtered_counts) / total
    lo_index = int(np.searchsorted(cumulative, trim_fraction, side="left"))
    hi_index = int(np.searchsorted(cumulative, 1.0 - trim_fraction, side="left"))
    lo_index = min(max(lo_index, 0), filtered_centers.size - 1)
    hi_index = min(max(hi_index, lo_index), filtered_centers.size - 1)
    return float(filtered_centers[lo_index]), float(filtered_centers[hi_index])


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


_TRAIN_SETUP_SPECS = (
    ControlSpec("max_gaussians", "slider_int", "Max Gaussians", {"value": 1000000, "min": 1000, "max": 10000000}),
    ControlSpec("training_steps_per_frame", "slider_int", "Steps / Frame", {"value": 3, "min": 1, "max": 8}),
    ControlSpec("background_mode", "combo", "Train Background", {"value": 1, "options": _TRAIN_BACKGROUND_MODE_LABELS}),
    ControlSpec("use_target_alpha_mask", "checkbox", "Use Target Alpha Mask", {"value": False}),
    ControlSpec("train_background_color", "color_edit3", "Train BG Color", {"value": (1.0, 1.0, 1.0)}),
    ControlSpec("refinement_interval", "input_int", "Refinement Interval", {"value": 200, "step": 10, "step_fast": 50}),
    ControlSpec("refinement_growth_ratio", "input_float", "Refinement Growth", {"value": 0.05, "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
    ControlSpec("refinement_growth_start_step", "slider_int", "Start Refinement After", {"value": 500, "min": 0, "max": 30000, "max_from": "lr_schedule_steps"}),
    ControlSpec("refinement_alpha_cull_threshold", "input_float", "Refinement Alpha Cull", {"value": 1e-2, "step": 1e-5, "step_fast": 1e-4, "format": "%.6e"}),
    ControlSpec("refinement_min_contribution_percent", "input_float", "Refinement Min Contribution", {"value": 1e-05, "step": 1e-6, "step_fast": 1e-5, "format": "%.6g%%"}),
    ControlSpec("refinement_min_contribution_decay", "input_float", "Refinement Min Contribution Decay", {"value": 0.995, "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}),
    ControlSpec("refinement_opacity_mul", "input_float", "Refinement Alpha Mul", {"value": 1.0, "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}),
    ControlSpec("refinement_loss_weight", "input_float", "Refinement Loss Weight", {"value": 0.25, "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
    ControlSpec("refinement_target_edge_weight", "input_float", "Refinement Edge Weight", {"value": 0.75, "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
    ControlSpec("train_downscale_mode", "combo", "Downscale Mode", {"value": 1, "options": _TRAIN_DOWNSCALE_MODE_LABELS}),
    ControlSpec("train_subsample_factor", "combo", "Subsampling", {"value": 0, "options": _TRAIN_SUBSAMPLE_LABELS}),
    ControlSpec("train_auto_start_downscale", "slider_int", "Auto Start Downscale", {"value": 16, "min": 1, "max": 16}),
    ControlSpec("train_downscale_base_iters", "input_int", "Downscale Base Iters", {"value": 200, "step": 25, "step_fast": 100}),
    ControlSpec("train_downscale_iter_step", "input_int", "Downscale Iter Step", {"value": 50, "step": 10, "step_fast": 50}),
    ControlSpec("train_downscale_max_iters", "input_int", "Downscale Max Iters", {"value": 30000, "step": 1000, "step_fast": 5000}),
    ControlSpec("seed", "slider_int", "Shuffle Seed", {"value": 1234, "min": 0, "max": 1000000}),
    ControlSpec("init_opacity", "input_float", "Init Opacity", {"value": 0.5, "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}),
)

_TRAIN_OPTIMIZER_SPECS = (
    ControlSpec("lr_schedule_enabled", "checkbox", "Use LR Schedule", {"value": True}),
    ControlSpec("lr_scale_mul", "input_float", "LR Mul Scale", {"value": 20.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
    ControlSpec("lr_rot_mul", "input_float", "LR Mul Rotation", {"value": 1.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
    ControlSpec("lr_color_mul", "input_float", "LR Mul Color", {"value": 5.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
    ControlSpec("lr_opacity_mul", "input_float", "LR Mul Opacity", {"value": 5.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
    ControlSpec("beta1", "input_float", "Beta1", {"value": 0.9, "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
    ControlSpec("beta2", "input_float", "Beta2", {"value": 0.999, "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}),
    ControlSpec("scale_l2", "input_float", "Scale Log Reg", {"value": 0.0, "step": 1e-5, "step_fast": 1e-4, "format": "%.8f"}),
    ControlSpec("scale_abs_reg", "input_float", "Scale Abs Reg", {"value": 0.01, "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}),
    ControlSpec("sh1_reg", "input_float", "SH Reg", {"value": 0.01, "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}),
    ControlSpec("opacity_reg", "input_float", "Opacity Reg", {"value": 0.01, "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}),
    ControlSpec("density_regularizer", "input_float", "Density Reg", {"value": 0.02, "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}),
    ControlSpec("depth_ratio_grad_min", "input_float", "Depth Ratio Grad Min", {"value": 0.0, "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}),
    ControlSpec("depth_ratio_grad_max", "input_float", "Depth Ratio Grad Max", {"value": 0.1, "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}),
    ControlSpec("max_allowed_density", "input_float", "Max Density", {"value": 12.0, "step": 1e-3, "step_fast": 1e-2, "format": "%.8f"}),
    ControlSpec("position_random_step_opacity_gate_center", "input_float", "Noise Gate Center", {"value": 0.005, "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}),
    ControlSpec("position_random_step_opacity_gate_sharpness", "input_float", "Noise Gate Sharpness", {"value": 100.0, "step": 1.0, "step_fast": 10.0, "format": "%.4g"}),
    ControlSpec("max_anisotropy", "input_float", "Max Anisotropy", {"value": 32.0, "step": 0.1, "step_fast": 0.5, "format": "%.6f"}),
    ControlSpec("grad_clip", "input_float", "Grad Clip", {"value": 10.0, "step": 0.1, "step_fast": 1.0, "format": "%.4f"}),
    ControlSpec("grad_norm_clip", "input_float", "Grad Norm Clip", {"value": 10.0, "step": 0.1, "step_fast": 1.0, "format": "%.4f"}),
    ControlSpec("max_update", "input_float", "Max Update", {"value": 0.05, "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}),
)

_SCHEDULE_STAGE_SPEC_TEMPLATE = {
    "end_step": ControlSpec("schedule_stage_end_step", "slider_int", "End Step", {"value": 0, "min": 0, "max": 30000, "max_from": "lr_schedule_steps"}),
    "lr": ControlSpec("schedule_stage_lr", "input_float", "LR Target", {"value": 1e-4, "step": 1e-6, "step_fast": 1e-5, "format": "%.8f"}),
    "lr_pos_mul": ControlSpec("schedule_stage_lr_pos_mul", "input_float", "LR Mul Position", {"value": 1.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
    "lr_sh_mul": ControlSpec("schedule_stage_lr_sh_mul", "input_float", "LR Mul SH", {"value": 0.05, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
    "depth_ratio_weight": ControlSpec("schedule_stage_depth_ratio_weight", "input_float", "Depth Ratio Reg", {"value": 0.001, "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}),
    "noise_lr": ControlSpec("schedule_stage_noise_lr", "input_float", "Noise LR", {"value": 0.0, "step": 100.0, "step_fast": 1000.0, "format": "%.4g"}),
    "sh_band": ControlSpec("schedule_stage_sh_band", "combo", "SH Band", {"value": 0, "options": _SH_BAND_LABELS}),
}

_SCHEDULE_STAGE_GROUPS = {
    "Stage 0": {
        "lr": "lr_schedule_start_lr",
        "lr_pos_mul": "lr_pos_mul",
        "lr_sh_mul": "lr_sh_mul",
        "depth_ratio_weight": "depth_ratio_weight",
        "noise_lr": "position_random_step_noise_lr",
        "sh_band": "sh_band",
    },
    "Stage 1": {
        "end_step": "lr_schedule_stage1_step",
        "lr": "lr_schedule_stage1_lr",
        "lr_pos_mul": "lr_pos_stage1_mul",
        "lr_sh_mul": "lr_sh_stage1_mul",
        "depth_ratio_weight": "depth_ratio_stage1_weight",
        "noise_lr": "position_random_step_noise_stage1_lr",
        "sh_band": "sh_band_stage1",
    },
    "Stage 2": {
        "end_step": "lr_schedule_stage2_step",
        "lr": "lr_schedule_stage2_lr",
        "lr_pos_mul": "lr_pos_stage2_mul",
        "lr_sh_mul": "lr_sh_stage2_mul",
        "depth_ratio_weight": "depth_ratio_stage2_weight",
        "noise_lr": "position_random_step_noise_stage2_lr",
        "sh_band": "sh_band_stage2",
    },
    "Stage 3": {
        "end_step": "lr_schedule_steps",
        "lr": "lr_schedule_end_lr",
        "lr_pos_mul": "lr_pos_stage3_mul",
        "lr_sh_mul": "lr_sh_stage3_mul",
        "depth_ratio_weight": "depth_ratio_stage3_weight",
        "noise_lr": "position_random_step_noise_stage3_lr",
        "sh_band": "sh_band_stage3",
    },
}

_SCHEDULE_STAGE_OVERRIDES = {
    "Stage 0": {
        "lr": {"kwargs": {"value": 0.005, "step": 1e-5, "step_fast": 1e-4, "format": "%.8f"}},
        "lr_pos_mul": {"kwargs": {"value": 1.0, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}},
        "lr_sh_mul": {"kwargs": {"value": 0.05, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}},
        "depth_ratio_weight": {"kwargs": {"value": 1.0, "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}},
        "noise_lr": {"kwargs": {"value": 5e5, "step": 100.0, "step_fast": 1000.0, "format": "%.4g"}},
        "sh_band": {"kwargs": {"value": 0, "options": _SH_BAND_LABELS}},
    },
    "Stage 1": {
        "end_step": {"kwargs": {"value": 3000, "min": 0, "max": 30000, "max_from": "lr_schedule_steps"}},
        "lr": {"kwargs": {"value": 0.002, "step": 1e-6, "step_fast": 1e-5, "format": "%.8f"}},
        "lr_pos_mul": {"kwargs": {"value": 0.75, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}},
        "lr_sh_mul": {"kwargs": {"value": 0.05, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}},
        "depth_ratio_weight": {"kwargs": {"value": 0.05, "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}},
        "noise_lr": {"kwargs": {"value": 466666.6666666667, "step": 100.0, "step_fast": 1000.0, "format": "%.4g"}},
        "sh_band": {"kwargs": {"value": 1, "options": _SH_BAND_LABELS}},
    },
    "Stage 2": {
        "end_step": {"kwargs": {"value": 14000, "min": 0, "max": 30000, "max_from": "lr_schedule_steps"}},
        "lr": {"kwargs": {"value": 0.001, "step": 1e-6, "step_fast": 1e-5, "format": "%.8f"}},
        "lr_pos_mul": {"kwargs": {"value": 0.2, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}},
        "lr_sh_mul": {"kwargs": {"value": 0.05, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}},
        "depth_ratio_weight": {"kwargs": {"value": 0.01, "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}},
        "noise_lr": {"kwargs": {"value": 416666.6666666667, "step": 100.0, "step_fast": 1000.0, "format": "%.4g"}},
        "sh_band": {"kwargs": {"value": 1, "options": _SH_BAND_LABELS}},
    },
    "Stage 3": {
        "end_step": {
            "kind": "input_int",
            "label": "End Step",
            "kwargs": {"value": 30000, "step": 1000, "step_fast": 5000},
        },
        "lr": {"kwargs": {"value": 1.5e-4, "step": 1e-6, "step_fast": 1e-5, "format": "%.8f"}},
        "lr_pos_mul": {"kwargs": {"value": 0.2, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}},
        "lr_sh_mul": {"kwargs": {"value": 0.05, "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}},
        "depth_ratio_weight": {"kwargs": {"value": 0.001, "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}},
        "noise_lr": {"kwargs": {"value": 0.0, "step": 100.0, "step_fast": 1000.0, "format": "%.4g"}},
        "sh_band": {"kwargs": {"value": 1, "options": _SH_BAND_LABELS}},
    },
}


def _build_schedule_stage_specs() -> dict[str, tuple[ControlSpec, ...]]:
    groups: dict[str, tuple[ControlSpec, ...]] = {}
    for stage_label, key_map in _SCHEDULE_STAGE_GROUPS.items():
        overrides = _SCHEDULE_STAGE_OVERRIDES.get(stage_label, {})
        specs: list[ControlSpec] = []
        ordered_keys = ("lr", "lr_pos_mul", "lr_sh_mul", "depth_ratio_weight", "noise_lr", "sh_band") if stage_label == "Stage 0" else ("end_step", "lr", "lr_pos_mul", "lr_sh_mul", "depth_ratio_weight", "noise_lr", "sh_band")
        for template_key in ordered_keys:
            if template_key not in key_map:
                continue
            mapped_key = key_map[template_key]
            template = _SCHEDULE_STAGE_SPEC_TEMPLATE[template_key]
            override = overrides.get(template_key, {})
            specs.append(
                ControlSpec(
                    key=mapped_key,
                    kind=str(override.get("kind", template.kind)),
                    label=str(override.get("label", template.label)),
                    kwargs=dict(template.kwargs if "kwargs" not in override else override["kwargs"]),
                )
            )
        groups[stage_label] = tuple(specs)
    return groups


SCHEDULE_STAGE_SPECS = _build_schedule_stage_specs()

GROUP_SPECS = {
    "View": (
        ControlSpec(_INTERFACE_SCALE_KEY, "combo", "Interface Scale", {"value": _DEFAULT_INTERFACE_SCALE_INDEX, "options": tuple(label for label, _ in _INTERFACE_SCALE_OPTIONS)}),
    ),
    "Main": (
        ControlSpec("loss_debug_view", "slider_int", "Debug View", {"value": 0, "min": 0, "max": len(LOSS_DEBUG_OPTIONS) - 1}),
        ControlSpec("loss_debug_frame", "slider_int", "Debug Frame", {"value": 0, "min": 0, "max": 10000}),
        ControlSpec(_LOSS_DEBUG_ABS_SCALE_KEY, "slider_float", "Abs Diff Scale", {"value": _LOSS_DEBUG_ABS_SCALE_DEFAULT, "min": _LOSS_DEBUG_ABS_SCALE_MIN, "max": _LOSS_DEBUG_ABS_SCALE_MAX, "format": "%.3gx", "logarithmic": True}),
    ),
    "Camera": (
        ControlSpec("move_speed", "input_float", "Move Speed", {"value": 2.0, "step": 0.1, "step_fast": 1.0, "format": "%.6g"}),
        ControlSpec("fov", "slider_float", "FOV", {"value": 60.0, "min": 25.0, "max": 100.0}),
        ControlSpec("render_background_mode", "combo", "Render Background", {"value": 1, "options": _VIEWER_BACKGROUND_MODE_LABELS}),
        ControlSpec("render_background_color", "color_edit3", "Render BG Color", {"value": (0.0, 0.0, 0.0)}),
    ),
    "Train Setup": _TRAIN_SETUP_SPECS,
    "Train Optimizer": _TRAIN_OPTIMIZER_SPECS + tuple(spec for specs in SCHEDULE_STAGE_SPECS.values() for spec in specs),
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
    ControlSpec("trans_threshold", "slider_float", "Trans Threshold", {"value": 0.005, "min": 0.001, "max": 0.2, "format": "%.2e"}),
    ControlSpec("cached_raster_grad_atomic_mode", "combo", "Cached Grad Atomics", {"value": 1, "options": _CACHED_RASTER_GRAD_ATOMIC_MODE_LABELS}),
    ControlSpec("cached_raster_grad_fixed_ro_local_range", "slider_float", "Cached Grad Pos Range", {"value": 0.01, "min": 1e-4, "max": 1024.0, "format": "%.4g", "logarithmic": True}),
    ControlSpec("cached_raster_grad_fixed_scale_range", "slider_float", "Cached Grad Scale Range", {"value": 0.01, "min": 1e-4, "max": 1024.0, "format": "%.4g", "logarithmic": True}),
    ControlSpec("cached_raster_grad_fixed_quat_range", "slider_float", "Cached Grad Rot Range", {"value": 0.01, "min": 1e-4, "max": 1024.0, "format": "%.4g", "logarithmic": True}),
    ControlSpec("cached_raster_grad_fixed_color_range", "slider_float", "Cached Grad Color Range", {"value": 0.2, "min": 1e-4, "max": 2048.0, "format": "%.4g", "logarithmic": True}),
    ControlSpec("cached_raster_grad_fixed_opacity_range", "slider_float", "Cached Grad Opacity Range", {"value": 0.2, "min": 1e-4, "max": 2048.0, "format": "%.4g", "logarithmic": True}),
)

DEBUG_RENDER_SPECS = (
    ControlSpec("debug_mode", "combo", "Mode", {"value": _DEBUG_MODE_VALUES.index("normal"), "options": _DEBUG_MODE_LABELS}),
    ControlSpec("debug_sh_coeff_index", "combo", "SH Coefficient", {"value": 0, "options": _DEBUG_SH_COEFF_LABELS}),
    ControlSpec("debug_grad_norm_threshold", "input_float", "Grad Norm Threshold", {"value": _DEBUG_GRAD_NORM_THRESHOLD_DEFAULT, "step": 1e-5, "step_fast": 1e-4, "format": "%.6g"}),
    ControlSpec("debug_ellipse_thickness_px", "slider_float", "Ellipse Thickness", {"value": 4.0, "min": 0.25, "max": 8.0, "format": "%.2f px"}),
    ControlSpec("debug_clone_count_min", "input_float", "Clone Count Min", {"value": 0.0, "step": 1.0, "step_fast": 4.0, "format": "%.5g"}),
    ControlSpec("debug_clone_count_max", "input_float", "Clone Count Max", {"value": 16.0, "step": 1.0, "step_fast": 4.0, "format": "%.5g"}),
    ControlSpec("debug_density_min", "input_float", "Density Min", {"value": 0.0, "step": 0.1, "step_fast": 1.0, "format": "%.5g"}),
    ControlSpec("debug_density_max", "input_float", "Density Max", {"value": 20.0, "step": 0.1, "step_fast": 1.0, "format": "%.5g"}),
    ControlSpec("debug_contribution_min", "input_float", "Contribution Min", {"value": 0.001, "step": 1e-4, "step_fast": 1e-3, "format": "%.6g%%"}),
    ControlSpec("debug_contribution_max", "input_float", "Contribution Max", {"value": 1.0, "step": 0.1, "step_fast": 1.0, "format": "%.6g%%"}),
    ControlSpec("debug_adam_momentum_threshold", "input_float", "Adam Momentum Threshold", {"value": _DEBUG_ADAM_MOMENTUM_THRESHOLD_DEFAULT, "step": 1e-5, "step_fast": 1e-4, "format": "%.6g"}),
    ControlSpec("debug_depth_mean_min", "input_float", "Depth Mean Min", {"value": 0.0, "step": 0.1, "step_fast": 1.0, "format": "%.5g"}),
    ControlSpec("debug_depth_mean_max", "input_float", "Depth Mean Max", {"value": 10.0, "step": 0.1, "step_fast": 1.0, "format": "%.5g"}),
    ControlSpec("debug_depth_std_min", "input_float", "Depth Std Min", {"value": 0.0, "step": 0.01, "step_fast": 0.1, "format": "%.5g"}),
    ControlSpec("debug_depth_std_max", "input_float", "Depth Std Max", {"value": 0.5, "step": 0.01, "step_fast": 0.1, "format": "%.5g"}),
    ControlSpec("debug_depth_local_mismatch_min", "input_float", "Depth Local Mismatch Min", {"value": 0.0, "step": 0.01, "step_fast": 0.1, "format": "%.5g"}),
    ControlSpec("debug_depth_local_mismatch_max", "input_float", "Depth Local Mismatch Max", {"value": 0.5, "step": 0.01, "step_fast": 0.1, "format": "%.5g"}),
    ControlSpec("debug_depth_local_mismatch_smooth_radius", "input_float", "Depth Smooth Radius", {"value": 2.0, "step": 0.1, "step_fast": 1.0, "format": "%.5g"}),
    ControlSpec("debug_depth_local_mismatch_reject_radius", "input_float", "Depth Reject Radius", {"value": 4.0, "step": 0.1, "step_fast": 1.0, "format": "%.5g"}),
)

_ALL_DEFAULTS = {spec.key: spec.kwargs["value"] for group in GROUP_SPECS.values() for spec in group if "value" in spec.kwargs}
_ALL_DEFAULTS.update({spec.key: spec.kwargs["value"] for spec in RENDER_PARAM_SPECS if "value" in spec.kwargs})
_ALL_DEFAULTS.update({spec.key: spec.kwargs["value"] for spec in DEBUG_RENDER_SPECS if "value" in spec.kwargs})

_OPTIMIZER_TAB_KEYS = {
    "Schedule": ("lr_schedule_enabled", "lr_scale_mul", "lr_rot_mul", "lr_color_mul", "lr_opacity_mul"),
    "Adam": ("beta1", "beta2"),
    "Regularization": ("scale_l2", "scale_abs_reg", "sh1_reg", "opacity_reg", "density_regularizer", "depth_ratio_grad_min", "depth_ratio_grad_max", "max_allowed_density", "position_random_step_opacity_gate_center", "position_random_step_opacity_gate_sharpness", "max_anisotropy", "grad_clip", "grad_norm_clip", "max_update"),
}

_TRAIN_OPTIMIZER_SPEC_BY_KEY = {spec.key: spec for spec in GROUP_SPECS["Train Optimizer"]}


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


def _control_bound(ui: ViewerUI, spec: ControlSpec, key: str, fallback: int) -> int:
    bound_key = spec.kwargs.get(key)
    if bound_key is None:
        return fallback
    return int(ui._values.get(str(bound_key), _ALL_DEFAULTS.get(str(bound_key), fallback)))


@dataclass(slots=True)
class ToolkitState:
    loss_history: deque = field(default_factory=partial(deque, maxlen=LOSS_HISTORY_SIZE))
    fps_history: deque = field(default_factory=partial(deque, maxlen=FPS_HISTORY_SIZE))
    psnr_history: deque = field(default_factory=partial(deque, maxlen=LOSS_HISTORY_SIZE))
    step_history: deque = field(default_factory=partial(deque, maxlen=LOSS_HISTORY_SIZE))
    step_time_history: deque = field(default_factory=partial(deque, maxlen=LOSS_HISTORY_SIZE))

    def clear_plot_history(self) -> None:
        self.loss_history.clear()
        self.fps_history.clear()
        self.psnr_history.clear()
        self.step_history.clear()
        self.step_time_history.clear()


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
            export_ply=_noop,
            browse_colmap_root=_noop,
            browse_colmap_images=_noop,
            browse_colmap_depth=_noop,
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
        self._dock_layout_initialized = False
        self._viewport_dock_id = 0
        self._toolkit_dock_id = 0
        self._toolkit_window_open = True
        self._toolkit_rect = (
            max(float(width) - float(max(int(width * TOOLKIT_WIDTH_FRACTION), 280)), 0.0),
            0.0,
            float(max(int(width * TOOLKIT_WIDTH_FRACTION), 280)),
            max(float(height), 1.0),
        )
        self._viewport_rect = (0.0, 0.0, max(float(width) - self._toolkit_rect[2], 1.0), max(float(height), 1.0))
        self._viewport_content_rect = self._viewport_rect
        self._viewport_window_focused = False
        self._viewport_input_active = False
        self._set_interface_scale(_INTERFACE_SCALE_OPTIONS[_DEFAULT_INTERFACE_SCALE_INDEX][1])

    def _set_current_context(self) -> None:
        imgui.set_current_context(self.ctx)

    def reset_plot_history(self) -> None:
        self.tk.clear_plot_history()

    def _apply_theme(self):
        self._set_current_context()
        imgui.style_colors_light()
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
        _c(imgui.Col_.text, imgui.ImVec4(0.12, 0.16, 0.22, 1.00))
        _c(imgui.Col_.text_disabled, imgui.ImVec4(0.28, 0.33, 0.41, 1.00))
        _c(imgui.Col_.window_bg, imgui.ImVec4(0.965, 0.970, 0.978, 1.00))
        _c(imgui.Col_.child_bg, imgui.ImVec4(0.975, 0.980, 0.988, 0.98))
        _c(imgui.Col_.popup_bg, imgui.ImVec4(0.995, 0.997, 1.000, 0.99))
        _c(imgui.Col_.border, imgui.ImVec4(0.66, 0.71, 0.79, 0.82))
        _c(imgui.Col_.border_shadow, imgui.ImVec4(0.0, 0.0, 0.0, 0.0))
        _c(imgui.Col_.title_bg, imgui.ImVec4(0.84, 0.90, 0.96, 1.0))
        _c(imgui.Col_.title_bg_active, imgui.ImVec4(0.76, 0.85, 0.95, 1.0))
        _c(imgui.Col_.menu_bar_bg, imgui.ImVec4(0.93, 0.95, 0.98, 1.0))
        _c(imgui.Col_.header, imgui.ImVec4(0.65, 0.79, 0.95, 0.62))
        _c(imgui.Col_.header_hovered, imgui.ImVec4(0.50, 0.72, 0.94, 0.86))
        _c(imgui.Col_.header_active, imgui.ImVec4(0.43, 0.67, 0.92, 1.00))
        _c(imgui.Col_.button, imgui.ImVec4(0.32, 0.53, 0.79, 0.92))
        _c(imgui.Col_.button_hovered, imgui.ImVec4(0.25, 0.47, 0.74, 0.96))
        _c(imgui.Col_.button_active, imgui.ImVec4(0.22, 0.42, 0.68, 1.00))
        _c(imgui.Col_.tab, imgui.ImVec4(0.84, 0.90, 0.96, 1.00))
        _c(imgui.Col_.tab_hovered, imgui.ImVec4(0.67, 0.81, 0.95, 0.92))
        _c(imgui.Col_.tab_selected, imgui.ImVec4(0.52, 0.73, 0.93, 1.00))
        _c(imgui.Col_.separator, imgui.ImVec4(0.66, 0.71, 0.79, 0.82))
        _c(imgui.Col_.separator_hovered, imgui.ImVec4(0.38, 0.60, 0.85, 0.85))
        _c(imgui.Col_.plot_lines, imgui.ImVec4(0.18, 0.48, 0.82, 1.00))
        _c(imgui.Col_.frame_bg, imgui.ImVec4(0.985, 0.989, 0.995, 1.00))
        _c(imgui.Col_.frame_bg_hovered, imgui.ImVec4(0.93, 0.96, 0.99, 1.00))
        _c(imgui.Col_.frame_bg_active, imgui.ImVec4(0.86, 0.92, 0.98, 1.00))
        _c(imgui.Col_.slider_grab, imgui.ImVec4(0.34, 0.58, 0.88, 0.82))
        _c(imgui.Col_.slider_grab_active, imgui.ImVec4(0.22, 0.48, 0.82, 1.00))
        _c(imgui.Col_.check_mark, imgui.ImVec4(0.18, 0.48, 0.82, 1.00))
        _c(imgui.Col_.scrollbar_bg, imgui.ImVec4(0.92, 0.94, 0.97, 1.00))
        _c(imgui.Col_.scrollbar_grab, imgui.ImVec4(0.70, 0.77, 0.86, 0.85))
        _c(imgui.Col_.scrollbar_grab_hovered, imgui.ImVec4(0.58, 0.69, 0.82, 0.92))
        _c(imgui.Col_.scrollbar_grab_active, imgui.ImVec4(0.46, 0.61, 0.78, 1.00))
        _c(imgui.Col_.resize_grip, imgui.ImVec4(0.52, 0.70, 0.92, 0.28))
        _c(imgui.Col_.resize_grip_hovered, imgui.ImVec4(0.40, 0.61, 0.88, 0.72))
        _c(imgui.Col_.resize_grip_active, imgui.ImVec4(0.28, 0.51, 0.82, 0.95))
        _c(imgui.Col_.text_selected_bg, imgui.ImVec4(0.56, 0.77, 0.96, 0.42))

    def _configure_default_font(self) -> None:
        io = imgui.get_io()
        atlas = io.fonts
        atlas.clear()
        font_path = _default_font_path()
        io.font_default = atlas.add_font_from_file_ttf(str(font_path), _FONT_ATLAS_SIZE_PX) if font_path is not None else atlas.add_font_default()
        markdown_font_base_path = _markdown_font_base_path()
        if markdown_font_base_path is not None:
            try:
                hello_imgui.set_assets_folder(str(_imgui_bundle_assets_path()))
                markdown_options = imgui_md.MarkdownOptions()
                markdown_options.font_options.font_base_path = str(markdown_font_base_path)
                markdown_options.font_options.regular_size = _FONT_ATLAS_SIZE_PX
                imgui_md.de_initialize_markdown()
                imgui_md.initialize_markdown(markdown_options)
                imgui_md.get_font_loader_function()()
            except Exception:
                pass
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
        handled = bool(simgui.handle_keyboard_event(event))
        return _should_capture_keyboard_for_ui(handled, self._viewport_input_active, bool(imgui.get_io().want_text_input))

    def handle_mouse_event(self, event) -> bool:
        if not self._alive:
            return False
        self._set_current_context()
        handled = bool(simgui.handle_mouse_event(event))
        pos = getattr(event, "pos", None)
        point = None if pos is None else (float(pos.x), float(pos.y))
        inside_viewport = _rect_contains(self._viewport_content_rect, point)
        event_type = getattr(event, "type", None)
        if inside_viewport and event_type in (spy.MouseEventType.button_down, spy.MouseEventType.move, spy.MouseEventType.scroll):
            self._viewport_input_active = True
        elif event_type == spy.MouseEventType.button_down and not inside_viewport:
            self._viewport_input_active = False
        return False if inside_viewport else handled

    def viewport_size(self) -> tuple[int, int]:
        return _clamp_viewport_size(self._viewport_content_rect[2], self._viewport_content_rect[3])

    def render(self, ui: ViewerUI, surface_texture: spy.Texture, command_encoder: spy.CommandEncoder, viewport_texture: spy.Texture | None = None) -> None:
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
        self._draw_viewport_window(ui, viewport_texture, width, height)
        self._draw_debug_colorbar(ui)
        self._draw_histogram_window(ui)
        self._draw_training_views_window(ui)
        imgui.render()
        draw_data = imgui.get_draw_data()
        self._frame_textures = simgui.sync_draw_data_textures(self.device, self.renderer, draw_data)
        simgui.render_imgui_draw_data(self.renderer, draw_data, surface_texture, command_encoder)

    def _draw_dockspace(self) -> None:
        viewport = imgui.get_main_viewport()
        dockspace_id = int(imgui.dock_space_over_viewport(viewport=viewport, flags=_DOCKSPACE_FLAGS))
        if dockspace_id != self._dockspace_id:
            self._dockspace_id = dockspace_id
            self._dock_layout_initialized = False
        self._ensure_default_dock_layout(viewport)

    def _ensure_default_dock_layout(self, viewport: object) -> None:
        if self._dock_layout_initialized or self._dockspace_id == 0:
            return
        imgui.internal.dock_builder_remove_node(self._dockspace_id)
        root_id = int(imgui.internal.dock_builder_add_node(self._dockspace_id, _DOCKSPACE_FLAGS))
        imgui.internal.dock_builder_set_node_size(root_id, viewport.work_size)
        split_ids = tuple(int(node_id) for node_id in imgui.internal.dock_builder_split_node_py(root_id, imgui.Dir_.right, TOOLKIT_WIDTH_FRACTION))
        leaf_ids = tuple(dict.fromkeys(node_id for node_id in split_ids if node_id != root_id))
        central_node = imgui.internal.dock_builder_get_central_node(root_id)
        viewport_dock_id = 0 if central_node is None else int(central_node.id_)
        toolkit_dock_id = next((node_id for node_id in leaf_ids if node_id != viewport_dock_id), self._dockspace_id)
        if viewport_dock_id == 0:
            viewport_dock_id = next((node_id for node_id in leaf_ids if node_id != toolkit_dock_id), self._dockspace_id)
        imgui.internal.dock_builder_dock_window(_TOOLKIT_WINDOW_NAME, toolkit_dock_id)
        imgui.internal.dock_builder_dock_window(_VIEWPORT_WINDOW_NAME, viewport_dock_id)
        imgui.internal.dock_builder_finish(root_id)
        self._viewport_dock_id = viewport_dock_id
        self._toolkit_dock_id = toolkit_dock_id
        self._dock_layout_initialized = True

    def _draw_panel(self, ui: ViewerUI, width: int, height: int) -> None:
        panel_x, panel_y, panel_width, panel_height = _panel_rect(width, height, self._menu_bar_height)
        imgui.set_next_window_pos(imgui.ImVec2(panel_x, panel_y), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(panel_width, panel_height), imgui.Cond_.first_use_ever.value)
        if self._toolkit_dock_id != 0:
            imgui.set_next_window_dock_id(self._toolkit_dock_id, imgui.Cond_.always.value)
        elif self._dockspace_id != 0:
            imgui.set_next_window_dock_id(self._dockspace_id, imgui.Cond_.first_use_ever.value)
        flags = imgui.WindowFlags_.no_collapse.value
        opened, self._toolkit_window_open = imgui.begin(_TOOLKIT_WINDOW_NAME, self._toolkit_window_open, flags=flags)
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

    def _draw_viewport_window(self, ui: ViewerUI, viewport_texture: spy.Texture | None, width: int, height: int) -> None:
        viewport_x = 0.0
        viewport_y = self._menu_bar_height
        viewport_width = max(float(width) - self._toolkit_rect[2], 1.0)
        viewport_height = max(float(height) - viewport_y, 1.0)
        imgui.set_next_window_pos(imgui.ImVec2(viewport_x, viewport_y), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(viewport_width, viewport_height), imgui.Cond_.first_use_ever.value)
        if self._viewport_dock_id != 0:
            imgui.set_next_window_dock_id(self._viewport_dock_id, imgui.Cond_.always.value)
        elif self._dockspace_id != 0:
            imgui.set_next_window_dock_id(self._dockspace_id, imgui.Cond_.always.value)
        flags = imgui.WindowFlags_.no_title_bar.value | imgui.WindowFlags_.no_collapse.value | imgui.WindowFlags_.no_scrollbar.value | imgui.WindowFlags_.no_scroll_with_mouse.value
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0.0, 0.0))
        opened = imgui.begin(_VIEWPORT_WINDOW_NAME, flags=flags)[0]
        imgui.pop_style_var()
        if opened:
            self._viewport_window_focused = bool(imgui.is_window_focused(int(imgui.FocusedFlags_.root_and_child_windows)))
            self._viewport_input_active = self._viewport_input_active or self._viewport_window_focused
            pos = imgui.get_window_pos()
            size = imgui.get_window_size()
            cursor = imgui.get_cursor_screen_pos()
            available = imgui.get_content_region_avail()
            content_x = float(cursor.x)
            content_y = float(cursor.y)
            content_width = max(float(available.x), 1.0)
            content_height = max(float(available.y), 1.0)
            self._viewport_rect = (float(pos.x), float(pos.y), float(size.x), float(size.y))
            self._viewport_content_rect = (content_x, content_y, content_width, content_height)
            image_size = imgui.ImVec2(content_width, content_height)
            if viewport_texture is not None:
                imgui.image(simgui.texture_ref(viewport_texture), image_size)
            else:
                imgui.dummy(image_size)
                draw_list = imgui.get_window_draw_list()
                draw_list.add_rect_filled(cursor, imgui.ImVec2(cursor.x + content_width, cursor.y + content_height), _color_u32(0.04, 0.045, 0.055, 1.0))
                label = "Load a scene to populate the viewport"
                text_size = imgui.calc_text_size(label)
                draw_list.add_text(
                    imgui.ImVec2(cursor.x + 0.5 * (content_width - float(text_size.x)), cursor.y + 0.5 * (content_height - float(text_size.y))),
                    _color_u32(0.72, 0.76, 0.82, 0.95),
                    label,
                )
            self._draw_viewport_camera_overlays(ui, cursor)
            overlay_origin = self._draw_viewport_view_menu(ui, cursor)
            self._draw_viewport_debug_overlay(ui, overlay_origin)
        else:
            self._viewport_window_focused = False
            self._viewport_input_active = False
        imgui.end()

    def _draw_viewport_view_menu(self, ui: ViewerUI, image_origin: imgui.ImVec2) -> imgui.ImVec2:
        style = imgui.get_style()
        scale = self._interface_scale_factor(ui)
        label = "View Mode"
        camera_label = "Cameras On" if bool(ui._values.get("show_camera_overlays", True)) else "Cameras Off"
        text_label = "Labels On" if bool(ui._values.get("show_camera_labels", False)) else "Labels Off"
        training_label = "Training Cameras On" if bool(ui._values.get("show_training_cameras", False)) else "Training Cameras Off"
        sh_band = min(max(int(ui._values.get("_viewport_sh_band", ui._values.get("sh_band", 0))), 0), len(_SH_BAND_LABELS) - 1)
        sh_label = _SH_BAND_LABELS[sh_band]
        label_size = imgui.calc_text_size(label)
        button_width = float(label_size.x) + 2.0 * float(style.frame_padding.x)
        button_pos = imgui.ImVec2(float(image_origin.x) + _VIEWPORT_OVERLAY_MARGIN * scale, float(image_origin.y) + _VIEWPORT_OVERLAY_MARGIN * scale)
        button_height = float(label_size.y) + 2.0 * float(style.frame_padding.y)
        current = min(max(int(ui._values.get("debug_mode", 0)), 0), len(_DEBUG_MODE_LABELS) - 1)
        current_label = _DEBUG_MODE_LABELS[current]
        imgui.push_id("viewport_view")
        imgui.set_cursor_screen_pos(button_pos)
        opened = _imgui_opened(imgui.small_button(label))
        imgui.same_line(0.0, 10.0 * scale)
        if _imgui_opened(imgui.small_button(camera_label)):
            ui._values["show_camera_overlays"] = not bool(ui._values.get("show_camera_overlays", True))
        imgui.same_line(0.0, 10.0 * scale)
        if _imgui_opened(imgui.small_button(text_label)):
            ui._values["show_camera_labels"] = not bool(ui._values.get("show_camera_labels", False))
        imgui.same_line(0.0, 10.0 * scale)
        if _imgui_opened(imgui.small_button(training_label)):
            ui._values["show_training_cameras"] = not bool(ui._values.get("show_training_cameras", False))
        imgui.same_line(0.0, 10.0 * scale)
        sh_label_width = max(float(imgui.calc_text_size(option).x) for option in _SH_BAND_LABELS)
        sh_combo_width = sh_label_width + 28.0 * scale
        sh_frame_padding = imgui.ImVec2(max(4.0 * scale, 1.0), max(1.0 * scale, 1.0))
        imgui.set_next_item_width(sh_combo_width)
        imgui.push_style_var(imgui.StyleVar_.frame_padding.value, sh_frame_padding)
        if imgui.begin_combo("##viewport_sh_band", sh_label):
            for idx, option in enumerate(_SH_BAND_LABELS):
                selected = idx == sh_band
                if imgui.selectable(option, selected)[0]:
                    ui._values["_viewport_sh_band"] = idx
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        imgui.pop_style_var()
        imgui.same_line(0.0, 10.0 * scale)
        label_pos = imgui.get_cursor_screen_pos()
        current_label_size = imgui.calc_text_size(current_label)
        badge_pad_x = 6.0 * scale
        badge_pad_y = 2.0 * scale
        draw_list = imgui.get_window_draw_list()
        draw_list.add_rect_filled(
            imgui.ImVec2(label_pos.x - badge_pad_x, label_pos.y - badge_pad_y),
            imgui.ImVec2(label_pos.x + float(current_label_size.x) + badge_pad_x, label_pos.y + float(current_label_size.y) + badge_pad_y),
            _color_u32(0.06, 0.08, 0.12, 0.68),
            4.0 * scale,
        )
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(0.985, 0.992, 1.000, 1.00))
        imgui.text_unformatted(current_label)
        imgui.pop_style_color()
        if opened:
            imgui.set_next_window_pos(imgui.ImVec2(button_pos.x, button_pos.y + button_height + _VIEWPORT_OVERLAY_MARGIN * scale), imgui.Cond_.appearing.value)
            imgui.open_popup("viewport_view_popup")
        if _imgui_opened(imgui.begin_popup("viewport_view_popup")):
            for idx, label in enumerate(_DEBUG_MODE_LABELS):
                selected = idx == current
                if _imgui_opened(imgui.selectable(label, selected)):
                    ui._values["debug_mode"] = idx
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_popup()
        imgui.pop_id()
        return imgui.ImVec2(button_pos.x, button_pos.y + button_height + _VIEWPORT_OVERLAY_MARGIN * scale)

    def _training_camera_debug_section_height(self, ui: ViewerUI) -> float:
        line_height = float(imgui.get_text_line_height_with_spacing())
        frame_height = float(imgui.get_frame_height())
        spacing_y = float(imgui.get_style().item_spacing.y)
        height = frame_height + spacing_y + frame_height + spacing_y + line_height
        if LOSS_DEBUG_OPTIONS[min(max(int(ui._values.get("loss_debug_view", 0)), 0), len(LOSS_DEBUG_OPTIONS) - 1)][0] == "abs_diff":
            height += frame_height + spacing_y
        if ui._texts.get("loss_debug_frame", ""):
            height += line_height + spacing_y
        return height

    def _draw_training_camera_debug_controls(self, ui: ViewerUI) -> None:
        view_idx = min(max(int(ui._values.get("loss_debug_view", 0)), 0), len(LOSS_DEBUG_OPTIONS) - 1)
        if imgui.begin_combo("##training_camera_view", LOSS_DEBUG_OPTIONS[view_idx][1]):
            for idx, (_key, label) in enumerate(LOSS_DEBUG_OPTIONS):
                if imgui.selectable(label, idx == view_idx)[0]:
                    ui._values["loss_debug_view"] = idx
                if idx == view_idx:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        if LOSS_DEBUG_OPTIONS[view_idx][0] == "abs_diff":
            self._draw_control(ui, next(spec for spec in GROUP_SPECS["Main"] if spec.key == _LOSS_DEBUG_ABS_SCALE_KEY), compact=True)
        frame_max = max(int(ui._values.get("_loss_debug_frame_max", 0)), 0)
        changed, val = imgui.slider_int("##training_camera_frame", int(ui._values.get("loss_debug_frame", 0)), 0, frame_max)
        if changed:
            ui._values["loss_debug_frame"] = val
        imgui.text_disabled(ui._texts.get("loss_debug_view", ""))
        frame_text = ui._texts.get("loss_debug_frame", "")
        if frame_text:
            imgui.text_disabled(_status_suffix(frame_text))

    def _draw_viewport_camera_overlays(self, ui: ViewerUI, image_origin: imgui.ImVec2) -> None:
        if not bool(ui._values.get("show_camera_overlays", True)):
            return
        overlays = tuple(ui._values.get("_training_view_overlay_segments", ()))
        if len(overlays) == 0:
            return
        draw_list = imgui.get_window_draw_list()
        base_x = float(image_origin.x)
        base_y = float(image_origin.y)
        scale = self._interface_scale_factor(ui)
        show_labels = bool(ui._values.get("show_camera_labels", False))
        label_pad_x = 5.0 * scale
        label_pad_y = 3.0 * scale
        label_font = imgui.get_font()
        label_font_size = float(imgui.get_font_size()) * 0.9
        for near_points, far_points, connectors, label_anchor, label_text, color, thickness in overlays:
            color_u32 = _color_u32(*color)
            near_polyline = [imgui.ImVec2(base_x + float(x), base_y + float(y)) for x, y in near_points]
            far_polyline = [imgui.ImVec2(base_x + float(x), base_y + float(y)) for x, y in far_points]
            if len(near_polyline) >= 2:
                draw_list.add_polyline(near_polyline, color_u32, imgui.ImDrawFlags_.closed.value, float(thickness))
            if len(far_polyline) >= 2:
                draw_list.add_polyline(far_polyline, color_u32, imgui.ImDrawFlags_.closed.value, float(thickness))
            for x0, y0, x1, y1 in connectors:
                draw_list.add_line(
                    imgui.ImVec2(base_x + float(x0), base_y + float(y0)),
                    imgui.ImVec2(base_x + float(x1), base_y + float(y1)),
                    color_u32,
                    float(thickness),
                )
            if show_labels and str(label_text):
                label_pos = imgui.ImVec2(base_x + float(label_anchor[0]) + 6.0 * scale, base_y + float(label_anchor[1]) - 18.0 * scale)
                label_size_raw = imgui.calc_text_size(str(label_text))
                label_size = imgui.ImVec2(float(label_size_raw.x) * 0.9, float(label_size_raw.y) * 0.9)
                draw_list.add_rect_filled(
                    imgui.ImVec2(label_pos.x - label_pad_x, label_pos.y - label_pad_y),
                    imgui.ImVec2(label_pos.x + float(label_size.x) + label_pad_x, label_pos.y + float(label_size.y) + label_pad_y),
                    _color_u32(0.04, 0.05, 0.07, 0.74),
                    4.0 * scale,
                )
                draw_list.add_text(label_font, label_font_size, label_pos, _color_u32(0.98, 0.99, 1.0, 1.0), str(label_text))

    def _draw_viewport_debug_overlay(self, ui: ViewerUI, overlay_origin: imgui.ImVec2) -> None:
        debug_mode = _DEBUG_MODE_VALUES[min(max(int(ui._values.get("debug_mode", 0)), 0), len(_DEBUG_MODE_VALUES) - 1)]
        show_training_cameras = bool(ui._values.get("show_training_cameras", False))
        control_keys = tuple(key for key in _renderer_debug_control_keys(debug_mode) if key != "debug_mode")
        if len(control_keys) == 0 and not show_training_cameras:
            return
        view_x0, view_y0, view_width, view_height = self._viewport_content_rect
        if view_width <= 1.0 or view_height <= 1.0:
            return
        scale = self._interface_scale_factor(ui)
        frame_height = float(imgui.get_frame_height())
        spacing_y = float(imgui.get_style().item_spacing.y)
        label_height = float(imgui.calc_text_size("Ag").y)
        padding = _VIEWPORT_OVERLAY_PADDING * scale
        width = min(_VIEWPORT_OVERLAY_WIDTH * scale, max(view_width - 2.0 * _VIEWPORT_OVERLAY_MARGIN * scale, _VIEWPORT_OVERLAY_MIN_WIDTH * scale))
        height = 2.0 * padding
        if show_training_cameras:
            height += self._training_camera_debug_section_height(ui)
            if len(control_keys) > 0:
                height += spacing_y + 2.0 * scale
        if len(control_keys) > 0:
            height += len(control_keys) * (label_height + frame_height + 2.0 * spacing_y)
        height = max(height, _VIEWPORT_OVERLAY_MIN_HEIGHT * scale)
        max_height = view_y0 + view_height - overlay_origin.y - _VIEWPORT_OVERLAY_MARGIN * scale
        if max_height <= 0.0:
            return
        height = min(height, max_height)
        imgui.push_style_color(imgui.Col_.child_bg.value, imgui.ImVec4(0.982, 0.987, 0.994, 0.94))
        imgui.push_style_color(imgui.Col_.border.value, imgui.ImVec4(0.65, 0.72, 0.80, 0.92))
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(padding, padding))
        imgui.set_cursor_screen_pos(overlay_origin)
        if _imgui_opened(imgui.begin_child("##viewport_debug_overlay", imgui.ImVec2(width, height), imgui.ChildFlags_.borders.value)):
            imgui.push_item_width(-1.0)
            if show_training_cameras:
                self._draw_training_camera_debug_controls(ui)
                if len(control_keys) > 0:
                    imgui.separator()
            for key in control_keys:
                self._draw_control(ui, next(spec for spec in DEBUG_RENDER_SPECS if spec.key == key), compact=True)
            imgui.pop_item_width()
        imgui.end_child()
        imgui.pop_style_var()
        imgui.pop_style_color(2)

    def _draw_debug_colorbar(self, ui: ViewerUI) -> None:
        mode = _debug_colorbar_mode(ui)
        if mode is None:
            return
        view_x0, view_y0, view_width, view_height = self._viewport_content_rect
        if view_width <= 1.0 or view_height <= 1.0:
            return
        draw_list = imgui.get_foreground_draw_list()
        scale = self._interface_scale_factor(ui)
        margin = _DEBUG_COLORBAR_MARGIN * scale
        side_pad = _DEBUG_COLORBAR_SIDE_PAD * scale
        top_pad = _DEBUG_COLORBAR_TOP_PAD * scale
        bottom_pad = _DEBUG_COLORBAR_BOTTOM_PAD * scale
        bar_height = _DEBUG_COLORBAR_HEIGHT * scale
        available_width = max(view_width - 2.0 * margin, 120.0)
        bar_width = min(_DEBUG_COLORBAR_MAX_WIDTH * scale, max(_DEBUG_COLORBAR_MIN_WIDTH * scale, available_width * 0.85))
        bar_width = max(min(bar_width, available_width - 2.0 * side_pad), 80.0)
        box_width = bar_width + 2.0 * side_pad
        box_height = top_pad + bar_height + bottom_pad
        box_x = max(view_x0 + 0.5 * (view_width - box_width), view_x0 + margin)
        box_y = max(view_y0 + view_height - margin - box_height, view_y0 + 4.0 * scale)
        x0 = box_x + side_pad
        y0 = box_y + top_pad
        x1 = x0 + bar_width
        y1 = y0 + bar_height
        draw_list.add_rect_filled(
            imgui.ImVec2(box_x, box_y),
            imgui.ImVec2(box_x + box_width, box_y + box_height),
            _color_u32(0.985, 0.989, 0.995, 0.94),
            8.0 * scale,
        )
        draw_list.add_text(imgui.ImVec2(box_x + 0.5 * side_pad, box_y + 0.25 * top_pad), _color_u32(0.20, 0.26, 0.34, 0.95), self._debug_colorbar_title(mode))
        self._draw_debug_colorbar_gradient(draw_list, x0, y0, x1, y1)
        draw_list.add_rect(imgui.ImVec2(x0, y0), imgui.ImVec2(x1, y1), _color_u32(0.50, 0.60, 0.72, 0.95), 2.0 * scale, 0, max(scale, 1.0))
        self._draw_debug_colorbar_ticks(draw_list, mode, x0, y0, x1, y1, ui, scale)

    def _draw_debug_colorbar_gradient(self, draw_list: object, x0: float, y0: float, x1: float, y1: float) -> None:
        width = max(x1 - x0, 1.0)
        for idx in range(_DEBUG_COLORBAR_STEPS):
            t0 = idx / _DEBUG_COLORBAR_STEPS
            t1 = (idx + 1) / _DEBUG_COLORBAR_STEPS
            rgb = _jet_colormap(0.5 * (t0 + t1))
            draw_list.add_rect_filled(
                imgui.ImVec2(x0 + t0 * width, y0),
                imgui.ImVec2(x0 + t1 * width, y1),
                _color_u32(*rgb),
            )

    def _draw_debug_colorbar_ticks(self, draw_list: object, mode: str, x0: float, y0: float, x1: float, y1: float, ui: ViewerUI, scale: float) -> None:
        for idx in range(_DEBUG_COLORBAR_TICKS):
            t = idx / max(_DEBUG_COLORBAR_TICKS - 1, 1)
            x = x0 + t * (x1 - x0)
            label = self._debug_colorbar_tick_label(mode, t, ui)
            label_size = imgui.calc_text_size(label)
            draw_list.add_line(imgui.ImVec2(x, y1 + 2.0 * scale), imgui.ImVec2(x, y1 + 10.0 * scale), _color_u32(0.44, 0.53, 0.64, 0.9), max(scale, 1.0))
            draw_list.add_text(imgui.ImVec2(x - 0.5 * float(label_size.x), y1 + 12.0 * scale), _color_u32(0.22, 0.28, 0.36, 0.95), label)

    def _debug_colorbar_title(self, mode: str) -> str:
        return {
            "processed_count": "Processed Count",
            "clone_count": "Clone Count",
            "splat_density": "Splat Density",
            "splat_spatial_density": "Spatial Density",
            "splat_screen_density": "Screen Density",
            "contribution_amount": "Contribution Amount",
            "adam_momentum": "Adam Momentum",
            "depth_mean": "Depth Mean",
            "depth_std": "Depth Std",
            "depth_local_mismatch": "Depth Local Mismatch",
            "grad_norm": "Grad Norm",
        }.get(mode, "Debug")

    def _debug_colorbar_tick_label(self, mode: str, t: float, ui: ViewerUI) -> str:
        if mode == "processed_count":
            value = _processed_count_tick_value(t, 32768)
            return f"{int(round(value)):,}"
        if mode == "grad_norm":
            threshold = float(ui._values.get("debug_grad_norm_threshold", _DEBUG_GRAD_NORM_THRESHOLD_DEFAULT))
            return f"{_threshold_band_tick_value(t, threshold):.1e}"
        if mode == "clone_count":
            return f"{_debug_range_tick_value(t, float(ui._values.get('debug_clone_count_min', 0.0)), float(ui._values.get('debug_clone_count_max', 16.0))):.3g}"
        if mode in ("splat_density", "splat_spatial_density", "splat_screen_density"):
            return f"{_debug_range_tick_value(t, float(ui._values.get('debug_density_min', 0.0)), float(ui._values.get('debug_density_max', 20.0))):.3g}"
        if mode == "contribution_amount":
            return f"{_contribution_amount_tick_value(t, float(ui._values.get('debug_contribution_min', 0.001)), float(ui._values.get('debug_contribution_max', 1.0))):.1e}"
        if mode == "adam_momentum":
            threshold = float(ui._values.get("debug_grad_norm_threshold", _DEBUG_GRAD_NORM_THRESHOLD_DEFAULT))
            return f"{_threshold_band_tick_value(t, threshold):.1e}"
        if mode == "depth_mean":
            return f"{_debug_range_tick_value(t, float(ui._values.get('debug_depth_mean_min', 0.0)), float(ui._values.get('debug_depth_mean_max', 10.0))):.3g}"
        if mode == "depth_std":
            return f"{_debug_range_tick_value(t, float(ui._values.get('debug_depth_std_min', 0.0)), float(ui._values.get('debug_depth_std_max', 0.5))):.3g}"
        if mode == "depth_local_mismatch":
            return f"{_debug_range_tick_value(t, float(ui._values.get('debug_depth_local_mismatch_min', 0.0)), float(ui._values.get('debug_depth_local_mismatch_max', 0.5))):.3g}"
        return ""

    def _draw_main_menu_bar(self, ui: ViewerUI) -> float:
        if not imgui.begin_main_menu_bar():
            return 0.0
        if imgui.begin_menu("File"):
            if _menu_item("Load PLY..."):
                self.callbacks.load_ply()
            if _menu_item("Export PLY...", enabled=bool(ui._values.get("_can_export_ply", False))):
                self.callbacks.export_ply()
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
            selected = bool(ui._values.get("show_training_views", False))
            if _menu_item("Training Views", selected=selected):
                ui._values["show_training_views"] = not selected
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
        self._dock_tool_window(imgui.Cond_.appearing.value)
        imgui.set_next_window_pos(imgui.ImVec2(24.0, self._menu_bar_height + 24.0), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(420.0, 220.0), imgui.Cond_.first_use_ever.value)
        opened, self._show_about = imgui.begin("About", True)
        if opened:
            _draw_markdown_text(self._about_text)
        imgui.end()

    def _draw_documentation_window(self) -> None:
        if not self._show_docs:
            return
        self._dock_tool_window(imgui.Cond_.appearing.value)
        imgui.set_next_window_pos(imgui.ImVec2(40.0, self._menu_bar_height + 32.0), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(760.0, 620.0), imgui.Cond_.first_use_ever.value)
        opened, self._show_docs = imgui.begin("Documentation", True)
        if opened:
            imgui.text_disabled("Local viewer documentation")
            imgui.separator()
            if imgui.begin_child("##docs_scroll", imgui.ImVec2(0.0, 0.0), imgui.ChildFlags_.borders.value):
                _draw_markdown_text(self._documentation_text)
                imgui.end_child()
        imgui.end()

    def close_colmap_import_window(self) -> None:
        self._show_colmap_import = False

    def _dock_tool_window(self, cond: int) -> None:
        if self._toolkit_dock_id != 0: imgui.set_next_window_dock_id(self._toolkit_dock_id, cond)

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
        self._dock_tool_window(imgui.Cond_.appearing.value)
        imgui.set_next_window_pos(imgui.ImVec2(56.0, self._menu_bar_height + 40.0), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(540.0, 0.0), imgui.Cond_.first_use_ever.value)
        opened, self._show_colmap_import = imgui.begin("COLMAP Import", True)
        import_active = bool(ui._values.get("_colmap_import_active", False))
        if import_active and not self._show_colmap_import:
            self._show_colmap_import = True
        if opened:
            imgui.text_wrapped("Select the dataset root, verify the RGB image folder, optionally provide a depth folder, choose import-time downscale and initialization mode, then import the dataset.")
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
            self._draw_import_path_selector(ui, label="Depth Folder", key="colmap_depth_root", button_label="Browse Depth Folder...", callback=self.callbacks.browse_colmap_depth)
            if imgui.is_item_hovered():
                imgui.set_item_tooltip("Optional root containing 16-bit depth PNGs matched to RGB images by relative path stem.")
            camera_rows = tuple(ui._values.get("_colmap_camera_rows", ()))
            if len(camera_rows) > 0:
                imgui.spacing()
                selected_camera_ids = tuple(int(camera_id) for camera_id in ui._values.get("colmap_selected_camera_ids", ()))
                camera_ids = tuple(int(row["camera_id"]) for row in camera_rows)
                selected = {camera_id for camera_id in selected_camera_ids if camera_id in camera_ids}
                selected_frame_count = sum(int(row["frame_count"]) for row in camera_rows if int(row["camera_id"]) in selected)
                total_frame_count = sum(int(row["frame_count"]) for row in camera_rows)
                imgui.text_disabled(f"Cameras: {len(selected)}/{len(camera_rows)} selected | Frames: {selected_frame_count}/{total_frame_count}")
                if imgui.button("All Cameras"):
                    selected = set(camera_ids)
                imgui.same_line()
                if imgui.button("No Cameras"):
                    selected.clear()
                table_height = min(max(88.0, 28.0 * float(len(camera_rows)) + 8.0), 180.0)
                if imgui.begin_child("##colmap_cameras", imgui.ImVec2(0.0, table_height), True):
                    flags = (
                        imgui.TableFlags_.row_bg.value
                        | imgui.TableFlags_.borders.value
                        | imgui.TableFlags_.resizable.value
                        | imgui.TableFlags_.scroll_x.value
                        | imgui.TableFlags_.scroll_y.value
                    )
                    if imgui.begin_table("##colmap_camera_table", 7, flags):
                        imgui.table_setup_column("Use", imgui.TableColumnFlags_.width_fixed.value, 36.0)
                        imgui.table_setup_column("Id", imgui.TableColumnFlags_.width_fixed.value, 42.0)
                        imgui.table_setup_column("Model", imgui.TableColumnFlags_.width_fixed.value, 132.0)
                        imgui.table_setup_column("Frames", imgui.TableColumnFlags_.width_fixed.value, 54.0)
                        imgui.table_setup_column("Res", imgui.TableColumnFlags_.width_fixed.value, 86.0)
                        imgui.table_setup_column("Focal", imgui.TableColumnFlags_.width_fixed.value, 120.0)
                        imgui.table_setup_column("Principal / Dist", imgui.TableColumnFlags_.width_stretch.value)
                        imgui.table_headers_row()
                        for row in camera_rows:
                            camera_id = int(row["camera_id"])
                            imgui.table_next_row()
                            imgui.table_next_column()
                            changed, value = imgui.checkbox(f"##colmap_camera_{camera_id}", camera_id in selected)
                            if changed:
                                if value:
                                    selected.add(camera_id)
                                else:
                                    selected.discard(camera_id)
                            imgui.table_next_column()
                            imgui.text_unformatted(str(camera_id))
                            imgui.table_next_column()
                            imgui.text_unformatted(str(row["model_name"]))
                            imgui.table_next_column()
                            imgui.text_unformatted(str(row["frame_count"]))
                            imgui.table_next_column()
                            imgui.text_unformatted(str(row["resolution_text"]))
                            imgui.table_next_column()
                            imgui.text_unformatted(str(row["focal_text"]))
                            imgui.table_next_column()
                            imgui.text_unformatted(f"{row['principal_text']} | {row['distortion_text']}")
                        imgui.end_table()
                    imgui.end_child()
                ui._values["colmap_selected_camera_ids"] = tuple(camera_id for camera_id in camera_ids if camera_id in selected)
            imgui.spacing()
            downscale_idx = max(0, min(int(ui._values.get("colmap_image_downscale_mode", 1)), len(_COLMAP_IMAGE_DOWNSCALE_LABELS) - 1))
            if imgui.begin_combo("Image Downscale", _COLMAP_IMAGE_DOWNSCALE_LABELS[downscale_idx]):
                for idx, label in enumerate(_COLMAP_IMAGE_DOWNSCALE_LABELS):
                    selected = idx == downscale_idx
                    if imgui.selectable(label, selected)[0]:
                        ui._values["colmap_image_downscale_mode"] = idx
                        downscale_idx = idx
                    if selected:
                        imgui.set_item_default_focus()
                imgui.end_combo()
            if downscale_idx == 1:
                changed, value = imgui.input_int("Max Size", int(ui._values.get("colmap_image_max_size", 2048)), 64, 256)
                if changed:
                    ui._values["colmap_image_max_size"] = max(int(value), 1)
                if imgui.is_item_hovered():
                    imgui.set_item_tooltip("Clamp imported training images so their longer side is at most this size while preserving aspect ratio. The importer never upscales.")
            elif downscale_idx == 2:
                changed, value = imgui.drag_float(
                    "Scale Factor",
                    float(ui._values.get("colmap_image_scale", 1.0)),
                    0.01,
                    1e-3,
                    1.0,
                    "%.4f",
                    imgui.SliderFlags_.logarithmic.value,
                )
                if changed:
                    ui._values["colmap_image_scale"] = min(max(float(value), 1e-3), 1.0)
                if imgui.is_item_hovered():
                    imgui.set_item_tooltip("Uniform scale applied to imported training images. Both axes stay proportional and the importer never upscales.")
            else:
                imgui.text_disabled("Imported images keep their source resolution.")
            imgui.spacing()
            changed, value = imgui.checkbox("Use Alpha Mask", bool(ui._values.get("use_target_alpha_mask", False)))
            if changed:
                ui._values["use_target_alpha_mask"] = bool(value)
            if imgui.is_item_hovered():
                imgui.set_item_tooltip("If imported images have alpha, transparent pixels are masked out of per-pixel training loss and gradients.")
            imgui.spacing()
            init_labels = _colmap_init_mode_labels(_valid_depth_root_text(ui._values.get("colmap_depth_root", "")))
            mode_idx = max(0, min(int(ui._values.get("colmap_init_mode", 0)), len(init_labels) - 1))
            if int(ui._values.get("colmap_init_mode", 0)) != mode_idx:
                ui._values["colmap_init_mode"] = mode_idx
            if imgui.begin_combo("Initialization", init_labels[mode_idx]):
                for idx, label in enumerate(init_labels):
                    selected = idx == mode_idx
                    if imgui.selectable(label, selected)[0]:
                        ui._values["colmap_init_mode"] = idx
                        mode_idx = idx
                    if selected:
                        imgui.set_item_default_focus()
                imgui.end_combo()
            if imgui.is_item_hovered():
                imgui.set_item_tooltip("COLMAP Pointcloud uses sparse points filtered by Min Camera Observations, Diffused Pointcloud resamples that filtered set, Custom PLY loads a chosen gaussian seed scene, and From Depth calibrates matched 16-bit PNG depth maps into a point cloud using an iteratively reweighted robust per-pose affine depth fit from all valid observed points, while rejecting projected samples that land on strong local depth-gradient spikes.")
            if mode_idx in (0, 1, 3):
                if mode_idx in (0, 1):
                    changed, value = imgui.drag_int(
                        "Min Camera Observations",
                        int(ui._values.get("colmap_min_track_length", DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH)),
                        0.25,
                        0,
                        32,
                    )
                    if changed:
                        ui._values["colmap_min_track_length"] = max(int(value), 0)
                    if imgui.is_item_hovered():
                        imgui.set_item_tooltip("Ignore sparse COLMAP points whose track is shorter than this many observing cameras. Set 0 to keep all sparse points.")
                if mode_idx == 3:
                    imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x)
                    imgui.text_disabled("From Depth matches RGB and depth by relative stem under Depth Folder, uses each pose's own positive COLMAP point observations, reprojects those 3D points through the frame camera model to sample depth, rejects projected samples whose local pixel-footprint gradients are a strong outlier relative to nearby gradients, then solves one iteratively reweighted robust affine map `a + b*d` per pose from the remaining observed points before sampling a dataset-wide calibrated point budget. Frames without usable depth stay in training but are skipped for depth-based initialization.")
                    imgui.pop_text_wrap_pos()
                    depth_value_mode = max(0, min(int(ui._values.get("colmap_depth_value_mode", 1)), len(_COLMAP_DEPTH_VALUE_MODE_LABELS) - 1))
                    if imgui.begin_combo("Depth Interpretation", _COLMAP_DEPTH_VALUE_MODE_LABELS[depth_value_mode]):
                        for idx, label in enumerate(_COLMAP_DEPTH_VALUE_MODE_LABELS):
                            selected = idx == depth_value_mode
                            if imgui.selectable(label, selected)[0]:
                                ui._values["colmap_depth_value_mode"] = idx
                                depth_value_mode = idx
                            if selected:
                                imgui.set_item_default_focus()
                        imgui.end_combo()
                    if imgui.is_item_hovered():
                        imgui.set_item_tooltip("Choose whether calibrated depth values represent Euclidean camera distance or camera-space z-depth before reverse projection.")
                    changed, value = imgui.drag_int(
                        "Depth Point Count",
                        int(ui._values.get("colmap_depth_point_count", 100000)),
                        1000.0,
                        1,
                        10000000,
                    )
                    if changed:
                        ui._values["colmap_depth_point_count"] = max(int(value), 1)
                    if imgui.is_item_hovered():
                        imgui.set_item_tooltip("Total calibrated points sampled across all matched RGB/depth pairs for depth-based initialization.")
                if mode_idx == 1:
                    changed, value = imgui.drag_int(
                        "Point Count",
                        int(ui._values.get("colmap_diffused_point_count", 100000)),
                        1000.0,
                        1,
                        10000000,
                    )
                    if changed:
                        ui._values["colmap_diffused_point_count"] = max(int(value), 1)
                    if imgui.is_item_hovered():
                        imgui.set_item_tooltip("Number of gaussians synthesized by resampling COLMAP points with replacement before diffusion.")
                    changed, value = imgui.drag_float(
                        "Diffusion Radius",
                        float(ui._values.get("colmap_diffusion_radius", 1.0)),
                        0.01,
                        0.0,
                        16.0,
                        "%.4f",
                        imgui.SliderFlags_.logarithmic.value,
                    )
                    if changed:
                        ui._values["colmap_diffusion_radius"] = max(float(value), 0.0)
                    if imgui.is_item_hovered():
                        imgui.set_item_tooltip("Local diffusion multiplier applied to each sampled point's original-cloud nearest-neighbor distance.")
                changed, value = imgui.drag_float(
                    "NN Radius Scale Coef",
                    float(ui._values.get("colmap_nn_radius_scale_coef", 0.5)),
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
            ui._values["_show_histograms_prev"] = False
            return
        if not bool(ui._values.get("_show_histograms_prev", False)):
            ui._values["_histograms_refresh_requested"] = True
        self._dock_tool_window(imgui.Cond_.appearing.value)
        imgui.set_next_window_pos(imgui.ImVec2(72.0, self._menu_bar_height + 56.0), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(_HISTOGRAM_WINDOW_WIDTH, _HISTOGRAM_WINDOW_HEIGHT), imgui.Cond_.first_use_ever.value)
        opened, show = imgui.begin("Histograms", True)
        ui._values["show_histograms"] = bool(show)
        ui._values["_show_histograms_prev"] = bool(show)
        if opened:
            self._draw_histogram_controls(ui)
            status = str(ui._texts.get("histogram_status", "")).strip()
            payload = ui._values.get("_histogram_payload")
            range_payload = ui._values.get("_histogram_range_payload")
            self._update_histogram_log_range(ui, payload, range_payload)
            if status:
                imgui.text_disabled(status)
                imgui.separator()
            if payload is None or getattr(payload, "counts", np.zeros((0, 0), dtype=np.int64)).size == 0 or int(np.sum(payload.counts)) == 0:
                imgui.text_wrapped("No cached ellipse gradient histogram data is available yet.")
                if range_payload is not None:
                    imgui.separator()
                    self._draw_histogram_range_debug(range_payload)
            else:
                self._update_histogram_y_limit(ui, payload)
                self._draw_histogram_groups(ui, payload)
                imgui.separator()
                self._draw_histogram_range_debug(range_payload)
        imgui.end()

    @staticmethod
    def _training_views_value_text(value: object, *, precision: int = 3, nan_text: str = "n/a") -> str:
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return nan_text
        return nan_text if not np.isfinite(scalar) else f"{scalar:.{precision}f}"

    def _draw_training_views_window(self, ui: ViewerUI) -> None:
        if not bool(ui._values.get("show_training_views", False)):
            return
        self._dock_tool_window(imgui.Cond_.appearing.value)
        imgui.set_next_window_pos(imgui.ImVec2(88.0, self._menu_bar_height + 64.0), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(980.0, 420.0), imgui.Cond_.first_use_ever.value)
        opened, show = imgui.begin("Training Views", True)
        ui._values["show_training_views"] = bool(show)
        if opened:
            rows = tuple(ui._values.get("_training_views_rows", ()))
            if len(rows) == 0:
                imgui.text_wrapped("No training views are available.")
            else:
                flags = (
                    imgui.TableFlags_.row_bg.value
                    | imgui.TableFlags_.borders.value
                    | imgui.TableFlags_.resizable.value
                    | imgui.TableFlags_.scroll_x.value
                    | imgui.TableFlags_.scroll_y.value
                    | imgui.TableFlags_.hideable.value
                )
                if imgui.begin_table("##training_views", 10, flags):
                    imgui.table_setup_column("Image", imgui.TableColumnFlags_.width_stretch.value)
                    imgui.table_setup_column("Res", imgui.TableColumnFlags_.width_fixed.value, 92.0)
                    imgui.table_setup_column("Fx", imgui.TableColumnFlags_.width_fixed.value, 76.0)
                    imgui.table_setup_column("Fy", imgui.TableColumnFlags_.width_fixed.value, 76.0)
                    imgui.table_setup_column("Cx", imgui.TableColumnFlags_.width_fixed.value, 76.0)
                    imgui.table_setup_column("Cy", imgui.TableColumnFlags_.width_fixed.value, 76.0)
                    imgui.table_setup_column("Near", imgui.TableColumnFlags_.width_fixed.value, 68.0)
                    imgui.table_setup_column("Far", imgui.TableColumnFlags_.width_fixed.value, 68.0)
                    imgui.table_setup_column("Loss", imgui.TableColumnFlags_.width_fixed.value, 88.0)
                    imgui.table_setup_column("PSNR", imgui.TableColumnFlags_.width_fixed.value, 80.0)
                    imgui.table_headers_row()
                    for row in rows:
                        imgui.table_next_row()
                        values = (
                            str(row.get("image_name", "")),
                            str(row.get("resolution", "")),
                            self._training_views_value_text(row.get("fx")),
                            self._training_views_value_text(row.get("fy")),
                            self._training_views_value_text(row.get("cx")),
                            self._training_views_value_text(row.get("cy")),
                            self._training_views_value_text(row.get("near"), precision=2),
                            self._training_views_value_text(row.get("far"), precision=2),
                            self._training_views_value_text(row.get("loss"), precision=4),
                            self._training_views_value_text(row.get("psnr"), precision=2),
                        )
                        for value in values:
                            imgui.table_next_column()
                            imgui.text_unformatted(value)
                    imgui.end_table()
        imgui.end()

    def _draw_histogram_controls(self, ui: ViewerUI) -> None:
        if imgui.button("Refresh"):
            ui._values["_histograms_refresh_requested"] = True
        imgui.same_line()
        if imgui.button("Update Y Scale"):
            ui._values["_histogram_update_y_limit"] = True
        imgui.same_line()
        if imgui.button("Update Log Range"):
            ui._values["_histogram_update_log_range"] = True
        flags = imgui.TableFlags_.sizing_stretch_prop.value | imgui.TableFlags_.pad_outer_x.value
        if imgui.begin_table("##hist_controls", 2, flags):
            imgui.table_setup_column("Label", imgui.TableColumnFlags_.width_fixed.value, _HISTOGRAM_CONTROL_LABEL_WIDTH)
            imgui.table_setup_column("Value", imgui.TableColumnFlags_.width_stretch.value)
            self._draw_histogram_control_row_int(ui, "Bin Count", "##hist_bin_count", "hist_bin_count", _HISTOGRAM_BIN_COUNT_DEFAULT, 1)
            self._draw_histogram_control_row_float(ui, "Min Log10", "##hist_min_log10", "hist_min_log10", _HISTOGRAM_MIN_LOG10_DEFAULT, "%.3f")
            self._draw_histogram_control_row_float(ui, "Max Log10", "##hist_max_log10", "hist_max_log10", _HISTOGRAM_MAX_LOG10_DEFAULT, "%.3f")
            self._draw_histogram_control_row_float(ui, "Y Limit", "##hist_y_limit", "hist_y_limit", _HISTOGRAM_Y_LIMIT_DEFAULT, "%.1f", min_value=1.0)
            imgui.end_table()
        imgui.separator()

    def _draw_histogram_control_row_int(self, ui: ViewerUI, label: str, input_id: str, value_key: str, default: int, min_value: int) -> None:
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.align_text_to_frame_padding()
        imgui.text_unformatted(label)
        imgui.table_next_column()
        imgui.set_next_item_width(-1.0)
        changed, value = imgui.input_int(input_id, int(ui._values.get(value_key, default)), 8, 32)
        if changed:
            ui._values[value_key] = max(int(value), int(min_value))

    def _draw_histogram_control_row_float(
        self,
        ui: ViewerUI,
        label: str,
        input_id: str,
        value_key: str,
        default: float,
        display_format: str,
        min_value: float | None = None,
    ) -> None:
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.align_text_to_frame_padding()
        imgui.text_unformatted(label)
        imgui.table_next_column()
        imgui.set_next_item_width(-1.0)
        changed, value = imgui.input_float(input_id, float(ui._values.get(value_key, default)), 0.25, 1.0, display_format)
        if changed:
            ui._values[value_key] = float(value) if min_value is None else max(float(value), float(min_value))

    def _update_histogram_y_limit(self, ui: ViewerUI, payload: object) -> None:
        if not bool(ui._values.get("_histogram_update_y_limit", False)):
            return
        counts = np.asarray(getattr(payload, "counts", np.zeros((0, 0), dtype=np.int64)), dtype=np.float64)
        ui._values["hist_y_limit"] = max(1.3 * float(np.max(counts) if counts.size > 0 else 0.0), 1.0)
        ui._values["_histogram_update_y_limit"] = False

    def _update_histogram_log_range(self, ui: ViewerUI, histogram_payload: object, range_payload: object) -> None:
        if not bool(ui._values.get("_histogram_update_log_range", False)):
            return
        log_range = _histogram_log_range_from_histogram(histogram_payload)
        if log_range is None:
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
        column_count = 1 if imgui.get_content_region_avail().x < (_HISTOGRAM_PLOT_MIN_COLUMN_WIDTH * 2.0) else 2
        for group_name, indices in _HISTOGRAM_GROUPS:
            valid = tuple(index for index in indices if 0 <= int(index) < counts.shape[0])
            if not valid:
                continue
            imgui.separator_text(group_name)
            if imgui.begin_table(f"##hist_{group_name}", column_count, imgui.TableFlags_.sizing_stretch_same.value):
                for index in valid:
                    imgui.table_next_column()
                    self._draw_histogram_plot(labels[index] if index < len(labels) else f"param {index}", centers, counts[index], y_limit)
                imgui.end_table()
            imgui.spacing()

    def _draw_histogram_plot(self, label: str, centers: np.ndarray, counts: np.ndarray, y_limit: float) -> None:
        imgui.text_disabled(label)
        plot_id = f"##plot_{label}"
        if implot.begin_plot(plot_id, imgui.ImVec2(-1, _HISTOGRAM_PLOT_HEIGHT)):
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
        depth_text = self._path_text(ui, "colmap_depth_root", "<none>")
        imgui.spacing()
        imgui.text_disabled(f"Root: {Path(root_text).name if root_text != '<none>' else root_text}")
        imgui.text_disabled(f"Images: {Path(images_text).name if images_text != '<none>' else images_text}")
        if depth_text != "<none>":
            imgui.text_disabled(f"Depth: {Path(depth_text).name}")
            imgui.text_disabled(f"Depth Mode: {_COLMAP_DEPTH_VALUE_MODE_LABELS[max(0, min(int(ui._values.get('colmap_depth_value_mode', 1)), len(_COLMAP_DEPTH_VALUE_MODE_LABELS) - 1))]}")
        imgui.text_disabled(f"Init: {_colmap_init_mode_label(ui)}")
        imgui.separator()

    def _section_camera(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Camera", imgui.TreeNodeFlags_.default_open.value):
            return
        changed, val = imgui.drag_float(
            "Move Speed", float(ui._values["move_speed"]),
            0.05, 0.0, 0.0, "%.4g", imgui.SliderFlags_.logarithmic.value
        )
        if changed:
            ui._values["move_speed"] = max(val, 0.0)
        if imgui.is_item_hovered():
            imgui.set_item_tooltip("Camera movement speed (scroll wheel also adjusts)")
        changed, val = imgui.slider_float("FOV", float(ui._values["fov"]), 25.0, 100.0, "%.1f\u00b0")
        if changed:
            ui._values["fov"] = val
        if imgui.is_item_hovered():
            imgui.set_item_tooltip("Vertical field of view in degrees")
        self._draw_control(ui, next(spec for spec in GROUP_SPECS["Camera"] if spec.key == "render_background_mode"))
        if int(ui._values.get("render_background_mode", 1)) == 1:
            self._draw_control(ui, next(spec for spec in GROUP_SPECS["Camera"] if spec.key == "render_background_color"))
        imgui.spacing()
        imgui.text_disabled("LMB drag=look | WASDQE=move | Wheel=speed")
        self._ctx_reset("camera_ctx", ui, ("move_speed", "fov", "render_background_mode", "render_background_color"))
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
        density_text = ui._texts.get("training_density", "Density Avg: n/a")
        psnr_text = ui._texts.get("training_psnr", "PSNR: n/a")
        if imgui.begin_table("##train_info", 2, imgui.TableFlags_.sizing_stretch_same.value):
            imgui.table_setup_column("L", imgui.TableColumnFlags_.width_fixed.value, 50)
            imgui.table_setup_column("V")
            for label, text in (("Step", training_text), ("Time", time_text), ("Avg", avg_iters_text), ("Loss", loss_text), ("MSE", mse_text), ("Density", density_text), ("PSNR", psnr_text)):
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

    def _section_training_setup(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Train Setup"):
            return
        for key in ("max_gaussians", "training_steps_per_frame", "background_mode", "refinement_interval", "refinement_growth_ratio", "refinement_growth_start_step", "refinement_alpha_cull_threshold", "refinement_min_contribution_percent", "refinement_min_contribution_decay", "refinement_opacity_mul", "refinement_loss_weight", "refinement_target_edge_weight", "train_downscale_mode", "train_subsample_factor"):
            self._draw_control(ui, next(spec for spec in GROUP_SPECS["Train Setup"] if spec.key == key))
        if int(ui._values.get("background_mode", 1)) == 0:
            self._draw_control(ui, next(spec for spec in GROUP_SPECS["Train Setup"] if spec.key == "train_background_color"))
        if int(ui._values.get("train_downscale_mode", 0)) == 0:
            for key in ("train_auto_start_downscale", "train_downscale_base_iters", "train_downscale_iter_step", "train_downscale_max_iters"):
                self._draw_control(ui, next(spec for spec in GROUP_SPECS["Train Setup"] if spec.key == key))
        for key in ("seed", "init_opacity"):
            self._draw_control(ui, next(spec for spec in GROUP_SPECS["Train Setup"] if spec.key == key))
        train_resolution = ui._texts.get("training_resolution", "")
        if train_resolution:
            _draw_disabled_wrapped_text(train_resolution)
        downscale_status = ui._texts.get("training_downscale", "")
        if downscale_status:
            _draw_disabled_wrapped_text(downscale_status)
        schedule_status = ui._texts.get("training_schedule", "")
        if schedule_status:
            _draw_disabled_wrapped_text(schedule_status)
        refinement_status = ui._texts.get("training_refinement", "")
        if refinement_status:
            _draw_disabled_wrapped_text(refinement_status)
        _draw_disabled_wrapped_text("COLMAP import chooses direct pointcloud init, diffused pointcloud init, or a custom PLY scene.")
        self._ctx_reset("train_setup_ctx", ui, [s.key for s in GROUP_SPECS["Train Setup"]])
        imgui.separator()

    def _draw_schedule_stage_tabs(self, ui: ViewerUI) -> None:
        if not imgui.begin_tab_bar("##schedule_stage_tabs"):
            return
        for stage_label, specs in SCHEDULE_STAGE_SPECS.items():
            if not imgui.begin_tab_item(stage_label)[0]:
                continue
            imgui.spacing()
            for spec in specs:
                self._draw_control(ui, spec)
            imgui.end_tab_item()
        imgui.end_tab_bar()

    def _section_optimizer(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Optimizer"):
            return
        if imgui.begin_tab_bar("##optim_tabs"):
            if imgui.begin_tab_item("Schedule")[0]:
                imgui.spacing()
                for key in _OPTIMIZER_TAB_KEYS["Schedule"]:
                    self._draw_control(ui, _TRAIN_OPTIMIZER_SPEC_BY_KEY[key])
                imgui.separator()
                self._draw_schedule_stage_tabs(ui)
                current_values = ui._texts.get("training_schedule_values", "")
                if current_values:
                    imgui.separator()
                    imgui.text_unformatted("Current Values")
                    _draw_disabled_wrapped_text(current_values)
                imgui.end_tab_item()
            if imgui.begin_tab_item("Adam")[0]:
                imgui.spacing()
                for key in _OPTIMIZER_TAB_KEYS["Adam"]:
                    self._draw_control(ui, _TRAIN_OPTIMIZER_SPEC_BY_KEY[key])
                imgui.end_tab_item()
            if imgui.begin_tab_item("Regularization")[0]:
                imgui.spacing()
                for key in _OPTIMIZER_TAB_KEYS["Regularization"]:
                    self._draw_control(ui, _TRAIN_OPTIMIZER_SPEC_BY_KEY[key])
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
        for key in ("radius_scale", "alpha_cutoff", "trans_threshold", "cached_raster_grad_atomic_mode"):
            self._draw_control(ui, next(spec for spec in RENDER_PARAM_SPECS if spec.key == key))

        imgui.separator_text("Cached Grad Ranges")
        for key in ("cached_raster_grad_fixed_ro_local_range", "cached_raster_grad_fixed_scale_range", "cached_raster_grad_fixed_quat_range", "cached_raster_grad_fixed_color_range", "cached_raster_grad_fixed_opacity_range"):
            self._draw_control(ui, next(spec for spec in RENDER_PARAM_SPECS if spec.key == key))
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
        "render_background_mode": "Choose whether the main renderer uses the training background color or a separate custom RGB background",
        "render_background_color": "Custom RGB background for the main renderer",
        "radius_scale": "Multiplier on top of true 3DGS gaussian size for rendering",
        "alpha_cutoff": "Minimum alpha threshold — splats below this are skipped",
        "trans_threshold": "Transmittance threshold for early ray termination",
        "cached_raster_grad_atomic_mode": "Choose float atomics or fixed-point atomics for cached ellipsoid gradient accumulation during raster backward",
        "cached_raster_grad_fixed_ro_local_range": "Symmetric [-X, X] range for avgInvScale-normalized cached position gradients",
        "cached_raster_grad_fixed_scale_range": "Symmetric [-X, X] range for avgInvScale-normalized cached scale gradients",
        "cached_raster_grad_fixed_quat_range": "Symmetric [-X, X] range for avgInvScale-normalized cached rotation gradients",
        "cached_raster_grad_fixed_color_range": "Symmetric [-X, X] range for cached color gradients",
        "cached_raster_grad_fixed_opacity_range": "Symmetric [-X, X] range for cached opacity gradients",
        "debug_mode": "Select the renderer debug output mode",
        "debug_sh_coeff_index": "Select which raw SH coefficient float3 to display; zero is mapped to 0.5 gray in this debug view",
        "debug_grad_norm_threshold": "Reference threshold for the gradient norm heatmap",
        "debug_ellipse_thickness_px": "Thickness used by ellipse outline debug rendering",
        "debug_clone_count_min": "Lower bound for the per-splat clone counter heatmap",
        "debug_clone_count_max": "Upper bound for the per-splat clone counter heatmap",
        "debug_density_min": "Lower bound for density debug heatmaps",
        "debug_density_max": "Upper bound for density debug heatmaps",
        "debug_contribution_min": "Lower bound for the log-scaled contribution heatmap in percent of observed dataset pixels",
        "debug_contribution_max": "Upper bound for the log-scaled contribution heatmap in percent of observed dataset pixels",
        "debug_adam_momentum_threshold": "Reference threshold for the per-splat Adam first-moment length heatmap; the displayed range is derived around it on a log scale",
        "debug_depth_mean_min": "Lower bound for depth mean debug heatmap",
        "debug_depth_mean_max": "Upper bound for depth mean debug heatmap",
        "debug_depth_std_min": "Lower bound for depth standard deviation heatmap",
        "debug_depth_std_max": "Upper bound for depth standard deviation heatmap",
        "debug_depth_local_mismatch_min": "Lower bound for the local depth mismatch ratio heatmap, normalized by mean depth",
        "debug_depth_local_mismatch_max": "Upper bound for the local depth mismatch ratio heatmap, normalized by mean depth",
        "debug_depth_local_mismatch_smooth_radius": "Sigma multiple for full local smoothing before the mismatch gate starts to fall off",
        "debug_depth_local_mismatch_reject_radius": "Base sigma multiple, based on mean splat sigma, beyond which depth mismatch stops contributing; the effective reject radius scales smoothly up to 2x with current splat alpha",
        "lr_pos_mul": "Learning rate multiplier for position",
        "lr_pos_stage1_mul": "Position learning-rate multiplier target reached at the end of Stage 1",
        "lr_pos_stage2_mul": "Position learning-rate multiplier target reached at the end of Stage 2",
        "lr_pos_stage3_mul": "Position learning-rate multiplier target reached at the end of Stage 3",
        "lr_sh_mul": "Learning rate multiplier for non-DC SH coefficients",
        "lr_sh_stage1_mul": "Non-DC SH learning-rate multiplier target reached at the end of Stage 1",
        "lr_sh_stage2_mul": "Non-DC SH learning-rate multiplier target reached at the end of Stage 2",
        "lr_sh_stage3_mul": "Non-DC SH learning-rate multiplier target reached at the end of Stage 3",
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
        "sh1_reg": "L1 regularization weight applied to all non-DC SH coefficients",
        "opacity_reg": "Opacity regularization weight (pushes toward 0 or 1)",
        "position_random_step_noise_lr": "Stage 0 post-step MCMC-style position noise multiplier; when scheduling is disabled this value is used for the whole run",
        "position_random_step_opacity_gate_center": "Opacity center for the random-step sigmoid gate; lower-opacity splats get stronger position noise",
        "position_random_step_opacity_gate_sharpness": "Steepness of the random-step opacity gate",
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
        "background_mode": "Choose whether training uses a fixed custom RGB background or a new seeded white-noise background each optimizer step",
        "train_background_color": "Custom RGB background used for training when Train Background is set to Custom",
        "sh_band": "Stage 0 SH band limit; SH0 uses only the DC term and SH3 enables the full coefficient set",
        "refinement_interval": "Run cull/split refinement every N training steps",
        "refinement_growth_ratio": "Target fractional scene growth per refinement step once densification is enabled",
        "refinement_growth_start_step": "Keep densification growth at zero until this training iteration, then use the configured refinement growth; slider range follows Schedule Steps",
        "refinement_alpha_cull_threshold": "Cull splats below this decoded alpha threshold during refinement",
        "refinement_min_contribution_percent": "Minimum accumulated alpha contribution, as a percent of observed dataset pixels, required for a splat to survive refinement",
        "refinement_min_contribution_decay": "Multiply the minimum contribution percent by this factor after each completed refinement pass",
        "refinement_opacity_mul": "Multiply every surviving splat alpha by this factor during each refinement rewrite pass",
        "refinement_loss_weight": "Weight of normalized per-pixel RGB loss in hybrid clone selection during refinement",
        "refinement_target_edge_weight": "Weight of normalized target-image Sobel edge intensity in hybrid clone selection during refinement",
        "density_regularizer": "Weight applied to the per-pixel hinge penalty max(density - max_allowed_density, 0)",
        "depth_ratio_weight": "Stage 0 depth-ratio regularizer weight; when scheduling is disabled this value is used for the whole run",
        "depth_ratio_grad_min": "Start of the high-gradient depth-ratio interval; gradients taper below this value",
        "depth_ratio_grad_max": "End of the high-gradient depth-ratio interval; gradients taper above this value",
        "max_allowed_density": "End-of-training per-pixel density threshold above which the density regularizer activates; runtime ramps from 5.0 to this value over the LR schedule",
        "lr_schedule_enabled": "Enable the piecewise-linear base learning-rate schedule",
        "lr_schedule_start_lr": "Stage 0 base learning rate; when scheduling is disabled this value is used for the whole run",
        "lr_schedule_stage1_step": "Training step where Stage 1 ends and the Stage 1 targets are reached; slider range follows the Stage 3 end step",
        "lr_schedule_stage2_step": "Training step where Stage 2 ends and the Stage 2 targets are reached; slider range follows the Stage 3 end step",
        "lr_schedule_stage1_lr": "Base learning-rate target reached at the end of Stage 1",
        "lr_schedule_stage2_lr": "Base learning-rate target reached at the end of Stage 2",
        "lr_schedule_end_lr": "Base learning-rate target reached at the end of Stage 3",
        "lr_schedule_steps": "Stage 3 end step and total step budget shared by the LR, depth-ratio, noise, and SH schedules",
        "depth_ratio_stage1_weight": "Depth-ratio regularizer target reached at the end of Stage 1",
        "depth_ratio_stage2_weight": "Depth-ratio regularizer target reached at the end of Stage 2",
        "depth_ratio_stage3_weight": "Depth-ratio regularizer target reached at the end of Stage 3",
        "position_random_step_noise_lr": "Initial post-step MCMC-style position noise multiplier before Stage 1; each stage tab sets the target noise LR reached at that stage end step",
        "position_random_step_noise_stage1_lr": "Position-noise LR target reached at the end of Stage 1",
        "position_random_step_noise_stage2_lr": "Position-noise LR target reached at the end of Stage 2",
        "position_random_step_noise_stage3_lr": "Position-noise LR target reached at the end of Stage 3",
        "sh_band_stage1": "SH band limit reached by the end of Stage 1",
        "sh_band_stage2": "SH band limit reached by the end of Stage 2",
        "sh_band_stage3": "SH band limit reached by the end of Stage 3",
        "train_downscale_mode": "Use Auto for scheduled downscale descent or choose a fixed manual override from 1x to 16x",
        "train_auto_start_downscale": "Initial downscale factor used at step 0 when Downscale Mode is Auto",
        "train_downscale_base_iters": "Number of iterations spent at the auto start factor before descending",
        "train_downscale_iter_step": "Additional iterations added to each lower auto downscale phase",
        "train_downscale_max_iters": "Displayed training schedule budget for the auto downscale progression; training does not stop automatically",
        _LOSS_DEBUG_ABS_SCALE_KEY: "Multiplier applied to absolute RGB difference before presenting the debug texture",
        "seed": "Random seed for training frame shuffle order",
        "init_opacity": "Initial opacity for new gaussians",
    }

    def _draw_control(self, ui: ViewerUI, spec: ControlSpec, compact: bool = False) -> None:
        key = spec.key
        if key not in ui._values:
            ui._values[key] = spec.kwargs.get("value", 0)
        label = f"##{key}" if compact else spec.label
        if compact:
            imgui.text_unformatted(spec.label)
        if spec.kind == "slider_float":
            changed, val = imgui.slider_float(
                label, float(ui._values[key]),
                float(spec.kwargs.get("min", 0.0)), float(spec.kwargs.get("max", 1.0)),
                spec.kwargs.get("format", "%.3f"),
                imgui.SliderFlags_.logarithmic.value if spec.kwargs.get("logarithmic") else 0
            )
            if changed:
                ui._values[key] = val
        elif spec.kind == "slider_int":
            min_value = int(spec.kwargs.get("min", 0))
            max_value = int(spec.kwargs.get("max", 100))
            if "min_from" in spec.kwargs:
                min_value = _control_bound(ui, spec, "min_from", min_value)
            if "max_from" in spec.kwargs:
                max_value = _control_bound(ui, spec, "max_from", max_value)
            max_value = max(max_value, min_value)
            current_value = min(max(int(ui._values[key]), min_value), max_value)
            if current_value != int(ui._values[key]):
                ui._values[key] = current_value
            changed, val = imgui.slider_int(
                label, current_value, min_value, max_value
            )
            if changed:
                ui._values[key] = min(max(int(val), min_value), max_value)
        elif spec.kind == "input_int":
            changed, val = imgui.input_int(
                label,
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
            if imgui.begin_combo(label, preview):
                for idx, option in enumerate(options):
                    selected = idx == current
                    if imgui.selectable(str(option), selected)[0]:
                        ui._values[key] = idx
                    if selected:
                        imgui.set_item_default_focus()
                imgui.end_combo()
        elif spec.kind == "input_float":
            changed, val = imgui.input_float(
                label, float(ui._values[key]),
                float(spec.kwargs.get("step", 0.0)),
                float(spec.kwargs.get("step_fast", 0.0)),
                spec.kwargs.get("format", "%.3f")
            )
            if changed:
                ui._values[key] = val
        elif spec.kind == "checkbox":
            changed, val = imgui.checkbox(label, bool(ui._values[key]))
            if changed:
                ui._values[key] = val
        elif spec.kind == "color_edit3":
            value = np.asarray(ui._values[key], dtype=np.float32).reshape(3)
            changed, color = imgui.color_edit3(label, imgui.ImVec4(float(value[0]), float(value[1]), float(value[2]), 1.0))
            if changed:
                ui._values[key] = (float(color.x), float(color.y), float(color.z))
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
    for spec in DEBUG_RENDER_SPECS:
        values[spec.key] = spec.kwargs.get("value", 0)
    values["radius_scale"] = float(renderer.radius_scale)
    values["alpha_cutoff"] = float(renderer.alpha_cutoff)
    values["trans_threshold"] = float(renderer.transmittance_threshold)
    values["cached_raster_grad_atomic_mode"] = 0 if getattr(renderer, "cached_raster_grad_atomic_mode", "fixed") == "float" else 1
    values["cached_raster_grad_fixed_ro_local_range"] = float(getattr(renderer, "cached_raster_grad_fixed_ro_local_range", 0.01))
    values["cached_raster_grad_fixed_scale_range"] = float(getattr(renderer, "cached_raster_grad_fixed_scale_range", 0.01))
    values["cached_raster_grad_fixed_quat_range"] = float(getattr(renderer, "cached_raster_grad_fixed_quat_range", 0.01))
    values["cached_raster_grad_fixed_color_range"] = float(getattr(renderer, "cached_raster_grad_fixed_color_range", 0.2))
    values["cached_raster_grad_fixed_opacity_range"] = float(getattr(renderer, "cached_raster_grad_fixed_opacity_range", 0.2))
    debug_mode = getattr(renderer, "debug_mode", "normal")
    values["debug_mode"] = _DEBUG_MODE_VALUES.index(debug_mode) if debug_mode in _DEBUG_MODE_VALUES else 0
    values["debug_grad_norm_threshold"] = float(getattr(renderer, "debug_grad_norm_threshold", _DEBUG_GRAD_NORM_THRESHOLD_DEFAULT))
    values["debug_ellipse_thickness_px"] = float(getattr(renderer, "debug_ellipse_thickness_px", 4.0))
    clone_count_range = tuple(getattr(renderer, "debug_clone_count_range", (0.0, 16.0)))
    density_range = tuple(getattr(renderer, "debug_density_range", (0.0, 20.0)))
    contribution_range = tuple(getattr(renderer, "debug_contribution_range", (0.001, 1.0)))
    adam_momentum_range = tuple(getattr(renderer, "debug_adam_momentum_range", _threshold_band_range(_DEBUG_ADAM_MOMENTUM_THRESHOLD_DEFAULT)))
    depth_mean_range = tuple(getattr(renderer, "debug_depth_mean_range", (0.0, 10.0)))
    depth_std_range = tuple(getattr(renderer, "debug_depth_std_range", (0.0, 0.5)))
    depth_local_mismatch_range = tuple(getattr(renderer, "debug_depth_local_mismatch_range", (0.0, 0.5)))
    values["debug_clone_count_min"] = float(clone_count_range[0])
    values["debug_clone_count_max"] = float(clone_count_range[1])
    values["debug_density_min"] = float(density_range[0])
    values["debug_density_max"] = float(density_range[1])
    values["debug_contribution_min"] = float(contribution_range[0])
    values["debug_contribution_max"] = float(contribution_range[1])
    values["debug_adam_momentum_threshold"] = _threshold_from_band_range(float(adam_momentum_range[0]), float(adam_momentum_range[1]), _DEBUG_ADAM_MOMENTUM_THRESHOLD_DEFAULT)
    values["debug_depth_mean_min"] = float(depth_mean_range[0])
    values["debug_depth_mean_max"] = float(depth_mean_range[1])
    values["debug_depth_std_min"] = float(depth_std_range[0])
    values["debug_depth_std_max"] = float(depth_std_range[1])
    values["debug_depth_local_mismatch_min"] = float(depth_local_mismatch_range[0])
    values["debug_depth_local_mismatch_max"] = float(depth_local_mismatch_range[1])
    values["debug_depth_local_mismatch_smooth_radius"] = float(getattr(renderer, "debug_depth_local_mismatch_smooth_radius", 2.0))
    values["debug_depth_local_mismatch_reject_radius"] = float(getattr(renderer, "debug_depth_local_mismatch_reject_radius", 4.0))
    values["debug_sh_coeff_index"] = int(getattr(renderer, "debug_sh_coeff_index", 0))
    values["colmap_root_path"] = ""
    values["colmap_database_path"] = ""
    values["colmap_images_root"] = ""
    values["colmap_depth_root"] = ""
    values["colmap_depth_value_mode"] = 1
    values["colmap_init_mode"] = 0
    values["colmap_custom_ply_path"] = ""
    values["colmap_selected_camera_ids"] = ()
    values["colmap_image_downscale_mode"] = 0
    values["colmap_image_max_size"] = 2048
    values["colmap_image_scale"] = 1.0
    values["colmap_nn_radius_scale_coef"] = 0.5
    values["colmap_min_track_length"] = DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH
    values["colmap_depth_point_count"] = 100000
    values["colmap_diffused_point_count"] = 100000
    values["colmap_diffusion_radius"] = 1.0
    values["show_histograms"] = False
    values["show_training_views"] = False
    values["show_camera_overlays"] = True
    values["show_camera_labels"] = False
    values["show_training_cameras"] = False
    values["hist_bin_count"] = _HISTOGRAM_BIN_COUNT_DEFAULT
    values["hist_min_log10"] = _HISTOGRAM_MIN_LOG10_DEFAULT
    values["hist_max_log10"] = _HISTOGRAM_MAX_LOG10_DEFAULT
    values["hist_y_limit"] = _HISTOGRAM_Y_LIMIT_DEFAULT
    values["_histograms_refresh_requested"] = False
    values["_show_histograms_prev"] = False
    values["_histogram_update_y_limit"] = True
    values["_histogram_update_log_range"] = False
    values["_histogram_payload"] = None
    values["_histogram_range_payload"] = None
    values["_training_views_rows"] = ()
    values["_training_view_overlay_segments"] = ()
    values["_loss_debug_frame_max"] = 0
    values["_viewport_sh_band"] = 3
    values["_viewport_sh_control_key"] = "sh_band"
    values["_viewport_sh_stage_label"] = "Stage 0"
    values["_colmap_camera_rows"] = ()
    values["_colmap_import_active"] = False
    values["_colmap_import_fraction"] = 0.0
    values["_can_export_ply"] = False

    texts: dict[str, str] = {
        key: "" for key in (
            "fps", "path", "scene_stats", "render_stats", "training",
            "training_time", "training_iters_avg", "training_loss", "training_mse", "training_density", "training_psnr", "training_instability", "error",
            "loss_debug_view", "loss_debug_frame",
            "colmap_import_status", "colmap_import_current",
            "training_resolution", "training_downscale", "training_schedule", "training_schedule_values", "training_refinement",
            "histogram_status",
            "setup_hint", "stability_hint",
        )
    }
    return ViewerUI(_values=values, _texts=texts)


def create_toolkit_window(device: spy.Device, width: int, height: int) -> ToolkitWindow:
    return ToolkitWindow(device=device, width=width, height=height)
