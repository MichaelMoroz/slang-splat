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
from imgui_bundle import imgui, imgui_md, implot

from ..metrics import PARAM_HISTOGRAM_SCALE_LINEAR, PARAM_HISTOGRAM_SCALE_LOG10
from ..repo_defaults import json_value
from ..app.training_controls import (
    SH_BAND_LABELS as _SH_BAND_LABELS,
    SCHEDULE_STAGE_CONTROL_DEFS,
    SCHEDULE_STAGE_GROUPS as _SCHEDULE_STAGE_GROUPS,
    TRAINING_OPTIMIZER_GROUP,
    TRAINING_OPTIMIZER_TAB_KEYS,
    TRAINING_SETUP_GROUP,
    TRAINING_STABILITY_GROUP,
    TRAINING_UI_GROUP_DEFS,
    TRAIN_STABILITY_PAIRED_KEYS,
)
from ..training.photometric_compensation import PhotometricCompensationHyperParams
from ..training.alpha_modes import TARGET_ALPHA_MODE_LABELS
from ..training.ppisp import PPISP_FIELD_SPECS
from .buffer_debug import ResourceDebugSnapshot, format_resource_bytes, write_resource_debug_log
from .state import DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH, LOSS_DEBUG_OPTIONS
from .ui_schema import (
    ControlSpec,
    GROUP_SPECS,
    SCHEDULE_STAGE_SPECS,
    UI_TOOLTIPS,
    _DEBUG_MODE_LABELS,
    _DEBUG_MODE_VALUES,
    _DEFAULT_INTERFACE_SCALE_INDEX,
    _HISTOGRAM_BIN_COUNT_DEFAULT,
    _HISTOGRAM_MAX_VALUE_DEFAULT,
    _HISTOGRAM_MIN_VALUE_DEFAULT,
    _HISTOGRAM_Y_LIMIT_DEFAULT,
    _INTERFACE_SCALE_KEY,
    _INTERFACE_SCALE_OPTIONS,
    _LOSS_DEBUG_ABS_SCALE_KEY,
    _OPTIMIZER_TAB_KEYS,
    _RENDERER_DEBUG_MODE_VALUES,
    _THEME_KEY,
    _TRAINING_RASTER_GRAD_KEYS,
    _TRAIN_OPTIMIZER_SPEC_BY_KEY,
    _VIEWER_CONTROL_DEFAULTS,
    _VIEWER_IMPORT_DEFAULTS,
    _VIEWER_IMPORT_EXPORT_FIELDS,
    _VIEWER_UI_DEFAULTS,
    _VIEWER_UI_EXPORT_FIELDS,
    _VIEWER_CONTROL_EXPORT_FIELDS,
    build_render_spec_bundle,
    _renderer_atomic_mode_index,
    _renderer_debug_mode_index,
    default_control_values,
    PPISP_DEBUG_MODE,
)
from .ui_pretty import draw_struct_sections, measure_struct_sections
from .ui_text import _build_about_text, _build_documentation_text, _draw_disabled_wrapped_text, _draw_markdown_text, _status_suffix
from ..renderer.render_params import RendererParams

TOOLKIT_WIDTH_FRACTION = 0.1875
_TOOLKIT_MIN_WIDTH = 280.0
LOSS_HISTORY_SIZE = 5000
FPS_HISTORY_SIZE = 128
_LOSS_HISTORY_WINDOW_STEPS = 30000
_LOSS_HISTORY_BUCKET_COUNT = 1000
_LOSS_HISTORY_BUCKET_SIZE = max(1, _LOSS_HISTORY_WINDOW_STEPS // _LOSS_HISTORY_BUCKET_COUNT)
_LOSS_DEBUG_ABS_SCALE_DEFAULT = 1.0
_LOSS_DEBUG_ABS_SCALE_MIN = 0.125
_LOSS_DEBUG_ABS_SCALE_MAX = 64.0
_THEME_OPTIONS = ("White", "Dark")
_BASE_FONT_SIZE_PX = 16.0
_FONT_ATLAS_SIZE_PX = _BASE_FONT_SIZE_PX * _INTERFACE_SCALE_OPTIONS[-1][1]
_COLMAP_INIT_MODE_BASE_LABELS = ("Point Sources",)
_COLMAP_INIT_MODE_DEPTH_LABEL = "From Depth"
_COLMAP_INIT_MODE_LABELS = _COLMAP_INIT_MODE_BASE_LABELS
_COLMAP_DEPTH_VALUE_MODE_LABELS = ("Depth Is Distance", "Depth Is Z-Depth")
_COLMAP_IMAGE_DOWNSCALE_LABELS = ("Original", "Max Size", "Scale Factor")
_DEBUG_GRAD_NORM_THRESHOLD_DEFAULT = 2e-4
_DEBUG_COLORBAR_HEIGHT = 28.0
_DEBUG_COLORBAR_MIN_WIDTH = 320.0
_DEBUG_COLORBAR_MAX_WIDTH = 640.0
_DEBUG_COLORBAR_MARGIN = 18.0
_DEBUG_COLORBAR_TICKS = 5
_DEBUG_COLORBAR_STEPS = 96
_DEBUG_COLORBAR_SIDE_PAD = 18.0
_DEBUG_COLORBAR_TOP_PAD = 30.0
_TRAINING_VIEWS_SORT_DEFAULT_COLUMN = "image_name"
_TRAINING_VIEWS_SORT_COLUMNS = (
    ("Image", "image_name", 1, imgui.TableColumnFlags_.width_stretch.value | imgui.TableColumnFlags_.default_sort.value, 0.0),
    ("Res", "resolution", 2, imgui.TableColumnFlags_.width_fixed.value, 92.0),
    ("Fx", "fx", 3, imgui.TableColumnFlags_.width_fixed.value, 76.0),
    ("Fy", "fy", 4, imgui.TableColumnFlags_.width_fixed.value, 76.0),
    ("Cx", "cx", 5, imgui.TableColumnFlags_.width_fixed.value, 76.0),
    ("Cy", "cy", 6, imgui.TableColumnFlags_.width_fixed.value, 76.0),
    ("Min Dist", "camera_min_dist", 7, imgui.TableColumnFlags_.width_fixed.value, 78.0),
    ("Loss", "loss", 8, imgui.TableColumnFlags_.width_fixed.value, 88.0),
    ("PSNR", "psnr", 9, imgui.TableColumnFlags_.width_fixed.value, 80.0),
)
_TRAINING_VIEWS_SORT_KEY_BY_USER_ID = {user_id: sort_key for _label, sort_key, user_id, _flags, _width in _TRAINING_VIEWS_SORT_COLUMNS}
_DEBUG_COLORBAR_BOTTOM_PAD = 30.0
_VIEWPORT_OVERLAY_MARGIN = 8.0
_VIEWPORT_OVERLAY_WIDTH = 320.0
_VIEWPORT_OVERLAY_MIN_WIDTH = 220.0
_VIEWPORT_OVERLAY_PADDING = 10.0
_VIEWPORT_OVERLAY_MIN_HEIGHT = 44.0
_TRAINING_CAMERA_COLMAP_POINT_RADIUS = 2.5
_TRAINING_CAMERA_COLMAP_POINT_SELECTED_RADIUS = 4.0
_TRAINING_CAMERA_COLMAP_POINT_HIT_RADIUS = 12.0
_TRAINING_CAMERA_COLMAP_POINT_CONTEXT_OFFSET = 12.0
_TRAINING_CAMERA_COLMAP_CONTEXT_VIEW_LIMIT = 16
_TRAINING_CAMERA_COLMAP_POINT_COLOR = (0.16, 0.86, 0.96, 0.90)
_TRAINING_CAMERA_COLMAP_POINT_SELECTED_COLOR = (1.00, 0.76, 0.20, 0.98)
_TRAINING_CAMERA_DEBUG_TEXT_FIELDS = (("loss_debug_frame", True), ("loss_debug_psnr", True))
_HISTOGRAM_AUTO_RANGE_KEEP_FRACTION = 0.99
_DOCKSPACE_FLAGS = int(imgui.DockNodeFlags_.none)
_TOOLKIT_WINDOW_NAME = "Toolkit"
_VIEWPORT_WINDOW_NAME = "Viewport###Viewport"
_HISTOGRAM_WINDOW_WIDTH = 1200.0
_HISTOGRAM_WINDOW_HEIGHT = 860.0
_HISTOGRAM_CONTROL_LABEL_WIDTH = 150.0
_HISTOGRAM_PLOT_HEIGHT = 230.0
_HISTOGRAM_PLOT_MIN_COLUMN_WIDTH = 460.0
_HISTOGRAM_LOG_Y_MIN = 1.0
_HISTOGRAM_LOG_Y_MAX_MIN = 1.01
_HISTOGRAM_LINEAR_TAB_LABEL = "Linear Values"
_HISTOGRAM_LOG10_TAB_LABEL = "Log10 Values"
_RESOURCE_DEBUG_WINDOW_WIDTH = 1120.0
_RESOURCE_DEBUG_WINDOW_HEIGHT = 620.0
_PHOTOMETRIC_UI_DEFAULTS = PhotometricCompensationHyperParams()
_DEFAULT_HISTOGRAM_GROUPS = (
    ("roLocal", (0, 1, 2)),
    ("scale", (3, 4, 5)),
    ("quat", (6, 7, 8, 9)),
    ("color", (10, 11, 12)),
    ("opacity", (13,)),
)
def _valid_depth_root_text(value: object) -> bool:
    text = str(value or "").strip()
    return bool(text) and Path(text).expanduser().is_dir()


def _colmap_init_mode_labels(depth_available: bool) -> tuple[str, ...]:
    return _COLMAP_INIT_MODE_BASE_LABELS + ((_COLMAP_INIT_MODE_DEPTH_LABEL,) if depth_available else ())


def _colmap_init_mode_label(ui: "ViewerUI", index: int | None = None) -> str:
    labels = _colmap_init_mode_labels(_valid_depth_root_text(ui._values.get("colmap_depth_root", "")))
    mode_idx = max(0, min(int(ui._values.get("colmap_init_mode", 0) if index is None else index), len(labels) - 1))
    return labels[mode_idx]


def _enabled_colmap_init_labels(ui: "ViewerUI") -> tuple[str, ...]:
    labels: list[str] = []
    if bool(ui._values.get("colmap_pointcloud_enabled", False)):
        labels.append("COLMAP Pointcloud")
    if bool(ui._values.get("colmap_diffused_enabled", False)):
        labels.append("Diffused COLMAP")
    if bool(ui._values.get("colmap_custom_ply_enabled", False)):
        labels.append("Custom PLY")
    if bool(ui._values.get("colmap_custom_mesh_enabled", False)):
        labels.append("Custom Mesh")
    if bool(ui._values.get("colmap_fibonacci_sphere_enabled", False)):
        labels.append("Fibonacci Sky Sphere")
    return tuple(labels)


def _colmap_init_summary(ui: "ViewerUI") -> str:
    labels = _enabled_colmap_init_labels(ui)
    return "None" if len(labels) == 0 else ", ".join(labels)


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


def _toolkit_panel_width(width: float, interface_scale: float) -> float:
    return float(max(int(float(width) * TOOLKIT_WIDTH_FRACTION), int(_TOOLKIT_MIN_WIDTH * max(float(interface_scale), 1.0))))


def _toolkit_width_fraction(width: float, interface_scale: float) -> float:
    return max(min(_toolkit_panel_width(width, interface_scale) / max(float(width), 1.0), 0.45), TOOLKIT_WIDTH_FRACTION)


def _panel_rect(width: int, height: int, menu_bar_height: float, interface_scale: float = 1.0) -> tuple[float, float, float, float]:
    panel_width = _toolkit_panel_width(width, interface_scale)
    return max(float(width) - panel_width, 0.0), float(menu_bar_height), panel_width, max(float(height) - float(menu_bar_height), 1.0)


def _clamp_viewport_size(width: float, height: float) -> tuple[int, int]:
    return max(int(round(float(width))), 1), max(int(round(float(height))), 1)


def _fit_aspect_rect(container_width: float, container_height: float, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    width = max(float(container_width), 1.0)
    height = max(float(container_height), 1.0)
    source_width = max(int(image_width), 1)
    source_height = max(int(image_height), 1)
    scale = min(width / float(source_width), height / float(source_height))
    draw_width = max(float(source_width) * scale, 1.0)
    draw_height = max(float(source_height) * scale, 1.0)
    return 0.5 * (width - draw_width), 0.5 * (height - draw_height), draw_width, draw_height


def _clamp_training_camera_center(center_x: float, center_y: float, zoom: float) -> tuple[float, float]:
    resolved_zoom = max(float(zoom), 1.0)
    half_span = 0.5 / resolved_zoom
    return min(max(float(center_x), half_span), 1.0 - half_span), min(max(float(center_y), half_span), 1.0 - half_span)


def _training_camera_uv_bounds(center_x: float, center_y: float, zoom: float) -> tuple[tuple[float, float], tuple[float, float]]:
    clamped_x, clamped_y = _clamp_training_camera_center(center_x, center_y, zoom)
    half_span = 0.5 / max(float(zoom), 1.0)
    return (clamped_x - half_span, clamped_y - half_span), (clamped_x + half_span, clamped_y + half_span)


def _rect_contains(rect: tuple[float, float, float, float], point: tuple[float, float] | None) -> bool:
    if point is None:
        return False
    x, y, width, height = rect
    px, py = point
    return px >= x and py >= y and px < x + width and py < y + height


def _should_capture_keyboard_for_ui(handled: bool, viewport_input_active: bool, want_text_input: bool, non_viewport_ui_focused: bool = False) -> bool:
    return bool(non_viewport_ui_focused) or (bool(handled) and not (bool(viewport_input_active) and not bool(want_text_input)))

def _point_in_any_rect(point: tuple[float, float] | None, rects: tuple[tuple[float, float, float, float], ...]) -> bool:
    return any(_rect_contains(rect, point) for rect in rects)

def _should_capture_mouse_for_ui(handled: bool, inside_viewport: bool, point_in_viewport_ui: bool) -> bool:
    return bool(handled) and not (bool(inside_viewport) and not bool(point_in_viewport_ui))


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


def _copy_vec2(value: object) -> imgui.ImVec2:
    return imgui.ImVec2(float(value.x), float(value.y))


def _viewer_theme_index(ui: object) -> int:
    values = getattr(ui, "_values", {})
    return min(max(int(values.get(_THEME_KEY, int(_VIEWER_CONTROL_DEFAULTS.get("theme", 0)))), 0), len(_THEME_OPTIONS) - 1)


def _theme_color(ui: object, light: tuple[float, float, float, float], dark: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return dark if _viewer_theme_index(ui) == 1 else light


def _debug_colorbar_mode(ui: "ViewerUI") -> str | None:
    index = min(max(int(ui._values.get("debug_mode", 0)), 0), len(_DEBUG_MODE_VALUES) - 1)
    mode = _DEBUG_MODE_VALUES[index]
    return None if mode in ("normal", PPISP_DEBUG_MODE, "ellipse_outlines", "sh_view_dependent", "sh_coefficient", "black_negative") else mode


def _renderer_debug_control_keys(mode: str) -> tuple[str, ...]:
    if mode == PPISP_DEBUG_MODE: return tuple(spec.key for spec in PPISP_FIELD_SPECS)
    if mode == "ellipse_outlines": return ("debug_mode", "debug_ellipse_thickness_px", "debug_gaussian_scale_multiplier", "debug_min_opacity", "debug_opacity_multiplier", "debug_ellipse_scale_multiplier")
    if mode == "grad_norm": return ("debug_mode", "debug_grad_norm_threshold")
    if mode == "sh_coefficient": return ("debug_mode", "debug_sh_coeff_index")
    if mode == "splat_age": return ("debug_mode", "debug_splat_age_min", "debug_splat_age_max")
    if mode in ("splat_density", "splat_spatial_density", "splat_screen_density"): return ("debug_mode", "debug_density_min", "debug_density_max")
    if mode == "contribution_amount": return ("debug_mode", "debug_contribution_min", "debug_contribution_max")
    if mode == "refinement_distribution": return ("debug_mode", "debug_refinement_distribution_min", "debug_refinement_distribution_max")
    if mode in ("adam_momentum", "adam_second_moment", "grad_variance"): return ("debug_mode", "debug_grad_norm_threshold")
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


def _refinement_distribution_tick_value(t: float, value_min: float, value_max: float) -> float:
    return _debug_range_tick_value(t, value_min, value_max)


def _debug_range_tick_value(t: float, value_min: float, value_max: float) -> float:
    lo = float(min(value_min, value_max))
    hi = float(max(value_min, value_max))
    return lo + _saturate(t) * (hi - lo)


def _histogram_centers(payload: object) -> np.ndarray:
    centers = getattr(payload, "bin_centers", None)
    if centers is None: centers = getattr(payload, "bin_centers_log10", np.zeros((0,), dtype=np.float64))
    return np.asarray(centers, dtype=np.float64)


def _histogram_param_value_scales(payload: object, row_count: int, default: str = PARAM_HISTOGRAM_SCALE_LINEAR) -> tuple[str, ...]:
    scales = tuple(str(scale) for scale in getattr(payload, "param_value_scales", ()))
    return scales if len(scales) == int(row_count) else (default,) * int(row_count)


def _histogram_centers_by_param(payload: object, row_count: int) -> np.ndarray:
    edges_by_param = np.asarray(getattr(payload, "bin_edges_by_param_log10", np.zeros((0, 0), dtype=np.float64)), dtype=np.float64)
    if edges_by_param.ndim == 2 and edges_by_param.shape[0] >= int(row_count) and edges_by_param.shape[1] > 1:
        return 0.5 * (edges_by_param[: int(row_count), :-1] + edges_by_param[: int(row_count), 1:])
    centers = _histogram_centers(payload)
    return np.repeat(centers[None, :], int(row_count), axis=0) if centers.size > 0 else np.zeros((int(row_count), 0), dtype=np.float64)


def _histogram_centers_for_param(payload: object, param_index: int) -> np.ndarray:
    centers = _histogram_centers_by_param(payload, int(param_index) + 1)
    return centers[int(param_index)] if centers.shape[0] > int(param_index) else np.zeros((0,), dtype=np.float64)


def _histogram_x_label_for_param(payload: object, param_index: int) -> str:
    scales = tuple(str(scale) for scale in getattr(payload, "param_value_scales", ()))
    return "log10(value)" if int(param_index) < len(scales) and scales[int(param_index)] == PARAM_HISTOGRAM_SCALE_LOG10 else "value"


def _histogram_tab_key(*parts: object) -> str:
    return "hist_tab_" + re.sub(r"[^0-9A-Za-z_]+", "_", "_".join(str(part) for part in parts)).strip("_")


def _histogram_group_type(payload: object, indices: tuple[int, ...]) -> str:
    counts = np.asarray(getattr(payload, "counts", np.zeros((0, 0), dtype=np.int64)))
    scales = _histogram_param_value_scales(payload, counts.shape[0] if counts.ndim == 2 else 0)
    valid_scales = tuple(scales[index] for index in indices if 0 <= int(index) < len(scales))
    return _HISTOGRAM_LOG10_TAB_LABEL if valid_scales and all(scale == PARAM_HISTOGRAM_SCALE_LOG10 for scale in valid_scales) else _HISTOGRAM_LINEAR_TAB_LABEL


def _histogram_range_from_histogram(payload: object) -> tuple[float, float] | None:
    if payload is None:
        return None
    counts = np.asarray(getattr(payload, "counts", np.zeros((0, 0), dtype=np.int64)), dtype=np.float64)
    if counts.ndim != 2 or counts.shape[0] == 0 or counts.shape[1] == 0:
        return None
    scales = _histogram_param_value_scales(payload, counts.shape[0])
    position_mask = np.zeros((counts.shape[0],), dtype=bool)
    for group_name, indices in tuple(getattr(payload, "param_groups", ())):
        if str(group_name) != "position":
            continue
        for index in indices:
            idx = int(index)
            if 0 <= idx < counts.shape[0]:
                position_mask[idx] = True
    if not np.any(position_mask):
        position_mask[:] = True
    row_mask = position_mask & np.asarray([scale == PARAM_HISTOGRAM_SCALE_LINEAR for scale in scales], dtype=bool)
    if not np.any(row_mask):
        return None
    centers_by_param = _histogram_centers_by_param(payload, counts.shape[0])
    if centers_by_param.shape != counts.shape:
        return None
    centers = centers_by_param[row_mask].reshape(-1)
    summed = counts[row_mask].reshape(-1)
    finite = np.isfinite(centers) & np.isfinite(summed) & (summed > 0.0)
    if not np.any(finite):
        return None
    filtered_centers = centers[finite]
    filtered_counts = summed[finite]
    order = np.argsort(filtered_centers)
    filtered_centers = filtered_centers[order]
    filtered_counts = filtered_counts[order]
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


def _histogram_range_from_ranges(payload: object) -> tuple[float, float] | None:
    if payload is None:
        return None
    min_values = np.asarray(getattr(payload, "min_values", np.zeros((0,), dtype=np.float32)), dtype=np.float64)
    max_values = np.asarray(getattr(payload, "max_values", np.zeros((0,), dtype=np.float32)), dtype=np.float64)
    if min_values.ndim != 1 or max_values.ndim != 1 or min_values.size != max_values.size or min_values.size == 0:
        return None
    scales = _histogram_param_value_scales(payload, min_values.size)
    position_mask = np.zeros((min_values.size,), dtype=bool)
    for group_name, indices in tuple(getattr(payload, "param_groups", ())):
        if str(group_name) != "position":
            continue
        for index in indices:
            idx = int(index)
            if 0 <= idx < min_values.size:
                position_mask[idx] = True
    if not np.any(position_mask):
        position_mask[:] = True
    scale_mask = position_mask & np.asarray([scale == PARAM_HISTOGRAM_SCALE_LINEAR for scale in scales], dtype=bool)
    finite_min = min_values[scale_mask & np.isfinite(min_values)]
    finite_max = max_values[scale_mask & np.isfinite(max_values)]
    if finite_min.size == 0 or finite_max.size == 0: return None
    return float(np.min(finite_min)), float(np.max(finite_max))


def _export_fields(values: dict[str, object], fields: tuple[tuple[str, object], ...]) -> dict[str, object]:
    return {key: cast(values[key]) for key, cast in fields}


def export_repo_defaults_from_ui_values(values: dict[str, object]) -> dict[str, dict[str, object]]:
    renderer_params = RendererParams.from_ui_values(values, _RENDERER_DEBUG_MODE_VALUES, _threshold_band_range)
    return {
        "renderer": json_value(renderer_params.renderer_kwargs()),
        "cli": {
            "common_render": json_value(renderer_params.cli_common_render_defaults_dict())
        },
        "viewer": {
            "controls": json_value(_export_fields(values, _VIEWER_CONTROL_EXPORT_FIELDS)),
            "import": json_value(_export_fields(values, _VIEWER_IMPORT_EXPORT_FIELDS)),
            "ui": json_value({
                **_export_fields(values, _VIEWER_UI_EXPORT_FIELDS[:-3]),
                "viewport_sh_band": int(values["_viewport_sh_band"]),
                "viewport_sh_control_key": str(values["_viewport_sh_control_key"]),
                "viewport_sh_stage_label": str(values["_viewport_sh_stage_label"]),
            }),
        },
    }


RENDER_PARAM_SPECS, DEBUG_RENDER_SPECS, _ALL_DEFAULTS = build_render_spec_bundle(_threshold_from_band_range)
_DEBUG_VIEW_CONTROL_SPECS = {spec.key: spec for spec in DEBUG_RENDER_SPECS + GROUP_SPECS["PPISP Debug"]}


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
    _control_proxies: dict[str, _ControlProxy] = field(default_factory=dict, init=False, repr=False)
    _text_proxies: dict[str, _TextProxy] = field(default_factory=dict, init=False, repr=False)
    _control_proxy_store_id: int = field(default=0, init=False, repr=False)
    _text_proxy_store_id: int = field(default=0, init=False, repr=False)
    _control_proxy_key_count: int = field(default=-1, init=False, repr=False)
    _text_proxy_key_count: int = field(default=-1, init=False, repr=False)

    @staticmethod
    def _sync_proxy_cache(store: dict[str, object], cache: dict[str, object], factory) -> dict[str, object]:
        for key in tuple(cache):
            if key not in store:
                del cache[key]
        for key in store:
            if key not in cache:
                cache[key] = factory(store, key)
        return cache

    def _control_cache(self) -> dict[str, _ControlProxy]:
        values = self._values
        value_count = len(values)
        if self._control_proxy_store_id != id(values) or self._control_proxy_key_count != value_count:
            ViewerUI._sync_proxy_cache(values, self._control_proxies, _ControlProxy)
            self._control_proxy_store_id = id(values)
            self._control_proxy_key_count = value_count
        return self._control_proxies

    def _text_cache(self) -> dict[str, _TextProxy]:
        texts = self._texts
        text_count = len(texts)
        if self._text_proxy_store_id != id(texts) or self._text_proxy_key_count != text_count:
            ViewerUI._sync_proxy_cache(texts, self._text_proxies, _TextProxy)
            self._text_proxy_store_id = id(texts)
            self._text_proxy_key_count = text_count
        return self._text_proxies

    @property
    def controls(self):
        return self._control_cache()

    @property
    def texts(self):
        return self._text_cache()

    def control(self, key: str) -> _ControlProxy:
        proxy = self._control_cache().get(key)
        if proxy is None:
            proxy = _ControlProxy(self._values, key)
            self._control_proxies[key] = proxy
            self._control_proxy_key_count = len(self._values)
        return proxy

    def text(self, key: str) -> _TextProxy:
        proxy = self._text_cache().get(key)
        if proxy is None:
            proxy = _TextProxy(self._texts, key)
            self._text_proxies[key] = proxy
            self._text_proxy_key_count = len(self._texts)
        return proxy


def _control_bound(ui: ViewerUI, spec: ControlSpec, key: str, fallback: int) -> int:
    bound_key = spec.kwargs.get(key)
    if bound_key is None:
        return fallback
    return int(ui._values.get(str(bound_key), _ALL_DEFAULTS.get(str(bound_key), fallback)))


@dataclass(slots=True)
class ToolkitState:
    loss_history: deque = field(default_factory=partial(deque, maxlen=_LOSS_HISTORY_BUCKET_COUNT))
    fps_history: deque = field(default_factory=partial(deque, maxlen=FPS_HISTORY_SIZE))
    ssim_history: deque = field(default_factory=partial(deque, maxlen=_LOSS_HISTORY_BUCKET_COUNT))
    psnr_history: deque = field(default_factory=partial(deque, maxlen=_LOSS_HISTORY_BUCKET_COUNT))
    step_history: deque = field(default_factory=partial(deque, maxlen=_LOSS_HISTORY_BUCKET_COUNT))
    step_time_history: deque = field(default_factory=partial(deque, maxlen=_LOSS_HISTORY_BUCKET_COUNT))
    photometric_loss_history: deque = field(default_factory=partial(deque, maxlen=_LOSS_HISTORY_BUCKET_COUNT))
    photometric_step_history: deque = field(default_factory=partial(deque, maxlen=_LOSS_HISTORY_BUCKET_COUNT))
    photometric_step_time_history: deque = field(default_factory=partial(deque, maxlen=_LOSS_HISTORY_BUCKET_COUNT))
    _plot_bucket_index: int = -1
    _plot_bucket_count: int = 0
    _plot_bucket_step_sum: float = 0.0
    _plot_bucket_time_sum: float = 0.0
    _plot_bucket_loss_sum: float = 0.0
    _plot_bucket_ssim_sum: float = 0.0
    _plot_bucket_psnr_sum: float = 0.0
    _photometric_plot_bucket_index: int = -1
    _photometric_plot_bucket_count: int = 0
    _photometric_plot_bucket_step_sum: float = 0.0
    _photometric_plot_bucket_time_sum: float = 0.0
    _photometric_plot_bucket_loss_sum: float = 0.0

    def _reset_plot_bucket(self) -> None:
        self._plot_bucket_index = -1
        self._plot_bucket_count = 0
        self._plot_bucket_step_sum = 0.0
        self._plot_bucket_time_sum = 0.0
        self._plot_bucket_loss_sum = 0.0
        self._plot_bucket_ssim_sum = 0.0
        self._plot_bucket_psnr_sum = 0.0

    def _reset_photometric_plot_bucket(self) -> None:
        self._photometric_plot_bucket_index = -1
        self._photometric_plot_bucket_count = 0
        self._photometric_plot_bucket_step_sum = 0.0
        self._photometric_plot_bucket_time_sum = 0.0
        self._photometric_plot_bucket_loss_sum = 0.0

    def append_training_plot_sample(self, step: int, timestamp: float, loss: float, ssim: float, psnr: float) -> None:
        bucket_index = max(int(step) - 1, 0) // _LOSS_HISTORY_BUCKET_SIZE
        if bucket_index != self._plot_bucket_index or self._plot_bucket_count <= 0 or not self.step_history:
            self._plot_bucket_index = bucket_index
            self._plot_bucket_count = 0
            self._plot_bucket_step_sum = 0.0
            self._plot_bucket_time_sum = 0.0
            self._plot_bucket_loss_sum = 0.0
            self._plot_bucket_ssim_sum = 0.0
            self._plot_bucket_psnr_sum = 0.0
        self._plot_bucket_count += 1
        count = float(self._plot_bucket_count)
        self._plot_bucket_step_sum += float(step)
        self._plot_bucket_time_sum += float(timestamp)
        self._plot_bucket_loss_sum += float(loss)
        self._plot_bucket_ssim_sum += float(ssim)
        self._plot_bucket_psnr_sum += float(psnr)
        averaged_step = self._plot_bucket_step_sum / count
        averaged_time = self._plot_bucket_time_sum / count
        averaged_loss = self._plot_bucket_loss_sum / count
        averaged_ssim = self._plot_bucket_ssim_sum / count
        averaged_psnr = self._plot_bucket_psnr_sum / count
        if self._plot_bucket_count == 1:
            self.step_history.append(averaged_step)
            self.step_time_history.append(averaged_time)
            self.loss_history.append(averaged_loss)
            self.ssim_history.append(averaged_ssim)
            self.psnr_history.append(averaged_psnr)
            return
        self.step_history[-1] = averaged_step
        self.step_time_history[-1] = averaged_time
        self.loss_history[-1] = averaged_loss
        self.ssim_history[-1] = averaged_ssim
        self.psnr_history[-1] = averaged_psnr

    def append_photometric_plot_sample(self, step: int, timestamp: float, loss: float) -> None:
        bucket_index = max(int(step) - 1, 0) // _LOSS_HISTORY_BUCKET_SIZE
        if bucket_index != self._photometric_plot_bucket_index or self._photometric_plot_bucket_count <= 0 or not self.photometric_step_history:
            self._photometric_plot_bucket_index = bucket_index
            self._photometric_plot_bucket_count = 0
            self._photometric_plot_bucket_step_sum = 0.0
            self._photometric_plot_bucket_time_sum = 0.0
            self._photometric_plot_bucket_loss_sum = 0.0
        self._photometric_plot_bucket_count += 1
        count = float(self._photometric_plot_bucket_count)
        self._photometric_plot_bucket_step_sum += float(step)
        self._photometric_plot_bucket_time_sum += float(timestamp)
        self._photometric_plot_bucket_loss_sum += float(loss)
        averaged_step = self._photometric_plot_bucket_step_sum / count
        averaged_time = self._photometric_plot_bucket_time_sum / count
        averaged_loss = self._photometric_plot_bucket_loss_sum / count
        if self._photometric_plot_bucket_count == 1:
            self.photometric_step_history.append(averaged_step)
            self.photometric_step_time_history.append(averaged_time)
            self.photometric_loss_history.append(averaged_loss)
            return
        self.photometric_step_history[-1] = averaged_step
        self.photometric_step_time_history[-1] = averaged_time
        self.photometric_loss_history[-1] = averaged_loss

    def clear_photometric_plot_history(self) -> None:
        self.photometric_loss_history.clear()
        self.photometric_step_history.clear()
        self.photometric_step_time_history.clear()
        self._reset_photometric_plot_bucket()

    def clear_plot_history(self) -> None:
        self.loss_history.clear()
        self.fps_history.clear()
        self.ssim_history.clear()
        self.psnr_history.clear()
        self.step_history.clear()
        self.step_time_history.clear()
        self._reset_plot_bucket()
        self.clear_photometric_plot_history()


def _noop() -> None:
    return None

class ToolkitWindow:
    """Dear ImGui overlay rendered into the active Slangpy AppWindow surface."""

    @staticmethod
    def _format_plot_metric_value(value: float, digits: int = 6) -> str:
        return "n/a" if not np.isfinite(value) else f"{float(value):.{max(int(digits), 1)}g}"

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
        self.callbacks = SimpleNamespace(
            load_ply=_noop,
            export_ply=_noop,
            browse_colmap_root=_noop,
            browse_colmap_images=_noop,
            browse_colmap_depth=_noop,
            browse_colmap_ply=_noop,
            browse_colmap_mesh=_noop,
            import_colmap=_noop,
            reload=_noop,
            reinitialize=_noop,
            request_exit=_noop,
            confirm_exit=_noop,
            cancel_exit=_noop,
            start_training=_noop,
            stop_training=_noop,
            start_photometric=_noop,
            stop_photometric=_noop,
            reset_photometric=_noop,
            move_to_training_camera=_noop,
            reset_camera=_noop,
            save_defaults=_noop,
        )
        self.tk = ToolkitState()
        self._alive = True
        self._capture_non_viewport_windows = True
        self._frame_textures: list[spy.Texture] = []
        self._last_frame_time = time.perf_counter()
        self._show_about = False
        self._show_docs = False
        self._show_colmap_import = False
        self._about_text = _build_about_text()
        self._documentation_text = _build_documentation_text()
        self._menu_bar_height = 0.0
        self._applied_interface_scale = -1.0
        self._applied_theme_index = -1
        self._dockspace_id = 0
        self._dock_layout_initialized = False
        self._viewport_dock_id = 0
        self._toolkit_dock_id = 0
        self._toolkit_window_open = True
        initial_scale = _INTERFACE_SCALE_OPTIONS[_DEFAULT_INTERFACE_SCALE_INDEX][1]
        initial_panel_width = _toolkit_panel_width(width, initial_scale)
        self._toolkit_rect = (
            max(float(width) - initial_panel_width, 0.0),
            0.0,
            initial_panel_width,
            max(float(height), 1.0),
        )
        self._viewport_rect = (0.0, 0.0, max(float(width) - self._toolkit_rect[2], 1.0), max(float(height), 1.0))
        self._viewport_content_rect = self._viewport_rect
        self._viewport_ui_capture_rects: tuple[tuple[float, float, float, float], ...] = ()
        self._non_viewport_ui_capture_rects: tuple[tuple[float, float, float, float], ...] = ()
        self._non_viewport_ui_focused = False
        self._viewport_window_focused = False
        self._viewport_input_active = False
        self._training_camera_view_zoom = 1.0
        self._training_camera_view_center = (0.5, 0.5)
        self._training_camera_selected_point_id: int | None = None
        self._training_camera_pending_point_focus: tuple[int, int, float, float] | None = None
        self._base_style = imgui.Style()
        self._apply_visual_state(initial_scale, int(_VIEWER_CONTROL_DEFAULTS.get("theme", 0)))

    def _set_current_context(self) -> None:
        imgui.set_current_context(self.ctx)

    def reset_plot_history(self) -> None:
        self.tk.clear_plot_history()

    def _restore_base_style(self) -> None:
        style = imgui.get_style()
        base = self._base_style
        scalar_fields = (
            "alpha",
            "disabled_alpha",
            "window_rounding",
            "window_border_size",
            "child_rounding",
            "child_border_size",
            "popup_rounding",
            "popup_border_size",
            "frame_rounding",
            "frame_border_size",
            "indent_spacing",
            "columns_min_spacing",
            "scrollbar_size",
            "scrollbar_padding",
            "scrollbar_rounding",
            "grab_min_size",
            "grab_rounding",
            "log_slider_deadzone",
            "image_border_size",
            "image_rounding",
            "tab_rounding",
            "tab_border_size",
            "tab_bar_border_size",
            "tab_bar_overline_size",
            "tab_min_width_base",
            "tab_min_width_shrink",
            "tab_close_button_min_width_selected",
            "tab_close_button_min_width_unselected",
            "table_angled_headers_angle",
            "separator_text_border_size",
            "mouse_cursor_scale",
            "curve_tessellation_tol",
            "circle_tessellation_max_error",
            "hover_stationary_delay",
            "hover_delay_short",
            "hover_delay_normal",
            "color_marker_size",
            "window_border_hover_padding",
            "docking_separator_size",
            "drag_drop_target_padding",
            "drag_drop_target_rounding",
            "drag_drop_target_border_size",
            "tree_lines_size",
            "tree_lines_rounding",
            "font_scale_main",
            "font_scale_dpi",
            "font_size_base",
        )
        vec2_fields = (
            "window_padding",
            "window_min_size",
            "window_title_align",
            "frame_padding",
            "item_spacing",
            "item_inner_spacing",
            "cell_padding",
            "touch_extra_padding",
            "button_text_align",
            "selectable_text_align",
            "separator_text_align",
            "separator_text_padding",
            "display_window_padding",
            "display_safe_area_padding",
            "table_angled_headers_text_align",
        )
        int_fields = (
            "window_menu_button_position",
            "color_button_position",
            "hover_flags_for_tooltip_mouse",
            "hover_flags_for_tooltip_nav",
            "layout_align",
            "tree_lines_flags",
        )
        bool_fields = (
            "anti_aliased_lines",
            "anti_aliased_lines_use_tex",
            "anti_aliased_fill",
            "docking_node_has_close_button",
        )
        for field in scalar_fields:
            setattr(style, field, getattr(base, field))
        for field in vec2_fields:
            setattr(style, field, _copy_vec2(getattr(base, field)))
        for field in int_fields:
            setattr(style, field, int(getattr(base, field)))
        for field in bool_fields:
            setattr(style, field, bool(getattr(base, field)))
        for color_idx in range(int(imgui.Col_.count)):
            color = base.color_(imgui.Col_(color_idx))
            style.set_color_(imgui.Col_(color_idx), imgui.ImVec4(float(color.x), float(color.y), float(color.z), float(color.w)))

    def _apply_theme(self, theme_index: int) -> None:
        self._set_current_context()
        self._restore_base_style()
        if int(theme_index) == 1:
            imgui.style_colors_dark()
        else:
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
        if int(theme_index) == 1:
            _c(imgui.Col_.text, imgui.ImVec4(0.90, 0.93, 0.97, 1.00))
            _c(imgui.Col_.text_disabled, imgui.ImVec4(0.55, 0.61, 0.69, 1.00))
            _c(imgui.Col_.window_bg, imgui.ImVec4(0.11, 0.12, 0.14, 1.00))
            _c(imgui.Col_.child_bg, imgui.ImVec4(0.13, 0.14, 0.17, 0.98))
            _c(imgui.Col_.popup_bg, imgui.ImVec4(0.14, 0.15, 0.18, 0.99))
            _c(imgui.Col_.border, imgui.ImVec4(0.32, 0.36, 0.43, 0.82))
            _c(imgui.Col_.border_shadow, imgui.ImVec4(0.0, 0.0, 0.0, 0.0))
            _c(imgui.Col_.title_bg, imgui.ImVec4(0.16, 0.20, 0.25, 1.0))
            _c(imgui.Col_.title_bg_active, imgui.ImVec4(0.21, 0.28, 0.36, 1.0))
            _c(imgui.Col_.title_bg_collapsed, imgui.ImVec4(0.14, 0.17, 0.21, 0.92))
            _c(imgui.Col_.menu_bar_bg, imgui.ImVec4(0.15, 0.18, 0.22, 1.0))
            _c(imgui.Col_.header, imgui.ImVec4(0.24, 0.39, 0.59, 0.68))
            _c(imgui.Col_.header_hovered, imgui.ImVec4(0.31, 0.49, 0.73, 0.86))
            _c(imgui.Col_.header_active, imgui.ImVec4(0.36, 0.55, 0.80, 1.00))
            _c(imgui.Col_.button, imgui.ImVec4(0.23, 0.42, 0.67, 0.92))
            _c(imgui.Col_.button_hovered, imgui.ImVec4(0.29, 0.50, 0.78, 0.96))
            _c(imgui.Col_.button_active, imgui.ImVec4(0.35, 0.58, 0.85, 1.00))
            _c(imgui.Col_.tab, imgui.ImVec4(0.17, 0.22, 0.28, 1.00))
            _c(imgui.Col_.tab_hovered, imgui.ImVec4(0.27, 0.42, 0.64, 0.92))
            _c(imgui.Col_.tab_selected, imgui.ImVec4(0.23, 0.42, 0.67, 1.00))
            _c(imgui.Col_.tab_dimmed, imgui.ImVec4(0.13, 0.17, 0.22, 1.00))
            _c(imgui.Col_.tab_dimmed_selected, imgui.ImVec4(0.19, 0.28, 0.41, 1.00))
            _c(imgui.Col_.tab_selected_overline, imgui.ImVec4(0.48, 0.73, 0.98, 1.00))
            _c(imgui.Col_.tab_dimmed_selected_overline, imgui.ImVec4(0.35, 0.56, 0.80, 0.85))
            _c(imgui.Col_.separator, imgui.ImVec4(0.32, 0.36, 0.43, 0.82))
            _c(imgui.Col_.separator_hovered, imgui.ImVec4(0.42, 0.57, 0.78, 0.85))
            _c(imgui.Col_.separator_active, imgui.ImVec4(0.52, 0.69, 0.92, 0.96))
            _c(imgui.Col_.plot_lines, imgui.ImVec4(0.45, 0.72, 0.98, 1.00))
            _c(imgui.Col_.plot_lines_hovered, imgui.ImVec4(0.68, 0.84, 1.00, 1.00))
            _c(imgui.Col_.plot_histogram, imgui.ImVec4(0.38, 0.66, 0.96, 0.92))
            _c(imgui.Col_.plot_histogram_hovered, imgui.ImVec4(0.58, 0.79, 1.00, 1.00))
            _c(imgui.Col_.frame_bg, imgui.ImVec4(0.15, 0.17, 0.20, 1.00))
            _c(imgui.Col_.frame_bg_hovered, imgui.ImVec4(0.18, 0.22, 0.27, 1.00))
            _c(imgui.Col_.frame_bg_active, imgui.ImVec4(0.20, 0.26, 0.33, 1.00))
            _c(imgui.Col_.slider_grab, imgui.ImVec4(0.41, 0.66, 0.96, 0.82))
            _c(imgui.Col_.slider_grab_active, imgui.ImVec4(0.52, 0.74, 0.99, 1.00))
            _c(imgui.Col_.check_mark, imgui.ImVec4(0.47, 0.73, 0.98, 1.00))
            _c(imgui.Col_.scrollbar_bg, imgui.ImVec4(0.09, 0.11, 0.13, 1.00))
            _c(imgui.Col_.scrollbar_grab, imgui.ImVec4(0.25, 0.30, 0.37, 0.85))
            _c(imgui.Col_.scrollbar_grab_hovered, imgui.ImVec4(0.33, 0.40, 0.49, 0.92))
            _c(imgui.Col_.scrollbar_grab_active, imgui.ImVec4(0.41, 0.49, 0.60, 1.00))
            _c(imgui.Col_.resize_grip, imgui.ImVec4(0.33, 0.51, 0.74, 0.28))
            _c(imgui.Col_.resize_grip_hovered, imgui.ImVec4(0.42, 0.63, 0.90, 0.72))
            _c(imgui.Col_.resize_grip_active, imgui.ImVec4(0.52, 0.74, 0.99, 0.95))
            _c(imgui.Col_.text_selected_bg, imgui.ImVec4(0.31, 0.49, 0.73, 0.42))
            _c(imgui.Col_.text_link, imgui.ImVec4(0.58, 0.80, 1.00, 1.00))
            _c(imgui.Col_.table_header_bg, imgui.ImVec4(0.18, 0.22, 0.27, 1.00))
            _c(imgui.Col_.table_border_strong, imgui.ImVec4(0.33, 0.39, 0.48, 0.92))
            _c(imgui.Col_.table_border_light, imgui.ImVec4(0.25, 0.30, 0.37, 0.82))
            _c(imgui.Col_.table_row_bg_alt, imgui.ImVec4(0.14, 0.16, 0.19, 0.72))
            _c(imgui.Col_.docking_preview, imgui.ImVec4(0.32, 0.54, 0.82, 0.62))
            _c(imgui.Col_.docking_empty_bg, imgui.ImVec4(0.09, 0.10, 0.12, 1.00))
            _c(imgui.Col_.drag_drop_target, imgui.ImVec4(0.55, 0.80, 1.00, 0.96))
            _c(imgui.Col_.drag_drop_target_bg, imgui.ImVec4(0.27, 0.46, 0.71, 0.32))
            _c(imgui.Col_.nav_cursor, imgui.ImVec4(0.48, 0.73, 0.98, 1.00))
            _c(imgui.Col_.nav_windowing_highlight, imgui.ImVec4(0.48, 0.73, 0.98, 1.00))
            _c(imgui.Col_.nav_windowing_dim_bg, imgui.ImVec4(0.0, 0.0, 0.0, 0.42))
            _c(imgui.Col_.modal_window_dim_bg, imgui.ImVec4(0.0, 0.0, 0.0, 0.52))
            _c(imgui.Col_.unsaved_marker, imgui.ImVec4(0.90, 0.66, 0.20, 1.00))
            _c(imgui.Col_.input_text_cursor, imgui.ImVec4(0.72, 0.82, 0.96, 1.00))
        else:
            _c(imgui.Col_.text, imgui.ImVec4(0.12, 0.16, 0.22, 1.00))
            _c(imgui.Col_.text_disabled, imgui.ImVec4(0.28, 0.33, 0.41, 1.00))
            _c(imgui.Col_.window_bg, imgui.ImVec4(0.965, 0.970, 0.978, 1.00))
            _c(imgui.Col_.child_bg, imgui.ImVec4(0.975, 0.980, 0.988, 0.98))
            _c(imgui.Col_.popup_bg, imgui.ImVec4(0.995, 0.997, 1.000, 0.99))
            _c(imgui.Col_.border, imgui.ImVec4(0.66, 0.71, 0.79, 0.82))
            _c(imgui.Col_.border_shadow, imgui.ImVec4(0.0, 0.0, 0.0, 0.0))
            _c(imgui.Col_.title_bg, imgui.ImVec4(0.84, 0.90, 0.96, 1.0))
            _c(imgui.Col_.title_bg_active, imgui.ImVec4(0.76, 0.85, 0.95, 1.0))
            _c(imgui.Col_.title_bg_collapsed, imgui.ImVec4(0.89, 0.93, 0.97, 0.96))
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
            _c(imgui.Col_.tab_dimmed, imgui.ImVec4(0.90, 0.94, 0.98, 1.00))
            _c(imgui.Col_.tab_dimmed_selected, imgui.ImVec4(0.72, 0.84, 0.96, 1.00))
            _c(imgui.Col_.tab_selected_overline, imgui.ImVec4(0.27, 0.54, 0.86, 1.00))
            _c(imgui.Col_.tab_dimmed_selected_overline, imgui.ImVec4(0.40, 0.63, 0.90, 0.90))
            _c(imgui.Col_.separator, imgui.ImVec4(0.66, 0.71, 0.79, 0.82))
            _c(imgui.Col_.separator_hovered, imgui.ImVec4(0.38, 0.60, 0.85, 0.85))
            _c(imgui.Col_.separator_active, imgui.ImVec4(0.22, 0.48, 0.82, 0.96))
            _c(imgui.Col_.plot_lines, imgui.ImVec4(0.18, 0.48, 0.82, 1.00))
            _c(imgui.Col_.plot_lines_hovered, imgui.ImVec4(0.10, 0.40, 0.78, 1.00))
            _c(imgui.Col_.plot_histogram, imgui.ImVec4(0.28, 0.56, 0.88, 0.92))
            _c(imgui.Col_.plot_histogram_hovered, imgui.ImVec4(0.16, 0.48, 0.84, 1.00))
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
            _c(imgui.Col_.text_link, imgui.ImVec4(0.16, 0.48, 0.82, 1.00))
            _c(imgui.Col_.table_header_bg, imgui.ImVec4(0.89, 0.93, 0.97, 1.00))
            _c(imgui.Col_.table_border_strong, imgui.ImVec4(0.63, 0.70, 0.79, 0.90))
            _c(imgui.Col_.table_border_light, imgui.ImVec4(0.77, 0.82, 0.89, 0.82))
            _c(imgui.Col_.table_row_bg_alt, imgui.ImVec4(0.94, 0.97, 1.00, 0.62))
            _c(imgui.Col_.docking_preview, imgui.ImVec4(0.35, 0.61, 0.90, 0.36))
            _c(imgui.Col_.docking_empty_bg, imgui.ImVec4(0.95, 0.97, 0.99, 1.00))
            _c(imgui.Col_.drag_drop_target, imgui.ImVec4(0.16, 0.48, 0.82, 0.96))
            _c(imgui.Col_.drag_drop_target_bg, imgui.ImVec4(0.48, 0.71, 0.95, 0.26))
            _c(imgui.Col_.nav_cursor, imgui.ImVec4(0.22, 0.48, 0.82, 1.00))
            _c(imgui.Col_.nav_windowing_highlight, imgui.ImVec4(0.22, 0.48, 0.82, 1.00))
            _c(imgui.Col_.nav_windowing_dim_bg, imgui.ImVec4(0.0, 0.0, 0.0, 0.15))
            _c(imgui.Col_.modal_window_dim_bg, imgui.ImVec4(0.0, 0.0, 0.0, 0.18))
            _c(imgui.Col_.unsaved_marker, imgui.ImVec4(0.90, 0.63, 0.08, 1.00))
            _c(imgui.Col_.input_text_cursor, imgui.ImVec4(0.16, 0.40, 0.74, 1.00))
        self._applied_theme_index = int(theme_index)

    def _configure_default_font(self) -> None:
        io = imgui.get_io()
        atlas = io.fonts
        atlas.clear()
        font_path = _default_font_path()
        io.font_default = atlas.add_font_from_file_ttf(str(font_path), _FONT_ATLAS_SIZE_PX) if font_path is not None else atlas.add_font_default()
        try:
            markdown_options = imgui_md.MarkdownOptions()
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

    def _theme_index(self, ui: ViewerUI) -> int:
        return min(max(int(ui._values.get(_THEME_KEY, int(_VIEWER_CONTROL_DEFAULTS.get("theme", 0)))), 0), len(_THEME_OPTIONS) - 1)

    def _set_interface_scale(self, scale: float) -> None:
        self._apply_visual_state(scale, self._applied_theme_index if self._applied_theme_index >= 0 else int(_VIEWER_CONTROL_DEFAULTS.get("theme", 0)))

    def _plot_scale(self, ui: ViewerUI) -> float:
        return self._interface_scale_factor(ui)

    def _apply_visual_state(self, scale: float, theme_index: int) -> None:
        clamped_scale = max(float(scale), 0.5)
        resolved_theme_index = min(max(int(theme_index), 0), len(_THEME_OPTIONS) - 1)
        if abs(clamped_scale - self._applied_interface_scale) <= 1e-6 and resolved_theme_index == self._applied_theme_index:
            return
        self._set_current_context()
        scale_changed = abs(clamped_scale - self._applied_interface_scale) > 1e-6
        self._apply_theme(resolved_theme_index)
        imgui.get_style().scale_all_sizes(clamped_scale)
        style = imgui.get_style()
        style.font_scale_main = clamped_scale * (_BASE_FONT_SIZE_PX / _FONT_ATLAS_SIZE_PX)
        self._applied_interface_scale = clamped_scale
        if scale_changed:
            self._dock_layout_initialized = False

    def _sync_interface_scale(self, ui: ViewerUI) -> None:
        self._apply_visual_state(self._interface_scale_factor(ui), self._theme_index(ui))

    def handle_keyboard_event(self, event) -> bool:
        if not self._alive:
            return False
        self._set_current_context()
        handled = bool(simgui.handle_keyboard_event(event))
        return _should_capture_keyboard_for_ui(handled, bool(getattr(self, "_viewport_input_active", False)), bool(imgui.get_io().want_text_input), bool(getattr(self, "_non_viewport_ui_focused", False)))

    def handle_mouse_event(self, event) -> bool:
        if not self._alive:
            return False
        self._set_current_context()
        handled = bool(simgui.handle_mouse_event(event))
        pos = getattr(event, "pos", None)
        point = None if pos is None else (float(pos.x), float(pos.y))
        inside_viewport = _rect_contains(self._viewport_content_rect, point)
        point_in_viewport_ui = _point_in_any_rect(point, self._viewport_ui_capture_rects)
        point_in_non_viewport_ui = _point_in_any_rect(point, getattr(self, "_non_viewport_ui_capture_rects", ()))
        event_type = getattr(event, "type", None)
        if point_in_non_viewport_ui:
            self._viewport_input_active = False
        elif inside_viewport and event_type in (spy.MouseEventType.button_down, spy.MouseEventType.move, spy.MouseEventType.scroll):
            self._viewport_input_active = True
        elif event_type == spy.MouseEventType.button_down and not inside_viewport:
            self._viewport_input_active = False
        return _should_capture_mouse_for_ui(handled, inside_viewport, point_in_viewport_ui or point_in_non_viewport_ui)

    def _append_viewport_ui_capture_rect(self, rect: tuple[float, float, float, float]) -> None:
        x, y, width, height = rect
        if width <= 0.0 or height <= 0.0:
            return
        self._viewport_ui_capture_rects = (*self._viewport_ui_capture_rects, (float(x), float(y), float(width), float(height)))

    def _append_non_viewport_ui_capture_rect(self, rect: tuple[float, float, float, float]) -> None:
        x, y, width, height = rect
        if width <= 0.0 or height <= 0.0:
            return
        self._non_viewport_ui_capture_rects = (*self._non_viewport_ui_capture_rects, (float(x), float(y), float(width), float(height)))

    def _register_non_viewport_window(self) -> None:
        if not bool(getattr(self, "_capture_non_viewport_windows", False)):
            return
        try:
            pos = imgui.get_window_pos()
            size = imgui.get_window_size()
            ToolkitWindow._append_non_viewport_ui_capture_rect(self, (float(pos.x), float(pos.y), float(size.x), float(size.y)))
        except Exception:
            pass
        try:
            focused = bool(imgui.is_window_focused(int(imgui.FocusedFlags_.root_and_child_windows)))
        except Exception:
            focused = False
        self._non_viewport_ui_focused = bool(self._non_viewport_ui_focused) or focused

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
        self._non_viewport_ui_capture_rects = ()
        self._non_viewport_ui_focused = False
        self._menu_bar_height = self._draw_main_menu_bar(ui)
        self._draw_dockspace()
        self._draw_panel(ui, width, height)
        self._draw_viewport_window(ui, viewport_texture, width, height)
        self._draw_debug_colorbar(ui)
        self._draw_histogram_window(ui)
        self._draw_photometric_compensation_window(ui)
        self._draw_training_views_window(ui)
        self._draw_resource_debug_window(ui)
        self._draw_exit_confirmation_modal(ui)
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
        split_fraction = _toolkit_width_fraction(float(viewport.work_size.x), self._applied_interface_scale if self._applied_interface_scale > 0.0 else 1.0)
        split_ids = tuple(int(node_id) for node_id in imgui.internal.dock_builder_split_node_py(root_id, imgui.Dir_.right, split_fraction))
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
        panel_x, panel_y, panel_width, panel_height = _panel_rect(width, height, self._menu_bar_height, self._interface_scale_factor(ui))
        imgui.set_next_window_pos(imgui.ImVec2(panel_x, panel_y), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(panel_width, panel_height), imgui.Cond_.first_use_ever.value)
        if self._toolkit_dock_id != 0:
            imgui.set_next_window_dock_id(self._toolkit_dock_id, imgui.Cond_.always.value)
        elif self._dockspace_id != 0:
            imgui.set_next_window_dock_id(self._dockspace_id, imgui.Cond_.first_use_ever.value)
        flags = imgui.WindowFlags_.no_collapse.value
        opened, self._toolkit_window_open = imgui.begin(_TOOLKIT_WINDOW_NAME, self._toolkit_window_open, flags=flags)
        ToolkitWindow._register_non_viewport_window(self)
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
            self._section_defaults_footer(ui)
        imgui.end()
        self._draw_about_window()
        self._draw_documentation_window()
        self._draw_colmap_import_window(ui)

    def _reset_training_camera_view(self) -> None:
        self._training_camera_view_zoom = 1.0
        self._training_camera_view_center = (0.5, 0.5)
        self._training_camera_selected_point_id = None
        self._training_camera_pending_point_focus = None

    def _apply_pending_training_camera_point_focus(self, ui: ViewerUI) -> None:
        pending_focus = getattr(self, "_training_camera_pending_point_focus", None)
        if pending_focus is None:
            return
        payload = ui._values.get("_training_camera_colmap_points_payload")
        if not isinstance(payload, dict):
            return
        target_image_id, point_id, focus_u, focus_v = pending_focus
        if int(payload.get("image_id", -1)) != int(target_image_id):
            return
        self._training_camera_view_center = _clamp_training_camera_center(
            float(focus_u),
            float(focus_v),
            self._training_camera_view_zoom,
        )
        self._training_camera_selected_point_id = int(point_id)
        self._training_camera_pending_point_focus = None

    def _focus_training_camera_point(self, ui: ViewerUI, frame_index: int, image_id: int, point_id: int, focus_u: float, focus_v: float) -> None:
        ui._values["loss_debug_frame"] = int(frame_index)
        self._training_camera_selected_point_id = int(point_id)
        self._training_camera_pending_point_focus = (int(image_id), int(point_id), float(focus_u), float(focus_v))
        ToolkitWindow._apply_pending_training_camera_point_focus(self, ui)

    def _draw_training_camera_viewport_image(self, ui: ViewerUI, viewport_texture: spy.Texture, content_width: float, content_height: float) -> None:
        texture_width = max(int(getattr(viewport_texture, "width", 1)), 1)
        texture_height = max(int(getattr(viewport_texture, "height", 1)), 1)
        origin = imgui.get_cursor_screen_pos()
        offset_x, offset_y, draw_width, draw_height = _fit_aspect_rect(content_width, content_height, texture_width, texture_height)
        image_origin = imgui.ImVec2(float(origin.x) + float(offset_x), float(origin.y) + float(offset_y))
        ToolkitWindow._apply_pending_training_camera_point_focus(self, ui)
        uv0, uv1 = _training_camera_uv_bounds(self._training_camera_view_center[0], self._training_camera_view_center[1], self._training_camera_view_zoom)
        imgui.set_cursor_screen_pos(image_origin)
        imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(0.0, 0.0))
        imgui.push_style_color(imgui.Col_.button.value, imgui.ImVec4(0.0, 0.0, 0.0, 0.0))
        imgui.push_style_color(imgui.Col_.button_hovered.value, imgui.ImVec4(0.0, 0.0, 0.0, 0.0))
        imgui.push_style_color(imgui.Col_.button_active.value, imgui.ImVec4(0.0, 0.0, 0.0, 0.0))
        imgui.set_next_item_allow_overlap()
        imgui.image_button(
            "##training_camera_viewport",
            simgui.texture_ref(viewport_texture),
            imgui.ImVec2(float(draw_width), float(draw_height)),
            imgui.ImVec2(*uv0),
            imgui.ImVec2(*uv1),
            imgui.ImVec4(0.0, 0.0, 0.0, 0.0),
            imgui.ImVec4(1.0, 1.0, 1.0, 1.0),
        )
        hovered = bool(imgui.is_item_hovered())
        active = bool(imgui.is_item_active())
        ToolkitWindow._draw_training_camera_colmap_overlay(self, ui, image_origin, draw_width, draw_height, uv0, uv1, hovered)
        if hovered and bool(imgui.is_mouse_double_clicked(0)):
            self._training_camera_view_zoom = 1.0
            self._training_camera_view_center = (0.5, 0.5)
        io = imgui.get_io()
        if hovered and abs(float(getattr(io, "mouse_wheel", 0.0))) > 1e-6:
            mouse_pos = imgui.get_mouse_pos()
            local_x = min(max((float(mouse_pos.x) - float(image_origin.x)) / max(float(draw_width), 1.0), 0.0), 1.0)
            local_y = min(max((float(mouse_pos.y) - float(image_origin.y)) / max(float(draw_height), 1.0), 0.0), 1.0)
            span = 1.0 / max(float(self._training_camera_view_zoom), 1.0)
            hover_u = float(uv0[0]) + local_x * span
            hover_v = float(uv0[1]) + local_y * span
            new_zoom = min(max(float(self._training_camera_view_zoom) * pow(1.2, float(io.mouse_wheel)), 1.0), 32.0)
            new_span = 1.0 / new_zoom
            self._training_camera_view_center = _clamp_training_camera_center(
                hover_u + (0.5 - local_x) * new_span,
                hover_v + (0.5 - local_y) * new_span,
                new_zoom,
            )
            self._training_camera_view_zoom = new_zoom
        if active and bool(imgui.is_mouse_dragging(0, 0.0)):
            span = 1.0 / max(float(self._training_camera_view_zoom), 1.0)
            delta = io.mouse_delta
            center_x, center_y = self._training_camera_view_center
            self._training_camera_view_center = _clamp_training_camera_center(
                float(center_x) - float(delta.x) * span / max(float(draw_width), 1.0),
                float(center_y) - float(delta.y) * span / max(float(draw_height), 1.0),
                self._training_camera_view_zoom,
            )
        imgui.pop_style_color(3)
        imgui.pop_style_var()

    def _draw_training_camera_colmap_overlay(
        self,
        ui: ViewerUI,
        image_origin: imgui.ImVec2,
        draw_width: float,
        draw_height: float,
        uv0: tuple[float, float],
        uv1: tuple[float, float],
        hovered: bool,
    ) -> None:
        if not bool(ui._values.get("show_training_camera_colmap_points", False)):
            self._training_camera_selected_point_id = None
            return
        payload = ui._values.get("_training_camera_colmap_points_payload")
        if not isinstance(payload, dict):
            self._training_camera_selected_point_id = None
            return
        point_uv = np.asarray(payload.get("uv", ()), dtype=np.float32).reshape(-1, 2)
        point_ids = np.asarray(payload.get("point_ids", ()), dtype=np.int64).reshape(-1)
        if point_uv.shape[0] == 0 or point_ids.size != point_uv.shape[0]:
            self._training_camera_selected_point_id = None
            return
        span_x = max(float(uv1[0]) - float(uv0[0]), 1e-6)
        span_y = max(float(uv1[1]) - float(uv0[1]), 1e-6)
        local_x = (point_uv[:, 0] - float(uv0[0])) / span_x
        local_y = (point_uv[:, 1] - float(uv0[1])) / span_y
        visible = (local_x >= 0.0) & (local_x <= 1.0) & (local_y >= 0.0) & (local_y <= 1.0)
        visible_indices = np.flatnonzero(visible)
        if visible_indices.size == 0:
            return
        screen_points = np.empty((int(visible_indices.size), 2), dtype=np.float32)
        screen_points[:, 0] = float(image_origin.x) + local_x[visible_indices] * float(draw_width)
        screen_points[:, 1] = float(image_origin.y) + local_y[visible_indices] * float(draw_height)
        draw_list = imgui.get_window_draw_list()
        selected_point_id = None if self._training_camera_selected_point_id is None else int(self._training_camera_selected_point_id)
        point_radius = float(_TRAINING_CAMERA_COLMAP_POINT_RADIUS)
        point_color = _color_u32(*_TRAINING_CAMERA_COLMAP_POINT_COLOR)
        selected_color = _color_u32(*_TRAINING_CAMERA_COLMAP_POINT_SELECTED_COLOR)
        draw_list.prim_reserve(6 * int(visible_indices.size), 4 * int(visible_indices.size))
        for point_x, point_y in screen_points.tolist():
            draw_list.prim_rect(
                imgui.ImVec2(float(point_x) - point_radius, float(point_y) - point_radius),
                imgui.ImVec2(float(point_x) + point_radius, float(point_y) + point_radius),
                point_color,
            )
        if hovered and bool(imgui.is_mouse_clicked(0)) and not bool(imgui.is_mouse_double_clicked(0)):
            mouse_pos = imgui.get_mouse_pos()
            delta = screen_points - np.asarray((float(mouse_pos.x), float(mouse_pos.y)), dtype=np.float32)[None, :]
            dist_sq = np.sum(delta * delta, axis=1)
            nearest_offset = int(np.argmin(dist_sq)) if dist_sq.size > 0 else -1
            hit_radius_sq = float(_TRAINING_CAMERA_COLMAP_POINT_HIT_RADIUS * _TRAINING_CAMERA_COLMAP_POINT_HIT_RADIUS)
            if nearest_offset >= 0 and float(dist_sq[nearest_offset]) <= hit_radius_sq:
                self._training_camera_selected_point_id = int(point_ids[int(visible_indices[nearest_offset])])
                selected_point_id = self._training_camera_selected_point_id
            else:
                self._training_camera_selected_point_id = None
                selected_point_id = None
        if selected_point_id is None:
            return
        selected_indices = np.flatnonzero(point_ids == int(selected_point_id))
        if selected_indices.size == 0:
            self._training_camera_selected_point_id = None
            return
        selected_index = int(selected_indices[0])
        if not bool(visible[selected_index]):
            return
        selected_screen_x = float(image_origin.x) + float(local_x[selected_index]) * float(draw_width)
        selected_screen_y = float(image_origin.y) + float(local_y[selected_index]) * float(draw_height)
        selected_radius = float(_TRAINING_CAMERA_COLMAP_POINT_SELECTED_RADIUS)
        draw_list.add_rect(
            imgui.ImVec2(selected_screen_x - selected_radius, selected_screen_y - selected_radius),
            imgui.ImVec2(selected_screen_x + selected_radius, selected_screen_y + selected_radius),
            selected_color,
        )
        ToolkitWindow._draw_training_camera_colmap_point_info(self, ui, payload, selected_index, selected_screen_x, selected_screen_y)

    def _draw_training_camera_colmap_point_info(self, ui: ViewerUI, payload: dict[str, object], point_index: int, point_x: float, point_y: float) -> None:
        point_ids = np.asarray(payload.get("point_ids", ()), dtype=np.int64).reshape(-1)
        point_xy = np.asarray(payload.get("xy", ()), dtype=np.float32).reshape(-1, 2)
        track_lengths = np.asarray(payload.get("track_lengths", ()), dtype=np.int32).reshape(-1)
        errors = np.asarray(payload.get("errors", ()), dtype=np.float32).reshape(-1)
        other_views = payload.get("other_views", ())
        if point_index < 0 or point_index >= point_ids.size or point_index >= point_xy.shape[0]:
            self._training_camera_selected_point_id = None
            return
        point_id = int(point_ids[point_index])
        point_error = float(errors[point_index]) if point_index < errors.size else float("nan")
        track_length = int(track_lengths[point_index]) if point_index < track_lengths.size else 0
        views = other_views[point_index] if point_index < len(other_views) else ()
        imgui.set_next_window_pos(
            imgui.ImVec2(point_x + _TRAINING_CAMERA_COLMAP_POINT_CONTEXT_OFFSET, point_y + _TRAINING_CAMERA_COLMAP_POINT_CONTEXT_OFFSET),
            imgui.Cond_.always.value,
        )
        flags = imgui.WindowFlags_.always_auto_resize.value | imgui.WindowFlags_.no_saved_settings.value
        opened, keep_open = imgui.begin(f"Point Match##training_camera_colmap_point_{point_id}", True, flags=flags)
        if not keep_open:
            self._training_camera_selected_point_id = None
        if opened:
            pos = imgui.get_window_pos()
            size = imgui.get_window_size()
            ToolkitWindow._append_viewport_ui_capture_rect(self, (float(pos.x), float(pos.y), float(size.x), float(size.y)))
            imgui.text_unformatted(f"Point {point_id}")
            imgui.text_disabled(f"xy: {float(point_xy[point_index, 0]):.1f}, {float(point_xy[point_index, 1]):.1f}")
            imgui.text_disabled(f"track length: {track_length}")
            imgui.text_disabled(f"reprojection error: {point_error:.4f}" if np.isfinite(point_error) else "reprojection error: n/a")
            imgui.separator()
            if views:
                imgui.text_unformatted(f"Other views ({len(views)})")
                for frame_index, image_id, image_name, _point_x, _point_y, point_u, point_v in views[:_TRAINING_CAMERA_COLMAP_CONTEXT_VIEW_LIMIT]:
                    label = f"[{int(image_id)}] {Path(str(image_name)).name}"
                    if int(frame_index) >= 0:
                        if imgui.selectable(label, False)[0]:
                            ToolkitWindow._focus_training_camera_point(self, ui, int(frame_index), int(image_id), point_id, float(point_u), float(point_v))
                    else:
                        imgui.begin_disabled()
                        imgui.selectable(label, False)
                        imgui.end_disabled()
                hidden_views = len(views) - _TRAINING_CAMERA_COLMAP_CONTEXT_VIEW_LIMIT
                if hidden_views > 0:
                    imgui.text_disabled(f"+{hidden_views} more")
            else:
                imgui.text_disabled("No other views")
        imgui.end()

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
        self._viewport_ui_capture_rects = ()
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
            show_training_cameras = bool(ui._values.get("show_training_cameras", False))
            if viewport_texture is not None:
                if show_training_cameras:
                    self._draw_training_camera_viewport_image(ui, viewport_texture, content_width, content_height)
                else:
                    self._reset_training_camera_view()
                    imgui.image(simgui.texture_ref(viewport_texture), image_size)
            else:
                self._reset_training_camera_view()
                imgui.dummy(image_size)
                draw_list = imgui.get_window_draw_list()
                placeholder_bg = _theme_color(ui, (0.92, 0.94, 0.97, 1.0), (0.04, 0.045, 0.055, 1.0))
                placeholder_text = _theme_color(ui, (0.30, 0.36, 0.45, 0.95), (0.72, 0.76, 0.82, 0.95))
                draw_list.add_rect_filled(cursor, imgui.ImVec2(cursor.x + content_width, cursor.y + content_height), _color_u32(*placeholder_bg))
                label = "Load a scene to populate the viewport"
                text_size = imgui.calc_text_size(label)
                draw_list.add_text(
                    imgui.ImVec2(cursor.x + 0.5 * (content_width - float(text_size.x)), cursor.y + 0.5 * (content_height - float(text_size.y))),
                    _color_u32(*placeholder_text),
                    label,
                )
            self._draw_viewport_camera_overlays(ui, cursor)
            overlay_origin = self._draw_viewport_view_menu(ui, cursor)
            self._draw_viewport_debug_overlay(ui, overlay_origin)
        else:
            self._viewport_window_focused = False
            self._viewport_input_active = False
        imgui.end()

    def _draw_viewport_view_toggles(self, ui: ViewerUI, scale: float) -> bool:
        opened = _imgui_opened(imgui.small_button("View Mode"))
        for key, enabled_label, disabled_label, default in (
            ("show_camera_overlays", "Cameras On", "Cameras Off", True),
            ("show_camera_labels", "Labels On", "Labels Off", False),
            ("show_camera_min_dist_spheres", "Min Dist On", "Min Dist Off", True),
            ("show_training_cameras", "Training Cameras On", "Training Cameras Off", False),
        ):
            imgui.same_line(0.0, 10.0 * scale)
            enabled = bool(ui._values.get(key, default))
            if _imgui_opened(imgui.small_button(enabled_label if enabled else disabled_label)):
                ui._values[key] = not enabled
        imgui.same_line(0.0, 10.0 * scale)
        if _imgui_opened(imgui.small_button("Reset Camera")):
            self.callbacks.reset_camera()
        return opened

    def _draw_viewport_sh_band_combo(self, ui: ViewerUI, scale: float) -> int:
        sh_band = min(max(int(ui._values.get("_viewport_sh_band", ui._values.get("sh_band", 0))), 0), len(_SH_BAND_LABELS) - 1)
        sh_label_width = max(float(imgui.calc_text_size(option).x) for option in _SH_BAND_LABELS)
        imgui.same_line(0.0, 10.0 * scale)
        imgui.set_next_item_width(sh_label_width + 28.0 * scale)
        imgui.push_style_var(imgui.StyleVar_.frame_padding.value, imgui.ImVec2(max(4.0 * scale, 1.0), max(1.0 * scale, 1.0)))
        if imgui.begin_combo("##viewport_sh_band", _SH_BAND_LABELS[sh_band]):
            for idx, option in enumerate(_SH_BAND_LABELS):
                selected = idx == sh_band
                if imgui.selectable(option, selected)[0]:
                    ui._values["_viewport_sh_band"] = idx
                if selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        imgui.pop_style_var()
        return min(max(int(ui._values.get("_viewport_sh_band", sh_band)), 0), len(_SH_BAND_LABELS) - 1)

    def _draw_viewport_mode_badge(self, ui: ViewerUI, scale: float, current_label: str) -> tuple[float, float]:
        imgui.same_line(0.0, 10.0 * scale)
        label_pos = imgui.get_cursor_screen_pos()
        current_label_size = imgui.calc_text_size(current_label)
        badge_pad_x = 6.0 * scale
        badge_pad_y = 2.0 * scale
        draw_list = imgui.get_window_draw_list()
        draw_list.add_rect_filled(
            imgui.ImVec2(label_pos.x - badge_pad_x, label_pos.y - badge_pad_y),
            imgui.ImVec2(label_pos.x + float(current_label_size.x) + badge_pad_x, label_pos.y + float(current_label_size.y) + badge_pad_y),
            _color_u32(*_theme_color(ui, (0.80, 0.88, 0.97, 0.82), (0.06, 0.08, 0.12, 0.68))),
            4.0 * scale,
        )
        badge_text_color = _theme_color(ui, (0.18, 0.26, 0.38, 1.0), (0.985, 0.992, 1.000, 1.0))
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*badge_text_color))
        imgui.text_unformatted(current_label)
        imgui.pop_style_color()
        row_right = float(label_pos.x) + float(current_label_size.x) + badge_pad_x
        row_height = max(float(current_label_size.y) + 2.0 * badge_pad_y, 1.0)
        return row_right, row_height

    def _draw_viewport_view_popup(self, ui: ViewerUI, current: int) -> None:
        if not _imgui_opened(imgui.begin_popup("viewport_view_popup")):
            return
        popup_pos = imgui.get_window_pos()
        popup_size = imgui.get_window_size()
        self._append_viewport_ui_capture_rect((float(popup_pos.x), float(popup_pos.y), float(popup_size.x), float(popup_size.y)))
        for idx, label in enumerate(_DEBUG_MODE_LABELS):
            selected = idx == current
            if _imgui_opened(imgui.selectable(label, selected)):
                ui._values["debug_mode"] = idx
            if selected:
                imgui.set_item_default_focus()
        imgui.end_popup()

    def _draw_viewport_view_menu(self, ui: ViewerUI, image_origin: imgui.ImVec2) -> imgui.ImVec2:
        style = imgui.get_style()
        scale = self._interface_scale_factor(ui)
        label_size = imgui.calc_text_size("View Mode")
        button_pos = imgui.ImVec2(float(image_origin.x) + _VIEWPORT_OVERLAY_MARGIN * scale, float(image_origin.y) + _VIEWPORT_OVERLAY_MARGIN * scale)
        button_height = float(label_size.y) + 2.0 * float(style.frame_padding.y)
        current = min(max(int(ui._values.get("debug_mode", 0)), 0), len(_DEBUG_MODE_LABELS) - 1)
        imgui.push_id("viewport_view")
        imgui.set_cursor_screen_pos(button_pos)
        opened = ToolkitWindow._draw_viewport_view_toggles(self, ui, scale)
        current_sh_band = ToolkitWindow._draw_viewport_sh_band_combo(self, ui, scale)
        row_right, row_height = ToolkitWindow._draw_viewport_mode_badge(self, ui, scale, _DEBUG_MODE_LABELS[current])
        self._append_viewport_ui_capture_rect((float(button_pos.x), float(button_pos.y), max(row_right - float(button_pos.x), 1.0), max(button_height, row_height)))
        if opened:
            imgui.set_next_window_pos(imgui.ImVec2(button_pos.x, button_pos.y + button_height + _VIEWPORT_OVERLAY_MARGIN * scale), imgui.Cond_.appearing.value)
            imgui.open_popup("viewport_view_popup")
        ToolkitWindow._draw_viewport_view_popup(self, ui, current)
        imgui.pop_id()
        ui._values["_viewport_sh_band"] = current_sh_band
        return imgui.ImVec2(button_pos.x, button_pos.y + button_height + _VIEWPORT_OVERLAY_MARGIN * scale)

    def _training_camera_debug_section_height(self, ui: ViewerUI) -> float:
        line_height = float(imgui.get_text_line_height_with_spacing())
        frame_height = float(imgui.get_frame_height())
        spacing_y = float(imgui.get_style().item_spacing.y)
        height = frame_height + spacing_y + frame_height + spacing_y + frame_height + spacing_y + frame_height + spacing_y + frame_height
        if LOSS_DEBUG_OPTIONS[min(max(int(ui._values.get("loss_debug_view", 0)), 0), len(LOSS_DEBUG_OPTIONS) - 1)][0] == "abs_diff":
            height += frame_height + spacing_y
        if bool(ui._values.get("show_training_camera_colmap_points", False)):
            height += line_height + spacing_y
        for key, _suffix_only in _TRAINING_CAMERA_DEBUG_TEXT_FIELDS:
            text = str(ui._texts.get(key, "")).strip()
            if text:
                height += max(text.count("\n") + 1, 1) * (line_height + spacing_y)
        sections = ui._values.get("_training_camera_struct_sections", ())
        if sections:
            overlay_width = max(_VIEWPORT_OVERLAY_WIDTH * self._interface_scale_factor(ui) - 2.0 * _VIEWPORT_OVERLAY_PADDING * self._interface_scale_factor(ui), 80.0)
            height += measure_struct_sections(sections, max_width=overlay_width) * (line_height + spacing_y)
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
        changed, full_res = imgui.checkbox("Full Resolution", bool(ui._values.get("training_camera_full_resolution", False)))
        if changed:
            ui._values["training_camera_full_resolution"] = bool(full_res)
        changed, show_points = imgui.checkbox("COLMAP Point Matches", bool(ui._values.get("show_training_camera_colmap_points", False)))
        if changed:
            ui._values["show_training_camera_colmap_points"] = bool(show_points)
            if not bool(show_points):
                self._training_camera_selected_point_id = None
        if bool(ui._values.get("show_training_camera_colmap_points", False)):
            payload = ui._values.get("_training_camera_colmap_points_payload")
            render_count = int(payload.get("render_count", 0)) if isinstance(payload, dict) else 0
            total_count = int(payload.get("total_count", render_count)) if isinstance(payload, dict) else render_count
            suffix = f"{render_count:,}" if total_count <= render_count else f"{render_count:,}/{total_count:,}"
            imgui.text_disabled(f"COLMAP points: {suffix}")
        if imgui.button("Move Main View Here"):
            self.callbacks.move_to_training_camera()
        for key, suffix_only in _TRAINING_CAMERA_DEBUG_TEXT_FIELDS:
            text = ui._texts.get(key, "")
            if text:
                _draw_disabled_wrapped_text(text, strip_label=suffix_only)
        sections = ui._values.get("_training_camera_struct_sections", ())
        if sections:
            draw_struct_sections(sections)

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
        for near_points, far_points, connectors, sphere_rings, label_anchor, label_text, color, thickness in overlays:
            color_u32 = _color_u32(*color)
            near_polyline = [imgui.ImVec2(base_x + float(x), base_y + float(y)) for x, y in near_points]
            far_polyline = [imgui.ImVec2(base_x + float(x), base_y + float(y)) for x, y in far_points]
            if len(near_polyline) >= 2:
                draw_list.add_polyline(near_polyline, color_u32, imgui.ImDrawFlags_.closed.value, float(thickness))
            if len(far_polyline) >= 2:
                draw_list.add_polyline(far_polyline, color_u32, imgui.ImDrawFlags_.closed.value, float(thickness))
            for ring in sphere_rings:
                ring_polyline = [imgui.ImVec2(base_x + float(x), base_y + float(y)) for x, y in ring]
                if len(ring_polyline) >= 2:
                    draw_list.add_polyline(ring_polyline, color_u32, imgui.ImDrawFlags_.closed.value, float(max(thickness * 0.85, 1.0)))
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
                    _color_u32(*_theme_color(ui, (0.95, 0.97, 0.99, 0.84), (0.04, 0.05, 0.07, 0.74))),
                    4.0 * scale,
                )
                draw_list.add_text(label_font, label_font_size, label_pos, _color_u32(*_theme_color(ui, (0.16, 0.22, 0.30, 1.0), (0.98, 0.99, 1.0, 1.0))), str(label_text))

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
        child_bg = _theme_color(ui, (0.982, 0.987, 0.994, 0.94), (0.13, 0.14, 0.17, 0.94))
        child_border = _theme_color(ui, (0.65, 0.72, 0.80, 0.92), (0.32, 0.36, 0.43, 0.92))
        imgui.push_style_color(imgui.Col_.child_bg.value, imgui.ImVec4(*child_bg))
        imgui.push_style_color(imgui.Col_.border.value, imgui.ImVec4(*child_border))
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(padding, padding))
        self._append_viewport_ui_capture_rect((float(overlay_origin.x), float(overlay_origin.y), float(width), float(height)))
        imgui.set_cursor_screen_pos(overlay_origin)
        if _imgui_opened(imgui.begin_child("##viewport_debug_overlay", imgui.ImVec2(width, height), imgui.ChildFlags_.borders.value)):
            imgui.push_item_width(-1.0)
            if show_training_cameras:
                self._draw_training_camera_debug_controls(ui)
                if len(control_keys) > 0:
                    imgui.separator()
            for key in control_keys:
                self._draw_control(ui, _DEBUG_VIEW_CONTROL_SPECS[key], compact=True)
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
            _color_u32(*_theme_color(ui, (0.985, 0.989, 0.995, 0.94), (0.13, 0.14, 0.17, 0.94))),
            8.0 * scale,
        )
        draw_list.add_text(imgui.ImVec2(box_x + 0.5 * side_pad, box_y + 0.25 * top_pad), _color_u32(*_theme_color(ui, (0.20, 0.26, 0.34, 0.95), (0.90, 0.93, 0.97, 0.95))), self._debug_colorbar_title(mode))
        self._draw_debug_colorbar_gradient(draw_list, x0, y0, x1, y1)
        draw_list.add_rect(imgui.ImVec2(x0, y0), imgui.ImVec2(x1, y1), _color_u32(*_theme_color(ui, (0.50, 0.60, 0.72, 0.95), (0.43, 0.57, 0.76, 0.95))), 2.0 * scale, 0, max(scale, 1.0))
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
            draw_list.add_line(imgui.ImVec2(x, y1 + 2.0 * scale), imgui.ImVec2(x, y1 + 10.0 * scale), _color_u32(*_theme_color(ui, (0.44, 0.53, 0.64, 0.9), (0.58, 0.68, 0.80, 0.9))), max(scale, 1.0))
            draw_list.add_text(imgui.ImVec2(x - 0.5 * float(label_size.x), y1 + 12.0 * scale), _color_u32(*_theme_color(ui, (0.22, 0.28, 0.36, 0.95), (0.86, 0.90, 0.95, 0.95))), label)

    def _debug_colorbar_title(self, mode: str) -> str:
        return {
            "processed_count": "Processed Count",
            "splat_age": "Splat Age",
            "splat_density": "Splat Density",
            "splat_spatial_density": "Spatial Density",
            "splat_screen_density": "Screen Density",
            "contribution_amount": "Contribution Amount",
            "adam_momentum": "Adam Momentum",
            "adam_second_moment": "Adam Second Moment",
            "grad_variance": "Grad Variance",
            "refinement_distribution": "Refinement Distribution",
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
        if mode == "splat_age":
            return f"{_debug_range_tick_value(t, float(ui._values.get('debug_splat_age_min', 0.0)), float(ui._values.get('debug_splat_age_max', 1.0))):.3g}"
        if mode in ("splat_density", "splat_spatial_density", "splat_screen_density"):
            return f"{_debug_range_tick_value(t, float(ui._values.get('debug_density_min', 0.0)), float(ui._values.get('debug_density_max', 20.0))):.3g}"
        if mode == "contribution_amount":
            return f"{_debug_range_tick_value(t, float(ui._values.get('debug_contribution_min', 0.0)), float(ui._values.get('debug_contribution_max', 1.0))):.3g}"
        if mode in ("adam_momentum", "adam_second_moment"):
            threshold = float(ui._values.get("debug_grad_norm_threshold", _DEBUG_GRAD_NORM_THRESHOLD_DEFAULT))
            return f"{_threshold_band_tick_value(t, threshold):.1e}"
        if mode == "grad_variance":
            threshold = float(ui._values.get("debug_grad_norm_threshold", _DEBUG_GRAD_NORM_THRESHOLD_DEFAULT))
            value = _threshold_band_tick_value(t, threshold)
            return f"{value * value:.1e}"
        if mode == "refinement_distribution":
            return f"{_refinement_distribution_tick_value(t, float(ui._values.get('debug_refinement_distribution_min', 0.0)), float(ui._values.get('debug_refinement_distribution_max', 1.0))):.3g}"
        if mode == "depth_mean":
            return f"{_debug_range_tick_value(t, float(ui._values.get('debug_depth_mean_min', 0.0)), float(ui._values.get('debug_depth_mean_max', 10.0))):.3g}"
        if mode == "depth_std":
            return f"{_debug_range_tick_value(t, float(ui._values.get('debug_depth_std_min', 0.0)), float(ui._values.get('debug_depth_std_max', 0.5))):.3g}"
        if mode == "depth_local_mismatch":
            return f"{_debug_range_tick_value(t, float(ui._values.get('debug_depth_local_mismatch_min', 0.0)), float(ui._values.get('debug_depth_local_mismatch_max', 0.5))):.3g}"
        return ""

    def _draw_file_menu(self, ui: ViewerUI) -> None:
        if not imgui.begin_menu("File"):
            return
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
        imgui.separator()
        if _menu_item("Exit"):
            self.callbacks.request_exit()
        imgui.end_menu()

    def _draw_view_menu(self, ui: ViewerUI) -> None:
        if not imgui.begin_menu("View"):
            return
        active_idx = min(max(int(ui._values.get(_INTERFACE_SCALE_KEY, _DEFAULT_INTERFACE_SCALE_INDEX)), 0), len(_INTERFACE_SCALE_OPTIONS) - 1)
        for idx, (label, _) in enumerate(_INTERFACE_SCALE_OPTIONS):
            if _menu_item(label, selected=idx == active_idx):
                ui._values[_INTERFACE_SCALE_KEY] = idx
        imgui.separator()
        if _menu_item("Reset Interface Scale"):
            ui._values[_INTERFACE_SCALE_KEY] = _DEFAULT_INTERFACE_SCALE_INDEX
        imgui.separator()
        if imgui.begin_menu("Theme"):
            active_theme_idx = self._theme_index(ui)
            for idx, label in enumerate(_THEME_OPTIONS):
                if _menu_item(label, selected=idx == active_theme_idx):
                    ui._values[_THEME_KEY] = idx
            imgui.separator()
            if _menu_item("Reset Theme"):
                ui._values[_THEME_KEY] = int(_VIEWER_CONTROL_DEFAULTS.get("theme", 0))
            imgui.end_menu()
        imgui.end_menu()

    def _draw_debug_menu(self, ui: ViewerUI) -> None:
        if not imgui.begin_menu("Debug"):
            return
        for key, label in (("show_resource_debug", "Buffers"), ("show_histograms", "Histograms"), ("show_photometric_compensation", "Photometric Compensation"), ("show_training_views", "Training Views")):
            selected = bool(ui._values.get(key, False))
            if _menu_item(label, selected=selected):
                ui._values[key] = not selected
        imgui.end_menu()

    def _draw_help_menu(self) -> None:
        if not imgui.begin_menu("Help"):
            return
        if _menu_item("Documentation"):
            self._show_docs = True
        if _menu_item("About"):
            self._show_about = True
        imgui.end_menu()

    def _draw_exit_confirmation_modal(self, ui: ViewerUI) -> None:
        if bool(ui._values.get("_exit_confirmation_open", False)):
            imgui.open_popup("Confirm Exit")
        scale = max(self._applied_interface_scale, 1.0)
        imgui.set_next_window_size(imgui.ImVec2(420.0 * scale, 0.0), imgui.Cond_.appearing.value)
        if not _imgui_opened(imgui.begin_popup_modal("Confirm Exit", None, flags=imgui.WindowFlags_.always_auto_resize.value)):
            return
        imgui.text_wrapped("Do you want to exit? Any unsaved data will be lost")
        imgui.separator()
        if imgui.button("Cancel", imgui.ImVec2(120.0 * scale, 0.0)):
            self.callbacks.cancel_exit()
            imgui.close_current_popup()
        imgui.same_line()
        if imgui.button("Exit", imgui.ImVec2(120.0 * scale, 0.0)):
            self.callbacks.confirm_exit()
            imgui.close_current_popup()
        imgui.end_popup()

    def _menu_bar_status_text(self, ui: ViewerUI) -> str:
        fps = self.tk.fps_history[-1] if self.tk.fps_history else 0.0
        used_bytes = ui._values.get("_menu_bar_device_vram_bytes")
        total_bytes = ui._values.get("_menu_bar_device_vram_total_bytes")
        dataset_bytes = ui._values.get("_menu_bar_dataset_vram_bytes")
        app_bytes = ui._values.get("_menu_bar_app_vram_bytes")
        tracked_total_bytes = ui._values.get("_menu_bar_total_vram_bytes")
        used_text = "n/a" if used_bytes is None else format_resource_bytes(int(used_bytes))
        total_text = None if total_bytes is None else format_resource_bytes(int(total_bytes))
        dataset_text = "n/a" if dataset_bytes is None else format_resource_bytes(int(dataset_bytes))
        app_text = "n/a" if app_bytes is None else format_resource_bytes(int(app_bytes))
        tracked_total_text = "n/a" if tracked_total_bytes is None else format_resource_bytes(int(tracked_total_bytes))
        fraction = self._menu_bar_vram_fraction(ui)
        if fraction is None:
            vram_text = used_text if total_text is None else f"{used_text} / {total_text}"
        else:
            vram_text = f"{fraction * 100.0:.0f}% ({used_text} / {total_text})"
        return f"FPS {fps:.1f} | VRAM {vram_text} | dataset: {dataset_text} | app: {app_text} | total: {tracked_total_text}"

    def _menu_bar_vram_fraction(self, ui: ViewerUI) -> float | None:
        used_bytes = ui._values.get("_menu_bar_device_vram_bytes")
        total_bytes = ui._values.get("_menu_bar_device_vram_total_bytes")
        if used_bytes is None or total_bytes is None:
            return None
        total = max(int(total_bytes), 0)
        if total <= 0:
            return None
        used = max(int(used_bytes), 0)
        return min(max(float(used) / float(total), 0.0), 1.0)

    def _menu_bar_vram_color(self, ui: ViewerUI) -> imgui.ImVec4 | None:
        fraction = self._menu_bar_vram_fraction(ui)
        if fraction is None:
            return None
        if fraction < 0.70:
            return imgui.ImVec4(0.2, 0.9, 0.3, 1.0)
        if fraction < 0.90:
            return imgui.ImVec4(1.0, 0.85, 0.2, 1.0)
        return imgui.ImVec4(1.0, 0.3, 0.3, 1.0)

    def _menu_bar_status_segments(self, ui: ViewerUI) -> tuple[tuple[str, imgui.ImVec4 | None], ...]:
        fps = self.tk.fps_history[-1] if self.tk.fps_history else 0.0
        used_bytes = ui._values.get("_menu_bar_device_vram_bytes")
        total_bytes = ui._values.get("_menu_bar_device_vram_total_bytes")
        dataset_bytes = ui._values.get("_menu_bar_dataset_vram_bytes")
        app_bytes = ui._values.get("_menu_bar_app_vram_bytes")
        tracked_total_bytes = ui._values.get("_menu_bar_total_vram_bytes")
        used_text = "n/a" if used_bytes is None else format_resource_bytes(int(used_bytes))
        total_text = None if total_bytes is None else format_resource_bytes(int(total_bytes))
        dataset_text = "n/a" if dataset_bytes is None else format_resource_bytes(int(dataset_bytes))
        app_text = "n/a" if app_bytes is None else format_resource_bytes(int(app_bytes))
        tracked_total_text = "n/a" if tracked_total_bytes is None else format_resource_bytes(int(tracked_total_bytes))
        fraction = self._menu_bar_vram_fraction(ui)
        if fraction is None:
            vram_value_text = used_text if total_text is None else f"{used_text} / {total_text}"
            return (
                (f"FPS {fps:.1f}", None),
                (" | VRAM ", None),
                (vram_value_text, self._menu_bar_vram_color(ui)),
                (f" | dataset: {dataset_text}", None),
                (f" | app: {app_text}", None),
                (f" | total: {tracked_total_text}", None),
            )
        return (
            (f"FPS {fps:.1f}", None),
            (" | VRAM ", None),
            (f"{fraction * 100.0:.0f}%", self._menu_bar_vram_color(ui)),
            (f" ({used_text} / {total_text})", None),
            (f" | dataset: {dataset_text}", None),
            (f" | app: {app_text}", None),
            (f" | total: {tracked_total_text}", None),
        )

    def _draw_menu_bar_status(self, ui: ViewerUI) -> None:
        segments = self._menu_bar_status_segments(ui)
        if len(segments) == 0:
            return
        style = imgui.get_style()
        current_x = float(imgui.get_cursor_pos_x())
        current_y = float(imgui.get_cursor_pos_y())
        text_width = sum(float(imgui.calc_text_size(text).x) for text, _color in segments)
        right_padding = float(style.item_spacing.x + style.window_padding.x)
        target_x = max(current_x, float(imgui.get_window_width()) - text_width - right_padding)
        segment_x = target_x
        for text, color in segments:
            imgui.set_cursor_pos_x(segment_x)
            imgui.set_cursor_pos_y(current_y)
            if color is not None:
                imgui.push_style_color(imgui.Col_.text.value, color)
            imgui.text_unformatted(text)
            if color is not None:
                imgui.pop_style_color()
            segment_x += float(imgui.calc_text_size(text).x)

    def _draw_main_menu_bar(self, ui: ViewerUI) -> float:
        if not imgui.begin_main_menu_bar():
            return 0.0
        ToolkitWindow._draw_file_menu(self, ui)
        ToolkitWindow._draw_view_menu(self, ui)
        ToolkitWindow._draw_debug_menu(self, ui)
        ToolkitWindow._draw_help_menu(self)
        self._draw_menu_bar_status(ui)
        menu_bar_height = float(imgui.get_window_height())
        imgui.end_main_menu_bar()
        return menu_bar_height

    def _draw_about_window(self) -> None:
        if not self._show_about:
            return
        self._dock_tool_window(imgui.Cond_.appearing.value)
        scale = max(self._applied_interface_scale, 1.0)
        imgui.set_next_window_pos(imgui.ImVec2(24.0 * scale, self._menu_bar_height + 24.0 * scale), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(420.0 * scale, 220.0 * scale), imgui.Cond_.first_use_ever.value)
        opened, self._show_about = imgui.begin("About", True)
        ToolkitWindow._register_non_viewport_window(self)
        if opened:
            _draw_markdown_text(self._about_text)
        imgui.end()

    def _draw_documentation_window(self) -> None:
        if not self._show_docs:
            return
        self._dock_tool_window(imgui.Cond_.appearing.value)
        scale = max(self._applied_interface_scale, 1.0)
        imgui.set_next_window_pos(imgui.ImVec2(40.0 * scale, self._menu_bar_height + 32.0 * scale), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(760.0 * scale, 620.0 * scale), imgui.Cond_.first_use_ever.value)
        opened, self._show_docs = imgui.begin("Documentation", True)
        ToolkitWindow._register_non_viewport_window(self)
        if opened:
            imgui.text_disabled("Local viewer documentation")
            imgui.separator()
            child_opened = _imgui_opened(imgui.begin_child("##docs_scroll", imgui.ImVec2(0.0, 0.0), imgui.ChildFlags_.borders.value))
            if child_opened:
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
        if imgui.button(button_label): callback()

    @staticmethod
    def _draw_combo(label: str, options: tuple[str, ...], current: int) -> int:
        current = max(0, min(int(current), len(options) - 1))
        if imgui.begin_combo(label, options[current]):
            for idx, option in enumerate(options):
                is_selected = idx == current
                if imgui.selectable(option, is_selected)[0]: current = idx
                if is_selected: imgui.set_item_default_focus()
            imgui.end_combo()
        return current

    @staticmethod
    def _set_tooltip(text: str) -> None:
        if imgui.is_item_hovered(): imgui.set_item_tooltip(text)

    @staticmethod
    def _draw_clamped_int(ui: ViewerUI, *, key: str, label: str, default: int, speed: float, min_value: int, max_value: int, tooltip: str) -> None:
        changed, value = imgui.drag_int(label, int(ui._values.get(key, default)), speed, min_value, max_value)
        if changed: ui._values[key] = min(max(int(value), min_value), max_value)
        ToolkitWindow._set_tooltip(tooltip)

    @staticmethod
    def _draw_clamped_float(ui: ViewerUI, *, key: str, label: str, default: float, speed: float, min_value: float, max_value: float, fmt: str, tooltip: str, flags: int = 0) -> None:
        changed, value = imgui.drag_float(label, float(ui._values.get(key, default)), speed, min_value, max_value, fmt, flags)
        if changed: ui._values[key] = min(max(float(value), min_value), max_value)
        ToolkitWindow._set_tooltip(tooltip)

    def _draw_colmap_camera_selection_table(self, ui: ViewerUI, camera_rows: tuple[dict[str, object], ...]) -> None:
        selected_camera_ids = tuple(int(camera_id) for camera_id in ui._values.get("colmap_selected_camera_ids", ()))
        camera_ids = tuple(int(row["camera_id"]) for row in camera_rows)
        selected = {camera_id for camera_id in selected_camera_ids if camera_id in camera_ids}
        selected_frame_count = sum(int(row["frame_count"]) for row in camera_rows if int(row["camera_id"]) in selected)
        total_frame_count = sum(int(row["frame_count"]) for row in camera_rows)
        imgui.text_disabled(f"Camera Models: {len(selected)}/{len(camera_rows)} selected | Frames/Poses: {selected_frame_count}/{total_frame_count}")
        point_stats = ui._values.get("_colmap_point_stats")
        if isinstance(point_stats, dict):
            total_points = max(int(point_stats.get("total_points", 0)), 0)
            tracked_points_min2 = max(int(point_stats.get("tracked_points_min2", 0)), 0)
            imgui.text_disabled(f"Points: {total_points:,} total | {tracked_points_min2:,} tracked (>=2 obs)")
        if imgui.button("All Models"): selected = set(camera_ids)
        imgui.same_line()
        if imgui.button("No Models"): selected.clear()
        table_height = min(max(88.0, 28.0 * float(len(camera_rows)) + 8.0), 180.0)
        child_opened = _imgui_opened(imgui.begin_child("##colmap_cameras", imgui.ImVec2(0.0, table_height), True))
        if child_opened:
            flags = (
                imgui.TableFlags_.row_bg.value
                | imgui.TableFlags_.borders.value
                | imgui.TableFlags_.resizable.value
                | imgui.TableFlags_.scroll_x.value
                | imgui.TableFlags_.scroll_y.value
            )
            if imgui.begin_table("##colmap_camera_table", 7, flags):
                for name, column_flags, width in (
                    ("Use", imgui.TableColumnFlags_.width_fixed.value, 36.0),
                    ("Camera Id", imgui.TableColumnFlags_.width_fixed.value, 72.0),
                    ("Model", imgui.TableColumnFlags_.width_fixed.value, 132.0),
                    ("Frames / Poses", imgui.TableColumnFlags_.width_fixed.value, 96.0),
                    ("Res", imgui.TableColumnFlags_.width_fixed.value, 86.0),
                    ("Focal", imgui.TableColumnFlags_.width_fixed.value, 120.0),
                    ("Principal / Dist", imgui.TableColumnFlags_.width_stretch.value, 0.0),
                ):
                    imgui.table_setup_column(name, column_flags, width)
                imgui.table_headers_row()
                for row in camera_rows:
                    camera_id = int(row["camera_id"])
                    imgui.table_next_row()
                    imgui.table_next_column()
                    changed, value = imgui.checkbox(f"##colmap_camera_{camera_id}", camera_id in selected)
                    if changed:
                        if value: selected.add(camera_id)
                        else: selected.discard(camera_id)
                    for text in (
                        str(camera_id),
                        str(row["model_name"]),
                        str(row["frame_count"]),
                        str(row["resolution_text"]),
                        str(row["focal_text"]),
                        f"{row['principal_text']} | {row['distortion_text']}",
                    ):
                        imgui.table_next_column()
                        imgui.text_unformatted(text)
                imgui.end_table()
        imgui.end_child()
        ui._values["colmap_selected_camera_ids"] = tuple(camera_id for camera_id in camera_ids if camera_id in selected)

    def _draw_colmap_downscale_controls(self, ui: ViewerUI) -> None:
        downscale_idx = ToolkitWindow._draw_combo("Image Downscale", _COLMAP_IMAGE_DOWNSCALE_LABELS, int(ui._values.get("colmap_image_downscale_mode", 1)))
        ui._values["colmap_image_downscale_mode"] = downscale_idx
        if downscale_idx == 1:
            changed, value = imgui.input_int("Max Size", int(ui._values.get("colmap_image_max_size", 2048)), 64, 256)
            if changed:
                ui._values["colmap_image_max_size"] = max(int(value), 1)
            ToolkitWindow._set_tooltip("Clamp imported training images so their longer side is at most this size while preserving aspect ratio. The importer never upscales.")
            return
        if downscale_idx == 2:
            ToolkitWindow._draw_clamped_float(
                ui,
                key="colmap_image_scale",
                label="Scale Factor",
                default=1.0,
                speed=0.01,
                min_value=1e-3,
                max_value=1.0,
                fmt="%.4f",
                tooltip="Uniform scale applied to imported training images. Both axes stay proportional and the importer never upscales.",
                flags=imgui.SliderFlags_.logarithmic.value,
            )
            return
        imgui.text_disabled("Imported images keep their source resolution.")

    def _draw_colmap_init_mode_controls(self, ui: ViewerUI) -> None:
        init_labels = _colmap_init_mode_labels(_valid_depth_root_text(ui._values.get("colmap_depth_root", "")))
        mode_idx = ToolkitWindow._draw_combo("Initialization", init_labels, min(int(ui._values.get("colmap_init_mode", 0)), len(init_labels) - 1))
        ui._values["colmap_init_mode"] = mode_idx
        ToolkitWindow._set_tooltip("Point Sources combines any enabled source rows below. From Depth calibrates matched 16-bit PNG depth maps into a point cloud using an iteratively reweighted robust per-pose affine depth fit from valid observed points while rejecting local depth-gradient spikes.")
        if mode_idx == 1:
            imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x)
            imgui.text_disabled("From Depth matches RGB and depth by relative stem under Depth Folder, uses each pose's own positive COLMAP point observations, reprojects those 3D points through the frame camera model to sample depth, rejects projected samples whose local pixel-footprint gradients are a strong outlier relative to nearby gradients, then solves one iteratively reweighted robust affine map `a + b*d` per pose from the remaining observed points before sampling a dataset-wide calibrated point budget. Frames without usable depth stay in training but are skipped for depth-based initialization.")
            imgui.pop_text_wrap_pos()
            ui._values["colmap_depth_value_mode"] = ToolkitWindow._draw_combo(
                "Depth Interpretation",
                _COLMAP_DEPTH_VALUE_MODE_LABELS,
                int(ui._values.get("colmap_depth_value_mode", 1)),
            )
            ToolkitWindow._set_tooltip("Choose whether calibrated depth values represent Euclidean camera distance or camera-space z-depth before reverse projection.")
            ToolkitWindow._draw_clamped_int(
                ui,
                key="colmap_depth_point_count",
                label="Depth Point Count",
                default=100000,
                speed=1000.0,
                min_value=1,
                max_value=10000000,
                tooltip="Total calibrated points sampled across all matched RGB/depth pairs for depth-based initialization.",
            )
            return

        imgui.text_wrapped("Enable any combination of initialization sources. Each row owns its own point budget and NN scale coefficient.")
        ToolkitWindow._draw_clamped_int(
            ui,
            key="colmap_min_track_length",
            label="Min Camera Observations",
            default=DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH,
            speed=0.25,
            min_value=0,
            max_value=32,
            tooltip="Ignore sparse COLMAP points whose track is shorter than this many observing cameras. Set 0 to keep all sparse points.",
        )
        imgui.spacing()
        flags = imgui.TableFlags_.row_bg.value | imgui.TableFlags_.borders.value | imgui.TableFlags_.resizable.value | imgui.TableFlags_.sizing_stretch_prop.value
        if imgui.begin_table("##colmap_init_sources", 5, flags):
            for label, column_flags, width in (
                ("Use", imgui.TableColumnFlags_.width_fixed.value, 38.0),
                ("Source", imgui.TableColumnFlags_.width_fixed.value, 150.0),
                ("Points", imgui.TableColumnFlags_.width_fixed.value, 96.0),
                ("Path / Radius", imgui.TableColumnFlags_.width_stretch.value, 0.0),
                ("NN Scale", imgui.TableColumnFlags_.width_fixed.value, 112.0),
            ):
                imgui.table_setup_column(label, column_flags, width)
            imgui.table_headers_row()

            def _row_toggle(key: str) -> bool:
                imgui.table_next_column()
                changed, enabled = imgui.checkbox(f"##{key}", bool(ui._values.get(key, False)))
                if changed:
                    ui._values[key] = bool(enabled)
                return bool(ui._values.get(key, False))

            def _row_nn_scale(key: str, default: float, enabled: bool) -> None:
                imgui.table_next_column()
                imgui.begin_disabled(not enabled)
                ToolkitWindow._draw_clamped_float(
                    ui,
                    key=key,
                    label=f"##{key}",
                    default=default,
                    speed=0.01,
                    min_value=1e-4,
                    max_value=16.0,
                    fmt="%.4f",
                    tooltip="Multiplier applied to the source-local nearest-neighbor radius when initializing gaussian scales.",
                    flags=imgui.SliderFlags_.logarithmic.value,
                )
                imgui.end_disabled()

            imgui.table_next_row()
            pointcloud_enabled = _row_toggle("colmap_pointcloud_enabled")
            imgui.table_next_column()
            imgui.text_unformatted("COLMAP Pointcloud")
            imgui.table_next_column()
            imgui.text_disabled("all")
            imgui.table_next_column()
            imgui.text_disabled("Sparse COLMAP points")
            _row_nn_scale("colmap_pointcloud_nn_radius_scale_coef", 0.5, pointcloud_enabled)

            imgui.table_next_row()
            diffused_enabled = _row_toggle("colmap_diffused_enabled")
            imgui.table_next_column()
            imgui.text_unformatted("Diffused COLMAP")
            imgui.table_next_column()
            imgui.begin_disabled(not diffused_enabled)
            ToolkitWindow._draw_clamped_int(
                ui,
                key="colmap_diffused_point_count",
                label="##colmap_diffused_point_count",
                default=500000,
                speed=1000.0,
                min_value=1,
                max_value=10000000,
                tooltip="Number of resampled diffused COLMAP points.",
            )
            imgui.end_disabled()
            imgui.table_next_column()
            imgui.begin_disabled(not diffused_enabled)
            ToolkitWindow._draw_clamped_float(
                ui,
                key="colmap_diffused_diffusion_radius",
                label="##colmap_diffused_diffusion_radius",
                default=1.0,
                speed=0.01,
                min_value=0.0,
                max_value=16.0,
                fmt="%.4f",
                tooltip="Local diffusion multiplier applied before point synthesis.",
                flags=imgui.SliderFlags_.logarithmic.value,
            )
            imgui.end_disabled()
            _row_nn_scale("colmap_diffused_nn_radius_scale_coef", 0.5, diffused_enabled)

            imgui.table_next_row()
            custom_ply_enabled = _row_toggle("colmap_custom_ply_enabled")
            imgui.table_next_column()
            imgui.text_unformatted("Custom PLY")
            imgui.table_next_column()
            imgui.text_disabled("scene")
            imgui.table_next_column()
            imgui.begin_disabled(not custom_ply_enabled)
            self._draw_import_path_selector(ui, label="PLY", key="colmap_custom_ply_path", button_label="Browse PLY...", callback=self.callbacks.browse_colmap_ply)
            imgui.end_disabled()
            _row_nn_scale("colmap_custom_ply_nn_radius_scale_coef", 1.0, custom_ply_enabled)

            imgui.table_next_row()
            custom_mesh_enabled = _row_toggle("colmap_custom_mesh_enabled")
            imgui.table_next_column()
            imgui.text_unformatted("Custom Mesh")
            imgui.table_next_column()
            imgui.begin_disabled(not custom_mesh_enabled)
            ToolkitWindow._draw_clamped_int(
                ui,
                key="colmap_custom_mesh_point_count",
                label="##colmap_custom_mesh_point_count",
                default=500000,
                speed=1000.0,
                min_value=1,
                max_value=10000000,
                tooltip="Number of uniformly sampled mesh surface points.",
            )
            imgui.end_disabled()
            imgui.table_next_column()
            imgui.begin_disabled(not custom_mesh_enabled)
            self._draw_import_path_selector(ui, label="Mesh", key="colmap_custom_mesh_path", button_label="Browse Mesh...", callback=self.callbacks.browse_colmap_mesh)
            imgui.end_disabled()
            _row_nn_scale("colmap_custom_mesh_nn_radius_scale_coef", 0.5, custom_mesh_enabled)

            imgui.table_next_row()
            fibonacci_enabled = _row_toggle("colmap_fibonacci_sphere_enabled")
            imgui.table_next_column()
            imgui.text_unformatted("Fibonacci Sky Sphere")
            imgui.table_next_column()
            imgui.begin_disabled(not fibonacci_enabled)
            ToolkitWindow._draw_clamped_int(
                ui,
                key="colmap_fibonacci_sphere_point_count",
                label="##colmap_fibonacci_sphere_point_count",
                default=50000,
                speed=1000.0,
                min_value=1,
                max_value=10000000,
                tooltip="Number of sky-sphere samples.",
            )
            imgui.end_disabled()
            imgui.table_next_column()
            imgui.begin_disabled(not fibonacci_enabled)
            ToolkitWindow._draw_clamped_float(
                ui,
                key="colmap_fibonacci_sphere_radius_multiplier",
                label="##colmap_fibonacci_sphere_radius_multiplier",
                default=2.0,
                speed=0.01,
                min_value=0.0,
                max_value=100.0,
                fmt="%.3f",
                tooltip="Multiplier applied to the max aligned COLMAP point distance from the mean camera pose; each shell point also gets deterministic radial jitter within +/-10%.",
                flags=imgui.SliderFlags_.logarithmic.value,
            )
            imgui.end_disabled()
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.table_next_column()
            imgui.text_disabled("Sphere Color")
            imgui.table_next_column()
            imgui.text_disabled("rgb")
            imgui.table_next_column()
            imgui.begin_disabled(not fibonacci_enabled)
            color = np.asarray(ui._values.get("colmap_fibonacci_sphere_color", (0.8, 0.8, 0.8)), dtype=np.float32).reshape(3)
            changed, edited_color = imgui.color_edit3(
                "##colmap_fibonacci_sphere_color",
                imgui.ImVec4(float(color[0]), float(color[1]), float(color[2]), 1.0),
            )
            if changed:
                ui._values["colmap_fibonacci_sphere_color"] = (float(edited_color.x), float(edited_color.y), float(edited_color.z))
            if imgui.is_item_hovered():
                imgui.set_item_tooltip("RGB color assigned to synthesized Fibonacci sky-sphere points.")
            imgui.end_disabled()
            imgui.table_next_column()
            _row_nn_scale("colmap_fibonacci_sphere_nn_radius_scale_coef", 1.0, fibonacci_enabled)

            imgui.end_table()

    def _draw_colmap_import_window(self, ui: ViewerUI) -> None:
        if not self._show_colmap_import:
            return
        self._dock_tool_window(imgui.Cond_.appearing.value)
        scale = max(self._applied_interface_scale, 1.0)
        imgui.set_next_window_pos(imgui.ImVec2(56.0 * scale, self._menu_bar_height + 40.0 * scale), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(540.0 * scale, 0.0), imgui.Cond_.first_use_ever.value)
        opened, self._show_colmap_import = imgui.begin("COLMAP Import", True)
        ToolkitWindow._register_non_viewport_window(self)
        import_active = bool(ui._values.get("_colmap_import_active", False))
        if import_active and not self._show_colmap_import:
            self._show_colmap_import = True
        if opened:
            imgui.text_wrapped("Select the dataset root, verify the RGB image folder, optionally provide a depth folder, choose import-time downscale, then enable any combination of initialization sources.")
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
            for label, key, button_label, callback, tooltip in (
                ("COLMAP Root", "colmap_root_path", "Browse Root...", self.callbacks.browse_colmap_root, None),
                ("Image Folder", "colmap_images_root", "Browse Image Folder...", self.callbacks.browse_colmap_images, None),
                ("Depth Folder", "colmap_depth_root", "Browse Depth Folder...", self.callbacks.browse_colmap_depth, "Optional root containing 16-bit depth PNGs matched to RGB images by relative path stem."),
            ):
                self._draw_import_path_selector(ui, label=label, key=key, button_label=button_label, callback=callback)
                if tooltip is not None:
                    ToolkitWindow._set_tooltip(tooltip)
                imgui.spacing()
            camera_rows = tuple(ui._values.get("_colmap_camera_rows", ()))
            if len(camera_rows) > 0:
                self._draw_colmap_camera_selection_table(ui, camera_rows)
            imgui.spacing()
            for label, key, tooltip in (
                ("Auto Rotate Scene", "colmap_auto_rotate_scene", "Apply the COLMAP import auto-alignment pass that reorients the reconstructed scene from the camera layout. Disable this to preserve the original COLMAP orientation."),
                ("Compress Dataset using BC7", "compress_dataset_using_bc7", "Compress imported training images into BC7 DDS files under Image Folder/cache and reuse that cache on later loads."),
                ("Initialize Colors From Images", "colmap_training_image_color_init", "After initialization, project each splat into all imported training images and use the nearest valid sampled color."),
            ):
                changed, value = imgui.checkbox(label, bool(ui._values.get(key, False)))
                if changed:
                    ui._values[key] = bool(value)
                ToolkitWindow._set_tooltip(tooltip)
                imgui.spacing()
            alpha_mode = min(max(int(ui._values.get("target_alpha_mode", 0)), 0), len(TARGET_ALPHA_MODE_LABELS) - 1)
            if imgui.begin_combo("Target Alpha", TARGET_ALPHA_MODE_LABELS[alpha_mode]):
                for idx, option in enumerate(TARGET_ALPHA_MODE_LABELS):
                    selected = idx == alpha_mode
                    if imgui.selectable(option, selected)[0]:
                        ui._values["target_alpha_mode"] = idx
                        ui._values["use_target_alpha_mask"] = idx == 1
                    if selected:
                        imgui.set_item_default_focus()
                imgui.end_combo()
            ToolkitWindow._set_tooltip("Choose whether target alpha is ignored, used to skip transparent pixels, or trained as part of the L1 target.")
            imgui.spacing()
            self._draw_colmap_downscale_controls(ui)
            imgui.spacing()
            self._draw_colmap_init_mode_controls(ui)
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
        scale = max(self._applied_interface_scale, 1.0)
        self._dock_tool_window(imgui.Cond_.appearing.value)
        imgui.set_next_window_pos(imgui.ImVec2(72.0 * scale, self._menu_bar_height + 56.0 * scale), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(_HISTOGRAM_WINDOW_WIDTH * scale, _HISTOGRAM_WINDOW_HEIGHT * scale), imgui.Cond_.first_use_ever.value)
        opened, show = imgui.begin("Histograms", True)
        ToolkitWindow._register_non_viewport_window(self)
        ui._values["show_histograms"] = bool(show)
        ui._values["_show_histograms_prev"] = bool(show)
        if opened:
            self._draw_histogram_controls(ui)
            status = str(ui._texts.get("histogram_status", "")).strip()
            payload = ui._values.get("_histogram_payload")
            range_payload = ui._values.get("_histogram_range_payload")
            self._update_histogram_range(ui, payload, range_payload)
            if status:
                imgui.text_disabled(status)
                imgui.separator()
            if payload is None or getattr(payload, "counts", np.zeros((0, 0), dtype=np.int64)).size == 0 or int(np.sum(payload.counts)) == 0:
                imgui.text_wrapped("No live splat histogram data is available yet.")
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

    @staticmethod
    def _update_training_views_sort_state(ui: ViewerUI) -> None:
        sort_specs = imgui.table_get_sort_specs()
        if sort_specs is None or int(getattr(sort_specs, "specs_count", 0)) <= 0:
            return
        try:
            spec = sort_specs.get_specs(0)
        except Exception:
            return
        sort_key = _TRAINING_VIEWS_SORT_KEY_BY_USER_ID.get(int(getattr(spec, "column_user_id", 0)), _TRAINING_VIEWS_SORT_DEFAULT_COLUMN)
        sort_direction = int(getattr(spec, "sort_direction", imgui.SortDirection_.ascending.value))
        if sort_direction == int(imgui.SortDirection_.none.value):
            return
        ui._values["_training_views_sort_column"] = sort_key
        ui._values["_training_views_sort_descending"] = sort_direction == int(imgui.SortDirection_.descending.value)
        try:
            sort_specs.specs_dirty = False
        except Exception:
            pass

    def _draw_training_views_window(self, ui: ViewerUI) -> None:
        if not bool(ui._values.get("show_training_views", False)):
            return
        scale = max(self._applied_interface_scale, 1.0)
        self._dock_tool_window(imgui.Cond_.appearing.value)
        imgui.set_next_window_pos(imgui.ImVec2(88.0 * scale, self._menu_bar_height + 64.0 * scale), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(980.0 * scale, 420.0 * scale), imgui.Cond_.first_use_ever.value)
        opened, show = imgui.begin("Training Views", True)
        ToolkitWindow._register_non_viewport_window(self)
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
                    | imgui.TableFlags_.sortable.value
                )
                if imgui.begin_table("##training_views", 9, flags):
                    for label, _sort_key, user_id, column_flags, width in _TRAINING_VIEWS_SORT_COLUMNS:
                        imgui.table_setup_column(label, column_flags, width, user_id)
                    imgui.table_headers_row()
                    ToolkitWindow._update_training_views_sort_state(ui)
                    for row in rows:
                        imgui.table_next_row()
                        values = (
                            str(row.get("image_name", "")),
                            str(row.get("resolution", "")),
                            self._training_views_value_text(row.get("fx")),
                            self._training_views_value_text(row.get("fy")),
                            self._training_views_value_text(row.get("cx")),
                            self._training_views_value_text(row.get("cy")),
                            self._training_views_value_text(row.get("camera_min_dist"), precision=3),
                            self._training_views_value_text(row.get("loss"), precision=4),
                            self._training_views_value_text(row.get("psnr"), precision=2),
                        )
                        for value in values:
                            imgui.table_next_column()
                            imgui.text_unformatted(value)
                    imgui.end_table()
        imgui.end()

    def _draw_photometric_compensation_window(self, ui: ViewerUI) -> None:
        if not bool(ui._values.get("show_photometric_compensation", False)):
            return
        scale = max(self._applied_interface_scale, 1.0)
        self._dock_tool_window(imgui.Cond_.appearing.value)
        imgui.set_next_window_pos(imgui.ImVec2(104.0 * scale, self._menu_bar_height + 72.0 * scale), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(760.0 * scale, 720.0 * scale), imgui.Cond_.first_use_ever.value)
        opened, show = imgui.begin("Photometric Compensation", True)
        ToolkitWindow._register_non_viewport_window(self)
        ui._values["show_photometric_compensation"] = bool(show)
        if opened:
            for key in ("photometric_status", "photometric_time", "photometric_loss", "photometric_regularization", "photometric_pairs"):
                text = str(ui._texts.get(key, "")).strip()
                if text:
                    imgui.text_wrapped(text)
            if bool(ui._values.get("_photometric_prepare_active", False)):
                progress = max(0.0, min(float(ui._values.get("_photometric_prepare_fraction", 0.0)), 1.0))
                current_name = str(ui._texts.get("photometric_prepare_current", "")).strip()
                imgui.progress_bar(progress, imgui.ImVec2(-1.0, 0.0))
                if current_name:
                    imgui.text_disabled(current_name)
            imgui.spacing()
            half_w = imgui.get_content_region_avail().x * 0.31
            if imgui.button("Start", imgui.ImVec2(half_w, 0.0)):
                self.callbacks.start_photometric()
            imgui.same_line()
            if imgui.button("Stop", imgui.ImVec2(half_w, 0.0)):
                self.callbacks.stop_photometric()
            imgui.same_line()
            if imgui.button("Reset", imgui.ImVec2(half_w, 0.0)):
                self.callbacks.reset_photometric()
            changed, apply_to_targets = imgui.checkbox("Apply to Gaussian Targets", bool(ui._values.get("photometric_apply_to_targets", True)))
            if changed:
                ui._values["photometric_apply_to_targets"] = bool(apply_to_targets)
            changed, steps_per_frame = imgui.drag_int("Steps / Frame", int(ui._values.get("photometric_steps_per_frame", 1)), 0.1, 1, 32)
            if changed:
                ui._values["photometric_steps_per_frame"] = max(int(steps_per_frame), 1)
            imgui.separator_text("Dataset")
            ToolkitWindow._draw_clamped_int(
                ui,
                key="photometric_batch_pair_count",
                label="Batch Pairs",
                default=int(_PHOTOMETRIC_UI_DEFAULTS.batch_pair_count),
                speed=32.0,
                min_value=1,
                max_value=65536,
                tooltip="Number of tracked observation pairs sampled per photometric optimizer step.",
            )
            ToolkitWindow._draw_clamped_int(
                ui,
                key="photometric_neighborhood_size",
                label="Neighborhood",
                default=int(_PHOTOMETRIC_UI_DEFAULTS.neighborhood_size),
                speed=1.0,
                min_value=1,
                max_value=15,
                tooltip="NxN patch size used for color comparison. Dataset rebuild occurs on Reset.",
            )
            ToolkitWindow._draw_clamped_int(
                ui,
                key="photometric_min_track_length",
                label="Min Track Length",
                default=int(_PHOTOMETRIC_UI_DEFAULTS.min_track_length),
                speed=1.0,
                min_value=2,
                max_value=32,
                tooltip="Minimum COLMAP observation count required to include a tracked point. Dataset rebuild occurs on Reset.",
            )
            imgui.text_disabled("Reset photometric training to apply neighborhood and track filters.")
            imgui.separator_text("Optimization")
            ToolkitWindow._draw_clamped_float(
                ui,
                key="photometric_learning_rate",
                label="Learning Rate",
                default=float(_PHOTOMETRIC_UI_DEFAULTS.learning_rate),
                speed=0.001,
                min_value=1e-4,
                max_value=1.0,
                fmt="%.4f",
                tooltip="Base learning rate for photometric compensation training.",
                flags=imgui.SliderFlags_.logarithmic.value,
            )
            ToolkitWindow._draw_clamped_float(
                ui,
                key="photometric_grad_component_clip",
                label="Grad Clip",
                default=float(_PHOTOMETRIC_UI_DEFAULTS.grad_component_clip),
                speed=0.05,
                min_value=1e-4,
                max_value=1000.0,
                fmt="%.4f",
                tooltip="Per-component gradient clipping threshold for photometric optimizer updates.",
                flags=imgui.SliderFlags_.logarithmic.value,
            )
            ToolkitWindow._draw_clamped_float(
                ui,
                key="photometric_grad_norm_clip",
                label="Grad Norm Clip",
                default=float(_PHOTOMETRIC_UI_DEFAULTS.grad_norm_clip),
                speed=0.05,
                min_value=1e-4,
                max_value=1000.0,
                fmt="%.4f",
                tooltip="Per-frame packed gradient norm clipping threshold for photometric optimizer updates.",
                flags=imgui.SliderFlags_.logarithmic.value,
            )
            ToolkitWindow._draw_clamped_float(
                ui,
                key="photometric_max_update",
                label="Max Update",
                default=float(_PHOTOMETRIC_UI_DEFAULTS.max_update),
                speed=0.001,
                min_value=1e-4,
                max_value=1.0,
                fmt="%.4f",
                tooltip="Maximum absolute parameter update applied per photometric optimizer step.",
                flags=imgui.SliderFlags_.logarithmic.value,
            )
            imgui.separator_text("Learning Rate Multipliers")
            for key, label, default, tooltip in (
                (
                    "photometric_exposure_lr_mul",
                    "Exposure LR",
                    float(_PHOTOMETRIC_UI_DEFAULTS.exposure_lr_mul),
                    "Learning-rate multiplier for exposure EV parameters.",
                ),
                (
                    "photometric_vignette_lr_mul",
                    "Vignette LR",
                    float(_PHOTOMETRIC_UI_DEFAULTS.vignette_lr_mul),
                    "Learning-rate multiplier for vignette parameters.",
                ),
                (
                    "photometric_chroma_lr_mul",
                    "Chroma LR",
                    float(_PHOTOMETRIC_UI_DEFAULTS.chroma_lr_mul),
                    "Learning-rate multiplier for chromatic-aberration parameters.",
                ),
                (
                    "photometric_crf_lr_mul",
                    "CRF LR",
                    float(_PHOTOMETRIC_UI_DEFAULTS.crf_lr_mul),
                    "Learning-rate multiplier for camera-response-curve parameters.",
                ),
            ):
                ToolkitWindow._draw_clamped_float(
                    ui,
                    key=key,
                    label=label,
                    default=default,
                    speed=0.0025,
                    min_value=0.0,
                    max_value=4.0,
                    fmt="%.4f",
                    tooltip=tooltip,
                )
            imgui.separator_text("Identity Regularization")
            for key, label, default, tooltip in (
                (
                    "photometric_exposure_regularize_weight",
                    "Exposure",
                    float(_PHOTOMETRIC_UI_DEFAULTS.exposure_regularize_weight),
                    "Quadratic identity regularization weight for exposure EV parameters.",
                ),
                (
                    "photometric_vignette_regularize_weight",
                    "Vignette",
                    float(_PHOTOMETRIC_UI_DEFAULTS.vignette_regularize_weight),
                    "Quadratic identity regularization weight for vignette parameters.",
                ),
                (
                    "photometric_chroma_regularize_weight",
                    "Chroma",
                    float(_PHOTOMETRIC_UI_DEFAULTS.chroma_regularize_weight),
                    "Quadratic identity regularization weight for chromatic-aberration parameters.",
                ),
                (
                    "photometric_crf_regularize_weight",
                    "CRF",
                    float(_PHOTOMETRIC_UI_DEFAULTS.crf_regularize_weight),
                    "Quadratic identity regularization weight for camera-response-curve parameters.",
                ),
            ):
                ToolkitWindow._draw_clamped_float(
                    ui,
                    key=key,
                    label=label,
                    default=default,
                    speed=0.0025,
                    min_value=0.0,
                    max_value=2.0,
                    fmt="%.4f",
                    tooltip=tooltip,
                )
            imgui.separator_text("L1 Regularization")
            for key, label, default, tooltip in (
                (
                    "photometric_exposure_l1_weight",
                    "Exposure L1",
                    float(_PHOTOMETRIC_UI_DEFAULTS.exposure_l1_weight),
                    "L1 identity regularization weight for exposure EV parameters.",
                ),
                (
                    "photometric_vignette_l1_weight",
                    "Vignette L1",
                    float(_PHOTOMETRIC_UI_DEFAULTS.vignette_l1_weight),
                    "L1 identity regularization weight for vignette parameters.",
                ),
                (
                    "photometric_chroma_l1_weight",
                    "Chroma L1",
                    float(_PHOTOMETRIC_UI_DEFAULTS.chroma_l1_weight),
                    "L1 identity regularization weight for chromatic-aberration parameters.",
                ),
                (
                    "photometric_crf_l1_weight",
                    "CRF L1",
                    float(_PHOTOMETRIC_UI_DEFAULTS.crf_l1_weight),
                    "L1 identity regularization weight for camera-response-curve parameters.",
                ),
            ):
                ToolkitWindow._draw_clamped_float(
                    ui,
                    key=key,
                    label=label,
                    default=default,
                    speed=0.0025,
                    min_value=0.0,
                    max_value=4.0,
                    fmt="%.4f",
                    tooltip=tooltip,
                )
            frame_max = max(int(ui._values.get("photometric_frame_max", 0)), 0)
            changed, selected_frame = imgui.slider_int("Frame", int(ui._values.get("photometric_selected_frame", 0)), 0, frame_max)
            if changed:
                ui._values["photometric_selected_frame"] = int(selected_frame)
            plot_scale = self._plot_scale(ui)
            loss_arr = np.array(self.tk.photometric_loss_history, dtype=np.float64)
            step_arr = np.array(self.tk.photometric_step_history, dtype=np.float64)
            if len(loss_arr) >= 2 and len(step_arr) >= 2:
                min_len = min(len(loss_arr), len(step_arr))
                s, l = step_arr[:min_len], loss_arr[:min_len]
                if implot.begin_plot("##PhotometricLoss", imgui.ImVec2(-1, 200.0 * plot_scale)):
                    implot.setup_axes("step", "loss", 0, implot.AxisFlags_.auto_fit.value)
                    implot.setup_axis_limits(implot.ImAxis_.x1.value, float(s[0]), float(s[-1]), implot.Cond_.always.value)
                    implot.setup_axis_scale(implot.ImAxis_.y1.value, implot.Scale_.log10.value)
                    loss_spec = implot.Spec()
                    loss_spec.line_color = imgui.ImVec4(0.95, 0.68, 0.22, 1.0)
                    implot.plot_line("Photometric Loss", s, l, spec=loss_spec)
                    implot.annotation(float(s[-1]), float(l[-1]), imgui.ImVec4(0.95, 0.68, 0.22, 1.0), imgui.ImVec2(-10, -10), True, f"{l[-1]:.2e}")
                    implot.end_plot()
            else:
                imgui.text_disabled("Waiting for photometric loss data...")
            sections = tuple(ui._values.get("_photometric_param_sections", ()))
            if sections:
                imgui.separator_text("Learned Parameters")
                draw_struct_sections(sections)
            else:
                imgui.separator_text("Learned Parameters")
                imgui.text_disabled("No photometric parameters are available yet.")
        imgui.end()

    def _draw_resource_debug_window(self, ui: ViewerUI) -> None:
        if not bool(ui._values.get("show_resource_debug", False)):
            return
        scale = max(self._applied_interface_scale, 1.0)
        self._dock_tool_window(imgui.Cond_.appearing.value)
        imgui.set_next_window_pos(imgui.ImVec2(104.0 * scale, self._menu_bar_height + 72.0 * scale), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(imgui.ImVec2(_RESOURCE_DEBUG_WINDOW_WIDTH * scale, _RESOURCE_DEBUG_WINDOW_HEIGHT * scale), imgui.Cond_.first_use_ever.value)
        opened, show = imgui.begin("Buffers", True)
        ToolkitWindow._register_non_viewport_window(self)
        ui._values["show_resource_debug"] = bool(show)
        if opened:
            if imgui.button("Refresh"):
                ui._values["_resource_debug_refresh_requested"] = True
            imgui.same_line()
            if imgui.button("Refresh VRAM"):
                ui._values["_resource_debug_refresh_requested"] = True
                ui._values["_resource_debug_process_vram_requested"] = True
            snapshot = ui._values.get("_resource_debug_snapshot")
            if not isinstance(snapshot, ResourceDebugSnapshot):
                imgui.text_wrapped("No resource allocation data is available yet.")
            else:
                self._draw_resource_debug_summary(snapshot)
                imgui.same_line()
                if imgui.button("Write Log"):
                    try:
                        path = write_resource_debug_log(snapshot)
                        ui._texts["resource_debug_status"] = f"Wrote {path}"
                    except Exception as exc:
                        ui._texts["resource_debug_status"] = f"Log failed: {exc}"
                status = str(ui._texts.get("resource_debug_status", "")).strip()
                if status:
                    imgui.same_line()
                    imgui.text_disabled(status)
                imgui.separator()
                self._draw_resource_debug_table(snapshot)
        imgui.end()

    def _draw_resource_debug_summary(self, snapshot: ResourceDebugSnapshot) -> None:
        imgui.text_unformatted(f"Total Consumption: {format_resource_bytes(snapshot.total_consumption)}")
        imgui.text_unformatted(
            f"Buffers: {snapshot.buffer_count:,} | total={format_resource_bytes(snapshot.buffer_total)} | "
            f"mean={format_resource_bytes(snapshot.buffer_mean)} | median={format_resource_bytes(snapshot.buffer_median)}"
        )
        imgui.text_unformatted(f"Textures: {snapshot.texture_count:,} | total={format_resource_bytes(snapshot.texture_total)}")
        if snapshot.process_vram is not None:
            source = f" ({snapshot.process_vram_source})" if snapshot.process_vram_source else ""
            delta = 0 if snapshot.process_vram_delta is None else snapshot.process_vram_delta
            imgui.text_unformatted(f"Process VRAM: {format_resource_bytes(snapshot.process_vram)}{source}")
            imgui.text_unformatted(f"Untracked / Driver Reserved: {format_resource_bytes(delta)}")

    def _draw_resource_debug_table(self, snapshot: ResourceDebugSnapshot) -> None:
        rows = tuple(sorted(snapshot.rows, key=lambda row: (-row.byte_size, row.order)))
        if len(rows) == 0:
            imgui.text_wrapped("No tracked buffers or textures are currently reachable from the viewer.")
            return
        flags = (
            imgui.TableFlags_.row_bg.value
            | imgui.TableFlags_.borders.value
            | imgui.TableFlags_.resizable.value
            | imgui.TableFlags_.scroll_x.value
            | imgui.TableFlags_.scroll_y.value
            | imgui.TableFlags_.hideable.value
        )
        if imgui.begin_table("##resource_debug", 6, flags):
            imgui.table_setup_column("Size", imgui.TableColumnFlags_.width_fixed.value, 104.0)
            imgui.table_setup_column("Type", imgui.TableColumnFlags_.width_fixed.value, 76.0)
            imgui.table_setup_column("Details", imgui.TableColumnFlags_.width_fixed.value, 180.0)
            imgui.table_setup_column("Name", imgui.TableColumnFlags_.width_stretch.value)
            imgui.table_setup_column("Owner", imgui.TableColumnFlags_.width_stretch.value)
            imgui.table_setup_column("Usage", imgui.TableColumnFlags_.width_stretch.value)
            imgui.table_headers_row()
            for row in rows:
                imgui.table_next_row()
                for value in (format_resource_bytes(row.byte_size), row.kind, row.details, row.name, row.owner, row.usage):
                    imgui.table_next_column()
                    imgui.text_unformatted(str(value))
            imgui.end_table()

    def _draw_histogram_controls(self, ui: ViewerUI) -> None:
        if imgui.button("Refresh"):
            ui._values["_histograms_refresh_requested"] = True
            ui._values["_histogram_open_tabs"] = {}
        imgui.same_line()
        if imgui.button("Update Y Scale"):
            ui._values["_histogram_update_y_limit"] = True
        imgui.same_line()
        if imgui.button("Update Range"):
            ui._values["_histogram_update_range"] = True
        imgui.same_line()
        changed, realtime = imgui.checkbox("Update Histograms in Realtime", bool(ui._values.get("_histograms_update_realtime", False)))
        if changed:
            ui._values["_histograms_update_realtime"] = bool(realtime)
            ui._values["_histograms_realtime_next_refresh_time"] = 0.0
            if realtime:
                ui._values["_histograms_refresh_requested"] = True
        flags = imgui.TableFlags_.sizing_stretch_prop.value | imgui.TableFlags_.pad_outer_x.value
        if imgui.begin_table("##hist_controls", 2, flags):
            imgui.table_setup_column("Label", imgui.TableColumnFlags_.width_fixed.value, _HISTOGRAM_CONTROL_LABEL_WIDTH)
            imgui.table_setup_column("Value", imgui.TableColumnFlags_.width_stretch.value)
            self._draw_histogram_control_row_int(ui, "Bin Count", "##hist_bin_count", "hist_bin_count", _HISTOGRAM_BIN_COUNT_DEFAULT, 1)
            self._draw_histogram_control_row_float(ui, "Min", "##hist_min_value", "hist_min_value", _HISTOGRAM_MIN_VALUE_DEFAULT, "%.3f")
            self._draw_histogram_control_row_float(ui, "Max", "##hist_max_value", "hist_max_value", _HISTOGRAM_MAX_VALUE_DEFAULT, "%.3f")
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
            if value_key == "hist_bin_count":
                ui._values["_histograms_refresh_requested"] = True

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
            if value_key in {"hist_min_value", "hist_max_value"}:
                ui._values["_histograms_refresh_requested"] = True

    def _update_histogram_y_limit(self, ui: ViewerUI, payload: object) -> None:
        if not bool(ui._values.get("_histogram_update_y_limit", False)):
            return
        counts = np.asarray(getattr(payload, "counts", np.zeros((0, 0), dtype=np.int64)), dtype=np.float64)
        ui._values["hist_y_limit"] = max(1.3 * float(np.max(counts) if counts.size > 0 else 0.0), 1.0)
        ui._values["_histogram_update_y_limit"] = False

    def _update_histogram_range(self, ui: ViewerUI, histogram_payload: object, range_payload: object) -> None:
        if not bool(ui._values.get("_histogram_update_range", False)):
            return
        value_range = _histogram_range_from_ranges(range_payload)
        if value_range is None:
            value_range = _histogram_range_from_histogram(histogram_payload)
        if value_range is not None:
            ui._values["hist_min_value"] = float(value_range[0])
            ui._values["hist_max_value"] = float(value_range[1])
            ui._values["_histograms_refresh_requested"] = True
        ui._values["_histogram_update_range"] = False

    def _draw_histogram_groups(self, ui: ViewerUI, payload: object) -> None:
        labels = tuple(str(label) for label in getattr(payload, "param_labels", ()))
        groups = tuple(getattr(payload, "param_groups", _DEFAULT_HISTOGRAM_GROUPS))
        counts = np.asarray(getattr(payload, "counts", np.zeros((0, 0), dtype=np.int64)), dtype=np.float64)
        if counts.ndim != 2 or counts.shape[1] == 0:
            imgui.text_wrapped("Histogram payload is malformed.")
            return
        y_limit = float(ui._values.get("hist_y_limit", _HISTOGRAM_Y_LIMIT_DEFAULT))
        column_count = 1 if imgui.get_content_region_avail().x < (_HISTOGRAM_PLOT_MIN_COLUMN_WIDTH * 2.0) else 2
        grouped: dict[str, list[tuple[str, tuple[int, ...]]]] = {_HISTOGRAM_LINEAR_TAB_LABEL: [], _HISTOGRAM_LOG10_TAB_LABEL: []}
        for group_name, indices in groups:
            valid = tuple(index for index in indices if 0 <= int(index) < counts.shape[0])
            if not valid:
                continue
            grouped.setdefault(_histogram_group_type(payload, valid), []).append((str(group_name), valid))
        if not imgui.begin_tab_bar("##hist_type_tabs"):
            return
        for type_label, type_groups in grouped.items():
            if not type_groups:
                continue
            if imgui.begin_tab_item(type_label)[0]:
                imgui.spacing()
                if imgui.begin_tab_bar(f"##hist_group_tabs_{type_label}"):
                    open_tabs = ui._values.setdefault("_histogram_open_tabs", {})
                    for group_name, valid in type_groups:
                        tab_key = _histogram_tab_key(type_label, group_name)
                        tab_open = bool(open_tabs.get(tab_key, True))
                        selected, tab_open = imgui.begin_tab_item(group_name, tab_open)
                        open_tabs[tab_key] = bool(tab_open)
                        if not selected:
                            continue
                        if imgui.begin_table(f"##hist_{type_label}_{group_name}", column_count, imgui.TableFlags_.sizing_stretch_same.value):
                            for index in valid:
                                imgui.table_next_column()
                                centers = _histogram_centers_for_param(payload, index)
                                self._draw_histogram_plot(ui, labels[index] if index < len(labels) else f"param {index}", _histogram_x_label_for_param(payload, index), centers, counts[index], y_limit)
                            imgui.end_table()
                        imgui.end_tab_item()
                    imgui.end_tab_bar()
                imgui.end_tab_item()
        imgui.end_tab_bar()

    def _draw_histogram_plot(self, ui: ViewerUI, label: str, x_label: str, centers: np.ndarray, counts: np.ndarray, y_limit: float) -> None:
        imgui.text_disabled(label)
        plot_id = f"##plot_{label}"
        if implot.begin_plot(plot_id, imgui.ImVec2(-1, _HISTOGRAM_PLOT_HEIGHT * self._plot_scale(ui))):
            implot.setup_axes(x_label, "count (log10)", 0, 0)
            if centers.size > 0:
                implot.setup_axis_limits(implot.ImAxis_.x1.value, float(centers[0]), float(centers[-1]), implot.Cond_.always.value)
            implot.setup_axis_scale(implot.ImAxis_.y1.value, implot.Scale_.log10.value)
            implot.setup_axis_limits(implot.ImAxis_.y1.value, _HISTOGRAM_LOG_Y_MIN, max(float(y_limit), _HISTOGRAM_LOG_Y_MAX_MIN), implot.Cond_.always.value)
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
        scales = _histogram_param_value_scales(payload, min_values.size)
        flags = imgui.TableFlags_.row_bg.value | imgui.TableFlags_.borders.value | imgui.TableFlags_.sizing_stretch_same.value
        if not imgui.begin_table("##hist_range_debug", 5, flags):
            return
        imgui.table_setup_column("Component")
        imgui.table_setup_column("Scale")
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
            imgui.text_unformatted(scales[index])
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
        imgui.text_disabled(f"Init: {_colmap_init_mode_label(ui) if int(ui._values.get('colmap_init_mode', 0)) == 1 else _colmap_init_summary(ui)}")
        imgui.separator()

    def _section_camera(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Camera", imgui.TreeNodeFlags_.default_open.value):
            return
        camera_specs = {spec.key: spec for spec in GROUP_SPECS["Camera"]}
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
        self._draw_control(ui, camera_specs["render_background_mode"])
        if int(ui._values.get("render_background_mode", 1)) == 1:
            self._draw_control(ui, camera_specs["render_background_color"])
        imgui.text_disabled("LMB drag=look | RMB drag=pan | WASDQE=move | Wheel=speed")
        self._ctx_reset("camera_ctx", ui, tuple(spec.key for spec in GROUP_SPECS["Camera"]))
        imgui.separator()

    def _section_training_control(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Training", imgui.TreeNodeFlags_.default_open.value):
            return
        # Training info in compact table
        training_text = ui._texts.get("training", "Training: not initialized")
        time_text = ui._texts.get("training_time", "Time: n/a")
        avg_iters_text = ui._texts.get("training_iters_avg", "Avg it/s: n/a")
        loss_text = ui._texts.get("training_loss", "Loss Avg: n/a")
        ssim_text = ui._texts.get("training_ssim", "SSIM: n/a")
        density_text = ui._texts.get("training_density", "Density Avg: n/a")
        psnr_text = ui._texts.get("training_psnr", "PSNR: n/a")
        if imgui.begin_table("##train_info", 2, imgui.TableFlags_.sizing_stretch_same.value):
            imgui.table_setup_column("L", imgui.TableColumnFlags_.width_fixed.value, 50)
            imgui.table_setup_column("V")
            for label, text in (("Step", training_text), ("Time", time_text), ("Avg", avg_iters_text), ("Loss", loss_text), ("SSIM", ssim_text), ("Density", density_text), ("PSNR", psnr_text)):
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
        background_mode = int(ui._values.get("background_mode", 1))
        train_downscale_mode = int(ui._values.get("train_downscale_mode", 0))
        for spec in GROUP_SPECS[TRAINING_SETUP_GROUP]:
            if spec.setup_visibility == "background_custom" and background_mode != 0:
                continue
            if spec.setup_visibility == "downscale_auto" and train_downscale_mode != 0:
                continue
            self._draw_control(ui, spec)
        for value_key, text_key in (
            ("_training_resolution_sections", "training_resolution"),
            ("_training_downscale_sections", "training_downscale"),
        ):
            sections = tuple(ui._values.get(value_key, ()))
            if len(sections) > 0:
                draw_struct_sections(sections)
                continue
            text = ui._texts.get(text_key, "")
            if text:
                _draw_disabled_wrapped_text(text)
        schedule_status = ui._texts.get("training_schedule", "")
        if schedule_status:
            _draw_disabled_wrapped_text(schedule_status)
        refinement_sections = tuple(ui._values.get("_training_refinement_sections", ()))
        if len(refinement_sections) > 0:
            draw_struct_sections(refinement_sections)
        else:
            refinement_status = ui._texts.get("training_refinement", "")
            if refinement_status:
                _draw_disabled_wrapped_text(refinement_status)
        _draw_disabled_wrapped_text("COLMAP import can combine sparse COLMAP points, diffused COLMAP points, custom PLY seeds, custom mesh samples, and a Fibonacci sky sphere in one initialization pass.")
        self._ctx_reset("train_setup_ctx", ui, [s.key for s in GROUP_SPECS[TRAINING_SETUP_GROUP]])
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
                schedule_sections = ui._values.get("_training_schedule_sections", ())
                if schedule_sections:
                    imgui.separator()
                    imgui.text_unformatted("Current Values")
                    draw_struct_sections(schedule_sections)
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
            if imgui.begin_tab_item("Raster Grads")[0]:
                imgui.spacing()
                for key in _TRAINING_RASTER_GRAD_KEYS:
                    self._draw_control(ui, next(spec for spec in RENDER_PARAM_SPECS if spec.key == key))
                imgui.end_tab_item()
            imgui.end_tab_bar()
        self._ctx_reset("optimizer_ctx", ui, [s.key for s in GROUP_SPECS[TRAINING_OPTIMIZER_GROUP]] + list(_TRAINING_RASTER_GRAD_KEYS))
        imgui.separator()

    def _section_stability(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Stability"):
            return
        pairs = TRAIN_STABILITY_PAIRED_KEYS
        if imgui.begin_table("##stab_pairs", 2, imgui.TableFlags_.sizing_stretch_same.value | imgui.TableFlags_.no_borders_in_body.value):
            imgui.table_setup_column("Min")
            imgui.table_setup_column("Max")
            for min_key, max_key in pairs:
                min_spec = next(s for s in GROUP_SPECS[TRAINING_STABILITY_GROUP] if s.key == min_key)
                max_spec = next(s for s in GROUP_SPECS[TRAINING_STABILITY_GROUP] if s.key == max_key)
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
        paired_keys = {k for min_k, max_k in pairs for k in (min_k, max_k)}
        for spec in GROUP_SPECS[TRAINING_STABILITY_GROUP]:
            if spec.key not in paired_keys:
                self._draw_control(ui, spec)
        imgui.text_disabled("Opacity bounds and anisotropy are clamped after each ADAM step")
        self._ctx_reset("stability_ctx", ui, [s.key for s in GROUP_SPECS[TRAINING_STABILITY_GROUP]])
        imgui.separator()

    def _section_render_params(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Render Params"):
            return
        for key in ("radius_scale", "alpha_cutoff", "sort_splats_by", "trans_threshold"):
            self._draw_control(ui, next(spec for spec in RENDER_PARAM_SPECS if spec.key == key))
        self._ctx_reset("render_ctx", ui, ["radius_scale", "alpha_cutoff", "sort_splats_by", "trans_threshold"])
        imgui.separator()

    def _section_defaults_footer(self, ui: ViewerUI) -> None:
        if imgui.button("Update Defaults"):
            self.callbacks.save_defaults()
        _draw_disabled_wrapped_text(ui._texts.get("defaults_status", ""))
        imgui.separator()

    def _section_performance(self, ui: ViewerUI) -> None:
        if not imgui.collapsing_header("Plots", imgui.TreeNodeFlags_.default_open.value):
            return
        plot_scale = self._plot_scale(ui)

        iters_per_second = self._iters_per_second(self.tk.step_history, self.tk.step_time_history)
        if iters_per_second > 0.0:
            imgui.text_disabled(f"iters/s {iters_per_second:.1f}")

        fps_arr = np.array(self.tk.fps_history, dtype=np.float64)
        if len(fps_arr) >= 2:
            imgui.text_disabled(f"avg {np.mean(fps_arr):.1f}  min {np.min(fps_arr):.1f}  max {np.max(fps_arr):.1f}")
            if implot.begin_plot("##FPS", imgui.ImVec2(-1, 110.0 * plot_scale)):
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
            if implot.begin_plot("##Loss", imgui.ImVec2(-1, 180.0 * plot_scale)):
                implot.setup_axes("step", "loss", 0, implot.AxisFlags_.auto_fit.value)
                implot.setup_axis_limits(implot.ImAxis_.x1.value, float(s[0]), float(s[-1]), implot.Cond_.always.value)
                implot.setup_axis_scale(implot.ImAxis_.y1.value, implot.Scale_.log10.value)
                implot.plot_line("Avg Loss", s, l, spec=loss_spec)
                implot.annotation(float(s[-1]), float(l[-1]), imgui.ImVec4(1.0, 0.6, 0.2, 1.0), imgui.ImVec2(-10, -10), True, f"{l[-1]:.2e}")
                implot.end_plot()

        ssim_arr = np.array(self.tk.ssim_history, dtype=np.float64)
        if len(ssim_arr) >= 2 and len(step_arr) >= 2:
            min_len = min(len(ssim_arr), len(step_arr))
            s, d = step_arr[:min_len], ssim_arr[:min_len]
            imgui.separator_text("SSIM")
            ssim_spec = implot.Spec()
            ssim_spec.line_color = imgui.ImVec4(0.9, 0.35, 0.35, 1.0)
            if implot.begin_plot("##SSIM", imgui.ImVec2(-1, 180.0 * plot_scale)):
                implot.setup_axes("step", "SSIM", 0, implot.AxisFlags_.auto_fit.value)
                implot.setup_axis_limits(implot.ImAxis_.x1.value, float(s[0]), float(s[-1]), implot.Cond_.always.value)
                implot.setup_axis_limits(implot.ImAxis_.y1.value, 0.0, 1.0, implot.Cond_.once.value)
                implot.plot_line("SSIM", s, d, spec=ssim_spec)
                implot.annotation(float(s[-1]), float(d[-1]), imgui.ImVec4(0.9, 0.35, 0.35, 1.0), imgui.ImVec2(-10, -10), True, ToolkitWindow._format_plot_metric_value(float(d[-1])))
                implot.end_plot()

        psnr_arr = np.array(self.tk.psnr_history, dtype=np.float64)
        if len(psnr_arr) >= 2 and len(step_arr) >= 2:
            min_len = min(len(psnr_arr), len(step_arr))
            s, p = step_arr[:min_len], psnr_arr[:min_len]
            imgui.separator_text("PSNR")
            psnr_spec = implot.Spec()
            psnr_spec.line_color = imgui.ImVec4(0.3, 0.85, 0.5, 1.0)
            if implot.begin_plot("##PSNR", imgui.ImVec2(-1, 180.0 * plot_scale)):
                implot.setup_axes("step", "PSNR (dB)", 0, implot.AxisFlags_.auto_fit.value)
                implot.setup_axis_limits(implot.ImAxis_.x1.value, float(s[0]), float(s[-1]), implot.Cond_.always.value)
                implot.plot_line("PSNR", s, p, spec=psnr_spec)
                implot.annotation(float(s[-1]), float(p[-1]), imgui.ImVec4(0.3, 0.85, 0.5, 1.0), imgui.ImVec2(-10, -10), True, f"{p[-1]:.2f}")
                implot.end_plot()

    # -- Helpers --

    _TOOLTIPS = UI_TOOLTIPS

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
        elif spec.kind in ("input_float2", "input_float3"):
            size = 2 if spec.kind == "input_float2" else 3
            value = np.asarray(ui._values[key], dtype=np.float32).reshape(-1)
            if value.size == 0:
                value = np.zeros((size,), dtype=np.float32)
            if value.size < size:
                value = np.pad(value, (0, size - value.size), mode="edge")
            changed, val = (imgui.drag_float2 if size == 2 else imgui.drag_float3)(
                label,
                [float(v) for v in value[:size]],
                float(spec.kwargs.get("step", 0.001)),
                0.0,
                0.0,
                spec.kwargs.get("format", "%.3f"),
            )
            if changed:
                ui._values[key] = tuple(float(v) for v in val[:size])
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

    def _reset_values(self, ui: ViewerUI, keys) -> None:
        for key in keys:
            if key in _ALL_DEFAULTS:
                ui._values[key] = _ALL_DEFAULTS[key]

    def _ctx_reset(self, ctx_id: str, ui: ViewerUI, keys) -> None:
        if imgui.begin_popup_context_item(ctx_id):
            if imgui.selectable("Reset to Defaults")[0]:
                self._reset_values(ui, keys)
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
    values[_THEME_KEY] = int(_VIEWER_CONTROL_DEFAULTS.get("theme", 0))
    for spec in RENDER_PARAM_SPECS:
        values[spec.key] = spec.kwargs.get("value", 0)
    for spec in DEBUG_RENDER_SPECS:
        values[spec.key] = spec.kwargs.get("value", 0)
    RendererParams.from_renderer(renderer).apply_ui_values(values, _renderer_atomic_mode_index, _renderer_debug_mode_index, _threshold_from_band_range)
    for key in ("colmap_root_path", "colmap_database_path", "colmap_images_root", "colmap_depth_root", "colmap_custom_ply_path", "colmap_custom_mesh_path"):
        values[key] = ""
    values["colmap_selected_camera_ids"] = ()
    for key, cast in _VIEWER_IMPORT_EXPORT_FIELDS:
        values[key] = cast(_VIEWER_IMPORT_DEFAULTS.get(key, False if cast is bool else 0 if cast is int else 20.0))
    values["colmap_fibonacci_sphere_radius_multiplier"] = float(_VIEWER_IMPORT_DEFAULTS.get("colmap_fibonacci_sphere_radius_multiplier", _VIEWER_IMPORT_DEFAULTS.get("colmap_fibonacci_sphere_radius", 2.0)))
    values["show_resource_debug"] = False
    values["show_photometric_compensation"] = False
    values["photometric_apply_to_targets"] = True
    values["photometric_steps_per_frame"] = 1
    values["photometric_selected_frame"] = 0
    values["photometric_batch_pair_count"] = int(_VIEWER_UI_DEFAULTS.get("photometric_batch_pair_count", _PHOTOMETRIC_UI_DEFAULTS.batch_pair_count))
    values["photometric_neighborhood_size"] = int(_VIEWER_UI_DEFAULTS.get("photometric_neighborhood_size", _PHOTOMETRIC_UI_DEFAULTS.neighborhood_size))
    values["photometric_min_track_length"] = int(_VIEWER_UI_DEFAULTS.get("photometric_min_track_length", _PHOTOMETRIC_UI_DEFAULTS.min_track_length))
    values["photometric_learning_rate"] = float(_VIEWER_UI_DEFAULTS.get("photometric_learning_rate", _PHOTOMETRIC_UI_DEFAULTS.learning_rate))
    values["photometric_grad_component_clip"] = float(_VIEWER_UI_DEFAULTS.get("photometric_grad_component_clip", _PHOTOMETRIC_UI_DEFAULTS.grad_component_clip))
    values["photometric_grad_norm_clip"] = float(_VIEWER_UI_DEFAULTS.get("photometric_grad_norm_clip", _PHOTOMETRIC_UI_DEFAULTS.grad_norm_clip))
    values["photometric_max_update"] = float(_VIEWER_UI_DEFAULTS.get("photometric_max_update", _PHOTOMETRIC_UI_DEFAULTS.max_update))
    values["photometric_exposure_lr_mul"] = float(_VIEWER_UI_DEFAULTS.get("photometric_exposure_lr_mul", _PHOTOMETRIC_UI_DEFAULTS.exposure_lr_mul))
    values["photometric_vignette_lr_mul"] = float(_VIEWER_UI_DEFAULTS.get("photometric_vignette_lr_mul", _PHOTOMETRIC_UI_DEFAULTS.vignette_lr_mul))
    values["photometric_chroma_lr_mul"] = float(_VIEWER_UI_DEFAULTS.get("photometric_chroma_lr_mul", _PHOTOMETRIC_UI_DEFAULTS.chroma_lr_mul))
    values["photometric_crf_lr_mul"] = float(_VIEWER_UI_DEFAULTS.get("photometric_crf_lr_mul", _PHOTOMETRIC_UI_DEFAULTS.crf_lr_mul))
    values["photometric_exposure_regularize_weight"] = float(_VIEWER_UI_DEFAULTS.get("photometric_exposure_regularize_weight", _PHOTOMETRIC_UI_DEFAULTS.exposure_regularize_weight))
    values["photometric_vignette_regularize_weight"] = float(_VIEWER_UI_DEFAULTS.get("photometric_vignette_regularize_weight", _PHOTOMETRIC_UI_DEFAULTS.vignette_regularize_weight))
    values["photometric_chroma_regularize_weight"] = float(_VIEWER_UI_DEFAULTS.get("photometric_chroma_regularize_weight", _PHOTOMETRIC_UI_DEFAULTS.chroma_regularize_weight))
    values["photometric_crf_regularize_weight"] = float(_VIEWER_UI_DEFAULTS.get("photometric_crf_regularize_weight", _PHOTOMETRIC_UI_DEFAULTS.crf_regularize_weight))
    values["photometric_exposure_l1_weight"] = float(_VIEWER_UI_DEFAULTS.get("photometric_exposure_l1_weight", _PHOTOMETRIC_UI_DEFAULTS.exposure_l1_weight))
    values["photometric_vignette_l1_weight"] = float(_VIEWER_UI_DEFAULTS.get("photometric_vignette_l1_weight", _PHOTOMETRIC_UI_DEFAULTS.vignette_l1_weight))
    values["photometric_chroma_l1_weight"] = float(_VIEWER_UI_DEFAULTS.get("photometric_chroma_l1_weight", _PHOTOMETRIC_UI_DEFAULTS.chroma_l1_weight))
    values["photometric_crf_l1_weight"] = float(_VIEWER_UI_DEFAULTS.get("photometric_crf_l1_weight", _PHOTOMETRIC_UI_DEFAULTS.crf_l1_weight))
    for key, cast in _VIEWER_UI_EXPORT_FIELDS[:-3]:
        default = False if cast is bool else 0 if cast is int else 0.0
        values[key] = cast(_VIEWER_UI_DEFAULTS.get(key, default))
    values.update({
        "_exit_confirmation_open": False,
        "_histograms_refresh_requested": False,
        "_histograms_update_realtime": False,
        "_histograms_realtime_next_refresh_time": 0.0,
        "_show_histograms_prev": False,
        "_histogram_update_y_limit": True,
        "_histogram_update_range": False,
        "_histogram_payload": None,
        "_histogram_range_payload": None,
        "_resource_debug_snapshot": None,
        "_resource_debug_next_update": 0.0,
        "_resource_debug_refresh_requested": True,
        "_resource_debug_process_vram_requested": False,
        "_menu_bar_device_vram_bytes": None,
        "_menu_bar_device_vram_source": "",
        "_menu_bar_device_vram_total_bytes": None,
        "_menu_bar_device_vram_total_source": "",
        "_menu_bar_dataset_vram_bytes": 0,
        "_menu_bar_app_vram_bytes": 0,
        "_menu_bar_total_vram_bytes": 0,
        "_menu_bar_resource_next_update": 0.0,
        "_training_views_rows": (),
        "_training_view_overlay_segments": (),
        "_training_views_sort_column": _TRAINING_VIEWS_SORT_DEFAULT_COLUMN,
        "_training_views_sort_descending": False,
        "_loss_debug_frame_max": 0,
        "training_camera_full_resolution": False,
        "_training_camera_pose_available": False,
        "_training_camera_struct_sections": (),
        "_training_camera_colmap_points_payload": None,
        "_training_resolution_sections": (),
        "_training_downscale_sections": (),
        "_training_schedule_sections": (),
        "_training_refinement_sections": (),
        "_photometric_param_sections": (),
        "photometric_frame_max": 0,
        "_viewport_sh_band": int(_VIEWER_UI_DEFAULTS["viewport_sh_band"]),
        "_viewport_sh_control_key": str(_VIEWER_UI_DEFAULTS["viewport_sh_control_key"]),
        "_viewport_sh_stage_label": str(_VIEWER_UI_DEFAULTS["viewport_sh_stage_label"]),
        "_colmap_point_stats": None,
        "_colmap_camera_rows": (),
        "_colmap_import_active": False,
        "_colmap_import_fraction": 0.0,
        "_photometric_prepare_active": False,
        "_photometric_prepare_fraction": 0.0,
        "_can_export_ply": False,
    })

    texts: dict[str, str] = {
        key: "" for key in (
            "fps", "path", "scene_stats", "render_stats", "training",
            "training_time", "training_iters_avg", "training_loss", "training_ssim", "training_density", "training_psnr", "training_instability", "error",
            "photometric_status", "photometric_time", "photometric_loss", "photometric_regularization", "photometric_pairs",
            "photometric_prepare_current",
            "loss_debug_frame", "loss_debug_psnr",
            "colmap_import_status", "colmap_import_current",
            "training_schedule",
            "histogram_status",
            "resource_debug_status",
            "setup_hint", "stability_hint", "defaults_status",
        )
    }
    return ViewerUI(_values=values, _texts=texts)


def create_toolkit_window(device: spy.Device, width: int, height: int) -> ToolkitWindow:
    return ToolkitWindow(device=device, width=width, height=height)
