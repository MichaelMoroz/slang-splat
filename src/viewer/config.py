from __future__ import annotations

from typing import Any

from ..app.training_controls import TRAINING_BUILD_ARG_UI_KEYS
from ..renderer.render_params import CACHED_RASTER_GRAD_ATOMIC_MODE_VALUES, SORT_SPLATS_BY_VALUES, cached_raster_grad_key
from .ui_schema import _DEBUG_MODE_VALUES

RUN_UI_VALUE_FIELDS: tuple[str, ...] = (
    "source",
    "ply_path",
    "colmap_root_path",
    "colmap_database_path",
    "colmap_images_root",
    "colmap_depth_root",
    "colmap_alpha_mask_root",
    "colmap_custom_ply_path",
    "colmap_custom_mesh_path",
    "colmap_selected_camera_ids",
    "seed",
    "train_iters",
    "output_ply",
)

_PATH_VALUE_FIELDS = frozenset(
    (
        "ply_path",
        "colmap_root_path",
        "colmap_database_path",
        "colmap_images_root",
        "colmap_depth_root",
        "colmap_alpha_mask_root",
        "colmap_custom_ply_path",
        "colmap_custom_mesh_path",
        "output_ply",
    )
)
_VIEWER_UI_KEY_MAP = {
    "viewport_sh_band": "_viewport_sh_band",
    "viewport_sh_control_key": "_viewport_sh_control_key",
    "viewport_sh_stage_label": "_viewport_sh_stage_label",
}
_RENDERER_FIELD_KEYS = {
    "radius_scale": "radius_scale",
    "alpha_cutoff": "alpha_cutoff",
    "max_anisotropy": "max_anisotropy",
    "transmittance_threshold": "trans_threshold",
    "debug_grad_norm_threshold": "debug_grad_norm_threshold",
    "debug_ellipse_thickness_px": "debug_ellipse_thickness_px",
    "debug_gaussian_scale_multiplier": "debug_gaussian_scale_multiplier",
    "debug_min_opacity": "debug_min_opacity",
    "debug_opacity_multiplier": "debug_opacity_multiplier",
    "debug_ellipse_scale_multiplier": "debug_ellipse_scale_multiplier",
    "debug_depth_local_mismatch_smooth_radius": "debug_depth_local_mismatch_smooth_radius",
    "debug_depth_local_mismatch_reject_radius": "debug_depth_local_mismatch_reject_radius",
    "debug_sh_coeff_index": "debug_sh_coeff_index",
}
_RENDERER_RANGE_KEYS = {
    "debug_splat_age_range": ("debug_splat_age_min", "debug_splat_age_max"),
    "debug_density_range": ("debug_density_min", "debug_density_max"),
    "debug_contribution_range": ("debug_contribution_min", "debug_contribution_max"),
    "debug_refinement_distribution_range": ("debug_refinement_distribution_min", "debug_refinement_distribution_max"),
    "debug_depth_mean_range": ("debug_depth_mean_min", "debug_depth_mean_max"),
    "debug_depth_std_range": ("debug_depth_std_min", "debug_depth_std_max"),
    "debug_depth_local_mismatch_range": ("debug_depth_local_mismatch_min", "debug_depth_local_mismatch_max"),
}


def _section(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    node: object = data
    for key in keys:
        if not isinstance(node, dict):
            return {}
        node = node.get(key, {})
    return node if isinstance(node, dict) else {}


def _as_tuple(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _set_run_value(values: dict[str, object], key: str, value: object) -> None:
    if key == "colmap_selected_camera_ids":
        values[key] = tuple(int(camera_id) for camera_id in _as_tuple(value))
    elif key in _PATH_VALUE_FIELDS:
        values[key] = "" if value is None else str(value)
    elif key in {"seed", "train_iters"}:
        values[key] = int(0 if value is None else value)
    else:
        values[key] = value


def _set_many(values: dict[str, object], section: dict[str, Any], *, key_map: dict[str, str] | None = None) -> None:
    resolved_map = {} if key_map is None else key_map
    for key, value in section.items():
        values[resolved_map.get(key, key)] = tuple(value) if isinstance(value, list) else value


def _renderer_enum_index(value: object, entries: tuple[str, ...], default: int = 0) -> int:
    if value is None:
        return default
    text = str(value)
    return entries.index(text) if text in entries else default


def _apply_renderer_overlay(values: dict[str, object], renderer: dict[str, Any]) -> None:
    for field_name, ui_key in _RENDERER_FIELD_KEYS.items():
        if field_name in renderer:
            values[ui_key] = renderer[field_name]
    if "sort_splats_by" in renderer:
        values["sort_splats_by"] = _renderer_enum_index(renderer["sort_splats_by"], SORT_SPLATS_BY_VALUES)
    if "debug_mode" in renderer:
        mode = "normal" if renderer["debug_mode"] is None else str(renderer["debug_mode"])
        values["debug_mode"] = _renderer_enum_index(mode, _DEBUG_MODE_VALUES)
    for field_name, (min_key, max_key) in _RENDERER_RANGE_KEYS.items():
        if field_name not in renderer:
            continue
        pair = _as_tuple(renderer[field_name])
        if len(pair) >= 2:
            values[min_key] = float(pair[0])
            values[max_key] = float(pair[1])
    for key in ("debug_show_ellipses", "debug_show_processed_count", "debug_show_grad_norm", "list_capacity_multiplier", "max_prepass_memory_mb"):
        if key in renderer:
            values[key] = renderer[key]
    atomic_key = cached_raster_grad_key("atomic_mode")
    if atomic_key in renderer:
        values[atomic_key] = _renderer_enum_index(renderer[atomic_key], CACHED_RASTER_GRAD_ATOMIC_MODE_VALUES)
    for field_name in ("include_depth", "fixed_ro_local_range", "fixed_scale_range", "fixed_quat_range", "fixed_color_range", "fixed_opacity_range"):
        key = cached_raster_grad_key(field_name)
        if key in renderer:
            values[key] = renderer[key]


def apply_config_overlay(values: dict[str, object], overlay: dict[str, Any]) -> dict[str, object]:
    training = _section(overlay, "training_build_args")
    for build_arg, value in training.items():
        control_key = TRAINING_BUILD_ARG_UI_KEYS.get(str(build_arg))
        if control_key is not None:
            values[control_key] = tuple(value) if isinstance(value, list) else value

    _apply_renderer_overlay(values, _section(overlay, "renderer"))
    _set_many(values, _section(overlay, "viewer", "controls"))
    _set_many(values, _section(overlay, "viewer", "import"))
    _set_many(values, _section(overlay, "viewer", "ui"), key_map=_VIEWER_UI_KEY_MAP)

    run = _section(overlay, "viewer", "run")
    for key in RUN_UI_VALUE_FIELDS:
        if key in run:
            _set_run_value(values, key, run[key])
    return values


def exported_run_values(values: dict[str, object], *, existing_run: dict[str, Any] | None = None) -> dict[str, Any]:
    run: dict[str, Any] = {}
    if existing_run is not None:
        for preserved_key in ("schedule", "stats", "photometric", "metrics", "render", "edit"):
            if preserved_key in existing_run:
                run[preserved_key] = existing_run[preserved_key]
    for key in RUN_UI_VALUE_FIELDS:
        value = values.get(key)
        if key in _PATH_VALUE_FIELDS:
            run[key] = None if value is None or str(value).strip() == "" else str(value)
        elif key == "colmap_selected_camera_ids":
            run[key] = [int(camera_id) for camera_id in _as_tuple(value)]
        elif key in {"seed", "train_iters"}:
            run[key] = int(0 if value is None else value)
        else:
            run[key] = value
    return run
