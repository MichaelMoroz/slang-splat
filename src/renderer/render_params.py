from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..repo_defaults import cli_defaults, renderer_defaults


_RENDERER_DEFAULTS = renderer_defaults()
_CLI_COMMON_RENDER_DEFAULTS = cli_defaults()["common_render"]
CACHED_RASTER_GRAD_ATOMIC_MODE_FLOAT = "float"
CACHED_RASTER_GRAD_ATOMIC_MODE_FIXED = "fixed"
CACHED_RASTER_GRAD_ATOMIC_MODE_VALUES = (
    CACHED_RASTER_GRAD_ATOMIC_MODE_FLOAT,
    CACHED_RASTER_GRAD_ATOMIC_MODE_FIXED,
)
CACHED_RASTER_GRAD_ATOMIC_MODE_LABELS = ("Float Atomics", "Fixed Point")


@dataclass(frozen=True, slots=True)
class ControlDef:
    key: str
    kind: str
    label: str
    kwargs: dict[str, object]
    tooltip: str
    cli_flags: tuple[str, ...] | None = None
    cli_kwargs: dict[str, object] | None = None


def cached_raster_grad_key(field_name: str) -> str:
    return f"cached_raster_grad_{field_name}"


def _cached_raster_grad_field_name(key: str) -> str:
    return key.removeprefix("cached_raster_grad_")


def _coerce_renderer_value(value: object, default: object) -> object:
    if isinstance(default, tuple):
        return tuple(float(v) for v in value)
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return None if value is None else str(value)


def _serialize_renderer_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(float(v) for v in value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value)


def _control_def(
    key: str,
    kind: str,
    label: str,
    tooltip: str,
    *,
    value: object | None = None,
    cli_flags: tuple[str, ...] | None = None,
    cli_kwargs: dict[str, object] | None = None,
    **kwargs: object,
) -> RendererControlDef:
    control_kwargs = dict(kwargs)
    if value is not None:
        control_kwargs["value"] = value
    return ControlDef(key, kind, label, control_kwargs, tooltip, cli_flags=cli_flags, cli_kwargs=cli_kwargs)


def _range_control_defs(
    key_prefix: str,
    label_prefix: str,
    default_range: tuple[float, float],
    tooltip_suffix: str,
    *,
    step: float,
    step_fast: float,
    display_format: str,
) -> tuple[ControlDef, ControlDef]:
    common_kwargs = {"step": step, "step_fast": step_fast, "format": display_format}
    return (
        _control_def(f"{key_prefix}_min", "input_float", f"{label_prefix} Min", f"Minimum {tooltip_suffix}", value=float(default_range[0]), **common_kwargs),
        _control_def(f"{key_prefix}_max", "input_float", f"{label_prefix} Max", f"Maximum {tooltip_suffix}", value=float(default_range[1]), **common_kwargs),
    )


_RENDERER_UI_FIELD_KEYS = (
    ("radius_scale", "radius_scale"),
    ("alpha_cutoff", "alpha_cutoff"),
    ("max_anisotropy", "max_anisotropy"),
    ("transmittance_threshold", "trans_threshold"),
    ("debug_grad_norm_threshold", "debug_grad_norm_threshold"),
    ("debug_ellipse_thickness_px", "debug_ellipse_thickness_px"),
    ("debug_gaussian_scale_multiplier", "debug_gaussian_scale_multiplier"),
    ("debug_min_opacity", "debug_min_opacity"),
    ("debug_opacity_multiplier", "debug_opacity_multiplier"),
    ("debug_ellipse_scale_multiplier", "debug_ellipse_scale_multiplier"),
    ("debug_depth_local_mismatch_smooth_radius", "debug_depth_local_mismatch_smooth_radius"),
    ("debug_depth_local_mismatch_reject_radius", "debug_depth_local_mismatch_reject_radius"),
    ("debug_sh_coeff_index", "debug_sh_coeff_index"),
)
_RENDERER_UI_RANGE_FIELDS = (
    ("debug_splat_age_range", "debug_splat_age_min", "debug_splat_age_max"),
    ("debug_density_range", "debug_density_min", "debug_density_max"),
    ("debug_contribution_range", "debug_contribution_min", "debug_contribution_max"),
    ("debug_refinement_distribution_range", "debug_refinement_distribution_min", "debug_refinement_distribution_max"),
    ("debug_depth_mean_range", "debug_depth_mean_min", "debug_depth_mean_max"),
    ("debug_depth_std_range", "debug_depth_std_min", "debug_depth_std_max"),
    ("debug_depth_local_mismatch_range", "debug_depth_local_mismatch_min", "debug_depth_local_mismatch_max"),
)
_RENDERER_ARG_FIELD_KEYS = (
    ("radius_scale", "radius_scale"),
    ("alpha_cutoff", "alpha_cutoff"),
    ("max_anisotropy", "max_anisotropy"),
    ("transmittance_threshold", "trans_threshold"),
    ("list_capacity_multiplier", "list_capacity_multiplier"),
    ("max_prepass_memory_mb", "prepass_memory_mb"),
)


@dataclass(frozen=True, slots=True)
class CachedRasterGradParams:
    atomic_mode: str = str(_RENDERER_DEFAULTS["cached_raster_grad_atomic_mode"])
    include_depth: bool = bool(_RENDERER_DEFAULTS.get("cached_raster_grad_include_depth", False))
    fixed_ro_local_range: float = float(_RENDERER_DEFAULTS["cached_raster_grad_fixed_ro_local_range"])
    fixed_scale_range: float = float(_RENDERER_DEFAULTS["cached_raster_grad_fixed_scale_range"])
    fixed_quat_range: float = float(_RENDERER_DEFAULTS.get("cached_raster_grad_fixed_quat_range", 0.01))
    fixed_color_range: float = float(_RENDERER_DEFAULTS["cached_raster_grad_fixed_color_range"])
    fixed_opacity_range: float = float(_RENDERER_DEFAULTS["cached_raster_grad_fixed_opacity_range"])

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(cls.__dataclass_fields__)

    @classmethod
    def control_keys(cls) -> tuple[str, ...]:
        return tuple(cached_raster_grad_key(name) for name in cls.field_names())

    @classmethod
    def from_renderer(cls, renderer: object) -> CachedRasterGradParams:
        defaults = cls()
        values = {
            field_name: getattr(renderer, cached_raster_grad_key(field_name), getattr(defaults, field_name))
            for field_name in cls.field_names()
        }
        return cls(**values)

    @classmethod
    def from_ui_values(cls, values: Mapping[str, object]) -> CachedRasterGradParams:
        defaults = cls()
        field_values: dict[str, object] = {}
        for field_name in cls.field_names():
            key = cached_raster_grad_key(field_name)
            raw_value = values.get(key, getattr(defaults, field_name))
            if field_name == "atomic_mode":
                mode_index = min(max(int(raw_value), 0), len(CACHED_RASTER_GRAD_ATOMIC_MODE_VALUES) - 1)
                field_values[field_name] = CACHED_RASTER_GRAD_ATOMIC_MODE_VALUES[mode_index]
            elif field_name == "include_depth":
                field_values[field_name] = bool(raw_value)
            else:
                field_values[field_name] = float(raw_value)
        return cls(**field_values)

    @classmethod
    def from_args(cls, args: object, defaults: Mapping[str, object] | None = None) -> CachedRasterGradParams:
        resolved_defaults = _CLI_COMMON_RENDER_DEFAULTS if defaults is None else defaults
        base_defaults = cls()
        field_values: dict[str, object] = {}
        for field_name in cls.field_names():
            key = cached_raster_grad_key(field_name)
            raw_value = getattr(args, key, resolved_defaults.get(key, getattr(base_defaults, field_name)))
            if field_name == "atomic_mode":
                field_values[field_name] = str(raw_value)
            elif field_name == "include_depth":
                field_values[field_name] = bool(raw_value)
            else:
                field_values[field_name] = float(raw_value)
        return cls(**field_values)

    def renderer_kwargs(self) -> dict[str, object]:
        return {cached_raster_grad_key(field_name): getattr(self, field_name) for field_name in self.field_names()}

    def apply_ui_values(self, values: dict[str, object], atomic_mode_index) -> None:
        for field_name in self.field_names():
            key = cached_raster_grad_key(field_name)
            field_value = getattr(self, field_name)
            values[key] = atomic_mode_index(field_value) if field_name == "atomic_mode" else field_value


@dataclass(frozen=True, slots=True)
class RendererParams:
    radius_scale: float = float(_RENDERER_DEFAULTS["radius_scale"])
    alpha_cutoff: float = float(_RENDERER_DEFAULTS["alpha_cutoff"])
    max_anisotropy: float = float(_RENDERER_DEFAULTS["max_anisotropy"])
    transmittance_threshold: float = float(_RENDERER_DEFAULTS["transmittance_threshold"])
    list_capacity_multiplier: int = int(_RENDERER_DEFAULTS["list_capacity_multiplier"])
    max_prepass_memory_mb: int = int(_RENDERER_DEFAULTS["max_prepass_memory_mb"])
    cached_raster_grad: CachedRasterGradParams = field(default_factory=CachedRasterGradParams)
    debug_mode: str | None = _RENDERER_DEFAULTS["debug_mode"]
    debug_grad_norm_threshold: float = float(_RENDERER_DEFAULTS["debug_grad_norm_threshold"])
    debug_ellipse_thickness_px: float = float(_RENDERER_DEFAULTS["debug_ellipse_thickness_px"])
    debug_gaussian_scale_multiplier: float = float(_RENDERER_DEFAULTS["debug_gaussian_scale_multiplier"])
    debug_min_opacity: float = float(_RENDERER_DEFAULTS["debug_min_opacity"])
    debug_opacity_multiplier: float = float(_RENDERER_DEFAULTS["debug_opacity_multiplier"])
    debug_ellipse_scale_multiplier: float = float(_RENDERER_DEFAULTS["debug_ellipse_scale_multiplier"])
    debug_splat_age_range: tuple[float, float] = tuple(float(v) for v in _RENDERER_DEFAULTS["debug_splat_age_range"])
    debug_density_range: tuple[float, float] = tuple(float(v) for v in _RENDERER_DEFAULTS["debug_density_range"])
    debug_contribution_range: tuple[float, float] = tuple(float(v) for v in _RENDERER_DEFAULTS["debug_contribution_range"])
    debug_refinement_distribution_range: tuple[float, float] = tuple(float(v) for v in _RENDERER_DEFAULTS["debug_refinement_distribution_range"])
    debug_adam_momentum_range: tuple[float, float] = tuple(float(v) for v in _RENDERER_DEFAULTS["debug_adam_momentum_range"])
    debug_depth_mean_range: tuple[float, float] = tuple(float(v) for v in _RENDERER_DEFAULTS["debug_depth_mean_range"])
    debug_depth_std_range: tuple[float, float] = tuple(float(v) for v in _RENDERER_DEFAULTS["debug_depth_std_range"])
    debug_depth_local_mismatch_range: tuple[float, float] = tuple(float(v) for v in _RENDERER_DEFAULTS["debug_depth_local_mismatch_range"])
    debug_depth_local_mismatch_smooth_radius: float = float(_RENDERER_DEFAULTS["debug_depth_local_mismatch_smooth_radius"])
    debug_depth_local_mismatch_reject_radius: float = float(_RENDERER_DEFAULTS["debug_depth_local_mismatch_reject_radius"])
    debug_sh_coeff_index: int = int(_RENDERER_DEFAULTS["debug_sh_coeff_index"])
    debug_show_ellipses: bool = bool(_RENDERER_DEFAULTS["debug_show_ellipses"])
    debug_show_processed_count: bool = bool(_RENDERER_DEFAULTS["debug_show_processed_count"])
    debug_show_grad_norm: bool = bool(_RENDERER_DEFAULTS["debug_show_grad_norm"])

    @classmethod
    def from_renderer(cls, renderer: object) -> RendererParams:
        defaults = cls()
        field_values = {
            field_name: _coerce_renderer_value(getattr(renderer, field_name, getattr(defaults, field_name)), getattr(defaults, field_name))
            for field_name in cls.__dataclass_fields__
            if field_name != "cached_raster_grad"
        }
        return cls(cached_raster_grad=CachedRasterGradParams.from_renderer(renderer), **field_values)

    @classmethod
    def from_ui_values(cls, values: Mapping[str, object], debug_mode_values: tuple[str, ...], threshold_band_range) -> RendererParams:
        defaults = cls()
        debug_mode_index = min(max(int(values.get("debug_mode", 0)), 0), len(debug_mode_values) - 1)
        resolved_debug_mode = debug_mode_values[debug_mode_index]
        adam_threshold = float(values.get("debug_adam_momentum_threshold", values.get("debug_grad_norm_threshold", defaults.debug_grad_norm_threshold)))
        field_values = {
            field_name: getattr(defaults, field_name)
            for field_name in cls.__dataclass_fields__
            if field_name != "cached_raster_grad"
        }
        for field_name, key in _RENDERER_UI_FIELD_KEYS:
            field_values[field_name] = _coerce_renderer_value(values.get(key, getattr(defaults, field_name)), getattr(defaults, field_name))
        for field_name, min_key, max_key in _RENDERER_UI_RANGE_FIELDS:
            default_min, default_max = getattr(defaults, field_name)
            field_values[field_name] = (float(values.get(min_key, default_min)), float(values.get(max_key, default_max)))
        field_values["list_capacity_multiplier"] = int(values.get("list_capacity_multiplier", defaults.list_capacity_multiplier))
        field_values["max_prepass_memory_mb"] = int(values.get("max_prepass_memory_mb", defaults.max_prepass_memory_mb))
        field_values["debug_mode"] = None if resolved_debug_mode == "normal" else resolved_debug_mode
        field_values["debug_adam_momentum_range"] = tuple(float(v) for v in threshold_band_range(adam_threshold))
        field_values["debug_show_ellipses"] = bool(values.get("debug_show_ellipses", defaults.debug_show_ellipses))
        field_values["debug_show_processed_count"] = bool(values.get("debug_show_processed_count", defaults.debug_show_processed_count))
        field_values["debug_show_grad_norm"] = bool(values.get("debug_show_grad_norm", defaults.debug_show_grad_norm))
        return cls(cached_raster_grad=CachedRasterGradParams.from_ui_values(values), **field_values)

    @classmethod
    def from_args(cls, args: object, defaults: Mapping[str, object] | None = None) -> RendererParams:
        resolved_defaults = _CLI_COMMON_RENDER_DEFAULTS if defaults is None else defaults
        base_defaults = cls()
        field_values = {
            field_name: getattr(base_defaults, field_name)
            for field_name in cls.__dataclass_fields__
            if field_name != "cached_raster_grad"
        }
        for field_name, key in _RENDERER_ARG_FIELD_KEYS:
            field_values[field_name] = _coerce_renderer_value(getattr(args, key, resolved_defaults.get(key, getattr(base_defaults, field_name))), getattr(base_defaults, field_name))
        return cls(cached_raster_grad=CachedRasterGradParams.from_args(args, resolved_defaults), **field_values)

    def renderer_kwargs(self) -> dict[str, object]:
        kwargs = {
            field_name: _serialize_renderer_value(getattr(self, field_name))
            for field_name in type(self).__dataclass_fields__
            if field_name != "cached_raster_grad"
        }
        kwargs.update(self.cached_raster_grad.renderer_kwargs())
        return kwargs

    def cli_common_render_defaults_dict(self) -> dict[str, object]:
        return {
            "prepass_memory_mb": int(self.max_prepass_memory_mb),
            "radius_scale": float(self.radius_scale),
            "alpha_cutoff": float(self.alpha_cutoff),
            "trans_threshold": float(self.transmittance_threshold),
            **self.cached_raster_grad.renderer_kwargs(),
            "debug_layers": False,
            "list_capacity_multiplier": int(self.list_capacity_multiplier),
        }

    def apply_ui_values(self, values: dict[str, object], atomic_mode_index, debug_mode_index, threshold_from_band_range) -> None:
        for field_name, key in _RENDERER_UI_FIELD_KEYS:
            values[key] = _serialize_renderer_value(getattr(self, field_name))
        for field_name, min_key, max_key in _RENDERER_UI_RANGE_FIELDS:
            value_min, value_max = getattr(self, field_name)
            values[min_key] = float(value_min)
            values[max_key] = float(value_max)
        self.cached_raster_grad.apply_ui_values(values, atomic_mode_index)
        values["debug_mode"] = debug_mode_index(self.debug_mode)
        values["debug_adam_momentum_threshold"] = threshold_from_band_range(float(self.debug_adam_momentum_range[0]), float(self.debug_adam_momentum_range[1]), 1e-2)


def runtime_renderer_params() -> RendererParams:
    return RendererParams(
        debug_refinement_distribution_range=(0.0, 1.0),
        debug_adam_momentum_range=(0.0, 0.1),
    )


_CACHED_RASTER_GRAD_CONTROL_DEFS = (
    ControlDef(
        cached_raster_grad_key("atomic_mode"),
        "combo",
        "Cached Grad Atomics",
        {"options": CACHED_RASTER_GRAD_ATOMIC_MODE_LABELS},
        "Choose float atomics or fixed-point atomics for cached raster gradient accumulation during raster backward",
        cli_flags=("--cached-raster-grad-atomic-mode",),
        cli_kwargs={"type": str, "choices": CACHED_RASTER_GRAD_ATOMIC_MODE_VALUES},
    ),
    ControlDef(
        cached_raster_grad_key("include_depth"),
        "checkbox",
        "Cached Grad Depth",
        {},
        "Include the camera-depth cached gradient channels in raster-backward atomics and gradient stats",
    ),
    ControlDef(
        cached_raster_grad_key("fixed_ro_local_range"),
        "slider_float",
        "Cached Grad Local Range",
        {"min": 1e-4, "max": 1024.0, "format": "%.4g", "logarithmic": True},
        "Symmetric [-X, X] range for avgInvScale-normalized cached local-origin gradients",
        cli_flags=("--cached-raster-grad-fixed-ro-local-range",),
        cli_kwargs={"type": float},
    ),
    ControlDef(
        cached_raster_grad_key("fixed_scale_range"),
        "slider_float",
        "Cached Grad Scale Range",
        {"min": 1e-4, "max": 4096.0, "format": "%.4g", "logarithmic": True},
        "Symmetric [-X, X] range for avgInvScale-normalized cached scale gradients",
        cli_flags=("--cached-raster-grad-fixed-scale-range",),
        cli_kwargs={"type": float},
    ),
    ControlDef(
        cached_raster_grad_key("fixed_quat_range"),
        "slider_float",
        "Cached Grad OffDiag Range",
        {"min": 1e-6, "max": 16.0, "format": "%.4g", "logarithmic": True},
        "Symmetric [-X, X] range for avgInvScale-normalized cached off-diagonal sigma gradients",
        cli_flags=("--cached-raster-grad-fixed-quat-range",),
        cli_kwargs={"type": float},
    ),
    ControlDef(
        cached_raster_grad_key("fixed_color_range"),
        "slider_float",
        "Cached Grad Color Range",
        {"min": 1e-4, "max": 2048.0, "format": "%.4g", "logarithmic": True},
        "Symmetric [-X, X] range for cached color gradients",
        cli_flags=("--cached-raster-grad-fixed-color-range",),
        cli_kwargs={"type": float},
    ),
    ControlDef(
        cached_raster_grad_key("fixed_opacity_range"),
        "slider_float",
        "Cached Grad Opacity Range",
        {"min": 1e-4, "max": 2048.0, "format": "%.4g", "logarithmic": True},
        "Symmetric [-X, X] range for cached opacity gradients",
        cli_flags=("--cached-raster-grad-fixed-opacity-range",),
        cli_kwargs={"type": float},
    ),
)


def build_cached_raster_grad_control_specs(control_spec_factory, atomic_mode_index) -> tuple[object, ...]:
    defaults = CachedRasterGradParams()
    return tuple(
        control_spec_factory(
            defn.key,
            defn.kind,
            defn.label,
            dict(defn.kwargs, value=atomic_mode_index(getattr(defaults, _cached_raster_grad_field_name(defn.key)))) if defn.key == cached_raster_grad_key("atomic_mode") else dict(defn.kwargs, value=getattr(defaults, _cached_raster_grad_field_name(defn.key))),
        )
        for defn in _CACHED_RASTER_GRAD_CONTROL_DEFS
    )


def build_cached_raster_grad_cli_args(arg_factory) -> tuple[object, ...]:
    defaults = CachedRasterGradParams.from_args(object())
    default_values = defaults.renderer_kwargs()
    return tuple(
        arg_factory(*defn.cli_flags, **dict(defn.cli_kwargs or {}, default=default_values[defn.key]))
        for defn in _CACHED_RASTER_GRAD_CONTROL_DEFS
        if defn.cli_flags is not None
    )


_RENDER_CONTROL_DEFS = (
    _control_def("radius_scale", "slider_float", "Radius Scale", "Multiplier on top of true 3DGS gaussian size for rendering", value=float(_RENDERER_DEFAULTS["radius_scale"]), min=0.25, max=4.0, format="%.3g", cli_flags=("--radius-scale",), cli_kwargs={"type": float}),
    _control_def("alpha_cutoff", "slider_float", "Alpha Cutoff", "Minimum alpha threshold; splats below this are skipped", value=float(_RENDERER_DEFAULTS["alpha_cutoff"]), min=0.0001, max=0.1, format="%.2e", cli_flags=("--alpha-cutoff",), cli_kwargs={"type": float}),
    _control_def("trans_threshold", "slider_float", "Trans Threshold", "Transmittance threshold for early ray termination", value=float(_RENDERER_DEFAULTS["transmittance_threshold"]), min=0.001, max=0.2, format="%.2e", cli_flags=("--trans-threshold",), cli_kwargs={"type": float}),
)

_DEBUG_RENDER_CONTROL_DEFS = (
    _control_def("debug_mode", "combo", "Mode", "Select the renderer debug output mode"),
    _control_def("debug_sh_coeff_index", "combo", "SH Coefficient", "Select which raw SH coefficient float3 to display; zero is mapped to 0.5 gray in this debug view"),
    _control_def("debug_grad_norm_threshold", "input_float", "Grad Norm Threshold", "Reference threshold for gradient norm and gradient variance heatmaps", value=float(_RENDERER_DEFAULTS["debug_grad_norm_threshold"]), step=1e-5, step_fast=1e-4, format="%.6g"),
    _control_def("debug_ellipse_thickness_px", "slider_float", "Ellipse Thickness", "Thickness used by ellipse outline debug rendering", value=float(_RENDERER_DEFAULTS["debug_ellipse_thickness_px"]), min=0.25, max=8.0, format="%.2f px"),
    _control_def("debug_gaussian_scale_multiplier", "slider_float", "Gaussian Scale", "Ellipse debug raster-loop multiplier applied to the cached Gaussian scale", value=float(_RENDERER_DEFAULTS["debug_gaussian_scale_multiplier"]), min=0.05, max=16.0, format="%.3gx", logarithmic=True),
    _control_def("debug_min_opacity", "slider_float", "Min Opacity", "Ellipse debug raster-loop opacity floor applied after the opacity multiplier", value=float(_RENDERER_DEFAULTS["debug_min_opacity"]), min=0.0, max=1.0, format="%.4f"),
    _control_def("debug_opacity_multiplier", "slider_float", "Opacity Mul", "Ellipse debug raster-loop multiplier applied to cached splat opacity", value=float(_RENDERER_DEFAULTS["debug_opacity_multiplier"]), min=0.0, max=16.0, format="%.3gx"),
    _control_def("debug_ellipse_scale_multiplier", "slider_float", "Ellipse Scale", "Additional multiplier on top of the cached gaussian scale for ellipse debug rendering", value=float(_RENDERER_DEFAULTS["debug_ellipse_scale_multiplier"]), min=0.05, max=16.0, format="%.3gx", logarithmic=True),
    *_range_control_defs("debug_splat_age", "Splat Age", tuple(float(v) for v in _RENDERER_DEFAULTS["debug_splat_age_range"]), "normalized splat age displayed in the splat age debug mode", step=0.05, step_fast=0.25, display_format="%.5g"),
    *_range_control_defs("debug_density", "Density", tuple(float(v) for v in _RENDERER_DEFAULTS["debug_density_range"]), "density shown in the density debug modes", step=0.1, step_fast=1.0, display_format="%.5g"),
    *_range_control_defs("debug_contribution", "Contribution", tuple(float(v) for v in _RENDERER_DEFAULTS["debug_contribution_range"]), "contribution shown in contribution debug mode", step=1e-4, step_fast=1e-3, display_format="%.6g"),
    *_range_control_defs("debug_refinement_distribution", "Refine Dist", tuple(float(v) for v in _RENDERER_DEFAULTS["debug_refinement_distribution_range"]), "refinement distribution value shown in the refinement debug view", step=1e-4, step_fast=1e-3, display_format="%.6g"),
    _control_def("debug_adam_momentum_threshold", "input_float", "Adam Momentum Threshold", "Reference threshold used to derive the symmetric Adam momentum debug band", step=1e-5, step_fast=1e-4, format="%.6g"),
    *_range_control_defs("debug_depth_mean", "Depth Mean", tuple(float(v) for v in _RENDERER_DEFAULTS["debug_depth_mean_range"]), "depth mean shown in depth mean debug mode", step=0.1, step_fast=1.0, display_format="%.5g"),
    *_range_control_defs("debug_depth_std", "Depth Std", tuple(float(v) for v in _RENDERER_DEFAULTS["debug_depth_std_range"]), "depth standard deviation shown in depth std debug mode", step=0.01, step_fast=0.1, display_format="%.5g"),
    *_range_control_defs("debug_depth_local_mismatch", "Depth Local Mismatch", tuple(float(v) for v in _RENDERER_DEFAULTS["debug_depth_local_mismatch_range"]), "local depth mismatch shown in the depth local mismatch debug view", step=0.01, step_fast=0.1, display_format="%.5g"),
    _control_def("debug_depth_local_mismatch_smooth_radius", "input_float", "Depth Smooth Radius", "Neighborhood radius used to smooth the local depth mismatch estimate", value=float(_RENDERER_DEFAULTS["debug_depth_local_mismatch_smooth_radius"]), step=0.1, step_fast=1.0, format="%.5g"),
    _control_def("debug_depth_local_mismatch_reject_radius", "input_float", "Depth Reject Radius", "Neighborhood radius used to reject inconsistent local depth samples", value=float(_RENDERER_DEFAULTS["debug_depth_local_mismatch_reject_radius"]), step=0.1, step_fast=1.0, format="%.5g"),
)


def build_renderer_control_specs(control_spec_factory, atomic_mode_index) -> tuple[object, ...]:
    return tuple(control_spec_factory(defn.key, defn.kind, defn.label, dict(defn.kwargs)) for defn in _RENDER_CONTROL_DEFS) + build_cached_raster_grad_control_specs(control_spec_factory, atomic_mode_index)


def build_debug_render_control_specs(control_spec_factory, debug_mode_index, debug_mode_labels, debug_sh_coeff_labels, threshold_from_band_range) -> tuple[object, ...]:
    default_overrides = {
        "debug_mode": {"value": debug_mode_index(_RENDERER_DEFAULTS["debug_mode"]), "options": debug_mode_labels},
        "debug_sh_coeff_index": {"value": int(_RENDERER_DEFAULTS["debug_sh_coeff_index"]), "options": debug_sh_coeff_labels},
        "debug_adam_momentum_threshold": {"value": threshold_from_band_range(float(_RENDERER_DEFAULTS["debug_adam_momentum_range"][0]), float(_RENDERER_DEFAULTS["debug_adam_momentum_range"][1]), 1e-2)},
    }
    specs: list[object] = []
    for defn in _DEBUG_RENDER_CONTROL_DEFS:
        kwargs = dict(defn.kwargs)
        kwargs.update(default_overrides.get(defn.key, {}))
        specs.append(control_spec_factory(defn.key, defn.kind, defn.label, kwargs))
    return tuple(specs)


def renderer_param_tooltips() -> dict[str, str]:
    return {
        **{defn.key: defn.tooltip for defn in _RENDER_CONTROL_DEFS},
        **{defn.key: defn.tooltip for defn in _CACHED_RASTER_GRAD_CONTROL_DEFS},
        **{defn.key: defn.tooltip for defn in _DEBUG_RENDER_CONTROL_DEFS},
    }


def build_renderer_cli_args(arg_factory) -> tuple[object, ...]:
    args = [
        arg_factory("--prepass-memory-mb", type=int, default=int(_CLI_COMMON_RENDER_DEFAULTS["prepass_memory_mb"])),
        *[
            arg_factory(*defn.cli_flags, **dict(defn.cli_kwargs or {}, default=defn.kwargs["value"]))
            for defn in _RENDER_CONTROL_DEFS
            if defn.cli_flags is not None
        ],
        *build_cached_raster_grad_cli_args(arg_factory),
    ]
    return tuple(args)
