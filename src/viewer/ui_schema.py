from __future__ import annotations

from dataclasses import dataclass

from ..repo_defaults import viewer_defaults
from ..app.training_controls import (
    SCHEDULE_STAGE_CONTROL_DEFS,
    TRAINING_OPTIMIZER_GROUP,
    TRAINING_OPTIMIZER_TAB_KEYS,
    TRAINING_SETUP_GROUP,
    TRAINING_STABILITY_GROUP,
    TRAINING_UI_GROUP_DEFS,
)
from ..renderer.render_params import CachedRasterGradParams, build_debug_render_control_specs, build_renderer_control_specs, renderer_param_tooltips
from ..training.ppisp import PPISP_FIELD_SPECS, ppisp_viewer_defaults, ppisp_viewer_export_fields
from .state import LOSS_DEBUG_OPTIONS

_LOSS_DEBUG_ABS_SCALE_KEY = "loss_debug_abs_scale"
_INTERFACE_SCALE_KEY = "interface_scale"
_THEME_KEY = "theme"
_GRAPHICS_API_KEY = "graphics_api"
_INTERFACE_SCALE_OPTIONS = (
    ("75%", 0.75),
    ("100%", 1.0),
    ("125%", 1.25),
    ("150%", 1.5),
    ("175%", 1.75),
    ("200%", 2.0),
    ("225%", 2.25),
    ("250%", 2.5),
    ("275%", 2.75),
    ("300%", 3.0),
)
_VIEWER_BACKGROUND_MODE_LABELS = ("Train Background", "Custom")
PPISP_DEBUG_MODE = "ppisp_tonemap"
_DEBUG_MODE_VALUES = (
    "normal",
    PPISP_DEBUG_MODE,
    "processed_count",
    "splat_age",
    "ellipse_outlines",
    "splat_density",
    "splat_spatial_density",
    "splat_screen_density",
    "contribution_amount",
    "current_frame_splat_contribution",
    "viewed_fraction_ema",
    "adam_momentum",
    "adam_second_moment",
    "grad_variance",
    "refinement_distribution",
    "depth_mean",
    "depth_std",
    "depth_local_mismatch",
    "grad_norm",
    "sh_view_dependent",
    "sh_coefficient",
    "black_negative",
)
_DEBUG_MODE_LABELS = (
    "Normal",
    "PPISP Tonemap",
    "Processed Count",
    "Splat Age",
    "Ellipse Outlines",
    "Splat Density",
    "Spatial Density",
    "Screen Density",
    "Contribution Amount",
    "Current Frame Splat Contribution",
    "Viewed Fraction EMA",
    "Adam Momentum",
    "Adam Second Moment",
    "Grad Variance",
    "Refinement Distribution",
    "Depth Mean",
    "Depth Std",
    "Depth Local Mismatch",
    "Grad Norm",
    "SH View-Dependent",
    "SH Coefficient",
    "Black/Negative Regions",
)
_RENDERER_DEBUG_MODE_VALUES = tuple("normal" if value == PPISP_DEBUG_MODE else value for value in _DEBUG_MODE_VALUES)
_DEBUG_SH_COEFF_LABELS = ("SH0 DC", "SH1 X", "SH1 Y", "SH1 Z", "SH2 0", "SH2 1", "SH2 2", "SH2 3", "SH2 4", "SH3 0", "SH3 1", "SH3 2", "SH3 3", "SH3 4", "SH3 5", "SH3 6")
_VIEWER_DEFAULTS = viewer_defaults()
_VIEWER_CONTROL_DEFAULTS = _VIEWER_DEFAULTS["controls"]
_VIEWER_IMPORT_DEFAULTS = _VIEWER_DEFAULTS["import"]
_VIEWER_UI_DEFAULTS = _VIEWER_DEFAULTS["ui"]
_PPISP_DEFAULTS = {**ppisp_viewer_defaults(), **{key: _VIEWER_CONTROL_DEFAULTS[key] for key in ppisp_viewer_defaults() if key in _VIEWER_CONTROL_DEFAULTS}}
_DEFAULT_INTERFACE_SCALE_INDEX = int(_VIEWER_CONTROL_DEFAULTS["interface_scale"])
_HISTOGRAM_BIN_COUNT_DEFAULT = int(_VIEWER_UI_DEFAULTS["hist_bin_count"])
_HISTOGRAM_MIN_VALUE_DEFAULT = float(_VIEWER_UI_DEFAULTS["hist_min_value"])
_HISTOGRAM_MAX_VALUE_DEFAULT = float(_VIEWER_UI_DEFAULTS["hist_max_value"])
_HISTOGRAM_Y_LIMIT_DEFAULT = float(_VIEWER_UI_DEFAULTS["hist_y_limit"])


def _renderer_atomic_mode_index(value: object) -> int:
    return 0 if str(value) == "float" else 1


def _renderer_debug_mode_index(value: object) -> int:
    mode = "normal" if value is None else str(value)
    return _DEBUG_MODE_VALUES.index(mode) if mode in _DEBUG_MODE_VALUES else 0


@dataclass(frozen=True, slots=True)
class ControlSpec:
    key: str
    kind: str
    label: str
    kwargs: dict[str, object]
    setup_visibility: str | None = None


def _control_spec(defn) -> ControlSpec:
    return ControlSpec(defn.key, defn.kind, defn.label, dict(defn.kwargs), getattr(defn, "setup_visibility", None))


_TRAIN_SETUP_SPECS = tuple(_control_spec(defn) for defn in TRAINING_UI_GROUP_DEFS[TRAINING_SETUP_GROUP])
_TRAIN_OPTIMIZER_SPECS = tuple(_control_spec(defn) for defn in TRAINING_UI_GROUP_DEFS[TRAINING_OPTIMIZER_GROUP])
SCHEDULE_STAGE_SPECS = {stage: tuple(_control_spec(defn) for defn in defs) for stage, defs in SCHEDULE_STAGE_CONTROL_DEFS.items()}
_TRAIN_STABILITY_SPECS = tuple(_control_spec(defn) for defn in TRAINING_UI_GROUP_DEFS[TRAINING_STABILITY_GROUP])

GROUP_SPECS = {
    "View": (
        ControlSpec(_INTERFACE_SCALE_KEY, "combo", "Interface Scale", {"value": int(_VIEWER_CONTROL_DEFAULTS["interface_scale"]), "options": tuple(label for label, _ in _INTERFACE_SCALE_OPTIONS)}),
    ),
    "Main": (
        ControlSpec("loss_debug_view", "slider_int", "Debug View", {"value": int(_VIEWER_CONTROL_DEFAULTS["loss_debug_view"]), "min": 0, "max": len(LOSS_DEBUG_OPTIONS) - 1}),
        ControlSpec("loss_debug_frame", "slider_int", "Debug Frame", {"value": int(_VIEWER_CONTROL_DEFAULTS["loss_debug_frame"]), "min": 0, "max": 10000}),
        ControlSpec(_LOSS_DEBUG_ABS_SCALE_KEY, "slider_float", "Abs Diff Scale", {"value": float(_VIEWER_CONTROL_DEFAULTS["loss_debug_abs_scale"]), "min": 0.125, "max": 64.0, "format": "%.3gx", "logarithmic": True}),
    ),
    "Camera": (
        ControlSpec("move_speed", "input_float", "Move Speed", {"value": float(_VIEWER_CONTROL_DEFAULTS["move_speed"]), "step": 0.1, "step_fast": 1.0, "format": "%.6g"}),
        ControlSpec("fov", "slider_float", "FOV", {"value": float(_VIEWER_CONTROL_DEFAULTS["fov"]), "min": 25.0, "max": 100.0}),
        ControlSpec("render_background_mode", "combo", "Render Background", {"value": int(_VIEWER_CONTROL_DEFAULTS["render_background_mode"]), "options": _VIEWER_BACKGROUND_MODE_LABELS}),
        ControlSpec("render_background_color", "color_edit3", "Render BG Color", {"value": tuple(float(v) for v in _VIEWER_CONTROL_DEFAULTS["render_background_color"])}),
    ),
    "PPISP Debug": (
        *(ControlSpec(spec.key, f"input_float{spec.size}" if spec.size > 1 else "input_float", spec.label, {"value": _PPISP_DEFAULTS[spec.key], "step": spec.step, "step_fast": spec.step_fast, "format": spec.fmt}) for spec in PPISP_FIELD_SPECS),
    ),
    "Train Setup": _TRAIN_SETUP_SPECS,
    "Train Optimizer": _TRAIN_OPTIMIZER_SPECS,
    "Train Stability": _TRAIN_STABILITY_SPECS,
}


def default_control_values(*group_names: str) -> dict[str, object]:
    groups = GROUP_SPECS.values() if not group_names else (GROUP_SPECS[name] for name in group_names)
    return {spec.key: spec.kwargs["value"] for specs in groups for spec in specs if "value" in spec.kwargs}


def build_render_spec_bundle(threshold_from_band_range):
    render_param_specs = build_renderer_control_specs(ControlSpec, _renderer_atomic_mode_index)
    debug_render_specs = build_debug_render_control_specs(ControlSpec, _renderer_debug_mode_index, _DEBUG_MODE_LABELS, _DEBUG_SH_COEFF_LABELS, threshold_from_band_range)
    all_defaults = {spec.key: spec.kwargs["value"] for group in GROUP_SPECS.values() for spec in group if "value" in spec.kwargs}
    all_defaults.update({spec.key: spec.kwargs["value"] for spec in render_param_specs if "value" in spec.kwargs})
    all_defaults.update({spec.key: spec.kwargs["value"] for spec in debug_render_specs if "value" in spec.kwargs})
    return render_param_specs, debug_render_specs, all_defaults


_VIEWER_CONTROL_EXPORT_FIELDS = (
    (_INTERFACE_SCALE_KEY, int),
    (_THEME_KEY, int),
    ("loss_debug_view", int),
    ("loss_debug_frame", int),
    (_LOSS_DEBUG_ABS_SCALE_KEY, float),
    ("move_speed", float),
    ("fov", float),
    ("render_background_mode", int),
    ("render_background_color", tuple),
    *ppisp_viewer_export_fields(),
    ("training_steps_per_frame", int),
    ("training_dataset_pool_size", int),
    ("train_background_color", tuple),
    ("seed", int),
    ("init_opacity", float),
)
_VIEWER_IMPORT_EXPORT_FIELDS = (
    ("colmap_depth_value_mode", int),
    ("colmap_init_mode", int),
    ("colmap_rotation_mode", int),
    ("colmap_custom_rotation_deg", tuple),
    ("compress_dataset_using_bc7", bool),
    ("colmap_training_image_color_init", bool),
    ("colmap_photometric_compensation_enabled", bool),
    ("colmap_use_alpha_masks", bool),
    ("colmap_image_downscale_mode", int),
    ("colmap_image_max_size", int),
    ("colmap_image_scale", float),
    ("colmap_nn_radius_scale_coef", float),
    ("colmap_min_track_length", int),
    ("colmap_init_neighbor_count", int),
    ("colmap_init_anisotropy_strength", float),
    ("colmap_depth_point_count", int),
    ("colmap_pointcloud_enabled", bool),
    ("colmap_pointcloud_nn_radius_scale_coef", float),
    ("colmap_diffused_enabled", bool),
    ("colmap_diffused_point_count", int),
    ("colmap_diffused_diffusion_radius", float),
    ("colmap_diffused_nn_radius_scale_coef", float),
    ("colmap_custom_ply_enabled", bool),
    ("colmap_custom_ply_nn_radius_scale_coef", float),
    ("colmap_custom_mesh_enabled", bool),
    ("colmap_custom_mesh_point_count", int),
    ("colmap_custom_mesh_nn_radius_scale_coef", float),
    ("target_alpha_mode", int),
    ("target_alpha_threshold", float),
    ("colmap_fibonacci_sphere_enabled", bool),
    ("colmap_fibonacci_sphere_point_count", int),
    ("colmap_fibonacci_sphere_radius_multiplier", float),
    ("colmap_fibonacci_sphere_color", tuple),
    ("colmap_fibonacci_sphere_upper_hemisphere_only", bool),
    ("colmap_fibonacci_sphere_nn_radius_scale_coef", float),
)
_VIEWER_UI_EXPORT_FIELDS = (
    (_GRAPHICS_API_KEY, str),
    ("show_histograms", bool),
    ("show_training_metrics", bool),
    ("show_training_views", bool),
    ("show_camera_overlays", bool),
    ("show_camera_labels", bool),
    ("show_training_cameras", bool),
    ("show_camera_min_dist_spheres", bool),
    ("hist_bin_count", int),
    ("hist_min_value", float),
    ("hist_max_value", float),
    ("hist_y_limit", float),
    ("photometric_batch_pair_count", int),
    ("photometric_neighborhood_size", int),
    ("photometric_min_track_length", int),
    ("photometric_learning_rate", float),
    ("photometric_target_average_exposure", float),
    ("photometric_enable_exposure", bool),
    ("photometric_enable_color", bool),
    ("photometric_enable_vignette", bool),
    ("photometric_enable_gamma", bool),
    ("photometric_grad_component_clip", float),
    ("photometric_max_update", float),
    ("photometric_exposure_lr_mul", float),
    ("photometric_vignette_lr_mul", float),
    ("photometric_chroma_lr_mul", float),
    ("photometric_crf_lr_mul", float),
    ("photometric_exposure_regularize_weight", float),
    ("photometric_vignette_regularize_weight", float),
    ("photometric_chroma_regularize_weight", float),
    ("photometric_crf_regularize_weight", float),
    ("photometric_gamma_regularize_weight", float),
    ("photometric_exposure_l1_weight", float),
    ("photometric_vignette_l1_weight", float),
    ("photometric_chroma_l1_weight", float),
    ("photometric_crf_l1_weight", float),
    ("photometric_gamma_l1_weight", float),
    ("_viewport_sh_band", int),
    ("_viewport_sh_control_key", str),
    ("_viewport_sh_stage_label", str),
)
_TRAINING_RASTER_GRAD_KEYS = CachedRasterGradParams.control_keys()
_OPTIMIZER_TAB_KEYS = TRAINING_OPTIMIZER_TAB_KEYS
_TRAIN_OPTIMIZER_SPEC_BY_KEY = {spec.key: spec for spec in GROUP_SPECS[TRAINING_OPTIMIZER_GROUP]}

UI_TOOLTIPS = {
    "render_background_mode": "Choose whether the main renderer uses the training background color or a separate custom RGB background",
    "render_background_color": "Custom RGB background for the main renderer",
    **renderer_param_tooltips(),
    "lr_pos_mul": "Learning rate multiplier for position",
    "lr_sh_mul": "Learning rate multiplier for non-DC SH coefficients",
    "lr_scale_mul": "Learning rate multiplier for scale",
    "lr_rot_mul": "Learning rate multiplier for rotation",
    "lr_color_mul": "Base learning rate multiplier for the SH0/DC color term before non-DC SH multipliers are applied",
    "lr_opacity_mul": "Learning rate multiplier for opacity",
    "beta1": "Adam first moment decay (momentum)",
    "beta2": "Adam second moment decay (RMSprop)",
    "grad_clip": "Per-parameter gradient clipping threshold",
    "max_update": "Maximum per-step parameter update magnitude",
    "scale_l2": "L2 regularization on log-scale",
    "scale_abs_reg": "Absolute scale regularization weight",
    "sh1_reg": "L1 regularization weight applied to all non-DC SH coefficients",
    "opacity_reg": "Opacity regularization weight (pushes opacity downward toward 0, strongest near 0.5)",
    "position_push_away_from_camera_step": "Stage 0 per-step position push multiplier; when scheduling is disabled this value is used for the whole run",
    "ssim_weight": "Blend weight for DSSIM in the RGB image loss; 0 keeps pure L1 and 1 uses pure DSSIM",
    "ssim_c2": "SSIM contrast/structure stabilizer constant used by the DSSIM path",
    "position_random_step_noise_lr": "Stage 0 post-step MCMC-style position noise multiplier; when scheduling is disabled this value is used for the whole run",
    "position_random_step_opacity_gate_center": "Opacity center for the random-step sigmoid gate; lower-opacity splats get stronger position noise",
    "position_random_step_opacity_gate_sharpness": "Steepness of the random-step opacity gate",
    "max_anisotropy": "Maximum ratio between largest and smallest scale axes",
    "min_opacity": "Floor for opacity",
    "max_opacity": "Base opacity ceiling used when the LR schedule is disabled",
    "position_abs_max": "Absolute position bounding box (per axis)",
    "camera_min_dist": "Cull any splat whose center is closer to the camera than this distance, regardless of projected size",
    "max_gaussians": "Maximum number of gaussians in the scene",
    "training_steps_per_frame": "Number of training optimizer steps to run before each viewer redraw; higher improves training throughput but reduces UI refresh rate",
    "training_dataset_pool_size": "Number of GPU textures the dataset streams through; 0 keeps the whole dataset resident in VRAM, and values below Steps / Frame submit more often",
    "background_mode": "Choose whether training uses a fixed custom RGB background or a new seeded white-noise background each optimizer step",
    "train_background_color": "Custom RGB background used for training when Train Background is set to Custom",
    "sh_band": "Stage 0 SH band limit; SH0 uses only the DC term and SH3 enables the full coefficient set",
    "refinement_interval": "Run cull/split refinement every N training steps",
    "refinement_growth_start_step": "Keep target-driven refinement at its prune-floor / zero-clone state until this training iteration; slider range follows Schedule Steps",
    "refinement_target_splat_ratio": "Stage 0 target fraction of Max Gaussians for the post-refinement scene; for example 0.10 means 10% of Max Gaussians",
    "refinement_max_growth_per_step": "Cap clone-driven scene growth per refinement pass; 0.15 allows at most 15% survivor growth in one pass",
    "refinement_max_prune_per_step": "Cap extra target-driven pruning per refinement pass; the Prune Lowest Ratio floor still applies even when it is higher",
    "refinement_alpha_cull_threshold": "Cull splats below this decoded alpha threshold during refinement",
    "refinement_min_contribution": "Minimum accumulated color-change contribution required for a splat to survive refinement, expressed in dequantized 1/256 contribution units",
    "refinement_min_contribution_decay": "Multiply the minimum accumulated color-change contribution threshold by this factor after each completed refinement pass",
    "refinement_ema_pose_count_decay": "Retained fraction for the refinement contribution and viewed-fraction EMAs after one full pass over the training pose set; the per-step EMA decay is derived automatically from this value",
    "refinement_viewed_fraction_zero_threshold": "Zero the viewed-fraction refinement signal used by the refinement distribution when viewed-fraction EMA drops below this value divided by the training camera count",
    "refinement_prune_lowest_contribution_ratio": "Stage 0 lowest-contribution prune ratio; when scheduling is disabled this value is used for the whole run",
    "refinement_opacity_mul": "Multiply every surviving splat alpha by this factor during each refinement rewrite pass",
    "refinement_sample_radius": "Radius of the centered local-space Fibonacci volume used when spawning new refinement samples",
    "refinement_clone_scale_mul": "Multiply the split-family sigma after the default family-size shrink used for refinement clones",
    "refinement_use_compact_split": "Use the compact split shrink rule instead of the legacy N^(-1/3) refinement shrink",
    "refinement_solve_opacity": "Solve one shared child alpha per split family so composed center transmittance matches the parent",
    "refinement_split_beta": "Exponent used by compact split shrink: child sigma scales as N^(-beta)",
    "refinement_grad_variance_weight_exponent": "Exponent a in pow(pixel_grad_variance, a) for refinement clone resampling",
    "refinement_contribution_weight_exponent": "Exponent b in pow(viewed_fraction_ema, b) for refinement clone resampling",
    "refinement_contribution_area_exponent": "Divide each per-view refinement contribution by visible screen area in pixels raised to this exponent before averaging it into history",
    "refinement_contribution_view_count_exponent": "When combining nonzero per-view refinement contributions, divide the accumulated contribution sum by nonzero view count raised to this exponent",
    "density_regularizer": "Weight applied to the per-pixel hinge penalty max(density - max_allowed_density, 0)",
    "sorting_order_dithering": "Stage 0 sort-camera dither amount; when scheduling is disabled this value is used for the whole run",
    "max_allowed_density": "End-of-training per-pixel density threshold above which the density regularizer activates; runtime ramps from 5.0 to this value over the LR schedule",
    "lr_schedule_enabled": "Enable the piecewise-linear base learning-rate schedule",
    "lr_schedule_start_lr": "Stage 0 base learning rate; when scheduling is disabled this value is used for the whole run",
    "lr_schedule_stage1_step": "Training step where Stage 1 ends and the Stage 1 targets are reached; slider range follows the Stage 4 end step",
    "lr_schedule_stage2_step": "Training step where Stage 2 ends and the Stage 2 targets are reached; slider range follows the Stage 4 end step",
    "lr_schedule_stage3_step": "Training step where Stage 3 ends and the Stage 3 targets are reached; slider range follows the Stage 4 end step",
    "lr_schedule_stage1_lr": "Base learning-rate target reached at the end of Stage 1",
    "lr_schedule_stage2_lr": "Base learning-rate target reached at the end of Stage 2",
    "lr_schedule_stage3_lr": "Base learning-rate target reached at the end of Stage 3",
    "lr_schedule_end_lr": "Base learning-rate target reached at the end of Stage 4",
    "lr_schedule_steps": "Stage 4 end step and total step budget shared by the LR, colorspace, noise, and SH schedules",
    "train_downscale_mode": "Use Auto for scheduled downscale descent or choose a fixed manual override from 1x to 16x",
    "train_auto_start_downscale": "Initial downscale factor used at step 0 when Downscale Mode is Auto",
    "train_downscale_base_iters": "Number of iterations spent at the auto start factor before descending",
    "train_downscale_iter_step": "Additional iterations added to each lower auto downscale phase",
    "train_downscale_max_iters": "Displayed training schedule budget for the auto downscale progression; training does not stop automatically",
    _LOSS_DEBUG_ABS_SCALE_KEY: "Multiplier applied to absolute RGB difference before presenting the debug texture",
    "seed": "Random seed for training frame shuffle order",
    "init_opacity": "Initial opacity for new gaussians",
    **{f"lr_pos_stage{stage}_mul": f"Position learning-rate multiplier target reached at the end of Stage {stage}" for stage in (1, 2, 3, 4)},
    **{f"lr_scale_stage{stage}_mul": f"Scale learning-rate multiplier target reached at the end of Stage {stage}" for stage in (1, 2, 3, 4)},
    **{f"lr_rot_stage{stage}_mul": f"Rotation learning-rate multiplier target reached at the end of Stage {stage}" for stage in (1, 2, 3, 4)},
    **{f"lr_color_stage{stage}_mul": f"Color learning-rate multiplier target reached at the end of Stage {stage}" for stage in (1, 2, 3, 4)},
    **{f"lr_opacity_stage{stage}_mul": f"Opacity learning-rate multiplier target reached at the end of Stage {stage}" for stage in (1, 2, 3, 4)},
    **{f"lr_sh_stage{stage}_mul": f"Non-DC SH learning-rate multiplier target reached at the end of Stage {stage}" for stage in (1, 2, 3, 4)},
    "max_opacity_stage0": "Stage 0 opacity ceiling; when LR scheduling is enabled this alpha limit is converted into the internal raw/logit opacity clamp",
    **{f"max_opacity_stage{stage}": f"Opacity ceiling target reached at the end of Stage {stage}; the exposed alpha limit is converted into the internal raw/logit opacity clamp" for stage in (1, 2, 3, 4)},
    **{f"position_push_away_from_camera_step_stage{stage}": f"Camera-push step target reached at the end of Stage {stage}" for stage in (1, 2, 3, 4)},
    **{f"refinement_prune_lowest_contribution_ratio_stage{stage}": f"Lowest-contribution prune ratio target reached at the end of Stage {stage}" for stage in (1, 2, 3, 4)},
    **{f"sorting_order_dithering_stage{stage}": f"Sort-camera dither target reached at the end of Stage {stage}" for stage in (1, 2, 3, 4)},
    **{f"position_random_step_noise_stage{stage}_lr": f"Position-noise LR target reached at the end of Stage {stage}" for stage in (1, 2, 3, 4)},
    **{f"sh_band_stage{stage}": f"SH band limit reached by the end of Stage {stage}" for stage in (1, 2, 3, 4)},
}
