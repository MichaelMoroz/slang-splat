from __future__ import annotations

from dataclasses import dataclass

from ..repo_defaults import viewer_defaults
from ..training.alpha_modes import TARGET_ALPHA_MODE_LABELS
from ..training import TRAIN_SUBSAMPLE_MAX_FACTOR
from ..training.defaults import (
    DEFAULT_LR_SCHEDULE_STEPS,
    DEFAULT_MAX_OPACITY_STAGE0,
    DEFAULT_MAX_OPACITY_STAGE1,
    DEFAULT_MAX_OPACITY_STAGE2,
    DEFAULT_MAX_OPACITY_STAGE3,
    DEFAULT_MAX_OPACITY_STAGE4,
    DEFAULT_REFINEMENT_CLONE_SCALE_MUL,
    DEFAULT_REFINEMENT_MIN_CONTRIBUTION_DECAY,
    DEFAULT_REFINEMENT_MIN_CONTRIBUTION,
    TRAINING_BUILD_ARG_DEFAULTS,
)
_VIEWER_CONTROL_DEFAULTS = viewer_defaults()["controls"]

TRAIN_BACKGROUND_MODE_LABELS = ("Custom", "Random")
TRAIN_DOWNSCALE_MODE_LABELS = ("Auto",) + tuple(f"{i}x" for i in range(1, 17))
TRAIN_SUBSAMPLE_LABELS = ("Auto", "Off") + tuple(f"1/{i}" for i in range(2, TRAIN_SUBSAMPLE_MAX_FACTOR + 1))
SH_BAND_LABELS = ("SH0", "SH1", "SH2", "SH3")

TRAINING_SETUP_GROUP = "Train Setup"
TRAINING_OPTIMIZER_GROUP = "Train Optimizer"
TRAINING_STABILITY_GROUP = "Train Stability"


@dataclass(frozen=True, slots=True)
class TrainingControlDef:
    key: str
    kind: str
    label: str
    kwargs: dict[str, object]
    group: str
    build_args: tuple[str, ...] = ()
    optimizer_tab: str | None = None
    schedule_stage: str | None = None
    schedule_slot: str | None = None
    setup_visibility: str | None = None


def _default(name: str) -> object:
    if name == "max_opacity_stage0":
        return float(TRAINING_BUILD_ARG_DEFAULTS.get(name, DEFAULT_MAX_OPACITY_STAGE0))
    if name == "max_opacity_stage1":
        return float(TRAINING_BUILD_ARG_DEFAULTS.get(name, DEFAULT_MAX_OPACITY_STAGE1))
    if name == "max_opacity_stage2":
        return float(TRAINING_BUILD_ARG_DEFAULTS.get(name, DEFAULT_MAX_OPACITY_STAGE2))
    if name == "max_opacity_stage3":
        return float(TRAINING_BUILD_ARG_DEFAULTS.get(name, DEFAULT_MAX_OPACITY_STAGE3))
    if name == "max_opacity_stage4":
        return float(TRAINING_BUILD_ARG_DEFAULTS.get(name, DEFAULT_MAX_OPACITY_STAGE4))
    if name == "refinement_grad_variance_weight_exponent":
        return TRAINING_BUILD_ARG_DEFAULTS.get(name, 0.1)
    if name == "refinement_contribution_weight_exponent":
        return TRAINING_BUILD_ARG_DEFAULTS.get(name, 0.1)
    if name == "refinement_contribution_area_exponent":
        return TRAINING_BUILD_ARG_DEFAULTS.get(name, 0.5)
    if name == "refinement_contribution_view_count_exponent":
        return TRAINING_BUILD_ARG_DEFAULTS.get(name, 0.5)
    return DEFAULT_REFINEMENT_CLONE_SCALE_MUL if name == "refinement_clone_scale_mul" else TRAINING_BUILD_ARG_DEFAULTS[name]


def _stage_default(name: str, base_name: str) -> object:
    return TRAINING_BUILD_ARG_DEFAULTS.get(name, TRAINING_BUILD_ARG_DEFAULTS[base_name])


def _control(
    key: str,
    kind: str,
    label: str,
    kwargs: dict[str, object],
    *,
    group: str,
    build_args: tuple[str, ...] = (),
    optimizer_tab: str | None = None,
    schedule_stage: str | None = None,
    schedule_slot: str | None = None,
    setup_visibility: str | None = None,
) -> TrainingControlDef:
    return TrainingControlDef(
        key=key,
        kind=kind,
        label=label,
        kwargs=dict(kwargs),
        group=group,
        build_args=tuple(build_args),
        optimizer_tab=optimizer_tab,
        schedule_stage=schedule_stage,
        schedule_slot=schedule_slot,
        setup_visibility=setup_visibility,
    )


TRAIN_SETUP_CONTROL_DEFS = (
    _control("max_gaussians", "input_int", "Max Gaussians", {"value": _default("max_gaussians"), "step": 1000, "step_fast": 10000}, group=TRAINING_SETUP_GROUP, build_args=("max_gaussians",)),
    _control("training_steps_per_frame", "input_int", "Steps / Frame", {"value": int(_VIEWER_CONTROL_DEFAULTS["training_steps_per_frame"]), "step": 1, "step_fast": 2}, group=TRAINING_SETUP_GROUP),
    _control("training_dataset_pool_size", "input_int", "Dataset Pool Size", {"value": int(_VIEWER_CONTROL_DEFAULTS["training_dataset_pool_size"]), "step": 1, "step_fast": 4}, group=TRAINING_SETUP_GROUP),
    _control("background_mode", "combo", "Train Background", {"value": _default("background_mode"), "options": TRAIN_BACKGROUND_MODE_LABELS}, group=TRAINING_SETUP_GROUP, build_args=("background_mode",)),
    _control("target_alpha_mode", "combo", "Target Alpha", {"value": int(_default("target_alpha_mode")), "options": TARGET_ALPHA_MODE_LABELS}, group=TRAINING_SETUP_GROUP, build_args=("target_alpha_mode",)),
    _control("target_alpha_threshold", "slider_float", "Target Alpha Threshold", {"value": _default("target_alpha_threshold"), "min": 0.0, "max": 1.0, "format": "%.3f"}, group=TRAINING_SETUP_GROUP, build_args=("target_alpha_threshold",)),
    _control("max_sh_band", "combo", "Global SH Cap", {"value": 3, "options": SH_BAND_LABELS}, group=TRAINING_SETUP_GROUP, build_args=("max_sh_band",)),
    _control("train_background_color", "color_edit3", "Train BG Color", {"value": tuple(float(v) for v in _VIEWER_CONTROL_DEFAULTS["train_background_color"])}, group=TRAINING_SETUP_GROUP, setup_visibility="background_custom"),
    _control("refinement_interval", "input_int", "Refinement Interval", {"value": _default("refinement_interval"), "step": 10, "step_fast": 50}, group=TRAINING_SETUP_GROUP, build_args=("refinement_interval",)),
    _control("refinement_growth_start_step", "input_int", "Start Refinement After", {"value": _default("refinement_growth_start_step"), "step": 100, "step_fast": 1000}, group=TRAINING_SETUP_GROUP, build_args=("refinement_growth_start_step",)),
    _control("refinement_max_growth_per_step", "input_float", "Max Growth / Step", {"value": _default("refinement_max_growth_per_step"), "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_max_growth_per_step",)),
    _control("refinement_max_prune_per_step", "input_float", "Max Prune / Step", {"value": _default("refinement_max_prune_per_step"), "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_max_prune_per_step",)),
    _control("refinement_alpha_cull_threshold", "input_float", "Refinement Alpha Cull", {"value": _default("refinement_alpha_cull_threshold"), "step": 1e-5, "step_fast": 1e-4, "format": "%.6e"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_alpha_cull_threshold",)),
    _control("refinement_min_contribution", "input_float", "Refinement Min Color Contribution", {"value": DEFAULT_REFINEMENT_MIN_CONTRIBUTION, "step": 1.0, "step_fast": 16.0, "format": "%.6g"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_min_contribution",)),
    _control("refinement_min_contribution_decay", "input_float", "Refinement Min Color Contribution Decay", {"value": DEFAULT_REFINEMENT_MIN_CONTRIBUTION_DECAY, "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_min_contribution_decay",)),
    _control("refinement_ema_pose_count_decay", "input_float", "Refinement EMA / Pose Count", {"value": _default("refinement_ema_pose_count_decay"), "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_ema_pose_count_decay",)),
    _control("refinement_viewed_fraction_zero_threshold", "input_float", "Viewed Fraction Zero / Pose Count", {"value": _default("refinement_viewed_fraction_zero_threshold"), "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_viewed_fraction_zero_threshold",)),
    _control("refinement_opacity_mul", "input_float", "Refinement Alpha Mul", {"value": _default("refinement_opacity_mul"), "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_opacity_mul",)),
    _control("refinement_sample_radius", "input_float", "Refinement Sample Radius", {"value": _default("refinement_sample_radius"), "step": 1e-2, "step_fast": 1e-1, "format": "%.5f"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_sample_radius",)),
    _control("refinement_clone_scale_mul", "input_float", "Refinement Clone Scale Mul", {"value": _default("refinement_clone_scale_mul"), "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_clone_scale_mul",)),
    _control("refinement_use_compact_split", "checkbox", "Use Compact Split", {"value": _default("refinement_use_compact_split")}, group=TRAINING_SETUP_GROUP, build_args=("refinement_use_compact_split",)),
    _control("refinement_solve_opacity", "checkbox", "Solve Split Opacity", {"value": _default("refinement_solve_opacity")}, group=TRAINING_SETUP_GROUP, build_args=("refinement_solve_opacity",)),
    _control("refinement_split_beta", "input_float", "Refinement Split Beta", {"value": _default("refinement_split_beta"), "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_split_beta",)),
    _control("refinement_grad_variance_weight_exponent", "input_float", "Refinement Variance Exponent", {"value": _default("refinement_grad_variance_weight_exponent"), "step": 1e-2, "step_fast": 1e-1, "format": "%.5f"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_grad_variance_weight_exponent",)),
    _control("refinement_contribution_weight_exponent", "input_float", "Refinement Viewed Fraction Exponent", {"value": _default("refinement_contribution_weight_exponent"), "step": 1e-2, "step_fast": 1e-1, "format": "%.5f"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_contribution_weight_exponent",)),
    _control("refinement_contribution_area_exponent", "input_float", "Contribution Area Exponent", {"value": _default("refinement_contribution_area_exponent"), "step": 1e-2, "step_fast": 1e-1, "format": "%.5f"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_contribution_area_exponent",)),
    _control("refinement_contribution_view_count_exponent", "input_float", "Contribution View Count Exponent", {"value": _default("refinement_contribution_view_count_exponent"), "step": 1e-2, "step_fast": 1e-1, "format": "%.5f"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_contribution_view_count_exponent",)),
    _control("train_downscale_mode", "combo", "Downscale Mode", {"value": _default("train_downscale_mode"), "options": TRAIN_DOWNSCALE_MODE_LABELS}, group=TRAINING_SETUP_GROUP, build_args=("train_downscale_mode",)),
    _control("train_subsample_factor", "combo", "Subsampling", {"value": _default("train_subsample_factor"), "options": TRAIN_SUBSAMPLE_LABELS}, group=TRAINING_SETUP_GROUP, build_args=("train_subsample_factor",)),
    _control("train_auto_start_downscale", "input_int", "Auto Start Downscale", {"value": _default("train_auto_start_downscale"), "step": 1, "step_fast": 2}, group=TRAINING_SETUP_GROUP, build_args=("train_auto_start_downscale",), setup_visibility="downscale_auto"),
    _control("train_downscale_base_iters", "input_int", "Downscale Base Iters", {"value": _default("train_downscale_base_iters"), "step": 25, "step_fast": 100}, group=TRAINING_SETUP_GROUP, build_args=("train_downscale_base_iters",), setup_visibility="downscale_auto"),
    _control("train_downscale_iter_step", "input_int", "Downscale Iter Step", {"value": _default("train_downscale_iter_step"), "step": 10, "step_fast": 50}, group=TRAINING_SETUP_GROUP, build_args=("train_downscale_iter_step",), setup_visibility="downscale_auto"),
    _control("train_downscale_max_iters", "input_int", "Downscale Max Iters", {"value": _default("train_downscale_max_iters"), "step": 1000, "step_fast": 5000}, group=TRAINING_SETUP_GROUP, build_args=("train_downscale_max_iters",), setup_visibility="downscale_auto"),
    _control("seed", "input_int", "Shuffle Seed", {"value": int(_VIEWER_CONTROL_DEFAULTS["seed"]), "step": 1, "step_fast": 100}, group=TRAINING_SETUP_GROUP),
    _control("init_opacity", "input_float", "Init Opacity", {"value": float(_VIEWER_CONTROL_DEFAULTS["init_opacity"]), "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}, group=TRAINING_SETUP_GROUP),
)

TRAIN_OPTIMIZER_CONTROL_DEFS = (
    _control("lr_schedule_enabled", "checkbox", "Use LR Schedule", {"value": _default("lr_schedule_enabled")}, group=TRAINING_OPTIMIZER_GROUP, build_args=("lr_schedule_enabled",), optimizer_tab="Schedule"),
    _control("lr_scale_mul", "input_float", "LR Mul Scale", {"value": _default("lr_scale_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("lr_scale_mul",)),
    _control("lr_rot_mul", "input_float", "LR Mul Rotation", {"value": _default("lr_rot_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("lr_rot_mul",)),
    _control("lr_color_mul", "input_float", "LR Mul SH0/DC", {"value": _default("lr_color_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("lr_color_mul",)),
    _control("lr_opacity_mul", "input_float", "LR Mul Opacity", {"value": _default("lr_opacity_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("lr_opacity_mul",)),
    _control("beta1", "input_float", "Beta1", {"value": _default("beta1"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("beta1",), optimizer_tab="Adam"),
    _control("beta2", "input_float", "Beta2", {"value": _default("beta2"), "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("beta2",), optimizer_tab="Adam"),
    _control("scale_l2", "input_float", "Scale Log Reg", {"value": _default("scale_l2_weight"), "step": 1e-5, "step_fast": 1e-4, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("scale_l2_weight",), optimizer_tab="Regularization"),
    _control("scale_abs_reg", "input_float", "Scale Abs Reg", {"value": _default("scale_abs_reg_weight"), "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("scale_abs_reg_weight",), optimizer_tab="Regularization"),
    _control("sh1_reg", "input_float", "SH Rest Reg", {"value": _default("sh1_reg_weight"), "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("sh1_reg_weight",), optimizer_tab="Regularization"),
    _control("opacity_reg", "input_float", "Opacity Reg", {"value": _default("opacity_reg_weight"), "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("opacity_reg_weight",), optimizer_tab="Regularization"),
    _control("density_regularizer", "input_float", "Density Reg", {"value": _default("density_regularizer"), "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("density_regularizer",), optimizer_tab="Regularization"),
    _control("raster_grad_distance_power", "input_float", "Grad Dist Power", {"value": _default("raster_grad_distance_power"), "step": 1e-2, "step_fast": 1e-1, "format": "%.5f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("raster_grad_distance_power",), optimizer_tab="Regularization"),
    _control("raster_grad_distance_bias", "input_float", "Grad Dist Bias", {"value": _default("raster_grad_distance_bias"), "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("raster_grad_distance_bias",), optimizer_tab="Regularization"),
    _control("ssim_c2", "input_float", "SSIM C2", {"value": _default("ssim_c2"), "step": 1e-5, "step_fast": 1e-4, "format": "%.6e"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("ssim_c2",), optimizer_tab="Regularization"),
    _control("max_allowed_density", "input_float", "Max Density", {"value": _default("max_allowed_density"), "step": 1e-3, "step_fast": 1e-2, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("max_allowed_density",), optimizer_tab="Regularization"),
    _control("position_random_step_opacity_gate_center", "input_float", "Noise Gate Center", {"value": _default("position_random_step_opacity_gate_center"), "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("position_random_step_opacity_gate_center",), optimizer_tab="Regularization"),
    _control("position_random_step_opacity_gate_sharpness", "input_float", "Noise Gate Sharpness", {"value": _default("position_random_step_opacity_gate_sharpness"), "step": 1.0, "step_fast": 10.0, "format": "%.4g"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("position_random_step_opacity_gate_sharpness",), optimizer_tab="Regularization"),
    _control("max_anisotropy", "input_float", "Max Anisotropy", {"value": _default("max_anisotropy"), "step": 0.1, "step_fast": 0.5, "format": "%.6f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("max_anisotropy",), optimizer_tab="Regularization"),
    _control("grad_clip", "input_float", "Grad Clip", {"value": _default("grad_clip"), "step": 0.1, "step_fast": 1.0, "format": "%.4f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("grad_clip",), optimizer_tab="Regularization"),
    _control("max_update", "input_float", "Max Update", {"value": _default("max_update"), "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("max_update",), optimizer_tab="Regularization"),
)

_SCHEDULE_STAGE_TEMPLATE = {
    "end_step": {"kind": "input_int", "label": "End Step", "kwargs": {"step": 100, "step_fast": 1000}},
    "lr": {"kind": "input_float", "label": "LR Target"},
    "lr_pos_mul": {"kind": "input_float", "label": "LR Mul Position"},
    "lr_scale_mul": {"kind": "input_float", "label": "LR Mul Scale"},
    "lr_rot_mul": {"kind": "input_float", "label": "LR Mul Rotation"},
    "lr_color_mul": {"kind": "input_float", "label": "LR Mul SH0/DC"},
    "lr_opacity_mul": {"kind": "input_float", "label": "LR Mul Opacity"},
    "lr_sh_mul": {"kind": "input_float", "label": "LR Mul SH"},
    "colorspace_mod": {"kind": "input_float", "label": "Colorspace Mod"},
    "ssim_weight": {"kind": "input_float", "label": "DSSIM Weight"},
    "max_visible_angle_deg": {"kind": "input_float", "label": "Max Visible Angle"},
    "max_opacity": {"kind": "input_float", "label": "Max Opacity"},
    "min_pixel_clamp": {"kind": "input_float", "label": "Min Pixel Clamp"},
    "sort_dither": {"kind": "input_float", "label": "Sort Dither"},
    "splat_target": {"kind": "input_float", "label": "Target Max Splats"},
    "prune_lowest": {"kind": "input_float", "label": "Prune Lowest Ratio"},
    "opacity_reg": {"kind": "input_float", "label": "Opacity Reg"},
    "cam_push": {"kind": "input_float", "label": "Cam Push Step"},
    "noise_lr": {"kind": "input_float", "label": "Noise LR"},
    "sh_band": {"kind": "combo", "label": "SH Band", "kwargs": {"options": SH_BAND_LABELS}},
}

_SCHEDULE_STAGE_CONFIGS = {
    "Stage 0": (
        ("lr", "lr_schedule_start_lr", ("base_lr", "lr_schedule_start_lr"), {"value": _default("lr_schedule_start_lr"), "step": 1e-5, "step_fast": 1e-4, "format": "%.8f"}),
        ("lr_pos_mul", "lr_pos_mul", ("lr_pos_mul",), {"value": _default("lr_pos_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_scale_mul", "lr_scale_mul", ("lr_scale_mul",), {"value": _default("lr_scale_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_rot_mul", "lr_rot_mul", ("lr_rot_mul",), {"value": _default("lr_rot_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_color_mul", "lr_color_mul", ("lr_color_mul",), {"value": _default("lr_color_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_opacity_mul", "lr_opacity_mul", ("lr_opacity_mul",), {"value": _default("lr_opacity_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_sh_mul", "lr_sh_mul", ("lr_sh_mul",), {"value": _default("lr_sh_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("colorspace_mod", "colorspace_mod", ("colorspace_mod",), {"value": _default("colorspace_mod"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ("ssim_weight", "ssim_weight", ("ssim_weight",), {"value": _default("ssim_weight"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ("max_visible_angle_deg", "max_visible_angle_deg", ("max_visible_angle_deg",), {"value": _default("max_visible_angle_deg"), "step": 1e-2, "step_fast": 1e-1, "format": "%.6f"}),
        ("max_opacity", "max_opacity_stage0", ("max_opacity_stage0",), {"value": _default("max_opacity_stage0"), "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}),
        ("min_pixel_clamp", "refinement_min_screen_radius_px", ("refinement_min_screen_radius_px",), {"value": _default("refinement_min_screen_radius_px"), "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}),
        ("sort_dither", "sorting_order_dithering", ("sorting_order_dithering",), {"value": _default("sorting_order_dithering"), "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}),
        ("splat_target", "refinement_target_splat_ratio", ("refinement_target_splat_ratio",), {"value": _default("refinement_target_splat_ratio"), "step": 1e-2, "step_fast": 5e-2, "format": "%.4f"}),
        ("prune_lowest", "refinement_prune_lowest_contribution_ratio", ("refinement_prune_lowest_contribution_ratio",), {"value": _default("refinement_prune_lowest_contribution_ratio"), "step": 1e-3, "step_fast": 1e-2, "format": "%.4f"}),
        ("opacity_reg", "opacity_reg", ("opacity_reg_weight",), {"value": _default("opacity_reg_weight"), "step": 1e-4, "step_fast": 1e-3, "format": "%.6g"}),
        ("cam_push", "position_push_away_from_camera_step", ("position_push_away_from_camera_step",), {"value": _default("position_push_away_from_camera_step"), "step": 1e-5, "step_fast": 1e-4, "format": "%.6g"}),
        ("noise_lr", "position_random_step_noise_lr", ("position_random_step_noise_lr",), {"value": _default("position_random_step_noise_lr"), "step": 100.0, "step_fast": 1000.0, "format": "%.4g"}),
        ("sh_band", "sh_band", ("sh_band",), {"value": _default("sh_band"), "options": SH_BAND_LABELS}),
    ),
    "Stage 1": (
        ("end_step", "lr_schedule_stage1_step", ("lr_schedule_stage1_step",), {"value": _default("lr_schedule_stage1_step"), "step": 100, "step_fast": 1000}),
        ("lr", "lr_schedule_stage1_lr", ("lr_schedule_stage1_lr",), {"value": _default("lr_schedule_stage1_lr"), "step": 1e-6, "step_fast": 1e-5, "format": "%.8f"}),
        ("lr_pos_mul", "lr_pos_stage1_mul", ("lr_pos_stage1_mul",), {"value": _default("lr_pos_stage1_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_scale_mul", "lr_scale_stage1_mul", ("lr_scale_stage1_mul",), {"value": _stage_default("lr_scale_stage1_mul", "lr_scale_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_rot_mul", "lr_rot_stage1_mul", ("lr_rot_stage1_mul",), {"value": _stage_default("lr_rot_stage1_mul", "lr_rot_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_color_mul", "lr_color_stage1_mul", ("lr_color_stage1_mul",), {"value": _stage_default("lr_color_stage1_mul", "lr_color_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_opacity_mul", "lr_opacity_stage1_mul", ("lr_opacity_stage1_mul",), {"value": _stage_default("lr_opacity_stage1_mul", "lr_opacity_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_sh_mul", "lr_sh_stage1_mul", ("lr_sh_stage1_mul",), {"value": _default("lr_sh_stage1_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("colorspace_mod", "colorspace_mod_stage1", ("colorspace_mod_stage1",), {"value": _default("colorspace_mod_stage1"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ("ssim_weight", "ssim_weight_stage1", ("ssim_weight_stage1",), {"value": _default("ssim_weight_stage1"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ("max_visible_angle_deg", "max_visible_angle_deg_stage1", ("max_visible_angle_deg_stage1",), {"value": _default("max_visible_angle_deg_stage1"), "step": 1e-2, "step_fast": 1e-1, "format": "%.6f"}),
        ("max_opacity", "max_opacity_stage1", ("max_opacity_stage1",), {"value": _default("max_opacity_stage1"), "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}),
        ("min_pixel_clamp", "refinement_min_screen_radius_px_stage1", ("refinement_min_screen_radius_px_stage1",), {"value": _default("refinement_min_screen_radius_px_stage1"), "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}),
        ("sort_dither", "sorting_order_dithering_stage1", ("sorting_order_dithering_stage1",), {"value": _default("sorting_order_dithering_stage1"), "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}),
        ("splat_target", "refinement_target_splat_ratio_stage1", ("refinement_target_splat_ratio_stage1",), {"value": _default("refinement_target_splat_ratio_stage1"), "step": 1e-2, "step_fast": 5e-2, "format": "%.4f"}),
        ("prune_lowest", "refinement_prune_lowest_contribution_ratio_stage1", ("refinement_prune_lowest_contribution_ratio_stage1",), {"value": _default("refinement_prune_lowest_contribution_ratio_stage1"), "step": 1e-3, "step_fast": 1e-2, "format": "%.4f"}),
        ("opacity_reg", "opacity_reg_stage1", ("opacity_reg_weight_stage1",), {"value": _default("opacity_reg_weight_stage1"), "step": 1e-4, "step_fast": 1e-3, "format": "%.6g"}),
        ("cam_push", "position_push_away_from_camera_step_stage1", ("position_push_away_from_camera_step_stage1",), {"value": _default("position_push_away_from_camera_step_stage1"), "step": 1e-5, "step_fast": 1e-4, "format": "%.6g"}),
        ("noise_lr", "position_random_step_noise_stage1_lr", ("position_random_step_noise_stage1_lr",), {"value": _default("position_random_step_noise_stage1_lr"), "step": 100.0, "step_fast": 1000.0, "format": "%.4g"}),
        ("sh_band", "sh_band_stage1", ("sh_band_stage1",), {"value": _default("sh_band_stage1"), "options": SH_BAND_LABELS}),
    ),
    "Stage 2": (
        ("end_step", "lr_schedule_stage2_step", ("lr_schedule_stage2_step",), {"value": _default("lr_schedule_stage2_step"), "step": 100, "step_fast": 1000}),
        ("lr", "lr_schedule_stage2_lr", ("lr_schedule_stage2_lr",), {"value": _default("lr_schedule_stage2_lr"), "step": 1e-6, "step_fast": 1e-5, "format": "%.8f"}),
        ("lr_pos_mul", "lr_pos_stage2_mul", ("lr_pos_stage2_mul",), {"value": _default("lr_pos_stage2_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_scale_mul", "lr_scale_stage2_mul", ("lr_scale_stage2_mul",), {"value": _stage_default("lr_scale_stage2_mul", "lr_scale_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_rot_mul", "lr_rot_stage2_mul", ("lr_rot_stage2_mul",), {"value": _stage_default("lr_rot_stage2_mul", "lr_rot_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_color_mul", "lr_color_stage2_mul", ("lr_color_stage2_mul",), {"value": _stage_default("lr_color_stage2_mul", "lr_color_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_opacity_mul", "lr_opacity_stage2_mul", ("lr_opacity_stage2_mul",), {"value": _stage_default("lr_opacity_stage2_mul", "lr_opacity_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_sh_mul", "lr_sh_stage2_mul", ("lr_sh_stage2_mul",), {"value": _default("lr_sh_stage2_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("colorspace_mod", "colorspace_mod_stage2", ("colorspace_mod_stage2",), {"value": _default("colorspace_mod_stage2"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ("ssim_weight", "ssim_weight_stage2", ("ssim_weight_stage2",), {"value": _default("ssim_weight_stage2"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ("max_visible_angle_deg", "max_visible_angle_deg_stage2", ("max_visible_angle_deg_stage2",), {"value": _default("max_visible_angle_deg_stage2"), "step": 1e-2, "step_fast": 1e-1, "format": "%.6f"}),
        ("max_opacity", "max_opacity_stage2", ("max_opacity_stage2",), {"value": _default("max_opacity_stage2"), "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}),
        ("min_pixel_clamp", "refinement_min_screen_radius_px_stage2", ("refinement_min_screen_radius_px_stage2",), {"value": _default("refinement_min_screen_radius_px_stage2"), "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}),
        ("sort_dither", "sorting_order_dithering_stage2", ("sorting_order_dithering_stage2",), {"value": _default("sorting_order_dithering_stage2"), "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}),
        ("splat_target", "refinement_target_splat_ratio_stage2", ("refinement_target_splat_ratio_stage2",), {"value": _default("refinement_target_splat_ratio_stage2"), "step": 1e-2, "step_fast": 5e-2, "format": "%.4f"}),
        ("prune_lowest", "refinement_prune_lowest_contribution_ratio_stage2", ("refinement_prune_lowest_contribution_ratio_stage2",), {"value": _default("refinement_prune_lowest_contribution_ratio_stage2"), "step": 1e-3, "step_fast": 1e-2, "format": "%.4f"}),
        ("opacity_reg", "opacity_reg_stage2", ("opacity_reg_weight_stage2",), {"value": _default("opacity_reg_weight_stage2"), "step": 1e-4, "step_fast": 1e-3, "format": "%.6g"}),
        ("cam_push", "position_push_away_from_camera_step_stage2", ("position_push_away_from_camera_step_stage2",), {"value": _default("position_push_away_from_camera_step_stage2"), "step": 1e-5, "step_fast": 1e-4, "format": "%.6g"}),
        ("noise_lr", "position_random_step_noise_stage2_lr", ("position_random_step_noise_stage2_lr",), {"value": _default("position_random_step_noise_stage2_lr"), "step": 100.0, "step_fast": 1000.0, "format": "%.4g"}),
        ("sh_band", "sh_band_stage2", ("sh_band_stage2",), {"value": _default("sh_band_stage2"), "options": SH_BAND_LABELS}),
    ),
    "Stage 3": (
        ("end_step", "lr_schedule_stage3_step", ("lr_schedule_stage3_step",), {"value": _default("lr_schedule_stage3_step"), "step": 100, "step_fast": 1000}),
        ("lr", "lr_schedule_stage3_lr", ("lr_schedule_stage3_lr",), {"value": _default("lr_schedule_stage3_lr"), "step": 1e-6, "step_fast": 1e-5, "format": "%.8f"}),
        ("lr_pos_mul", "lr_pos_stage3_mul", ("lr_pos_stage3_mul",), {"value": _default("lr_pos_stage3_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_scale_mul", "lr_scale_stage3_mul", ("lr_scale_stage3_mul",), {"value": _stage_default("lr_scale_stage3_mul", "lr_scale_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_rot_mul", "lr_rot_stage3_mul", ("lr_rot_stage3_mul",), {"value": _stage_default("lr_rot_stage3_mul", "lr_rot_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_color_mul", "lr_color_stage3_mul", ("lr_color_stage3_mul",), {"value": _stage_default("lr_color_stage3_mul", "lr_color_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_opacity_mul", "lr_opacity_stage3_mul", ("lr_opacity_stage3_mul",), {"value": _stage_default("lr_opacity_stage3_mul", "lr_opacity_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_sh_mul", "lr_sh_stage3_mul", ("lr_sh_stage3_mul",), {"value": _default("lr_sh_stage3_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("colorspace_mod", "colorspace_mod_stage3", ("colorspace_mod_stage3",), {"value": _default("colorspace_mod_stage3"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ("ssim_weight", "ssim_weight_stage3", ("ssim_weight_stage3",), {"value": _default("ssim_weight_stage3"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ("max_visible_angle_deg", "max_visible_angle_deg_stage3", ("max_visible_angle_deg_stage3",), {"value": _default("max_visible_angle_deg_stage3"), "step": 1e-2, "step_fast": 1e-1, "format": "%.6f"}),
        ("max_opacity", "max_opacity_stage3", ("max_opacity_stage3",), {"value": _default("max_opacity_stage3"), "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}),
        ("min_pixel_clamp", "refinement_min_screen_radius_px_stage3", ("refinement_min_screen_radius_px_stage3",), {"value": _default("refinement_min_screen_radius_px_stage3"), "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}),
        ("sort_dither", "sorting_order_dithering_stage3", ("sorting_order_dithering_stage3",), {"value": _default("sorting_order_dithering_stage3"), "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}),
        ("splat_target", "refinement_target_splat_ratio_stage3", ("refinement_target_splat_ratio_stage3",), {"value": _default("refinement_target_splat_ratio_stage3"), "step": 1e-2, "step_fast": 5e-2, "format": "%.4f"}),
        ("prune_lowest", "refinement_prune_lowest_contribution_ratio_stage3", ("refinement_prune_lowest_contribution_ratio_stage3",), {"value": _default("refinement_prune_lowest_contribution_ratio_stage3"), "step": 1e-3, "step_fast": 1e-2, "format": "%.4f"}),
        ("opacity_reg", "opacity_reg_stage3", ("opacity_reg_weight_stage3",), {"value": _default("opacity_reg_weight_stage3"), "step": 1e-4, "step_fast": 1e-3, "format": "%.6g"}),
        ("cam_push", "position_push_away_from_camera_step_stage3", ("position_push_away_from_camera_step_stage3",), {"value": _default("position_push_away_from_camera_step_stage3"), "step": 1e-5, "step_fast": 1e-4, "format": "%.6g"}),
        ("noise_lr", "position_random_step_noise_stage3_lr", ("position_random_step_noise_stage3_lr",), {"value": _default("position_random_step_noise_stage3_lr"), "step": 100.0, "step_fast": 1000.0, "format": "%.4g"}),
        ("sh_band", "sh_band_stage3", ("sh_band_stage3",), {"value": _default("sh_band_stage3"), "options": SH_BAND_LABELS}),
    ),
    "Stage 4": (
        ("end_step", "lr_schedule_steps", ("lr_schedule_steps",), {"value": _default("lr_schedule_steps"), "step": 1000, "step_fast": 5000}),
        ("lr", "lr_schedule_end_lr", ("lr_schedule_end_lr",), {"value": _default("lr_schedule_end_lr"), "step": 1e-6, "step_fast": 1e-5, "format": "%.8f"}),
        ("lr_pos_mul", "lr_pos_stage4_mul", ("lr_pos_stage4_mul",), {"value": _default("lr_pos_stage4_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_scale_mul", "lr_scale_stage4_mul", ("lr_scale_stage4_mul",), {"value": _stage_default("lr_scale_stage4_mul", "lr_scale_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_rot_mul", "lr_rot_stage4_mul", ("lr_rot_stage4_mul",), {"value": _stage_default("lr_rot_stage4_mul", "lr_rot_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_color_mul", "lr_color_stage4_mul", ("lr_color_stage4_mul",), {"value": _stage_default("lr_color_stage4_mul", "lr_color_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_opacity_mul", "lr_opacity_stage4_mul", ("lr_opacity_stage4_mul",), {"value": _stage_default("lr_opacity_stage4_mul", "lr_opacity_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_sh_mul", "lr_sh_stage4_mul", ("lr_sh_stage4_mul",), {"value": _default("lr_sh_stage4_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("colorspace_mod", "colorspace_mod_stage4", ("colorspace_mod_stage4",), {"value": _default("colorspace_mod_stage4"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ("ssim_weight", "ssim_weight_stage4", ("ssim_weight_stage4",), {"value": _default("ssim_weight_stage4"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ("max_visible_angle_deg", "max_visible_angle_deg_stage4", ("max_visible_angle_deg_stage4",), {"value": _default("max_visible_angle_deg_stage4"), "step": 1e-2, "step_fast": 1e-1, "format": "%.6f"}),
        ("max_opacity", "max_opacity_stage4", ("max_opacity_stage4",), {"value": _default("max_opacity_stage4"), "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}),
        ("min_pixel_clamp", "refinement_min_screen_radius_px_stage4", ("refinement_min_screen_radius_px_stage4",), {"value": _default("refinement_min_screen_radius_px_stage4"), "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}),
        ("sort_dither", "sorting_order_dithering_stage4", ("sorting_order_dithering_stage4",), {"value": _default("sorting_order_dithering_stage4"), "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}),
        ("splat_target", "refinement_target_splat_ratio_stage4", ("refinement_target_splat_ratio_stage4",), {"value": _default("refinement_target_splat_ratio_stage4"), "step": 1e-2, "step_fast": 5e-2, "format": "%.4f"}),
        ("prune_lowest", "refinement_prune_lowest_contribution_ratio_stage4", ("refinement_prune_lowest_contribution_ratio_stage4",), {"value": _default("refinement_prune_lowest_contribution_ratio_stage4"), "step": 1e-3, "step_fast": 1e-2, "format": "%.4f"}),
        ("opacity_reg", "opacity_reg_stage4", ("opacity_reg_weight_stage4",), {"value": _default("opacity_reg_weight_stage4"), "step": 1e-4, "step_fast": 1e-3, "format": "%.6g"}),
        ("cam_push", "position_push_away_from_camera_step_stage4", ("position_push_away_from_camera_step_stage4",), {"value": _default("position_push_away_from_camera_step_stage4"), "step": 1e-5, "step_fast": 1e-4, "format": "%.6g"}),
        ("noise_lr", "position_random_step_noise_stage4_lr", ("position_random_step_noise_stage4_lr",), {"value": _default("position_random_step_noise_stage4_lr"), "step": 100.0, "step_fast": 1000.0, "format": "%.4g"}),
        ("sh_band", "sh_band_stage4", ("sh_band_stage4",), {"value": _default("sh_band_stage4"), "options": SH_BAND_LABELS}),
    ),
}


def _build_schedule_stage_defs() -> dict[str, tuple[TrainingControlDef, ...]]:
    groups: dict[str, tuple[TrainingControlDef, ...]] = {}
    for stage_label, entries in _SCHEDULE_STAGE_CONFIGS.items():
        groups[stage_label] = tuple(
            _control(
                key=key,
                kind=str(_SCHEDULE_STAGE_TEMPLATE[slot]["kind"]),
                label=str(_SCHEDULE_STAGE_TEMPLATE[slot]["label"]),
                kwargs=dict(_SCHEDULE_STAGE_TEMPLATE[slot].get("kwargs", {}) | kwargs),
                group=TRAINING_OPTIMIZER_GROUP,
                build_args=build_args,
                schedule_stage=stage_label,
                schedule_slot=slot,
            )
            for slot, key, build_args, kwargs in entries
        )
    return groups


SCHEDULE_STAGE_CONTROL_DEFS = _build_schedule_stage_defs()

TRAIN_STABILITY_CONTROL_DEFS = (
    _control("min_opacity", "input_float", "Min Opacity", {"value": _default("min_opacity"), "step": 1e-5, "step_fast": 1e-4, "format": "%.8f"}, group=TRAINING_STABILITY_GROUP, build_args=("min_opacity",)),
    _control("max_opacity", "input_float", "Max Opacity", {"value": _default("max_opacity"), "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}, group=TRAINING_STABILITY_GROUP, build_args=("max_opacity",)),
    _control("position_abs_max", "input_float", "Pos Abs Max", {"value": _default("position_abs_max"), "step": 10.0, "step_fast": 100.0, "format": "%.3f"}, group=TRAINING_STABILITY_GROUP, build_args=("position_abs_max",)),
    _control("camera_min_dist", "input_float", "Cam Min Dist", {"value": _default("camera_min_dist"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}, group=TRAINING_STABILITY_GROUP, build_args=("camera_min_dist",)),
)

TRAINING_UI_GROUP_DEFS = {
    TRAINING_SETUP_GROUP: TRAIN_SETUP_CONTROL_DEFS,
    TRAINING_OPTIMIZER_GROUP: TRAIN_OPTIMIZER_CONTROL_DEFS + tuple(spec for specs in SCHEDULE_STAGE_CONTROL_DEFS.values() for spec in specs),
    TRAINING_STABILITY_GROUP: TRAIN_STABILITY_CONTROL_DEFS,
}

TRAINING_CONTROL_DEFAULTS = {
    spec.key: spec.kwargs["value"]
    for specs in TRAINING_UI_GROUP_DEFS.values()
    for spec in specs
    if "value" in spec.kwargs
}

TRAINING_BUILD_ARG_UI_KEYS = {
    build_arg: spec.key
    for specs in TRAINING_UI_GROUP_DEFS.values()
    for spec in specs
    for build_arg in spec.build_args
}

TRAINING_OPTIMIZER_TAB_KEYS = {
    tab: tuple(spec.key for spec in TRAIN_OPTIMIZER_CONTROL_DEFS if spec.optimizer_tab == tab)
    for tab in ("Schedule", "Adam", "Regularization")
}

TRAIN_STABILITY_PAIRED_KEYS = (("min_opacity", "max_opacity"),)

SCHEDULE_STAGE_GROUPS = {
    stage_label: {spec.schedule_slot: spec.key for spec in specs if spec.schedule_slot is not None}
    for stage_label, specs in SCHEDULE_STAGE_CONTROL_DEFS.items()
}

def training_control_defaults(*group_names: str) -> dict[str, object]:
    groups = TRAINING_UI_GROUP_DEFS.values() if not group_names else (TRAINING_UI_GROUP_DEFS[name] for name in group_names)
    return {spec.key: spec.kwargs["value"] for specs in groups for spec in specs if "value" in spec.kwargs}
