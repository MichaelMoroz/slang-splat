from __future__ import annotations

from dataclasses import dataclass

from ..training.defaults import (
    DEFAULT_LR_SCHEDULE_STEPS,
    DEFAULT_REFINEMENT_MIN_CONTRIBUTION_DECAY,
    DEFAULT_REFINEMENT_MIN_CONTRIBUTION_PERCENT,
    TRAINING_BUILD_ARG_DEFAULTS,
)

TRAIN_BACKGROUND_MODE_LABELS = ("Custom", "Random")
TRAIN_DOWNSCALE_MODE_LABELS = ("Auto",) + tuple(f"{i}x" for i in range(1, 17))
TRAIN_SUBSAMPLE_LABELS = ("Auto", "Off", "1/2", "1/3", "1/4")
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


@dataclass(frozen=True, slots=True)
class TrainingCliArgDef:
    flags: tuple[str, ...]
    kwargs: dict[str, object]
    dest: str
    build_arg: str | None = None


def _default(name: str) -> object:
    return TRAINING_BUILD_ARG_DEFAULTS[name]


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
    )


def _cli_arg(*flags: str, dest: str, build_arg: str | None = None, **kwargs: object) -> TrainingCliArgDef:
    return TrainingCliArgDef(flags=tuple(flags), kwargs=dict(kwargs, dest=dest), dest=dest, build_arg=build_arg)


TRAIN_SETUP_CONTROL_DEFS = (
    _control("max_gaussians", "slider_int", "Max Gaussians", {"value": _default("max_gaussians"), "min": 1000, "max": 10000000}, group=TRAINING_SETUP_GROUP, build_args=("max_gaussians",)),
    _control("training_steps_per_frame", "slider_int", "Steps / Frame", {"value": 3, "min": 1, "max": 8}, group=TRAINING_SETUP_GROUP),
    _control("background_mode", "combo", "Train Background", {"value": _default("background_mode"), "options": TRAIN_BACKGROUND_MODE_LABELS}, group=TRAINING_SETUP_GROUP, build_args=("background_mode",)),
    _control("use_target_alpha_mask", "checkbox", "Use Target Alpha Mask", {"value": _default("use_target_alpha_mask")}, group=TRAINING_SETUP_GROUP, build_args=("use_target_alpha_mask",)),
    _control("train_background_color", "color_edit3", "Train BG Color", {"value": (1.0, 1.0, 1.0)}, group=TRAINING_SETUP_GROUP),
    _control("refinement_interval", "input_int", "Refinement Interval", {"value": _default("refinement_interval"), "step": 10, "step_fast": 50}, group=TRAINING_SETUP_GROUP, build_args=("refinement_interval",)),
    _control("refinement_growth_ratio", "input_float", "Refinement Growth", {"value": _default("refinement_growth_ratio"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_growth_ratio",)),
    _control("refinement_growth_start_step", "slider_int", "Start Refinement After", {"value": _default("refinement_growth_start_step"), "min": 0, "max": DEFAULT_LR_SCHEDULE_STEPS, "max_from": "lr_schedule_steps"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_growth_start_step",)),
    _control("refinement_alpha_cull_threshold", "input_float", "Refinement Alpha Cull", {"value": _default("refinement_alpha_cull_threshold"), "step": 1e-5, "step_fast": 1e-4, "format": "%.6e"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_alpha_cull_threshold",)),
    _control("refinement_min_contribution_percent", "input_float", "Refinement Min Contribution", {"value": DEFAULT_REFINEMENT_MIN_CONTRIBUTION_PERCENT, "step": 1e-6, "step_fast": 1e-5, "format": "%.6g%%"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_min_contribution_percent",)),
    _control("refinement_min_contribution_decay", "input_float", "Refinement Min Contribution Decay", {"value": DEFAULT_REFINEMENT_MIN_CONTRIBUTION_DECAY, "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_min_contribution_decay",)),
    _control("refinement_opacity_mul", "input_float", "Refinement Alpha Mul", {"value": _default("refinement_opacity_mul"), "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_opacity_mul",)),
    _control("refinement_sample_radius", "input_float", "Refinement Sample Radius", {"value": _default("refinement_sample_radius"), "step": 1e-2, "step_fast": 1e-1, "format": "%.5f"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_sample_radius",)),
    _control("refinement_loss_weight", "input_float", "Refinement Loss Weight", {"value": _default("refinement_loss_weight"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_loss_weight",)),
    _control("refinement_target_edge_weight", "input_float", "Refinement Edge Weight", {"value": _default("refinement_target_edge_weight"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}, group=TRAINING_SETUP_GROUP, build_args=("refinement_target_edge_weight",)),
    _control("train_downscale_mode", "combo", "Downscale Mode", {"value": _default("train_downscale_mode"), "options": TRAIN_DOWNSCALE_MODE_LABELS}, group=TRAINING_SETUP_GROUP, build_args=("train_downscale_mode",)),
    _control("train_subsample_factor", "combo", "Subsampling", {"value": _default("train_subsample_factor"), "options": TRAIN_SUBSAMPLE_LABELS}, group=TRAINING_SETUP_GROUP, build_args=("train_subsample_factor",)),
    _control("train_auto_start_downscale", "slider_int", "Auto Start Downscale", {"value": _default("train_auto_start_downscale"), "min": 1, "max": 16}, group=TRAINING_SETUP_GROUP, build_args=("train_auto_start_downscale",)),
    _control("train_downscale_base_iters", "input_int", "Downscale Base Iters", {"value": _default("train_downscale_base_iters"), "step": 25, "step_fast": 100}, group=TRAINING_SETUP_GROUP, build_args=("train_downscale_base_iters",)),
    _control("train_downscale_iter_step", "input_int", "Downscale Iter Step", {"value": _default("train_downscale_iter_step"), "step": 10, "step_fast": 50}, group=TRAINING_SETUP_GROUP, build_args=("train_downscale_iter_step",)),
    _control("train_downscale_max_iters", "input_int", "Downscale Max Iters", {"value": _default("train_downscale_max_iters"), "step": 1000, "step_fast": 5000}, group=TRAINING_SETUP_GROUP, build_args=("train_downscale_max_iters",)),
    _control("seed", "slider_int", "Shuffle Seed", {"value": 1234, "min": 0, "max": 1000000}, group=TRAINING_SETUP_GROUP),
    _control("init_opacity", "input_float", "Init Opacity", {"value": 0.5, "step": 1e-3, "step_fast": 1e-2, "format": "%.5f"}, group=TRAINING_SETUP_GROUP),
)

TRAIN_OPTIMIZER_CONTROL_DEFS = (
    _control("lr_schedule_enabled", "checkbox", "Use LR Schedule", {"value": _default("lr_schedule_enabled")}, group=TRAINING_OPTIMIZER_GROUP, build_args=("lr_schedule_enabled",), optimizer_tab="Schedule"),
    _control("lr_scale_mul", "input_float", "LR Mul Scale", {"value": _default("lr_scale_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("lr_scale_mul",), optimizer_tab="Schedule"),
    _control("lr_rot_mul", "input_float", "LR Mul Rotation", {"value": _default("lr_rot_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("lr_rot_mul",), optimizer_tab="Schedule"),
    _control("lr_color_mul", "input_float", "LR Mul SH0/DC", {"value": _default("lr_color_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("lr_color_mul",), optimizer_tab="Schedule"),
    _control("lr_opacity_mul", "input_float", "LR Mul Opacity", {"value": _default("lr_opacity_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("lr_opacity_mul",), optimizer_tab="Schedule"),
    _control("beta1", "input_float", "Beta1", {"value": _default("beta1"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("beta1",), optimizer_tab="Adam"),
    _control("beta2", "input_float", "Beta2", {"value": _default("beta2"), "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("beta2",), optimizer_tab="Adam"),
    _control("scale_l2", "input_float", "Scale Log Reg", {"value": _default("scale_l2_weight"), "step": 1e-5, "step_fast": 1e-4, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("scale_l2_weight",), optimizer_tab="Regularization"),
    _control("scale_abs_reg", "input_float", "Scale Abs Reg", {"value": _default("scale_abs_reg_weight"), "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("scale_abs_reg_weight",), optimizer_tab="Regularization"),
    _control("sh1_reg", "input_float", "SH Rest Reg", {"value": _default("sh1_reg_weight"), "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("sh1_reg_weight",), optimizer_tab="Regularization"),
    _control("opacity_reg", "input_float", "Opacity Reg", {"value": _default("opacity_reg_weight"), "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("opacity_reg_weight",), optimizer_tab="Regularization"),
    _control("density_regularizer", "input_float", "Density Reg", {"value": _default("density_regularizer"), "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("density_regularizer",), optimizer_tab="Regularization"),
    _control("color_non_negative_reg", "input_float", "Color >= 0 Reg", {"value": _default("color_non_negative_reg"), "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("color_non_negative_reg",), optimizer_tab="Regularization"),
    _control("depth_ratio_grad_min", "input_float", "Depth Ratio Grad Min", {"value": _default("depth_ratio_grad_min"), "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("depth_ratio_grad_min",), optimizer_tab="Regularization"),
    _control("depth_ratio_grad_max", "input_float", "Depth Ratio Grad Max", {"value": _default("depth_ratio_grad_max"), "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("depth_ratio_grad_max",), optimizer_tab="Regularization"),
    _control("ssim_c2", "input_float", "SSIM C2", {"value": _default("ssim_c2"), "step": 1e-5, "step_fast": 1e-4, "format": "%.6e"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("ssim_c2",), optimizer_tab="Regularization"),
    _control("max_allowed_density", "input_float", "Max Density", {"value": _default("max_allowed_density"), "step": 1e-3, "step_fast": 1e-2, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("max_allowed_density",), optimizer_tab="Regularization"),
    _control("position_random_step_opacity_gate_center", "input_float", "Noise Gate Center", {"value": _default("position_random_step_opacity_gate_center"), "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("position_random_step_opacity_gate_center",), optimizer_tab="Regularization"),
    _control("position_random_step_opacity_gate_sharpness", "input_float", "Noise Gate Sharpness", {"value": _default("position_random_step_opacity_gate_sharpness"), "step": 1.0, "step_fast": 10.0, "format": "%.4g"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("position_random_step_opacity_gate_sharpness",), optimizer_tab="Regularization"),
    _control("max_anisotropy", "input_float", "Max Anisotropy", {"value": _default("max_anisotropy"), "step": 0.1, "step_fast": 0.5, "format": "%.6f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("max_anisotropy",), optimizer_tab="Regularization"),
    _control("grad_clip", "input_float", "Grad Clip", {"value": _default("grad_clip"), "step": 0.1, "step_fast": 1.0, "format": "%.4f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("grad_clip",), optimizer_tab="Regularization"),
    _control("grad_norm_clip", "input_float", "Grad Norm Clip", {"value": _default("grad_norm_clip"), "step": 0.1, "step_fast": 1.0, "format": "%.4f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("grad_norm_clip",), optimizer_tab="Regularization"),
    _control("max_update", "input_float", "Max Update", {"value": _default("max_update"), "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}, group=TRAINING_OPTIMIZER_GROUP, build_args=("max_update",), optimizer_tab="Regularization"),
)

_SCHEDULE_STAGE_TEMPLATE = {
    "end_step": {"kind": "slider_int", "label": "End Step", "kwargs": {"min": 0, "max": DEFAULT_LR_SCHEDULE_STEPS, "max_from": "lr_schedule_steps"}},
    "lr": {"kind": "input_float", "label": "LR Target"},
    "lr_pos_mul": {"kind": "input_float", "label": "LR Mul Position"},
    "lr_sh_mul": {"kind": "input_float", "label": "LR Mul SH"},
    "depth_ratio_weight": {"kind": "input_float", "label": "Depth Ratio Reg"},
    "ssim_weight": {"kind": "input_float", "label": "DSSIM Weight"},
    "max_screen_fraction": {"kind": "input_float", "label": "Max Splat Screen Size"},
    "noise_lr": {"kind": "input_float", "label": "Noise LR"},
    "sh_band": {"kind": "combo", "label": "SH Band", "kwargs": {"options": SH_BAND_LABELS}},
}

_SCHEDULE_STAGE_CONFIGS = {
    "Stage 0": (
        ("lr", "lr_schedule_start_lr", ("base_lr", "lr_schedule_start_lr"), {"value": _default("lr_schedule_start_lr"), "step": 1e-5, "step_fast": 1e-4, "format": "%.8f"}),
        ("lr_pos_mul", "lr_pos_mul", ("lr_pos_mul",), {"value": _default("lr_pos_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_sh_mul", "lr_sh_mul", ("lr_sh_mul",), {"value": _default("lr_sh_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("depth_ratio_weight", "depth_ratio_weight", ("depth_ratio_weight",), {"value": _default("depth_ratio_weight"), "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}),
        ("ssim_weight", "ssim_weight", ("ssim_weight",), {"value": _default("ssim_weight"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ("max_screen_fraction", "max_screen_fraction", ("max_screen_fraction",), {"value": _default("max_screen_fraction"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ("noise_lr", "position_random_step_noise_lr", ("position_random_step_noise_lr",), {"value": _default("position_random_step_noise_lr"), "step": 100.0, "step_fast": 1000.0, "format": "%.4g"}),
        ("sh_band", "sh_band", ("sh_band",), {"value": _default("sh_band"), "options": SH_BAND_LABELS}),
    ),
    "Stage 1": (
        ("end_step", "lr_schedule_stage1_step", ("lr_schedule_stage1_step",), {"value": _default("lr_schedule_stage1_step"), "min": 0, "max": DEFAULT_LR_SCHEDULE_STEPS, "max_from": "lr_schedule_steps"}),
        ("lr", "lr_schedule_stage1_lr", ("lr_schedule_stage1_lr",), {"value": _default("lr_schedule_stage1_lr"), "step": 1e-6, "step_fast": 1e-5, "format": "%.8f"}),
        ("lr_pos_mul", "lr_pos_stage1_mul", ("lr_pos_stage1_mul",), {"value": _default("lr_pos_stage1_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_sh_mul", "lr_sh_stage1_mul", ("lr_sh_stage1_mul",), {"value": _default("lr_sh_stage1_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("depth_ratio_weight", "depth_ratio_stage1_weight", ("depth_ratio_stage1_weight",), {"value": _default("depth_ratio_stage1_weight"), "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}),
        ("ssim_weight", "ssim_weight_stage1", ("ssim_weight_stage1",), {"value": _default("ssim_weight_stage1"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ("max_screen_fraction", "max_screen_fraction_stage1", ("max_screen_fraction_stage1",), {"value": _default("max_screen_fraction_stage1"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ("noise_lr", "position_random_step_noise_stage1_lr", ("position_random_step_noise_stage1_lr",), {"value": _default("position_random_step_noise_stage1_lr"), "step": 100.0, "step_fast": 1000.0, "format": "%.4g"}),
        ("sh_band", "sh_band_stage1", ("sh_band_stage1",), {"value": _default("sh_band_stage1"), "options": SH_BAND_LABELS}),
    ),
    "Stage 2": (
        ("end_step", "lr_schedule_stage2_step", ("lr_schedule_stage2_step",), {"value": _default("lr_schedule_stage2_step"), "min": 0, "max": DEFAULT_LR_SCHEDULE_STEPS, "max_from": "lr_schedule_steps"}),
        ("lr", "lr_schedule_stage2_lr", ("lr_schedule_stage2_lr",), {"value": _default("lr_schedule_stage2_lr"), "step": 1e-6, "step_fast": 1e-5, "format": "%.8f"}),
        ("lr_pos_mul", "lr_pos_stage2_mul", ("lr_pos_stage2_mul",), {"value": _default("lr_pos_stage2_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_sh_mul", "lr_sh_stage2_mul", ("lr_sh_stage2_mul",), {"value": _default("lr_sh_stage2_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("depth_ratio_weight", "depth_ratio_stage2_weight", ("depth_ratio_stage2_weight",), {"value": _default("depth_ratio_stage2_weight"), "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}),
        ("ssim_weight", "ssim_weight_stage2", ("ssim_weight_stage2",), {"value": _default("ssim_weight_stage2"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ("max_screen_fraction", "max_screen_fraction_stage2", ("max_screen_fraction_stage2",), {"value": _default("max_screen_fraction_stage2"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ("noise_lr", "position_random_step_noise_stage2_lr", ("position_random_step_noise_stage2_lr",), {"value": _default("position_random_step_noise_stage2_lr"), "step": 100.0, "step_fast": 1000.0, "format": "%.4g"}),
        ("sh_band", "sh_band_stage2", ("sh_band_stage2",), {"value": _default("sh_band_stage2"), "options": SH_BAND_LABELS}),
    ),
    "Stage 3": (
        ("end_step", "lr_schedule_steps", ("lr_schedule_steps",), {"value": _default("lr_schedule_steps"), "step": 1000, "step_fast": 5000}),
        ("lr", "lr_schedule_end_lr", ("lr_schedule_end_lr",), {"value": _default("lr_schedule_end_lr"), "step": 1e-6, "step_fast": 1e-5, "format": "%.8f"}),
        ("lr_pos_mul", "lr_pos_stage3_mul", ("lr_pos_stage3_mul",), {"value": _default("lr_pos_stage3_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("lr_sh_mul", "lr_sh_stage3_mul", ("lr_sh_stage3_mul",), {"value": _default("lr_sh_stage3_mul"), "step": 1e-2, "step_fast": 1e-1, "format": "%.8f"}),
        ("depth_ratio_weight", "depth_ratio_stage3_weight", ("depth_ratio_stage3_weight",), {"value": _default("depth_ratio_stage3_weight"), "step": 1e-4, "step_fast": 1e-3, "format": "%.8f"}),
        ("ssim_weight", "ssim_weight_stage3", ("ssim_weight_stage3",), {"value": _default("ssim_weight_stage3"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ("max_screen_fraction", "max_screen_fraction_stage3", ("max_screen_fraction_stage3",), {"value": _default("max_screen_fraction_stage3"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}),
        ("noise_lr", "position_random_step_noise_stage3_lr", ("position_random_step_noise_stage3_lr",), {"value": _default("position_random_step_noise_stage3_lr"), "step": 100.0, "step_fast": 1000.0, "format": "%.4g"}),
        ("sh_band", "sh_band_stage3", ("sh_band_stage3",), {"value": _default("sh_band_stage3"), "options": SH_BAND_LABELS}),
    ),
}


def _build_schedule_stage_defs() -> dict[str, tuple[TrainingControlDef, ...]]:
    groups: dict[str, tuple[TrainingControlDef, ...]] = {}
    for stage_label, entries in _SCHEDULE_STAGE_CONFIGS.items():
        groups[stage_label] = tuple(
            _control(
                key=key,
                kind=str(_SCHEDULE_STAGE_TEMPLATE[slot]["kind"] if slot != "end_step" or stage_label != "Stage 3" else "input_int"),
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
    _control("max_scale", "input_float", "Max Scale", {"value": _default("max_scale"), "step": 1e-2, "step_fast": 0.1, "format": "%.5f"}, group=TRAINING_STABILITY_GROUP, build_args=("max_scale",)),
    _control("min_opacity", "input_float", "Min Opacity", {"value": _default("min_opacity"), "step": 1e-5, "step_fast": 1e-4, "format": "%.8f"}, group=TRAINING_STABILITY_GROUP, build_args=("min_opacity",)),
    _control("max_opacity", "input_float", "Max Opacity", {"value": _default("max_opacity"), "step": 1e-4, "step_fast": 1e-3, "format": "%.6f"}, group=TRAINING_STABILITY_GROUP, build_args=("max_opacity",)),
    _control("position_abs_max", "input_float", "Pos Abs Max", {"value": _default("position_abs_max"), "step": 10.0, "step_fast": 100.0, "format": "%.3f"}, group=TRAINING_STABILITY_GROUP, build_args=("position_abs_max",)),
    _control("train_near", "input_float", "Train Near", {"value": _default("near"), "step": 1e-3, "step_fast": 1e-2, "format": "%.6f"}, group=TRAINING_STABILITY_GROUP, build_args=("near",)),
    _control("train_far", "input_float", "Train Far", {"value": _default("far"), "step": 1.0, "step_fast": 10.0, "format": "%.3f"}, group=TRAINING_STABILITY_GROUP, build_args=("far",)),
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

TRAIN_SETUP_PRIMARY_KEYS = (
    "max_gaussians",
    "training_steps_per_frame",
    "background_mode",
    "refinement_interval",
    "refinement_growth_ratio",
    "refinement_growth_start_step",
    "refinement_alpha_cull_threshold",
    "refinement_min_contribution_percent",
    "refinement_min_contribution_decay",
    "refinement_opacity_mul",
    "refinement_sample_radius",
    "refinement_loss_weight",
    "refinement_target_edge_weight",
    "train_downscale_mode",
    "train_subsample_factor",
)
TRAIN_SETUP_AUTO_DOWNSCALE_KEYS = (
    "train_auto_start_downscale",
    "train_downscale_base_iters",
    "train_downscale_iter_step",
    "train_downscale_max_iters",
)
TRAIN_SETUP_TRAILING_KEYS = ("seed", "init_opacity")
TRAIN_STABILITY_PAIRED_KEYS = (("min_opacity", "max_opacity"),)

SCHEDULE_STAGE_GROUPS = {
    stage_label: {spec.schedule_slot: spec.key for spec in specs if spec.schedule_slot is not None}
    for stage_label, specs in SCHEDULE_STAGE_CONTROL_DEFS.items()
}

TRAINING_CLI_ARG_DEFS = (
    _cli_arg("--lr-base", dest="lr_base", build_arg="base_lr", type=float, default=_default("base_lr")),
    _cli_arg("--lr-mul-pos", dest="lr_mul_pos", build_arg="lr_pos_mul", type=float, default=_default("lr_pos_mul")),
    _cli_arg("--lr-mul-scale", dest="lr_mul_scale", build_arg="lr_scale_mul", type=float, default=_default("lr_scale_mul")),
    _cli_arg("--lr-mul-rot", dest="lr_mul_rot", build_arg="lr_rot_mul", type=float, default=_default("lr_rot_mul")),
    _cli_arg("--lr-mul-color", dest="lr_mul_color", build_arg="lr_color_mul", type=float, default=_default("lr_color_mul")),
    _cli_arg("--lr-mul-opacity", dest="lr_mul_opacity", build_arg="lr_opacity_mul", type=float, default=_default("lr_opacity_mul")),
    _cli_arg("--beta1", dest="beta1", build_arg="beta1", type=float, default=_default("beta1")),
    _cli_arg("--beta2", dest="beta2", build_arg="beta2", type=float, default=_default("beta2")),
    _cli_arg("--grad-clip", dest="grad_clip", build_arg="grad_clip", type=float, default=_default("grad_clip")),
    _cli_arg("--grad-norm-clip", dest="grad_norm_clip", build_arg="grad_norm_clip", type=float, default=_default("grad_norm_clip")),
    _cli_arg("--max-update", dest="max_update", build_arg="max_update", type=float, default=_default("max_update")),
    _cli_arg("--max-scale", dest="max_scale", build_arg="max_scale", type=float, default=_default("max_scale")),
    _cli_arg("--min-opacity", dest="min_opacity", build_arg="min_opacity", type=float, default=_default("min_opacity")),
    _cli_arg("--max-opacity", dest="max_opacity", build_arg="max_opacity", type=float, default=_default("max_opacity")),
    _cli_arg("--position-abs-max", dest="position_abs_max", build_arg="position_abs_max", type=float, default=_default("position_abs_max")),
    _cli_arg("--loss-grad-clip", dest="loss_grad_clip", type=float, default=_default("grad_clip")),
    _cli_arg("--near", dest="near", build_arg="near", type=float, default=_default("near")),
    _cli_arg("--far", dest="far", build_arg="far", type=float, default=_default("far")),
    _cli_arg("--scale-l2", dest="scale_l2", build_arg="scale_l2_weight", type=float, default=_default("scale_l2_weight")),
    _cli_arg("--scale-abs-reg", dest="scale_abs_reg", build_arg="scale_abs_reg_weight", type=float, default=_default("scale_abs_reg_weight")),
    _cli_arg("--sh1-reg", dest="sh1_reg", build_arg="sh1_reg_weight", type=float, default=_default("sh1_reg_weight")),
    _cli_arg("--opacity-reg", dest="opacity_reg", build_arg="opacity_reg_weight", type=float, default=_default("opacity_reg_weight")),
    _cli_arg("--density-reg", dest="density_reg", build_arg="density_regularizer", type=float, default=_default("density_regularizer")),
    _cli_arg("--depth-ratio-weight", dest="depth_ratio_weight", build_arg="depth_ratio_weight", type=float, default=_default("depth_ratio_weight")),
    _cli_arg("--ssim-weight", dest="ssim_weight", build_arg="ssim_weight", type=float, default=_default("ssim_weight")),
    _cli_arg("--refinement-loss-weight", dest="refinement_loss_weight", build_arg="refinement_loss_weight", type=float, default=_default("refinement_loss_weight")),
    _cli_arg("--refinement-target-edge-weight", dest="refinement_target_edge_weight", build_arg="refinement_target_edge_weight", type=float, default=_default("refinement_target_edge_weight")),
    _cli_arg("--depth-ratio-grad-min", dest="depth_ratio_grad_min", build_arg="depth_ratio_grad_min", type=float, default=_default("depth_ratio_grad_min")),
    _cli_arg("--depth-ratio-grad-max", dest="depth_ratio_grad_max", build_arg="depth_ratio_grad_max", type=float, default=_default("depth_ratio_grad_max")),
    _cli_arg("--max-allowed-density-start", dest="max_allowed_density_start", build_arg="max_allowed_density_start", type=float, default=_default("max_allowed_density_start")),
    _cli_arg("--max-allowed-density", dest="max_allowed_density", build_arg="max_allowed_density", type=float, default=_default("max_allowed_density")),
    _cli_arg("--max-anisotropy", dest="max_anisotropy", build_arg="max_anisotropy", type=float, default=_default("max_anisotropy")),
)


def training_control_defaults(*group_names: str) -> dict[str, object]:
    groups = TRAINING_UI_GROUP_DEFS.values() if not group_names else (TRAINING_UI_GROUP_DEFS[name] for name in group_names)
    return {spec.key: spec.kwargs["value"] for specs in groups for spec in specs if "value" in spec.kwargs}


def training_cli_build_kwargs(args: object) -> dict[str, object]:
    return {spec.build_arg: getattr(args, spec.dest) for spec in TRAINING_CLI_ARG_DEFS if spec.build_arg is not None}
