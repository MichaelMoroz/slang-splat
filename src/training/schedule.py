from __future__ import annotations

import math
from typing import Any

from .defaults import DEFAULT_LR_SCHEDULE_STEPS, DEFAULT_LR_STAGE1_STEP, DEFAULT_LR_STAGE2_STEP, DEFAULT_LR_STAGE3_STEP, DEFAULT_MAX_OPACITY_STAGE0, DEFAULT_MAX_OPACITY_STAGE1, DEFAULT_MAX_OPACITY_STAGE2, DEFAULT_MAX_OPACITY_STAGE3, DEFAULT_MAX_OPACITY_STAGE4, DEFAULT_REFINEMENT_MIN_CONTRIBUTION_DECAY

_SCHEDULE_REFERENCE_STEPS = DEFAULT_LR_SCHEDULE_STEPS
_DEFAULT_MAX_SH_BAND = 3


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _schedule_duration(training_hparams: Any) -> int:
    return _coerce_int(getattr(training_hparams, "lr_schedule_steps", _SCHEDULE_REFERENCE_STEPS), _SCHEDULE_REFERENCE_STEPS)


def _step_progress(step: int, max_step: int) -> float:
    if int(max_step) == 0:
        return 0.0
    return _coerce_int(step, 0) / float(max_step)


def _schedule_progress(training_hparams: Any, step: int) -> float:
    duration = _schedule_duration(training_hparams)
    if duration <= 0:
        return 0.0
    return min(max(_coerce_int(step, 0), 0), duration) / float(duration)


def resolve_lr_schedule_breakpoints(training_hparams: Any) -> tuple[int, int, int, int]:
    return (
        _coerce_int(getattr(training_hparams, "lr_schedule_stage1_step", DEFAULT_LR_STAGE1_STEP), DEFAULT_LR_STAGE1_STEP),
        _coerce_int(getattr(training_hparams, "lr_schedule_stage2_step", DEFAULT_LR_STAGE2_STEP), DEFAULT_LR_STAGE2_STEP),
        _coerce_int(getattr(training_hparams, "lr_schedule_stage3_step", DEFAULT_LR_STAGE3_STEP), DEFAULT_LR_STAGE3_STEP),
        _schedule_duration(training_hparams),
    )


def resolve_stage_schedule_steps(training_hparams: Any) -> tuple[int, int, int, int]:
    return resolve_lr_schedule_breakpoints(training_hparams)


def _piecewise_linear_schedule(progress: float, milestones: tuple[tuple[float, float], ...]) -> float:
    resolved_progress = min(max(float(progress), 0.0), 1.0)
    if not milestones:
        return 0.0
    if resolved_progress <= float(milestones[0][0]):
        return float(milestones[0][1])
    for index in range(len(milestones) - 1):
        start_progress, start_value = milestones[index]
        end_progress, end_value = milestones[index + 1]
        if resolved_progress <= float(end_progress):
            span = max(float(end_progress) - float(start_progress), 1e-8)
            t = (resolved_progress - float(start_progress)) / span
            return float(start_value) + (float(end_value) - float(start_value)) * t
    return float(milestones[-1][1])


def _stage_progress_milestones(training_hparams: Any) -> tuple[float, float, float, float]:
    stage1, stage2, stage3, stage4 = resolve_stage_schedule_steps(training_hparams)
    if stage4 <= 0:
        return 0.0, 0.0, 0.0, 1.0
    stage1_progress = min(max(_step_progress(stage1, stage4), 0.0), 1.0)
    stage2_progress = min(max(_step_progress(stage2, stage4), stage1_progress), 1.0)
    stage3_progress = min(max(_step_progress(stage3, stage4), stage2_progress), 1.0)
    return stage1_progress, stage2_progress, stage3_progress, 1.0


def _resolve_staged_linear_value(training_hparams: Any, step: int, initial_value: float, stage_values: tuple[float, float, float, float]) -> float:
    stage1_progress, stage2_progress, stage3_progress, stage4_progress = _stage_progress_milestones(training_hparams)
    milestones = (
        (0.0, float(initial_value)),
        (stage1_progress, float(stage_values[0])),
        (stage2_progress, float(stage_values[1])),
        (stage3_progress, float(stage_values[2])),
        (stage4_progress, float(stage_values[3])),
    )
    return _piecewise_linear_schedule(_schedule_progress(training_hparams, step), milestones)


def resolve_base_learning_rate(training_hparams: Any, step: int) -> float:
    enabled = bool(getattr(training_hparams, "lr_schedule_enabled", True))
    start = _coerce_float(getattr(training_hparams, "lr_schedule_start_lr", 1e-3), 1e-3)
    if not enabled:
        return start
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        start,
        (
            _coerce_float(getattr(training_hparams, "lr_schedule_stage1_lr", 0.002), 0.002),
            _coerce_float(getattr(training_hparams, "lr_schedule_stage2_lr", 0.001), 0.001),
            _coerce_float(getattr(training_hparams, "lr_schedule_stage3_lr", getattr(training_hparams, "lr_schedule_end_lr", 1.5e-4)), 1.5e-4),
            _coerce_float(getattr(training_hparams, "lr_schedule_end_lr", 0.001), 0.001),
        ),
    )


def resolve_cosine_base_learning_rate(training_hparams: Any, step: int) -> float:
    return resolve_base_learning_rate(training_hparams, step)


def resolve_learning_rate_scale(training_hparams: Any, step: int) -> float:
    start = _coerce_float(getattr(training_hparams, "lr_schedule_start_lr", 1e-3), 1e-3)
    return resolve_base_learning_rate(training_hparams, step) / (start if start != 0.0 else 1.0)


def resolve_position_lr_mul(training_hparams: Any, step: int) -> float:
    start = _coerce_float(getattr(training_hparams, "lr_pos_mul", 1.0), 1.0)
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return start
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        start,
        (
            _coerce_float(getattr(training_hparams, "lr_pos_stage1_mul", 1.0), 1.0),
            _coerce_float(getattr(training_hparams, "lr_pos_stage2_mul", 1.0), 1.0),
            _coerce_float(getattr(training_hparams, "lr_pos_stage3_mul", 1.0), 1.0),
            _coerce_float(getattr(training_hparams, "lr_pos_stage4_mul", getattr(training_hparams, "lr_pos_stage3_mul", 1.0)), 1.0),
        ),
    )


def resolve_scale_lr_mul(training_hparams: Any, step: int) -> float:
    start = _coerce_float(getattr(training_hparams, "lr_scale_mul", 1.0), 1.0)
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return start
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        start,
        (
            _coerce_float(getattr(training_hparams, "lr_scale_stage1_mul", start), start),
            _coerce_float(getattr(training_hparams, "lr_scale_stage2_mul", start), start),
            _coerce_float(getattr(training_hparams, "lr_scale_stage3_mul", start), start),
            _coerce_float(getattr(training_hparams, "lr_scale_stage4_mul", getattr(training_hparams, "lr_scale_stage3_mul", start)), start),
        ),
    )


def resolve_rotation_lr_mul(training_hparams: Any, step: int) -> float:
    start = _coerce_float(getattr(training_hparams, "lr_rot_mul", 1.0), 1.0)
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return start
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        start,
        (
            _coerce_float(getattr(training_hparams, "lr_rot_stage1_mul", start), start),
            _coerce_float(getattr(training_hparams, "lr_rot_stage2_mul", start), start),
            _coerce_float(getattr(training_hparams, "lr_rot_stage3_mul", start), start),
            _coerce_float(getattr(training_hparams, "lr_rot_stage4_mul", getattr(training_hparams, "lr_rot_stage3_mul", start)), start),
        ),
    )


def resolve_color_lr_mul(training_hparams: Any, step: int) -> float:
    start = _coerce_float(getattr(training_hparams, "lr_color_mul", 1.0), 1.0)
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return start
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        start,
        (
            _coerce_float(getattr(training_hparams, "lr_color_stage1_mul", start), start),
            _coerce_float(getattr(training_hparams, "lr_color_stage2_mul", start), start),
            _coerce_float(getattr(training_hparams, "lr_color_stage3_mul", start), start),
            _coerce_float(getattr(training_hparams, "lr_color_stage4_mul", getattr(training_hparams, "lr_color_stage3_mul", start)), start),
        ),
    )


def resolve_opacity_lr_mul(training_hparams: Any, step: int) -> float:
    start = _coerce_float(getattr(training_hparams, "lr_opacity_mul", 1.0), 1.0)
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return start
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        start,
        (
            _coerce_float(getattr(training_hparams, "lr_opacity_stage1_mul", start), start),
            _coerce_float(getattr(training_hparams, "lr_opacity_stage2_mul", start), start),
            _coerce_float(getattr(training_hparams, "lr_opacity_stage3_mul", start), start),
            _coerce_float(getattr(training_hparams, "lr_opacity_stage4_mul", getattr(training_hparams, "lr_opacity_stage3_mul", start)), start),
        ),
    )


def resolve_sh_lr_mul(training_hparams: Any, step: int) -> float:
    start = _coerce_float(getattr(training_hparams, "lr_sh_mul", 1.0), 1.0)
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return start
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        start,
        (
            _coerce_float(getattr(training_hparams, "lr_sh_stage1_mul", 1.0), 1.0),
            _coerce_float(getattr(training_hparams, "lr_sh_stage2_mul", 1.0), 1.0),
            _coerce_float(getattr(training_hparams, "lr_sh_stage3_mul", 1.0), 1.0),
            _coerce_float(getattr(training_hparams, "lr_sh_stage4_mul", getattr(training_hparams, "lr_sh_stage3_mul", 1.0)), 1.0),
        ),
    )


def resolve_ssim_weight(training_hparams: Any, step: int) -> float:
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return _coerce_float(getattr(training_hparams, "ssim_weight", 0.05), 0.05)
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        _coerce_float(getattr(training_hparams, "ssim_weight", 0.05), 0.05),
        (
            _coerce_float(getattr(training_hparams, "ssim_weight_stage1", 0.1), 0.1),
            _coerce_float(getattr(training_hparams, "ssim_weight_stage2", 0.3), 0.3),
            _coerce_float(getattr(training_hparams, "ssim_weight_stage3", 0.4), 0.4),
            _coerce_float(getattr(training_hparams, "ssim_weight_stage4", getattr(training_hparams, "ssim_weight_stage3", 0.4)), 0.4),
        ),
    )


def resolve_max_visible_angle_deg(training_hparams: Any, step: int) -> float:
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return _coerce_float(getattr(training_hparams, "max_visible_angle_deg", 1.0), 1.0)
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        _coerce_float(getattr(training_hparams, "max_visible_angle_deg", 1.0), 1.0),
        (
            _coerce_float(getattr(training_hparams, "max_visible_angle_deg_stage1", 1.0), 1.0),
            _coerce_float(getattr(training_hparams, "max_visible_angle_deg_stage2", 1.0), 1.0),
            _coerce_float(getattr(training_hparams, "max_visible_angle_deg_stage3", 1.0), 1.0),
            _coerce_float(getattr(training_hparams, "max_visible_angle_deg_stage4", getattr(training_hparams, "max_visible_angle_deg_stage3", 1.0)), 1.0),
        ),
    )


def resolve_refinement_min_screen_radius_px(training_hparams: Any, step: int) -> float:
    start = _coerce_float(getattr(training_hparams, "refinement_min_screen_radius_px", 0.05), 0.05)
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return start
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        start,
        (
            _coerce_float(getattr(training_hparams, "refinement_min_screen_radius_px_stage1", start), start),
            _coerce_float(getattr(training_hparams, "refinement_min_screen_radius_px_stage2", start), start),
            _coerce_float(getattr(training_hparams, "refinement_min_screen_radius_px_stage3", start), start),
            _coerce_float(getattr(training_hparams, "refinement_min_screen_radius_px_stage4", getattr(training_hparams, "refinement_min_screen_radius_px_stage3", start)), start),
        ),
    )


def resolve_position_random_step_noise_lr(training_hparams: Any, step: int) -> float:
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return _coerce_float(getattr(training_hparams, "position_random_step_noise_lr", 5e5), 5e5)
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        _coerce_float(getattr(training_hparams, "position_random_step_noise_lr", 5e5), 5e5),
        (
            _coerce_float(getattr(training_hparams, "position_random_step_noise_stage1_lr", 0.0), 0.0),
            _coerce_float(getattr(training_hparams, "position_random_step_noise_stage2_lr", 0.0), 0.0),
            _coerce_float(getattr(training_hparams, "position_random_step_noise_stage3_lr", 0.0), 0.0),
            _coerce_float(getattr(training_hparams, "position_random_step_noise_stage4_lr", getattr(training_hparams, "position_random_step_noise_stage3_lr", 0.0)), 0.0),
        ),
    )


def resolve_opacity_reg_weight(training_hparams: Any, step: int) -> float:
    start = _coerce_float(getattr(training_hparams, "opacity_reg_weight", 0.0), 0.0)
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return start
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        start,
        (
            _coerce_float(getattr(training_hparams, "opacity_reg_weight_stage1", start), start),
            _coerce_float(getattr(training_hparams, "opacity_reg_weight_stage2", start), start),
            _coerce_float(getattr(training_hparams, "opacity_reg_weight_stage3", start), start),
            _coerce_float(getattr(training_hparams, "opacity_reg_weight_stage4", getattr(training_hparams, "opacity_reg_weight_stage3", start)), start),
        ),
    )


def resolve_max_opacity(training_hparams: Any, step: int) -> float:
    fallback = _coerce_float(getattr(training_hparams, "max_opacity", 1.0), 1.0)
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return fallback
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        _coerce_float(getattr(training_hparams, "max_opacity_stage0", DEFAULT_MAX_OPACITY_STAGE0), DEFAULT_MAX_OPACITY_STAGE0),
        (
            _coerce_float(getattr(training_hparams, "max_opacity_stage1", DEFAULT_MAX_OPACITY_STAGE1), DEFAULT_MAX_OPACITY_STAGE1),
            _coerce_float(getattr(training_hparams, "max_opacity_stage2", DEFAULT_MAX_OPACITY_STAGE2), DEFAULT_MAX_OPACITY_STAGE2),
            _coerce_float(getattr(training_hparams, "max_opacity_stage3", DEFAULT_MAX_OPACITY_STAGE3), DEFAULT_MAX_OPACITY_STAGE3),
            _coerce_float(getattr(training_hparams, "max_opacity_stage4", DEFAULT_MAX_OPACITY_STAGE4), DEFAULT_MAX_OPACITY_STAGE4),
        ),
    )


def resolve_position_push_away_from_camera_step(training_hparams: Any, step: int) -> float:
    start = _coerce_float(getattr(training_hparams, "position_push_away_from_camera_step", 0.0), 0.0)
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return start
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        start,
        (
            _coerce_float(getattr(training_hparams, "position_push_away_from_camera_step_stage1", start), start),
            _coerce_float(getattr(training_hparams, "position_push_away_from_camera_step_stage2", start), start),
            _coerce_float(getattr(training_hparams, "position_push_away_from_camera_step_stage3", start), start),
            _coerce_float(getattr(training_hparams, "position_push_away_from_camera_step_stage4", getattr(training_hparams, "position_push_away_from_camera_step_stage3", start)), start),
        ),
    )


def resolve_sorting_order_dithering(training_hparams: Any, step: int) -> float:
    start = _coerce_float(getattr(training_hparams, "sorting_order_dithering", 0.5), 0.5)
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return start
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        start,
        (
            _coerce_float(getattr(training_hparams, "sorting_order_dithering_stage1", 0.2), 0.2),
            _coerce_float(getattr(training_hparams, "sorting_order_dithering_stage2", 0.05), 0.05),
            _coerce_float(getattr(training_hparams, "sorting_order_dithering_stage3", 0.01), 0.01),
            _coerce_float(getattr(training_hparams, "sorting_order_dithering_stage4", getattr(training_hparams, "sorting_order_dithering_stage3", 0.01)), 0.01),
        ),
    )


def resolve_colorspace_mod(training_hparams: Any, step: int) -> float:
    start = _coerce_float(getattr(training_hparams, "colorspace_mod", 1.0), 1.0)
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return start
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        start,
        (
            _coerce_float(getattr(training_hparams, "colorspace_mod_stage1", 1.0), 1.0),
            _coerce_float(getattr(training_hparams, "colorspace_mod_stage2", 1.0), 1.0),
            _coerce_float(getattr(training_hparams, "colorspace_mod_stage3", 1.0), 1.0),
            _coerce_float(getattr(training_hparams, "colorspace_mod_stage4", getattr(training_hparams, "colorspace_mod_stage3", 1.0)), 1.0),
        ),
    )


def _clamp_sh_band(value: Any, default: int) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = int(default)
    return min(max(resolved, 0), _DEFAULT_MAX_SH_BAND)


def _resolve_legacy_sh_band(training_hparams: Any, key: str, default: bool) -> int:
    return _DEFAULT_MAX_SH_BAND if bool(getattr(training_hparams, key, default)) else 0


def resolve_max_sh_band(training_hparams: Any) -> int:
    return _clamp_sh_band(getattr(training_hparams, "max_sh_band", _DEFAULT_MAX_SH_BAND), _DEFAULT_MAX_SH_BAND)


def resolve_sh_band(training_hparams: Any, step: int) -> int:
    resolved_band: int
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        if hasattr(training_hparams, "sh_band"):
            resolved_band = _clamp_sh_band(getattr(training_hparams, "sh_band", 0), 0)
        else:
            resolved_band = _resolve_legacy_sh_band(training_hparams, "use_sh", False)
        return min(resolved_band, resolve_max_sh_band(training_hparams))
    stage1, stage2, stage3, stage4 = resolve_stage_schedule_steps(training_hparams)
    current_step = max(int(step), 0)
    if current_step < stage1:
        resolved_band = _clamp_sh_band(getattr(training_hparams, "sh_band", _resolve_legacy_sh_band(training_hparams, "use_sh", False)), 0)
    elif current_step < stage2:
        resolved_band = _clamp_sh_band(getattr(training_hparams, "sh_band_stage1", _resolve_legacy_sh_band(training_hparams, "use_sh_stage1", False)), _DEFAULT_MAX_SH_BAND)
    elif current_step < stage3:
        resolved_band = _clamp_sh_band(getattr(training_hparams, "sh_band_stage2", _resolve_legacy_sh_band(training_hparams, "use_sh_stage2", True)), _DEFAULT_MAX_SH_BAND)
    elif current_step < stage4:
        resolved_band = _clamp_sh_band(getattr(training_hparams, "sh_band_stage3", _resolve_legacy_sh_band(training_hparams, "use_sh_stage3", True)), _DEFAULT_MAX_SH_BAND)
    else:
        resolved_band = _clamp_sh_band(getattr(training_hparams, "sh_band_stage4", getattr(training_hparams, "sh_band_stage3", _resolve_legacy_sh_band(training_hparams, "use_sh_stage4", True))), _DEFAULT_MAX_SH_BAND)
    return min(resolved_band, resolve_max_sh_band(training_hparams))


def resolve_use_sh(training_hparams: Any, step: int) -> bool:
    return resolve_sh_band(training_hparams, step) > 0


def resolve_max_allowed_density(training_hparams: Any, step: int) -> float:
    end = _coerce_float(getattr(training_hparams, "max_allowed_density", 12.0), 12.0)
    start = _coerce_float(getattr(training_hparams, "max_allowed_density_start", 5.0), 5.0)
    enabled = bool(getattr(training_hparams, "lr_schedule_enabled", True))
    duration = _coerce_int(getattr(training_hparams, "lr_schedule_steps", 30_000), 30_000)
    if not enabled or duration <= 0:
        return start
    progress = min(max(int(step), 0), duration) / float(duration)
    return start + (end - start) * progress


def _resolve_refinement_start_step(training_hparams: Any) -> int:
    return _coerce_int(getattr(training_hparams, "refinement_growth_start_step", 500), 500)


def resolve_refinement_target_splat_ratio(training_hparams: Any, step: int) -> float:
    start = _coerce_float(getattr(training_hparams, "refinement_target_splat_ratio", 0.10), 0.10)
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return start
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        start,
        (
            _coerce_float(getattr(training_hparams, "refinement_target_splat_ratio_stage1", 0.20), 0.20),
            _coerce_float(getattr(training_hparams, "refinement_target_splat_ratio_stage2", 0.50), 0.50),
            _coerce_float(getattr(training_hparams, "refinement_target_splat_ratio_stage3", 1.00), 1.00),
            _coerce_float(getattr(training_hparams, "refinement_target_splat_ratio_stage4", getattr(training_hparams, "refinement_target_splat_ratio_stage3", 1.00)), 1.00),
        ),
    )


def resolve_refinement_active_target_splat_ratio(training_hparams: Any, step: int) -> float:
    return resolve_refinement_target_splat_ratio(training_hparams, step) if int(step) >= _resolve_refinement_start_step(training_hparams) else 0.0


def _resolve_refinement_target_splat_count(training_hparams: Any, step: int) -> int:
    max_gaussians = max(int(getattr(training_hparams, "max_gaussians", 0)), 0)
    if max_gaussians <= 0:
        return 0
    target_ratio = resolve_refinement_target_splat_ratio(training_hparams, step)
    return min(max(int(math.ceil(float(max_gaussians) * target_ratio)), 0), max_gaussians)


def _estimate_refinement_survivor_count(splat_count: int, prune_ratio: float) -> int:
    count = max(int(splat_count), 0)
    if count <= 0:
        return 0
    prune_count = min(int(math.ceil(float(count) * min(max(float(prune_ratio), 0.0), 1.0))), count)
    return max(count - prune_count, 0)


def resolve_refinement_prune_lowest_contribution_ratio(training_hparams: Any, step: int) -> float:
    start = _coerce_float(getattr(training_hparams, "refinement_prune_lowest_contribution_ratio", 0.0), 0.0)
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return start
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        start,
        (
            _coerce_float(getattr(training_hparams, "refinement_prune_lowest_contribution_ratio_stage1", start), start),
            _coerce_float(getattr(training_hparams, "refinement_prune_lowest_contribution_ratio_stage2", start), start),
            _coerce_float(getattr(training_hparams, "refinement_prune_lowest_contribution_ratio_stage3", start), start),
            _coerce_float(
                getattr(
                    training_hparams,
                    "refinement_prune_lowest_contribution_ratio_stage4",
                    getattr(training_hparams, "refinement_prune_lowest_contribution_ratio_stage3", start),
                ),
                start,
            ),
        ),
    )


def resolve_refinement_prune_ratio(training_hparams: Any, splat_count: int, step: int = 0) -> float:
    base_ratio = resolve_refinement_prune_lowest_contribution_ratio(training_hparams, step)
    current_count = max(int(splat_count), 0)
    if current_count <= 0 or int(step) < _resolve_refinement_start_step(training_hparams):
        return base_ratio
    target_count = _resolve_refinement_target_splat_count(training_hparams, step)
    if target_count <= 0 or target_count >= current_count:
        return base_ratio
    required_ratio = 1.0 - (float(target_count) / float(current_count))
    prune_cap = max(float(getattr(training_hparams, "refinement_max_prune_per_step", 0.15)), 0.0)
    return min(max(base_ratio, min(required_ratio, prune_cap)), 1.0)


def resolve_refinement_min_contribution(training_hparams: Any, step: int, frame_count: int = 1) -> float:
    base_threshold = _coerce_float(getattr(training_hparams, "refinement_min_contribution", 512.0), 512.0)
    decay = _coerce_float(getattr(training_hparams, "refinement_min_contribution_decay", DEFAULT_REFINEMENT_MIN_CONTRIBUTION_DECAY), DEFAULT_REFINEMENT_MIN_CONTRIBUTION_DECAY)
    interval = resolve_effective_refinement_interval(training_hparams, frame_count)
    completed_refinement_steps = max(int(step), 0) // interval
    return base_threshold * math.pow(decay, completed_refinement_steps)


def resolve_effective_refinement_interval(training_hparams: Any, frame_count: int = 1) -> int:
    interval = max(int(getattr(training_hparams, "refinement_interval", 200)), 1)
    return max(interval, max(int(frame_count), 1))


def should_run_refinement_step(training_hparams: Any, step: int, frame_count: int = 1) -> bool:
    interval = resolve_effective_refinement_interval(training_hparams, frame_count)
    return int(step) > 0 and int(step) % interval == 0


def resolve_refinement_clone_budget(training_hparams: Any, splat_count: int, step: int = 0, frame_count: int = 1) -> int:
    if int(step) < _resolve_refinement_start_step(training_hparams):
        return 0
    max_gaussians = max(int(getattr(training_hparams, "max_gaussians", 0)), 0)
    if max_gaussians <= 0:
        return 0
    current_count = max(int(splat_count), 0)
    if current_count <= 0:
        return 0
    target_count = _resolve_refinement_target_splat_count(training_hparams, step)
    if target_count <= 0:
        return 0
    prune_ratio = resolve_refinement_prune_ratio(training_hparams, current_count, step)
    survivor_count = _estimate_refinement_survivor_count(current_count, prune_ratio)
    growth_cap_ratio = max(float(getattr(training_hparams, "refinement_max_growth_per_step", 0.15)), 0.0)
    growth_cap = int(math.ceil(float(survivor_count) * growth_cap_ratio))
    target_clones = min(max(target_count - survivor_count, 0), growth_cap)
    max_gaussians = max(int(getattr(training_hparams, "max_gaussians", 0)), 0)
    if max_gaussians > 0:
        target_clones = min(target_clones, max(max_gaussians - survivor_count, 0))
    return max(int(target_clones), 0)
