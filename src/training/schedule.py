from __future__ import annotations

import math
from typing import Any

from .defaults import DEFAULT_LR_SCHEDULE_STEPS, DEFAULT_LR_STAGE1_STEP, DEFAULT_LR_STAGE2_STEP, DEFAULT_REFINEMENT_MIN_CONTRIBUTION_DECAY

_SCHEDULE_REFERENCE_STEPS = DEFAULT_LR_SCHEDULE_STEPS
_DEFAULT_MAX_SH_BAND = 3


def _schedule_duration(training_hparams: Any) -> int:
    return max(int(getattr(training_hparams, "lr_schedule_steps", _SCHEDULE_REFERENCE_STEPS)), 1)


def _clamp_schedule_step(step: int, max_step: int) -> int:
    return min(max(int(step), 0), max(int(max_step), 0))


def _step_progress(step: int, max_step: int) -> float:
    return _clamp_schedule_step(step, max_step) / float(max(max_step, 1))


def _schedule_progress(training_hparams: Any, step: int) -> float:
    return min(max(int(step), 0), _schedule_duration(training_hparams)) / float(_schedule_duration(training_hparams))


def resolve_lr_schedule_breakpoints(training_hparams: Any) -> tuple[int, int, int]:
    max_step = _schedule_duration(training_hparams)
    stage1 = _clamp_schedule_step(getattr(training_hparams, "lr_schedule_stage1_step", DEFAULT_LR_STAGE1_STEP), max_step)
    stage2 = max(stage1, _clamp_schedule_step(getattr(training_hparams, "lr_schedule_stage2_step", DEFAULT_LR_STAGE2_STEP), max_step))
    return stage1, stage2, max_step


def resolve_stage_schedule_steps(training_hparams: Any) -> tuple[int, int, int]:
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


def _stage_progress_milestones(training_hparams: Any) -> tuple[float, float, float]:
    stage1, stage2, stage3 = resolve_stage_schedule_steps(training_hparams)
    return _step_progress(stage1, stage3), _step_progress(stage2, stage3), 1.0


def _resolve_staged_linear_value(training_hparams: Any, step: int, initial_value: float, stage_values: tuple[float, float, float]) -> float:
    stage1_progress, stage2_progress, stage3_progress = _stage_progress_milestones(training_hparams)
    milestones = (
        (0.0, float(initial_value)),
        (stage1_progress, float(stage_values[0])),
        (stage2_progress, float(stage_values[1])),
        (stage3_progress, float(stage_values[2])),
    )
    return _piecewise_linear_schedule(_schedule_progress(training_hparams, step), milestones)


def resolve_base_learning_rate(training_hparams: Any, step: int) -> float:
    enabled = bool(getattr(training_hparams, "lr_schedule_enabled", True))
    start = max(float(getattr(training_hparams, "lr_schedule_start_lr", 1e-3)), 1e-8)
    if not enabled:
        return start
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        start,
        (
            max(float(getattr(training_hparams, "lr_schedule_stage1_lr", 0.002)), 1e-8),
            max(float(getattr(training_hparams, "lr_schedule_stage2_lr", 0.001)), 1e-8),
            max(float(getattr(training_hparams, "lr_schedule_end_lr", 1.5e-4)), 1e-8),
        ),
    )


def resolve_cosine_base_learning_rate(training_hparams: Any, step: int) -> float:
    return resolve_base_learning_rate(training_hparams, step)


def resolve_learning_rate_scale(training_hparams: Any, step: int) -> float:
    start = max(float(getattr(training_hparams, "lr_schedule_start_lr", 1e-3)), 1e-8)
    return resolve_base_learning_rate(training_hparams, step) / start


def resolve_position_lr_mul(training_hparams: Any, step: int) -> float:
    start = max(float(getattr(training_hparams, "lr_pos_mul", 1.0)), 1e-8)
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return start
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        start,
        (
            max(float(getattr(training_hparams, "lr_pos_stage1_mul", 1.0)), 1e-8),
            max(float(getattr(training_hparams, "lr_pos_stage2_mul", 1.0)), 1e-8),
            max(float(getattr(training_hparams, "lr_pos_stage3_mul", 1.0)), 1e-8),
        ),
    )


def resolve_sh_lr_mul(training_hparams: Any, step: int) -> float:
    start = max(float(getattr(training_hparams, "lr_sh_mul", 1.0)), 1e-8)
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return start
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        start,
        (
            max(float(getattr(training_hparams, "lr_sh_stage1_mul", 1.0)), 1e-8),
            max(float(getattr(training_hparams, "lr_sh_stage2_mul", 1.0)), 1e-8),
            max(float(getattr(training_hparams, "lr_sh_stage3_mul", 1.0)), 1e-8),
        ),
    )


def resolve_depth_ratio_weight(training_hparams: Any, step: int) -> float:
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return max(float(getattr(training_hparams, "depth_ratio_weight", 0.05)), 0.0)
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        max(float(getattr(training_hparams, "depth_ratio_weight", 0.05)), 0.0),
        (
            max(float(getattr(training_hparams, "depth_ratio_stage1_weight", 0.05)), 0.0),
            max(float(getattr(training_hparams, "depth_ratio_stage2_weight", 0.01)), 0.0),
            max(float(getattr(training_hparams, "depth_ratio_stage3_weight", 0.001)), 0.0),
        ),
    )


def resolve_ssim_weight(training_hparams: Any, step: int) -> float:
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return min(max(float(getattr(training_hparams, "ssim_weight", 0.05)), 0.0), 1.0)
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        min(max(float(getattr(training_hparams, "ssim_weight", 0.05)), 0.0), 1.0),
        (
            min(max(float(getattr(training_hparams, "ssim_weight_stage1", 0.1)), 0.0), 1.0),
            min(max(float(getattr(training_hparams, "ssim_weight_stage2", 0.3)), 0.0), 1.0),
            min(max(float(getattr(training_hparams, "ssim_weight_stage3", 0.4)), 0.0), 1.0),
        ),
    )


def resolve_max_visible_angle_deg(training_hparams: Any, step: int) -> float:
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return min(max(float(getattr(training_hparams, "max_visible_angle_deg", 1.0)), 1e-8), 89.999)
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        min(max(float(getattr(training_hparams, "max_visible_angle_deg", 1.0)), 1e-8), 89.999),
        (
            min(max(float(getattr(training_hparams, "max_visible_angle_deg_stage1", 1.0)), 1e-8), 89.999),
            min(max(float(getattr(training_hparams, "max_visible_angle_deg_stage2", 1.0)), 1e-8), 89.999),
            min(max(float(getattr(training_hparams, "max_visible_angle_deg_stage3", 1.0)), 1e-8), 89.999),
        ),
    )


def resolve_position_random_step_noise_lr(training_hparams: Any, step: int) -> float:
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return max(float(getattr(training_hparams, "position_random_step_noise_lr", 5e5)), 0.0)
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        max(float(getattr(training_hparams, "position_random_step_noise_lr", 5e5)), 0.0),
        (
            max(float(getattr(training_hparams, "position_random_step_noise_stage1_lr", 0.0)), 0.0),
            max(float(getattr(training_hparams, "position_random_step_noise_stage2_lr", 0.0)), 0.0),
            max(float(getattr(training_hparams, "position_random_step_noise_stage3_lr", 0.0)), 0.0),
        ),
    )


def _clamp_unit_interval(value: Any, default: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        resolved = float(default)
    return min(max(resolved, 0.0), 1.0)


def resolve_sorting_order_dithering(training_hparams: Any, step: int) -> float:
    start = _clamp_unit_interval(getattr(training_hparams, "sorting_order_dithering", 0.5), 0.5)
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return start
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        start,
        (
            _clamp_unit_interval(getattr(training_hparams, "sorting_order_dithering_stage1", 0.2), 0.2),
            _clamp_unit_interval(getattr(training_hparams, "sorting_order_dithering_stage2", 0.05), 0.05),
            _clamp_unit_interval(getattr(training_hparams, "sorting_order_dithering_stage3", 0.01), 0.01),
        ),
    )


def resolve_colorspace_mod(training_hparams: Any, step: int) -> float:
    start = max(float(getattr(training_hparams, "colorspace_mod", 0.5)), 1e-8)
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        return start
    return _resolve_staged_linear_value(
        training_hparams,
        step,
        start,
        (
            max(float(getattr(training_hparams, "colorspace_mod_stage1", 0.75)), 1e-8),
            max(float(getattr(training_hparams, "colorspace_mod_stage2", 0.9)), 1e-8),
            max(float(getattr(training_hparams, "colorspace_mod_stage3", 1.0)), 1e-8),
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


def resolve_sh_band(training_hparams: Any, step: int) -> int:
    if not bool(getattr(training_hparams, "lr_schedule_enabled", True)):
        if hasattr(training_hparams, "sh_band"):
            return _clamp_sh_band(getattr(training_hparams, "sh_band", 0), 0)
        return _resolve_legacy_sh_band(training_hparams, "use_sh", False)
    stage1, stage2, stage3 = resolve_stage_schedule_steps(training_hparams)
    current_step = max(int(step), 0)
    if current_step < stage1:
        return _clamp_sh_band(getattr(training_hparams, "sh_band", _resolve_legacy_sh_band(training_hparams, "use_sh", False)), 0)
    if current_step < stage2:
        return _clamp_sh_band(getattr(training_hparams, "sh_band_stage1", _resolve_legacy_sh_band(training_hparams, "use_sh_stage1", False)), _DEFAULT_MAX_SH_BAND)
    if current_step < stage3:
        return _clamp_sh_band(getattr(training_hparams, "sh_band_stage2", _resolve_legacy_sh_band(training_hparams, "use_sh_stage2", True)), _DEFAULT_MAX_SH_BAND)
    return _clamp_sh_band(getattr(training_hparams, "sh_band_stage3", _resolve_legacy_sh_band(training_hparams, "use_sh_stage3", True)), _DEFAULT_MAX_SH_BAND)


def resolve_use_sh(training_hparams: Any, step: int) -> bool:
    return resolve_sh_band(training_hparams, step) > 0


def resolve_max_allowed_density(training_hparams: Any, step: int) -> float:
    end = max(float(getattr(training_hparams, "max_allowed_density", 12.0)), 0.0)
    start = max(float(getattr(training_hparams, "max_allowed_density_start", 5.0)), 0.0)
    enabled = bool(getattr(training_hparams, "lr_schedule_enabled", True))
    duration = max(int(getattr(training_hparams, "lr_schedule_steps", 30_000)), 1)
    if not enabled or end <= start:
        return end if end <= start else start
    progress = min(max(int(step), 0), duration) / float(duration)
    return start + (end - start) * progress


def resolve_refinement_growth_ratio(training_hparams: Any, step: int) -> float:
    growth_ratio = max(float(getattr(training_hparams, "refinement_growth_ratio", 0.075)), 0.0)
    start_step = max(int(getattr(training_hparams, "refinement_growth_start_step", 500)), 0)
    return growth_ratio if int(step) >= start_step else 0.0


def resolve_refinement_min_contribution(training_hparams: Any, step: int, frame_count: int = 1) -> int:
    base_threshold = max(int(getattr(training_hparams, "refinement_min_contribution", 512)), 0)
    decay = min(max(float(getattr(training_hparams, "refinement_min_contribution_decay", DEFAULT_REFINEMENT_MIN_CONTRIBUTION_DECAY)), 0.0), 1.0)
    interval = resolve_effective_refinement_interval(training_hparams, frame_count)
    completed_refinement_steps = max(int(step), 0) // interval
    return max(int(round(base_threshold * math.pow(decay, completed_refinement_steps))), 0)


def resolve_effective_refinement_interval(training_hparams: Any, frame_count: int = 1) -> int:
    interval = max(int(getattr(training_hparams, "refinement_interval", 200)), 1)
    return max(interval, max(int(frame_count), 1))


def should_run_refinement_step(training_hparams: Any, step: int, frame_count: int = 1) -> bool:
    interval = resolve_effective_refinement_interval(training_hparams, frame_count)
    return int(step) > 0 and int(step) % interval == 0


def resolve_refinement_clone_budget(training_hparams: Any, splat_count: int, step: int = 0, frame_count: int = 1) -> int:
    growth_ratio = resolve_refinement_growth_ratio(training_hparams, step)
    max_gaussians = max(int(getattr(training_hparams, "max_gaussians", 0)), 0)
    if max_gaussians > 0 and int(splat_count) >= max_gaussians:
        return 0
    target_clones = int(math.ceil(float(max(int(splat_count), 0)) * growth_ratio))
    if max_gaussians > 0:
        target_clones = min(target_clones, max(max_gaussians - int(splat_count), 0))
    return max(int(target_clones), 0)
