from __future__ import annotations

import math
from typing import Any

DEFAULT_REFINEMENT_MIN_CONTRIBUTION_DECAY = 0.995
_SCHEDULE_REFERENCE_STEPS = 30_000
_DEFAULT_LR_STAGE1_STEP = 2000
_DEFAULT_LR_STAGE2_STEP = 5000


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
    stage1 = _clamp_schedule_step(getattr(training_hparams, "lr_schedule_stage1_step", _DEFAULT_LR_STAGE1_STEP), max_step)
    stage2 = max(stage1, _clamp_schedule_step(getattr(training_hparams, "lr_schedule_stage2_step", _DEFAULT_LR_STAGE2_STEP), max_step))
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
            max(float(getattr(training_hparams, "lr_schedule_end_lr", 1e-4)), 1e-8),
        ),
    )


def resolve_cosine_base_learning_rate(training_hparams: Any, step: int) -> float:
    return resolve_base_learning_rate(training_hparams, step)


def resolve_learning_rate_scale(training_hparams: Any, step: int) -> float:
    start = max(float(getattr(training_hparams, "lr_schedule_start_lr", 1e-3)), 1e-8)
    return resolve_base_learning_rate(training_hparams, step) / start


def resolve_depth_ratio_weight(training_hparams: Any, step: int) -> float:
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


def resolve_position_random_step_noise_lr(training_hparams: Any, step: int) -> float:
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


def _resolve_stage_bool(training_hparams: Any, step: int, keys: tuple[str, str, str], defaults: tuple[bool, bool, bool]) -> bool:
    stage1, stage2, _ = resolve_stage_schedule_steps(training_hparams)
    values = tuple(bool(getattr(training_hparams, key, default)) for key, default in zip(keys, defaults))
    current_step = max(int(step), 0)
    if current_step < stage1:
        return values[0]
    if current_step < stage2:
        return values[1]
    return values[2]


def resolve_use_sh(training_hparams: Any, step: int) -> bool:
    return bool(getattr(training_hparams, "use_sh", True)) and _resolve_stage_bool(
        training_hparams,
        step,
        ("use_sh_stage1", "use_sh_stage2", "use_sh_stage3"),
        (False, False, True),
    )


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


def resolve_refinement_min_contribution_percent(training_hparams: Any, step: int, frame_count: int = 1) -> float:
    base_threshold = max(float(getattr(training_hparams, "refinement_min_contribution_percent", 1e-05)), 0.0)
    decay = min(max(float(getattr(training_hparams, "refinement_min_contribution_decay", DEFAULT_REFINEMENT_MIN_CONTRIBUTION_DECAY)), 0.0), 1.0)
    interval = resolve_effective_refinement_interval(training_hparams, frame_count)
    completed_refinement_steps = max(int(step), 0) // interval
    return base_threshold * math.pow(decay, completed_refinement_steps)


def resolve_effective_refinement_interval(training_hparams: Any, frame_count: int = 1) -> int:
    interval = max(int(getattr(training_hparams, "refinement_interval", 200)), 1)
    return max(interval, max(int(frame_count), 1))


def should_run_refinement_step(training_hparams: Any, step: int, frame_count: int = 1) -> bool:
    interval = resolve_effective_refinement_interval(training_hparams, frame_count)
    return int(step) > 0 and int(step) % interval == 0


def resolve_clone_probability_threshold(training_hparams: Any, splat_count: int, pixel_count: int, step: int = 0, frame_count: int = 1) -> float:
    interval = resolve_effective_refinement_interval(training_hparams, frame_count)
    growth_ratio = resolve_refinement_growth_ratio(training_hparams, step)
    pixels = max(int(pixel_count), 1)
    max_gaussians = max(int(getattr(training_hparams, "max_gaussians", 0)), 0)
    if max_gaussians > 0 and int(splat_count) >= max_gaussians:
        return 0.0
    target_clones = float(splat_count) * growth_ratio
    if max_gaussians > 0:
        target_clones = min(target_clones, float(max(max_gaussians - int(splat_count), 0)))
    return min(max(target_clones / float(interval * pixels), 0.0), 1.0)
