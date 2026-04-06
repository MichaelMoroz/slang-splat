from __future__ import annotations

import math
from typing import Any

DEFAULT_REFINEMENT_MIN_CONTRIBUTION_DECAY = 0.995
_SCHEDULE_REFERENCE_STEPS = 30_000
_LR_MILESTONES = ((0.0, 1.0), (2000.0 / _SCHEDULE_REFERENCE_STEPS, 0.4), (5000.0 / _SCHEDULE_REFERENCE_STEPS, 0.2))
_DEPTH_RATIO_WEIGHT_MILESTONES = (
    (0.0, 1.0),
    (1000.0 / _SCHEDULE_REFERENCE_STEPS, 0.25),
    (2000.0 / _SCHEDULE_REFERENCE_STEPS, 0.05),
    (5000.0 / _SCHEDULE_REFERENCE_STEPS, 0.01),
    (1.0, 0.001),
)
_POSITION_RANDOM_STEP_NOISE_MILESTONES = ((0.0, 1.0), (1.0, 0.0))
_USE_SH_START_PROGRESS = 5000.0 / _SCHEDULE_REFERENCE_STEPS


def _schedule_duration(training_hparams: Any) -> int:
    return max(int(getattr(training_hparams, "lr_schedule_steps", _SCHEDULE_REFERENCE_STEPS)), 1)


def _schedule_progress(training_hparams: Any, step: int) -> float:
    return min(max(int(step), 0), _schedule_duration(training_hparams)) / float(_schedule_duration(training_hparams))


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


def resolve_base_learning_rate(training_hparams: Any, step: int) -> float:
    enabled = bool(getattr(training_hparams, "lr_schedule_enabled", True))
    start = max(float(getattr(training_hparams, "lr_schedule_start_lr", 1e-3)), 1e-8)
    end = max(float(getattr(training_hparams, "lr_schedule_end_lr", 1e-4)), 1e-8)
    if not enabled:
        return start
    end_ratio = end / start
    return start * _piecewise_linear_schedule(_schedule_progress(training_hparams, step), _LR_MILESTONES + ((1.0, end_ratio),))


def resolve_cosine_base_learning_rate(training_hparams: Any, step: int) -> float:
    return resolve_base_learning_rate(training_hparams, step)


def resolve_learning_rate_scale(training_hparams: Any, step: int) -> float:
    start = max(float(getattr(training_hparams, "lr_schedule_start_lr", 1e-3)), 1e-8)
    return resolve_base_learning_rate(training_hparams, step) / start


def resolve_depth_ratio_weight(training_hparams: Any, step: int) -> float:
    base_weight = max(float(getattr(training_hparams, "depth_ratio_weight", 0.05)), 0.0)
    return base_weight * _piecewise_linear_schedule(_schedule_progress(training_hparams, step), _DEPTH_RATIO_WEIGHT_MILESTONES)


def resolve_position_random_step_noise_lr(training_hparams: Any, step: int) -> float:
    base_noise_lr = max(float(getattr(training_hparams, "position_random_step_noise_lr", 5e5)), 0.0)
    return base_noise_lr * _piecewise_linear_schedule(_schedule_progress(training_hparams, step), _POSITION_RANDOM_STEP_NOISE_MILESTONES)


def resolve_use_sh(training_hparams: Any, step: int) -> bool:
    return bool(getattr(training_hparams, "use_sh", True)) and _schedule_progress(training_hparams, step) >= _USE_SH_START_PROGRESS


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
