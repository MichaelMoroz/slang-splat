from __future__ import annotations

import math
from typing import Any

DEFAULT_REFINEMENT_CONTRIBUTION_CULL_DECAY = 0.95


def resolve_cosine_base_learning_rate(training_hparams: Any, step: int) -> float:
    enabled = bool(getattr(training_hparams, "lr_schedule_enabled", True))
    start = max(float(getattr(training_hparams, "lr_schedule_start_lr", 1e-3)), 1e-8)
    end = max(float(getattr(training_hparams, "lr_schedule_end_lr", 1e-4)), 1e-8)
    duration = max(int(getattr(training_hparams, "lr_schedule_steps", 30_000)), 1)
    if not enabled:
        return start
    progress = min(max(int(step), 0), duration) / float(duration)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def resolve_learning_rate_scale(training_hparams: Any, step: int) -> float:
    start = max(float(getattr(training_hparams, "lr_schedule_start_lr", 1e-3)), 1e-8)
    return resolve_cosine_base_learning_rate(training_hparams, step) / start


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
    growth_ratio = max(float(getattr(training_hparams, "refinement_growth_ratio", 0.02)), 0.0)
    start_step = max(int(getattr(training_hparams, "refinement_growth_start_step", 500)), 0)
    return growth_ratio if int(step) >= start_step else 0.0


def resolve_refinement_contribution_cull_threshold(training_hparams: Any, step: int, frame_count: int = 1) -> float:
    base_threshold = max(float(getattr(training_hparams, "refinement_contribution_cull_threshold", 0.001)), 0.0)
    decay = min(max(float(getattr(training_hparams, "refinement_contribution_cull_decay", DEFAULT_REFINEMENT_CONTRIBUTION_CULL_DECAY)), 0.0), 1.0)
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
