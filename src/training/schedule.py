from __future__ import annotations

import math
from typing import Any


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


def should_run_maintenance_step(training_hparams: Any, step: int) -> bool:
    interval = max(int(getattr(training_hparams, "maintenance_interval", 200)), 1)
    return int(step) > 0 and int(step) % interval == 0


def resolve_clone_probability_threshold(training_hparams: Any, splat_count: int, pixel_count: int) -> float:
    interval = max(int(getattr(training_hparams, "maintenance_interval", 200)), 1)
    growth_ratio = max(float(getattr(training_hparams, "maintenance_growth_ratio", 0.05)), 0.0)
    pixels = max(int(pixel_count), 1)
    max_gaussians = max(int(getattr(training_hparams, "max_gaussians", 0)), 0)
    if max_gaussians > 0 and int(splat_count) >= max_gaussians:
        return 0.0
    target_clones = float(splat_count) * growth_ratio
    if max_gaussians > 0:
        target_clones = min(target_clones, float(max(max_gaussians - int(splat_count), 0)))
    return min(max(target_clones / float(interval * pixels), 0.0), 1.0)