from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

TRAINING_PROFILE_AUTO = "auto"
TRAINING_PROFILE_LEGACY = "legacy"
TRAINING_PROFILE_BICYCLE_IMAGES4_PSNR = "bicycle-images4-psnr"

_BICYCLE_DATASET_NAME = "bicycle"
_BICYCLE_IMAGES_SUBDIR = "images_4"
_BICYCLE_INIT_OPACITY = 0.1
_BICYCLE_MAX_GAUSSIANS = 200_000
_BICYCLE_DENSIFY_FROM_ITER = 500
_BICYCLE_DENSIFY_UNTIL_ITER = 15_000
_BICYCLE_DENSIFICATION_INTERVAL = 100
_BICYCLE_DENSIFY_GRAD_THRESHOLD = 1.5e-4
_BICYCLE_PERCENT_DENSE = 0.01
_PAPER_POSITION_LR = 1.6e-4
_PAPER_SCALE_LR = 5e-3
_PAPER_ROTATION_LR = 1e-3
_PAPER_COLOR_LR = 2.5e-3
_PAPER_OPACITY_LR = 5e-2


@dataclass(frozen=True, slots=True)
class TrainingProfile:
    name: str
    adam_overrides: dict[str, float] = field(default_factory=dict)
    stability_overrides: dict[str, float] = field(default_factory=dict)
    training_overrides: dict[str, float | int | bool | tuple[float, float, float]] = field(default_factory=dict)
    init_opacity_override: float | None = None


_PROFILES = {
    TRAINING_PROFILE_LEGACY: TrainingProfile(name=TRAINING_PROFILE_LEGACY),
    TRAINING_PROFILE_BICYCLE_IMAGES4_PSNR: TrainingProfile(
        name=TRAINING_PROFILE_BICYCLE_IMAGES4_PSNR,
        adam_overrides={
            "position_lr": _PAPER_POSITION_LR,
            "scale_lr": _PAPER_SCALE_LR,
            "rotation_lr": _PAPER_ROTATION_LR,
            "color_lr": _PAPER_COLOR_LR,
            "opacity_lr": _PAPER_OPACITY_LR,
        },
        training_overrides={
            "scale_l2_weight": 0.0,
            "opacity_reg_weight": 0.0,
            "lambda_dssim": 0.0,
            "mcmc_position_noise_enabled": False,
            "max_gaussians": _BICYCLE_MAX_GAUSSIANS,
            "densify_from_iter": _BICYCLE_DENSIFY_FROM_ITER,
            "densify_until_iter": _BICYCLE_DENSIFY_UNTIL_ITER,
            "densification_interval": _BICYCLE_DENSIFICATION_INTERVAL,
            "densify_grad_threshold": _BICYCLE_DENSIFY_GRAD_THRESHOLD,
            "percent_dense": _BICYCLE_PERCENT_DENSE,
            "screen_size_prune_threshold": 0.0,
            "world_size_prune_ratio": 0.0,
            "opacity_reset_interval": 0,
        },
        init_opacity_override=_BICYCLE_INIT_OPACITY,
    ),
}

TRAINING_PROFILE_CHOICES = tuple((TRAINING_PROFILE_AUTO, *tuple(_PROFILES.keys())))


def resolve_training_profile(
    profile_name: str | None,
    *,
    dataset_root: Path | None = None,
    images_subdir: str | None = None,
) -> TrainingProfile:
    normalized = TRAINING_PROFILE_AUTO if profile_name is None else str(profile_name).strip().lower()
    if normalized == TRAINING_PROFILE_AUTO:
        dataset_name = "" if dataset_root is None else dataset_root.resolve().name.lower()
        image_dir = "" if images_subdir is None else str(images_subdir).strip().lower()
        return (
            _PROFILES[TRAINING_PROFILE_BICYCLE_IMAGES4_PSNR]
            if dataset_name == _BICYCLE_DATASET_NAME and image_dir == _BICYCLE_IMAGES_SUBDIR
            else _PROFILES[TRAINING_PROFILE_LEGACY]
        )
    if normalized not in _PROFILES:
        raise ValueError(f"Unknown training profile '{profile_name}'. Expected one of {', '.join(TRAINING_PROFILE_CHOICES)}.")
    return _PROFILES[normalized]
