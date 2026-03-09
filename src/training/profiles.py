from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

TRAINING_PROFILE_AUTO = "auto"
TRAINING_PROFILE_LEGACY = "legacy"


@dataclass(frozen=True, slots=True)
class TrainingProfile:
    name: str
    adam_overrides: dict[str, float] = field(default_factory=dict)
    stability_overrides: dict[str, float] = field(default_factory=dict)
    training_overrides: dict[str, float | int | bool | tuple[float, float, float]] = field(default_factory=dict)
    init_opacity_override: float | None = None


_PROFILES = {
    TRAINING_PROFILE_LEGACY: TrainingProfile(name=TRAINING_PROFILE_LEGACY),
}

TRAINING_PROFILE_CHOICES = (TRAINING_PROFILE_AUTO, TRAINING_PROFILE_LEGACY)


def resolve_training_profile(
    profile_name: str | None,
    *,
    dataset_root: Path | None = None,
    images_subdir: str | None = None,
) -> TrainingProfile:
    del dataset_root, images_subdir
    normalized = TRAINING_PROFILE_AUTO if profile_name is None else str(profile_name).strip().lower()
    if normalized == TRAINING_PROFILE_AUTO:
        return _PROFILES[TRAINING_PROFILE_LEGACY]
    if normalized not in _PROFILES:
        raise ValueError(f"Unknown training profile '{profile_name}'. Expected one of {', '.join(TRAINING_PROFILE_CHOICES)}.")
    return _PROFILES[normalized]
