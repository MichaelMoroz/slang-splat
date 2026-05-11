from __future__ import annotations

TARGET_ALPHA_MODE_OFF = 0
TARGET_ALPHA_MODE_SKIP_MASK = 1
TARGET_ALPHA_MODE_ALPHA_TARGET = 2
TARGET_ALPHA_MODE_LABELS = ("Off", "Skip Mask", "Alpha Training Target")


def resolve_target_alpha_mode(mode: object | None, legacy_use_target_alpha_mask: bool = False) -> int:
    if mode is None:
        return TARGET_ALPHA_MODE_SKIP_MASK if bool(legacy_use_target_alpha_mask) else TARGET_ALPHA_MODE_OFF
    if isinstance(mode, bool):
        return TARGET_ALPHA_MODE_SKIP_MASK if mode else TARGET_ALPHA_MODE_OFF
    try:
        resolved = int(mode)
    except (TypeError, ValueError):
        resolved = TARGET_ALPHA_MODE_SKIP_MASK if bool(legacy_use_target_alpha_mask) else TARGET_ALPHA_MODE_OFF
    return resolved if resolved in (TARGET_ALPHA_MODE_OFF, TARGET_ALPHA_MODE_SKIP_MASK, TARGET_ALPHA_MODE_ALPHA_TARGET) else (TARGET_ALPHA_MODE_SKIP_MASK if bool(legacy_use_target_alpha_mask) else TARGET_ALPHA_MODE_OFF)


def target_alpha_skip_mask_enabled(mode: int) -> bool:
    return int(mode) == TARGET_ALPHA_MODE_SKIP_MASK


def target_alpha_l1_enabled(mode: int) -> bool:
    return int(mode) == TARGET_ALPHA_MODE_ALPHA_TARGET