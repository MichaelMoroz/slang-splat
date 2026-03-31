from __future__ import annotations

import numpy as np

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SUPPORTED_SH_COEFF_COUNT = 4


def rgb_to_sh0(colors: np.ndarray) -> np.ndarray:
    values = np.asarray(colors, dtype=np.float32)
    return ((values - 0.5) / SH_C0).astype(np.float32, copy=False)


def pad_sh_coeffs(sh_coeffs: np.ndarray, coeff_count: int = SUPPORTED_SH_COEFF_COUNT) -> np.ndarray:
    values = np.asarray(sh_coeffs, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("sh_coeffs must have shape [count, coeff_count, 3].")
    target = max(int(coeff_count), 1)
    padded = np.zeros((values.shape[0], target, 3), dtype=np.float32)
    copy_count = min(values.shape[1], target)
    padded[:, :copy_count, :] = values[:, :copy_count, :]
    return padded

def resolve_supported_sh_coeffs(sh_coeffs: np.ndarray, colors: np.ndarray) -> np.ndarray:
    resolved = pad_sh_coeffs(sh_coeffs, SUPPORTED_SH_COEFF_COUNT)
    source = np.asarray(sh_coeffs, dtype=np.float32)
    if source.shape[1] >= SUPPORTED_SH_COEFF_COUNT:
        return resolved
    if source.shape[1] == 0 or np.all(np.abs(source[:, 0, :]) <= 1e-8):
        resolved[:, 0, :] = rgb_to_sh0(colors)
    return resolved


def sh_coeffs_to_display_colors(sh_coeffs: np.ndarray) -> np.ndarray:
    coeffs = pad_sh_coeffs(sh_coeffs, 1)
    return np.clip(0.5 + SH_C0 * coeffs[:, 0, :], 0.0, 1.0).astype(np.float32, copy=False)


def evaluate_sh0_sh1(sh_coeffs: np.ndarray, view_dirs: np.ndarray) -> np.ndarray:
    coeffs = pad_sh_coeffs(sh_coeffs)
    dirs = np.asarray(view_dirs, dtype=np.float32).reshape(coeffs.shape[0], 3)
    lengths = np.linalg.norm(dirs, axis=1, keepdims=True)
    safe_dirs = np.divide(dirs, np.maximum(lengths, 1e-8), out=np.zeros_like(dirs), where=lengths > 1e-8)
    return (
        0.5
        + SH_C0 * coeffs[:, 0, :]
        - SH_C1 * safe_dirs[:, 1:2] * coeffs[:, 1, :]
        + SH_C1 * safe_dirs[:, 2:3] * coeffs[:, 2, :]
        - SH_C1 * safe_dirs[:, 0:1] * coeffs[:, 3, :]
    ).astype(np.float32, copy=False)