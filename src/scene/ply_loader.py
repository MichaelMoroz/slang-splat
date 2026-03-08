from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from plyfile import PlyData

from .gaussian_scene import GaussianScene

SH_C0 = 0.28209479177387814
_sigmoid = lambda values: 1.0 / (1.0 + np.exp(-values))


def _sorted_props(names: Iterable[str], prefix: str) -> list[str]:
    filtered = [name for name in names if name.startswith(prefix)]
    if not filtered:
        return []

    def to_index(name: str) -> int:
        suffix = name.split("_")[-1]
        if suffix.isdigit():
            return int(suffix)
        digits = "".join(ch for ch in suffix if ch.isdigit())
        return int(digits) if digits else 0

    return sorted(filtered, key=to_index)


def load_gaussian_ply(path: str | Path) -> GaussianScene:
    ply_data = PlyData.read(str(path))
    if len(ply_data.elements) == 0:
        raise ValueError("PLY file does not contain any elements.")
    vertex = ply_data.elements[0]
    n = int(vertex.count)

    positions = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )
    opacities = _sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))

    property_names = [prop.name for prop in vertex.properties]

    scale_names = _sorted_props(property_names, "scale_")
    if scale_names:
        raw_scales = np.stack([np.asarray(vertex[name], dtype=np.float32) for name in scale_names], axis=1)
        scales = np.exp(raw_scales)
    else:
        scales = np.ones((n, 3), dtype=np.float32)

    rot_names = _sorted_props(property_names, "rot")
    if rot_names:
        rotations = np.stack([np.asarray(vertex[name], dtype=np.float32) for name in rot_names], axis=1)
        if rotations.shape[1] != 4:
            raise ValueError(f"Expected 4 rotation components, found {rotations.shape[1]}.")
        rotations = rotations / np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-8)
    else:
        rotations = np.zeros((n, 4), dtype=np.float32)
        rotations[:, 0] = 1.0

    f_dc = np.stack(
        [
            np.asarray(vertex["f_dc_0"], dtype=np.float32),
            np.asarray(vertex["f_dc_1"], dtype=np.float32),
            np.asarray(vertex["f_dc_2"], dtype=np.float32),
        ],
        axis=1,
    )

    rest_names = _sorted_props(property_names, "f_rest_")
    if rest_names:
        if len(rest_names) % 3 != 0:
            raise ValueError("Expected a multiple-of-3 number of f_rest_* properties.")
        component_count = 1 + len(rest_names) // 3
        sh_coeffs = np.zeros((n, component_count, 3), dtype=np.float32)
        sh_coeffs[:, 0, :] = f_dc
        rest_data = np.stack([np.asarray(vertex[name], dtype=np.float32) for name in rest_names], axis=1)
        sh_coeffs[:, 1:, :] = rest_data.reshape(n, 3, component_count - 1).transpose(0, 2, 1)
    else:
        sh_coeffs = f_dc[:, None, :]

    colors = np.clip(0.5 + SH_C0 * sh_coeffs[:, 0, :], 0.0, 1.0)
    return GaussianScene(
        positions=positions,
        scales=scales.astype(np.float32),
        rotations=rotations.astype(np.float32),
        opacities=opacities.astype(np.float32),
        colors=colors.astype(np.float32),
        sh_coeffs=sh_coeffs.astype(np.float32),
    )
