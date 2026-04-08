from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from plyfile import PlyData, PlyElement

from .gaussian_scene import GaussianScene
from .sh_utils import SH_C0, SUPPORTED_SH_COEFF_COUNT, pad_sh_coeffs, sh_coeffs_to_display_colors
_LOGIT_EPS = 1e-6


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _logit(values: np.ndarray) -> np.ndarray:
    clamped = np.clip(np.asarray(values, dtype=np.float32), _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return np.log(clamped / (1.0 - clamped))


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
        scales = np.stack([np.asarray(vertex[name], dtype=np.float32) for name in scale_names], axis=1)
    else:
        scales = np.zeros((n, 3), dtype=np.float32)

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

    sh_coeffs = pad_sh_coeffs(sh_coeffs, SUPPORTED_SH_COEFF_COUNT)
    colors = sh_coeffs_to_display_colors(sh_coeffs)
    return GaussianScene(
        positions=positions,
        scales=scales.astype(np.float32),
        rotations=rotations.astype(np.float32),
        opacities=opacities.astype(np.float32),
        colors=colors.astype(np.float32),
        sh_coeffs=sh_coeffs.astype(np.float32),
    )


def save_gaussian_ply(path: str | Path, scene: GaussianScene, *, include_sh: bool = True) -> Path:
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = int(scene.count)
    sh_coeffs = np.asarray(scene.sh_coeffs, dtype=np.float32)
    if sh_coeffs.ndim != 3 or sh_coeffs.shape[0] != count or sh_coeffs.shape[2] != 3 or sh_coeffs.shape[1] <= 0:
        raise ValueError("GaussianScene.sh_coeffs must have shape [count, coeff_count, 3] with coeff_count >= 1.")
    export_coeff_count = SUPPORTED_SH_COEFF_COUNT if include_sh else 1
    sh_coeffs = pad_sh_coeffs(sh_coeffs, export_coeff_count)
    rest_coeff_count = int(max(export_coeff_count - 1, 0))
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("opacity", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        *(("f_rest_" + str(index), "f4") for index in range(rest_coeff_count * 3)),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    rows = np.empty((count,), dtype=dtype)
    positions = np.asarray(scene.positions, dtype=np.float32)
    rows["x"], rows["y"], rows["z"] = positions[:, 0], positions[:, 1], positions[:, 2]
    rows["opacity"] = _logit(scene.opacities).astype(np.float32)
    rows["f_dc_0"], rows["f_dc_1"], rows["f_dc_2"] = sh_coeffs[:, 0, 0], sh_coeffs[:, 0, 1], sh_coeffs[:, 0, 2]
    if rest_coeff_count > 0:
        rest = np.transpose(sh_coeffs[:, 1:, :], (0, 2, 1)).reshape(count, rest_coeff_count * 3)
        for index in range(rest.shape[1]):
            rows[f"f_rest_{index}"] = rest[:, index]
    scales = np.asarray(scene.scales, dtype=np.float32)
    rows["scale_0"], rows["scale_1"], rows["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]
    rotations = np.asarray(scene.rotations, dtype=np.float32)
    rotations = rotations / np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-8)
    rows["rot_0"], rows["rot_1"], rows["rot_2"], rows["rot_3"] = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
    PlyData([PlyElement.describe(rows, "vertex")], text=False).write(output_path)
    return output_path
