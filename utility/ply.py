from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from plyfile import PlyData

_SH_C0 = 0.28209479177387814
_LOGIT_EPS = 1e-6


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _sorted_props(names: Iterable[str], prefix: str) -> list[str]:
    props = [name for name in names if name.startswith(prefix)]
    if not props:
        return []

    def _index(name: str) -> int:
        suffix = name.split("_")[-1]
        if suffix.isdigit():
            return int(suffix)
        digits = "".join(ch for ch in suffix if ch.isdigit())
        return int(digits) if digits else 0

    return sorted(props, key=_index)


def load_gaussian_ply(path: str | Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    if not ply.elements:
        raise ValueError("PLY file does not contain any elements.")
    vertex = ply.elements[0]
    count = int(vertex.count)
    names = [prop.name for prop in vertex.properties]

    splats = np.zeros((14, count), dtype=np.float32)
    splats[0] = np.asarray(vertex["x"], dtype=np.float32)
    splats[1] = np.asarray(vertex["y"], dtype=np.float32)
    splats[2] = np.asarray(vertex["z"], dtype=np.float32)

    scale_names = _sorted_props(names, "scale_")
    if scale_names:
        if len(scale_names) != 3:
            raise ValueError(f"Expected 3 scale components, found {len(scale_names)}.")
        splats[3:6] = np.stack([np.asarray(vertex[name], dtype=np.float32) for name in scale_names], axis=0)

    rot_names = _sorted_props(names, "rot")
    if rot_names:
        if len(rot_names) != 4:
            raise ValueError(f"Expected 4 rotation components, found {len(rot_names)}.")
        rotation = np.stack([np.asarray(vertex[name], dtype=np.float32) for name in rot_names], axis=0)
        rotation /= np.maximum(np.linalg.norm(rotation, axis=0, keepdims=True), 1e-8)
        splats[6:10] = rotation
    else:
        splats[6] = 1.0

    f_dc = np.stack(
        [
            np.asarray(vertex["f_dc_0"], dtype=np.float32),
            np.asarray(vertex["f_dc_1"], dtype=np.float32),
            np.asarray(vertex["f_dc_2"], dtype=np.float32),
        ],
        axis=0,
    )
    splats[10:13] = np.clip(0.5 + _SH_C0 * f_dc, 0.0, 1.0)
    splats[13] = _sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))
    return np.ascontiguousarray(splats)
