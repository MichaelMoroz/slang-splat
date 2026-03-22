from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from plyfile import PlyData, PlyElement

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utility.ply import load_gaussian_ply


def test_load_gaussian_ply_returns_module_layout(tmp_path: Path) -> None:
    rows = np.empty(
        (2,),
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("opacity", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ],
    )
    rows["x"] = [1.0, 4.0]
    rows["y"] = [2.0, 5.0]
    rows["z"] = [3.0, 6.0]
    rows["opacity"] = [0.0, 2.0]
    rows["f_dc_0"] = [0.1, 0.2]
    rows["f_dc_1"] = [0.2, 0.3]
    rows["f_dc_2"] = [0.3, 0.4]
    rows["scale_0"] = [0.4, 0.5]
    rows["scale_1"] = [0.6, 0.7]
    rows["scale_2"] = [0.8, 0.9]
    rows["rot_0"] = [1.0, 1.0]
    rows["rot_1"] = [0.0, 0.0]
    rows["rot_2"] = [0.0, 0.0]
    rows["rot_3"] = [0.0, 0.0]
    path = tmp_path / "scene.ply"
    PlyData([PlyElement.describe(rows, "vertex")], text=False).write(path)

    splats = load_gaussian_ply(path)

    assert splats.shape == (14, 2)
    np.testing.assert_allclose(splats[0:3], np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32))
    np.testing.assert_allclose(splats[3:6], np.array([[0.4, 0.5], [0.6, 0.7], [0.8, 0.9]], dtype=np.float32))
    np.testing.assert_allclose(splats[6:10], np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32))
    assert np.all((splats[10:13] >= 0.0) & (splats[10:13] <= 1.0))
    assert np.all((splats[13] > 0.0) & (splats[13] < 1.0))
