from __future__ import annotations

import numpy as np
from plyfile import PlyData, PlyElement

from src.scene import load_gaussian_ply


def test_load_gaussian_ply_applies_basic_transforms(tmp_path):
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("opacity", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("f_rest_0", "f4"),
        ("f_rest_1", "f4"),
        ("f_rest_2", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
    rows = np.array(
        [
            (
                1.0,
                2.0,
                3.0,
                0.0,
                0.5,
                0.0,
                -0.5,
                1.0,
                2.0,
                3.0,
                0.0,
                1.0,
                -1.0,
                2.0,
                0.0,
                0.0,
                0.0,
            )
        ],
        dtype=dtype,
    )
    path = tmp_path / "scene.ply"
    PlyData([PlyElement.describe(rows, "vertex")], text=True).write(path)

    scene = load_gaussian_ply(path)
    assert scene.count == 1
    np.testing.assert_allclose(scene.positions[0], [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(scene.scales[0], [0.0, 1.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(scene.rotations[0]), 1.0, atol=1e-6)
    np.testing.assert_allclose(scene.opacities[0], 0.5, atol=1e-6)
    assert scene.sh_coeffs.shape == (1, 2, 3)
