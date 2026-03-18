from __future__ import annotations

import numpy as np
from plyfile import PlyData, PlyElement

from src.scene import GaussianScene, load_gaussian_ply, save_gaussian_ply


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


def test_save_gaussian_ply_round_trips_scene_data(tmp_path) -> None:
    scene = GaussianScene(
        positions=np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]], dtype=np.float32),
        scales=np.array([[0.0, 1.0, -1.0], [0.25, -0.5, 0.75]], dtype=np.float32),
        rotations=np.array([[2.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]], dtype=np.float32),
        opacities=np.array([0.2, 0.8], dtype=np.float32),
        colors=np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]], dtype=np.float32),
        sh_coeffs=np.array(
            [
                [[0.5, 0.0, -0.5], [1.0, 2.0, 3.0]],
                [[-0.25, 0.75, 0.5], [0.1, 0.2, 0.3]],
            ],
            dtype=np.float32,
        ),
    )

    path = tmp_path / "exported_scene.ply"
    saved_path = save_gaussian_ply(path, scene)
    loaded = load_gaussian_ply(saved_path)

    assert saved_path == path.resolve()
    np.testing.assert_allclose(loaded.positions, scene.positions, atol=1e-6)
    np.testing.assert_allclose(loaded.scales, scene.scales, atol=1e-6)
    np.testing.assert_allclose(loaded.opacities, scene.opacities, atol=2e-6)
    np.testing.assert_allclose(loaded.sh_coeffs, scene.sh_coeffs, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(loaded.rotations, axis=1), np.ones((scene.count,), dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(
        loaded.rotations,
        scene.rotations / np.maximum(np.linalg.norm(scene.rotations, axis=1, keepdims=True), 1e-8),
        atol=1e-6,
    )
