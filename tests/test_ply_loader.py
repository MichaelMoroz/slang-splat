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
    assert scene.sh_coeffs.shape == (1, 16, 3)


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
    np.testing.assert_allclose(loaded.sh_coeffs[:, : scene.sh_coeffs.shape[1], :], scene.sh_coeffs, atol=1e-6)
    np.testing.assert_allclose(loaded.sh_coeffs[:, scene.sh_coeffs.shape[1] :, :], 0.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(loaded.rotations, axis=1), np.ones((scene.count,), dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(
        loaded.rotations,
        scene.rotations / np.maximum(np.linalg.norm(scene.rotations, axis=1, keepdims=True), 1e-8),
        atol=1e-6,
    )


def test_save_gaussian_ply_without_sh_omits_rest_properties(tmp_path) -> None:
    scene = GaussianScene(
        positions=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        scales=np.array([[0.0, 1.0, -1.0]], dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.2], dtype=np.float32),
        colors=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        sh_coeffs=np.array([[[0.5, 0.0, -0.5], [1.0, 2.0, 3.0]]], dtype=np.float32),
    )

    path = tmp_path / "exported_scene_dc_only.ply"
    saved_path = save_gaussian_ply(path, scene, include_sh=False)
    ply = PlyData.read(saved_path)
    property_names = tuple(prop.name for prop in ply.elements[0].properties)
    loaded = load_gaussian_ply(saved_path)

    assert saved_path == path.resolve()
    assert property_names == (
        "x",
        "y",
        "z",
        "opacity",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    )
    np.testing.assert_allclose(loaded.sh_coeffs[:, 0, :], scene.sh_coeffs[:, 0, :], atol=1e-6)
    np.testing.assert_allclose(loaded.sh_coeffs[:, 1:, :], 0.0, atol=1e-6)


def _write_point_cloud_ply(path, positions, colors=None, color_fields=("red", "green", "blue"), color_dtype="u1"):
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if colors is not None:
        fields += [(name, color_dtype) for name in color_fields]
    rows = np.empty((positions.shape[0],), dtype=fields)
    rows["x"], rows["y"], rows["z"] = positions[:, 0], positions[:, 1], positions[:, 2]
    if colors is not None:
        for index, name in enumerate(color_fields):
            rows[name] = colors[:, index]
    PlyData([PlyElement.describe(rows, "vertex")], text=False).write(str(path))


def test_load_point_cloud_ply_auto_detects_and_builds_opaque_splats(tmp_path):
    rng = np.random.default_rng(0)
    positions = rng.uniform(-2.0, 2.0, size=(50, 3)).astype(np.float32)
    colors = np.array([[255, 128, 0]] * 50, dtype=np.uint8)
    path = tmp_path / "cloud.ply"
    _write_point_cloud_ply(path, positions, colors)

    scene = load_gaussian_ply(path)

    assert scene.count == 50
    np.testing.assert_array_equal(scene.positions, positions)
    # Fully opaque, identity rotation, DC-only SH (no padding to 16 coeffs).
    np.testing.assert_allclose(scene.opacities, 1.0)
    np.testing.assert_allclose(scene.rotations, np.tile([1.0, 0.0, 0.0, 0.0], (50, 1)))
    assert scene.sh_coeffs.shape == (50, 1, 3)
    # 8-bit colors decoded to [0, 1].
    np.testing.assert_allclose(scene.colors[0], np.array([1.0, 0.502, 0.0]), atol=2e-3)
    # Isotropic, finite scales.
    assert np.all(scene.scales[:, 0] == scene.scales[:, 1]) and np.all(scene.scales[:, 0] == scene.scales[:, 2])
    assert np.all(np.isfinite(scene.scales))


def test_load_point_cloud_ply_scales_track_local_density(tmp_path):
    # A dense cluster plus a far-apart sparse set: nearest-neighbor scaling must give the
    # dense points smaller splats than the sparse ones (no single global scale).
    rng = np.random.default_rng(1)
    dense = rng.normal(0.0, 0.02, size=(200, 3)).astype(np.float32)
    sparse = rng.uniform(-5.0, 5.0, size=(20, 3)).astype(np.float32) + 50.0
    positions = np.concatenate([dense, sparse], axis=0)
    path = tmp_path / "density.ply"
    _write_point_cloud_ply(path, positions, colors=None)

    scene = load_gaussian_ply(path)
    linear_scales = np.exp(scene.scales[:, 0])

    assert np.median(linear_scales[:200]) < np.median(linear_scales[200:])


def test_load_point_cloud_ply_without_colors_defaults_to_gray(tmp_path):
    positions = np.zeros((4, 3), dtype=np.float32)
    positions[:, 0] = np.arange(4)
    path = tmp_path / "nocolor.ply"
    _write_point_cloud_ply(path, positions, colors=None)

    scene = load_gaussian_ply(path)

    assert scene.count == 4
    np.testing.assert_allclose(scene.colors, 0.5)


def test_load_point_cloud_ply_float_colors_not_rescaled(tmp_path):
    positions = np.zeros((3, 3), dtype=np.float32)
    colors = np.array([[0.1, 0.2, 0.3]] * 3, dtype=np.float32)
    path = tmp_path / "floatcolor.ply"
    _write_point_cloud_ply(path, positions, colors, color_fields=("r", "g", "b"), color_dtype="f4")

    scene = load_gaussian_ply(path)

    np.testing.assert_allclose(scene.colors, colors, atol=1e-6)
