"""Parity tests: the GPU splat-edit kernels must match the pure-numpy reference."""
from __future__ import annotations

import numpy as np

from src.renderer import GaussianRenderer
from src.scene import splat_edit
from src.scene.gaussian_scene import GaussianScene
from src.scene.sh_utils import SUPPORTED_SH_COEFF_COUNT, rgb_to_sh0, sh_coeffs_to_display_colors


def _scene(count: int, seed: int = 0) -> GaussianScene:
    rng = np.random.default_rng(seed)
    quats = rng.normal(size=(count, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    colors = rng.uniform(0.0, 1.0, size=(count, 3)).astype(np.float32)
    scene = GaussianScene(
        positions=rng.uniform(-2.0, 2.0, size=(count, 3)).astype(np.float32),
        scales=np.log(rng.uniform(0.01, 0.3, size=(count, 3))).astype(np.float32),
        rotations=quats,
        opacities=rng.uniform(0.05, 0.95, size=(count,)).astype(np.float32),
        colors=colors,
        sh_coeffs=np.zeros((count, SUPPORTED_SH_COEFF_COUNT, 3), dtype=np.float32),
    )
    scene.sh_coeffs[:, 0, :] = rgb_to_sh0(colors)
    return scene


def _renderer(device, scene: GaussianScene) -> GaussianRenderer:
    renderer = GaussianRenderer(device, width=64, height=64, allocate_training_work_buffers=False, allocate_grad_work_buffers=False)
    renderer.set_scene(scene)
    return renderer


def _mask(renderer: GaussianRenderer) -> np.ndarray:
    count = int(renderer._scene_count)
    return renderer._read_array(renderer._ensure_highlight_buffer(count), np.uint32, count).astype(bool)


def _euler_matrix(euler_deg) -> np.ndarray:
    ax, ay, az = np.deg2rad(euler_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return (rz @ ry @ rx).astype(np.float32)


def test_select_box_oriented_matches_reference(device) -> None:
    scene = _scene(400, seed=1)
    renderer = _renderer(device, scene)
    center = np.array([0.2, -0.1, 0.3], dtype=np.float32)
    half = np.array([0.8, 0.5, 1.1], dtype=np.float32)
    rot = _euler_matrix((20.0, -35.0, 50.0))
    renderer.edit_select_box(center, rot.T, half, renderer.SELECT_MODE_REPLACE)
    expected = splat_edit.select_in_box(scene, center, half, rotation=rot)
    np.testing.assert_array_equal(_mask(renderer), expected)


def test_select_range_matches_reference_all_scalars(device) -> None:
    scene = _scene(400, seed=2)
    renderer = _renderer(device, scene)
    for kind, gpu_kind, lo, hi in (
        ("scale", renderer.SELECT_SCALAR_SCALE, 0.05, 0.2),
        ("opacity", renderer.SELECT_SCALAR_OPACITY, 0.3, 0.7),
        ("color", renderer.SELECT_SCALAR_COLOR, 0.3, 0.6),
    ):
        renderer.edit_select_range(gpu_kind, lo, hi, renderer.SELECT_MODE_REPLACE)
        expected = splat_edit.select_in_range(splat_edit.selection_scalar(scene, kind), lo, hi)
        np.testing.assert_array_equal(_mask(renderer), expected)


def test_select_combine_modes(device) -> None:
    scene = _scene(200, seed=3)
    renderer = _renderer(device, scene)
    a = splat_edit.select_in_range(splat_edit.selection_scalar(scene, "opacity"), 0.0, 0.5)
    b = splat_edit.select_in_range(splat_edit.selection_scalar(scene, "opacity"), 0.3, 1.0)
    renderer.edit_select_range(renderer.SELECT_SCALAR_OPACITY, 0.0, 0.5, renderer.SELECT_MODE_REPLACE)
    renderer.edit_select_range(renderer.SELECT_SCALAR_OPACITY, 0.3, 1.0, renderer.SELECT_MODE_INTERSECT)
    np.testing.assert_array_equal(_mask(renderer), a & b)
    renderer.edit_select_range(renderer.SELECT_SCALAR_OPACITY, 0.0, 0.5, renderer.SELECT_MODE_REPLACE)
    renderer.edit_select_range(renderer.SELECT_SCALAR_OPACITY, 0.3, 1.0, renderer.SELECT_MODE_SUBTRACT)
    np.testing.assert_array_equal(_mask(renderer), a & ~b)


def test_edit_properties_matches_reference(device) -> None:
    scene = _scene(200, seed=4)
    renderer = _renderer(device, scene)
    sel = splat_edit.select_in_range(splat_edit.selection_scalar(scene, "opacity"), 0.2, 0.8)
    renderer.edit_select_range(renderer.SELECT_SCALAR_OPACITY, 0.2, 0.8, renderer.SELECT_MODE_REPLACE)
    renderer.edit_properties(color=(0.2, 0.5, 0.9), opacity=0.4, scale=0.05)
    expected = splat_edit.edit_properties(scene, sel, color=(0.2, 0.5, 0.9), opacity=0.4, total_scale_value=0.05)
    groups = renderer.read_scene_groups(renderer._scene_count)
    gpu_opacity = 1.0 / (1.0 + np.exp(-groups["color_alpha"][:, 3]))
    gpu_colors = sh_coeffs_to_display_colors(groups["sh_coeffs"])
    gpu_scale = np.exp(np.mean(groups["scales"][:, :3], axis=1))
    np.testing.assert_allclose(gpu_opacity, expected.opacities, atol=1e-3)
    np.testing.assert_allclose(gpu_colors, expected.colors, atol=2e-3)
    np.testing.assert_allclose(gpu_scale, splat_edit.total_scale(expected), rtol=2e-3)


def test_resample_zero_percent_deletes_selection(device) -> None:
    scene = _scene(300, seed=5)
    renderer = _renderer(device, scene)
    sel = splat_edit.select_in_range(splat_edit.selection_scalar(scene, "opacity"), 0.0, 0.5)
    renderer.edit_select_range(renderer.SELECT_SCALAR_OPACITY, 0.0, 0.5, renderer.SELECT_MODE_REPLACE)
    survivors = renderer.edit_resample(0.0, seed=1)
    assert survivors == scene.count - int(sel.sum())
    # Every surviving position must be one of the originally unselected splats.
    survivor_pos = renderer.read_scene_groups(survivors)["positions"][:, :3]
    unselected = {tuple(np.round(p, 5)) for p in scene.positions[~sel]}
    assert all(tuple(np.round(p, 5)) in unselected for p in survivor_pos)
    assert renderer.edit_selection_count() == 0


def test_resample_densify_adds_expected_children(device) -> None:
    scene = _scene(300, seed=6)
    renderer = _renderer(device, scene)
    sel = splat_edit.select_in_range(splat_edit.selection_scalar(scene, "scale"), 0.05, 0.3)
    n_sel = int(sel.sum())
    renderer.edit_select_range(renderer.SELECT_SCALAR_SCALE, 0.05, 0.3, renderer.SELECT_MODE_REPLACE)
    total = renderer.edit_resample(2.5, seed=2)
    assert total == scene.count + int(round(1.5 * n_sel))
    # Children (appended after the originals) are shrunk by the 3DGS split factor.
    child_scale = np.exp(renderer.read_scene_groups(total)["scales"][scene.count:, :3])
    assert 0.4 < child_scale.mean() / np.exp(scene.scales[sel]).mean() < 0.9
