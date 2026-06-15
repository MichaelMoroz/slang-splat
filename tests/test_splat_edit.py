from __future__ import annotations

import numpy as np
import pytest

from src.scene.gaussian_scene import GaussianScene
from src.scene.sh_utils import SUPPORTED_SH_COEFF_COUNT, rgb_to_sh0, sh_coeffs_to_display_colors
from src.scene import splat_edit


def _make_scene(count: int, *, seed: int = 0) -> GaussianScene:
    rng = np.random.default_rng(seed)
    sh = np.zeros((count, SUPPORTED_SH_COEFF_COUNT, 3), dtype=np.float32)
    sh[:, 0, :] = rng.uniform(-0.5, 0.5, size=(count, 3))
    colors = sh_coeffs_to_display_colors(sh)
    return GaussianScene(
        positions=rng.uniform(-2.0, 2.0, size=(count, 3)).astype(np.float32),
        scales=rng.uniform(-3.0, 0.0, size=(count, 3)).astype(np.float32),
        rotations=rng.normal(size=(count, 4)).astype(np.float32),
        opacities=rng.uniform(0.0, 1.0, size=(count,)).astype(np.float32),
        colors=colors,
        sh_coeffs=sh,
    )


def test_total_scale_is_linear_geometric_mean() -> None:
    scene = _make_scene(5)
    expected = np.exp(np.mean(scene.scales[:, :3], axis=1))
    np.testing.assert_allclose(splat_edit.total_scale(scene), expected, rtol=1e-6)


def test_select_in_range_inclusive_and_order_independent() -> None:
    values = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    mask = splat_edit.select_in_range(values, 0.5, 2.0)
    np.testing.assert_array_equal(mask, [False, True, True, True, False])
    # Swapped bounds behave identically.
    np.testing.assert_array_equal(mask, splat_edit.select_in_range(values, 2.0, 0.5))


def test_select_in_box_axis_aligned() -> None:
    scene = _make_scene(50, seed=3)
    center = np.array([0.0, 0.0, 0.0])
    half = np.array([1.0, 1.0, 1.0])
    mask = splat_edit.select_in_box(scene, center, half)
    expected = np.all(np.abs(scene.positions) <= 1.0, axis=1)
    np.testing.assert_array_equal(mask, expected)


def test_select_in_box_oriented_matches_axis_aligned_after_rotation() -> None:
    # A 45deg rotation about Z; points placed on the rotated axes should be selected.
    angle = np.pi / 4.0
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
    # Two points: one along the box's local +x (inside), one along world +x past the half-extent.
    local_points = np.array([[1.4, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    world_points = (rotation @ local_points.T).T
    scene = GaussianScene(
        positions=world_points,
        scales=np.zeros((2, 3), np.float32),
        rotations=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)).astype(np.float32),
        opacities=np.ones((2,), np.float32),
        colors=np.ones((2, 3), np.float32),
        sh_coeffs=np.zeros((2, SUPPORTED_SH_COEFF_COUNT, 3), np.float32),
    )
    half = np.array([1.5, 0.2, 0.2])
    mask = splat_edit.select_in_box(scene, np.zeros(3), half, rotation=rotation)
    np.testing.assert_array_equal(mask, [True, False])


def test_log10_histogram_counts_sum_to_sample_count() -> None:
    values = np.array([0.01, 0.1, 0.1, 1.0, 10.0, 10.0, 100.0])
    counts, edges = splat_edit.log10_histogram(values, bins=8)
    assert counts.sum() == values.size
    assert edges.shape[0] == 9
    assert edges[0] <= values.min() and edges[-1] >= values.max() - 1e-9


def test_log10_histogram_handles_nonpositive_and_empty() -> None:
    counts, edges = splat_edit.log10_histogram(np.array([0.0, -1.0, np.nan]), bins=4)
    assert counts.sum() == 0
    assert edges.shape[0] == 5


def test_resample_sparsify_removes_only_selected() -> None:
    scene = _make_scene(100, seed=1)
    mask = np.zeros(100, dtype=bool)
    mask[:40] = True
    new_scene, new_mask = splat_edit.resample_selection(scene, mask, 0.5, rng=np.random.default_rng(7))
    # 60 unselected kept + 20 of the selected retained.
    assert new_scene.count == 80
    assert int(new_mask.sum()) == 20
    # Unselected splats (originals 40..99) are preserved verbatim somewhere in the result.
    assert new_scene.count == int((~mask).sum()) + 20


def test_resample_densify_adds_selected_children() -> None:
    scene = _make_scene(20, seed=2)
    mask = np.zeros(20, dtype=bool)
    mask[:10] = True
    new_scene, new_mask = splat_edit.resample_selection(scene, mask, 2.0, rng=np.random.default_rng(9))
    assert new_scene.count == 30  # 20 original + 10 new children
    assert int(new_mask.sum()) == 20  # 10 selected originals + 10 children
    # Children are shrunk relative to their parents.
    child_scales = new_scene.scales[20:, :3]
    assert np.all(child_scales < scene.scales[:10, :3].max() + 1e-3)


def test_resample_noop_when_ratio_one_or_empty_selection() -> None:
    scene = _make_scene(10)
    mask = np.zeros(10, dtype=bool)
    mask[:5] = True
    same_scene, same_mask = splat_edit.resample_selection(scene, mask, 1.0)
    assert same_scene is scene
    np.testing.assert_array_equal(same_mask, mask)
    empty = np.zeros(10, dtype=bool)
    s2, m2 = splat_edit.resample_selection(scene, empty, 0.1)
    assert s2 is scene and not m2.any()


def test_edit_properties_sets_color_opacity_scale_on_selection_only() -> None:
    scene = _make_scene(8, seed=5)
    original = _make_scene(8, seed=5)
    mask = np.zeros(8, dtype=bool)
    mask[2:5] = True
    edited = splat_edit.edit_properties(
        scene, mask, color=(1.0, 0.0, 0.0), opacity=0.25, total_scale_value=0.5
    )
    # Selected opacity set exactly.
    np.testing.assert_allclose(edited.opacities[mask], 0.25, atol=1e-6)
    # Selected base color matches target (display space, clipped).
    np.testing.assert_allclose(edited.colors[mask], np.broadcast_to([1.0, 0.0, 0.0], (3, 3)), atol=1e-5)
    np.testing.assert_allclose(edited.sh_coeffs[mask, 0, :], np.broadcast_to(rgb_to_sh0(np.array([[1.0, 0.0, 0.0]]))[0], (3, 3)), atol=1e-5)
    # Selected geometric-mean scale equals target.
    np.testing.assert_allclose(splat_edit.total_scale(edited)[mask], 0.5, rtol=1e-5)
    # Unselected splats are untouched.
    np.testing.assert_array_equal(edited.opacities[~mask], original.opacities[~mask])
    np.testing.assert_array_equal(edited.scales[~mask], original.scales[~mask])
    np.testing.assert_array_equal(edited.colors[~mask], original.colors[~mask])


def test_edit_properties_preserves_anisotropy_ratio() -> None:
    scene = _make_scene(6, seed=11)
    mask = np.ones(6, dtype=bool)
    before_ratio = scene.scales[:, 0] - scene.scales[:, 1]  # log-space differences = ratios
    edited = splat_edit.edit_properties(scene, mask, total_scale_value=2.0)
    after_ratio = edited.scales[:, 0] - edited.scales[:, 1]
    np.testing.assert_allclose(after_ratio, before_ratio, atol=1e-5)


def test_mask_length_validation() -> None:
    scene = _make_scene(4)
    with pytest.raises(ValueError):
        splat_edit.resample_selection(scene, np.zeros(3, dtype=bool), 0.5)
    with pytest.raises(ValueError):
        splat_edit.edit_properties(scene, np.zeros(5, dtype=bool), opacity=0.5)
