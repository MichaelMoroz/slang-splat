from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.scene.gaussian_scene import GaussianScene
from src.scene.sh_utils import SUPPORTED_SH_COEFF_COUNT, sh_coeffs_to_display_colors
from src.scene import splat_edit
from src.viewer import splat_editor as ed


def _scene(count: int, seed: int = 0) -> GaussianScene:
    rng = np.random.default_rng(seed)
    sh = np.zeros((count, SUPPORTED_SH_COEFF_COUNT, 3), dtype=np.float32)
    sh[:, 0, :] = rng.uniform(-0.4, 0.4, size=(count, 3))
    return GaussianScene(
        positions=rng.uniform(-1.0, 1.0, size=(count, 3)).astype(np.float32),
        scales=rng.uniform(-3.0, 0.0, size=(count, 3)).astype(np.float32),
        rotations=np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)).astype(np.float32),
        opacities=rng.uniform(0.05, 0.95, size=(count,)).astype(np.float32),
        colors=sh_coeffs_to_display_colors(sh),
        sh_coeffs=sh,
    )


def test_camera_view_projection_matches_overlay_projection() -> None:
    # The gizmo's view/projection must reproduce the same world->screen mapping the
    # viewport overlays use, with ImGuizmo's NDC->pixel convention (NDC.y up).
    from src.renderer import Camera
    from src.viewer.presenter_state import _project_overlay_points

    camera = Camera.look_at(position=(1.5, 0.8, 3.0), target=(0.1, -0.2, 0.0), up=(0.0, 1.0, 0.0), near=0.1, far=50.0)
    width, height = 640, 480
    view, proj = ed.camera_view_projection_matrices(camera, width, height)
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.5, 1.5, size=(40, 3)).astype(np.float32)
    ref_screen, ref_valid = _project_overlay_points(camera, points, width, height)

    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    clip = (proj.astype(np.float64) @ (view.astype(np.float64) @ homogeneous.astype(np.float64).T)).T
    in_front = clip[:, 3] > 1e-6
    ndc = clip[in_front, :3] / clip[in_front, 3:4]
    gizmo_x = (ndc[:, 0] * 0.5 + 0.5) * width
    gizmo_y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
    ref = ref_screen[in_front]
    valid = ref_valid[in_front]
    np.testing.assert_allclose(gizmo_x[valid], ref[valid, 0], atol=1e-2)
    np.testing.assert_allclose(gizmo_y[valid], ref[valid, 1], atol=1e-2)


def test_box_model_matrix_round_trips_through_imguizmo_decompose() -> None:
    imguizmo = pytest.importorskip("imgui_bundle").imguizmo.im_guizmo
    state = ed.SplatEditorState()
    state.box_center = np.array([0.3, -0.4, 1.2], dtype=np.float32)
    state.box_half_extents = np.array([0.5, 0.8, 0.25], dtype=np.float32)
    state.box_rotation_euler = np.array([15.0, -30.0, 45.0], dtype=np.float32)
    model = ed.box_model_matrix(state)
    mat16 = imguizmo.Matrix16(list(model.flatten(order="F").astype(float)))
    comp = imguizmo.decompose_matrix_to_components(mat16)
    np.testing.assert_allclose(list(comp.translation.values), state.box_center, atol=1e-4)
    np.testing.assert_allclose(list(comp.scale.values), state.box_half_extents, atol=1e-4)
    np.testing.assert_allclose(list(comp.rotation.values), state.box_rotation_euler, atol=1e-3)


def _highlight_mask(renderer) -> np.ndarray:
    count = int(renderer._scene_count)
    flags = renderer._read_array(renderer._ensure_highlight_buffer(count), np.uint32, count)
    return flags.astype(bool)


def _loaded_viewer(device, scene: GaussianScene):
    from src.renderer import GaussianRenderer

    renderer = GaussianRenderer(device, width=64, height=64, allocate_training_work_buffers=False, allocate_grad_work_buffers=False)
    renderer.set_scene(scene)
    return SimpleNamespace(s=SimpleNamespace(scene=scene, renderer=renderer, trainer=None, training_renderer=None, splat_editor=None))


def test_box_selection_highlights_and_counts(device) -> None:
    scene = _scene(60, seed=1)
    viewer = _loaded_viewer(device, scene)
    ed.init_box_to_scene(viewer)
    state = ed.editor_state(viewer)
    state.box_half_extents = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    state.box_center = np.zeros(3, dtype=np.float32)
    ed.select_box(viewer, mode="replace")
    expected = np.all(np.abs(scene.positions) <= 0.5, axis=1)
    assert state.selected_count == int(expected.sum())
    np.testing.assert_array_equal(_highlight_mask(viewer.s.renderer), expected)
    assert bool(viewer.s.renderer._highlight_enabled) == bool(expected.any())


def test_range_selection_add_accumulates(device) -> None:
    scene = _scene(40, seed=2)
    viewer = _loaded_viewer(device, scene)
    ed.refresh_histograms(viewer, force=True)
    state = ed.editor_state(viewer)
    state.ranges[splat_edit.SELECT_OPACITY] = (0.0, 0.5)
    ed.select_range(viewer, splat_edit.SELECT_OPACITY, mode="replace")
    np.testing.assert_array_equal(_highlight_mask(viewer.s.renderer), scene.opacities <= 0.5)
    state.ranges[splat_edit.SELECT_SCALE] = (0.0, float("inf"))
    ed.select_range(viewer, splat_edit.SELECT_SCALE, mode="add")
    assert state.selected_count == scene.count  # scale range covers everything


def test_invert_and_clear(device) -> None:
    scene = _scene(20, seed=3)
    viewer = _loaded_viewer(device, scene)
    state = ed.editor_state(viewer)
    ed.init_box_to_scene(viewer)
    state.box_center = np.zeros(3, dtype=np.float32)
    state.box_half_extents = np.array([0.4, 0.4, 0.4], dtype=np.float32)
    ed.select_box(viewer, mode="replace")
    inside = int(np.all(np.abs(scene.positions) <= 0.4, axis=1).sum())
    ed.invert_selection(viewer)
    assert state.selected_count == scene.count - inside
    ed.clear_selection(viewer)
    assert state.selected_count == 0
    assert not bool(viewer.s.renderer._highlight_enabled)


def test_resample_sparsify_deletes_selection_at_zero_percent(device) -> None:
    scene = _scene(100, seed=4)
    viewer = _loaded_viewer(device, scene)
    state = ed.editor_state(viewer)
    ed.init_box_to_scene(viewer)
    state.box_center = np.zeros(3, dtype=np.float32)
    state.box_half_extents = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    selected = ed.select_box(viewer, mode="replace")
    state.resample_percent = 0.0
    assert ed.apply_resample(viewer) is True
    assert int(viewer.s.renderer._scene_count) == scene.count - selected
    assert state.selected_count == 0


def test_resample_densify_grows_scene(device) -> None:
    scene = _scene(100, seed=5)
    viewer = _loaded_viewer(device, scene)
    state = ed.editor_state(viewer)
    ed.init_box_to_scene(viewer)
    state.box_center = np.zeros(3, dtype=np.float32)
    state.box_half_extents = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    selected = ed.select_box(viewer, mode="replace")
    state.resample_percent = 200.0
    assert ed.apply_resample(viewer) is True
    assert int(viewer.s.renderer._scene_count) == scene.count + selected


def test_edit_properties_sets_opacity_on_selection_only(device) -> None:
    scene = _scene(50, seed=6)
    viewer = _loaded_viewer(device, scene)
    state = ed.editor_state(viewer)
    ed.init_box_to_scene(viewer)
    state.box_center = np.zeros(3, dtype=np.float32)
    state.box_half_extents = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    ed.select_box(viewer, mode="replace")
    sel = np.all(np.abs(scene.positions) <= 0.5, axis=1)
    assert ed.apply_edit_properties(viewer) is False  # nothing enabled
    state.edit_opacity_enabled = True
    state.edit_opacity = 0.3
    assert ed.apply_edit_properties(viewer) is True
    opacities = viewer.s.renderer.read_live_scene().opacities
    np.testing.assert_allclose(opacities[sel], 0.3, atol=1e-3)
    np.testing.assert_allclose(opacities[~sel], scene.opacities[~sel], atol=1e-3)


def test_sync_preview_enables_for_box_and_clears_when_unconstrained(device) -> None:
    scene = _scene(80, seed=8)
    viewer = _loaded_viewer(device, scene)
    state = ed.editor_state(viewer)
    ed.init_box_to_scene(viewer)
    ed.refresh_histograms(viewer, force=True)

    # Box enabled -> preview on with the box bound.
    state.box_enabled = True
    state.preview_enabled = True
    ed.sync_preview(viewer)
    assert bool(viewer.s.renderer._preview_enabled) is True
    assert bool(viewer.s.renderer._preview_box_enabled) is True

    # Box off and every range at full extent -> nothing to preview -> cleared.
    state.box_enabled = False
    ed.sync_preview(viewer)
    assert bool(viewer.s.renderer._preview_enabled) is False

    # Narrowing a histogram range re-enables the preview even without a box.
    edges = state.histograms[ed.splat_edit.SELECT_OPACITY][1]
    state.ranges[ed.splat_edit.SELECT_OPACITY] = (float(edges[0]), 0.5 * float(edges[0] + edges[-1]))
    ed.sync_preview(viewer)
    assert bool(viewer.s.renderer._preview_enabled) is True
    assert bool(viewer.s.renderer._preview_box_enabled) is False

    # The preview toggle overrides everything.
    state.preview_enabled = False
    ed.sync_preview(viewer)
    assert bool(viewer.s.renderer._preview_enabled) is False
