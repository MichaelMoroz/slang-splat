from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.scene.gaussian_scene import GaussianScene
from src.scene.sh_utils import SUPPORTED_SH_COEFF_COUNT, sh_coeffs_to_display_colors
from src.viewer import splat_editor as ed
from src.viewer.state import SceneCountProxy


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


class _FakeRenderer:
    def __init__(self) -> None:
        self.scene: GaussianScene | None = None
        self.highlight_mask: np.ndarray | None = None

    def set_scene(self, scene: GaussianScene) -> None:
        self.scene = scene

    def set_selection_highlight(self, mask, *, color=None, mix=None) -> None:
        self.highlight_mask = None if mask is None else np.asarray(mask, dtype=bool).copy()


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
    # ImGuizmo worldToPos: x = (ndc.x*0.5+0.5)*W ; y = (1-(ndc.y*0.5+0.5))*H
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


def _loaded_viewer(scene: GaussianScene) -> SimpleNamespace:
    renderer = _FakeRenderer()
    renderer.set_scene(scene)
    return SimpleNamespace(s=SimpleNamespace(scene=scene, renderer=renderer, trainer=None, training_renderer=None))


def test_box_selection_highlights_and_counts() -> None:
    scene = _scene(60, seed=1)
    viewer = _loaded_viewer(scene)
    ed.init_box_to_scene(viewer)
    state = ed.editor_state(viewer)
    # Shrink the box to a sub-region and select.
    state.box_half_extents = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    state.box_center = np.zeros(3, dtype=np.float32)
    ed.select_box(viewer)
    expected = np.all(np.abs(scene.positions) <= 0.5, axis=1)
    np.testing.assert_array_equal(state.selection, expected)
    # Highlight pushed to the renderer mirrors the selection.
    np.testing.assert_array_equal(viewer.s.renderer.highlight_mask, expected)


def test_range_selection_add_accumulates() -> None:
    scene = _scene(40, seed=2)
    viewer = _loaded_viewer(scene)
    ed.refresh_histograms(viewer, force=True)
    state = ed.editor_state(viewer)
    # Opacity range: low half.
    state.ranges[ed.splat_edit.SELECT_OPACITY] = (0.0, 0.5)
    ed.select_range(viewer, ed.splat_edit.SELECT_OPACITY)
    first = state.selection.copy()
    np.testing.assert_array_equal(first, scene.opacities <= 0.5)
    # Adding a scale range ORs in more splats.
    state.ranges[ed.splat_edit.SELECT_SCALE] = (0.0, float("inf"))
    ed.select_range(viewer, ed.splat_edit.SELECT_SCALE)
    assert int(state.selection.sum()) == scene.count  # scale range covers everything


def test_invert_and_clear() -> None:
    scene = _scene(20, seed=3)
    viewer = _loaded_viewer(scene)
    state = ed.editor_state(viewer)
    ed.ensure_selection(state, scene.count)
    state.selection[:5] = True
    ed.invert_selection(viewer)
    assert int(state.selection.sum()) == 15
    ed.clear_selection(viewer)
    assert int(state.selection.sum()) == 0
    assert viewer.s.renderer.highlight_mask is None


def test_resample_sparsify_writes_back_loaded_scene() -> None:
    scene = _scene(100, seed=4)
    viewer = _loaded_viewer(scene)
    state = ed.editor_state(viewer)
    ed.ensure_selection(state, scene.count)
    state.selection[:50] = True
    state.resample_percent = 50.0
    assert ed.apply_resample(viewer) is True
    assert viewer.s.scene.count == 75  # 50 unselected + 25 kept
    assert viewer.s.renderer.scene is viewer.s.scene
    assert int(state.selection.sum()) == 25


def test_edit_properties_requires_enabled_property() -> None:
    scene = _scene(10, seed=5)
    viewer = _loaded_viewer(scene)
    state = ed.editor_state(viewer)
    ed.ensure_selection(state, scene.count)
    state.selection[:] = True
    assert ed.apply_edit_properties(viewer) is False  # nothing enabled
    state.edit_opacity_enabled = True
    state.edit_opacity = 0.3
    assert ed.apply_edit_properties(viewer) is True
    np.testing.assert_allclose(viewer.s.scene.opacities, 0.3, atol=1e-6)


def test_trainer_editing_uses_replace_scene_and_training_renderer() -> None:
    scene = _scene(30, seed=6)
    training_renderer = _FakeRenderer()
    training_renderer.set_scene(scene)
    replaced: dict[str, GaussianScene] = {}

    class _FakeTrainer:
        _scene_count = scene.count

        def read_live_scene(self):
            return scene

        def replace_scene(self, new_scene):
            replaced["scene"] = new_scene
            type(self)._scene_count = new_scene.count

    viewer = SimpleNamespace(
        s=SimpleNamespace(scene=SceneCountProxy(scene.count), renderer=_FakeRenderer(), trainer=_FakeTrainer(), training_renderer=training_renderer)
    )
    state = ed.editor_state(viewer)
    ed.ensure_selection(state, scene.count)
    state.selection[:10] = True
    state.resample_percent = 200.0
    assert ed.apply_resample(viewer) is True
    assert replaced["scene"].count == 40  # 30 + 10 children
    # Highlight goes to the training renderer (the one drawing the viewport).
    assert training_renderer.highlight_mask is not None
    assert isinstance(viewer.s.scene, SceneCountProxy) and viewer.s.scene.count == 40
