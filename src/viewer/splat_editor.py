"""Controller for the viewer's "Edit Splat" tools.

Drives the renderer's GPU edit kernels for the active scene (loaded PLY or live
trainer): the selection lives entirely in the GPU highlight buffer, and box /
histogram-range selection, property edits, and resampling all run in place on the
param buffer with no CPU round-trip. The pure-numpy :mod:`src.scene.splat_edit`
ops remain as the CPU reference the kernels are validated against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..scene.gaussian_scene import GaussianScene
from ..scene import splat_edit
from .state import SceneCountProxy

_HISTOGRAM_BINS = 64
_DEFAULT_BOX_PADDING = 0.02
DEFAULT_HIGHLIGHT_COLOR = (1.0, 0.55, 0.1)
DEFAULT_HIGHLIGHT_MIX = 0.65


@dataclass(slots=True)
class SplatEditorState:
    """Mutable UI/selection state for the splat editor, stored on the viewer."""

    selected_count: int = 0
    scene_count: int = 0
    # Oriented bounding box (world-from-box rotation columns are the box axes).
    box_center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    box_half_extents: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))
    box_rotation_euler: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    box_initialized: bool = False
    box_enabled: bool = True
    gizmo_operation: str = "universal"
    # Per-scalar histogram range selection.
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    histograms: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    histograms_dirty: bool = True
    # Edit parameters.
    resample_percent: float = 100.0
    edit_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    edit_color_enabled: bool = False
    edit_opacity: float = 1.0
    edit_opacity_enabled: bool = False
    edit_scale: float = 0.01
    edit_scale_enabled: bool = False
    highlight_color: tuple[float, float, float] = DEFAULT_HIGHLIGHT_COLOR
    highlight_mix: float = DEFAULT_HIGHLIGHT_MIX
    status: str = ""


def camera_view_projection_matrices(camera: object, width: int, height: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Build (view, projection) 4x4 matrices matching the viewport's pinhole projection.

    The pair reproduces the world->screen mapping used by the camera overlays, in the
    convention ImGuizmo expects: ``screen = NDC->pixel`` with ``NDC.y`` up. ``view`` maps
    world->camera, ``projection`` maps camera->clip. Returns ``None`` if the camera lacks
    the required intrinsics.
    """
    if not (hasattr(camera, "basis") and hasattr(camera, "focal_pixels_xy") and hasattr(camera, "principal_point")):
        return None
    try:
        right, up, forward = (np.asarray(axis, dtype=np.float64).reshape(3) for axis in camera.basis())
        position = np.asarray(getattr(camera, "position"), dtype=np.float64).reshape(3)
        fx, fy = camera.focal_pixels_xy(int(width), int(height))
        cx, cy = camera.principal_point(int(width), int(height))
    except Exception:
        return None
    w = float(max(int(width), 1))
    h = float(max(int(height), 1))
    if not (np.isfinite(fx) and np.isfinite(fy) and fx > 1e-8 and fy > 1e-8):
        return None
    view = np.eye(4, dtype=np.float64)
    view[0, :3], view[1, :3], view[2, :3] = right, up, forward
    view[0, 3], view[1, 3], view[2, 3] = -float(right @ position), -float(up @ position), -float(forward @ position)
    near = float(getattr(camera, "near", 0.05) or 0.05)
    far = float(getattr(camera, "far", 1000.0) or 1000.0)
    if far <= near:
        far = near + 1.0
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = 2.0 * float(fx) / w
    proj[0, 2] = 2.0 * float(cx) / w - 1.0
    proj[1, 1] = -2.0 * float(fy) / h
    proj[1, 2] = 1.0 - 2.0 * float(cy) / h
    proj[2, 2] = (far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = 1.0
    return view.astype(np.float32), proj.astype(np.float32)


def box_model_matrix(state: SplatEditorState) -> np.ndarray:
    """Box-to-world 4x4 model matrix (columns scaled by half-extents)."""
    model = np.eye(4, dtype=np.float32)
    model[:3, :3] = box_rotation_matrix(state) * np.asarray(state.box_half_extents, dtype=np.float32).reshape(1, 3)
    model[:3, 3] = np.asarray(state.box_center, dtype=np.float32).reshape(3)
    return model


def box_rotation_matrix(state: SplatEditorState) -> np.ndarray:
    """World-from-box rotation matrix from the box's XYZ euler angles (degrees)."""
    angles = np.deg2rad(np.asarray(state.box_rotation_euler, dtype=np.float64).reshape(3))
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return (rot_z @ rot_y @ rot_x).astype(np.float32)


def editor_state(viewer: object) -> SplatEditorState:
    state = getattr(viewer.s, "splat_editor", None)
    if state is None:
        state = SplatEditorState()
        viewer.s.splat_editor = state
    return state


def is_editing_trainer(viewer: object) -> bool:
    return getattr(viewer.s, "trainer", None) is not None


def _display_renderer(viewer: object) -> object | None:
    """The renderer that draws the viewport for the currently edited scene."""
    if is_editing_trainer(viewer) and getattr(viewer.s, "training_renderer", None) is not None:
        return viewer.s.training_renderer
    return getattr(viewer.s, "renderer", None)


def current_scene(viewer: object) -> GaussianScene | None:
    """Read the editable scene from the GPU param buffer (for histograms / bounds)."""
    renderer = _display_renderer(viewer)
    if renderer is None or int(getattr(renderer, "_scene_count", 0)) <= 0:
        return None
    try:
        return renderer.read_live_scene()
    except Exception:
        return None


def has_editable_scene(viewer: object) -> bool:
    return _scene_count(viewer) > 0


def _scene_count(viewer: object) -> int:
    renderer = _display_renderer(viewer)
    return int(getattr(renderer, "_scene_count", 0)) if renderer is not None else 0


_SELECT_MODES = {"replace": 0, "add": 1, "subtract": 2, "intersect": 3}


def selected_count(viewer: object) -> int:
    return int(editor_state(viewer).selected_count)


def sync_selection_to_scene(viewer: object) -> object | None:
    """Clear a stale GPU selection if the scene count changed under the editor (e.g. refinement)."""
    renderer = _display_renderer(viewer)
    if renderer is None:
        return None
    state = editor_state(viewer)
    count = int(getattr(renderer, "_scene_count", 0))
    if int(state.scene_count) != count:
        state.scene_count = count
        state.selected_count = int(renderer.edit_select_set_all(renderer.SELECT_SET_CLEAR)) if count > 0 else 0
    return renderer


def refresh_histograms(viewer: object, *, force: bool = False) -> None:
    state = editor_state(viewer)
    if not (force or state.histograms_dirty):
        return
    scene = current_scene(viewer)
    if scene is None or scene.count == 0:
        state.histograms = {}
        state.histograms_dirty = False
        return
    for kind in splat_edit.SELECTION_SCALARS:
        values = splat_edit.selection_scalar(scene, kind)
        counts, edges = splat_edit.log10_histogram(values, _HISTOGRAM_BINS)
        state.histograms[kind] = (counts, edges)
        if kind not in state.ranges:
            state.ranges[kind] = (float(edges[0]), float(edges[-1]))
    state.histograms_dirty = False


def init_box_to_scene(viewer: object, *, force: bool = False) -> None:
    state = editor_state(viewer)
    if state.box_initialized and not force:
        return
    scene = current_scene(viewer)
    if scene is None or scene.count == 0:
        return
    lo, hi = splat_edit.scene_bounds(scene)
    center = (lo + hi) * 0.5
    extent = np.maximum((hi - lo) * 0.5, 1e-4)
    state.box_center = center.astype(np.float32)
    state.box_half_extents = (extent * (1.0 + _DEFAULT_BOX_PADDING)).astype(np.float32)
    state.box_rotation_euler = np.zeros(3, dtype=np.float32)
    state.box_initialized = True


# --- selection mutation -------------------------------------------------------

def select_box(viewer: object, mode: str = "add") -> int:
    renderer = sync_selection_to_scene(viewer)
    state = editor_state(viewer)
    if renderer is None or _scene_count(viewer) == 0:
        return 0
    init_box_to_scene(viewer)
    axes = box_rotation_matrix(state).T  # rows = box axes (columns of world-from-box)
    hit = renderer.edit_select_box(state.box_center, axes, state.box_half_extents, _SELECT_MODES.get(mode, 1))
    state.selected_count = int(hit)
    sync_highlight(viewer)
    state.status = f"Selected {int(hit)} splats total (box)."
    return int(hit)


def select_range(viewer: object, kind: str, mode: str = "add") -> int:
    renderer = sync_selection_to_scene(viewer)
    state = editor_state(viewer)
    if renderer is None or _scene_count(viewer) == 0:
        return 0
    low, high = state.ranges.get(kind, (float("-inf"), float("inf")))
    scalar_kind = {"scale": renderer.SELECT_SCALAR_SCALE, "opacity": renderer.SELECT_SCALAR_OPACITY, "color": renderer.SELECT_SCALAR_COLOR}[kind]
    hit = renderer.edit_select_range(scalar_kind, low, high, _SELECT_MODES.get(mode, 1))
    state.selected_count = int(hit)
    sync_highlight(viewer)
    state.status = f"Selected {int(hit)} splats total by {kind} range."
    return int(hit)


def invert_selection(viewer: object) -> None:
    renderer = sync_selection_to_scene(viewer)
    state = editor_state(viewer)
    if renderer is None or _scene_count(viewer) == 0:
        return
    state.selected_count = int(renderer.edit_select_set_all(renderer.SELECT_SET_INVERT))
    sync_highlight(viewer)
    state.status = f"Inverted selection ({state.selected_count} total)."


def clear_selection(viewer: object) -> None:
    renderer = sync_selection_to_scene(viewer)
    state = editor_state(viewer)
    if renderer is not None and _scene_count(viewer) > 0:
        renderer.edit_select_set_all(renderer.SELECT_SET_CLEAR)
    state.selected_count = 0
    sync_highlight(viewer)
    state.status = "Cleared selection."


# --- highlight ----------------------------------------------------------------

def sync_highlight(viewer: object) -> None:
    renderer = _display_renderer(viewer)
    if renderer is None or not hasattr(renderer, "set_highlight_visible"):
        return
    state = editor_state(viewer)
    renderer.set_highlight_appearance(state.highlight_color, state.highlight_mix)
    renderer.set_highlight_visible(state.selected_count > 0)


def clear_highlight(viewer: object) -> None:
    renderer = _display_renderer(viewer)
    if renderer is not None and hasattr(renderer, "set_highlight_visible"):
        renderer.set_highlight_visible(False)


# --- edit operations ----------------------------------------------------------

def apply_resample(viewer: object) -> bool:
    renderer = sync_selection_to_scene(viewer)
    state = editor_state(viewer)
    if renderer is None or _scene_count(viewer) == 0:
        state.status = "No scene to edit."
        return False
    if state.selected_count == 0:
        state.status = "Resample skipped: nothing selected."
        return False
    ratio = max(float(state.resample_percent), 0.0) / 100.0
    if abs(ratio - 1.0) < 1e-9:
        state.status = "Resample made no change (ratio = 100%)."
        return False
    before = _scene_count(viewer)
    seed = int(np.random.default_rng().integers(0, 2**32, dtype=np.uint64))
    if is_editing_trainer(viewer):
        new_count = int(viewer.s.trainer.resample_selection(ratio, seed))
        viewer.s.scene = SceneCountProxy(new_count)
    else:
        new_count = int(renderer.edit_resample(ratio, seed))
    state.scene_count = new_count
    state.selected_count = int(renderer.edit_selection_count())
    state.histograms_dirty = True
    sync_highlight(viewer)
    delta = new_count - before
    verb = "Added" if delta >= 0 else "Removed"
    state.status = f"Resampled selection: {verb} {abs(delta)} splats ({before} -> {new_count})."
    return True


def apply_edit_properties(viewer: object) -> bool:
    renderer = sync_selection_to_scene(viewer)
    state = editor_state(viewer)
    if renderer is None or _scene_count(viewer) == 0:
        state.status = "No scene to edit."
        return False
    if state.selected_count == 0:
        state.status = "Edit skipped: nothing selected."
        return False
    if not (state.edit_color_enabled or state.edit_opacity_enabled or state.edit_scale_enabled):
        state.status = "Edit skipped: enable at least one property."
        return False
    renderer.edit_properties(
        color=state.edit_color if state.edit_color_enabled else None,
        opacity=state.edit_opacity if state.edit_opacity_enabled else None,
        scale=state.edit_scale if state.edit_scale_enabled else None,
    )
    if is_editing_trainer(viewer):
        viewer.s.scene = SceneCountProxy(_scene_count(viewer))
    state.histograms_dirty = True
    state.status = f"Edited properties of {state.selected_count} splats."
    return True
