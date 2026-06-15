"""Controller for the viewer's "Edit Splat" tools.

Bridges the pure :mod:`src.scene.splat_edit` operations to the live viewer:
maintains the selection mask, applies bounding-box / histogram-range selections,
runs resample / property edits, writes the result back to the active scene (loaded
PLY or the live trainer), and keeps the GPU selection highlight in sync.
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

    selection: np.ndarray | None = None
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
    """Read the editable scene (live trainer state takes priority over a loaded PLY)."""
    trainer = getattr(viewer.s, "trainer", None)
    if trainer is not None:
        try:
            return trainer.read_live_scene()
        except Exception:
            return None
    scene = getattr(viewer.s, "scene", None)
    return scene if isinstance(scene, GaussianScene) else None


def has_editable_scene(viewer: object) -> bool:
    trainer = getattr(viewer.s, "trainer", None)
    if trainer is not None:
        return int(getattr(trainer, "_scene_count", 0)) > 0
    return isinstance(getattr(viewer.s, "scene", None), GaussianScene)


def _scene_count(viewer: object) -> int:
    trainer = getattr(viewer.s, "trainer", None)
    if trainer is not None:
        return int(getattr(trainer, "_scene_count", 0))
    scene = getattr(viewer.s, "scene", None)
    return int(scene.count) if isinstance(scene, GaussianScene) else 0


def ensure_selection(state: SplatEditorState, count: int) -> np.ndarray:
    """Return a selection mask sized to ``count``, resetting it if the count changed."""
    if state.selection is None or int(state.selection.shape[0]) != int(count):
        state.selection = np.zeros((max(int(count), 0),), dtype=bool)
        state.scene_count = int(count)
    return state.selection


def selected_count(viewer: object) -> int:
    state = editor_state(viewer)
    if state.selection is None:
        return 0
    return int(state.selection.sum())


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

def _apply_selection(viewer: object, new_mask: np.ndarray, mode: str) -> None:
    state = editor_state(viewer)
    mask = ensure_selection(state, new_mask.shape[0])
    if mode == "replace":
        state.selection = new_mask.copy()
    elif mode == "subtract":
        state.selection = mask & ~new_mask
    elif mode == "intersect":
        state.selection = mask & new_mask
    else:  # add
        state.selection = mask | new_mask
    sync_highlight(viewer)


def select_box(viewer: object, mode: str = "add") -> int:
    state = editor_state(viewer)
    scene = current_scene(viewer)
    if scene is None or scene.count == 0:
        return 0
    init_box_to_scene(viewer)
    mask = splat_edit.select_in_box(scene, state.box_center, state.box_half_extents, rotation=box_rotation_matrix(state))
    _apply_selection(viewer, mask, mode)
    state.status = f"Selected {int(mask.sum())} splats in box ({selected_count(viewer)} total)."
    return int(mask.sum())


def select_range(viewer: object, kind: str, mode: str = "add") -> int:
    state = editor_state(viewer)
    scene = current_scene(viewer)
    if scene is None or scene.count == 0:
        return 0
    low, high = state.ranges.get(kind, (float("-inf"), float("inf")))
    values = splat_edit.selection_scalar(scene, kind)
    mask = splat_edit.select_in_range(values, low, high)
    _apply_selection(viewer, mask, mode)
    state.status = f"Selected {int(mask.sum())} splats by {kind} range ({selected_count(viewer)} total)."
    return int(mask.sum())


def invert_selection(viewer: object) -> None:
    state = editor_state(viewer)
    mask = ensure_selection(state, _scene_count(viewer))
    state.selection = ~mask
    state.status = f"Inverted selection ({selected_count(viewer)} total)."
    sync_highlight(viewer)


def clear_selection(viewer: object) -> None:
    state = editor_state(viewer)
    state.selection = np.zeros((_scene_count(viewer),), dtype=bool)
    state.status = "Cleared selection."
    sync_highlight(viewer)


# --- highlight ----------------------------------------------------------------

def sync_highlight(viewer: object) -> None:
    renderer = _display_renderer(viewer)
    if renderer is None or not hasattr(renderer, "set_selection_highlight"):
        return
    state = editor_state(viewer)
    mask = state.selection
    if mask is None or not bool(mask.any()):
        renderer.set_selection_highlight(None)
        return
    renderer.set_selection_highlight(mask, color=state.highlight_color, mix=state.highlight_mix)


def clear_highlight(viewer: object) -> None:
    renderer = _display_renderer(viewer)
    if renderer is not None and hasattr(renderer, "set_selection_highlight"):
        renderer.set_selection_highlight(None)


# --- edit operations ----------------------------------------------------------

def _write_back(viewer: object, new_scene: GaussianScene, new_mask: np.ndarray) -> None:
    state = editor_state(viewer)
    if is_editing_trainer(viewer):
        viewer.s.trainer.replace_scene(new_scene)
        viewer.s.scene = SceneCountProxy(new_scene.count)
    else:
        viewer.s.scene = new_scene
        renderer = getattr(viewer.s, "renderer", None)
        if renderer is not None:
            renderer.set_scene(new_scene)
    state.selection = np.asarray(new_mask, dtype=bool).reshape(-1)
    state.scene_count = int(new_scene.count)
    state.histograms_dirty = True
    sync_highlight(viewer)


def apply_resample(viewer: object) -> bool:
    state = editor_state(viewer)
    scene = current_scene(viewer)
    if scene is None or scene.count == 0:
        state.status = "No scene to edit."
        return False
    mask = ensure_selection(state, scene.count)
    if not mask.any():
        state.status = "Resample skipped: nothing selected."
        return False
    ratio = max(float(state.resample_percent), 0.0) / 100.0
    before = scene.count
    new_scene, new_mask = splat_edit.resample_selection(scene, mask, ratio)
    if new_scene is scene:
        state.status = "Resample made no change (ratio = 100%)."
        return False
    _write_back(viewer, new_scene, new_mask)
    delta = new_scene.count - before
    verb = "Added" if delta >= 0 else "Removed"
    state.status = f"Resampled selection: {verb} {abs(delta)} splats ({before} -> {new_scene.count})."
    return True


def apply_edit_properties(viewer: object) -> bool:
    state = editor_state(viewer)
    scene = current_scene(viewer)
    if scene is None or scene.count == 0:
        state.status = "No scene to edit."
        return False
    mask = ensure_selection(state, scene.count)
    if not mask.any():
        state.status = "Edit skipped: nothing selected."
        return False
    if not (state.edit_color_enabled or state.edit_opacity_enabled or state.edit_scale_enabled):
        state.status = "Edit skipped: enable at least one property."
        return False
    new_scene = splat_edit.edit_properties(
        scene,
        mask,
        color=state.edit_color if state.edit_color_enabled else None,
        opacity=state.edit_opacity if state.edit_opacity_enabled else None,
        total_scale_value=state.edit_scale if state.edit_scale_enabled else None,
    )
    _write_back(viewer, new_scene, mask)
    state.status = f"Edited properties of {int(mask.sum())} splats."
    return True
