"""Pure-numpy Gaussian splat selection and editing operations.

These helpers operate on :class:`GaussianScene` arrays only (no GPU state) so the
viewer's "Edit Splat" tools can be unit tested in isolation. Field conventions
match the rest of the scene pipeline:

* ``scales`` are stored in log-space (linear size = ``exp(scale)``).
* ``opacities`` are linear alpha in ``[0, 1]``.
* ``colors`` are display RGB; ``sh_coeffs[:, 0]`` is the DC base color.
* ``rotations`` are normalized quaternions ``(w, x, y, z)``.
"""

from __future__ import annotations

import numpy as np

from .gaussian_scene import GaussianScene
from .sh_utils import rgb_to_sh0, sh_coeffs_to_display_colors

# Selection scalar kinds exposed as histogram/range axes in the editor.
SELECT_SCALE = "scale"
SELECT_OPACITY = "opacity"
SELECT_COLOR = "color"
SELECTION_SCALARS = (SELECT_SCALE, SELECT_OPACITY, SELECT_COLOR)

# Rec. 709 luminance weights for the single-scalar "color" selection axis.
_LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# Child scale shrink factor used when densifying (matches the 3DGS split heuristic).
_DENSIFY_SCALE_SHRINK = 1.6


def _empty_mask(count: int) -> np.ndarray:
    return np.zeros((max(int(count), 0),), dtype=bool)


def total_scale(scene: GaussianScene) -> np.ndarray:
    """Per-splat isotropic size: geometric mean of the linear axis scales."""
    scales = np.asarray(scene.scales, dtype=np.float32)
    if scales.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.exp(np.mean(scales[:, :3], axis=1)).astype(np.float32, copy=False)


def color_luminance(scene: GaussianScene) -> np.ndarray:
    colors = np.asarray(scene.colors, dtype=np.float32)
    if colors.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    return (colors[:, :3] @ _LUMA_WEIGHTS).astype(np.float32, copy=False)


def selection_scalar(scene: GaussianScene, kind: str) -> np.ndarray:
    """Return the per-splat scalar used for histogram/range selection."""
    if kind == SELECT_SCALE:
        return total_scale(scene)
    if kind == SELECT_OPACITY:
        return np.asarray(scene.opacities, dtype=np.float32).reshape(-1)
    if kind == SELECT_COLOR:
        return color_luminance(scene)
    raise ValueError(f"Unknown selection scalar kind: {kind}")


def log10_histogram(
    values: np.ndarray,
    bins: int = 64,
    *,
    floor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a log10-spaced histogram, mirroring the Histograms window style.

    Returns ``(counts, edges)`` where ``edges`` are in linear value space (length
    ``bins + 1``). Non-positive and non-finite samples are clamped to ``floor``.
    """
    bin_count = max(int(bins), 1)
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    samples = samples[np.isfinite(samples)]
    positive = samples[samples > 0.0]
    if positive.size == 0:
        return np.zeros((bin_count,), dtype=np.int64), np.logspace(-1.0, 0.0, bin_count + 1)
    floor_value = float(positive.min()) if floor is None else max(float(floor), 1e-30)
    v_min = max(float(positive.min()), floor_value)
    v_max = float(positive.max())
    log_min = np.log10(v_min)
    log_max = np.log10(v_max)
    if not np.isfinite(log_max) or log_max <= log_min:
        log_max = log_min + 1.0
    log_edges = np.linspace(log_min, log_max, bin_count + 1)
    clamped = np.clip(samples, v_min, 10.0**log_max)
    counts, _ = np.histogram(np.log10(clamped), bins=log_edges)
    return counts.astype(np.int64, copy=False), (10.0**log_edges).astype(np.float64, copy=False)


def select_in_range(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Boolean mask for samples within ``[low, high]`` (inclusive)."""
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    lo, hi = (float(low), float(high)) if low <= high else (float(high), float(low))
    return (samples >= lo) & (samples <= hi)


def quaternion_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    """Convert ``(w, x, y, z)`` quaternions to ``[..., 3, 3]`` rotation matrices."""
    quats = np.asarray(quaternions, dtype=np.float64).reshape(-1, 4)
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = np.divide(quats, np.maximum(norms, 1e-12), out=np.zeros_like(quats), where=norms > 0.0)
    quats[norms.reshape(-1) <= 0.0, 0] = 1.0
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    matrices = np.empty((quats.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[:, 0, 1] = 2.0 * (x * y - w * z)
    matrices[:, 0, 2] = 2.0 * (x * z + w * y)
    matrices[:, 1, 0] = 2.0 * (x * y + w * z)
    matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[:, 1, 2] = 2.0 * (y * z - w * x)
    matrices[:, 2, 0] = 2.0 * (x * z - w * y)
    matrices[:, 2, 1] = 2.0 * (y * z + w * x)
    matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices


def select_in_box(
    scene: GaussianScene,
    center: np.ndarray,
    half_extents: np.ndarray,
    rotation: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean mask of splats whose centers fall inside an oriented box.

    ``rotation`` is the box's 3x3 orientation (world-from-box); ``None`` selects an
    axis-aligned box. ``half_extents`` are the box half-sizes along its local axes.
    """
    positions = np.asarray(scene.positions, dtype=np.float64)
    if positions.shape[0] == 0:
        return _empty_mask(0)
    center_arr = np.asarray(center, dtype=np.float64).reshape(3)
    half = np.abs(np.asarray(half_extents, dtype=np.float64).reshape(3))
    local = positions - center_arr
    if rotation is not None:
        rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        # Project world offsets into box-local axes (inverse of world-from-box).
        local = local @ rot
    return np.all(np.abs(local) <= half, axis=1)


def scene_bounds(scene: GaussianScene) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(min_xyz, max_xyz)`` of splat centers (zeros for an empty scene)."""
    positions = np.asarray(scene.positions, dtype=np.float32)
    if positions.shape[0] == 0:
        return np.zeros((3,), dtype=np.float32), np.zeros((3,), dtype=np.float32)
    return positions.min(axis=0).astype(np.float32), positions.max(axis=0).astype(np.float32)


def _index_scene(scene: GaussianScene, index: np.ndarray) -> GaussianScene:
    return GaussianScene(
        positions=scene.positions[index],
        scales=scene.scales[index],
        rotations=scene.rotations[index],
        opacities=scene.opacities[index],
        colors=scene.colors[index],
        sh_coeffs=scene.sh_coeffs[index],
    )


def _concat_scenes(a: GaussianScene, b: GaussianScene) -> GaussianScene:
    return GaussianScene(
        positions=np.concatenate([a.positions, b.positions], axis=0),
        scales=np.concatenate([a.scales, b.scales], axis=0),
        rotations=np.concatenate([a.rotations, b.rotations], axis=0),
        opacities=np.concatenate([a.opacities, b.opacities], axis=0),
        colors=np.concatenate([a.colors, b.colors], axis=0),
        sh_coeffs=np.concatenate([a.sh_coeffs, b.sh_coeffs], axis=0),
    )


def _make_children(scene: GaussianScene, parents: np.ndarray, rng: np.random.Generator) -> GaussianScene:
    """Build child splats by offsetting parents within their covariance and shrinking."""
    parent_idx = np.asarray(parents, dtype=np.intp).reshape(-1)
    child = _index_scene(scene, parent_idx)
    linear_scales = np.exp(np.asarray(child.scales[:, :3], dtype=np.float64))
    rotation = quaternion_to_matrix(child.rotations)
    local = rng.normal(size=(parent_idx.shape[0], 3)) * linear_scales
    offsets = np.einsum("nij,nj->ni", rotation, local)
    child.positions[:, :3] = (np.asarray(child.positions[:, :3], dtype=np.float64) + offsets).astype(np.float32)
    child.scales[:, :3] = (np.asarray(child.scales[:, :3], dtype=np.float32) - np.log(_DENSIFY_SCALE_SHRINK)).astype(np.float32)
    return child


def resample_selection(
    scene: GaussianScene,
    mask: np.ndarray,
    ratio: float,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[GaussianScene, np.ndarray]:
    """Sparsify (``ratio < 1``) or densify (``ratio > 1``) the selected splats.

    Returns the new scene and a selection mask aligned to it (kept and newly added
    splats stay selected). Unselected splats are never touched.
    """
    selection = np.asarray(mask, dtype=bool).reshape(-1)
    count = scene.count
    if selection.shape[0] != count:
        raise ValueError("Selection mask length must match the scene splat count.")
    selected_idx = np.where(selection)[0]
    n_selected = int(selected_idx.shape[0])
    resample_ratio = max(float(ratio), 0.0)
    if n_selected == 0 or abs(resample_ratio - 1.0) < 1e-9:
        return scene, selection.copy()
    generator = np.random.default_rng() if rng is None else rng
    target = int(round(resample_ratio * n_selected))

    if target < n_selected:
        drop_count = n_selected - target
        drop_idx = generator.permutation(selected_idx)[:drop_count]
        keep = np.ones((count,), dtype=bool)
        keep[drop_idx] = False
        new_scene = _index_scene(scene, np.where(keep)[0])
        return new_scene, selection[keep].copy()

    add_count = target - n_selected
    parent_choice = selected_idx[generator.integers(0, n_selected, size=add_count)]
    children = _make_children(scene, parent_choice, generator)
    new_scene = _concat_scenes(scene, children)
    new_mask = np.concatenate([selection, np.ones((add_count,), dtype=bool)], axis=0)
    return new_scene, new_mask


def edit_properties(
    scene: GaussianScene,
    mask: np.ndarray,
    *,
    color: tuple[float, float, float] | np.ndarray | None = None,
    opacity: float | None = None,
    total_scale_value: float | None = None,
) -> GaussianScene:
    """Return a copy of ``scene`` with selected splat properties overridden.

    ``color`` sets the DC base color (higher-order SH is preserved); ``opacity`` sets
    linear alpha; ``total_scale_value`` rescales each splat so its geometric-mean size
    equals the target while preserving anisotropy.
    """
    selection = np.asarray(mask, dtype=bool).reshape(-1)
    if selection.shape[0] != scene.count:
        raise ValueError("Selection mask length must match the scene splat count.")
    edited = _index_scene(scene, np.arange(scene.count, dtype=np.intp))  # full deep-ish copy
    if not np.any(selection):
        return edited

    if opacity is not None:
        edited.opacities[selection] = np.float32(np.clip(float(opacity), 0.0, 1.0))

    if color is not None:
        rgb = np.clip(np.asarray(color, dtype=np.float32).reshape(3), 0.0, 1.0)
        edited.sh_coeffs[selection, 0, :] = rgb_to_sh0(rgb[None, :])[0]
        edited.colors[selection, :3] = sh_coeffs_to_display_colors(edited.sh_coeffs[selection])

    if total_scale_value is not None:
        target_log = np.log(max(float(total_scale_value), 1e-12))
        current_log_mean = np.mean(edited.scales[selection, :3], axis=1, keepdims=True)
        edited.scales[selection, :3] = (edited.scales[selection, :3] + (target_log - current_log_mean)).astype(np.float32)

    return edited
