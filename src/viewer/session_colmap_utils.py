from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np

from ..scene._internal.colmap_ops import DEPTH_INIT_VALUE_DISTANCE, DEPTH_INIT_VALUE_Z_DEPTH, build_colmap_image_path_index, resolve_colmap_image_path
from ..scene._internal.colmap_binary import _resolve_colmap_sparse_paths
from ..scene import load_colmap_reconstruction
from ..scene._internal.colmap_types import ColmapFrame
from ..training.alpha_modes import TARGET_ALPHA_MODE_OFF, resolve_target_alpha_mode, target_alpha_skip_mask_enabled
from ..training.defaults import TRAINING_BUILD_ARG_DEFAULTS
from .state import ColmapImportSettings, COLMAP_ROTATION_MODE_AUTO, COLMAP_ROTATION_MODE_CUSTOM, COLMAP_ROTATION_MODE_NONE

_COLMAP_IMPORT_POINTCLOUD = "pointcloud"
_COLMAP_IMPORT_DIFFUSED_POINTCLOUD = "diffused_pointcloud"
_COLMAP_IMPORT_CUSTOM_PLY = "custom_ply"
_COLMAP_IMPORT_CUSTOM_MESH = "custom_mesh"
_COLMAP_IMPORT_DEPTH = "depth"
_COLMAP_DEPTH_VALUE_DISTANCE = DEPTH_INIT_VALUE_DISTANCE
_COLMAP_DEPTH_VALUE_Z_DEPTH = DEPTH_INIT_VALUE_Z_DEPTH
_COLMAP_IMAGE_DOWNSCALE_ORIGINAL = "original"
_COLMAP_IMAGE_DOWNSCALE_MAX_SIZE = "max_size"
_COLMAP_IMAGE_DOWNSCALE_SCALE = "scale"
_COLMAP_CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    6: "FULL_OPENCV",
}
_COLMAP_DB_SAMPLE_LIMIT = 64
_COLMAP_DB_SEARCH_PATTERNS = ("database.db", "*.db", "*.sqlite", "*.sqlite3")
_COLMAP_IMPORT_IMAGES_PER_TICK = 1
_DEFAULT_TARGET_ALPHA_THRESHOLD = float(TRAINING_BUILD_ARG_DEFAULTS["target_alpha_threshold"])


def _set_ui_path(viewer: object, key: str, path: Path | None) -> None:
    viewer.ui._values[key] = "" if path is None else str(Path(path).resolve())


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    for path in paths:
        resolved = Path(path).resolve()
        if resolved in unique:
            continue
        unique.append(resolved)
    return unique


def _has_colmap_sparse(root: Path) -> bool:
    root_path = Path(root).resolve()
    try:
        _resolve_colmap_sparse_paths(root_path / "sparse" / "0", root_path / "sparse", root_path)
    except FileNotFoundError:
        return False
    return True


def _looks_like_colmap_database(database_path: Path) -> bool:
    db_path = Path(database_path).resolve()
    if not db_path.exists() or not db_path.is_file():
        return False
    try:
        with sqlite3.connect(str(db_path)) as conn:
            table_names = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    except sqlite3.Error:
        return False
    return "images" in table_names


def _find_colmap_database(dataset_root: Path) -> Path:
    root = Path(dataset_root).resolve()
    for pattern in _COLMAP_DB_SEARCH_PATTERNS:
        for candidate in sorted(root.rglob(pattern)):
            if _looks_like_colmap_database(candidate):
                return candidate.resolve()
    raise FileNotFoundError(f"Could not find a COLMAP database under {root}")


def _find_optional_colmap_database(dataset_root: Path) -> Path | None:
    try:
        return _find_colmap_database(dataset_root)
    except FileNotFoundError:
        return None


def _resolve_colmap_root_from_selection(dataset_root: Path) -> Path:
    root = Path(dataset_root).resolve()
    candidates = [root]
    candidates.extend(path.resolve() for path in sorted(root.rglob("*")) if path.is_dir())
    for candidate in candidates:
        if _has_colmap_sparse(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find COLMAP sparse reconstruction under {root}")


def _database_image_names(database_path: Path, limit: int = _COLMAP_DB_SAMPLE_LIMIT) -> list[str]:
    db_path = Path(database_path).resolve()
    with sqlite3.connect(str(db_path)) as conn:
        table = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images'").fetchone()
        if table is None:
            raise RuntimeError(f"COLMAP database has no images table: {db_path}")
        rows = conn.execute("SELECT name FROM images ORDER BY image_id LIMIT ?", (int(max(limit, 1)),)).fetchall()
    names = [str(row[0]).strip() for row in rows if row and str(row[0]).strip()]
    if not names:
        raise RuntimeError(f"COLMAP database has no image names: {db_path}")
    return names


def _dataset_directories(dataset_root: Path) -> list[Path]:
    root = Path(dataset_root).resolve()
    candidates = [root]
    candidates.extend(path.resolve() for path in sorted(root.rglob("*")) if path.is_dir())
    return _unique_paths(candidates)


def _looks_like_depth_directory(path: Path) -> bool:
    value = str(Path(path).resolve()).lower()
    return "depth" in value or "depth" in Path(path).name.lower()


def _looks_like_alpha_mask_directory(path: Path) -> bool:
    name = Path(path).name.lower()
    return "mask" in name or "alpha" in name


def _sample_colmap_image_names(image_names: list[str], limit: int = _COLMAP_DB_SAMPLE_LIMIT) -> list[str]:
    """Sample image names spread across the whole list.

    COLMAP names are ordered by image id, so a reconstruction that spans several
    capture subfolders (e.g. ``A/000.png`` ... ``C/268.png``) but only ships one
    subset on disk would never match if we only looked at the leading names. Spreading
    the sample ensures every subset is represented.
    """
    names = [str(name).strip() for name in image_names if str(name).strip()]
    cap = max(int(limit), 1)
    if len(names) <= cap:
        return names
    step = len(names) / float(cap)
    sampled = [names[min(int(index * step), len(names) - 1)] for index in range(cap)]
    seen: set[str] = set()
    return [name for name in sampled if not (name in seen or seen.add(name))]


def _suggest_images_root_from_dataset_root(dataset_root: Path, image_names: list[str]) -> Path:
    root = Path(dataset_root).resolve()
    sample_names = _sample_colmap_image_names(image_names)
    candidates = [candidate for candidate in _dataset_directories(root) if not _looks_like_depth_directory(candidate)]
    # Fast path: a candidate that directly contains a sampled image (handles
    # subfolder-prefixed names and datasets that only ship part of the reconstruction).
    for candidate in candidates:
        if any((candidate / image_name).exists() for image_name in sample_names):
            return candidate
    # Robust fallback: match by basename / relative stem (handles extension differences,
    # e.g. COLMAP names ``frame.jpg`` resolving to ``frame.png`` on disk).
    best_candidate: Path | None = None
    best_key = (0, -1)
    for candidate in candidates:
        index = build_colmap_image_path_index(candidate)
        hits = sum(1 for image_name in sample_names if resolve_colmap_image_path(candidate, image_name, image_path_index=index) is not None)
        # Prefer the most images matched, then the most specific (deepest) folder so
        # plain names land on the leaf image directory rather than an ancestor.
        key = (hits, len(candidate.parts))
        if hits > 0 and key > best_key:
            best_candidate, best_key = candidate, key
    if best_candidate is not None:
        return best_candidate
    raise FileNotFoundError(f"Could not find an image folder under {root} for COLMAP images: {sample_names[:4]}")


def _suggest_alpha_mask_root_from_dataset_root(dataset_root: Path, images_root: Path, image_names: list[str]) -> Path | None:
    root = Path(dataset_root).resolve()
    resolved_images_root = Path(images_root).resolve()
    sample_names = image_names[: min(len(image_names), _COLMAP_DB_SAMPLE_LIMIT)]
    sibling_dirs = () if not resolved_images_root.parent.exists() else tuple(path.resolve() for path in sorted(resolved_images_root.parent.iterdir()) if path.is_dir())
    for candidate in _unique_paths([*sibling_dirs, *_dataset_directories(root)]):
        if candidate == resolved_images_root or _looks_like_depth_directory(candidate) or not _looks_like_alpha_mask_directory(candidate):
            continue
        if any(resolve_colmap_image_path(candidate, image_name) is not None for image_name in sample_names):
            return candidate
    return None


def _camera_rows(recon: object) -> tuple[dict[str, object], ...]:
    frame_counts: dict[int, int] = {}
    for image in getattr(recon, "images", {}).values():
        camera_id = int(getattr(image, "camera_id", -1))
        frame_counts[camera_id] = frame_counts.get(camera_id, 0) + 1
    rows: list[dict[str, object]] = []
    for camera_id, camera in sorted(getattr(recon, "cameras", {}).items()):
        distortion_values = tuple(float(getattr(camera, name, 0.0)) for name in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"))
        while len(distortion_values) > 2 and abs(distortion_values[-1]) <= 1e-12:
            distortion_values = distortion_values[:-1]
        rows.append(
            {
                "camera_id": int(camera_id),
                "model_name": _COLMAP_CAMERA_MODEL_NAMES.get(int(getattr(camera, "model_id", -1)), f"MODEL_{int(getattr(camera, 'model_id', -1))}"),
                "frame_count": int(frame_counts.get(int(camera_id), 0)),
                "resolution_text": f"{int(camera.width)}x{int(camera.height)}",
                "focal_text": f"{float(camera.fx):.2f}, {float(camera.fy):.2f}",
                "principal_text": f"{float(camera.cx):.2f}, {float(camera.cy):.2f}",
                "distortion_text": ", ".join(f"{value:.4g}" for value in distortion_values),
            }
        )
    return tuple(rows)


def _point_preview_stats(recon: object) -> dict[str, int]:
    track_lengths = getattr(recon, "point_track_length_table", None)
    if track_lengths is not None:
        track_arr = np.asarray(track_lengths, dtype=np.int32).reshape(-1)
        return {
            "total_points": int(track_arr.size),
            "tracked_points_min2": int(np.count_nonzero(track_arr >= 2)),
        }
    points = tuple(getattr(recon, "points3d", {}).values())
    return {
        "total_points": int(len(points)),
        "tracked_points_min2": int(sum(int(getattr(point, "track_length", 0)) >= 2 for point in points)),
    }


def _normalized_selected_camera_ids(camera_rows: tuple[dict[str, object], ...], selected_camera_ids: tuple[int, ...] | None = None) -> tuple[int, ...]:
    camera_ids = tuple(int(row["camera_id"]) for row in camera_rows)
    if selected_camera_ids is None:
        return camera_ids
    selected = {int(camera_id) for camera_id in selected_camera_ids}
    return tuple(camera_id for camera_id in camera_ids if camera_id in selected)


def _set_colmap_camera_preview(viewer: object, recon: object, selected_camera_ids: tuple[int, ...] | None = None) -> tuple[int, ...]:
    rows = _camera_rows(recon)
    selected_ids = _normalized_selected_camera_ids(rows, selected_camera_ids)
    viewer.ui._values["_colmap_camera_rows"] = rows
    viewer.ui._values["_colmap_point_stats"] = _point_preview_stats(recon)
    viewer.ui._values["colmap_selected_camera_ids"] = selected_ids
    return selected_ids


def _update_import_settings(
    viewer: object,
    *,
    dataset_root: Path,
    database_path: Path | None,
    images_root: Path,
    alpha_mask_root: Path | None,
    use_alpha_masks: bool,
    depth_root: Path | None,
    selected_camera_ids: tuple[int, ...],
    depth_value_mode: str,
    init_mode: str,
    rotation_mode: int,
    custom_rotation_deg: tuple[float, float, float],
    compress_dataset_using_bc7: bool,
    training_image_color_init: bool,
    photometric_compensation_enabled: bool,
    custom_ply_path: Path | None,
    image_downscale_mode: str,
    image_downscale_max_size: int,
    image_downscale_scale: float,
    nn_radius_scale_coef: float,
    min_track_length: int,
    init_neighbor_count: int,
    init_anisotropy_strength: float,
    depth_point_count: int,
    diffused_point_count: int,
    fibonacci_sphere_point_count: int,
    fibonacci_sphere_radius_multiplier: float,
    fibonacci_sphere_color: tuple[float, float, float],
    fibonacci_sphere_upper_hemisphere_only: bool = False,
    target_alpha_mode: int | None = None,
    target_alpha_threshold: float = _DEFAULT_TARGET_ALPHA_THRESHOLD,
    use_target_alpha_mask: bool = False,
    pointcloud_enabled: bool | None = None,
    pointcloud_nn_radius_scale_coef: float | None = None,
    diffused_enabled: bool | None = None,
    diffused_diffusion_radius: float = 1.0,
    diffused_nn_radius_scale_coef: float | None = None,
    custom_ply_enabled: bool | None = None,
    custom_ply_nn_radius_scale_coef: float | None = None,
    custom_mesh_enabled: bool | None = None,
    custom_mesh_path: Path | None = None,
    custom_mesh_point_count: int | None = None,
    custom_mesh_nn_radius_scale_coef: float | None = None,
    fibonacci_sphere_enabled: bool | None = None,
    fibonacci_sphere_nn_radius_scale_coef: float | None = None,
) -> None:
    resolved_pointcloud_enabled = bool(pointcloud_enabled)
    resolved_pointcloud_nn_radius_scale_coef = float(max(pointcloud_nn_radius_scale_coef if pointcloud_nn_radius_scale_coef is not None else nn_radius_scale_coef, 1e-4))
    resolved_diffused_enabled = bool(diffused_enabled)
    resolved_diffused_diffusion_radius = max(float(diffused_diffusion_radius), 0.0)
    resolved_diffused_nn_radius_scale_coef = float(max(diffused_nn_radius_scale_coef if diffused_nn_radius_scale_coef is not None else nn_radius_scale_coef, 1e-4))
    resolved_custom_ply_enabled = bool(custom_ply_enabled)
    resolved_custom_ply_nn_radius_scale_coef = float(max(custom_ply_nn_radius_scale_coef if custom_ply_nn_radius_scale_coef is not None else 1.0, 1e-4))
    resolved_custom_mesh_enabled = bool(custom_mesh_enabled)
    resolved_custom_mesh_path = custom_mesh_path
    resolved_custom_mesh_point_count = max(int(custom_mesh_point_count if custom_mesh_point_count is not None else diffused_point_count), 1)
    resolved_custom_mesh_nn_radius_scale_coef = float(max(custom_mesh_nn_radius_scale_coef if custom_mesh_nn_radius_scale_coef is not None else nn_radius_scale_coef, 1e-4))
    resolved_fibonacci_sphere_enabled = bool(fibonacci_sphere_enabled)
    resolved_fibonacci_sphere_nn_radius_scale_coef = float(max(fibonacci_sphere_nn_radius_scale_coef if fibonacci_sphere_nn_radius_scale_coef is not None else 1.0, 1e-4))
    resolved_fibonacci_sphere_color = tuple(float(v) for v in np.clip(np.asarray(fibonacci_sphere_color, dtype=np.float32).reshape(3), 0.0, 1.0))
    resolved_rotation_mode = min(max(int(rotation_mode), COLMAP_ROTATION_MODE_NONE), COLMAP_ROTATION_MODE_AUTO)
    resolved_custom_rotation_deg = tuple(float(v) for v in np.asarray(custom_rotation_deg, dtype=np.float32).reshape(3))
    resolved_target_alpha_mode = resolve_target_alpha_mode(target_alpha_mode, legacy_use_target_alpha_mask=use_target_alpha_mask)
    resolved_target_alpha_threshold = float(np.clip(target_alpha_threshold, 0.0, 1.0))
    viewer.s.colmap_import = ColmapImportSettings(
        database_path=None if database_path is None else Path(database_path).resolve(),
        images_root=Path(images_root).resolve(),
        alpha_mask_root=None if alpha_mask_root is None else Path(alpha_mask_root).resolve(),
        use_alpha_masks=bool(use_alpha_masks and alpha_mask_root is not None),
        depth_root=None if depth_root is None else Path(depth_root).resolve(),
        selected_camera_ids=tuple(int(camera_id) for camera_id in selected_camera_ids),
        depth_value_mode=str(depth_value_mode),
        init_mode=str(init_mode),
        rotation_mode=resolved_rotation_mode,
        custom_rotation_deg=resolved_custom_rotation_deg,
        compress_dataset_using_bc7=bool(compress_dataset_using_bc7),
        training_image_color_init=bool(training_image_color_init),
        photometric_compensation_enabled=bool(photometric_compensation_enabled),
        custom_ply_path=None if custom_ply_path is None else Path(custom_ply_path).resolve(),
        image_downscale_mode=str(image_downscale_mode),
        image_downscale_max_size=max(int(image_downscale_max_size), 1),
        image_downscale_scale=float(np.clip(image_downscale_scale, 1e-6, 1.0)),
        nn_radius_scale_coef=float(max(nn_radius_scale_coef, 1e-4)),
        min_track_length=max(int(min_track_length), 0),
        init_neighbor_count=max(int(init_neighbor_count), 2),
        init_anisotropy_strength=float(np.clip(init_anisotropy_strength, 0.0, 1.0)),
        depth_point_count=max(int(depth_point_count), 1),
        diffused_point_count=max(int(diffused_point_count), 1),
        fibonacci_sphere_point_count=max(int(fibonacci_sphere_point_count), 0),
        fibonacci_sphere_radius_multiplier=max(float(fibonacci_sphere_radius_multiplier), 0.0),
        fibonacci_sphere_color=resolved_fibonacci_sphere_color,
        fibonacci_sphere_upper_hemisphere_only=bool(fibonacci_sphere_upper_hemisphere_only),
        target_alpha_mode=resolved_target_alpha_mode,
        target_alpha_threshold=resolved_target_alpha_threshold,
        use_target_alpha_mask=target_alpha_skip_mask_enabled(resolved_target_alpha_mode),
        pointcloud_enabled=resolved_pointcloud_enabled,
        pointcloud_nn_radius_scale_coef=resolved_pointcloud_nn_radius_scale_coef,
        diffused_enabled=resolved_diffused_enabled,
        diffused_diffusion_radius=resolved_diffused_diffusion_radius,
        diffused_nn_radius_scale_coef=resolved_diffused_nn_radius_scale_coef,
        custom_ply_enabled=resolved_custom_ply_enabled,
        custom_ply_nn_radius_scale_coef=resolved_custom_ply_nn_radius_scale_coef,
        custom_mesh_enabled=resolved_custom_mesh_enabled,
        custom_mesh_path=None if resolved_custom_mesh_path is None else Path(resolved_custom_mesh_path).resolve(),
        custom_mesh_point_count=resolved_custom_mesh_point_count,
        custom_mesh_nn_radius_scale_coef=resolved_custom_mesh_nn_radius_scale_coef,
        fibonacci_sphere_enabled=resolved_fibonacci_sphere_enabled,
        fibonacci_sphere_nn_radius_scale_coef=resolved_fibonacci_sphere_nn_radius_scale_coef,
    )
    _set_ui_path(viewer, "colmap_root_path", dataset_root)
    _set_ui_path(viewer, "colmap_database_path", database_path)
    _set_ui_path(viewer, "colmap_images_root", images_root)
    _set_ui_path(viewer, "colmap_alpha_mask_root", alpha_mask_root)
    viewer.ui._values["colmap_use_alpha_masks"] = bool(use_alpha_masks and alpha_mask_root is not None)
    _set_ui_path(viewer, "colmap_depth_root", depth_root)
    viewer.ui._values["colmap_selected_camera_ids"] = tuple(int(camera_id) for camera_id in selected_camera_ids)
    viewer.ui._values["colmap_depth_value_mode"] = 0 if str(depth_value_mode) == _COLMAP_DEPTH_VALUE_DISTANCE else 1
    viewer.ui._values["colmap_init_mode"] = (
        1 if str(init_mode) == _COLMAP_IMPORT_DEPTH else 0
    )
    viewer.ui._values["colmap_rotation_mode"] = resolved_rotation_mode
    viewer.ui._values["colmap_custom_rotation_deg"] = resolved_custom_rotation_deg
    viewer.ui._values["compress_dataset_using_bc7"] = bool(compress_dataset_using_bc7)
    viewer.ui._values["colmap_training_image_color_init"] = bool(training_image_color_init)
    viewer.ui._values["colmap_photometric_compensation_enabled"] = bool(photometric_compensation_enabled)
    _set_ui_path(viewer, "colmap_custom_ply_path", custom_ply_path)
    viewer.ui._values["colmap_image_downscale_mode"] = 1 if str(image_downscale_mode) == _COLMAP_IMAGE_DOWNSCALE_MAX_SIZE else 2 if str(image_downscale_mode) == _COLMAP_IMAGE_DOWNSCALE_SCALE else 0
    viewer.ui._values["colmap_image_max_size"] = max(int(image_downscale_max_size), 1)
    viewer.ui._values["colmap_image_scale"] = float(np.clip(image_downscale_scale, 1e-6, 1.0))
    viewer.ui._values["colmap_nn_radius_scale_coef"] = float(max(nn_radius_scale_coef, 1e-4))
    viewer.ui._values["colmap_min_track_length"] = max(int(min_track_length), 0)
    viewer.ui._values["colmap_init_neighbor_count"] = max(int(init_neighbor_count), 2)
    viewer.ui._values["colmap_init_anisotropy_strength"] = float(np.clip(init_anisotropy_strength, 0.0, 1.0))
    viewer.ui._values["colmap_depth_point_count"] = max(int(depth_point_count), 1)
    viewer.ui._values["colmap_pointcloud_enabled"] = resolved_pointcloud_enabled
    viewer.ui._values["colmap_pointcloud_nn_radius_scale_coef"] = resolved_pointcloud_nn_radius_scale_coef
    viewer.ui._values["colmap_diffused_enabled"] = resolved_diffused_enabled
    viewer.ui._values["colmap_diffused_point_count"] = max(int(diffused_point_count), 1)
    viewer.ui._values["colmap_diffused_diffusion_radius"] = resolved_diffused_diffusion_radius
    viewer.ui._values["colmap_diffused_nn_radius_scale_coef"] = resolved_diffused_nn_radius_scale_coef
    viewer.ui._values["colmap_custom_ply_enabled"] = resolved_custom_ply_enabled
    viewer.ui._values["colmap_custom_ply_nn_radius_scale_coef"] = resolved_custom_ply_nn_radius_scale_coef
    viewer.ui._values["colmap_custom_mesh_enabled"] = resolved_custom_mesh_enabled
    _set_ui_path(viewer, "colmap_custom_mesh_path", resolved_custom_mesh_path)
    viewer.ui._values["colmap_custom_mesh_point_count"] = resolved_custom_mesh_point_count
    viewer.ui._values["colmap_custom_mesh_nn_radius_scale_coef"] = resolved_custom_mesh_nn_radius_scale_coef
    viewer.ui._values["colmap_fibonacci_sphere_enabled"] = resolved_fibonacci_sphere_enabled
    viewer.ui._values["colmap_fibonacci_sphere_point_count"] = max(int(fibonacci_sphere_point_count), 0)
    viewer.ui._values["colmap_fibonacci_sphere_radius_multiplier"] = max(float(fibonacci_sphere_radius_multiplier), 0.0)
    viewer.ui._values["colmap_fibonacci_sphere_color"] = resolved_fibonacci_sphere_color
    viewer.ui._values["colmap_fibonacci_sphere_upper_hemisphere_only"] = bool(fibonacci_sphere_upper_hemisphere_only)
    viewer.ui._values["colmap_fibonacci_sphere_nn_radius_scale_coef"] = resolved_fibonacci_sphere_nn_radius_scale_coef
    viewer.ui._values["target_alpha_mode"] = resolved_target_alpha_mode
    viewer.ui._values["target_alpha_threshold"] = resolved_target_alpha_threshold
    viewer.ui._values["use_target_alpha_mask"] = target_alpha_skip_mask_enabled(resolved_target_alpha_mode)


def _load_aligned_colmap_reconstruction(
    colmap_root: Path,
    rotation_mode: int = COLMAP_ROTATION_MODE_AUTO,
    custom_rotation_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    recon = load_colmap_reconstruction(Path(colmap_root).resolve())
    resolved_rotation_mode = min(max(int(rotation_mode), COLMAP_ROTATION_MODE_NONE), COLMAP_ROTATION_MODE_AUTO)
    if resolved_rotation_mode == COLMAP_ROTATION_MODE_NONE:
        return recon
    if resolved_rotation_mode == COLMAP_ROTATION_MODE_CUSTOM:
        from ..scene._internal.colmap_ops import transform_colmap_reconstruction_custom_rotation

        aligned_recon, _ = transform_colmap_reconstruction_custom_rotation(recon, custom_rotation_deg)
        return aligned_recon
    from ..scene import transform_colmap_reconstruction_pca
    aligned_recon, _ = transform_colmap_reconstruction_pca(recon)
    return aligned_recon
