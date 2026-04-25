from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np

from ..scene._internal.colmap_ops import DEPTH_INIT_VALUE_DISTANCE, DEPTH_INIT_VALUE_Z_DEPTH
from ..scene import load_colmap_reconstruction
from ..scene._internal.colmap_types import ColmapFrame
from .state import ColmapImportSettings

_COLMAP_IMPORT_POINTCLOUD = "pointcloud"
_COLMAP_IMPORT_DIFFUSED_POINTCLOUD = "diffused_pointcloud"
_COLMAP_IMPORT_CUSTOM_PLY = "custom_ply"
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
}
_COLMAP_DB_SAMPLE_LIMIT = 64
_COLMAP_DB_SEARCH_PATTERNS = ("database.db", "*.db", "*.sqlite", "*.sqlite3")
_COLMAP_IMPORT_IMAGES_PER_TICK = 1


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
    sparse_dir = Path(root).resolve() / "sparse" / "0"
    return all((sparse_dir / name).exists() for name in ("cameras.bin", "images.bin", "points3D.bin")) or all(
        (sparse_dir / name).exists() for name in ("cameras.txt", "images.txt", "points3D.txt")
    )


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


def _suggest_images_root_from_dataset_root(dataset_root: Path, image_names: list[str]) -> Path:
    root = Path(dataset_root).resolve()
    sample_names = image_names[: min(len(image_names), _COLMAP_DB_SAMPLE_LIMIT)]
    for candidate in _dataset_directories(root):
        if _looks_like_depth_directory(candidate):
            continue
        if any((candidate / image_name).exists() for image_name in sample_names):
            return candidate
    raise FileNotFoundError(f"Could not find an image folder under {root} for COLMAP images: {sample_names[:4]}")


def _camera_rows(recon: object) -> tuple[dict[str, object], ...]:
    frame_counts: dict[int, int] = {}
    for image in getattr(recon, "images", {}).values():
        camera_id = int(getattr(image, "camera_id", -1))
        frame_counts[camera_id] = frame_counts.get(camera_id, 0) + 1
    rows: list[dict[str, object]] = []
    for camera_id, camera in sorted(getattr(recon, "cameras", {}).items()):
        rows.append(
            {
                "camera_id": int(camera_id),
                "model_name": _COLMAP_CAMERA_MODEL_NAMES.get(int(getattr(camera, "model_id", -1)), f"MODEL_{int(getattr(camera, 'model_id', -1))}"),
                "frame_count": int(frame_counts.get(int(camera_id), 0)),
                "resolution_text": f"{int(camera.width)}x{int(camera.height)}",
                "focal_text": f"{float(camera.fx):.2f}, {float(camera.fy):.2f}",
                "principal_text": f"{float(camera.cx):.2f}, {float(camera.cy):.2f}",
                "distortion_text": f"{float(getattr(camera, 'k1', 0.0)):.4g}, {float(getattr(camera, 'k2', 0.0)):.4g}",
            }
        )
    return tuple(rows)


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
    viewer.ui._values["colmap_selected_camera_ids"] = selected_ids
    return selected_ids


def _update_import_settings(
    viewer: object,
    *,
    dataset_root: Path,
    database_path: Path | None,
    images_root: Path,
    depth_root: Path | None,
    selected_camera_ids: tuple[int, ...],
    depth_value_mode: str,
    init_mode: str,
    compress_dataset_using_bc7: bool,
    custom_ply_path: Path | None,
    image_downscale_mode: str,
    image_downscale_max_size: int,
    image_downscale_scale: float,
    nn_radius_scale_coef: float,
    min_track_length: int,
    depth_point_count: int,
    diffused_point_count: int,
    diffusion_radius: float,
    fibonacci_sphere_point_count: int,
    fibonacci_sphere_radius: float,
    use_target_alpha_mask: bool,
) -> None:
    viewer.s.colmap_import = ColmapImportSettings(
        database_path=None if database_path is None else Path(database_path).resolve(),
        images_root=Path(images_root).resolve(),
        depth_root=None if depth_root is None else Path(depth_root).resolve(),
        selected_camera_ids=tuple(int(camera_id) for camera_id in selected_camera_ids),
        depth_value_mode=str(depth_value_mode),
        init_mode=str(init_mode),
        compress_dataset_using_bc7=bool(compress_dataset_using_bc7),
        custom_ply_path=None if custom_ply_path is None else Path(custom_ply_path).resolve(),
        image_downscale_mode=str(image_downscale_mode),
        image_downscale_max_size=max(int(image_downscale_max_size), 1),
        image_downscale_scale=float(np.clip(image_downscale_scale, 1e-6, 1.0)),
        nn_radius_scale_coef=float(max(nn_radius_scale_coef, 1e-4)),
        min_track_length=max(int(min_track_length), 0),
        depth_point_count=max(int(depth_point_count), 1),
        diffused_point_count=max(int(diffused_point_count), 1),
        diffusion_radius=max(float(diffusion_radius), 0.0),
        fibonacci_sphere_point_count=max(int(fibonacci_sphere_point_count), 0),
        fibonacci_sphere_radius=max(float(fibonacci_sphere_radius), 0.0),
        use_target_alpha_mask=bool(use_target_alpha_mask),
    )
    _set_ui_path(viewer, "colmap_root_path", dataset_root)
    _set_ui_path(viewer, "colmap_database_path", database_path)
    _set_ui_path(viewer, "colmap_images_root", images_root)
    _set_ui_path(viewer, "colmap_depth_root", depth_root)
    viewer.ui._values["colmap_selected_camera_ids"] = tuple(int(camera_id) for camera_id in selected_camera_ids)
    viewer.ui._values["colmap_depth_value_mode"] = 0 if str(depth_value_mode) == _COLMAP_DEPTH_VALUE_DISTANCE else 1
    viewer.ui._values["colmap_init_mode"] = (
        1 if str(init_mode) == _COLMAP_IMPORT_DIFFUSED_POINTCLOUD else
        2 if str(init_mode) == _COLMAP_IMPORT_CUSTOM_PLY else
        3 if str(init_mode) == _COLMAP_IMPORT_DEPTH else
        0
    )
    viewer.ui._values["compress_dataset_using_bc7"] = bool(compress_dataset_using_bc7)
    _set_ui_path(viewer, "colmap_custom_ply_path", custom_ply_path)
    viewer.ui._values["colmap_image_downscale_mode"] = 1 if str(image_downscale_mode) == _COLMAP_IMAGE_DOWNSCALE_MAX_SIZE else 2 if str(image_downscale_mode) == _COLMAP_IMAGE_DOWNSCALE_SCALE else 0
    viewer.ui._values["colmap_image_max_size"] = max(int(image_downscale_max_size), 1)
    viewer.ui._values["colmap_image_scale"] = float(np.clip(image_downscale_scale, 1e-6, 1.0))
    viewer.ui._values["colmap_nn_radius_scale_coef"] = float(max(nn_radius_scale_coef, 1e-4))
    viewer.ui._values["colmap_min_track_length"] = max(int(min_track_length), 0)
    viewer.ui._values["colmap_depth_point_count"] = max(int(depth_point_count), 1)
    viewer.ui._values["colmap_diffused_point_count"] = max(int(diffused_point_count), 1)
    viewer.ui._values["colmap_diffusion_radius"] = max(float(diffusion_radius), 0.0)
    viewer.ui._values["colmap_fibonacci_sphere_point_count"] = max(int(fibonacci_sphere_point_count), 0)
    viewer.ui._values["colmap_fibonacci_sphere_radius"] = max(float(fibonacci_sphere_radius), 0.0)
    viewer.ui._values["use_target_alpha_mask"] = bool(use_target_alpha_mask)


def _load_aligned_colmap_reconstruction(colmap_root: Path):
    recon = load_colmap_reconstruction(Path(colmap_root).resolve())
    from ..scene import transform_colmap_reconstruction_pca
    aligned_recon, _ = transform_colmap_reconstruction_pca(recon)
    return aligned_recon