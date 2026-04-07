from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import sqlite3
import time

import numpy as np
from PIL import Image
import slangpy as spy

from ..app.shared import apply_training_profile, estimate_point_bounds, estimate_scene_bounds, renderer_kwargs
from ..common import SHADER_ROOT, clamp_index
from ..metrics import ParamTensorRanges
from ..renderer import GaussianRenderSettings, GaussianRenderer
from ..scene import (
    GaussianScene,
    build_training_frames_from_root,
    initialize_scene_from_colmap_diffused_points,
    initialize_scene_from_colmap_points,
    initialize_scene_from_points_colors,
    load_colmap_reconstruction,
    load_gaussian_ply,
    resolve_colmap_init_hparams,
    sample_colmap_diffused_points,
    transform_colmap_reconstruction_pca,
)
from ..scene._internal.colmap_ops import point_nn_scales, resolve_training_frame_image_size
from ..scene._internal.colmap_ops import (
    TRAINING_FRAME_LOAD_THREADS,
    build_depth_path_index,
    generate_depth_init_points,
    load_rgba8_image,
    load_training_frame_rgba8,
    load_training_frame_rgba8_with_depth_payload,
    match_depth_path,
)
from ..training import GaussianTrainer, resolve_effective_train_render_factor, resolve_training_resolution
from ..scene._internal.colmap_types import ColmapFrame, point_tables
from .state import ColmapImportProgress, ColmapImportSettings, SceneCountProxy

_COLMAP_IMPORT_POINTCLOUD = "pointcloud"
_COLMAP_IMPORT_DIFFUSED_POINTCLOUD = "diffused_pointcloud"
_COLMAP_IMPORT_CUSTOM_PLY = "custom_ply"
_COLMAP_IMPORT_DEPTH = "depth"
_COLMAP_IMAGE_DOWNSCALE_ORIGINAL = "original"
_COLMAP_IMAGE_DOWNSCALE_MAX_SIZE = "max_size"
_COLMAP_IMAGE_DOWNSCALE_SCALE = "scale"
_COLMAP_DB_SAMPLE_LIMIT = 64
_COLMAP_DB_SEARCH_PATTERNS = ("database.db", "*.db", "*.sqlite", "*.sqlite3")
_COLMAP_IMPORT_IMAGES_PER_TICK = 1
_TRAINING_RUNTIME_PARAM_NAMES = (
    "train_downscale_mode",
    "train_auto_start_downscale",
    "train_downscale_base_iters",
    "train_downscale_iter_step",
    "train_downscale_max_iters",
    "train_downscale_factor",
    "train_subsample_factor",
)


def _load_aligned_colmap_reconstruction(colmap_root: Path):
    recon = load_colmap_reconstruction(Path(colmap_root).resolve())
    aligned_recon, _ = transform_colmap_reconstruction_pca(recon)
    return aligned_recon


def _clear(viewer: object, *attrs: str) -> None:
    for attr in attrs:
        value = getattr(viewer.s, attr)
        if value is not None:
            setattr(viewer.s, attr, None)
            del value


def _apply_initial_camera_fit(viewer: object, fallback_bounds: object | None = None, fallback_factory=None) -> None:
    fit_training_views = getattr(viewer, "apply_camera_fit_to_training_views", None)
    if callable(fit_training_views) and fit_training_views(getattr(viewer.s, "training_frames", ())):
        return
    if fallback_factory is not None:
        fallback_bounds = fallback_factory()
    viewer.apply_camera_fit(fallback_bounds)


def _point_tables(recon: object) -> tuple[np.ndarray, np.ndarray]:
    xyz, rgb = point_tables(recon)
    if xyz.shape[0] != rgb.shape[0] or xyz.shape[0] <= 0:
        raise RuntimeError("COLMAP point tables are empty or mismatched.")
    return xyz, rgb


def _ui_path_string(viewer: object, key: str) -> str:
    return str(viewer.ui._values.get(key, "")).strip()


def _set_ui_path(viewer: object, key: str, path: Path | None) -> None:
    viewer.ui._values[key] = "" if path is None else str(Path(path).resolve())


def _ui_import_mode(viewer: object) -> str:
    mode_idx = int(viewer.ui._values.get("colmap_init_mode", 0))
    if mode_idx == 1: return _COLMAP_IMPORT_DIFFUSED_POINTCLOUD
    if mode_idx == 2: return _COLMAP_IMPORT_CUSTOM_PLY
    if mode_idx == 3: return _COLMAP_IMPORT_DEPTH
    return _COLMAP_IMPORT_POINTCLOUD


def _ui_image_downscale_mode(viewer: object) -> str:
    mode_idx = int(viewer.ui._values.get("colmap_image_downscale_mode", 0))
    if mode_idx == 1:
        return _COLMAP_IMAGE_DOWNSCALE_MAX_SIZE
    if mode_idx == 2:
        return _COLMAP_IMAGE_DOWNSCALE_SCALE
    return _COLMAP_IMAGE_DOWNSCALE_ORIGINAL


def _profile_images_subdir(viewer: object) -> str | None:
    import_state = getattr(viewer.s, "colmap_import", None)
    images_root = None if import_state is None else import_state.images_root
    if viewer.s.colmap_root is None or images_root is None:
        return None
    root = Path(viewer.s.colmap_root).resolve()
    images_path = Path(images_root).resolve()
    try:
        return str(images_path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return images_path.name


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
    return all((sparse_dir / name).exists() for name in ("cameras.bin", "images.bin", "points3D.bin"))


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


def _reconstruction_image_names(colmap_root: Path, limit: int = _COLMAP_DB_SAMPLE_LIMIT) -> list[str]:
    recon = load_colmap_reconstruction(Path(colmap_root).resolve())
    names = [str(image.name).strip() for _, image in sorted(recon.images.items()) if str(image.name).strip()]
    if not names:
        raise RuntimeError(f"COLMAP reconstruction has no image names: {Path(colmap_root).resolve()}")
    return names[: int(max(limit, 1))]


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


def _update_import_settings(
    viewer: object,
    *,
    dataset_root: Path,
    database_path: Path | None,
    images_root: Path,
    depth_root: Path | None,
    init_mode: str,
    custom_ply_path: Path | None,
    image_downscale_mode: str,
    image_downscale_max_size: int,
    image_downscale_scale: float,
    nn_radius_scale_coef: float,
    depth_point_count: int,
    diffused_point_count: int,
    diffusion_radius: float,
) -> None:
    viewer.s.colmap_import = ColmapImportSettings(
        database_path=None if database_path is None else Path(database_path).resolve(),
        images_root=Path(images_root).resolve(),
        depth_root=None if depth_root is None else Path(depth_root).resolve(),
        init_mode=str(init_mode),
        custom_ply_path=None if custom_ply_path is None else Path(custom_ply_path).resolve(),
        image_downscale_mode=str(image_downscale_mode),
        image_downscale_max_size=max(int(image_downscale_max_size), 1),
        image_downscale_scale=float(np.clip(image_downscale_scale, 1e-6, 1.0)),
        nn_radius_scale_coef=float(max(nn_radius_scale_coef, 1e-4)),
        depth_point_count=max(int(depth_point_count), 1),
        diffused_point_count=max(int(diffused_point_count), 1),
        diffusion_radius=max(float(diffusion_radius), 0.0),
    )
    _set_ui_path(viewer, "colmap_root_path", dataset_root)
    _set_ui_path(viewer, "colmap_database_path", database_path)
    _set_ui_path(viewer, "colmap_images_root", images_root)
    _set_ui_path(viewer, "colmap_depth_root", depth_root)
    viewer.ui._values["colmap_init_mode"] = (
        1 if str(init_mode) == _COLMAP_IMPORT_DIFFUSED_POINTCLOUD else
        2 if str(init_mode) == _COLMAP_IMPORT_CUSTOM_PLY else
        3 if str(init_mode) == _COLMAP_IMPORT_DEPTH else
        0
    )
    _set_ui_path(viewer, "colmap_custom_ply_path", custom_ply_path)
    viewer.ui._values["colmap_image_downscale_mode"] = 1 if str(image_downscale_mode) == _COLMAP_IMAGE_DOWNSCALE_MAX_SIZE else 2 if str(image_downscale_mode) == _COLMAP_IMAGE_DOWNSCALE_SCALE else 0
    viewer.ui._values["colmap_image_max_size"] = max(int(image_downscale_max_size), 1)
    viewer.ui._values["colmap_image_scale"] = float(np.clip(image_downscale_scale, 1e-6, 1.0))
    viewer.ui._values["colmap_nn_radius_scale_coef"] = float(max(nn_radius_scale_coef, 1e-4))
    viewer.ui._values["colmap_depth_point_count"] = max(int(depth_point_count), 1)
    viewer.ui._values["colmap_diffused_point_count"] = max(int(diffused_point_count), 1)
    viewer.ui._values["colmap_diffusion_radius"] = max(float(diffusion_radius), 0.0)


def _append_training_frame(progress: ColmapImportProgress, image_id: int, image: object) -> None:
    if progress.recon is None:
        raise RuntimeError("COLMAP import progress is missing reconstruction state.")
    image_path = (progress.images_root / image.name).resolve()
    camera = progress.recon.cameras.get(image.camera_id)
    if camera is None or not image_path.exists():
        return
    with Image.open(image_path) as pil_image:
        src_width, src_height = pil_image.size
    width, height = resolve_training_frame_image_size(
        src_width,
        src_height,
        downscale_mode=progress.image_downscale_mode,
        downscale_max_size=progress.image_downscale_max_size,
        downscale_scale=progress.image_downscale_scale,
    )
    sx, sy = float(width) / float(camera.width), float(height) / float(camera.height)
    frame = ColmapFrame(
        image_id,
        image_path,
        image.q_wxyz.astype(np.float32),
        image.t_xyz.astype(np.float32),
        float(camera.fx) * sx,
        float(camera.fy) * sy,
        float(camera.cx) * sx,
        float(camera.cy) * sy,
        int(width),
        int(height),
        float(getattr(camera, "k1", 0.0)),
        float(getattr(camera, "k2", 0.0)),
    )
    if progress.init_mode == _COLMAP_IMPORT_DEPTH:
        progress.frame_images.append(image)
        progress.depth_paths.append(None if progress.depth_index is None else match_depth_path(progress.images_root, image_path, progress.depth_index))
    progress.frames.append(frame)


def _create_native_dataset_texture(viewer: object, image_path: Path, *, target_size: tuple[int, int] | None = None) -> spy.Texture:
    return _create_native_dataset_texture_from_rgba8(viewer, load_rgba8_image(image_path, target_size=target_size))


def _create_native_dataset_texture_from_rgba8(viewer: object, rgba8: np.ndarray) -> spy.Texture:
    texture = viewer.device.create_texture(
        format=spy.Format.rgba8_unorm_srgb,
        width=int(rgba8.shape[1]),
        height=int(rgba8.shape[0]),
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
    )
    texture.copy_from_numpy(np.ascontiguousarray(rgba8, dtype=np.uint8))
    return texture


def _close_colmap_texture_loader(progress: ColmapImportProgress) -> None:
    loader = progress.native_rgba8_loader
    progress.native_rgba8_loader = None
    progress.native_rgba8_iter = None
    if loader is not None:
        loader.shutdown(wait=False, cancel_futures=False)


def _start_colmap_texture_loader(progress: ColmapImportProgress) -> None:
    _close_colmap_texture_loader(progress)
    loader = ThreadPoolExecutor(max_workers=TRAINING_FRAME_LOAD_THREADS, thread_name_prefix="viewer-target")
    progress.native_rgba8_loader = loader
    if progress.init_mode == _COLMAP_IMPORT_DEPTH:
        if progress.recon is None:
            raise RuntimeError("Depth import mode requires reconstruction state before texture loading.")
        tasks = []
        for frame, image, depth_path in zip(progress.frames, progress.frame_images, progress.depth_paths, strict=False):
            camera = progress.recon.cameras.get(image.camera_id)
            if camera is None:
                raise RuntimeError(f"Missing COLMAP camera {image.camera_id} for {image.name}.")
            tasks.append((progress.recon, image, camera, frame, depth_path))
        progress.native_rgba8_iter = loader.map(load_training_frame_rgba8_with_depth_payload, tasks)
        return
    progress.native_rgba8_iter = loader.map(load_training_frame_rgba8, progress.frames)


def _create_native_dataset_textures(viewer: object, frames: list[ColmapFrame]) -> list[spy.Texture]:
    textures: list[spy.Texture] = []
    with ThreadPoolExecutor(max_workers=TRAINING_FRAME_LOAD_THREADS, thread_name_prefix="viewer-target") as executor:
        for rgba8 in executor.map(load_training_frame_rgba8, frames):
            textures.append(_create_native_dataset_texture_from_rgba8(viewer, rgba8))
    return textures


def _build_depth_init_source(
    recon: object,
    images_root: Path,
    frames: list[ColmapFrame],
    *,
    depth_root: Path,
    depth_point_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    depth_index = build_depth_path_index(depth_root)
    tasks = []
    for frame in frames:
        image = recon.images.get(int(frame.image_id))
        if image is None:
            continue
        camera = recon.cameras.get(int(image.camera_id))
        if camera is None:
            continue
        depth_path = match_depth_path(images_root, frame.image_path, depth_index)
        if depth_path is None:
            continue
        tasks.append((recon, image, camera, frame, depth_path))
    if len(tasks) == 0:
        raise RuntimeError("Depth initialization found no matched RGB/depth frame pairs.")
    payloads = []
    with ThreadPoolExecutor(max_workers=TRAINING_FRAME_LOAD_THREADS, thread_name_prefix="viewer-depth") as executor:
        for _, payload in executor.map(load_training_frame_rgba8_with_depth_payload, tasks):
            if payload is not None:
                payloads.append(payload)
    return generate_depth_init_points(payloads, depth_point_count, seed)


def choose_colmap_root(viewer: object, dataset_root: Path) -> None:
    root = Path(dataset_root).resolve()
    colmap_root = _resolve_colmap_root_from_selection(root)
    db_path = _find_optional_colmap_database(root)
    image_names = _reconstruction_image_names(colmap_root) if db_path is None else _database_image_names(db_path)
    _set_ui_path(viewer, "colmap_root_path", root)
    _set_ui_path(viewer, "colmap_database_path", db_path)
    _set_ui_path(viewer, "colmap_images_root", _suggest_images_root_from_dataset_root(root, image_names))
    viewer.s.last_error = ""


def choose_colmap_images_root(viewer: object, images_root: Path) -> None:
    _set_ui_path(viewer, "colmap_images_root", Path(images_root).resolve())
    viewer.s.last_error = ""


def choose_colmap_depth_root(viewer: object, depth_root: Path) -> None:
    _set_ui_path(viewer, "colmap_depth_root", Path(depth_root).resolve())
    viewer.s.last_error = ""


def choose_colmap_custom_ply(viewer: object, ply_path: Path) -> None:
    _set_ui_path(viewer, "colmap_custom_ply_path", Path(ply_path).resolve())
    viewer.s.last_error = ""


def _pointcloud_init_hparams(recon: object, max_gaussians: int, init_hparams: object, nn_radius_scale_coef: float):
    resolved = resolve_colmap_init_hparams(recon, max_gaussians, init_hparams)
    xyz, _ = _point_tables(recon)
    chosen_count = xyz.shape[0] if max_gaussians <= 0 else min(max(int(max_gaussians), 1), xyz.shape[0])
    median_nn_scale = float(np.median(point_nn_scales(np.ascontiguousarray(xyz[:chosen_count], dtype=np.float32)))) if chosen_count > 0 else 1.0
    return replace(resolved, base_scale=float(max(float(nn_radius_scale_coef), 1e-4) * max(median_nn_scale, 1e-6)))


def _pointcloud_init_hparams_from_positions(recon: object, positions: np.ndarray, max_gaussians: int, init_hparams: object, nn_radius_scale_coef: float):
    resolved = resolve_colmap_init_hparams(recon, max_gaussians, init_hparams)
    chosen_count = positions.shape[0] if max_gaussians <= 0 else min(max(int(max_gaussians), 1), positions.shape[0])
    chosen_positions = np.ascontiguousarray(positions[:chosen_count], dtype=np.float32)
    median_nn_scale = float(np.median(point_nn_scales(chosen_positions))) if chosen_count > 0 else 1.0
    return replace(resolved, base_scale=float(max(float(nn_radius_scale_coef), 1e-4) * max(median_nn_scale, 1e-6)))


def _diffused_pointcloud_init_hparams(
    recon: object,
    point_count: int,
    diffusion_radius: float,
    seed: int,
    init_hparams: object,
    nn_radius_scale_coef: float,
):
    resolved = resolve_colmap_init_hparams(recon, point_count, init_hparams)
    positions, _ = sample_colmap_diffused_points(recon, point_count, diffusion_radius, seed)
    median_nn_scale = float(np.median(point_nn_scales(np.ascontiguousarray(positions, dtype=np.float32)))) if positions.shape[0] > 0 else 1.0
    return replace(resolved, base_scale=float(max(float(nn_radius_scale_coef), 1e-4) * max(median_nn_scale, 1e-6)))


def _diffused_pointcloud_init_hparams_from_positions(recon: object, positions: np.ndarray, init_hparams: object, nn_radius_scale_coef: float):
    resolved = resolve_colmap_init_hparams(recon, int(positions.shape[0]), init_hparams)
    chosen_positions = np.ascontiguousarray(positions, dtype=np.float32)
    median_nn_scale = float(np.median(point_nn_scales(chosen_positions))) if chosen_positions.shape[0] > 0 else 1.0
    return replace(resolved, base_scale=float(max(float(nn_radius_scale_coef), 1e-4) * max(median_nn_scale, 1e-6)))


def _copy_gaussian_scene(scene: GaussianScene) -> GaussianScene:
    return GaussianScene(
        positions=np.array(scene.positions, dtype=np.float32, copy=True),
        scales=np.array(scene.scales, dtype=np.float32, copy=True),
        rotations=np.array(scene.rotations, dtype=np.float32, copy=True),
        opacities=np.array(scene.opacities, dtype=np.float32, copy=True),
        colors=np.array(scene.colors, dtype=np.float32, copy=True),
        sh_coeffs=np.array(scene.sh_coeffs, dtype=np.float32, copy=True),
    )


def _clear_cached_init_source(viewer: object) -> None:
    setattr(viewer.s, "cached_init_point_positions", None)
    setattr(viewer.s, "cached_init_point_colors", None)
    setattr(viewer.s, "cached_init_scene", None)
    setattr(viewer.s, "cached_init_signature", None)


def _cached_init_signature(viewer: object, init: object) -> tuple[object, ...] | None:
    import_cfg = getattr(viewer.s, "colmap_import", None)
    if import_cfg is None:
        return None
    if str(import_cfg.init_mode) == _COLMAP_IMPORT_DEPTH:
        return (
            None if viewer.s.colmap_root is None else str(Path(viewer.s.colmap_root).resolve()),
            str(import_cfg.init_mode),
            None if import_cfg.images_root is None else str(Path(import_cfg.images_root).resolve()),
            None if import_cfg.depth_root is None else str(Path(import_cfg.depth_root).resolve()),
            str(import_cfg.image_downscale_mode),
            int(import_cfg.image_downscale_max_size),
            round(float(import_cfg.image_downscale_scale), 6),
            int(import_cfg.depth_point_count),
        )
    return (
        None if viewer.s.colmap_root is None else str(Path(viewer.s.colmap_root).resolve()),
        str(import_cfg.init_mode),
        None if import_cfg.custom_ply_path is None else str(Path(import_cfg.custom_ply_path).resolve()),
        int(import_cfg.diffused_point_count),
        round(float(import_cfg.diffusion_radius), 6),
        int(init.seed),
    )


def _ensure_cached_init_source(viewer: object, init: object) -> None:
    if viewer.s.colmap_recon is None:
        return
    import_cfg = viewer.s.colmap_import
    signature = _cached_init_signature(viewer, init)
    cached_signature = getattr(viewer.s, "cached_init_signature", None)
    cached_scene = getattr(viewer.s, "cached_init_scene", None)
    cached_positions = getattr(viewer.s, "cached_init_point_positions", None)
    cached_colors = getattr(viewer.s, "cached_init_point_colors", None)
    if signature is not None and cached_signature == signature:
        if str(import_cfg.init_mode) == _COLMAP_IMPORT_CUSTOM_PLY:
            if cached_scene is not None:
                return
        elif cached_positions is not None and cached_colors is not None:
            return
    if import_cfg.init_mode == _COLMAP_IMPORT_DEPTH:
        raise RuntimeError("Depth initialization cache is unavailable. Re-import the dataset to rebuild the calibrated point cloud.")
    _clear_cached_init_source(viewer)
    if import_cfg.init_mode == _COLMAP_IMPORT_CUSTOM_PLY:
        if import_cfg.custom_ply_path is None:
            raise RuntimeError("Custom PLY initialization requires a selected PLY file.")
        setattr(viewer.s, "cached_init_scene", _copy_gaussian_scene(load_gaussian_ply(import_cfg.custom_ply_path)))
    elif import_cfg.init_mode == _COLMAP_IMPORT_DIFFUSED_POINTCLOUD:
        positions, colors = sample_colmap_diffused_points(viewer.s.colmap_recon, import_cfg.diffused_point_count, import_cfg.diffusion_radius, init.seed)
        setattr(viewer.s, "cached_init_point_positions", np.array(positions, dtype=np.float32, copy=True))
        setattr(viewer.s, "cached_init_point_colors", np.array(colors, dtype=np.float32, copy=True))
    else:
        positions, colors = _point_tables(viewer.s.colmap_recon)
        setattr(viewer.s, "cached_init_point_positions", np.array(positions, dtype=np.float32, copy=True))
        setattr(viewer.s, "cached_init_point_colors", np.array(colors, dtype=np.float32, copy=True))
    setattr(viewer.s, "cached_init_signature", signature)


def _invalidate(viewer: object, *targets: str) -> None:
    for target in targets or ("main", "debug"):
        setattr(viewer.s, f"synced_step_{target}", -1)


def _reset_loss_debug(viewer: object) -> None:
    viewer.s.viewport_texture = None
    viewer.s.loss_debug_texture = None
    viewer.s.debug_present_texture = None
    _clear(viewer, "debug_renderer")
    _invalidate(viewer, "debug")


def _reset_training_visual_state(viewer: object) -> None:
    if hasattr(viewer, "toolkit") and getattr(viewer, "toolkit") is not None:
        reset_plot_history = getattr(viewer.toolkit, "reset_plot_history", None)
        if callable(reset_plot_history):
            reset_plot_history()
    viewer.s.cached_raster_grad_histograms = None
    viewer.s.cached_raster_grad_ranges = None
    viewer.s.cached_raster_grad_histogram_mode = ""
    viewer.s.cached_raster_grad_histogram_step = -1
    viewer.s.cached_raster_grad_histogram_scene_count = -1
    viewer.s.cached_raster_grad_histogram_signature = None
    viewer.s.cached_raster_grad_histogram_status = ""


def _reset_training_runtime(viewer: object) -> None:
    viewer.s.scene_init_signature = None
    viewer.s.training_active = False
    viewer.s.training_elapsed_s = 0.0
    viewer.s.training_resume_time = None
    viewer.s.trainer = None
    if viewer.s.renderer is not None:
        viewer.s.renderer.set_debug_grad_norm_buffer(None)
        viewer.s.renderer.set_debug_clone_count_buffer(None)
    viewer.s.applied_renderer_params_training = None
    viewer.s.applied_renderer_params_debug = None
    viewer.s.applied_training_signature = None
    viewer.s.applied_training_runtime_signature = None
    viewer.s.applied_training_runtime_factor = None
    viewer.s.cached_training_setup_signature = None
    viewer.s.cached_training_setup = None
    viewer.s.pending_training_runtime_resize = False
    _reset_training_visual_state(viewer)
    _reset_loss_debug(viewer)
    _clear(viewer, "training_renderer")


def _reset_loaded_runtime(viewer: object) -> None:
    viewer.s.colmap_import_progress = None
    _reset_training_runtime(viewer)
    viewer.s.colmap_point_positions_buffer = viewer.s.colmap_point_colors_buffer = None
    viewer.s.colmap_point_count = 0
    _clear_cached_init_source(viewer)
    viewer.s.suggested_init_hparams = viewer.s.suggested_init_count = None
    viewer.s.applied_renderer_params_main = None
    viewer.s.cached_training_setup_signature = None
    viewer.s.cached_training_setup = None
    update_debug_frame_slider_range(viewer)


def _clear_loaded_scene(viewer: object) -> None:
    _reset_loaded_runtime(viewer)
    viewer.s.scene = None
    viewer.s.scene_path = None
    viewer.s.colmap_root = None
    viewer.s.colmap_recon = None
    viewer.s.training_frames = []
    if viewer.s.renderer is not None:
        viewer.s.renderer.clear_scene_resources()


def _scene_signature(viewer: object):
    if viewer.s.colmap_root is None or viewer.s.colmap_recon is None or not viewer.s.training_frames:
        return None
    init = viewer.init_params()
    import_cfg = viewer.s.colmap_import
    return (
        str(viewer.s.colmap_root.resolve()),
        len(viewer.s.training_frames),
        init.seed,
        str(import_cfg.init_mode),
        None if import_cfg.images_root is None else str(import_cfg.images_root.resolve()),
        None if import_cfg.depth_root is None else str(import_cfg.depth_root.resolve()),
        None if import_cfg.custom_ply_path is None else str(import_cfg.custom_ply_path.resolve()),
        str(import_cfg.image_downscale_mode),
        int(import_cfg.image_downscale_max_size),
        round(float(import_cfg.image_downscale_scale), 6),
        round(float(import_cfg.nn_radius_scale_coef), 6),
        int(import_cfg.depth_point_count),
        int(import_cfg.diffused_point_count),
        round(float(import_cfg.diffusion_radius), 6),
        None if init.hparams.initial_opacity is None else round(float(init.hparams.initial_opacity), 8),
    )


def create_debug_shaders(viewer: object) -> None:
    shader_path = str(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang")
    viewer.s.debug_abs_diff_kernel = viewer.device.create_compute_kernel(viewer.device.load_program(shader_path, ["csComposeAbsDiffDebug"]))
    viewer.s.debug_edge_kernel = viewer.device.create_compute_kernel(viewer.device.load_program(shader_path, ["csComposeEdgeDebug"]))
    viewer.s.debug_letterbox_kernel = viewer.device.create_compute_kernel(viewer.device.load_program(shader_path, ["csComposeLetterboxDebug"]))


def update_debug_frame_slider_range(viewer: object) -> None:
    slider = viewer.c("loss_debug_frame")
    max_index = max(len(viewer.s.training_frames) - 1, 0)
    if hasattr(slider, "min"):
        slider.min = 0
        slider.max = int(max_index)
    slider.value = clamp_index(int(slider.value), max_index + 1)


def _training_debug_clone_count_buffer(viewer: object):
    return (
        viewer.s.trainer.refinement_buffers["clone_counts"]
        if viewer.s.trainer is not None and "clone_counts" in viewer.s.trainer.refinement_buffers
        else None
    )


def _training_debug_splat_contribution_buffer(viewer: object):
    return (
        viewer.s.trainer.refinement_buffers["splat_contribution"]
        if viewer.s.trainer is not None and "splat_contribution" in viewer.s.trainer.refinement_buffers
        else None
    )


def _training_debug_adam_moments_buffer(viewer: object):
    return viewer.s.trainer.adam_optimizer.buffers["adam_moments"] if viewer.s.trainer is not None else None


def _apply_debug_buffers(viewer: object, renderer: GaussianRenderer | None) -> None:
    if renderer is None:
        return
    renderer.set_debug_grad_norm_buffer(
        viewer.s.training_renderer.work_buffers["debug_grad_norm"]
        if viewer.s.training_renderer is not None and viewer.s.trainer is not None
        else None
    )
    renderer.set_debug_clone_count_buffer(_training_debug_clone_count_buffer(viewer))
    bind_contribution = getattr(renderer, "set_debug_splat_contribution_buffer", None)
    if callable(bind_contribution):
        bind_contribution(_training_debug_splat_contribution_buffer(viewer))
    bind_adam_moments = getattr(renderer, "set_debug_adam_moments_buffer", None)
    if callable(bind_adam_moments):
        bind_adam_moments(_training_debug_adam_moments_buffer(viewer))
    set_contribution_pixels = getattr(renderer, "set_debug_contribution_observed_pixel_count", None)
    if callable(set_contribution_pixels):
        set_contribution_pixels(0 if viewer.s.trainer is None else viewer.s.trainer.observed_contribution_pixel_count)


def ensure_renderer(viewer: object, attr: str, width: int, height: int, allow_debug_overlays: bool) -> GaussianRenderer:
    size, renderer = (int(width), int(height)), getattr(viewer.s, attr)
    if renderer is not None and (renderer.width, renderer.height) == size:
        return renderer
    previous_renderer = renderer
    renderer = GaussianRenderSettings(width=size[0], height=size[1], **renderer_kwargs(viewer.renderer_params(allow_debug_overlays))).create_renderer(viewer.device)
    if isinstance(viewer.s.scene, GaussianScene):
        renderer.set_scene(viewer.s.scene)
    setattr(viewer.s, attr, renderer)
    if attr != "training_renderer":
        _apply_debug_buffers(viewer, renderer)
    if previous_renderer is not None:
        del previous_renderer
    _invalidate(viewer, "debug" if attr == "debug_renderer" else "main", "debug")
    return renderer


def _create_renderer(viewer: object, width: int, height: int, allow_debug_overlays: bool) -> GaussianRenderer:
    return GaussianRenderSettings(width=int(width), height=int(height), **renderer_kwargs(viewer.renderer_params(allow_debug_overlays))).create_renderer(viewer.device)


def _renderer_params_signature(params: object) -> tuple[object, ...]:
    return tuple(getattr(params, name) for name in _field_names(params))


def _field_names(value: object) -> tuple[str, ...]:
    fields = getattr(value, "__dataclass_fields__", None)
    if fields is not None:
        return tuple(fields)
    return tuple(vars(value))


def _training_params_signature(params: object) -> tuple[object, ...]:
    adam = tuple(getattr(params.adam, name) for name in _field_names(params.adam))
    stability = tuple(getattr(params.stability, name) for name in _field_names(params.stability))
    training = tuple(getattr(params.training, name) for name in _field_names(params.training))
    return adam + stability + training


def _training_live_params_signature(params: object) -> tuple[object, ...]:
    adam = tuple(getattr(params.adam, name) for name in _field_names(params.adam))
    stability = tuple(getattr(params.stability, name) for name in _field_names(params.stability))
    training = tuple(
        getattr(params.training, name)
        for name in _field_names(params.training)
        if name not in _TRAINING_RUNTIME_PARAM_NAMES
    )
    return adam + stability + training


def _training_runtime_signature(params: object) -> tuple[object, ...]:
    return tuple(getattr(params.training, name, None) for name in _TRAINING_RUNTIME_PARAM_NAMES)


def recreate_renderer(viewer: object, width: int, height: int) -> None:
    ensure_renderer(viewer, "renderer", width, height, allow_debug_overlays=True)
    _reset_loss_debug(viewer)


def ensure_training_runtime_resolution(viewer: object) -> bool:
    if viewer.s.trainer is None or viewer.s.training_renderer is None or not viewer.s.training_frames:
        return False
    if bool(getattr(viewer.s, "pending_training_runtime_resize", False)):
        _, params, _, _ = resolve_effective_training_setup(viewer)
        runtime_signature = _training_runtime_signature(params)
        if getattr(viewer.s, "applied_training_runtime_signature", None) != runtime_signature:
            viewer.s.trainer.update_hyperparams(params.adam, params.stability, params.training)
            viewer.s.applied_training_signature = _training_live_params_signature(params)
            viewer.s.applied_training_runtime_signature = runtime_signature
    current_factor = int(viewer.s.trainer.effective_train_render_factor()) if hasattr(viewer.s.trainer, "effective_train_render_factor") else int(viewer.s.trainer.effective_train_downscale_factor())
    current_size = (int(viewer.s.training_renderer.width), int(viewer.s.training_renderer.height))
    if viewer.s.applied_training_runtime_factor == current_factor:
        desired_width, desired_height = viewer.s.trainer.training_resolution(0)
        if current_size == (int(desired_width), int(desired_height)):
            return False
    desired_width, desired_height = viewer.s.trainer.training_resolution(0)
    desired_size = (int(desired_width), int(desired_height))
    if current_size == desired_size:
        viewer.s.applied_training_runtime_factor = current_factor
        viewer.s.pending_training_runtime_resize = False
        return True
    previous_renderer = viewer.s.training_renderer
    renderer = _create_renderer(viewer, desired_size[0], desired_size[1], allow_debug_overlays=False)
    enc = viewer.device.create_command_encoder()
    previous_renderer.copy_scene_state_to(enc, renderer)
    viewer.device.submit_command_buffer(enc.finish())
    viewer.s.training_renderer = renderer
    viewer.s.trainer.rebind_renderer(renderer)
    _apply_debug_buffers(viewer, viewer.s.renderer)
    _apply_debug_buffers(viewer, viewer.s.debug_renderer)
    viewer.s.applied_training_runtime_factor = current_factor
    viewer.s.pending_training_runtime_resize = False
    _invalidate(viewer)
    _reset_loss_debug(viewer)
    return True


def _resolve_training_setup_signature(viewer: object, init: object, params: object, images_subdir: str | None) -> tuple[object, ...]:
    return (
        int(init.seed),
        None if init.hparams.initial_opacity is None else round(float(init.hparams.initial_opacity), 8),
        *(_training_params_signature(params)),
        None if viewer.s.colmap_root is None else str(viewer.s.colmap_root.resolve()),
        images_subdir,
    )


def _refresh_training_frames(viewer: object) -> None:
    if viewer.s.colmap_recon is None:
        return
    import_cfg = getattr(viewer.s, "colmap_import", None)
    images_root = None if import_cfg is None else getattr(import_cfg, "images_root", None)
    if images_root is None:
        return
    viewer.s.training_frames = build_training_frames_from_root(
        viewer.s.colmap_recon,
        Path(images_root).resolve(),
        downscale_mode=str(getattr(import_cfg, "image_downscale_mode", _COLMAP_IMAGE_DOWNSCALE_ORIGINAL)),
        downscale_max_size=int(getattr(import_cfg, "image_downscale_max_size", 2048)),
        downscale_scale=float(getattr(import_cfg, "image_downscale_scale", 1.0)),
    )


def _build_initial_training_scene(viewer: object, init: object, params: object, init_hparams: object) -> tuple[GaussianScene, float | None]:
    if viewer.s.colmap_recon is None:
        raise RuntimeError("Training scene initialization requires a loaded COLMAP reconstruction.")
    import_cfg = viewer.s.colmap_import
    _ensure_cached_init_source(viewer, init)
    if import_cfg.init_mode == _COLMAP_IMPORT_CUSTOM_PLY:
        if viewer.s.cached_init_scene is None:
            raise RuntimeError("Cached custom PLY scene is unavailable.")
        return _copy_gaussian_scene(viewer.s.cached_init_scene), None

    positions = getattr(viewer.s, "cached_init_point_positions", None)
    colors = getattr(viewer.s, "cached_init_point_colors", None)
    if positions is None or colors is None:
        raise RuntimeError("Cached pointcloud initializer data is unavailable.")

    if import_cfg.init_mode == _COLMAP_IMPORT_DIFFUSED_POINTCLOUD:
        resolved_init = _diffused_pointcloud_init_hparams_from_positions(viewer.s.colmap_recon, positions, init_hparams, import_cfg.nn_radius_scale_coef)
        return initialize_scene_from_points_colors(positions, colors, init.seed, resolved_init), float(max(resolved_init.base_scale, 1e-8))

    resolved_init = _pointcloud_init_hparams_from_positions(viewer.s.colmap_recon, positions, params.training.max_gaussians, init_hparams, import_cfg.nn_radius_scale_coef)
    chosen_count = positions.shape[0] if params.training.max_gaussians <= 0 else min(max(int(params.training.max_gaussians), 1), positions.shape[0])
    scene = initialize_scene_from_points_colors(positions[:chosen_count], colors[:chosen_count], init.seed, resolved_init)
    return scene, float(max(resolved_init.base_scale, 1e-8))


def resolve_effective_training_setup(viewer: object):
    init = viewer.init_params()
    raw_params = viewer.training_params()
    images_subdir = _profile_images_subdir(viewer)
    signature = _resolve_training_setup_signature(viewer, init, raw_params, images_subdir)
    if viewer.s.cached_training_setup_signature == signature and viewer.s.cached_training_setup is not None:
        return viewer.s.cached_training_setup
    params, profile = apply_training_profile(raw_params, "auto", dataset_root=viewer.s.colmap_root, images_subdir=images_subdir)
    init_hparams = init.hparams if profile.init_opacity_override is None else replace(init.hparams, initial_opacity=profile.init_opacity_override)
    viewer.s.cached_training_setup_signature = signature
    viewer.s.cached_training_setup = (init, params, init_hparams, profile)
    return viewer.s.cached_training_setup


def apply_live_params(viewer: object, force_init_defaults: bool = False) -> None:
    del force_init_defaults
    use_sh = bool(viewer.training_params().training.use_sh)
    viewer.s.background = viewer.render_background()
    renderer_specs = (
        ("renderer", True, "applied_renderer_params_main", True),
        ("training_renderer", False, "applied_renderer_params_training", False),
        ("debug_renderer", True, "applied_renderer_params_debug", True),
    )
    for attr, allow_debug, state_attr, force_sh in renderer_specs:
        renderer = getattr(viewer.s, attr)
        if renderer is None:
            setattr(viewer.s, state_attr, None)
            continue
        renderer.use_sh = True if force_sh else use_sh
        params = viewer.renderer_params(allow_debug)
        signature = _renderer_params_signature(params)
        if getattr(viewer.s, state_attr) == signature:
            continue
        for key, value in renderer_kwargs(params).items():
            setattr(renderer, key, value)
        setattr(viewer.s, state_attr, signature)
    _apply_debug_buffers(viewer, viewer.s.renderer)
    _apply_debug_buffers(viewer, viewer.s.debug_renderer)
    if viewer.s.trainer is not None:
        viewer.s.trainer.compute_debug_grad_norm = bool(viewer.s.renderer is not None and viewer.s.renderer.debug_show_grad_norm)
        _, params, _, _ = resolve_effective_training_setup(viewer)
        signature = _training_live_params_signature(params)
        runtime_signature = _training_runtime_signature(params)
        if viewer.s.applied_training_signature != signature:
            viewer.s.trainer.update_hyperparams(params.adam, params.stability, params.training)
            viewer.s.applied_training_signature = signature
            viewer.s.applied_training_runtime_signature = runtime_signature
            viewer.s.pending_training_runtime_resize = True
        elif getattr(viewer.s, "applied_training_runtime_signature", None) != runtime_signature:
            viewer.s.pending_training_runtime_resize = True


def sync_scene_from_training_renderer(viewer: object, dst_renderer: GaussianRenderer, target: str, force: bool = False) -> None:
    if viewer.s.training_renderer is None or viewer.s.trainer is None:
        return
    step = int(viewer.s.trainer.state.step)
    if not force and getattr(viewer.s, f"synced_step_{target}") == step:
        return
    enc = viewer.device.create_command_encoder()
    viewer.s.training_renderer.copy_scene_state_to(enc, dst_renderer)
    viewer.device.submit_command_buffer(enc.finish())
    setattr(viewer.s, f"synced_step_{target}", step)


def refresh_cached_raster_grad_histograms(viewer: object, force: bool = False) -> None:
    refresh_requested = bool(force or viewer.ui._values.get("_histograms_refresh_requested", False))
    if viewer.s.trainer is None or viewer.s.training_renderer is None:
        viewer.s.cached_raster_grad_histograms = None
        viewer.s.cached_raster_grad_ranges = None
        viewer.s.cached_raster_grad_histogram_status = "Histograms require an initialized training scene."
        viewer.ui._values["_histograms_refresh_requested"] = False
        return
    bin_count = max(int(viewer.ui._values.get("hist_bin_count", 64)), 1)
    min_log10 = float(viewer.ui._values.get("hist_min_log10", -8.0))
    max_log10 = float(viewer.ui._values.get("hist_max_log10", 2.0))
    step = int(viewer.s.trainer.state.step)
    scene_count = int(viewer.s.trainer.scene.count)
    mode = str(viewer.s.training_renderer.cached_raster_grad_atomic_mode)
    signature = (step, mode, scene_count, bin_count, min_log10, max_log10)
    if not refresh_requested:
        return
    viewer.s.cached_raster_grad_histograms = viewer.s.training_renderer.compute_cached_raster_grad_component_histograms(
        viewer.s.trainer.metrics,
        scene_count,
        bin_count=bin_count,
        min_log10=min_log10,
        max_log10=max_log10,
    )
    grad_ranges = viewer.s.training_renderer.compute_cached_raster_grad_component_ranges(viewer.s.trainer.metrics, scene_count)
    sh_ranges_fn = getattr(viewer.s.training_renderer, "compute_sh_component_ranges", None)
    sh_ranges = sh_ranges_fn(scene_count) if callable(sh_ranges_fn) else None
    viewer.s.cached_raster_grad_ranges = _concat_param_tensor_ranges(grad_ranges, sh_ranges)
    viewer.s.cached_raster_grad_histogram_mode = mode
    viewer.s.cached_raster_grad_histogram_step = step
    viewer.s.cached_raster_grad_histogram_scene_count = scene_count
    viewer.s.cached_raster_grad_histogram_signature = signature
    total = int(np.sum(viewer.s.cached_raster_grad_histograms.counts))
    viewer.s.cached_raster_grad_histogram_status = (
        f"Cached ellipse grads + SH ranges | mode={mode} | step={step:,} | samples={scene_count:,} | populated={total:,}"
        if total > 0 or step > 0
        else "No cached ellipse backward gradients have been produced yet."
    )
    viewer.ui._values["_histograms_refresh_requested"] = False


def _concat_param_tensor_ranges(*payloads: object) -> object:
    valid = [payload for payload in payloads if payload is not None]
    if len(valid) == 0:
        return None
    if len(valid) == 1:
        return valid[0]
    min_values = []
    max_values = []
    labels: list[str] = []
    for payload in valid:
        payload_min = np.asarray(getattr(payload, "min_values", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        payload_max = np.asarray(getattr(payload, "max_values", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        if payload_min.size != payload_max.size:
            continue
        min_values.append(payload_min)
        max_values.append(payload_max)
        labels.extend(str(label) for label in getattr(payload, "param_labels", ()))
    if len(min_values) == 0:
        return None
    return ParamTensorRanges(
        min_values=np.concatenate(min_values, axis=0),
        max_values=np.concatenate(max_values, axis=0),
        param_labels=tuple(labels),
    )


def load_scene(viewer: object, path: Path) -> None:
    scene = load_gaussian_ply(path)
    _reset_loaded_runtime(viewer)
    viewer.s.scene = scene
    viewer.s.scene_path = path
    viewer.s.colmap_root = None
    viewer.s.colmap_recon = None
    viewer.s.training_frames = []
    viewer.s.renderer.set_scene(scene)
    viewer.apply_camera_fit(estimate_scene_bounds(scene))
    viewer.s.last_error = ""
    print(f"Loaded scene: {path} ({scene.count:,} splats)")


def _finish_import_colmap_dataset(
    viewer: object,
    *,
    colmap_root: Path,
    database_path: Path | None,
    images_root: Path,
    depth_root: Path | None = None,
    init_mode: str,
    custom_ply_path: Path | None,
    image_downscale_mode: str,
    image_downscale_max_size: int,
    image_downscale_scale: float,
    nn_radius_scale_coef: float,
    depth_point_count: int = 100000,
    diffused_point_count: int = 100000,
    diffusion_radius: float = 1.0,
    recon: object = None,
    training_frames: list[ColmapFrame] | None = None,
    frame_targets_native: list[spy.Texture] | None = None,
    cached_init_point_positions: np.ndarray | None = None,
    cached_init_point_colors: np.ndarray | None = None,
) -> None:
    if recon is None or training_frames is None:
        raise RuntimeError("COLMAP import finalize requires reconstruction and training frames.")
    xyz, _ = _point_tables(recon)
    _reset_loaded_runtime(viewer)
    _reset_training_visual_state(viewer)
    _update_import_settings(
        viewer,
        dataset_root=colmap_root,
        database_path=database_path,
        images_root=images_root,
        depth_root=depth_root,
        init_mode=init_mode,
        custom_ply_path=custom_ply_path,
        image_downscale_mode=image_downscale_mode,
        image_downscale_max_size=image_downscale_max_size,
        image_downscale_scale=image_downscale_scale,
        nn_radius_scale_coef=nn_radius_scale_coef,
        depth_point_count=depth_point_count,
        diffused_point_count=diffused_point_count,
        diffusion_radius=diffusion_radius,
    )
    viewer.s.colmap_root = Path(colmap_root)
    viewer.s.colmap_recon = recon
    viewer.s.training_frames = list(training_frames)
    viewer.s.colmap_point_count = int(xyz.shape[0])
    viewer.s.scene_path = None
    if cached_init_point_positions is not None and cached_init_point_colors is not None:
        viewer.s.cached_init_point_positions = np.array(cached_init_point_positions, dtype=np.float32, copy=True)
        viewer.s.cached_init_point_colors = np.array(cached_init_point_colors, dtype=np.float32, copy=True)
        init_params_fn = getattr(viewer, "init_params", None)
        if callable(init_params_fn):
            viewer.s.cached_init_signature = _cached_init_signature(viewer, init_params_fn())
    apply_live_params(viewer)
    _apply_initial_camera_fit(viewer, fallback_factory=lambda: estimate_point_bounds(xyz))
    initialize_training_scene(viewer, frame_targets_native=frame_targets_native)
    viewer.s.last_error = ""
    print(
        f"Loaded COLMAP: db={None if database_path is None else Path(database_path).resolve()} root={colmap_root} "
        f"frames={len(training_frames)} images={Path(images_root).resolve()} init={init_mode}"
    )


def import_colmap_dataset(
    viewer: object,
    *,
    colmap_root: Path,
    database_path: Path | None,
    images_root: Path,
    depth_root: Path | None = None,
    init_mode: str,
    custom_ply_path: Path | None,
    image_downscale_mode: str,
    image_downscale_max_size: int,
    image_downscale_scale: float,
    nn_radius_scale_coef: float,
    depth_point_count: int = 100000,
    diffused_point_count: int = 100000,
    diffusion_radius: float = 1.0,
) -> None:
    _clear_loaded_scene(viewer)
    root = Path(colmap_root).resolve()
    recon = _load_aligned_colmap_reconstruction(root)
    training_frames = build_training_frames_from_root(
        recon,
        images_root,
        downscale_mode=image_downscale_mode,
        downscale_max_size=image_downscale_max_size,
        downscale_scale=image_downscale_scale,
    )
    frame_targets_native = _create_native_dataset_textures(viewer, training_frames)
    cached_init_point_positions = None
    cached_init_point_colors = None
    if init_mode == _COLMAP_IMPORT_DEPTH:
        if depth_root is None:
            raise RuntimeError("Depth initialization requires a selected depth folder.")
        cached_init_point_positions, cached_init_point_colors = _build_depth_init_source(
            recon,
            Path(images_root).resolve(),
            training_frames,
            depth_root=Path(depth_root).resolve(),
            depth_point_count=depth_point_count,
            seed=int(viewer.init_params().seed),
        )
    _finish_import_colmap_dataset(
        viewer,
        colmap_root=root,
        database_path=database_path,
        images_root=images_root,
        depth_root=depth_root,
        init_mode=init_mode,
        custom_ply_path=custom_ply_path,
        image_downscale_mode=image_downscale_mode,
        image_downscale_max_size=image_downscale_max_size,
        image_downscale_scale=image_downscale_scale,
        nn_radius_scale_coef=nn_radius_scale_coef,
        depth_point_count=depth_point_count,
        diffused_point_count=diffused_point_count,
        diffusion_radius=diffusion_radius,
        recon=recon,
        training_frames=training_frames,
        frame_targets_native=frame_targets_native,
        cached_init_point_positions=cached_init_point_positions,
        cached_init_point_colors=cached_init_point_colors,
    )


def import_colmap_from_ui(viewer: object) -> None:
    colmap_root = Path(_ui_path_string(viewer, "colmap_root_path")).expanduser()
    database_path_text = _ui_path_string(viewer, "colmap_database_path")
    database_path = None if not database_path_text else Path(database_path_text).expanduser()
    images_root = Path(_ui_path_string(viewer, "colmap_images_root")).expanduser()
    depth_root_text = _ui_path_string(viewer, "colmap_depth_root")
    depth_root = None if not depth_root_text else Path(depth_root_text).expanduser()
    init_mode = _ui_import_mode(viewer)
    custom_ply_text = _ui_path_string(viewer, "colmap_custom_ply_path")
    custom_ply_path = None if not custom_ply_text else Path(custom_ply_text).expanduser()
    image_downscale_mode = _ui_image_downscale_mode(viewer)
    image_downscale_max_size = max(int(viewer.ui._values.get("colmap_image_max_size", 2048)), 1)
    image_downscale_scale = float(np.clip(viewer.ui._values.get("colmap_image_scale", 1.0), 1e-6, 1.0))
    nn_radius_scale_coef = float(viewer.ui._values.get("colmap_nn_radius_scale_coef", 0.5))
    depth_point_count = max(int(viewer.ui._values.get("colmap_depth_point_count", 100000)), 1)
    diffused_point_count = max(int(viewer.ui._values.get("colmap_diffused_point_count", 100000)), 1)
    diffusion_radius = max(float(viewer.ui._values.get("colmap_diffusion_radius", 1.0)), 0.0)
    if not colmap_root.exists():
        raise FileNotFoundError(f"COLMAP root does not exist: {colmap_root}")
    if not _has_colmap_sparse(colmap_root):
        colmap_root = _resolve_colmap_root_from_selection(colmap_root)
    if database_path is not None and not database_path.exists():
        database_path = _find_optional_colmap_database(colmap_root)
    if not images_root.exists():
        raise FileNotFoundError(f"COLMAP image folder does not exist: {images_root}")
    if init_mode == _COLMAP_IMPORT_DEPTH and (depth_root is None or not depth_root.exists()):
        raise FileNotFoundError(f"Depth folder does not exist: {depth_root}")
    if init_mode == _COLMAP_IMPORT_CUSTOM_PLY and (custom_ply_path is None or not custom_ply_path.exists()):
        raise FileNotFoundError(f"Custom PLY does not exist: {custom_ply_path}")
    _clear_loaded_scene(viewer)
    viewer.s.colmap_import_progress = ColmapImportProgress(
        dataset_root=Path(_ui_path_string(viewer, "colmap_root_path")).expanduser().resolve(),
        colmap_root=colmap_root.resolve(),
        database_path=None if database_path is None else database_path.resolve(),
        images_root=images_root.resolve(),
        depth_root=None if depth_root is None else depth_root.resolve(),
        init_mode=init_mode,
        custom_ply_path=None if custom_ply_path is None else custom_ply_path.resolve(),
        image_downscale_mode=image_downscale_mode,
        image_downscale_max_size=image_downscale_max_size,
        image_downscale_scale=image_downscale_scale,
        nn_radius_scale_coef=float(max(nn_radius_scale_coef, 1e-4)),
        depth_point_count=depth_point_count,
        diffused_point_count=diffused_point_count,
        diffusion_radius=diffusion_radius,
    )
    viewer.s.last_error = ""


def advance_colmap_import(viewer: object) -> None:
    progress = getattr(viewer.s, "colmap_import_progress", None)
    if progress is None:
        return
    try:
        if progress.phase == "prepare":
            progress.recon = _load_aligned_colmap_reconstruction(progress.colmap_root)
            progress.image_items = sorted(progress.recon.images.items())
            progress.depth_index = build_depth_path_index(progress.depth_root) if progress.init_mode == _COLMAP_IMPORT_DEPTH and progress.depth_root is not None else None
            progress.total = max(len(progress.image_items), 1)
            progress.current = 0
            progress.current_name = ""
            progress.phase = "scan_frames"
            return
        if progress.phase == "scan_frames":
            for _ in range(_COLMAP_IMPORT_IMAGES_PER_TICK):
                if progress.current >= len(progress.image_items):
                    break
                image_id, image = progress.image_items[progress.current]
                progress.current_name = Path(str(image.name)).name
                _append_training_frame(progress, image_id, image)
                progress.current += 1
            if progress.current < len(progress.image_items):
                return
            if not progress.frames:
                raise RuntimeError(f"No training frames were found in {progress.images_root}.")
            _start_colmap_texture_loader(progress)
            progress.phase = "load_textures"
            progress.total = len(progress.frames)
            progress.current = 0
            progress.current_name = ""
            return
        if progress.phase == "load_textures":
            for _ in range(_COLMAP_IMPORT_IMAGES_PER_TICK):
                if progress.current >= len(progress.frames):
                    break
                frame = progress.frames[progress.current]
                progress.current_name = Path(frame.image_path).name
                if progress.native_rgba8_iter is None:
                    _start_colmap_texture_loader(progress)
                load_result = next(progress.native_rgba8_iter)
                rgba8, payload = load_result if progress.init_mode == _COLMAP_IMPORT_DEPTH else (load_result, None)
                progress.native_textures.append(_create_native_dataset_texture_from_rgba8(viewer, rgba8))
                if payload is not None:
                    progress.depth_init_payloads.append(payload)
                progress.current += 1
            if progress.current < len(progress.frames):
                return
            _close_colmap_texture_loader(progress)
            progress.phase = "finalize"
            progress.current_name = ""
            return
        if progress.phase == "finalize":
            if progress.recon is None:
                raise RuntimeError("COLMAP import lost reconstruction state before finalize.")
            cached_init_point_positions = None
            cached_init_point_colors = None
            if progress.init_mode == _COLMAP_IMPORT_DEPTH:
                cached_init_point_positions, cached_init_point_colors = generate_depth_init_points(
                    progress.depth_init_payloads,
                    progress.depth_point_count,
                    int(viewer.init_params().seed),
                )
            _finish_import_colmap_dataset(
                viewer,
                colmap_root=progress.colmap_root,
                database_path=progress.database_path,
                images_root=progress.images_root,
                depth_root=progress.depth_root,
                init_mode=progress.init_mode,
                custom_ply_path=progress.custom_ply_path,
                image_downscale_mode=progress.image_downscale_mode,
                image_downscale_max_size=progress.image_downscale_max_size,
                image_downscale_scale=progress.image_downscale_scale,
                nn_radius_scale_coef=progress.nn_radius_scale_coef,
                depth_point_count=progress.depth_point_count,
                diffused_point_count=progress.diffused_point_count,
                diffusion_radius=progress.diffusion_radius,
                recon=progress.recon,
                training_frames=progress.frames,
                frame_targets_native=progress.native_textures,
                cached_init_point_positions=cached_init_point_positions,
                cached_init_point_colors=cached_init_point_colors,
            )
            viewer.toolkit.close_colmap_import_window()
            return
        raise RuntimeError(f"Unknown COLMAP import phase: {progress.phase}")
    except Exception:
        if progress is not None:
            _close_colmap_texture_loader(progress)
        viewer.s.colmap_import_progress = None
        raise


def initialize_training_scene(viewer: object, frame_targets_native: list[spy.Texture] | None = None) -> None:
    if viewer.s.colmap_recon is None and viewer.s.colmap_root is None:
        return
    _reset_training_runtime(viewer)
    if not viewer.s.training_frames:
        _refresh_training_frames(viewer)
    if viewer.s.colmap_recon is None or not viewer.s.training_frames:
        return
    init, params, init_hparams, profile = resolve_effective_training_setup(viewer)
    try:
        factor = resolve_effective_train_render_factor(params.training, 0, int(viewer.s.training_frames[0].width), int(viewer.s.training_frames[0].height))
    except TypeError:
        factor = resolve_effective_train_render_factor(params.training, 0)
    width, height = resolve_training_resolution(
        int(viewer.s.training_frames[0].width),
        int(viewer.s.training_frames[0].height),
        int(factor),
    )
    renderer = ensure_renderer(viewer, "training_renderer", width, height, allow_debug_overlays=False)
    scene, scale_reg_reference = _build_initial_training_scene(viewer, init, params, init_hparams)
    apply_live_params(viewer)
    trainer_kwargs = dict(
        device=viewer.device,
        renderer=renderer,
        scene=scene,
        frames=viewer.s.training_frames,
        adam_hparams=params.adam,
        stability_hparams=params.stability,
        training_hparams=params.training,
        seed=init.seed,
    )
    if scale_reg_reference is not None:
        trainer_kwargs["scale_reg_reference"] = scale_reg_reference
    if frame_targets_native is not None:
        trainer_kwargs["frame_targets_native"] = frame_targets_native
    viewer.s.trainer = GaussianTrainer(**trainer_kwargs)
    viewer.s.scene = SceneCountProxy(scene.count)
    _apply_initial_camera_fit(viewer, fallback_factory=lambda: estimate_scene_bounds(scene))
    enc = viewer.device.create_command_encoder()
    renderer.copy_scene_state_to(enc, viewer.s.renderer)
    viewer.device.submit_command_buffer(enc.finish())
    _apply_debug_buffers(viewer, viewer.s.renderer)
    _apply_debug_buffers(viewer, viewer.s.debug_renderer)
    viewer.s.training_active = False
    viewer.s.training_elapsed_s = 0.0
    viewer.s.training_resume_time = None
    viewer.s.applied_renderer_params_training = _renderer_params_signature(viewer.renderer_params(False))
    viewer.s.applied_training_signature = _training_live_params_signature(params)
    viewer.s.applied_training_runtime_signature = _training_runtime_signature(params)
    viewer.s.applied_training_runtime_factor = int(viewer.s.trainer.effective_train_render_factor(0)) if hasattr(viewer.s.trainer, "effective_train_render_factor") else int(viewer.s.trainer.effective_train_downscale_factor(0))
    viewer.s.pending_training_runtime_resize = False
    _invalidate(viewer)
    viewer.s.scene_init_signature = _scene_signature(viewer)
    _reset_training_visual_state(viewer)
    update_debug_frame_slider_range(viewer)
    _reset_loss_debug(viewer)
    viewer.s.last_error = ""
    print(f"Initialized training scene ({scene.count:,} gaussians, profile={profile.name})")


def training_elapsed_seconds(viewer: object, now: float | None = None) -> float:
    elapsed = float(getattr(viewer.s, "training_elapsed_s", 0.0))
    resume_time = getattr(viewer.s, "training_resume_time", None)
    if viewer.s.training_active and resume_time is not None:
        current_time = float(time.perf_counter() if now is None else now)
        elapsed += max(current_time - float(resume_time), 0.0)
    return elapsed


def set_training_active(viewer: object, active: bool) -> None:
    if active and viewer.s.trainer is None:
        initialize_training_scene(viewer)
    now = float(time.perf_counter())
    if viewer.s.training_active and viewer.s.training_resume_time is not None:
        viewer.s.training_elapsed_s += max(now - float(viewer.s.training_resume_time), 0.0)
        viewer.s.training_resume_time = None
    viewer.s.training_active = bool(active and viewer.s.trainer is not None)
    if viewer.s.training_active:
        viewer.s.training_resume_time = now
