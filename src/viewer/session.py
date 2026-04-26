from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import subprocess
import time
import urllib.request

import numpy as np
from PIL import Image
import slangpy as spy

from ..app.shared import apply_training_profile, estimate_point_bounds, estimate_scene_bounds
from ..utility import SHADER_ROOT, clamp_index, load_compute_kernels
from ..metrics import ParamTensorRanges
from ..renderer import GaussianRenderSettings, GaussianRenderer
from ..scene import (
    GaussianScene,
    build_training_frames_from_root,
    initialize_scene_from_points_colors,
    load_colmap_reconstruction,
    load_gaussian_ply,
    resolve_colmap_init_hparams,
    sample_colmap_diffused_points,
    sample_colmap_fibonacci_sphere_points,
)
from ..scene._internal.colmap_ops import point_nn_scales, resolve_training_frame_image_size
from ..training import resolve_sh_band
from ..scene._internal.colmap_ops import (
    DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH,
    DEPTH_INIT_VALUE_DISTANCE,
    DEPTH_INIT_VALUE_Z_DEPTH,
    TRAINING_FRAME_LOAD_THREADS,
    build_depth_path_index,
    generate_depth_init_points,
    load_training_frame_rgba8,
    load_training_frame_rgba8_with_depth_payload,
    match_depth_path,
)
from ..training import GaussianTrainer, resolve_effective_train_render_factor, resolve_training_resolution
from ..scene._internal.colmap_types import ColmapFrame, point_tables
from .session_colmap_utils import (
    _COLMAP_CAMERA_MODEL_NAMES,
    _COLMAP_DB_SAMPLE_LIMIT,
    _COLMAP_DEPTH_VALUE_DISTANCE,
    _COLMAP_DEPTH_VALUE_Z_DEPTH,
    _COLMAP_IMAGE_DOWNSCALE_MAX_SIZE,
    _COLMAP_IMAGE_DOWNSCALE_ORIGINAL,
    _COLMAP_IMAGE_DOWNSCALE_SCALE,
    _COLMAP_IMPORT_CUSTOM_PLY,
    _COLMAP_IMPORT_DEPTH,
    _COLMAP_IMPORT_DIFFUSED_POINTCLOUD,
    _COLMAP_IMPORT_IMAGES_PER_TICK,
    _COLMAP_IMPORT_POINTCLOUD,
    _camera_rows as _camera_rows_impl,
    _database_image_names,
    _find_colmap_database,
    _find_optional_colmap_database,
    _has_colmap_sparse as _has_colmap_sparse_impl,
    _load_aligned_colmap_reconstruction as _load_aligned_colmap_reconstruction_impl,
    _normalized_selected_camera_ids as _normalized_selected_camera_ids_impl,
    _resolve_colmap_root_from_selection as _resolve_colmap_root_from_selection_impl,
    _set_colmap_camera_preview as _set_colmap_camera_preview_impl,
    _set_ui_path as _set_ui_path_impl,
    _suggest_images_root_from_dataset_root,
    _update_import_settings as _update_import_settings_impl,
)
from .session_dataset_utils import (
    _CompressedDatasetTexture,
    _bc_payload_byte_count,
    _create_native_dataset_texture_from_bc_payload,
    _create_native_dataset_texture_from_rgba8,
    _dataset_bc7_cache_dir,
    _dataset_bc7_cache_path,
    _parse_compressed_dataset_texture,
)
from .state import ColmapImportProgress, ColmapImportSettings, SceneCountProxy

_FIBONACCI_SPHERE_EQUAL_AREA_RADIUS_FACTOR = 2.0
_FIBONACCI_SPHERE_SCALE_MIN = 1e-4
_FIBONACCI_SPHERE_SCALE_MAX = 1e4
_DATASET_BC7_TEXCONV_URL = "https://github.com/microsoft/DirectXTex/releases/download/oct2024/texconv.exe"
_DATASET_BC7_TEXCONV_PATH = Path(__file__).resolve().parents[2] / "temp" / "bc_texture_tools" / "texconv.exe"
_TRAINING_RUNTIME_PARAM_NAMES = (
    "train_downscale_mode",
    "train_auto_start_downscale",
    "train_downscale_base_iters",
    "train_downscale_iter_step",
    "train_downscale_max_iters",
    "train_downscale_factor",
    "train_subsample_factor",
)


_has_colmap_sparse = _has_colmap_sparse_impl
_resolve_colmap_root_from_selection = _resolve_colmap_root_from_selection_impl
_load_aligned_colmap_reconstruction = _load_aligned_colmap_reconstruction_impl
_set_ui_path = _set_ui_path_impl


def _camera_rows(recon: object) -> tuple[dict[str, object], ...]:
    unsupported_model_ids = sorted(
        {
            int(getattr(camera, "model_id", -1))
            for camera in getattr(recon, "cameras", {}).values()
            if int(getattr(camera, "model_id", -1)) not in _COLMAP_CAMERA_MODEL_NAMES
        }
    )
    if unsupported_model_ids:
        raise ValueError(f"Unsupported COLMAP camera model id {unsupported_model_ids[0]}")
    return _camera_rows_impl(recon)


_normalized_selected_camera_ids = _normalized_selected_camera_ids_impl


def _set_colmap_camera_preview(
    viewer: object, recon: object, selected_camera_ids: tuple[int, ...] | None = None
) -> tuple[int, ...]:
    rows = _camera_rows(recon)
    selected_ids = _normalized_selected_camera_ids(rows, selected_camera_ids)
    viewer.ui._values["_colmap_camera_rows"] = rows
    viewer.ui._values["colmap_selected_camera_ids"] = selected_ids
    return selected_ids


_update_import_settings = _update_import_settings_impl


def _ensure_dataset_bc7_texconv() -> Path:
    path = _DATASET_BC7_TEXCONV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    urllib.request.urlretrieve(_DATASET_BC7_TEXCONV_URL, path)
    return path


def _compress_dataset_frame_to_bc7_cache(frame: ColmapFrame, images_root: Path) -> Path:
    texconv_path = _ensure_dataset_bc7_texconv()
    cache_path = _dataset_bc7_cache_path(images_root, frame)
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    source_path = frame.image_path
    temp_source_path: Path | None = None
    target_size = (max(int(frame.width), 1), max(int(frame.height), 1))
    with Image.open(frame.image_path) as pil_image:
        if pil_image.size != target_size:
            temp_source_path = cache_path.with_suffix(".png")
            resized = pil_image.convert("RGBA")
            if resized.size != target_size:
                resized = resized.resize(target_size, Image.Resampling.LANCZOS)
            resized.save(temp_source_path)
            source_path = temp_source_path
    try:
        subprocess.run(
            [
                str(texconv_path),
                "-y",
                "-srgbi",
                "-srgbo",
                "-f",
                "BC7_UNORM_SRGB",
                "-m",
                "1",
                "-o",
                str(cache_path.parent),
                str(source_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"texconv failed for {frame.image_path.name}: {exc.stderr.strip() or exc.stdout.strip()}") from exc
    finally:
        if temp_source_path is not None:
            temp_source_path.unlink(missing_ok=True)
    generated_candidates = (cache_path, cache_path.with_suffix(".DDS"))
    generated_path = next((path for path in generated_candidates if path.exists()), None)
    if generated_path is not None and generated_path != cache_path:
        generated_path.replace(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Expected BC7 cache file was not created: {cache_path}")
    return cache_path


def _load_or_create_bc7_dataset_texture(frame: ColmapFrame, images_root: Path) -> _CompressedDatasetTexture:
    cache_path = _dataset_bc7_cache_path(images_root, frame)
    if not cache_path.exists():
        cache_path = _compress_dataset_frame_to_bc7_cache(frame, images_root)
    payload = _parse_compressed_dataset_texture(cache_path)
    if payload.format != spy.Format.bc7_unorm_srgb or payload.width != int(frame.width) or payload.height != int(frame.height):
        cache_path.unlink(missing_ok=True)
        cache_path = _compress_dataset_frame_to_bc7_cache(frame, images_root)
        payload = _parse_compressed_dataset_texture(cache_path)
    if payload.format != spy.Format.bc7_unorm_srgb:
        raise RuntimeError(f"Expected BC7 UNORM SRGB cache for {frame.image_path.name}, got {payload.format}")
    if payload.width != int(frame.width) or payload.height != int(frame.height):
        raise RuntimeError(
            f"BC7 cache size mismatch for {frame.image_path.name}: expected {int(frame.width)}x{int(frame.height)}, "
            f"got {int(payload.width)}x{int(payload.height)}"
        )
    return payload


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


def _point_tables(recon: object, min_track_length: int = DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    threshold = max(int(min_track_length), 0)
    xyz, rgb = point_tables(recon, min_track_length=threshold)
    if xyz.shape[0] != rgb.shape[0] or xyz.shape[0] <= 0:
        if threshold <= 0:
            raise RuntimeError("COLMAP point tables are empty or mismatched.")
        raise RuntimeError(f"COLMAP point tables are empty or mismatched after filtering to points seen by at least {threshold} cameras.")
    return xyz, rgb


def _ui_path_string(viewer: object, key: str) -> str:
    return str(viewer.ui._values.get(key, "")).strip()


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


def _ui_depth_value_mode(viewer: object) -> str:
    return _COLMAP_DEPTH_VALUE_DISTANCE if int(viewer.ui._values.get("colmap_depth_value_mode", 1)) == 0 else _COLMAP_DEPTH_VALUE_Z_DEPTH


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


def _append_training_frame(progress: ColmapImportProgress, image_id: int, image: object) -> None:
    if progress.recon is None:
        raise RuntimeError("COLMAP import progress is missing reconstruction state.")
    selected = {int(camera_id) for camera_id in getattr(progress, "selected_camera_ids", ())}
    if selected and int(image.camera_id) not in selected:
        return
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


def _load_dataset_texture(frame: ColmapFrame, images_root: Path, compress_dataset_using_bc7: bool) -> np.ndarray | _CompressedDatasetTexture:
    return _load_or_create_bc7_dataset_texture(frame, images_root) if bool(compress_dataset_using_bc7) else load_training_frame_rgba8(frame)


def _load_dataset_texture_with_depth_payload(task: tuple[ColmapReconstruction, object, object, ColmapFrame, Path | None, str, bool, Path]) -> tuple[np.ndarray | _CompressedDatasetTexture, object | None]:
    recon, image, camera, frame, depth_path, depth_value_mode, compress_dataset_using_bc7, images_root = task
    rgba8, payload = load_training_frame_rgba8_with_depth_payload((recon, image, camera, frame, depth_path, depth_value_mode))
    if not bool(compress_dataset_using_bc7):
        return rgba8, payload
    cache_path = _dataset_bc7_cache_path(images_root, frame)
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        staging_path = cache_path.with_suffix(".png")
        Image.fromarray(np.ascontiguousarray(rgba8, dtype=np.uint8), mode="RGBA").save(staging_path)
        try:
            subprocess.run(
                [
                    str(_ensure_dataset_bc7_texconv()),
                    "-y",
                    "-srgbi",
                    "-srgbo",
                    "-f",
                    "BC7_UNORM_SRGB",
                    "-m",
                    "1",
                    "-o",
                    str(cache_path.parent),
                    str(staging_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"texconv failed for {frame.image_path.name}: {exc.stderr.strip() or exc.stdout.strip()}") from exc
        finally:
            if staging_path.exists():
                staging_path.unlink()
    return _parse_compressed_dataset_texture(cache_path), payload


def _close_colmap_texture_loader(progress: ColmapImportProgress) -> None:
    loader = progress.native_rgba8_loader
    progress.native_rgba8_loader = None
    progress.native_rgba8_iter = None
    if loader is not None:
        loader.shutdown(wait=False, cancel_futures=False)


def _start_colmap_texture_loader(progress: ColmapImportProgress) -> None:
    _close_colmap_texture_loader(progress)
    if bool(progress.compress_dataset_using_bc7):
        _ensure_dataset_bc7_texconv()
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
            tasks.append((progress.recon, image, camera, frame, depth_path, progress.depth_value_mode, progress.compress_dataset_using_bc7, progress.images_root))
        progress.native_rgba8_iter = loader.map(_load_dataset_texture_with_depth_payload, tasks)
        return
    progress.native_rgba8_iter = loader.map(lambda frame: _load_dataset_texture(frame, progress.images_root, progress.compress_dataset_using_bc7), progress.frames)


def _create_native_dataset_textures(
    viewer: object,
    frames: list[ColmapFrame],
    images_root: Path | None = None,
    compress_dataset_using_bc7: bool | None = None,
) -> list[spy.Texture]:
    import_cfg = getattr(viewer.s, "colmap_import", None)
    resolved_images_root = images_root if images_root is not None else None if import_cfg is None else getattr(import_cfg, "images_root", None)
    if resolved_images_root is None:
        raise RuntimeError("Dataset texture creation requires a resolved images root.")
    resolved_images_root = Path(resolved_images_root).resolve()
    use_bc7 = bool(
        compress_dataset_using_bc7
        if compress_dataset_using_bc7 is not None
        else False if import_cfg is None else getattr(import_cfg, "compress_dataset_using_bc7", False)
    )
    textures: list[spy.Texture] = []
    if use_bc7:
        _ensure_dataset_bc7_texconv()
    with ThreadPoolExecutor(max_workers=TRAINING_FRAME_LOAD_THREADS, thread_name_prefix="viewer-target") as executor:
        for payload in executor.map(lambda frame: _load_dataset_texture(frame, resolved_images_root, use_bc7), frames):
            textures.append(_create_native_dataset_texture_from_bc_payload(viewer, payload) if isinstance(payload, _CompressedDatasetTexture) else _create_native_dataset_texture_from_rgba8(viewer, payload))
    return textures


def _build_depth_init_source(
    recon: object,
    images_root: Path,
    frames: list[ColmapFrame],
    *,
    depth_root: Path,
    depth_value_mode: str,
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
        tasks.append((recon, image, camera, frame, depth_path, depth_value_mode))
    if len(tasks) == 0:
        raise RuntimeError("Depth initialization found no matched RGB/depth frame pairs.")
    payloads = []
    with ThreadPoolExecutor(max_workers=TRAINING_FRAME_LOAD_THREADS, thread_name_prefix="viewer-depth") as executor:
        for _, payload in executor.map(load_training_frame_rgba8_with_depth_payload, tasks):
            if payload is not None:
                payloads.append(payload)
    return generate_depth_init_points(payloads, depth_point_count, seed, depth_value_mode)


def choose_colmap_root(viewer: object, dataset_root: Path) -> None:
    root = Path(dataset_root).resolve()
    colmap_root = _resolve_colmap_root_from_selection(root)
    recon = load_colmap_reconstruction(colmap_root)
    db_path = _find_optional_colmap_database(root)
    image_names = [str(image.name).strip() for _, image in sorted(recon.images.items()) if str(image.name).strip()] if db_path is None else _database_image_names(db_path)
    _set_ui_path(viewer, "colmap_root_path", root)
    _set_ui_path(viewer, "colmap_database_path", db_path)
    _set_ui_path(viewer, "colmap_images_root", _suggest_images_root_from_dataset_root(root, image_names))
    _set_colmap_camera_preview(viewer, recon)
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


def _pointcloud_init_hparams_from_positions(recon: object, positions: np.ndarray, max_gaussians: int, init_hparams: object, nn_radius_scale_coef: float, min_track_length: int):
    resolved = resolve_colmap_init_hparams(recon, max_gaussians, init_hparams, min_track_length=min_track_length)
    chosen_count = positions.shape[0] if max_gaussians <= 0 else min(max(int(max_gaussians), 1), positions.shape[0])
    chosen_positions = np.ascontiguousarray(positions[:chosen_count], dtype=np.float32)
    median_nn_scale = float(np.median(point_nn_scales(chosen_positions))) if chosen_count > 0 else 1.0
    return replace(resolved, base_scale=float(max(float(nn_radius_scale_coef), 1e-4) * max(median_nn_scale, 1e-6)))


def _diffused_pointcloud_init_hparams_from_positions(recon: object, positions: np.ndarray, init_hparams: object, nn_radius_scale_coef: float, min_track_length: int):
    resolved = resolve_colmap_init_hparams(recon, int(positions.shape[0]), init_hparams, min_track_length=min_track_length)
    chosen_positions = np.ascontiguousarray(positions, dtype=np.float32)
    median_nn_scale = float(np.median(point_nn_scales(chosen_positions))) if chosen_positions.shape[0] > 0 else 1.0
    return replace(resolved, base_scale=float(max(float(nn_radius_scale_coef), 1e-4) * max(median_nn_scale, 1e-6)))


def _append_fibonacci_sphere_points(
    recon: object,
    positions: np.ndarray,
    colors: np.ndarray,
    point_count: int,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = max(int(point_count), 0)
    if count <= 0:
        return positions, colors
    sphere_positions, sphere_colors = sample_colmap_fibonacci_sphere_points(recon, count, radius)
    return (
        np.ascontiguousarray(np.concatenate((np.asarray(positions, dtype=np.float32), sphere_positions), axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate((np.asarray(colors, dtype=np.float32), sphere_colors), axis=0), dtype=np.float32),
    )


def _resolve_fibonacci_sphere_count(total_count: int, point_count: int) -> int:
    return min(max(int(point_count), 0), max(int(total_count), 0))


def _fibonacci_sphere_dense_overlap_scale(point_count: int, radius: float) -> float:
    if point_count <= 0 or radius <= 0.0:
        return _FIBONACCI_SPHERE_SCALE_MIN
    scale = _FIBONACCI_SPHERE_EQUAL_AREA_RADIUS_FACTOR * float(radius) / float(np.sqrt(float(point_count)))
    return float(np.clip(scale, _FIBONACCI_SPHERE_SCALE_MIN, _FIBONACCI_SPHERE_SCALE_MAX))


def _apply_fibonacci_sphere_dense_overlap_scales(scene: GaussianScene, point_count: int, radius: float) -> GaussianScene:
    if int(point_count) <= 0 or not hasattr(scene, "scales"):
        return scene
    count = min(int(point_count), int(scene.scales.shape[0]))
    scene.scales[-count:, :] = np.float32(np.log(_fibonacci_sphere_dense_overlap_scale(count, radius)))
    return scene


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
    min_track_length = int(getattr(import_cfg, "min_track_length", DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH))
    if import_cfg is None:
        return None
    if str(import_cfg.init_mode) == _COLMAP_IMPORT_DEPTH:
        return (
            None if viewer.s.colmap_root is None else str(Path(viewer.s.colmap_root).resolve()),
            str(import_cfg.init_mode),
            None if import_cfg.images_root is None else str(Path(import_cfg.images_root).resolve()),
            None if import_cfg.depth_root is None else str(Path(import_cfg.depth_root).resolve()),
            str(import_cfg.depth_value_mode),
            str(import_cfg.image_downscale_mode),
            int(import_cfg.image_downscale_max_size),
            round(float(import_cfg.image_downscale_scale), 6),
            min_track_length,
            int(import_cfg.depth_point_count),
            int(getattr(import_cfg, "fibonacci_sphere_point_count", 0)),
            round(float(getattr(import_cfg, "fibonacci_sphere_radius", 20.0)), 6),
        )
    return (
        None if viewer.s.colmap_root is None else str(Path(viewer.s.colmap_root).resolve()),
        str(import_cfg.init_mode),
        None if import_cfg.custom_ply_path is None else str(Path(import_cfg.custom_ply_path).resolve()),
        min_track_length,
        int(import_cfg.diffused_point_count),
        round(float(import_cfg.diffusion_radius), 6),
        int(getattr(import_cfg, "fibonacci_sphere_point_count", 0)),
        round(float(getattr(import_cfg, "fibonacci_sphere_radius", 20.0)), 6),
        int(init.seed),
    )


def _ensure_cached_init_source(viewer: object, init: object) -> None:
    if viewer.s.colmap_recon is None:
        return
    import_cfg = viewer.s.colmap_import
    min_track_length = int(getattr(import_cfg, "min_track_length", DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH))
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
        positions, colors = sample_colmap_diffused_points(
            viewer.s.colmap_recon,
            import_cfg.diffused_point_count,
            import_cfg.diffusion_radius,
            init.seed,
            min_track_length=min_track_length,
        )
    else:
        positions, colors = _point_tables(viewer.s.colmap_recon, min_track_length)
    if import_cfg.init_mode != _COLMAP_IMPORT_CUSTOM_PLY:
        positions, colors = _append_fibonacci_sphere_points(
            viewer.s.colmap_recon,
            positions,
            colors,
            int(getattr(import_cfg, "fibonacci_sphere_point_count", 0)),
            float(getattr(import_cfg, "fibonacci_sphere_radius", 20.0)),
        )
        setattr(viewer.s, "cached_init_point_positions", np.array(positions, dtype=np.float32, copy=True))
        setattr(viewer.s, "cached_init_point_colors", np.array(colors, dtype=np.float32, copy=True))
    setattr(viewer.s, "cached_init_signature", signature)


def _invalidate(viewer: object, *targets: str) -> None:
    for target in targets or ("main", "debug"):
        setattr(viewer.s, f"synced_step_{target}", -1)


def _reset_loss_debug(viewer: object) -> None:
    viewer.s.viewport_texture = None
    viewer.s.loss_debug_texture = None
    viewer.s.debug_target_texture = None
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
    viewer.s.training_active = False
    viewer.s.training_elapsed_s = 0.0
    viewer.s.training_resume_time = None
    viewer.s.trainer = None
    if viewer.s.renderer is not None:
        viewer.s.renderer.set_debug_grad_norm_buffer(None)
        clear_grad_stats = getattr(viewer.s.renderer, "set_debug_grad_stats_buffer", None)
        if callable(clear_grad_stats):
            clear_grad_stats(None)
        viewer.s.renderer.set_debug_splat_age_buffer(None)
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
    _clear_cached_init_source(viewer)
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


def create_debug_shaders(viewer: object) -> None:
    kernels = load_compute_kernels(
        viewer.device,
        SHADER_ROOT / "renderer" / "gaussian_training_stage.slang",
        {
            "debug_abs_diff_kernel": "csComposeAbsDiffDebug",
            "debug_edge_kernel": "csComposeEdgeDebug",
            "debug_dssim_features_kernel": "csComputeSSIMFeaturesDebug",
            "debug_dssim_compose_kernel": "csComposeDSSIMDebug",
            "debug_letterbox_kernel": "csComposeLetterboxDebug",
            "debug_target_sample_kernel": "csSampleTrainingDebugTarget",
        },
    )
    viewer.s.debug_abs_diff_kernel = kernels["debug_abs_diff_kernel"]
    viewer.s.debug_edge_kernel = kernels["debug_edge_kernel"]
    viewer.s.debug_dssim_features_kernel = kernels["debug_dssim_features_kernel"]
    viewer.s.debug_dssim_compose_kernel = kernels["debug_dssim_compose_kernel"]
    viewer.s.debug_letterbox_kernel = kernels["debug_letterbox_kernel"]
    viewer.s.debug_target_sample_kernel = kernels["debug_target_sample_kernel"]


def update_debug_frame_slider_range(viewer: object) -> None:
    slider = viewer.c("loss_debug_frame")
    max_index = max(len(viewer.s.training_frames) - 1, 0)
    if hasattr(slider, "min"):
        slider.min = 0
        slider.max = int(max_index)
    slider.value = clamp_index(int(slider.value), max_index + 1)


def _training_debug_splat_age_buffer(viewer: object):
    return (
        viewer.s.trainer.refinement_buffers["splat_age"]
        if viewer.s.trainer is not None and "splat_age" in viewer.s.trainer.refinement_buffers
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
    bind_grad_stats = getattr(renderer, "set_debug_grad_stats_buffer", None)
    if callable(bind_grad_stats):
        refinement_buffers = getattr(viewer.s.trainer, "refinement_buffers", {}) if viewer.s.trainer is not None else {}
        bind_grad_stats(
            refinement_buffers["gradient_stats"]
            if "gradient_stats" in refinement_buffers
            else None
        )
    renderer.set_debug_splat_age_buffer(_training_debug_splat_age_buffer(viewer))
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
    if renderer is not None:
        renderer_size = (
            int(getattr(renderer, "_render_capacity_width", renderer.width)),
            int(getattr(renderer, "_render_capacity_height", renderer.height)),
        ) if attr == "training_renderer" else (int(renderer.width), int(renderer.height))
        if renderer_size == size: return renderer
    previous_renderer = renderer
    renderer = GaussianRenderSettings.from_renderer_params(size[0], size[1], viewer.renderer_params(allow_debug_overlays)).create_renderer(viewer.device)
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
    return GaussianRenderSettings.from_renderer_params(int(width), int(height), viewer.renderer_params(allow_debug_overlays)).create_renderer(viewer.device)


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
    current_size = (
        int(getattr(viewer.s.training_renderer, "_render_capacity_width", viewer.s.training_renderer.width)),
        int(getattr(viewer.s.training_renderer, "_render_capacity_height", viewer.s.training_renderer.height)),
    )
    if viewer.s.applied_training_runtime_factor == current_factor:
        desired_width, desired_height = viewer.s.trainer.max_training_resolution()
        if current_size == (int(desired_width), int(desired_height)):
            return False
    desired_width, desired_height = viewer.s.trainer.max_training_resolution()
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
        selected_camera_ids=tuple(int(camera_id) for camera_id in getattr(import_cfg, "selected_camera_ids", ())),
        downscale_mode=str(getattr(import_cfg, "image_downscale_mode", _COLMAP_IMAGE_DOWNSCALE_ORIGINAL)),
        downscale_max_size=int(getattr(import_cfg, "image_downscale_max_size", 2048)),
        downscale_scale=float(getattr(import_cfg, "image_downscale_scale", 1.0)),
    )


def _build_initial_training_scene(viewer: object, init: object, params: object, init_hparams: object) -> tuple[GaussianScene, float | None]:
    if viewer.s.colmap_recon is None:
        raise RuntimeError("Training scene initialization requires a loaded COLMAP reconstruction.")
    import_cfg = viewer.s.colmap_import
    min_track_length = int(getattr(import_cfg, "min_track_length", DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH))
    _ensure_cached_init_source(viewer, init)
    if import_cfg.init_mode == _COLMAP_IMPORT_CUSTOM_PLY:
        if viewer.s.cached_init_scene is None:
            raise RuntimeError("Cached custom PLY scene is unavailable.")
        return _copy_gaussian_scene(viewer.s.cached_init_scene), None

    positions = getattr(viewer.s, "cached_init_point_positions", None)
    colors = getattr(viewer.s, "cached_init_point_colors", None)
    if positions is None or colors is None:
        raise RuntimeError("Cached pointcloud initializer data is unavailable.")

    sphere_count = _resolve_fibonacci_sphere_count(int(positions.shape[0]), int(getattr(import_cfg, "fibonacci_sphere_point_count", 0)))
    sphere_radius = float(getattr(import_cfg, "fibonacci_sphere_radius", 20.0))

    if import_cfg.init_mode == _COLMAP_IMPORT_DIFFUSED_POINTCLOUD:
        base_count = int(positions.shape[0]) - sphere_count
        init_positions = positions[:base_count] if sphere_count > 0 and base_count > 0 else positions
        resolved_init = _diffused_pointcloud_init_hparams_from_positions(viewer.s.colmap_recon, init_positions, init_hparams, import_cfg.nn_radius_scale_coef, min_track_length)
        scene = initialize_scene_from_points_colors(positions, colors, init.seed, resolved_init)
        scene = _apply_fibonacci_sphere_dense_overlap_scales(scene, sphere_count, sphere_radius)
        return scene, float(max(resolved_init.base_scale, 1e-8))

    base_count = int(positions.shape[0]) - sphere_count
    chosen_base_count = base_count if params.training.max_gaussians <= 0 else min(max(int(params.training.max_gaussians), 1), base_count)
    init_positions = positions[:base_count] if sphere_count > 0 and base_count > 0 else positions
    if sphere_count > 0:
        chosen_positions = np.concatenate((positions[:chosen_base_count], positions[base_count:]), axis=0)
        chosen_colors = np.concatenate((colors[:chosen_base_count], colors[base_count:]), axis=0)
    else:
        chosen_positions = positions[:chosen_base_count]
        chosen_colors = colors[:chosen_base_count]
    resolved_init = _pointcloud_init_hparams_from_positions(viewer.s.colmap_recon, init_positions, params.training.max_gaussians, init_hparams, import_cfg.nn_radius_scale_coef, min_track_length)
    scene = initialize_scene_from_points_colors(chosen_positions, chosen_colors, init.seed, resolved_init)
    scene = _apply_fibonacci_sphere_dense_overlap_scales(scene, sphere_count, sphere_radius)
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
    training_params = viewer.training_params().training
    trainer = viewer.s.trainer
    active_sh_band = resolve_sh_band(trainer.training, trainer.state.step) if trainer is not None and hasattr(trainer, "training") and hasattr(trainer, "state") else int(getattr(training_params, "sh_band", 3 if bool(getattr(training_params, "use_sh", False)) else 0))
    viewport_sh_band = min(max(int(viewer.ui._values.get("_viewport_sh_band", active_sh_band)), 0), 3) if hasattr(viewer, "ui") else active_sh_band
    viewer.s.background = viewer.render_background()
    renderer_specs = (
        ("renderer", True, "applied_renderer_params_main"),
        ("training_renderer", False, "applied_renderer_params_training"),
        ("debug_renderer", True, "applied_renderer_params_debug"),
    )
    for attr, allow_debug, state_attr in renderer_specs:
        renderer = getattr(viewer.s, attr)
        if renderer is None:
            setattr(viewer.s, state_attr, None)
            continue
        renderer.sh_band = active_sh_band if attr == "training_renderer" else viewport_sh_band
        renderer.debug_refinement_grad_variance_weight_exponent = float(getattr(training_params, "refinement_grad_variance_weight_exponent", getattr(renderer, "debug_refinement_grad_variance_weight_exponent", 0.0)))
        renderer.debug_refinement_contribution_weight_exponent = float(getattr(training_params, "refinement_contribution_weight_exponent", getattr(renderer, "debug_refinement_contribution_weight_exponent", 0.0)))
        params = viewer.renderer_params(allow_debug)
        signature = _renderer_params_signature(params)
        if getattr(viewer.s, state_attr) == signature:
            continue
        runtime_kwargs = params.renderer_kwargs()
        if params.debug_mode is None: runtime_kwargs["debug_mode"] = GaussianRenderer.DEBUG_MODE_NORMAL
        for key, value in runtime_kwargs.items():
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
    min_value = float(viewer.ui._values.get("hist_min_value", -1.0))
    max_value = float(viewer.ui._values.get("hist_max_value", 1.0))
    step = int(viewer.s.trainer.state.step)
    scene_count = int(viewer.s.trainer.scene.count)
    signature = (step, scene_count, bin_count, min_value, max_value)
    if not refresh_requested:
        return
    viewer.s.cached_raster_grad_histograms = viewer.s.training_renderer.compute_scene_param_histograms(
        scene_count,
        bin_count=bin_count,
        min_value=min_value,
        max_value=max_value,
    )
    viewer.s.cached_raster_grad_ranges = viewer.s.training_renderer.compute_scene_param_ranges(scene_count)
    viewer.s.cached_raster_grad_histogram_mode = ""
    viewer.s.cached_raster_grad_histogram_step = step
    viewer.s.cached_raster_grad_histogram_scene_count = scene_count
    viewer.s.cached_raster_grad_histogram_signature = signature
    total = int(np.sum(viewer.s.cached_raster_grad_histograms.counts))
    viewer.s.cached_raster_grad_histogram_status = (
        f"Splat parameters | step={step:,} | samples={scene_count:,} | populated={total:,}"
        if total > 0 or step > 0
        else "No live splat parameter histogram data is available yet."
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
    selected_camera_ids: tuple[int, ...] = (),
    depth_value_mode: str = _COLMAP_DEPTH_VALUE_Z_DEPTH,
    init_mode: str,
    compress_dataset_using_bc7: bool = False,
    custom_ply_path: Path | None,
    image_downscale_mode: str,
    image_downscale_max_size: int,
    image_downscale_scale: float,
    nn_radius_scale_coef: float,
    min_track_length: int = DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH,
    depth_point_count: int = 100000,
    diffused_point_count: int = 100000,
    diffusion_radius: float = 1.0,
    fibonacci_sphere_point_count: int = 0,
    fibonacci_sphere_radius: float = 20.0,
    use_target_alpha_mask: bool = False,
    recon: object = None,
    training_frames: list[ColmapFrame] | None = None,
    frame_targets_native: list[spy.Texture] | None = None,
    cached_init_point_positions: np.ndarray | None = None,
    cached_init_point_colors: np.ndarray | None = None,
) -> None:
    if recon is None or training_frames is None:
        raise RuntimeError("COLMAP import finalize requires reconstruction and training frames.")
    resolved_selected_camera_ids = _normalized_selected_camera_ids(_camera_rows(recon), None if len(selected_camera_ids) == 0 else selected_camera_ids)
    xyz, _ = _point_tables(recon, min_track_length)
    _reset_loaded_runtime(viewer)
    _reset_training_visual_state(viewer)
    _update_import_settings(
        viewer,
        dataset_root=colmap_root,
        database_path=database_path,
        images_root=images_root,
        depth_root=depth_root,
        selected_camera_ids=resolved_selected_camera_ids,
        depth_value_mode=depth_value_mode,
        init_mode=init_mode,
        compress_dataset_using_bc7=compress_dataset_using_bc7,
        custom_ply_path=custom_ply_path,
        image_downscale_mode=image_downscale_mode,
        image_downscale_max_size=image_downscale_max_size,
        image_downscale_scale=image_downscale_scale,
        nn_radius_scale_coef=nn_radius_scale_coef,
        min_track_length=min_track_length,
        depth_point_count=depth_point_count,
        diffused_point_count=diffused_point_count,
        diffusion_radius=diffusion_radius,
        fibonacci_sphere_point_count=fibonacci_sphere_point_count,
        fibonacci_sphere_radius=fibonacci_sphere_radius,
        use_target_alpha_mask=use_target_alpha_mask,
    )
    viewer.s.colmap_root = Path(colmap_root)
    viewer.s.colmap_recon = recon
    _set_colmap_camera_preview(viewer, recon, resolved_selected_camera_ids)
    viewer.s.training_frames = list(training_frames)
    viewer.s.scene_path = None
    if cached_init_point_positions is not None and cached_init_point_colors is not None:
        cached_init_point_positions, cached_init_point_colors = _append_fibonacci_sphere_points(
            recon,
            cached_init_point_positions,
            cached_init_point_colors,
            fibonacci_sphere_point_count,
            fibonacci_sphere_radius,
        )
        viewer.s.cached_init_point_positions = np.array(cached_init_point_positions, dtype=np.float32, copy=True)
        viewer.s.cached_init_point_colors = np.array(cached_init_point_colors, dtype=np.float32, copy=True)
        init_params_fn = getattr(viewer, "init_params", None)
        if callable(init_params_fn):
            viewer.s.cached_init_signature = _cached_init_signature(viewer, init_params_fn())
    else:
        init_params_fn = getattr(viewer, "init_params", None)
        if init_mode != _COLMAP_IMPORT_DEPTH and callable(init_params_fn):
            _ensure_cached_init_source(viewer, init_params_fn())
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
    selected_camera_ids: tuple[int, ...] = (),
    depth_value_mode: str = _COLMAP_DEPTH_VALUE_Z_DEPTH,
    init_mode: str,
    custom_ply_path: Path | None,
    image_downscale_mode: str,
    image_downscale_max_size: int,
    image_downscale_scale: float,
    nn_radius_scale_coef: float,
    min_track_length: int = DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH,
    depth_point_count: int = 100000,
    diffused_point_count: int = 100000,
    diffusion_radius: float = 1.0,
    fibonacci_sphere_point_count: int = 0,
    fibonacci_sphere_radius: float = 20.0,
    use_target_alpha_mask: bool = False,
    compress_dataset_using_bc7: bool = False,
) -> None:
    _clear_loaded_scene(viewer)
    root = Path(colmap_root).resolve()
    recon = _load_aligned_colmap_reconstruction(root)
    viewer.s.colmap_import = ColmapImportSettings(
        images_root=Path(images_root).resolve(),
        compress_dataset_using_bc7=bool(compress_dataset_using_bc7),
        nn_radius_scale_coef=float(nn_radius_scale_coef),
    )
    resolved_selected_camera_ids = _normalized_selected_camera_ids(_camera_rows(recon), None if len(selected_camera_ids) == 0 else selected_camera_ids)
    training_frames = build_training_frames_from_root(
        recon,
        images_root,
        selected_camera_ids=resolved_selected_camera_ids,
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
            depth_value_mode=depth_value_mode,
            depth_point_count=depth_point_count,
            seed=int(viewer.init_params().seed),
        )
    _finish_import_colmap_dataset(
        viewer,
        colmap_root=root,
        database_path=database_path,
        images_root=images_root,
        depth_root=depth_root,
        selected_camera_ids=resolved_selected_camera_ids,
        depth_value_mode=depth_value_mode,
        init_mode=init_mode,
        compress_dataset_using_bc7=compress_dataset_using_bc7,
        custom_ply_path=custom_ply_path,
        image_downscale_mode=image_downscale_mode,
        image_downscale_max_size=image_downscale_max_size,
        image_downscale_scale=image_downscale_scale,
        nn_radius_scale_coef=nn_radius_scale_coef,
        min_track_length=min_track_length,
        depth_point_count=depth_point_count,
        diffused_point_count=diffused_point_count,
        diffusion_radius=diffusion_radius,
        fibonacci_sphere_point_count=fibonacci_sphere_point_count,
        fibonacci_sphere_radius=fibonacci_sphere_radius,
        use_target_alpha_mask=use_target_alpha_mask,
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
    depth_value_mode = _ui_depth_value_mode(viewer)
    init_mode = _ui_import_mode(viewer)
    custom_ply_text = _ui_path_string(viewer, "colmap_custom_ply_path")
    custom_ply_path = None if not custom_ply_text else Path(custom_ply_text).expanduser()
    image_downscale_mode = _ui_image_downscale_mode(viewer)
    selected_camera_ids = tuple(int(camera_id) for camera_id in viewer.ui._values.get("colmap_selected_camera_ids", ()))
    camera_rows = tuple(viewer.ui._values.get("_colmap_camera_rows", ()))
    image_downscale_max_size = max(int(viewer.ui._values.get("colmap_image_max_size", 2048)), 1)
    image_downscale_scale = float(np.clip(viewer.ui._values.get("colmap_image_scale", 1.0), 1e-6, 1.0))
    nn_radius_scale_coef = float(viewer.ui._values.get("colmap_nn_radius_scale_coef", 0.5))
    min_track_length = max(int(viewer.ui._values.get("colmap_min_track_length", DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH)), 0)
    depth_point_count = max(int(viewer.ui._values.get("colmap_depth_point_count", 100000)), 1)
    diffused_point_count = max(int(viewer.ui._values.get("colmap_diffused_point_count", 100000)), 1)
    diffusion_radius = max(float(viewer.ui._values.get("colmap_diffusion_radius", 1.0)), 0.0)
    fibonacci_sphere_point_count = max(int(viewer.ui._values.get("colmap_fibonacci_sphere_point_count", 0)), 0)
    fibonacci_sphere_radius = max(float(viewer.ui._values.get("colmap_fibonacci_sphere_radius", 20.0)), 0.0)
    use_target_alpha_mask = bool(viewer.ui._values.get("use_target_alpha_mask", False))
    compress_dataset_using_bc7 = bool(viewer.ui._values.get("compress_dataset_using_bc7", False))
    if not colmap_root.exists():
        raise FileNotFoundError(f"COLMAP root does not exist: {colmap_root}")
    if not _has_colmap_sparse(colmap_root):
        colmap_root = _resolve_colmap_root_from_selection(colmap_root)
    if database_path is not None and not database_path.exists():
        database_path = _find_optional_colmap_database(colmap_root)
    if not images_root.exists():
        raise FileNotFoundError(f"COLMAP image folder does not exist: {images_root}")
    if len(camera_rows) > 0 and len(selected_camera_ids) == 0:
        raise ValueError("Select at least one COLMAP camera model before importing.")
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
        selected_camera_ids=selected_camera_ids,
        depth_root=None if depth_root is None else depth_root.resolve(),
        depth_value_mode=depth_value_mode,
        init_mode=init_mode,
        custom_ply_path=None if custom_ply_path is None else custom_ply_path.resolve(),
        compress_dataset_using_bc7=compress_dataset_using_bc7,
        image_downscale_mode=image_downscale_mode,
        image_downscale_max_size=image_downscale_max_size,
        image_downscale_scale=image_downscale_scale,
        nn_radius_scale_coef=float(max(nn_radius_scale_coef, 1e-4)),
        min_track_length=min_track_length,
        depth_point_count=depth_point_count,
        diffused_point_count=diffused_point_count,
        diffusion_radius=diffusion_radius,
        fibonacci_sphere_point_count=fibonacci_sphere_point_count,
        fibonacci_sphere_radius=fibonacci_sphere_radius,
        use_target_alpha_mask=use_target_alpha_mask,
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
                image_source, payload = load_result if progress.init_mode == _COLMAP_IMPORT_DEPTH else (load_result, None)
                progress.native_textures.append(
                    _create_native_dataset_texture_from_bc_payload(viewer, image_source)
                    if isinstance(image_source, _CompressedDatasetTexture)
                    else _create_native_dataset_texture_from_rgba8(viewer, image_source)
                )
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
                    progress.depth_value_mode,
                )
            _finish_import_colmap_dataset(
                viewer,
                colmap_root=progress.colmap_root,
                database_path=progress.database_path,
                images_root=progress.images_root,
                depth_root=progress.depth_root,
                selected_camera_ids=tuple(int(camera_id) for camera_id in getattr(progress, "selected_camera_ids", ())),
                depth_value_mode=progress.depth_value_mode,
                init_mode=progress.init_mode,
                compress_dataset_using_bc7=progress.compress_dataset_using_bc7,
                custom_ply_path=progress.custom_ply_path,
                image_downscale_mode=progress.image_downscale_mode,
                image_downscale_max_size=progress.image_downscale_max_size,
                image_downscale_scale=progress.image_downscale_scale,
                nn_radius_scale_coef=progress.nn_radius_scale_coef,
                min_track_length=progress.min_track_length,
                depth_point_count=progress.depth_point_count,
                diffused_point_count=progress.diffused_point_count,
                diffusion_radius=progress.diffusion_radius,
                fibonacci_sphere_point_count=progress.fibonacci_sphere_point_count,
                fibonacci_sphere_radius=progress.fibonacci_sphere_radius,
                use_target_alpha_mask=progress.use_target_alpha_mask,
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
    resolutions = []
    for frame in viewer.s.training_frames:
        try:
            factor = resolve_effective_train_render_factor(params.training, 0, int(frame.width), int(frame.height))
        except TypeError:
            factor = resolve_effective_train_render_factor(params.training, 0)
        resolutions.append(resolve_training_resolution(int(frame.width), int(frame.height), int(factor)))
    width, height = max(w for w, _ in resolutions), max(h for _, h in resolutions)
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
    elif getattr(viewer.s.colmap_import, "images_root", None) is not None and all(hasattr(frame, "image_path") for frame in viewer.s.training_frames):
        trainer_kwargs["frame_targets_native"] = _create_native_dataset_textures(viewer, viewer.s.training_frames)
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
    _reset_training_visual_state(viewer)
    update_debug_frame_slider_range(viewer)
    _reset_loss_debug(viewer)
    viewer.s.last_error = ""
    print(f"Initialized training scene ({scene.count:,} gaussians, profile={profile.name})")


def reinitialize_training_scene(viewer: object) -> None:
    frame_targets_native = None
    trainer = getattr(viewer.s, "trainer", None)
    training_frames = tuple(getattr(viewer.s, "training_frames", ()))
    existing_targets = getattr(trainer, "_frame_targets_native", None) if trainer is not None else None
    if isinstance(existing_targets, list) and len(existing_targets) == len(training_frames) and len(existing_targets) > 0:
        frame_targets_native = list(existing_targets)
    initialize_training_scene(viewer, frame_targets_native=frame_targets_native)


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
