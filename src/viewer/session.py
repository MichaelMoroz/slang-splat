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

from ..app.shared import apply_training_profile, estimate_point_bounds, estimate_scene_bounds, fit_camera
from ..utility import SHADER_ROOT, clamp_index, drain_all_deferred_resource_releases, load_compute_kernels
from ..metrics import PARAM_HISTOGRAM_SCALE_LINEAR, PARAM_HISTOGRAM_SCALE_LOG10, ParamLog10Histograms, ParamTensorRanges
from ..renderer import GaussianRenderSettings, GaussianRenderer
from ..scene import (
    GaussianScene,
    build_training_frames_from_root,
    initialize_scene_from_points_colors,
    load_colmap_reconstruction,
    load_gaussian_ply,
    resolve_colmap_fibonacci_sphere_radius,
    resolve_colmap_init_hparams,
    resolve_points_init_hparams,
    sample_colmap_diffused_points,
    sample_colmap_fibonacci_sphere_points,
    sample_mesh_surface_points,
)
from ..scene._internal.colmap_ops import point_nn_scales, resolve_training_frame_image_size
from ..training import resolve_sh_band
from ..scene._internal.colmap_ops import (
    DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH,
    DEPTH_INIT_VALUE_DISTANCE,
    DEPTH_INIT_VALUE_Z_DEPTH,
    FIBONACCI_SPHERE_COLOR,
    TRAINING_FRAME_LOAD_THREADS,
    build_depth_path_index,
    generate_depth_init_points,
    load_training_frame_rgba8,
    load_training_frame_rgba8_with_depth_payload,
    match_depth_path,
)
from ..training.alpha_modes import TARGET_ALPHA_MODE_OFF, resolve_target_alpha_mode
from ..training import (
    GaussianTrainer,
    PhotometricCompensationHyperParams,
    PhotometricCompensationTrainer,
    resolve_effective_train_render_factor,
    resolve_training_resolution,
)
from ..training.image_color_init import TrainingImageColorInitializer
from ..scene._internal.colmap_types import ColmapFrame, ColmapReconstruction, point_tables
from .session_colmap_utils import (
    _COLMAP_CAMERA_MODEL_NAMES,
    _COLMAP_DB_SAMPLE_LIMIT,
    _COLMAP_DEPTH_VALUE_DISTANCE,
    _COLMAP_DEPTH_VALUE_Z_DEPTH,
    _COLMAP_IMAGE_DOWNSCALE_MAX_SIZE,
    _COLMAP_IMAGE_DOWNSCALE_ORIGINAL,
    _COLMAP_IMAGE_DOWNSCALE_SCALE,
    _COLMAP_IMPORT_CUSTOM_PLY,
    _COLMAP_IMPORT_CUSTOM_MESH,
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
    _point_preview_stats as _point_preview_stats_impl,
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

_REFINEMENT_HISTOGRAM_LOG10_FALLBACK_RANGE = (-6.0, 1.0)
_HISTOGRAM_RANGE_EPS = 1e-6
_DATASET_BC7_TEXCONV_URL = "https://github.com/microsoft/DirectXTex/releases/download/oct2024/texconv.exe"
_DATASET_BC7_TEXCONV_PATH = Path(__file__).resolve().parents[2] / "temp" / "bc_texture_tools" / "texconv.exe"
_PERIODIC_RENDERER_REALLOCATION_INTERVAL_S = 120.0
_TRAINING_RUNTIME_PARAM_NAMES = (
    "max_sh_band",
    "train_downscale_mode",
    "train_auto_start_downscale",
    "train_downscale_base_iters",
    "train_downscale_iter_step",
    "train_downscale_max_iters",
    "train_downscale_factor",
    "train_subsample_factor",
)
_MESH_TO_COLMAP_COORDINATE_TRANSFORM = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


_has_colmap_sparse = _has_colmap_sparse_impl
_resolve_colmap_root_from_selection = _resolve_colmap_root_from_selection_impl
_load_aligned_colmap_reconstruction = _load_aligned_colmap_reconstruction_impl
_set_ui_path = _set_ui_path_impl


def _param_value_scales(payload: object, row_count: int, default: str = PARAM_HISTOGRAM_SCALE_LINEAR) -> tuple[str, ...]:
    scales = tuple(str(scale) for scale in getattr(payload, "param_value_scales", ()))
    return scales if len(scales) == int(row_count) else (default,) * int(row_count)


def _param_range_bounds(payload: object, value_scale: str, fallback: tuple[float, float]) -> tuple[float, float]:
    if payload is None:
        return fallback
    min_values = np.asarray(getattr(payload, "min_values", np.zeros((0,), dtype=np.float32)), dtype=np.float64).reshape(-1)
    max_values = np.asarray(getattr(payload, "max_values", np.zeros((0,), dtype=np.float32)), dtype=np.float64).reshape(-1)
    if min_values.size == 0 or min_values.size != max_values.size:
        return fallback
    scales = _param_value_scales(payload, min_values.size)
    scale_mask = np.asarray([scale == value_scale for scale in scales], dtype=bool)
    values = np.concatenate((min_values[scale_mask], max_values[scale_mask]), axis=0)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fallback
    lo, hi = float(np.min(values)), float(np.max(values))
    return (lo, hi) if hi > lo else (lo, lo + _HISTOGRAM_RANGE_EPS)


def _scene_histogram_bounds(payload: object, position_min: float, position_max: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    if payload is None:
        return None, None
    min_values = np.asarray(getattr(payload, "min_values", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    max_values = np.asarray(getattr(payload, "max_values", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if min_values.size == 0 or min_values.size != max_values.size:
        return None, None
    scales = _param_value_scales(payload, min_values.size)
    bounds_min = min_values.astype(np.float32, copy=True)
    bounds_max = max_values.astype(np.float32, copy=True)
    position_lo, position_hi = float(position_min), float(position_max)
    if position_hi <= position_lo:
        position_hi = position_lo + _HISTOGRAM_RANGE_EPS
    for group_name, indices in tuple(getattr(payload, "param_groups", ())):
        if str(group_name) != "position":
            continue
        for index in indices:
            idx = int(index)
            if 0 <= idx < min_values.size:
                bounds_min[idx] = np.float32(position_lo)
                bounds_max[idx] = np.float32(position_hi)
    for index, scale in enumerate(scales):
        fallback = _REFINEMENT_HISTOGRAM_LOG10_FALLBACK_RANGE if scale == PARAM_HISTOGRAM_SCALE_LOG10 else (position_lo, position_hi)
        if not np.isfinite(bounds_min[index]):
            bounds_min[index] = np.float32(fallback[0])
        if not np.isfinite(bounds_max[index]):
            bounds_max[index] = np.float32(fallback[1])
        if bounds_max[index] <= bounds_min[index]:
            bounds_max[index] = np.float32(float(bounds_min[index]) + _HISTOGRAM_RANGE_EPS)
    return bounds_min, bounds_max


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
    viewer.ui._values["_colmap_point_stats"] = _point_preview_stats_impl(recon)
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
            clear_scene_resources = getattr(value, "clear_scene_resources", None)
            if callable(clear_scene_resources):
                clear_scene_resources()
            setattr(viewer.s, attr, None)
            del value


def _flush_deferred_resources(viewer: object) -> None:
    wait = getattr(getattr(viewer, "device", None), "wait", None)
    if callable(wait):
        wait()
    drain_all_deferred_resource_releases(min_age=0, advance_generation=False)


def _training_camera(frames: object, near: float, far: float):
    for frame in tuple(frames):
        make_camera = getattr(frame, "make_camera", None)
        if not callable(make_camera):
            continue
        try:
            camera = make_camera(near=float(near), far=float(far))
        except TypeError:
            camera = make_camera()
        except Exception:
            continue
        position = np.asarray(getattr(camera, "position", ()), dtype=np.float32).reshape(-1)
        target = np.asarray(getattr(camera, "target", ()), dtype=np.float32).reshape(-1)
        up = np.asarray(getattr(camera, "up", ()), dtype=np.float32).reshape(-1)
        if position.size >= 3 and target.size >= 3 and up.size >= 3 and np.all(np.isfinite(position[:3])) and np.all(np.isfinite(target[:3])) and np.all(np.isfinite(up[:3])):
            return camera
    return None


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
    return _COLMAP_IMPORT_DEPTH if int(viewer.ui._values.get("colmap_init_mode", 0)) == 1 else _COLMAP_IMPORT_POINTCLOUD


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
        camera_id=int(image.camera_id),
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


def choose_colmap_custom_mesh(viewer: object, mesh_path: Path) -> None:
    resolved_path = Path(mesh_path).resolve()
    _set_ui_path(viewer, "colmap_custom_mesh_path", resolved_path)
    viewer.s.last_error = ""


def _import_cfg_enabled(import_cfg: object, attr_name: str, default: bool = False) -> bool:
    return bool(getattr(import_cfg, attr_name, default))


def _import_cfg_nn_radius_scale_coef(import_cfg: object, attr_name: str, default: float = 0.5, fallback_attr: str | None = "nn_radius_scale_coef") -> float:
    fallback = default if fallback_attr is None else getattr(import_cfg, fallback_attr, default)
    return float(max(getattr(import_cfg, attr_name, fallback), 1e-4))


def _fibonacci_source_enabled(import_cfg: object) -> bool:
    return _import_cfg_enabled(import_cfg, "fibonacci_sphere_enabled", int(getattr(import_cfg, "fibonacci_sphere_point_count", 0)) > 0)


def _uses_depth_init(import_cfg: object) -> bool:
    if import_cfg is None:
        return False
    return str(getattr(import_cfg, "init_mode", _COLMAP_IMPORT_POINTCLOUD)) == _COLMAP_IMPORT_DEPTH and not any((
        _import_cfg_enabled(import_cfg, "pointcloud_enabled"),
        _import_cfg_enabled(import_cfg, "diffused_enabled"),
        _import_cfg_enabled(import_cfg, "custom_ply_enabled"),
        _import_cfg_enabled(import_cfg, "custom_mesh_enabled"),
    ))


def _enabled_init_source_names(import_cfg: object) -> tuple[str, ...]:
    names: list[str] = []
    if _import_cfg_enabled(import_cfg, "pointcloud_enabled"):
        names.append(_COLMAP_IMPORT_POINTCLOUD)
    if _import_cfg_enabled(import_cfg, "diffused_enabled"):
        names.append(_COLMAP_IMPORT_DIFFUSED_POINTCLOUD)
    if _import_cfg_enabled(import_cfg, "custom_ply_enabled"):
        names.append(_COLMAP_IMPORT_CUSTOM_PLY)
    if _import_cfg_enabled(import_cfg, "custom_mesh_enabled"):
        names.append(_COLMAP_IMPORT_CUSTOM_MESH)
    if _fibonacci_source_enabled(import_cfg):
        names.append("fibonacci_sphere")
    return tuple(names)


def _diffused_diffusion_radius(import_cfg: object) -> float:
    return max(float(getattr(import_cfg, "diffused_diffusion_radius", 1.0)), 0.0)


def _custom_mesh_path(import_cfg: object) -> Path | None:
    mesh_path = getattr(import_cfg, "custom_mesh_path", None)
    return None if mesh_path is None else Path(mesh_path)


def _custom_mesh_point_count(import_cfg: object) -> int:
    return max(int(getattr(import_cfg, "custom_mesh_point_count", getattr(import_cfg, "diffused_point_count", 100000))), 1)


def _fibonacci_radius_multiplier(import_cfg: object) -> float:
    return max(float(getattr(import_cfg, "fibonacci_sphere_radius_multiplier", getattr(import_cfg, "fibonacci_sphere_radius", 2.0))), 0.0)


def _fibonacci_sphere_color(import_cfg: object) -> tuple[float, float, float]:
    return tuple(float(v) for v in np.clip(np.asarray(getattr(import_cfg, "fibonacci_sphere_color", FIBONACCI_SPHERE_COLOR), dtype=np.float32).reshape(3), 0.0, 1.0))


def _concat_gaussian_scenes(scenes: list[GaussianScene]) -> GaussianScene:
    if len(scenes) == 0:
        raise RuntimeError("Enable at least one initialization source before building the training scene.")
    if len(scenes) == 1:
        return scenes[0]
    return GaussianScene(
        positions=np.ascontiguousarray(np.concatenate([np.asarray(scene.positions, dtype=np.float32) for scene in scenes], axis=0), dtype=np.float32),
        scales=np.ascontiguousarray(np.concatenate([np.asarray(scene.scales, dtype=np.float32) for scene in scenes], axis=0), dtype=np.float32),
        rotations=np.ascontiguousarray(np.concatenate([np.asarray(scene.rotations, dtype=np.float32) for scene in scenes], axis=0), dtype=np.float32),
        opacities=np.ascontiguousarray(np.concatenate([np.asarray(scene.opacities, dtype=np.float32) for scene in scenes], axis=0), dtype=np.float32),
        colors=np.ascontiguousarray(np.concatenate([np.asarray(scene.colors, dtype=np.float32) for scene in scenes], axis=0), dtype=np.float32),
        sh_coeffs=np.ascontiguousarray(np.concatenate([np.asarray(scene.sh_coeffs, dtype=np.float32) for scene in scenes], axis=0), dtype=np.float32),
    )


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


def _sampled_point_init_hparams_from_positions(positions: np.ndarray, max_gaussians: int, init_hparams: object, nn_radius_scale_coef: float):
    resolved = resolve_points_init_hparams(positions, max_gaussians, init_hparams)
    chosen_count = positions.shape[0] if max_gaussians <= 0 else min(max(int(max_gaussians), 1), positions.shape[0])
    chosen_positions = np.ascontiguousarray(positions[:chosen_count], dtype=np.float32)
    median_nn_scale = float(np.median(point_nn_scales(chosen_positions))) if chosen_count > 0 else 1.0
    return replace(
        resolved,
        position_jitter_std=0.0,
        base_scale=float(max(float(nn_radius_scale_coef), 1e-4) * max(median_nn_scale, 1e-6)),
    )


def _append_fibonacci_sphere_points(
    recon: object,
    positions: np.ndarray,
    colors: np.ndarray,
    point_count: int,
    radius_multiplier: float,
    sphere_color: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    count = max(int(point_count), 0)
    if count <= 0:
        return positions, colors
    sphere_positions, sphere_colors = sample_colmap_fibonacci_sphere_points(recon, count, radius_multiplier, sphere_color=sphere_color)
    return (
        np.ascontiguousarray(np.concatenate((np.asarray(positions, dtype=np.float32), sphere_positions), axis=0), dtype=np.float32),
        np.ascontiguousarray(np.concatenate((np.asarray(colors, dtype=np.float32), sphere_colors), axis=0), dtype=np.float32),
    )


def _resolve_fibonacci_sphere_count(total_count: int, point_count: int) -> int:
    return min(max(int(point_count), 0), max(int(total_count), 0))


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
    setattr(viewer.s, "cached_init_signature", None)
    setattr(viewer.s, "cached_init_pointcloud_positions", None)
    setattr(viewer.s, "cached_init_pointcloud_colors", None)
    setattr(viewer.s, "cached_init_diffused_positions", None)
    setattr(viewer.s, "cached_init_diffused_colors", None)
    setattr(viewer.s, "cached_init_custom_ply_scene", None)
    setattr(viewer.s, "cached_init_custom_mesh_positions", None)
    setattr(viewer.s, "cached_init_custom_mesh_colors", None)
    setattr(viewer.s, "cached_init_fibonacci_positions", None)
    setattr(viewer.s, "cached_init_fibonacci_colors", None)


def _cached_init_signature(viewer: object, init: object) -> tuple[object, ...] | None:
    import_cfg = getattr(viewer.s, "colmap_import", None)
    min_track_length = int(getattr(import_cfg, "min_track_length", DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH))
    if import_cfg is None:
        return None
    if _uses_depth_init(import_cfg):
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
            round(float(_fibonacci_radius_multiplier(import_cfg)), 6),
            tuple(round(float(v), 6) for v in _fibonacci_sphere_color(import_cfg)),
        )
    return (
        None if viewer.s.colmap_root is None else str(Path(viewer.s.colmap_root).resolve()),
        tuple(_enabled_init_source_names(import_cfg)),
        min_track_length,
        round(float(_import_cfg_nn_radius_scale_coef(import_cfg, "pointcloud_nn_radius_scale_coef")), 6),
        int(getattr(import_cfg, "diffused_point_count", 0)),
        round(float(_diffused_diffusion_radius(import_cfg)), 6),
        round(float(_import_cfg_nn_radius_scale_coef(import_cfg, "diffused_nn_radius_scale_coef")), 6),
        None if getattr(import_cfg, "custom_ply_path", None) is None else str(Path(import_cfg.custom_ply_path).resolve()),
        round(float(_import_cfg_nn_radius_scale_coef(import_cfg, "custom_ply_nn_radius_scale_coef", default=1.0, fallback_attr=None)), 6),
        None if _custom_mesh_path(import_cfg) is None else str(Path(_custom_mesh_path(import_cfg)).resolve()),
        int(_custom_mesh_point_count(import_cfg)),
        round(float(_import_cfg_nn_radius_scale_coef(import_cfg, "custom_mesh_nn_radius_scale_coef")), 6),
        int(getattr(import_cfg, "fibonacci_sphere_point_count", 0)),
        round(float(_fibonacci_radius_multiplier(import_cfg)), 6),
        tuple(round(float(v), 6) for v in _fibonacci_sphere_color(import_cfg)),
        round(float(_import_cfg_nn_radius_scale_coef(import_cfg, "fibonacci_sphere_nn_radius_scale_coef", default=1.0, fallback_attr=None)), 6),
        int(init.seed),
    )


def _store_point_source_cache(viewer: object, source_name: str, positions: np.ndarray, colors: np.ndarray) -> None:
    positions_copy = np.array(positions, dtype=np.float32, copy=True)
    colors_copy = np.array(colors, dtype=np.float32, copy=True)
    if source_name == _COLMAP_IMPORT_POINTCLOUD:
        viewer.s.cached_init_pointcloud_positions = positions_copy
        viewer.s.cached_init_pointcloud_colors = colors_copy
    elif source_name == _COLMAP_IMPORT_DIFFUSED_POINTCLOUD:
        viewer.s.cached_init_diffused_positions = positions_copy
        viewer.s.cached_init_diffused_colors = colors_copy
    elif source_name == _COLMAP_IMPORT_CUSTOM_MESH:
        viewer.s.cached_init_custom_mesh_positions = positions_copy
        viewer.s.cached_init_custom_mesh_colors = colors_copy
    elif source_name == "fibonacci_sphere":
        viewer.s.cached_init_fibonacci_positions = positions_copy
        viewer.s.cached_init_fibonacci_colors = colors_copy


def _has_cached_init_source(viewer: object, source_name: str) -> bool:
    if source_name == _COLMAP_IMPORT_CUSTOM_PLY:
        return getattr(viewer.s, "cached_init_custom_ply_scene", None) is not None
    if source_name == _COLMAP_IMPORT_POINTCLOUD:
        positions = getattr(viewer.s, "cached_init_pointcloud_positions", None)
        colors = getattr(viewer.s, "cached_init_pointcloud_colors", None)
        return positions is not None and colors is not None
    if source_name == _COLMAP_IMPORT_DIFFUSED_POINTCLOUD:
        positions = getattr(viewer.s, "cached_init_diffused_positions", None)
        colors = getattr(viewer.s, "cached_init_diffused_colors", None)
        return positions is not None and colors is not None
    if source_name == _COLMAP_IMPORT_CUSTOM_MESH:
        positions = getattr(viewer.s, "cached_init_custom_mesh_positions", None)
        colors = getattr(viewer.s, "cached_init_custom_mesh_colors", None)
        return positions is not None and colors is not None
    if source_name == "fibonacci_sphere":
        positions = getattr(viewer.s, "cached_init_fibonacci_positions", None)
        colors = getattr(viewer.s, "cached_init_fibonacci_colors", None)
        return positions is not None and colors is not None
    return False


def _load_enabled_init_source_payloads(viewer: object, init: object) -> None:
    import_cfg = viewer.s.colmap_import
    min_track_length = int(getattr(import_cfg, "min_track_length", DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH))
    for source_name in _enabled_init_source_names(import_cfg):
        if source_name == _COLMAP_IMPORT_POINTCLOUD:
            positions, colors = _point_tables(viewer.s.colmap_recon, min_track_length)
            _store_point_source_cache(viewer, source_name, positions, colors)
            continue
        if source_name == _COLMAP_IMPORT_DIFFUSED_POINTCLOUD:
            positions, colors = sample_colmap_diffused_points(
                viewer.s.colmap_recon,
                int(getattr(import_cfg, "diffused_point_count", 100000)),
                _diffused_diffusion_radius(import_cfg),
                int(init.seed),
                min_track_length=min_track_length,
            )
            _store_point_source_cache(viewer, source_name, positions, colors)
            continue
        if source_name == _COLMAP_IMPORT_CUSTOM_PLY:
            if getattr(import_cfg, "custom_ply_path", None) is None:
                raise RuntimeError("Custom PLY initialization requires a selected PLY file.")
            viewer.s.cached_init_custom_ply_scene = _copy_gaussian_scene(load_gaussian_ply(import_cfg.custom_ply_path))
            continue
        if source_name == _COLMAP_IMPORT_CUSTOM_MESH:
            mesh_path = _custom_mesh_path(import_cfg)
            if mesh_path is None:
                raise RuntimeError("Custom mesh initialization requires a selected mesh file.")
            positions, colors = sample_mesh_surface_points(mesh_path, _custom_mesh_point_count(import_cfg), int(init.seed))
            positions = np.ascontiguousarray(np.asarray(positions, dtype=np.float32) @ _MESH_TO_COLMAP_COORDINATE_TRANSFORM[:3, :3].T, dtype=np.float32)
            _store_point_source_cache(viewer, source_name, positions, colors)
            continue
        if source_name == "fibonacci_sphere":
            positions, colors = sample_colmap_fibonacci_sphere_points(
                viewer.s.colmap_recon,
                int(getattr(import_cfg, "fibonacci_sphere_point_count", 0)),
                _fibonacci_radius_multiplier(import_cfg),
                sphere_color=_fibonacci_sphere_color(import_cfg),
            )
            _store_point_source_cache(viewer, source_name, positions, colors)


def _ensure_cached_init_source(viewer: object, init: object) -> None:
    if viewer.s.colmap_recon is None:
        return
    import_cfg = viewer.s.colmap_import
    signature = _cached_init_signature(viewer, init)
    cached_signature = getattr(viewer.s, "cached_init_signature", None)
    cached_ply_scene = getattr(viewer.s, "cached_init_custom_ply_scene", None)
    cached_positions = getattr(viewer.s, "cached_init_point_positions", None)
    cached_colors = getattr(viewer.s, "cached_init_point_colors", None)
    if signature is not None and cached_signature == signature:
        if _uses_depth_init(import_cfg):
            if cached_positions is not None and cached_colors is not None:
                return
        elif all(_has_cached_init_source(viewer, source_name) for source_name in _enabled_init_source_names(import_cfg)):
            return
    if _uses_depth_init(import_cfg):
        raise RuntimeError("Depth initialization cache is unavailable. Re-import the dataset to rebuild the calibrated point cloud.")
    _clear_cached_init_source(viewer)
    _load_enabled_init_source_payloads(viewer, init)
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


def _reset_training_runtime(viewer: object, *, preserve_frame_targets: bool = False) -> None:
    viewer.s.training_active = False
    viewer.s.training_elapsed_s = 0.0
    viewer.s.training_resume_time = None
    trainer = getattr(viewer.s, "trainer", None)
    release_resources = getattr(trainer, "release_resources", None)
    if callable(release_resources):
        release_resources(preserve_frame_targets=bool(preserve_frame_targets))
    viewer.s.trainer = None
    _clear_debug_buffers(getattr(viewer.s, "renderer", None))
    _clear_debug_buffers(getattr(viewer.s, "debug_renderer", None))
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
    _flush_deferred_resources(viewer)


def _reset_loaded_runtime(viewer: object) -> None:
    viewer.s.colmap_import_progress = None
    _reset_training_runtime(viewer)
    reset_photometric_compensation(viewer)
    _clear_cached_init_source(viewer)
    _clear_main_camera_reset_state(viewer)
    viewer.s.training_camera_colmap_observation_index = None
    viewer.s.training_camera_colmap_observation_signature = None
    viewer.s.training_camera_colmap_payload = None
    viewer.s.training_camera_colmap_payload_signature = None
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
    _flush_deferred_resources(viewer)


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


def _selected_training_debug_frame_index(viewer: object) -> int:
    frames = tuple(viewer.s.training_frames)
    if len(frames) == 0:
        return 0
    try:
        value = int(viewer.c("loss_debug_frame").value)
    except Exception:
        value = int(viewer.ui._values.get("loss_debug_frame", 0))
    return clamp_index(value, len(frames))


def selected_training_debug_camera(viewer: object):
    state = viewer.s
    frames = tuple(state.training_frames)
    if len(frames) == 0:
        return None, None, 0
    frame_idx = _selected_training_debug_frame_index(viewer)
    frame = frames[frame_idx]
    trainer = state.trainer
    if trainer is not None and hasattr(trainer, "make_frame_camera"):
        try:
            width, height = trainer.frame_size(frame_idx) if hasattr(trainer, "frame_size") else (int(getattr(frame, "width", 1)), int(getattr(frame, "height", 1)))
            return frame, trainer.make_frame_camera(frame_idx, int(width), int(height)), frame_idx
        except Exception:
            pass
    make_camera = getattr(frame, "make_camera", None)
    if not callable(make_camera):
        return frame, None, frame_idx
    try:
        camera = make_camera(near=float(getattr(state, "near", 0.1)), far=float(getattr(state, "far", 120.0)))
    except TypeError:
        camera = make_camera()
    except Exception:
        camera = None
    return frame, camera, frame_idx


def move_main_camera_to_selected_training_frame(viewer: object) -> int:
    frame, camera, frame_idx = selected_training_debug_camera(viewer)
    if frame is None or camera is None:
        raise RuntimeError("Training camera pose is unavailable.")
    apply_camera_pose = getattr(viewer, "apply_camera_pose", None)
    if not callable(apply_camera_pose):
        raise RuntimeError("Viewer cannot apply camera poses.")
    apply_camera_pose(camera)
    viewer.ui._values["show_training_cameras"] = False
    return int(frame_idx)


def _clear_main_camera_reset_state(viewer: object) -> None:
    state = viewer.s
    state.camera_reset_position = None
    state.camera_reset_up = None
    state.camera_reset_yaw = None
    state.camera_reset_pitch = None
    state.camera_reset_near = None
    state.camera_reset_far = None
    state.camera_reset_move_speed = None


def _store_main_camera_reset_state(viewer: object) -> None:
    state = viewer.s
    position = np.asarray(getattr(state, "camera_pos", ()), dtype=np.float32).reshape(-1)
    up = np.asarray(getattr(state, "up", ()), dtype=np.float32).reshape(-1)
    if position.size < 3 or up.size < 3 or not np.all(np.isfinite(position[:3])) or not np.all(np.isfinite(up[:3])):
        return
    state.camera_reset_position = tuple(float(value) for value in position[:3])
    state.camera_reset_up = tuple(float(value) for value in up[:3])
    state.camera_reset_yaw = float(getattr(state, "yaw", 0.0))
    state.camera_reset_pitch = float(getattr(state, "pitch", 0.0))
    state.camera_reset_near = float(getattr(state, "near", 0.1))
    state.camera_reset_far = float(getattr(state, "far", 120.0))
    state.camera_reset_move_speed = float(getattr(state, "move_speed", 2.0))


def _restore_main_camera_reset_state(viewer: object) -> bool:
    state = viewer.s
    position = getattr(state, "camera_reset_position", None)
    up = getattr(state, "camera_reset_up", None)
    yaw = getattr(state, "camera_reset_yaw", None)
    pitch = getattr(state, "camera_reset_pitch", None)
    near = getattr(state, "camera_reset_near", None)
    far = getattr(state, "camera_reset_far", None)
    move_speed = getattr(state, "camera_reset_move_speed", None)
    if None in (position, up, yaw, pitch, near, far, move_speed):
        return False
    state.camera_pos = spy.float3(*position)
    state.up = spy.float3(*up)
    state.yaw = float(yaw)
    state.pitch = float(pitch)
    state.near = float(near)
    state.far = float(far)
    state.move_speed = float(move_speed)
    state.move_vel = spy.float3(0.0, 0.0, 0.0)
    state.rot_vel = spy.float2(0.0, 0.0)
    control = getattr(viewer, "c", None)
    if callable(control):
        try:
            control("move_speed").value = float(move_speed)
        except Exception:
            pass
    return True


def _camera_reset_scene(viewer: object) -> GaussianScene | None:
    state = viewer.s
    scene = getattr(state, "scene", None)
    if isinstance(scene, GaussianScene) and scene.count > 0:
        return scene
    return None


def _camera_reset_points(viewer: object) -> np.ndarray | None:
    state = viewer.s
    for attr in (
        "cached_init_point_positions",
        "cached_init_pointcloud_positions",
        "cached_init_diffused_positions",
        "cached_init_custom_mesh_positions",
        "cached_init_fibonacci_positions",
    ):
        points = getattr(state, attr, None)
        if isinstance(points, np.ndarray) and points.ndim == 2 and points.shape[0] > 0:
            return points
    recon = getattr(state, "colmap_recon", None)
    if recon is None:
        return None
    min_track_length = int(getattr(getattr(state, "colmap_import", None), "min_track_length", DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH))
    try:
        points, _ = _point_tables(recon, min_track_length=min_track_length)
    except Exception:
        return None
    return points if points.ndim == 2 and points.shape[0] > 0 else None


def reset_main_camera(viewer: object) -> None:
    state = viewer.s
    if _restore_main_camera_reset_state(viewer):
        viewer.ui._values["show_training_cameras"] = False
        return
    fallback_bounds = None
    scene = _camera_reset_scene(viewer)
    if scene is not None:
        fallback_bounds = estimate_scene_bounds(scene)
    else:
        points = _camera_reset_points(viewer)
        if points is not None:
            fallback_bounds = estimate_point_bounds(points)
    fit = None
    if fallback_bounds is not None:
        try:
            fit = fit_camera(fallback_bounds, getattr(state, "fov_y", 60.0))
        except Exception:
            fit = None
    training_frames = getattr(state, "training_frames", ())
    if fallback_bounds is None and not training_frames:
        raise RuntimeError("Viewer scene bounds are unavailable.")
    camera = _training_camera(
        training_frames,
        getattr(state, "near", 0.1) if fit is None else fit.near,
        getattr(state, "far", 120.0) if fit is None else fit.far,
    )
    apply_camera_position = getattr(viewer, "apply_camera_position", None)
    if camera is not None and callable(apply_camera_position):
        apply_camera_position(camera, near=None if fit is None else fit.near, far=None if fit is None else fit.far, move_speed=None if fit is None else fit.move_speed)
    elif fallback_bounds is not None:
        viewer.apply_camera_fit(fallback_bounds)
    else:
        raise RuntimeError("Viewer scene bounds are unavailable.")
    _store_main_camera_reset_state(viewer)
    viewer.ui._values["show_training_cameras"] = False


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


def _clear_debug_buffers(renderer: GaussianRenderer | None) -> None:
    if renderer is None:
        return
    renderer.set_debug_grad_norm_buffer(None)
    bind_grad_stats = getattr(renderer, "set_debug_grad_stats_buffer", None)
    if callable(bind_grad_stats):
        bind_grad_stats(None)
    renderer.set_debug_splat_age_buffer(None)
    bind_contribution = getattr(renderer, "set_debug_splat_contribution_buffer", None)
    if callable(bind_contribution):
        bind_contribution(None)
    bind_adam_moments = getattr(renderer, "set_debug_adam_moments_buffer", None)
    if callable(bind_adam_moments):
        bind_adam_moments(None)
    set_contribution_pixels = getattr(renderer, "set_debug_contribution_observed_pixel_count", None)
    if callable(set_contribution_pixels):
        set_contribution_pixels(0)


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


def ensure_renderer(viewer: object, attr: str, width: int, height: int, allow_debug_overlays: bool, *, force_recreate: bool = False) -> GaussianRenderer:
    size, renderer = (int(width), int(height)), getattr(viewer.s, attr)
    if renderer is not None:
        renderer_size = (
            int(getattr(renderer, "_render_capacity_width", renderer.width)),
            int(getattr(renderer, "_render_capacity_height", renderer.height)),
        ) if attr == "training_renderer" else (int(renderer.width), int(renderer.height))
        if renderer_size == size and not force_recreate: return renderer
    previous_renderer = renderer
    renderer = _create_renderer(viewer, size[0], size[1], allow_debug_overlays)
    if isinstance(viewer.s.scene, GaussianScene):
        renderer.set_scene(viewer.s.scene)
    setattr(viewer.s, attr, renderer)
    if attr != "training_renderer":
        _apply_debug_buffers(viewer, renderer)
    if previous_renderer is not None:
        clear_scene_resources = getattr(previous_renderer, "clear_scene_resources", None)
        if callable(clear_scene_resources):
            clear_scene_resources()
        del previous_renderer
    _invalidate(viewer, "debug" if attr == "debug_renderer" else "main", "debug")
    return renderer


def _replace_training_renderer(viewer: object, width: int, height: int, *, reset_loss_debug: bool = True) -> GaussianRenderer:
    previous_renderer = getattr(viewer.s, "training_renderer", None)
    if previous_renderer is None:
        raise RuntimeError("Training renderer is unavailable.")
    renderer = _create_renderer(viewer, int(width), int(height), allow_debug_overlays=False)
    enc = viewer.device.create_command_encoder()
    previous_renderer.copy_scene_state_to(enc, renderer)
    viewer.device.submit_command_buffer(enc.finish())
    viewer.s.training_renderer = renderer
    viewer.s.trainer.rebind_renderer(renderer)
    _apply_debug_buffers(viewer, viewer.s.renderer)
    _apply_debug_buffers(viewer, viewer.s.debug_renderer)
    _invalidate(viewer)
    if reset_loss_debug:
        _reset_loss_debug(viewer)
    clear_scene_resources = getattr(previous_renderer, "clear_scene_resources", None)
    if callable(clear_scene_resources):
        clear_scene_resources()
    return renderer


def _periodic_renderer_reallocation_due(viewer: object, current_time: float) -> bool:
    last_reallocation_time = getattr(viewer.s, "last_periodic_renderer_reallocation_time", None)
    if last_reallocation_time is None:
        return False
    return float(current_time) - float(last_reallocation_time) >= _PERIODIC_RENDERER_REALLOCATION_INTERVAL_S


def maybe_reallocate_renderers(viewer: object, render_width: int, render_height: int, current_time: float) -> bool:
    now = float(current_time)
    if getattr(viewer.s, "last_periodic_renderer_reallocation_time", None) is None:
        viewer.s.last_periodic_renderer_reallocation_time = now
        return False
    if not _periodic_renderer_reallocation_due(viewer, now):
        return False
    recycled = False
    loss_debug_reset = False
    if getattr(viewer.s, "renderer", None) is not None:
        recreate_renderer(viewer, int(render_width), int(render_height))
        recycled = True
        loss_debug_reset = True
    debug_renderer = getattr(viewer.s, "debug_renderer", None)
    if debug_renderer is not None:
        ensure_renderer(viewer, "debug_renderer", int(debug_renderer.width), int(debug_renderer.height), allow_debug_overlays=True, force_recreate=True)
        recycled = True
    if getattr(viewer.s, "trainer", None) is not None and getattr(viewer.s, "training_renderer", None) is not None:
        training_renderer = viewer.s.training_renderer
        _replace_training_renderer(
            viewer,
            int(getattr(training_renderer, "_render_capacity_width", training_renderer.width)),
            int(getattr(training_renderer, "_render_capacity_height", training_renderer.height)),
            reset_loss_debug=not loss_debug_reset,
        )
        recycled = True
    if recycled:
        viewer.s.last_periodic_renderer_reallocation_time = now
    return recycled


def _create_renderer(viewer: object, width: int, height: int, allow_debug_overlays: bool) -> GaussianRenderer:
    renderer = GaussianRenderSettings.from_renderer_params(int(width), int(height), viewer.renderer_params(allow_debug_overlays)).create_renderer(viewer.device)
    if not allow_debug_overlays or getattr(viewer.s, "trainer", None) is not None:
        _, params, _, _ = resolve_effective_training_setup(viewer)
        renderer.max_sh_band = int(getattr(params.training, "max_sh_band", 3))
    return renderer


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
    ensure_renderer(viewer, "renderer", width, height, allow_debug_overlays=True, force_recreate=True)
    _reset_loss_debug(viewer)


def ensure_training_runtime_resolution(viewer: object) -> bool:
    if viewer.s.trainer is None or viewer.s.training_renderer is None or not viewer.s.training_frames:
        return False
    runtime_changed = False
    if bool(getattr(viewer.s, "pending_training_runtime_resize", False)):
        _, params, _, _ = resolve_effective_training_setup(viewer)
        runtime_signature = _training_runtime_signature(params)
        if getattr(viewer.s, "applied_training_runtime_signature", None) != runtime_signature:
            viewer.s.trainer.update_hyperparams(params.adam, params.stability, params.training)
            viewer.s.applied_training_signature = _training_live_params_signature(params)
            viewer.s.applied_training_runtime_signature = runtime_signature
            runtime_changed = True
    current_factor = int(viewer.s.trainer.effective_train_render_factor()) if hasattr(viewer.s.trainer, "effective_train_render_factor") else int(viewer.s.trainer.effective_train_downscale_factor())
    current_size = (
        int(getattr(viewer.s.training_renderer, "_render_capacity_width", viewer.s.training_renderer.width)),
        int(getattr(viewer.s.training_renderer, "_render_capacity_height", viewer.s.training_renderer.height)),
    )
    desired_width, desired_height = viewer.s.trainer.max_training_resolution()
    desired_size = (int(desired_width), int(desired_height))
    if viewer.s.applied_training_runtime_factor == current_factor and current_size == desired_size and not runtime_changed:
        viewer.s.pending_training_runtime_resize = False
        return False
    if current_size == desired_size:
        if runtime_changed:
            _replace_training_renderer(viewer, desired_size[0], desired_size[1])
        viewer.s.applied_training_runtime_factor = current_factor
        viewer.s.pending_training_runtime_resize = False
        return True
    _replace_training_renderer(viewer, desired_size[0], desired_size[1])
    viewer.s.applied_training_runtime_factor = current_factor
    viewer.s.pending_training_runtime_resize = False
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
    if _uses_depth_init(import_cfg):
        positions = getattr(viewer.s, "cached_init_point_positions", None)
        colors = getattr(viewer.s, "cached_init_point_colors", None)
        if positions is None or colors is None:
            raise RuntimeError("Cached pointcloud initializer data is unavailable.")
        sphere_count = _resolve_fibonacci_sphere_count(int(positions.shape[0]), int(getattr(import_cfg, "fibonacci_sphere_point_count", 0)))
        sphere_radius = resolve_colmap_fibonacci_sphere_radius(viewer.s.colmap_recon, _fibonacci_radius_multiplier(import_cfg))
        base_count = int(positions.shape[0]) - sphere_count
        chosen_base_count = base_count if params.training.max_gaussians <= 0 else min(max(int(params.training.max_gaussians), 1), base_count)
        init_positions = positions[:base_count] if sphere_count > 0 and base_count > 0 else positions
        if sphere_count > 0:
            chosen_positions = np.concatenate((positions[:chosen_base_count], positions[base_count:]), axis=0)
            chosen_colors = np.concatenate((colors[:chosen_base_count], colors[base_count:]), axis=0)
        else:
            chosen_positions = positions[:chosen_base_count]
            chosen_colors = colors[:chosen_base_count]
        resolved_init = _pointcloud_init_hparams_from_positions(viewer.s.colmap_recon, init_positions, params.training.max_gaussians, init_hparams, getattr(import_cfg, "nn_radius_scale_coef", 0.5), min_track_length)
        scene = initialize_scene_from_points_colors(chosen_positions, chosen_colors, init.seed, resolved_init)
        return scene, float(max(resolved_init.base_scale, 1e-8))
    source_scenes: list[GaussianScene] = []

    if _import_cfg_enabled(import_cfg, "pointcloud_enabled"):
        positions = getattr(viewer.s, "cached_init_pointcloud_positions", None)
        colors = getattr(viewer.s, "cached_init_pointcloud_colors", None)
        if positions is None or colors is None:
            raise RuntimeError("Cached COLMAP pointcloud initializer data is unavailable.")
        resolved_init = _pointcloud_init_hparams_from_positions(viewer.s.colmap_recon, positions, int(positions.shape[0]), init_hparams, _import_cfg_nn_radius_scale_coef(import_cfg, "pointcloud_nn_radius_scale_coef"), min_track_length)
        source_scenes.append(initialize_scene_from_points_colors(positions, colors, init.seed, resolved_init))

    if _import_cfg_enabled(import_cfg, "diffused_enabled"):
        positions = getattr(viewer.s, "cached_init_diffused_positions", None)
        colors = getattr(viewer.s, "cached_init_diffused_colors", None)
        if positions is None or colors is None:
            raise RuntimeError("Cached diffused initializer data is unavailable.")
        resolved_init = _diffused_pointcloud_init_hparams_from_positions(viewer.s.colmap_recon, positions, init_hparams, _import_cfg_nn_radius_scale_coef(import_cfg, "diffused_nn_radius_scale_coef"), min_track_length)
        source_scenes.append(initialize_scene_from_points_colors(positions, colors, init.seed, resolved_init))

    if _import_cfg_enabled(import_cfg, "custom_ply_enabled"):
        cached_ply_scene = getattr(viewer.s, "cached_init_custom_ply_scene", None)
        if cached_ply_scene is None:
            raise RuntimeError("Cached custom PLY scene is unavailable.")
        source_scenes.append(_copy_gaussian_scene(cached_ply_scene))

    if _import_cfg_enabled(import_cfg, "custom_mesh_enabled"):
        positions = getattr(viewer.s, "cached_init_custom_mesh_positions", None)
        colors = getattr(viewer.s, "cached_init_custom_mesh_colors", None)
        if positions is None or colors is None:
            raise RuntimeError("Cached custom mesh initializer data is unavailable.")
        resolved_init = _sampled_point_init_hparams_from_positions(positions, int(positions.shape[0]), init_hparams, _import_cfg_nn_radius_scale_coef(import_cfg, "custom_mesh_nn_radius_scale_coef"))
        source_scenes.append(initialize_scene_from_points_colors(positions, colors, init.seed, resolved_init))

    if _fibonacci_source_enabled(import_cfg):
        positions = getattr(viewer.s, "cached_init_fibonacci_positions", None)
        colors = getattr(viewer.s, "cached_init_fibonacci_colors", None)
        if positions is None or colors is None:
            raise RuntimeError("Cached Fibonacci sphere initializer data is unavailable.")
        resolved_init = _sampled_point_init_hparams_from_positions(positions, int(positions.shape[0]), init_hparams, _import_cfg_nn_radius_scale_coef(import_cfg, "fibonacci_sphere_nn_radius_scale_coef", default=1.0, fallback_attr=None))
        source_scenes.append(initialize_scene_from_points_colors(positions, colors, init.seed, resolved_init))

    return _concat_gaussian_scenes(source_scenes), None


def _apply_training_image_color_init(viewer: object, trainer: GaussianTrainer, encoder: spy.CommandEncoder) -> None:
    import_cfg = getattr(viewer.s, "colmap_import", None)
    if not bool(getattr(import_cfg, "training_image_color_init", False)):
        return
    frame_textures = list(getattr(trainer, "_frame_targets_native", ()))
    if len(frame_textures) == 0:
        return
    TrainingImageColorInitializer(viewer.device).apply(
        encoder,
        trainer.renderer,
        list(getattr(trainer, "frames", ())),
        frame_textures,
        int(getattr(trainer.scene, "count", 0)),
    )


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
    resolved_params = None
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
        if trainer is not None and attr != "training_renderer":
            if resolved_params is None:
                _, resolved_params, _, _ = resolve_effective_training_setup(viewer)
            renderer.max_sh_band = int(getattr(resolved_params.training, "max_sh_band", getattr(renderer, "max_sh_band", 3)))
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
        if resolved_params is None:
            _, resolved_params, _, _ = resolve_effective_training_setup(viewer)
        params = resolved_params
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


def _histogram_scene_source(viewer: object) -> tuple[GaussianRenderer | None, int, int, object | None, object | None]:
    trainer = getattr(viewer.s, "trainer", None)
    training_renderer = getattr(viewer.s, "training_renderer", None)
    if trainer is not None and training_renderer is not None:
        return training_renderer, int(trainer.scene.count), int(trainer.state.step), getattr(trainer, "metrics", None), trainer
    renderer = getattr(viewer.s, "renderer", None)
    scene = getattr(viewer.s, "scene", None)
    scene_count = int(getattr(scene, "count", 0)) if scene is not None else 0
    return renderer, scene_count, 0, None, None


def refresh_cached_raster_grad_histograms(viewer: object, force: bool = False) -> None:
    refresh_requested = bool(force or viewer.ui._values.get("_histograms_refresh_requested", False))
    renderer, scene_count, step, metrics, trainer = _histogram_scene_source(viewer)
    if renderer is None or scene_count <= 0:
        viewer.s.cached_raster_grad_histograms = None
        viewer.s.cached_raster_grad_ranges = None
        viewer.s.cached_raster_grad_histogram_status = "Histograms require a loaded or initialized scene."
        viewer.ui._values["_histograms_refresh_requested"] = False
        return
    bin_count = max(int(viewer.ui._values.get("hist_bin_count", 64)), 1)
    min_value = float(viewer.ui._values.get("hist_min_value", -1.0))
    max_value = float(viewer.ui._values.get("hist_max_value", 1.0))
    signature = (step, scene_count, bin_count, min_value, max_value)
    if not refresh_requested:
        return
    scene_ranges = renderer.compute_scene_param_ranges(scene_count, metrics=metrics)
    scene_min_values, scene_max_values = _scene_histogram_bounds(scene_ranges, min_value, max_value)
    scene_histograms = renderer.compute_scene_param_histograms(
        scene_count,
        bin_count=bin_count,
        min_value=min_value,
        max_value=max_value,
        param_min_values=scene_min_values,
        param_max_values=scene_max_values,
        metrics=metrics,
    )
    refinement_ranges = None
    refinement_histograms = None
    if trainer is not None:
        compute_refinement_ranges = getattr(trainer, "compute_refinement_distribution_ranges", None)
        refinement_ranges = compute_refinement_ranges(scene_count) if callable(compute_refinement_ranges) else None
        refinement_min_log10, refinement_max_log10 = _param_range_bounds(refinement_ranges, PARAM_HISTOGRAM_SCALE_LOG10, _REFINEMENT_HISTOGRAM_LOG10_FALLBACK_RANGE)
        compute_refinement_histograms = getattr(trainer, "compute_refinement_distribution_histograms", None)
        refinement_histograms = compute_refinement_histograms(scene_count, bin_count=bin_count, min_log10=refinement_min_log10, max_log10=refinement_max_log10) if callable(compute_refinement_histograms) else None
    viewer.s.cached_raster_grad_histograms = _concat_param_histograms(scene_histograms, refinement_histograms)
    viewer.s.cached_raster_grad_ranges = _concat_param_tensor_ranges(scene_ranges, refinement_ranges)
    viewer.s.cached_raster_grad_histogram_mode = ""
    viewer.s.cached_raster_grad_histogram_step = step
    viewer.s.cached_raster_grad_histogram_scene_count = scene_count
    viewer.s.cached_raster_grad_histogram_signature = signature
    total = int(np.sum(viewer.s.cached_raster_grad_histograms.counts))
    viewer.s.cached_raster_grad_histogram_status = (
        (
            f"Splat histograms | step={step:,} | samples={scene_count:,} | populated={total:,}"
            if trainer is not None
            else f"Splat histograms | samples={scene_count:,} | populated={total:,}"
        )
        if total > 0 or step > 0 or trainer is None
        else "No live splat histogram data is available yet."
    )
    viewer.ui._values["_histograms_refresh_requested"] = False


def _concat_param_histograms(*payloads: object) -> object:
    valid = [payload for payload in payloads if payload is not None]
    if len(valid) == 0:
        return None
    if len(valid) == 1:
        return valid[0]
    counts: list[np.ndarray] = []
    row_edges: list[np.ndarray] = []
    labels: list[str] = []
    value_scales: list[str] = []
    groups: list[tuple[str, tuple[int, ...]]] = []
    offset = 0
    bin_edges = np.asarray(getattr(valid[0], "bin_edges_log10", np.zeros((0,), dtype=np.float64)), dtype=np.float64).reshape(-1)
    for payload in valid:
        payload_counts = np.asarray(getattr(payload, "counts", np.zeros((0, 0), dtype=np.int64)), dtype=np.int64)
        payload_edges = np.asarray(getattr(payload, "bin_edges_by_param_log10", np.zeros((0, 0), dtype=np.float64)), dtype=np.float64)
        if payload_edges.ndim != 2 or payload_edges.shape[0] != payload_counts.shape[0]:
            shared_edges = np.asarray(getattr(payload, "bin_edges_log10", np.zeros((0,), dtype=np.float64)), dtype=np.float64).reshape(-1)
            payload_edges = np.repeat(shared_edges[None, :], payload_counts.shape[0], axis=0) if shared_edges.size > 0 else np.zeros((payload_counts.shape[0], 0), dtype=np.float64)
        if payload_counts.ndim != 2 or payload_edges.shape != (payload_counts.shape[0], payload_counts.shape[1] + 1) or payload_counts.shape[1] != max(bin_edges.size - 1, 0):
            continue
        row_count = int(payload_counts.shape[0])
        counts.append(payload_counts)
        row_edges.append(payload_edges)
        payload_labels = tuple(str(label) for label in getattr(payload, "param_labels", ()))
        value_scales.extend(_param_value_scales(payload, row_count))
        labels.extend(payload_labels[index] if index < len(payload_labels) else f"param {offset + index}" for index in range(row_count))
        for group_name, indices in tuple(getattr(payload, "param_groups", ())):
            group_indices = tuple(offset + int(index) for index in indices if 0 <= int(index) < row_count)
            if group_indices:
                groups.append((str(group_name), group_indices))
        offset += row_count
    if len(counts) == 0:
        return None
    return ParamLog10Histograms(
        counts=np.concatenate(counts, axis=0),
        bin_edges_log10=bin_edges.copy(),
        param_labels=tuple(labels),
        param_groups=tuple(groups),
        param_value_scales=tuple(value_scales),
        bin_edges_by_param_log10=np.concatenate(row_edges, axis=0),
    )


def _concat_param_tensor_ranges(*payloads: object) -> object:
    valid = [payload for payload in payloads if payload is not None]
    if len(valid) == 0:
        return None
    if len(valid) == 1:
        return valid[0]
    min_values = []
    max_values = []
    labels: list[str] = []
    value_scales: list[str] = []
    groups: list[tuple[str, tuple[int, ...]]] = []
    offset = 0
    for payload in valid:
        payload_min = np.asarray(getattr(payload, "min_values", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        payload_max = np.asarray(getattr(payload, "max_values", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        if payload_min.size != payload_max.size:
            continue
        min_values.append(payload_min)
        max_values.append(payload_max)
        row_count = int(payload_min.size)
        payload_labels = tuple(str(label) for label in getattr(payload, "param_labels", ()))
        value_scales.extend(_param_value_scales(payload, row_count))
        labels.extend(payload_labels[index] if index < len(payload_labels) else f"param {offset + index}" for index in range(row_count))
        for group_name, indices in tuple(getattr(payload, "param_groups", ())):
            group_indices = tuple(offset + int(index) for index in indices if 0 <= int(index) < row_count)
            if group_indices:
                groups.append((str(group_name), group_indices))
        offset += row_count
    if len(min_values) == 0:
        return None
    return ParamTensorRanges(
        min_values=np.concatenate(min_values, axis=0),
        max_values=np.concatenate(max_values, axis=0),
        param_labels=tuple(labels),
        param_groups=tuple(groups),
        param_value_scales=tuple(value_scales),
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
    reset_main_camera(viewer)
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
    auto_rotate_scene: bool = True,
    compress_dataset_using_bc7: bool = False,
    training_image_color_init: bool = False,
    custom_ply_path: Path | None,
    image_downscale_mode: str,
    image_downscale_max_size: int,
    image_downscale_scale: float,
    nn_radius_scale_coef: float,
    min_track_length: int = DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH,
    depth_point_count: int = 100000,
    diffused_point_count: int = 100000,
    fibonacci_sphere_point_count: int = 0,
    fibonacci_sphere_radius_multiplier: float = 2.0,
    fibonacci_sphere_color: tuple[float, float, float] = tuple(float(v) for v in FIBONACCI_SPHERE_COLOR),
    target_alpha_mode: int | None = None,
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
        auto_rotate_scene=auto_rotate_scene,
        compress_dataset_using_bc7=compress_dataset_using_bc7,
        training_image_color_init=training_image_color_init,
        custom_ply_path=custom_ply_path,
        image_downscale_mode=image_downscale_mode,
        image_downscale_max_size=image_downscale_max_size,
        image_downscale_scale=image_downscale_scale,
        nn_radius_scale_coef=nn_radius_scale_coef,
        min_track_length=min_track_length,
        depth_point_count=depth_point_count,
        diffused_point_count=diffused_point_count,
        fibonacci_sphere_point_count=fibonacci_sphere_point_count,
        fibonacci_sphere_radius_multiplier=fibonacci_sphere_radius_multiplier,
        fibonacci_sphere_color=fibonacci_sphere_color,
        target_alpha_mode=target_alpha_mode,
        use_target_alpha_mask=use_target_alpha_mask,
        pointcloud_enabled=pointcloud_enabled,
        pointcloud_nn_radius_scale_coef=pointcloud_nn_radius_scale_coef,
        diffused_enabled=diffused_enabled,
        diffused_diffusion_radius=diffused_diffusion_radius,
        diffused_nn_radius_scale_coef=diffused_nn_radius_scale_coef,
        custom_ply_enabled=custom_ply_enabled,
        custom_ply_nn_radius_scale_coef=custom_ply_nn_radius_scale_coef,
        custom_mesh_enabled=custom_mesh_enabled,
        custom_mesh_path=custom_mesh_path,
        custom_mesh_point_count=custom_mesh_point_count,
        custom_mesh_nn_radius_scale_coef=custom_mesh_nn_radius_scale_coef,
        fibonacci_sphere_enabled=fibonacci_sphere_enabled,
        fibonacci_sphere_nn_radius_scale_coef=fibonacci_sphere_nn_radius_scale_coef,
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
            fibonacci_sphere_radius_multiplier,
            fibonacci_sphere_color,
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
    reset_main_camera(viewer)
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
    auto_rotate_scene: bool = True,
    custom_ply_path: Path | None,
    image_downscale_mode: str,
    image_downscale_max_size: int,
    image_downscale_scale: float,
    nn_radius_scale_coef: float,
    min_track_length: int = DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH,
    depth_point_count: int = 100000,
    diffused_point_count: int = 100000,
    fibonacci_sphere_point_count: int = 0,
    fibonacci_sphere_radius_multiplier: float = 2.0,
    fibonacci_sphere_color: tuple[float, float, float] = tuple(float(v) for v in FIBONACCI_SPHERE_COLOR),
    target_alpha_mode: int | None = None,
    use_target_alpha_mask: bool = False,
    compress_dataset_using_bc7: bool = False,
    training_image_color_init: bool = False,
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
    _clear_loaded_scene(viewer)
    root = Path(colmap_root).resolve()
    recon = _load_aligned_colmap_reconstruction(root, auto_rotate_scene=auto_rotate_scene)
    viewer.s.colmap_import = ColmapImportSettings(
        images_root=Path(images_root).resolve(),
        auto_rotate_scene=bool(auto_rotate_scene),
        compress_dataset_using_bc7=bool(compress_dataset_using_bc7),
        training_image_color_init=bool(training_image_color_init),
        nn_radius_scale_coef=float(nn_radius_scale_coef),
        pointcloud_enabled=bool(pointcloud_enabled),
        pointcloud_nn_radius_scale_coef=float(max(pointcloud_nn_radius_scale_coef if pointcloud_nn_radius_scale_coef is not None else nn_radius_scale_coef, 1e-4)),
        diffused_enabled=bool(diffused_enabled),
        diffused_diffusion_radius=max(float(diffused_diffusion_radius), 0.0),
        diffused_nn_radius_scale_coef=float(max(diffused_nn_radius_scale_coef if diffused_nn_radius_scale_coef is not None else nn_radius_scale_coef, 1e-4)),
        custom_ply_enabled=bool(custom_ply_enabled),
        custom_ply_nn_radius_scale_coef=float(max(custom_ply_nn_radius_scale_coef if custom_ply_nn_radius_scale_coef is not None else 1.0, 1e-4)),
        custom_mesh_enabled=bool(custom_mesh_enabled),
        custom_mesh_path=None if custom_mesh_path is None else Path(custom_mesh_path).resolve(),
        custom_mesh_point_count=max(int(custom_mesh_point_count if custom_mesh_point_count is not None else diffused_point_count), 1),
        custom_mesh_nn_radius_scale_coef=float(max(custom_mesh_nn_radius_scale_coef if custom_mesh_nn_radius_scale_coef is not None else nn_radius_scale_coef, 1e-4)),
        fibonacci_sphere_enabled=bool(fibonacci_sphere_enabled),
        fibonacci_sphere_point_count=max(int(fibonacci_sphere_point_count), 0),
        fibonacci_sphere_radius_multiplier=max(float(fibonacci_sphere_radius_multiplier), 0.0),
        fibonacci_sphere_color=tuple(float(v) for v in np.clip(np.asarray(fibonacci_sphere_color, dtype=np.float32).reshape(3), 0.0, 1.0)),
        target_alpha_mode=resolve_target_alpha_mode(target_alpha_mode, legacy_use_target_alpha_mask=use_target_alpha_mask),
        fibonacci_sphere_nn_radius_scale_coef=float(max(fibonacci_sphere_nn_radius_scale_coef if fibonacci_sphere_nn_radius_scale_coef is not None else 1.0, 1e-4)),
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
        auto_rotate_scene=auto_rotate_scene,
        compress_dataset_using_bc7=compress_dataset_using_bc7,
        training_image_color_init=training_image_color_init,
        custom_ply_path=custom_ply_path,
        image_downscale_mode=image_downscale_mode,
        image_downscale_max_size=image_downscale_max_size,
        image_downscale_scale=image_downscale_scale,
        nn_radius_scale_coef=nn_radius_scale_coef,
        min_track_length=min_track_length,
        depth_point_count=depth_point_count,
        diffused_point_count=diffused_point_count,
        fibonacci_sphere_point_count=fibonacci_sphere_point_count,
        fibonacci_sphere_radius_multiplier=fibonacci_sphere_radius_multiplier,
        target_alpha_mode=target_alpha_mode,
        use_target_alpha_mask=use_target_alpha_mask,
        pointcloud_enabled=pointcloud_enabled,
        pointcloud_nn_radius_scale_coef=pointcloud_nn_radius_scale_coef,
        diffused_enabled=diffused_enabled,
        diffused_diffusion_radius=diffused_diffusion_radius,
        diffused_nn_radius_scale_coef=diffused_nn_radius_scale_coef,
        custom_ply_enabled=custom_ply_enabled,
        custom_ply_nn_radius_scale_coef=custom_ply_nn_radius_scale_coef,
        custom_mesh_enabled=custom_mesh_enabled,
        custom_mesh_path=custom_mesh_path,
        custom_mesh_point_count=custom_mesh_point_count,
        custom_mesh_nn_radius_scale_coef=custom_mesh_nn_radius_scale_coef,
        fibonacci_sphere_enabled=fibonacci_sphere_enabled,
        fibonacci_sphere_nn_radius_scale_coef=fibonacci_sphere_nn_radius_scale_coef,
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
    custom_mesh_text = _ui_path_string(viewer, "colmap_custom_mesh_path")
    custom_mesh_path = None if not custom_mesh_text else Path(custom_mesh_text).expanduser()
    image_downscale_mode = _ui_image_downscale_mode(viewer)
    selected_camera_ids = tuple(int(camera_id) for camera_id in viewer.ui._values.get("colmap_selected_camera_ids", ()))
    camera_rows = tuple(viewer.ui._values.get("_colmap_camera_rows", ()))
    image_downscale_max_size = max(int(viewer.ui._values.get("colmap_image_max_size", 2048)), 1)
    image_downscale_scale = float(np.clip(viewer.ui._values.get("colmap_image_scale", 1.0), 1e-6, 1.0))
    nn_radius_scale_coef = float(viewer.ui._values.get("colmap_nn_radius_scale_coef", 0.5))
    min_track_length = max(int(viewer.ui._values.get("colmap_min_track_length", DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH)), 0)
    depth_point_count = max(int(viewer.ui._values.get("colmap_depth_point_count", 100000)), 1)
    diffused_point_count = max(int(viewer.ui._values.get("colmap_diffused_point_count", 100000)), 1)
    fibonacci_sphere_point_count = max(int(viewer.ui._values.get("colmap_fibonacci_sphere_point_count", 0)), 0)
    fibonacci_sphere_radius_multiplier = max(float(viewer.ui._values.get("colmap_fibonacci_sphere_radius_multiplier", viewer.ui._values.get("colmap_fibonacci_sphere_radius", 2.0))), 0.0)
    fibonacci_sphere_color = tuple(float(v) for v in np.clip(np.asarray(viewer.ui._values.get("colmap_fibonacci_sphere_color", FIBONACCI_SPHERE_COLOR), dtype=np.float32).reshape(3), 0.0, 1.0))
    pointcloud_enabled = bool(viewer.ui._values.get("colmap_pointcloud_enabled", False))
    pointcloud_nn_radius_scale_coef = float(viewer.ui._values.get("colmap_pointcloud_nn_radius_scale_coef", nn_radius_scale_coef))
    diffused_enabled = bool(viewer.ui._values.get("colmap_diffused_enabled", False))
    diffused_diffusion_radius = max(float(viewer.ui._values.get("colmap_diffused_diffusion_radius", 1.0)), 0.0)
    diffused_nn_radius_scale_coef = float(viewer.ui._values.get("colmap_diffused_nn_radius_scale_coef", nn_radius_scale_coef))
    custom_ply_enabled = bool(viewer.ui._values.get("colmap_custom_ply_enabled", False))
    custom_ply_nn_radius_scale_coef = float(viewer.ui._values.get("colmap_custom_ply_nn_radius_scale_coef", 1.0))
    custom_mesh_enabled = bool(viewer.ui._values.get("colmap_custom_mesh_enabled", False))
    custom_mesh_point_count = max(int(viewer.ui._values.get("colmap_custom_mesh_point_count", diffused_point_count)), 1)
    custom_mesh_nn_radius_scale_coef = float(viewer.ui._values.get("colmap_custom_mesh_nn_radius_scale_coef", nn_radius_scale_coef))
    fibonacci_sphere_enabled = bool(viewer.ui._values.get("colmap_fibonacci_sphere_enabled", fibonacci_sphere_point_count > 0))
    fibonacci_sphere_nn_radius_scale_coef = float(viewer.ui._values.get("colmap_fibonacci_sphere_nn_radius_scale_coef", 1.0))
    auto_rotate_scene = bool(viewer.ui._values.get("colmap_auto_rotate_scene", True))
    target_alpha_mode = resolve_target_alpha_mode(viewer.ui._values.get("target_alpha_mode", None), legacy_use_target_alpha_mask=bool(viewer.ui._values.get("use_target_alpha_mask", False)))
    compress_dataset_using_bc7 = bool(viewer.ui._values.get("compress_dataset_using_bc7", False))
    training_image_color_init = bool(viewer.ui._values.get("colmap_training_image_color_init", False))
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
    if init_mode != _COLMAP_IMPORT_DEPTH:
        if custom_ply_enabled and (custom_ply_path is None or not custom_ply_path.exists()):
            raise FileNotFoundError(f"Custom PLY does not exist: {custom_ply_path}")
        if custom_mesh_enabled and (custom_mesh_path is None or not custom_mesh_path.exists()):
            raise FileNotFoundError(f"Custom mesh does not exist: {custom_mesh_path}")
        if not any((pointcloud_enabled, diffused_enabled, custom_ply_enabled, custom_mesh_enabled, fibonacci_sphere_enabled)):
            raise ValueError("Enable at least one initialization source before importing.")
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
        auto_rotate_scene=auto_rotate_scene,
        custom_ply_path=None if custom_ply_path is None else custom_ply_path.resolve(),
        compress_dataset_using_bc7=compress_dataset_using_bc7,
        training_image_color_init=training_image_color_init,
        image_downscale_mode=image_downscale_mode,
        image_downscale_max_size=image_downscale_max_size,
        image_downscale_scale=image_downscale_scale,
        nn_radius_scale_coef=float(max(nn_radius_scale_coef, 1e-4)),
        min_track_length=min_track_length,
        depth_point_count=depth_point_count,
        diffused_point_count=diffused_point_count,
        fibonacci_sphere_point_count=fibonacci_sphere_point_count,
        fibonacci_sphere_radius_multiplier=fibonacci_sphere_radius_multiplier,
        fibonacci_sphere_color=fibonacci_sphere_color,
        target_alpha_mode=target_alpha_mode,
        pointcloud_enabled=pointcloud_enabled,
        pointcloud_nn_radius_scale_coef=float(max(pointcloud_nn_radius_scale_coef, 1e-4)),
        diffused_enabled=diffused_enabled,
        diffused_diffusion_radius=diffused_diffusion_radius,
        diffused_nn_radius_scale_coef=float(max(diffused_nn_radius_scale_coef, 1e-4)),
        custom_ply_enabled=custom_ply_enabled,
        custom_ply_nn_radius_scale_coef=float(max(custom_ply_nn_radius_scale_coef, 1e-4)),
        custom_mesh_enabled=custom_mesh_enabled,
        custom_mesh_path=None if custom_mesh_path is None else custom_mesh_path.resolve(),
        custom_mesh_point_count=custom_mesh_point_count,
        custom_mesh_nn_radius_scale_coef=float(max(custom_mesh_nn_radius_scale_coef, 1e-4)),
        fibonacci_sphere_enabled=fibonacci_sphere_enabled,
        fibonacci_sphere_nn_radius_scale_coef=float(max(fibonacci_sphere_nn_radius_scale_coef, 1e-4)),
    )
    viewer.s.last_error = ""


def advance_colmap_import(viewer: object) -> None:
    progress = getattr(viewer.s, "colmap_import_progress", None)
    if progress is None:
        return
    try:
        if progress.phase == "prepare":
            progress.recon = _load_aligned_colmap_reconstruction(progress.colmap_root, auto_rotate_scene=progress.auto_rotate_scene)
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
                auto_rotate_scene=progress.auto_rotate_scene,
                compress_dataset_using_bc7=progress.compress_dataset_using_bc7,
                training_image_color_init=progress.training_image_color_init,
                custom_ply_path=progress.custom_ply_path,
                image_downscale_mode=progress.image_downscale_mode,
                image_downscale_max_size=progress.image_downscale_max_size,
                image_downscale_scale=progress.image_downscale_scale,
                nn_radius_scale_coef=progress.nn_radius_scale_coef,
                min_track_length=progress.min_track_length,
                depth_point_count=progress.depth_point_count,
                diffused_point_count=progress.diffused_point_count,
                fibonacci_sphere_point_count=progress.fibonacci_sphere_point_count,
                fibonacci_sphere_radius_multiplier=progress.fibonacci_sphere_radius_multiplier,
                fibonacci_sphere_color=progress.fibonacci_sphere_color,
                target_alpha_mode=progress.target_alpha_mode,
                pointcloud_enabled=getattr(progress, "pointcloud_enabled", None),
                pointcloud_nn_radius_scale_coef=getattr(progress, "pointcloud_nn_radius_scale_coef", None),
                diffused_enabled=getattr(progress, "diffused_enabled", None),
                diffused_diffusion_radius=getattr(progress, "diffused_diffusion_radius", None),
                diffused_nn_radius_scale_coef=getattr(progress, "diffused_nn_radius_scale_coef", None),
                custom_ply_enabled=getattr(progress, "custom_ply_enabled", None),
                custom_ply_nn_radius_scale_coef=getattr(progress, "custom_ply_nn_radius_scale_coef", None),
                custom_mesh_enabled=getattr(progress, "custom_mesh_enabled", None),
                custom_mesh_path=getattr(progress, "custom_mesh_path", None),
                custom_mesh_point_count=getattr(progress, "custom_mesh_point_count", None),
                custom_mesh_nn_radius_scale_coef=getattr(progress, "custom_mesh_nn_radius_scale_coef", None),
                fibonacci_sphere_enabled=getattr(progress, "fibonacci_sphere_enabled", None),
                fibonacci_sphere_nn_radius_scale_coef=getattr(progress, "fibonacci_sphere_nn_radius_scale_coef", None),
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
    _reset_training_runtime(viewer, preserve_frame_targets=frame_targets_native is not None)
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
    sync_photometric_target_provider(viewer)
    capped_main_sh = int(getattr(params.training, "max_sh_band", 3))
    if getattr(viewer.s, "renderer", None) is not None:
        viewer.s.renderer.max_sh_band = capped_main_sh
    if getattr(viewer.s, "debug_renderer", None) is not None:
        viewer.s.debug_renderer.max_sh_band = capped_main_sh
    viewer.s.scene = SceneCountProxy(scene.count)
    reset_main_camera(viewer)
    enc = viewer.device.create_command_encoder()
    _apply_training_image_color_init(viewer, viewer.s.trainer, enc)
    renderer.copy_scene_state_to(enc, viewer.s.renderer, include_work_buffers=False)
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


def photometric_elapsed_seconds(viewer: object, now: float | None = None) -> float:
    elapsed = float(getattr(viewer.s, "photometric_elapsed_s", 0.0))
    resume_time = getattr(viewer.s, "photometric_resume_time", None)
    if bool(getattr(viewer.s, "photometric_active", False)) and resume_time is not None:
        current_time = float(time.perf_counter() if now is None else now)
        elapsed += max(current_time - float(resume_time), 0.0)
    return elapsed


def sync_photometric_target_provider(viewer: object) -> None:
    trainer = getattr(viewer.s, "trainer", None)
    if trainer is None:
        return
    photometric_trainer = getattr(viewer.s, "photometric_trainer", None)
    apply_to_targets = bool(getattr(getattr(viewer, "ui", None), "_values", {}).get("photometric_apply_to_targets", True))
    provider = photometric_trainer.provider if photometric_trainer is not None and apply_to_targets else None
    if getattr(trainer, "target_tonemap_provider", None) is provider:
        return
    trainer.set_target_tonemap_provider(provider)


def _photometric_frame_source_textures(viewer: object) -> list[spy.Texture] | None:
    trainer = getattr(viewer.s, "trainer", None)
    frames = tuple(getattr(viewer.s, "training_frames", ()))
    native_targets = getattr(trainer, "_frame_targets_native", None)
    if trainer is None or not isinstance(native_targets, list) or not frames:
        return None
    if len(native_targets) != len(frames) or any(texture is None for texture in native_targets):
        return None
    return list(native_targets)


def _photometric_hparams(viewer: object) -> PhotometricCompensationHyperParams:
    values = getattr(getattr(viewer, "ui", None), "_values", {})
    defaults = PhotometricCompensationHyperParams()
    return PhotometricCompensationHyperParams(
        batch_pair_count=int(values.get("photometric_batch_pair_count", defaults.batch_pair_count)),
        neighborhood_size=int(values.get("photometric_neighborhood_size", defaults.neighborhood_size)),
        min_track_length=int(values.get("photometric_min_track_length", defaults.min_track_length)),
        learning_rate=float(values.get("photometric_learning_rate", defaults.learning_rate)),
        exposure_lr_mul=float(values.get("photometric_exposure_lr_mul", defaults.exposure_lr_mul)),
        vignette_lr_mul=float(values.get("photometric_vignette_lr_mul", defaults.vignette_lr_mul)),
        chroma_lr_mul=float(values.get("photometric_chroma_lr_mul", defaults.chroma_lr_mul)),
        crf_lr_mul=float(values.get("photometric_crf_lr_mul", defaults.crf_lr_mul)),
        exposure_regularize_weight=float(values.get("photometric_exposure_regularize_weight", defaults.exposure_regularize_weight)),
        vignette_regularize_weight=float(values.get("photometric_vignette_regularize_weight", defaults.vignette_regularize_weight)),
        chroma_regularize_weight=float(values.get("photometric_chroma_regularize_weight", defaults.chroma_regularize_weight)),
        crf_regularize_weight=float(values.get("photometric_crf_regularize_weight", defaults.crf_regularize_weight)),
        exposure_l1_weight=float(values.get("photometric_exposure_l1_weight", defaults.exposure_l1_weight)),
        vignette_l1_weight=float(values.get("photometric_vignette_l1_weight", defaults.vignette_l1_weight)),
        chroma_l1_weight=float(values.get("photometric_chroma_l1_weight", defaults.chroma_l1_weight)),
        crf_l1_weight=float(values.get("photometric_crf_l1_weight", defaults.crf_l1_weight)),
        grad_component_clip=float(values.get("photometric_grad_component_clip", defaults.grad_component_clip)),
        grad_norm_clip=float(values.get("photometric_grad_norm_clip", defaults.grad_norm_clip)),
        max_update=float(values.get("photometric_max_update", defaults.max_update)),
    )


def sync_photometric_hparams(viewer: object) -> None:
    trainer = getattr(viewer.s, "photometric_trainer", None)
    if trainer is None:
        return
    values = getattr(getattr(viewer, "ui", None), "_values", {})
    trainer.hparams = replace(
        trainer.hparams,
        batch_pair_count=int(values.get("photometric_batch_pair_count", trainer.hparams.batch_pair_count)),
        learning_rate=float(values.get("photometric_learning_rate", trainer.hparams.learning_rate)),
        exposure_lr_mul=float(values.get("photometric_exposure_lr_mul", trainer.hparams.exposure_lr_mul)),
        vignette_lr_mul=float(values.get("photometric_vignette_lr_mul", trainer.hparams.vignette_lr_mul)),
        chroma_lr_mul=float(values.get("photometric_chroma_lr_mul", trainer.hparams.chroma_lr_mul)),
        crf_lr_mul=float(values.get("photometric_crf_lr_mul", trainer.hparams.crf_lr_mul)),
        exposure_regularize_weight=float(values.get("photometric_exposure_regularize_weight", trainer.hparams.exposure_regularize_weight)),
        vignette_regularize_weight=float(values.get("photometric_vignette_regularize_weight", trainer.hparams.vignette_regularize_weight)),
        chroma_regularize_weight=float(values.get("photometric_chroma_regularize_weight", trainer.hparams.chroma_regularize_weight)),
        crf_regularize_weight=float(values.get("photometric_crf_regularize_weight", trainer.hparams.crf_regularize_weight)),
        exposure_l1_weight=float(values.get("photometric_exposure_l1_weight", trainer.hparams.exposure_l1_weight)),
        vignette_l1_weight=float(values.get("photometric_vignette_l1_weight", trainer.hparams.vignette_l1_weight)),
        chroma_l1_weight=float(values.get("photometric_chroma_l1_weight", trainer.hparams.chroma_l1_weight)),
        crf_l1_weight=float(values.get("photometric_crf_l1_weight", trainer.hparams.crf_l1_weight)),
        grad_component_clip=float(values.get("photometric_grad_component_clip", trainer.hparams.grad_component_clip)),
        grad_norm_clip=float(values.get("photometric_grad_norm_clip", trainer.hparams.grad_norm_clip)),
        max_update=float(values.get("photometric_max_update", trainer.hparams.max_update)),
    )


def _photometric_pair_dataset_ready(trainer: object | None) -> bool:
    if trainer is None:
        return False
    return bool(getattr(trainer, "_pair_dataset_uploaded", True))


def _begin_photometric_pair_dataset_prepare(trainer: object | None) -> None:
    if trainer is None:
        return
    begin_prepare_pair_dataset = getattr(trainer, "begin_prepare_pair_dataset", None)
    if callable(begin_prepare_pair_dataset):
        begin_prepare_pair_dataset()
        return
    prepare_pair_dataset = getattr(trainer, "prepare_pair_dataset", None)
    if callable(prepare_pair_dataset):
        prepare_pair_dataset()


def _advance_photometric_pair_dataset_prepare(trainer: object | None, *, frame_budget: int = 1) -> bool:
    if trainer is None:
        return False
    advance_prepare_pair_dataset = getattr(trainer, "advance_prepare_pair_dataset", None)
    if callable(advance_prepare_pair_dataset):
        return bool(advance_prepare_pair_dataset(frame_budget=frame_budget))
    if not _photometric_pair_dataset_ready(trainer):
        prepare_pair_dataset = getattr(trainer, "prepare_pair_dataset", None)
        if callable(prepare_pair_dataset):
            prepare_pair_dataset()
    return _photometric_pair_dataset_ready(trainer)


def initialize_photometric_compensation(viewer: object, *, activate_when_ready: bool = False) -> None:
    if not viewer.s.training_frames:
        _refresh_training_frames(viewer)
    if viewer.s.colmap_recon is None or not viewer.s.training_frames:
        return
    reset_photometric_compensation(viewer, clear_history=False)
    viewer.s.photometric_trainer = PhotometricCompensationTrainer(
        device=viewer.device,
        reconstruction=viewer.s.colmap_recon,
        frames=viewer.s.training_frames,
        hparams=_photometric_hparams(viewer),
        frame_source_textures=_photometric_frame_source_textures(viewer),
    )
    viewer.s.photometric_prepare_pending_active = bool(activate_when_ready)
    _begin_photometric_pair_dataset_prepare(viewer.s.photometric_trainer)
    viewer.s.photometric_active = False
    viewer.s.photometric_elapsed_s = 0.0
    viewer.s.photometric_resume_time = None
    sync_photometric_target_provider(viewer)
    if activate_when_ready and _photometric_pair_dataset_ready(viewer.s.photometric_trainer):
        set_photometric_active(viewer, True)


def advance_photometric_initialization(viewer: object) -> None:
    trainer = getattr(viewer.s, "photometric_trainer", None)
    if trainer is None or _photometric_pair_dataset_ready(trainer):
        return
    if not _advance_photometric_pair_dataset_prepare(trainer, frame_budget=1):
        return
    if bool(getattr(viewer.s, "photometric_prepare_pending_active", False)):
        set_photometric_active(viewer, True)


def reset_photometric_compensation(viewer: object, *, clear_history: bool = True) -> None:
    now = float(time.perf_counter())
    photometric_active = bool(getattr(viewer.s, "photometric_active", False))
    photometric_resume_time = getattr(viewer.s, "photometric_resume_time", None)
    if photometric_active and photometric_resume_time is not None:
        viewer.s.photometric_elapsed_s = float(getattr(viewer.s, "photometric_elapsed_s", 0.0)) + max(now - float(photometric_resume_time), 0.0)
    viewer.s.photometric_active = False
    viewer.s.photometric_prepare_pending_active = False
    viewer.s.photometric_resume_time = None
    trainer = getattr(viewer.s, "photometric_trainer", None)
    release_resources = getattr(trainer, "release_resources", None)
    if callable(release_resources):
        release_resources()
    viewer.s.photometric_trainer = None
    viewer.s.photometric_elapsed_s = 0.0
    sync_photometric_target_provider(viewer)
    if clear_history and hasattr(viewer, "toolkit") and hasattr(getattr(viewer, "toolkit", None), "tk"):
        viewer.toolkit.tk.clear_photometric_plot_history()


def set_photometric_active(viewer: object, active: bool) -> None:
    if active and getattr(viewer.s, "photometric_trainer", None) is None:
        initialize_photometric_compensation(viewer, activate_when_ready=True)
    now = float(time.perf_counter())
    photometric_resume_time = getattr(viewer.s, "photometric_resume_time", None)
    if bool(getattr(viewer.s, "photometric_active", False)) and photometric_resume_time is not None:
        viewer.s.photometric_elapsed_s = float(getattr(viewer.s, "photometric_elapsed_s", 0.0)) + max(now - float(photometric_resume_time), 0.0)
        viewer.s.photometric_resume_time = None
    trainer = getattr(viewer.s, "photometric_trainer", None)
    pair_dataset_ready = _photometric_pair_dataset_ready(trainer)
    viewer.s.photometric_prepare_pending_active = bool(active and trainer is not None and not pair_dataset_ready)
    viewer.s.photometric_active = bool(active and trainer is not None and pair_dataset_ready)
    if viewer.s.photometric_active:
        viewer.s.photometric_resume_time = now


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
