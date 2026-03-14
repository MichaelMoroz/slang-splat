from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sqlite3
import time

import numpy as np
import slangpy as spy

from ..app.shared import apply_training_profile, estimate_point_bounds, estimate_scene_bounds, renderer_kwargs
from ..common import SHADER_ROOT, clamp_index
from ..renderer import GaussianRenderer
from ..scene import GaussianScene, build_training_frames_from_root, initialize_scene_from_colmap_points, load_colmap_reconstruction, load_gaussian_ply, resolve_colmap_init_hparams
from ..scene._internal.colmap_ops import point_nn_scales
from ..training import GaussianTrainer
from ..scene._internal.colmap_types import point_tables
from .state import ColmapImportSettings, SceneCountProxy

_COLMAP_IMPORT_POINTCLOUD = "pointcloud"
_COLMAP_IMPORT_CUSTOM_PLY = "custom_ply"
_COLMAP_DB_SAMPLE_LIMIT = 64
_COLMAP_DB_SEARCH_PATTERNS = ("database.db", "*.db", "*.sqlite", "*.sqlite3")


def _clear(viewer: object, *attrs: str) -> None:
    for attr in attrs:
        value = getattr(viewer.s, attr)
        if value is not None:
            setattr(viewer.s, attr, None)
            del value


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
    return _COLMAP_IMPORT_CUSTOM_PLY if int(viewer.ui._values.get("colmap_init_mode", 0)) == 1 else _COLMAP_IMPORT_POINTCLOUD


def _profile_images_subdir(viewer: object) -> str | None:
    images_root = viewer.s.colmap_import.images_root
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


def _suggest_images_root_from_dataset_root(dataset_root: Path, image_names: list[str]) -> Path:
    root = Path(dataset_root).resolve()
    sample_names = image_names[: min(len(image_names), _COLMAP_DB_SAMPLE_LIMIT)]
    for candidate in _dataset_directories(root):
        if any((candidate / image_name).exists() for image_name in sample_names):
            return candidate
    raise FileNotFoundError(f"Could not find an image folder under {root} for COLMAP images: {sample_names[:4]}")


def _update_import_settings(
    viewer: object,
    *,
    dataset_root: Path,
    database_path: Path | None,
    images_root: Path,
    init_mode: str,
    custom_ply_path: Path | None,
    nn_radius_scale_coef: float,
) -> None:
    viewer.s.colmap_import = ColmapImportSettings(
        database_path=None if database_path is None else Path(database_path).resolve(),
        images_root=Path(images_root).resolve(),
        init_mode=str(init_mode),
        custom_ply_path=None if custom_ply_path is None else Path(custom_ply_path).resolve(),
        nn_radius_scale_coef=float(max(nn_radius_scale_coef, 1e-4)),
    )
    _set_ui_path(viewer, "colmap_root_path", dataset_root)
    _set_ui_path(viewer, "colmap_database_path", database_path)
    _set_ui_path(viewer, "colmap_images_root", images_root)
    viewer.ui._values["colmap_init_mode"] = 1 if str(init_mode) == _COLMAP_IMPORT_CUSTOM_PLY else 0
    _set_ui_path(viewer, "colmap_custom_ply_path", custom_ply_path)
    viewer.ui._values["colmap_nn_radius_scale_coef"] = float(max(nn_radius_scale_coef, 1e-4))


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


def choose_colmap_custom_ply(viewer: object, ply_path: Path) -> None:
    _set_ui_path(viewer, "colmap_custom_ply_path", Path(ply_path).resolve())
    viewer.s.last_error = ""


def _pointcloud_init_hparams(recon: object, max_gaussians: int, init_hparams: object, nn_radius_scale_coef: float):
    resolved = resolve_colmap_init_hparams(recon, max_gaussians, init_hparams)
    xyz, _ = _point_tables(recon)
    chosen_count = xyz.shape[0] if max_gaussians <= 0 else min(max(int(max_gaussians), 1), xyz.shape[0])
    median_nn_scale = float(np.median(point_nn_scales(np.ascontiguousarray(xyz[:chosen_count], dtype=np.float32)))) if chosen_count > 0 else 1.0
    return replace(resolved, base_scale=float(max(float(nn_radius_scale_coef), 1e-4) * max(median_nn_scale, 1e-6)))


def _invalidate(viewer: object, *targets: str) -> None:
    for target in targets or ("main", "debug"):
        setattr(viewer.s, f"synced_step_{target}", -1)


def _reset_loss_debug(viewer: object) -> None:
    viewer.s.loss_debug_texture = None
    viewer.s.debug_present_texture = None
    _clear(viewer, "debug_renderer")
    _invalidate(viewer, "debug")


def _reset_loaded_runtime(viewer: object) -> None:
    viewer.s.scene_init_signature = None
    viewer.s.training_active = False
    viewer.s.training_elapsed_s = 0.0
    viewer.s.training_resume_time = None
    viewer.s.trainer = None
    if viewer.s.renderer is not None:
        viewer.s.renderer.set_debug_grad_norm_buffer(None)
    viewer.s.colmap_point_positions_buffer = viewer.s.colmap_point_colors_buffer = None
    viewer.s.colmap_point_count = 0
    viewer.s.suggested_init_hparams = viewer.s.suggested_init_count = None
    update_debug_frame_slider_range(viewer)
    _reset_loss_debug(viewer)
    _clear(viewer, "training_renderer")


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
        None if import_cfg.custom_ply_path is None else str(import_cfg.custom_ply_path.resolve()),
        round(float(import_cfg.nn_radius_scale_coef), 6),
        None if init.hparams.initial_opacity is None else round(float(init.hparams.initial_opacity), 8),
    )


def create_debug_shaders(viewer: object) -> None:
    shader_path = str(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang")
    viewer.s.debug_abs_diff_kernel = viewer.device.create_compute_kernel(viewer.device.load_program(shader_path, ["csComposeAbsDiffDebug"]))
    viewer.s.debug_letterbox_kernel = viewer.device.create_compute_kernel(viewer.device.load_program(shader_path, ["csComposeLetterboxDebug"]))


def update_debug_frame_slider_range(viewer: object) -> None:
    slider = viewer.c("loss_debug_frame")
    max_index = max(len(viewer.s.training_frames) - 1, 0)
    if hasattr(slider, "min"):
        slider.min = 0
        slider.max = int(max_index)
    slider.value = clamp_index(int(slider.value), max_index + 1)


def ensure_renderer(viewer: object, attr: str, width: int, height: int, allow_debug_overlays: bool) -> GaussianRenderer:
    size, renderer = (int(width), int(height)), getattr(viewer.s, attr)
    if renderer is not None and (renderer.width, renderer.height) == size:
        return renderer
    _clear(viewer, attr)
    renderer = GaussianRenderer(viewer.device, width=size[0], height=size[1], **renderer_kwargs(viewer.renderer_params(allow_debug_overlays)))
    if isinstance(viewer.s.scene, GaussianScene):
        renderer.set_scene(viewer.s.scene)
    setattr(viewer.s, attr, renderer)
    _invalidate(viewer, "debug" if attr == "debug_renderer" else "main", "debug")
    return renderer


def recreate_renderer(viewer: object, width: int, height: int) -> None:
    ensure_renderer(viewer, "renderer", width, height, allow_debug_overlays=True)
    _reset_loss_debug(viewer)


def resolve_effective_training_setup(viewer: object):
    init = viewer.init_params()
    params, profile = apply_training_profile(
        viewer.training_params(),
        "auto",
        dataset_root=viewer.s.colmap_root,
        images_subdir=_profile_images_subdir(viewer),
    )
    init_hparams = init.hparams if profile.init_opacity_override is None else replace(init.hparams, initial_opacity=profile.init_opacity_override)
    return init, params, init_hparams, profile


def apply_live_params(viewer: object, force_init_defaults: bool = False) -> None:
    for renderer, allow_debug in ((viewer.s.renderer, True), (viewer.s.training_renderer, False), (viewer.s.debug_renderer, True)):
        if renderer is not None:
            for key, value in renderer_kwargs(viewer.renderer_params(allow_debug)).items():
                setattr(renderer, key, value)
    if viewer.s.renderer is not None:
        viewer.s.renderer.set_debug_grad_norm_buffer(
            viewer.s.training_renderer.work_buffers["debug_grad_norm"]
            if viewer.s.training_renderer is not None and viewer.s.trainer is not None
            else None
        )
    if viewer.s.trainer is not None:
        viewer.s.trainer.compute_debug_grad_norm = bool(viewer.s.renderer is not None and viewer.s.renderer.debug_show_grad_norm)
        _, params, _, _ = resolve_effective_training_setup(viewer)
        viewer.s.trainer.update_hyperparams(params.adam, params.stability, params.training)


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


def import_colmap_dataset(
    viewer: object,
    *,
    colmap_root: Path,
    database_path: Path | None,
    images_root: Path,
    init_mode: str,
    custom_ply_path: Path | None,
    nn_radius_scale_coef: float,
) -> None:
    root = Path(colmap_root).resolve()
    recon = load_colmap_reconstruction(root)
    xyz, _ = _point_tables(recon)
    _reset_loaded_runtime(viewer)
    _update_import_settings(
        viewer,
        dataset_root=root,
        database_path=database_path,
        images_root=images_root,
        init_mode=init_mode,
        custom_ply_path=custom_ply_path,
        nn_radius_scale_coef=nn_radius_scale_coef,
    )
    viewer.s.colmap_root = Path(root)
    viewer.s.colmap_recon = recon
    viewer.s.training_frames = build_training_frames_from_root(recon, images_root)
    viewer.s.colmap_point_count = int(xyz.shape[0])
    viewer.s.scene_path = None
    apply_live_params(viewer)
    viewer.apply_camera_fit(estimate_point_bounds(xyz))
    initialize_training_scene(viewer)
    viewer.s.last_error = ""
    print(
        f"Loaded COLMAP: db={None if database_path is None else Path(database_path).resolve()} root={root} "
        f"frames={len(viewer.s.training_frames)} images={Path(images_root).resolve()} init={init_mode}"
    )


def import_colmap_from_ui(viewer: object) -> None:
    colmap_root = Path(_ui_path_string(viewer, "colmap_root_path")).expanduser()
    database_path_text = _ui_path_string(viewer, "colmap_database_path")
    database_path = None if not database_path_text else Path(database_path_text).expanduser()
    images_root = Path(_ui_path_string(viewer, "colmap_images_root")).expanduser()
    init_mode = _ui_import_mode(viewer)
    custom_ply_text = _ui_path_string(viewer, "colmap_custom_ply_path")
    custom_ply_path = None if not custom_ply_text else Path(custom_ply_text).expanduser()
    nn_radius_scale_coef = float(viewer.ui._values.get("colmap_nn_radius_scale_coef", 0.25))
    if not colmap_root.exists():
        raise FileNotFoundError(f"COLMAP root does not exist: {colmap_root}")
    if not _has_colmap_sparse(colmap_root):
        colmap_root = _resolve_colmap_root_from_selection(colmap_root)
    if database_path is not None and not database_path.exists():
        database_path = _find_optional_colmap_database(colmap_root)
    if not images_root.exists():
        raise FileNotFoundError(f"COLMAP image folder does not exist: {images_root}")
    if init_mode == _COLMAP_IMPORT_CUSTOM_PLY and (custom_ply_path is None or not custom_ply_path.exists()):
        raise FileNotFoundError(f"Custom PLY does not exist: {custom_ply_path}")
    import_colmap_dataset(
        viewer,
        colmap_root=colmap_root,
        database_path=database_path,
        images_root=images_root,
        init_mode=init_mode,
        custom_ply_path=custom_ply_path,
        nn_radius_scale_coef=nn_radius_scale_coef,
    )


def initialize_training_scene(viewer: object) -> None:
    if viewer.s.colmap_recon is None or not viewer.s.training_frames:
        return
    init, params, init_hparams, profile = resolve_effective_training_setup(viewer)
    width, height = int(viewer.s.training_frames[0].width), int(viewer.s.training_frames[0].height)
    renderer = ensure_renderer(viewer, "training_renderer", width, height, allow_debug_overlays=False)
    import_cfg = viewer.s.colmap_import
    resolved_init = None
    if import_cfg.init_mode == _COLMAP_IMPORT_CUSTOM_PLY:
        if import_cfg.custom_ply_path is None:
            raise RuntimeError("Custom PLY initialization requires a selected PLY file.")
        scene = load_gaussian_ply(import_cfg.custom_ply_path)
        scale_reg_reference = None
    else:
        resolved_init = _pointcloud_init_hparams(viewer.s.colmap_recon, params.training.max_gaussians, init_hparams, import_cfg.nn_radius_scale_coef)
        scene = initialize_scene_from_colmap_points(
            recon=viewer.s.colmap_recon,
            max_gaussians=params.training.max_gaussians,
            seed=init.seed,
            init_hparams=resolved_init,
        )
        scale_reg_reference = float(max(resolved_init.base_scale, 1e-8))
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
    viewer.s.trainer = GaussianTrainer(**trainer_kwargs)
    viewer.s.scene = SceneCountProxy(scene.count)
    viewer.apply_camera_fit(estimate_scene_bounds(scene))
    enc = viewer.device.create_command_encoder()
    renderer.copy_scene_state_to(enc, viewer.s.renderer)
    viewer.device.submit_command_buffer(enc.finish())
    viewer.s.training_active = False
    viewer.s.training_elapsed_s = 0.0
    viewer.s.training_resume_time = None
    _invalidate(viewer)
    viewer.s.scene_init_signature = _scene_signature(viewer)
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
