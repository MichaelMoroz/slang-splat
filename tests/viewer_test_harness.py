from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sqlite3
import struct
import time

import numpy as np
from PIL import Image

from src.app.shared import build_training_params
from src.app.training_controls import training_control_defaults
from src.scene._internal.colmap_types import ColmapFrame, ColmapImage, ColmapPoint3D, ColmapReconstruction
from src.training.ppisp import PPISP_FIELD_SPECS
from src.viewer import presenter
from src.viewer.buffer_debug import ResourceDebugRow, ResourceDebugSnapshot
from src.viewer import session as viewer_session
from src.viewer.state import COLMAP_ROTATION_MODE_AUTO

_DEFAULT_TRAINING_PARAMS = build_training_params(background=(1.0, 1.0, 1.0))
_TRAINING_CONTROL_DEFAULTS = training_control_defaults()
_TRAINING_VISUAL_CACHE_DEFAULTS = {
    "cached_raster_grad_histograms": None,
    "cached_raster_grad_ranges": None,
    "cached_raster_grad_histogram_mode": "",
    "cached_raster_grad_histogram_step": -1,
    "cached_raster_grad_histogram_scene_count": -1,
    "cached_raster_grad_histogram_signature": None,
    "cached_raster_grad_histogram_status": "",
}
_CAMERA_PACKS = {
    0: ("<ddd", (64.0, 32.0, 32.0)),
    1: ("<dddd", (64.0, 64.0, 32.0, 32.0)),
    2: ("<dddd", (64.0, 32.0, 32.0, 0.01)),
    3: ("<ddddd", (64.0, 32.0, 32.0, 0.01, -0.01)),
    4: ("<dddddddd", (64.0, 64.0, 32.0, 32.0, 0.01, -0.01, 0.0, 0.0)),
    6: ("<dddddddddddd", (64.0, 64.0, 32.0, 32.0, 0.01, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
}


def _renderer_params(**kwargs):
    values = {
        "debug_mode": None,
        "debug_show_ellipses": False,
        "debug_show_processed_count": False,
        "debug_show_grad_norm": False,
        **kwargs,
    }

    class _Params(SimpleNamespace):
        __dataclass_fields__ = {key: None for key in values}

        def renderer_kwargs(self):
            return dict(values)

    return _Params(**values)


class _FinishedEncoder:
    def finish(self):
        return "finished"


class _DebugViewportRenderer:
    def __init__(self):
        self.grad_buffer = "stale-grad"
        self.splat_age_buffer = "stale-splat-age"
        self.copy_targets = []

    def set_debug_grad_norm_buffer(self, buffer):
        self.grad_buffer = buffer

    def set_debug_splat_age_buffer(self, buffer):
        self.splat_age_buffer = buffer

    def copy_scene_state_to(self, encoder, dst):
        del encoder
        self.copy_targets.append(dst)


class _DebugTrainingRenderer:
    width = 32
    height = 32
    work_buffers = {"debug_grad_norm": "new-grad"}

    def __init__(self):
        self.copy_calls = []
        self.grad_buffer = None
        self.splat_age_buffer = None

    def copy_scene_state_to(self, encoder, dst, *, include_work_buffers=True):
        del encoder
        self.copy_calls.append((dst, bool(include_work_buffers)))
        dst.copy_targets.append(self)

    def set_debug_grad_norm_buffer(self, buffer):
        self.grad_buffer = buffer

    def set_debug_splat_age_buffer(self, buffer):
        self.splat_age_buffer = buffer


class _TonemapAwareTrainer(SimpleNamespace):
    def set_target_tonemap_provider(self, provider):
        self.target_tonemap_provider = provider


def _training_setup(*, max_sh_band=None):
    training = {"max_gaussians": 8}
    if max_sh_band is not None:
        training["max_sh_band"] = max_sh_band
    return (
        SimpleNamespace(seed=7),
        SimpleNamespace(training=SimpleNamespace(**training), adam=SimpleNamespace(tag="adam"), stability=SimpleNamespace(tag="stability")),
        SimpleNamespace(),
        SimpleNamespace(name="test"),
    )


def _patch_fixed_training_resolution(monkeypatch):
    monkeypatch.setattr(viewer_session, "resolve_effective_train_render_factor", lambda training, step, width=None, height=None: 1)
    monkeypatch.setattr(viewer_session, "resolve_training_resolution", lambda width, height, factor: (width, height))


def _make_training_scene_viewer(
    *,
    calls=None,
    ui_values=None,
    training_frames=None,
    trainer=None,
    renderer=None,
    debug_renderer=None,
    training_renderer=None,
    colmap_import=None,
    photometric_trainer=None,
    photometric_active=False,
    photometric_elapsed_s=0.0,
    photometric_resume_time=None,
    training_active=True,
    training_elapsed_s=12.0,
    training_resume_time=3.0,
    renderer_mode=None,
    last_error="stale",
    colmap_root=None,
    state_overrides=None,
):
    state = {
        "colmap_recon": object(),
        "training_frames": [SimpleNamespace(width=32, height=32)] if training_frames is None else training_frames,
        "colmap_import": colmap_import or SimpleNamespace(init_mode="pointcloud", custom_ply_path=None, nn_radius_scale_coef=0.5, diffused_point_count=100),
        "trainer": trainer,
        "photometric_trainer": photometric_trainer,
        "photometric_active": photometric_active,
        "photometric_elapsed_s": photometric_elapsed_s,
        "photometric_resume_time": photometric_resume_time,
        "renderer": renderer or _DebugViewportRenderer(),
        "debug_renderer": debug_renderer or _DebugViewportRenderer(),
        "training_renderer": training_renderer,
        "training_active": training_active,
        "training_elapsed_s": training_elapsed_s,
        "training_resume_time": training_resume_time,
        "scene": None,
        "applied_renderer_params_training": None,
        "applied_renderer_params_debug": None,
        "applied_training_signature": None,
        "applied_training_runtime_signature": None,
        "applied_training_runtime_factor": None,
        "pending_training_runtime_resize": True,
        "training_runtime_factor_changed": False,
        "last_training_batch_steps": 0,
        "last_error": last_error,
        "colmap_root": colmap_root,
        **_TRAINING_VISUAL_CACHE_DEFAULTS,
    }
    if state_overrides:
        state.update(state_overrides)
    return SimpleNamespace(
        device=SimpleNamespace(create_command_encoder=lambda: _FinishedEncoder(), submit_command_buffer=lambda command_buffer: None),
        toolkit=SimpleNamespace(reset_plot_history=(lambda: calls.append("reset_plot_history")) if calls is not None else (lambda: None)),
        ui=SimpleNamespace(_values={} if ui_values is None else dict(ui_values)),
        init_params=lambda: SimpleNamespace(seed=7),
        renderer_params=lambda allow_debug_overlays: SimpleNamespace(mode=renderer_mode if renderer_mode is not None else ("debug" if allow_debug_overlays else "train")),
        training_params=lambda: object(),
        apply_camera_fit=lambda bounds: None,
        s=SimpleNamespace(**state),
    )


def _assign_training_renderer(calls, training_renderer):
    def _ensure_renderer(viewer_obj, attr, width, height, allow_debug_overlays):
        del attr, width, height, allow_debug_overlays
        calls.append(f"ensure_renderer_cleared={viewer_obj.s.training_renderer is None}")
        viewer_obj.s.training_renderer = training_renderer
        return training_renderer

    return _ensure_renderer


def _capturing_gaussian_trainer(captured, trainer):
    def _gaussian_trainer(**kwargs):
        captured["frame_targets_native"] = kwargs.get("frame_targets_native")
        return trainer

    return _gaussian_trainer


def _patch_training_scene_bootstrap(
    monkeypatch,
    *,
    training_setup=None,
    ensure_renderer,
    gaussian_trainer,
    apply_live_params=None,
    build_initial_training_scene=None,
    reset_camera=None,
    renderer_signature=None,
    training_live_signature=None,
    training_params_signature=None,
    training_runtime_signature=None,
    update_slider=None,
    reset_training_visual=None,
    reset_loss_debug=None,
    sync_photometric_target_provider=None,
    apply_training_image_color_init=None,
    apply_debug_buffers=None,
):
    monkeypatch.setattr(viewer_session, "resolve_effective_training_setup", lambda viewer_obj: _training_setup() if training_setup is None else training_setup)
    _patch_fixed_training_resolution(monkeypatch)
    for name, value in (
        ("ensure_renderer", ensure_renderer),
        ("_build_initial_training_scene", build_initial_training_scene or (lambda viewer_obj, init, params, init_hparams: (SimpleNamespace(count=8), 0.25))),
        ("apply_live_params", apply_live_params or (lambda viewer_obj: None)),
        ("GaussianTrainer", gaussian_trainer),
        ("reset_main_camera", reset_camera or (lambda viewer_obj: None)),
        ("_renderer_params_signature", renderer_signature or (lambda params: (params.mode,))),
        ("_training_live_params_signature", training_live_signature or (lambda params: ("training-live",))),
        ("_training_params_signature", training_params_signature or (lambda params: ("training",))),
        ("_training_runtime_signature", training_runtime_signature or (lambda params: ("training-runtime",))),
        ("update_debug_frame_slider_range", update_slider or (lambda viewer_obj: None)),
    ):
        monkeypatch.setattr(viewer_session, name, value)
    if reset_training_visual is not None:
        monkeypatch.setattr(viewer_session, "_reset_training_visual_state", reset_training_visual)
    if reset_loss_debug is not None:
        monkeypatch.setattr(viewer_session, "_reset_loss_debug", reset_loss_debug)
    if sync_photometric_target_provider is not None:
        monkeypatch.setattr(viewer_session, "sync_photometric_target_provider", sync_photometric_target_provider)
    if apply_training_image_color_init is not None:
        monkeypatch.setattr(viewer_session, "_apply_training_image_color_init", apply_training_image_color_init)
    if apply_debug_buffers is not None:
        monkeypatch.setattr(viewer_session, "_apply_debug_buffers", apply_debug_buffers)


def _identity_q():
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _write_cameras_bin(path: Path, model_id=1):
    fmt, values = _CAMERA_PACKS.get(model_id, ("<dddddddd", (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)))
    with path.open("wb") as handle:
        handle.write(struct.pack("<QiiQQ", 1, 7, model_id, 64, 64))
        handle.write(struct.pack(fmt, *values))


def _write_images_bin(path: Path, image_names):
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(image_names)))
        for image_id, image_name in enumerate(image_names, start=1):
            handle.write(struct.pack("<i", image_id))
            handle.write(struct.pack("<ddddddd", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0))
            handle.write(struct.pack("<i", 7))
            handle.write(image_name.encode("utf-8") + b"\x00" + struct.pack("<Q", 0))


def _write_points3d_bin(path: Path):
    with path.open("wb") as handle:
        handle.write(struct.pack("<QQdddBBBdQ", 1, 11, 0.0, 0.0, 0.0, 255, 255, 255, 0.0, 3))
        handle.write(struct.pack("<ii", 1, 0) * 3)


def _write_database(path: Path, image_names):
    with sqlite3.connect(str(path)) as conn:
        conn.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        conn.executemany("INSERT INTO images(image_id, name) VALUES (?, ?)", list(enumerate(image_names, start=1)))
        conn.commit()


def _write_test_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(path)


def _build_colmap_tree(tmp_path: Path, *, image_names, image_root_rel: Path, model_id=1):
    root = tmp_path / "scene"
    sparse = root / "sparse" / "0"
    images_root = root / image_root_rel
    sparse.mkdir(parents=True)
    (root / "distorted").mkdir(parents=True)
    images_root.mkdir(parents=True, exist_ok=True)
    _write_cameras_bin(sparse / "cameras.bin", model_id)
    _write_images_bin(sparse / "images.bin", image_names)
    _write_points3d_bin(sparse / "points3D.bin")
    _write_database(root / "distorted" / "database.db", image_names)
    for image_name in image_names:
        _write_test_image((images_root / image_name).resolve())
    return root / "distorted" / "database.db", images_root.resolve()


def _build_colmap_tree_without_database(tmp_path: Path, *, image_names, image_root_rel: Path, model_id=1):
    root = tmp_path / "scene_no_db"
    sparse = root / "sparse" / "0"
    images_root = root / image_root_rel
    sparse.mkdir(parents=True)
    images_root.mkdir(parents=True, exist_ok=True)
    _write_cameras_bin(sparse / "cameras.bin", model_id)
    _write_images_bin(sparse / "images.bin", image_names)
    _write_points3d_bin(sparse / "points3D.bin")
    for image_name in image_names:
        _write_test_image((images_root / image_name).resolve())
    return root.resolve(), images_root.resolve()


def _make_import_renderer(calls=None):
    record = (lambda name, value=None: calls.append((name, value))) if calls is not None else (lambda name, value=None: None)
    return SimpleNamespace(
        clear_scene_resources=lambda: record("clear_renderer"),
        set_debug_grad_norm_buffer=lambda buffer: record("clear_grad_debug", buffer),
        set_debug_splat_age_buffer=lambda buffer: record("clear_splat_age_debug", buffer),
    )


def _make_import_viewer(
    *,
    calls=None,
    ui_values=None,
    toolkit=None,
    init_params=None,
    trainer=None,
    training_frames=None,
    scene=None,
    scene_path=None,
    colmap_root=None,
    colmap_recon=None,
    colmap_import_progress=None,
    last_error="",
    state_overrides=None,
):
    state = {
        "renderer": _make_import_renderer(calls),
        "trainer": trainer,
        "training_active": False,
        "training_elapsed_s": 0.0,
        "training_resume_time": None,
        "training_renderer": None,
        "training_frames": [] if training_frames is None else training_frames,
        "scene": scene,
        "scene_path": scene_path,
        "colmap_root": colmap_root,
        "colmap_recon": colmap_recon,
        "colmap_import_progress": colmap_import_progress,
        "applied_renderer_params_training": None,
        "applied_renderer_params_debug": None,
        "applied_training_signature": None,
        "applied_training_runtime_factor": None,
        "pending_training_runtime_resize": False,
        "applied_renderer_params_main": None,
        "cached_training_setup_signature": None,
        "cached_training_setup": None,
        "last_error": last_error,
    }
    if state_overrides:
        state.update(state_overrides)
    viewer = SimpleNamespace(ui=SimpleNamespace(_values={} if ui_values is None else dict(ui_values)), s=SimpleNamespace(**state), c=lambda key: SimpleNamespace(value=0))
    if toolkit is not None:
        viewer.toolkit = toolkit
    if init_params is not None:
        viewer.init_params = init_params
    return viewer


def _patch_import_runtime_resets(monkeypatch, calls=None):
    def _record(name):
        return (lambda viewer_obj: calls.append((name, None))) if calls is not None else (lambda viewer_obj: None)

    for attr, name in (
        ("update_debug_frame_slider_range", "update_slider"),
        ("_clear_cached_init_source", "clear_cached_init"),
        ("_reset_training_visual_state", "reset_training_visual"),
        ("_reset_loss_debug", "reset_loss_debug"),
    ):
        monkeypatch.setattr(viewer_session, attr, _record(name))


def _patch_import_pipeline(monkeypatch, *, recon, load_reconstruction=None, build_training_frames=None, create_textures=None, finish=None):
    monkeypatch.setattr(viewer_session, "_load_aligned_colmap_reconstruction", load_reconstruction or (lambda root, rotation_mode=COLMAP_ROTATION_MODE_AUTO, custom_rotation_deg=(0.0, 0.0, 0.0): recon))
    monkeypatch.setattr(
        viewer_session,
        "build_training_frames_from_root",
        build_training_frames or (lambda root, recon_obj, image_paths=None: [ColmapFrame(image_id=1, camera_id=7, width=64, height=64, fx=64.0, fy=64.0, cx=32.0, cy=32.0, q_wxyz=_identity_q(), t_xyz=np.asarray((0.0, 0.0, -2.0), dtype=np.float32), image_path=Path("image.png"))]),
    )
    monkeypatch.setattr(viewer_session, "_create_native_dataset_textures", create_textures or (lambda viewer_obj, frames: ["texture"] * len(frames)))
    monkeypatch.setattr(viewer_session, "_finish_import_colmap_dataset", finish or (lambda viewer_obj, **kwargs: None))


def _patch_advance_import_pipeline(monkeypatch, *, recon, load_reconstruction=None, load_training_frame=None, create_texture=None, finish=None):
    monkeypatch.setattr(viewer_session, "_load_aligned_colmap_reconstruction", load_reconstruction or (lambda root, rotation_mode=COLMAP_ROTATION_MODE_AUTO, custom_rotation_deg=(0.0, 0.0, 0.0): recon))
    monkeypatch.setattr(viewer_session, "load_training_frame_rgba8", load_training_frame or (lambda frame: np.zeros((frame.height, frame.width, 4), dtype=np.uint8)))
    monkeypatch.setattr(viewer_session, "_create_native_dataset_texture_from_rgba8", create_texture or (lambda viewer_obj, rgba8: rgba8))
    monkeypatch.setattr(viewer_session, "_finish_import_colmap_dataset", finish or (lambda viewer_obj, **kwargs: None))


def _training_hparams(**overrides):
    return replace(_DEFAULT_TRAINING_PARAMS.training, **overrides)


def _training_camera_colmap_viewer(*, observations, points, loss_debug_frame=0, state_overrides=None):
    recon = ColmapReconstruction(
        root=Path("dataset"),
        sparse_dir=Path("dataset/sparse/0"),
        cameras={},
        images={
            int(spec["image_id"]): ColmapImage(
                image_id=int(spec["image_id"]),
                q_wxyz=np.asarray(spec.get("q_wxyz", (1.0, 0.0, 0.0, 0.0)), dtype=np.float32),
                t_xyz=np.asarray(spec.get("t_xyz", (0.0, 0.0, 0.0)), dtype=np.float32),
                camera_id=int(spec.get("camera_id", 7)),
                name=str(spec["name"]),
                points2d_xy=np.asarray(spec["points2d_xy"], dtype=np.float32).reshape((-1, 2)),
                points2d_point3d_ids=np.asarray(spec["point_ids"], dtype=np.int64),
            )
            for spec in observations
        },
        points3d={
            int(spec["point_id"]): ColmapPoint3D(
                point_id=int(spec["point_id"]),
                xyz=np.asarray(spec.get("xyz", (0.0, 1.0, 2.0)), dtype=np.float32),
                rgb=np.asarray(spec.get("rgb", (1.0, 0.5, 0.25)), dtype=np.float32),
                error=float(spec.get("error", 0.125)),
                track_length=int(spec.get("track_length", 2)),
            )
            for spec in points
        },
    )
    state = {
        "training_frames": [
            SimpleNamespace(image_id=int(spec["image_id"]), width=int(spec["width"]), height=int(spec["height"]), image_path=Path(spec["name"]))
            for spec in observations
        ],
        "colmap_recon": recon,
        "training_camera_colmap_observation_index": None,
        "training_camera_colmap_observation_signature": None,
        "training_camera_colmap_payload": None,
        "training_camera_colmap_payload_signature": None,
        "training_camera_colmap_payload_cache": None,
        "training_camera_colmap_payload_cache_signature": None,
    }
    if state_overrides:
        state.update(state_overrides)
    viewer = SimpleNamespace()
    viewer.ui = SimpleNamespace(controls={"loss_debug_frame": _control(loss_debug_frame)})
    viewer.c = lambda key: viewer.ui.controls[key]
    viewer.s = SimpleNamespace(**state)
    return viewer


def _resource_debug_snapshot(*rows, total_consumption=None, process_vram=None, process_vram_delta=None, process_vram_source=""):
    debug_rows = tuple(ResourceDebugRow(*row) if not isinstance(row, ResourceDebugRow) else row for row in rows)
    buffer_sizes = sorted(max(int(row.byte_size), 0) for row in debug_rows if row.kind == "Buffer")
    texture_total = sum(max(int(row.byte_size), 0) for row in debug_rows if row.kind == "Texture")
    total = sum(max(int(row.byte_size), 0) for row in debug_rows) if total_consumption is None else max(int(total_consumption), 0)
    return ResourceDebugSnapshot(
        rows=debug_rows,
        total_consumption=total,
        buffer_count=len(buffer_sizes),
        buffer_total=sum(buffer_sizes),
        buffer_mean=float(sum(buffer_sizes) / len(buffer_sizes)) if buffer_sizes else 0.0,
        buffer_median=float(np.median(buffer_sizes)) if buffer_sizes else 0.0,
        texture_count=sum(1 for row in debug_rows if row.kind == "Texture"),
        texture_total=texture_total,
        process_vram=process_vram,
        process_vram_delta=process_vram_delta,
        process_vram_source=process_vram_source,
    )


def _buffer_snapshot(byte_size, *, process_vram=None, process_vram_delta=None, process_vram_source=""):
    return _resource_debug_snapshot(("Buffer", "renderer.buf", "viewer.main_renderer.buf", byte_size, "buf", "rw", 1), process_vram=process_vram, process_vram_delta=process_vram_delta, process_vram_source=process_vram_source)


def _patch_vram_queries(monkeypatch, *, used=None, used_source="", capacity=None, capacity_source=""):
    monkeypatch.setattr(presenter, "query_total_device_vram_used_cached", lambda _device, *, allow_heap_query=True: (used, used_source))
    monkeypatch.setattr(presenter, "query_total_device_vram_capacity", lambda _device: (capacity, capacity_source))


def _viewer_controls(loss_debug):
    values = dict(_TRAINING_CONTROL_DEFAULTS)
    values.update({"debug_mode": 0 if loss_debug else 1, "loss_debug_frame": 0, "loss_debug_view": 0, "loss_debug_abs_scale": 1.0, "images_subdir": 0, "training_steps_per_frame": 3, "train_auto_start_downscale": 1, "train_downscale_factor": 1, "train_subsample_factor": 0})
    return {key: _control(value) for key, value in values.items()}


class _DummyEncoder:
    def __init__(self):
        self.clear_calls = []
        self.groups = []

    def clear_texture_float(self, texture, clear_value):
        self.clear_calls.append((texture, clear_value))

    def push_debug_group(self, label, _color):
        self.groups.append(("push", label))

    def pop_debug_group(self):
        self.groups.append(("pop", None))

    def finish(self):
        return "finished"


class _DummyRenderer:
    def __init__(self, width=640, height=360):
        self.width = width
        self.height = height
        self.sh_band = 0
        self.debug_mode = "normal"
        self.max_sh_band = 3
        self.training_forward_calls = []
        self.render_calls = []
        self.render_linear_calls = []
        self.render_ppisp_calls = []
        self.resolution_calls = []

    def set_render_resolution(self, width, height):
        self.width, self.height = int(width), int(height)
        self.resolution_calls.append((self.width, self.height))
        return True

    def render_to_texture(self, camera, background, read_stats, command_encoder):
        self.render_calls.append({"camera": camera, "background": background, "read_stats": read_stats, "command_encoder": command_encoder})
        return "main_render_tex", {"generated_entries": 1, "written_entries": 2, "overflow": False}

    def render_linear_to_texture(self, camera, background, read_stats, command_encoder):
        self.render_linear_calls.append({"camera": camera, "background": background, "read_stats": read_stats, "command_encoder": command_encoder})
        return "main_linear_tex", {"generated_entries": 3, "written_entries": 4, "overflow": False}

    def render_ppisp_to_texture(self, camera, ppisp_tonemap, background, read_stats, command_encoder):
        self.render_ppisp_calls.append({"camera": camera, "ppisp_tonemap": ppisp_tonemap, "background": background, "read_stats": read_stats, "command_encoder": command_encoder})
        return "main_ppisp_tex", {"generated_entries": 5, "written_entries": 6, "overflow": False}

    def render_training_forward_to_texture(self, camera, background, read_stats, command_encoder, **kwargs):
        self.training_forward_calls.append({"camera": camera, "background": background, "read_stats": read_stats, "command_encoder": command_encoder, **kwargs})
        return "training_preview_tex", {"generated_entries": 1, "written_entries": 2, "overflow": False}


class _DummyTrainer:
    def __init__(self):
        self.state = SimpleNamespace(step=0, last_loss=0.0, avg_loss=0.0, last_mse=0.0, avg_mse=0.0, last_ssim=1.0, avg_ssim=1.0, last_psnr=float("inf"), avg_psnr=float("inf"), avg_density_loss=0.0, last_frame_index=0, last_instability="")
        self.scene = SimpleNamespace(count=4)
        self.native_target_is_linear = False
        self.training = _training_hparams(train_auto_start_downscale=1, train_subsample_factor=1)
        self.step_calls = 0
        self.step_batch_calls = []
        self.training_resolution_calls = []
        self.sample_vars_calls = []
        self.background_seed_calls = []
        self.hparam_calls = []
        self.sort_calls = []
        self.target_calls = []
        self.subsample_factor = 1
        self.target_tonemap_provider = None
        self.refinement_buffers = {"clone_counts": "clone_counts", "splat_contribution": "splat_contribution"}

    def step(self):
        self.step_calls += 1

    def step_batch(self, steps):
        steps = int(steps)
        self.step_batch_calls.append(steps)
        self.step_calls += steps
        return steps

    def make_frame_camera(self, frame_index, width, height):
        return (frame_index, width, height)

    def effective_train_downscale_factor(self):
        return 1

    def effective_train_subsample_factor(self, frame_index=0, step=None):
        return int(self.subsample_factor)

    def effective_train_render_factor(self):
        return 1

    def training_resolution(self, frame_index=0, step=None):
        self.training_resolution_calls.append((int(frame_index), int(step or 0)))
        return (320, 180)

    def frame_size(self, frame_index):
        return (640, 360)

    def current_base_lr(self):
        return 0.005

    def get_frame_target_texture(self, frame_index, native_resolution=True, encoder=None, step=None, apply_target_tonemap=True):
        self.target_calls.append((int(frame_index), bool(native_resolution), bool(apply_target_tonemap)))
        return f"target_tex_{frame_index}_{native_resolution}"

    def target_texture_is_linear(self, target_texture=None):
        if target_texture is None:
            return False
        name = str(target_texture)
        return name.endswith("_False") or (self.native_target_is_linear and name.endswith("_True"))

    def training_background(self):
        return np.asarray([0.25, 0.5, 0.75], dtype=np.float32)

    def training_background_seed(self, seed_index=None):
        self.background_seed_calls.append(None if seed_index is None else int(seed_index))
        return 1000 + int(seed_index or 0)

    def training_sample_vars(self, frame_index, step=None, sample_seed_step=None):
        self.sample_vars_calls.append((int(frame_index), int(step or 0), None if sample_seed_step is None else int(sample_seed_step)))
        return {"g_TrainingSubsample": {"enabled": np.uint32(1 if self.subsample_factor > 1 else 0), "factor": np.uint32(2), "nativeWidth": np.uint32(640), "nativeHeight": np.uint32(360), "frameIndex": np.uint32(frame_index), "stepIndex": np.uint32(0 if sample_seed_step is None else sample_seed_step)}}

    def apply_renderer_training_hparams(self, step=None, renderer=None):
        self.hparam_calls.append((None if step is None else int(step), renderer))
        if renderer is not None:
            renderer.sh_band = 2

    def sorting_dither(self, frame_index, step, camera):
        self.sort_calls.append((int(frame_index), int(step), camera))
        return SimpleNamespace(position=np.asarray([1.0, 2.0, 3.0], dtype=np.float32), sigma=0.125, seed=555)

    def renderer_training_workspace(self):
        return {"workspace": True}

    def frame_metrics_snapshot(self):
        return {"loss": np.asarray([0.25], dtype=np.float64), "mse": np.asarray([0.125], dtype=np.float64), "psnr": np.asarray([32.5], dtype=np.float64), "visited": np.asarray([True], dtype=bool)}


class _CaptureKernel:
    def __init__(self):
        self.calls = []

    def dispatch(self, *, thread_count, vars, command_encoder):
        self.calls.append({"thread_count": thread_count, "vars": vars, "command_encoder": command_encoder})


def _control(value):
    return SimpleNamespace(value=value)


def _text():
    return SimpleNamespace(text="")


def _section_dict(sections):
    return {title: dict(rows) for title, rows in sections}


def _render_context(width=640, height=360):
    return SimpleNamespace(surface_texture=SimpleNamespace(width=width, height=height), command_encoder=_DummyEncoder())


def _patch_render_frame(
    monkeypatch,
    calls,
    *,
    apply_live_params=None,
    ensure_training_runtime_resolution=None,
    recreate_renderer=None,
    maybe_reallocate_renderers=None,
    advance_colmap_import=None,
    render_debug_view=None,
    render_main_view=None,
    update_ui_text=None,
    refresh_cached_raster_grad_histograms=None,
):
    for name, value in (
        ("apply_live_params", apply_live_params or (lambda viewer_obj: calls.append("apply"))),
        ("ensure_training_runtime_resolution", ensure_training_runtime_resolution or (lambda viewer_obj: calls.append("train_resize"))),
        ("recreate_renderer", recreate_renderer or (lambda viewer_obj, width, height: calls.append("resize"))),
        ("update_debug_frame_slider_range", lambda viewer_obj: None),
        ("advance_photometric_initialization", lambda viewer_obj: None),
        ("advance_dataset_metrics", lambda viewer_obj: None),
    ):
        monkeypatch.setattr(presenter.session, name, value)
    if maybe_reallocate_renderers is not None:
        monkeypatch.setattr(presenter.session, "maybe_reallocate_renderers", maybe_reallocate_renderers)
    if advance_colmap_import is not None:
        monkeypatch.setattr(presenter.session, "advance_colmap_import", advance_colmap_import)
    if refresh_cached_raster_grad_histograms is not None:
        monkeypatch.setattr(presenter.session, "refresh_cached_raster_grad_histograms", refresh_cached_raster_grad_histograms)
    monkeypatch.setattr(presenter, "_render_debug_view", render_debug_view or (lambda viewer_obj, encoder, width, height, render_frame_index: calls.append("debug") or "debug_tex"))
    monkeypatch.setattr(presenter, "_render_main_view", render_main_view or (lambda viewer_obj, encoder: calls.append("main") or "main_tex"))
    monkeypatch.setattr(presenter, "update_ui_text", update_ui_text or (lambda viewer_obj, dt: calls.append("ui")))


def _viewer(loss_debug):
    trainer = _DummyTrainer()
    texts = {key: _text() for key in ("fps", "images_subdir", "loss_debug_frame", "loss_debug_psnr", "path", "scene_stats", "render_stats", "training", "training_time", "training_iters_avg", "training_loss", "training_ssim", "training_density", "training_psnr", "training_instability", "training_resolution", "training_downscale", "training_schedule", "training_refinement", "colmap_import_status", "colmap_import_current", "histogram_status", "error")}
    frame = SimpleNamespace(image_id=5, camera_id=7, image_path=Path("frame.png"), width=640, height=360, fx=525.0, fy=520.0, cx=320.0, cy=180.0, k1=0.01, k2=-0.02, p1=0.001, p2=-0.002, k3=0.003, k4=-0.004, k5=0.005, k6=-0.006, make_camera=lambda near=0.1, far=120.0: SimpleNamespace(position=np.array([1.0, 2.0, 3.0], dtype=np.float32), target=np.array([1.5, 2.0, 4.0], dtype=np.float32), up=np.array([0.0, 1.0, 0.0], dtype=np.float32), near=near, far=far))
    values = {spec.key: spec.default for spec in PPISP_FIELD_SPECS}
    values.update({"show_histograms": False, "_histogram_payload": None, "_histogram_range_payload": None, "show_training_cameras": bool(loss_debug), "show_training_views": False, "show_camera_overlays": False, "show_camera_labels": False, "training_camera_full_resolution": False, "training_camera_ppisp_tonemap": True})
    viewer = SimpleNamespace(device=SimpleNamespace(), toolkit=SimpleNamespace(viewport_size=lambda: (640, 360)), loss_debug_view_options=(("rendered", "Rendered"), ("target", "Target"), ("abs_diff", "Abs Diff"), ("dssim", "DSSIM"), ("rendered_edges", "Rendered Edges"), ("target_edges", "Target Edges")))
    viewer.ui = SimpleNamespace(controls=_viewer_controls(loss_debug), texts=texts, _values=values, _texts={key: value.text for key, value in texts.items()})
    viewer.c = lambda key: viewer.ui.controls[key]
    viewer.t = lambda key: viewer.ui.texts[key]
    viewer.camera = lambda: "camera"
    viewer.update_camera = lambda dt: None
    viewer.s = SimpleNamespace(fps_smooth=60.0, last_time=time.perf_counter(), last_interaction_time=0.0, renderer=_DummyRenderer(), debug_renderer=None, scene=SimpleNamespace(count=4), stats={}, scene_path=None, colmap_root=Path("dataset"), training_frames=[frame], trainer=trainer, training_active=True, training_renderer=_DummyRenderer(), background=(0.0, 0.0, 0.0), training_elapsed_s=0.0, training_resume_time=None, last_render_exception="", last_error="", last_training_batch_steps=0, viewport_texture=None, debug_target_texture=None, debug_dssim_features_kernel=None, debug_dssim_compose_kernel=None, debug_dssim_blur=None, debug_dssim_resolution=None, debug_dssim_moments=None, debug_dssim_blurred_moments=None, debug_target_sample_kernel=None, colmap_import_progress=None, cached_raster_grad_ranges=None, cached_training_setup_signature=None, cached_training_setup=None, render_frame_index=0)
    viewer.init_params = lambda: SimpleNamespace(seed=7, hparams=SimpleNamespace(initial_opacity=None))
    viewer.renderer_params = lambda allow_debug_overlays: _renderer_params(debug=bool(allow_debug_overlays))
    viewer.training_params = lambda: replace(_DEFAULT_TRAINING_PARAMS, training=viewer.s.trainer.training) if viewer.s.trainer is not None else _DEFAULT_TRAINING_PARAMS
    return viewer
