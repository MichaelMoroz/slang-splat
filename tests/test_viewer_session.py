from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from src.scene import GaussianInitHyperParams, GaussianScene
from src.scene._internal.colmap_types import ColmapFrame
from src.viewer import session
from src.viewer.state import ColmapImportProgress, ColmapImportSettings


def _viewer() -> SimpleNamespace:
    return SimpleNamespace(
        s=SimpleNamespace(
            trainer=SimpleNamespace(state=SimpleNamespace(step=0)),
            training_active=False,
            training_elapsed_s=0.0,
            training_resume_time=None,
        )
    )


def _renderer_params(**kwargs) -> SimpleNamespace:
    values = {
        "debug_mode": None,
        "debug_show_ellipses": False,
        "debug_show_processed_count": False,
        "debug_show_grad_norm": False,
        **kwargs,
    }

    class _Params(SimpleNamespace):
        __dataclass_fields__ = {key: None for key in values}

        def renderer_kwargs(self) -> dict[str, object]:
            return dict(values)

    return _Params(**values)


def _identity_q() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _write_cameras_bin(path: Path, model_id: int = 1) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<i", 7))
        handle.write(struct.pack("<i", model_id))
        handle.write(struct.pack("<Q", 64))
        handle.write(struct.pack("<Q", 64))
        if model_id == 0:
            handle.write(struct.pack("<ddd", 64.0, 32.0, 32.0))
        elif model_id == 1:
            handle.write(struct.pack("<dddd", 64.0, 64.0, 32.0, 32.0))
        elif model_id == 2:
            handle.write(struct.pack("<dddd", 64.0, 32.0, 32.0, 0.01))
        elif model_id == 3:
            handle.write(struct.pack("<ddddd", 64.0, 32.0, 32.0, 0.01, -0.01))
        elif model_id == 4:
            handle.write(struct.pack("<dddddddd", 64.0, 64.0, 32.0, 32.0, 0.01, -0.01, 0.0, 0.0))
        elif model_id == 6:
            handle.write(struct.pack("<dddddddddddd", 64.0, 64.0, 32.0, 32.0, 0.01, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        else:
            handle.write(struct.pack("<dddddddd", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))


def _write_images_bin(path: Path, image_names: list[str]) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(image_names)))
        for image_id, image_name in enumerate(image_names, start=1):
            handle.write(struct.pack("<i", image_id))
            handle.write(struct.pack("<dddd", 1.0, 0.0, 0.0, 0.0))
            handle.write(struct.pack("<ddd", 0.0, 0.0, -2.0))
            handle.write(struct.pack("<i", 7))
            handle.write(image_name.encode("utf-8"))
            handle.write(b"\x00")
            handle.write(struct.pack("<Q", 0))


def _write_points3d_bin(path: Path) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<Q", 11))
        handle.write(struct.pack("<ddd", 0.0, 0.0, 0.0))
        handle.write(struct.pack("<BBB", 255, 255, 255))
        handle.write(struct.pack("<d", 0.0))
        handle.write(struct.pack("<Q", 3))
        handle.write(struct.pack("<ii", 1, 0) * 3)


def _write_database(path: Path, image_names: list[str]) -> None:
    with sqlite3.connect(str(path)) as conn:
        conn.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        conn.executemany("INSERT INTO images(image_id, name) VALUES (?, ?)", list(enumerate(image_names, start=1)))
        conn.commit()


def _build_colmap_tree(tmp_path: Path, *, image_names: list[str], image_root_rel: Path, model_id: int = 1) -> tuple[Path, Path]:
    root = tmp_path / "scene"
    sparse = root / "sparse" / "0"
    database_path = root / "distorted" / "database.db"
    images_root = root / image_root_rel
    sparse.mkdir(parents=True)
    database_path.parent.mkdir(parents=True)
    images_root.mkdir(parents=True, exist_ok=True)
    _write_cameras_bin(sparse / "cameras.bin", model_id=model_id)
    _write_images_bin(sparse / "images.bin", image_names)
    _write_points3d_bin(sparse / "points3D.bin")
    _write_database(database_path, image_names)
    for image_name in image_names:
        image_path = (images_root / image_name).resolve()
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(image_path)
    return database_path, images_root.resolve()


def _build_colmap_tree_without_database(tmp_path: Path, *, image_names: list[str], image_root_rel: Path, model_id: int = 1) -> tuple[Path, Path]:
    root = tmp_path / "scene_no_db"
    sparse = root / "sparse" / "0"
    images_root = root / image_root_rel
    sparse.mkdir(parents=True)
    images_root.mkdir(parents=True, exist_ok=True)
    _write_cameras_bin(sparse / "cameras.bin", model_id=model_id)
    _write_images_bin(sparse / "images.bin", image_names)
    _write_points3d_bin(sparse / "points3D.bin")
    for image_name in image_names:
        image_path = (images_root / image_name).resolve()
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(image_path)
    return root.resolve(), images_root.resolve()


def test_bc7_cache_compression_uses_resized_frame_image(tmp_path: Path, monkeypatch) -> None:
    images_root = tmp_path / "images"
    images_root.mkdir()
    image_path = images_root / "frame.png"
    Image.fromarray(np.full((4, 8, 4), 127, dtype=np.uint8), mode="RGBA").save(image_path)
    frame = ColmapFrame(
        image_id=1,
        image_path=image_path,
        q_wxyz=_identity_q(),
        t_xyz=np.zeros((3,), dtype=np.float32),
        fx=4.0,
        fy=2.0,
        cx=2.0,
        cy=1.0,
        width=4,
        height=2,
    )
    seen: dict[str, object] = {}

    def _fake_run(args, **_kwargs):
        source_path = Path(args[-1])
        with Image.open(source_path) as source_image:
            seen["source_size"] = source_image.size
        out_dir = Path(args[args.index("-o") + 1])
        (out_dir / source_path.with_suffix(".dds").name).write_bytes(b"dds")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(session, "_ensure_dataset_bc7_texconv", lambda: tmp_path / "texconv.exe")
    monkeypatch.setattr(session.subprocess, "run", _fake_run)

    cache_path = session._compress_dataset_frame_to_bc7_cache(frame, images_root)

    assert seen["source_size"] == (4, 2)
    assert cache_path.exists()


def test_set_training_active_accumulates_elapsed_time_on_pause(monkeypatch) -> None:
    viewer = _viewer()
    times = iter((10.0, 14.5))
    monkeypatch.setattr(session.time, "perf_counter", lambda: next(times))

    session.set_training_active(viewer, True)
    session.set_training_active(viewer, False)

    assert viewer.s.training_active is False
    assert viewer.s.training_resume_time is None
    assert viewer.s.training_elapsed_s == 4.5


def test_reinitialize_training_scene_reuses_existing_native_targets(monkeypatch) -> None:
    textures = [object(), object()]
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            trainer=SimpleNamespace(_frame_targets_native=textures),
            training_frames=[SimpleNamespace(), SimpleNamespace()],
        )
    )
    captured: list[object] = []

    monkeypatch.setattr(
        session,
        "initialize_training_scene",
        lambda viewer_obj, frame_targets_native=None: captured.append(frame_targets_native),
    )

    session.reinitialize_training_scene(viewer)

    assert captured == [textures]


def test_reset_training_runtime_releases_trainer_resources(monkeypatch) -> None:
    calls: list[object] = []
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            trainer=SimpleNamespace(release_resources=lambda preserve_frame_targets=False: calls.append(("release", preserve_frame_targets))),
            training_active=True,
            training_elapsed_s=12.0,
            training_resume_time=3.0,
            renderer=None,
            applied_renderer_params_training="training",
            applied_renderer_params_debug="debug",
            applied_training_signature="sig",
            applied_training_runtime_signature="runtime",
            applied_training_runtime_factor=2,
            cached_training_setup_signature="cached-sig",
            cached_training_setup="cached",
            pending_training_runtime_resize=True,
        )
    )
    monkeypatch.setattr(session, "_reset_training_visual_state", lambda viewer_obj: calls.append(("visual", viewer_obj)))
    monkeypatch.setattr(session, "_reset_loss_debug", lambda viewer_obj: calls.append(("loss", viewer_obj)))
    monkeypatch.setattr(session, "_clear", lambda viewer_obj, *attrs: calls.append(("clear", attrs)))

    session._reset_training_runtime(viewer, preserve_frame_targets=True)

    assert calls[0] == ("release", True)
    assert calls[1][0] == "visual"
    assert calls[2][0] == "loss"
    assert calls[3] == ("clear", ("training_renderer",))
    assert viewer.s.trainer is None
    assert viewer.s.training_active is False
    assert viewer.s.training_elapsed_s == 0.0
    assert viewer.s.training_resume_time is None
    assert viewer.s.applied_renderer_params_training is None
    assert viewer.s.applied_renderer_params_debug is None
    assert viewer.s.applied_training_signature is None
    assert viewer.s.applied_training_runtime_signature is None
    assert viewer.s.applied_training_runtime_factor is None
    assert viewer.s.cached_training_setup_signature is None
    assert viewer.s.cached_training_setup is None
    assert viewer.s.pending_training_runtime_resize is False


def test_reset_training_runtime_clears_all_training_debug_bindings(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class _BoundRenderer:
        def __init__(self, prefix: str) -> None:
            self.prefix = prefix

        def set_debug_grad_norm_buffer(self, buffer) -> None:
            calls.append((f"{self.prefix}_grad_norm", buffer))

        def set_debug_grad_stats_buffer(self, buffer) -> None:
            calls.append((f"{self.prefix}_grad_stats", buffer))

        def set_debug_splat_age_buffer(self, buffer) -> None:
            calls.append((f"{self.prefix}_splat_age", buffer))

        def set_debug_splat_contribution_buffer(self, buffer) -> None:
            calls.append((f"{self.prefix}_contribution", buffer))

        def set_debug_adam_moments_buffer(self, buffer) -> None:
            calls.append((f"{self.prefix}_adam", buffer))

        def set_debug_contribution_observed_pixel_count(self, value) -> None:
            calls.append((f"{self.prefix}_pixels", value))

    viewer = SimpleNamespace(
        s=SimpleNamespace(
            trainer=SimpleNamespace(release_resources=lambda preserve_frame_targets=False: calls.append(("release", preserve_frame_targets))),
            training_active=True,
            training_elapsed_s=5.0,
            training_resume_time=1.0,
            renderer=_BoundRenderer("main"),
            debug_renderer=_BoundRenderer("debug"),
            applied_renderer_params_training="training",
            applied_renderer_params_debug="debug",
            applied_training_signature="sig",
            applied_training_runtime_signature="runtime",
            applied_training_runtime_factor=2,
            cached_training_setup_signature="cached-sig",
            cached_training_setup="cached",
            pending_training_runtime_resize=True,
        )
    )
    monkeypatch.setattr(session, "_reset_training_visual_state", lambda viewer_obj: calls.append(("visual", viewer_obj)))
    monkeypatch.setattr(session, "_reset_loss_debug", lambda viewer_obj: calls.append(("loss", viewer_obj)))
    monkeypatch.setattr(session, "_clear", lambda viewer_obj, *attrs: calls.append(("clear", attrs)))

    session._reset_training_runtime(viewer)

    assert ("main_grad_norm", None) in calls
    assert ("main_grad_stats", None) in calls
    assert ("main_splat_age", None) in calls
    assert ("main_contribution", None) in calls
    assert ("main_adam", None) in calls
    assert ("main_pixels", 0) in calls
    assert ("debug_grad_norm", None) in calls
    assert ("debug_grad_stats", None) in calls
    assert ("debug_splat_age", None) in calls
    assert ("debug_contribution", None) in calls
    assert ("debug_adam", None) in calls
    assert ("debug_pixels", 0) in calls


def test_move_main_camera_to_selected_training_frame_applies_pose_and_exits_mode() -> None:
    calls: list[tuple[str, object]] = []
    pose = SimpleNamespace(
        position=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        target=np.array([1.0, 2.0, 4.0], dtype=np.float32),
        up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        near=0.25,
        far=80.0,
    )
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={"show_training_cameras": True, "loss_debug_frame": 1}),
        c=lambda _key: SimpleNamespace(value=1),
        s=SimpleNamespace(
            near=0.1,
            far=120.0,
            training_frames=[SimpleNamespace(width=320, height=180), SimpleNamespace(width=640, height=360)],
            trainer=SimpleNamespace(
                frame_size=lambda frame_idx: (640, 360) if int(frame_idx) == 1 else (320, 180),
                make_frame_camera=lambda frame_idx, width, height: calls.append(("make", (int(frame_idx), int(width), int(height)))) or pose,
            ),
        ),
        apply_camera_pose=lambda camera, **kwargs: calls.append(("apply", (camera, kwargs))),
    )

    frame_idx = session.move_main_camera_to_selected_training_frame(viewer)

    assert frame_idx == 1
    assert calls == [
        ("make", (1, 640, 360)),
        ("apply", (pose, {})),
    ]
    assert viewer.ui._values["show_training_cameras"] is False


def test_reset_training_runtime_waits_and_drains_deferred_resources(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    viewer = SimpleNamespace(
        device=SimpleNamespace(wait=lambda: calls.append(("wait", None))),
        s=SimpleNamespace(
            trainer=SimpleNamespace(release_resources=lambda preserve_frame_targets=False: calls.append(("release", preserve_frame_targets))),
            training_active=True,
            training_elapsed_s=1.0,
            training_resume_time=2.0,
            renderer=None,
            debug_renderer=None,
            applied_renderer_params_training="training",
            applied_renderer_params_debug="debug",
            applied_training_signature="sig",
            applied_training_runtime_signature="runtime",
            applied_training_runtime_factor=2,
            cached_training_setup_signature="cached-sig",
            cached_training_setup="cached",
            pending_training_runtime_resize=True,
        ),
    )
    monkeypatch.setattr(session, "_reset_training_visual_state", lambda viewer_obj: calls.append(("visual", viewer_obj)))
    monkeypatch.setattr(session, "_reset_loss_debug", lambda viewer_obj: calls.append(("loss", viewer_obj)))
    monkeypatch.setattr(session, "_clear", lambda viewer_obj, *attrs: calls.append(("clear", attrs)))
    monkeypatch.setattr(session, "drain_all_deferred_resource_releases", lambda min_age=0, advance_generation=False: calls.append(("drain", (min_age, advance_generation))) or (0, 0))

    session._reset_training_runtime(viewer)

    assert ("wait", None) in calls
    assert ("drain", (0, False)) in calls


def test_clear_loaded_scene_waits_and_drains_deferred_resources(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    viewer = SimpleNamespace(
        device=SimpleNamespace(wait=lambda: calls.append(("wait", None))),
        s=SimpleNamespace(
            scene=object(),
            scene_path="scene.ply",
            colmap_root="dataset",
            colmap_recon=object(),
            training_frames=[object()],
            renderer=SimpleNamespace(clear_scene_resources=lambda: calls.append(("clear_scene_resources", None))),
        ),
    )
    monkeypatch.setattr(session, "_reset_loaded_runtime", lambda viewer_obj: calls.append(("reset_loaded_runtime", viewer_obj)))
    monkeypatch.setattr(session, "drain_all_deferred_resource_releases", lambda min_age=0, advance_generation=False: calls.append(("drain", (min_age, advance_generation))) or (0, 0))

    session._clear_loaded_scene(viewer)

    assert calls[0][0] == "reset_loaded_runtime"
    assert ("clear_scene_resources", None) in calls
    assert ("wait", None) in calls
    assert ("drain", (0, False)) in calls
    assert viewer.s.scene is None
    assert viewer.s.scene_path is None
    assert viewer.s.colmap_root is None
    assert viewer.s.colmap_recon is None
    assert viewer.s.training_frames == []


def test_training_elapsed_seconds_includes_current_active_segment() -> None:
    viewer = _viewer()
    viewer.s.training_active = True
    viewer.s.training_elapsed_s = 12.0
    viewer.s.training_resume_time = 20.0

    assert session.training_elapsed_seconds(viewer, now=27.5) == 19.5


def test_ensure_training_runtime_resolution_rebinds_renderer_without_reset(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class _Encoder:
        def finish(self) -> str:
            return "finished"

    class _OldRenderer:
        width = 64
        height = 64

        def copy_scene_state_to(self, encoder, dst) -> None:
            calls.append(("copy", dst))

        def clear_scene_resources(self) -> None:
            calls.append(("clear", None))

    class _MainRenderer:
        def __init__(self) -> None:
            self.bound = None
            self.grad_stats_bound = None
            self.contribution_bound = None
            self.contribution_pixels = None

        def set_debug_grad_norm_buffer(self, buffer) -> None:
            self.bound = buffer

        def set_debug_grad_stats_buffer(self, buffer) -> None:
            self.grad_stats_bound = buffer

        def set_debug_splat_age_buffer(self, buffer) -> None:
            del buffer

        def set_debug_splat_contribution_buffer(self, buffer) -> None:
            self.contribution_bound = buffer

        def set_debug_contribution_observed_pixel_count(self, value) -> None:
            self.contribution_pixels = value

    new_renderer = SimpleNamespace(width=32, height=32, work_buffers={"debug_grad_norm": "grad_norm"})
    trainer = SimpleNamespace(
        compute_debug_grad_norm=True,
        refinement_buffers={"splat_contribution": "contrib", "gradient_stats": "grad_stats"},
        observed_contribution_pixel_count=2048,
        effective_train_downscale_factor=lambda: 2,
        max_training_resolution=lambda: (32, 32),
        training_resolution=lambda frame_index=0: (32, 32),
        rebind_renderer=lambda renderer: calls.append(("rebind", renderer)),
    )
    viewer = SimpleNamespace(
        device=SimpleNamespace(
            create_command_encoder=lambda: _Encoder(),
            submit_command_buffer=lambda command_buffer: calls.append(("submit", command_buffer)),
        ),
        s=SimpleNamespace(
            trainer=trainer,
            training_renderer=_OldRenderer(),
            training_frames=[SimpleNamespace(width=64, height=64)],
            renderer=_MainRenderer(),
            debug_renderer=None,
            applied_training_runtime_factor=None,
            pending_training_runtime_resize=False,
        ),
    )

    monkeypatch.setattr(session, "_create_renderer", lambda viewer_obj, width, height, allow_debug_overlays: new_renderer)
    monkeypatch.setattr(session, "_invalidate", lambda viewer_obj, *targets: calls.append(("invalidate", targets)))
    monkeypatch.setattr(session, "_reset_loss_debug", lambda viewer_obj: calls.append(("reset_loss_debug", None)))

    session.ensure_training_runtime_resolution(viewer)

    assert viewer.s.training_renderer is new_renderer
    assert viewer.s.renderer.bound == "grad_norm"
    assert viewer.s.renderer.grad_stats_bound == "grad_stats"
    assert viewer.s.renderer.contribution_bound == "contrib"
    assert viewer.s.renderer.contribution_pixels == 2048
    assert calls == [
        ("copy", new_renderer),
        ("submit", "finished"),
        ("rebind", new_renderer),
        ("invalidate", ()),
        ("reset_loss_debug", None),
        ("clear", None),
    ]


def test_ensure_training_runtime_resolution_clears_pending_flag_without_realloc(monkeypatch) -> None:
    update_calls: list[tuple[object, object, object]] = []
    replace_calls: list[tuple[int, int]] = []

    params = SimpleNamespace(
        adam=SimpleNamespace(__dataclass_fields__={"lr": None}, lr=1e-3),
        stability=SimpleNamespace(__dataclass_fields__={"eps": None}, eps=1e-8),
        training=SimpleNamespace(
            __dataclass_fields__={
                "max_sh_band": None,
                "train_downscale_mode": None,
                "train_auto_start_downscale": None,
                "train_downscale_base_iters": None,
                "train_downscale_iter_step": None,
                "train_downscale_max_iters": None,
                "train_downscale_factor": None,
                "train_subsample_factor": None,
            },
            max_sh_band=3,
            train_downscale_mode=1,
            train_auto_start_downscale=1,
            train_downscale_base_iters=200,
            train_downscale_iter_step=50,
            train_downscale_max_iters=30_000,
            train_downscale_factor=1,
            train_subsample_factor=1,
        ),
    )
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            trainer=SimpleNamespace(
                update_hyperparams=lambda adam, stability, training: update_calls.append((adam, stability, training)),
                effective_train_render_factor=lambda: 2,
                max_training_resolution=lambda: (32, 32),
            ),
            training_renderer=SimpleNamespace(width=32, height=32),
            training_frames=[SimpleNamespace(width=64, height=64)],
            applied_training_signature=session._training_live_params_signature(params),
            applied_training_runtime_signature=session._training_runtime_signature(params),
            applied_training_runtime_factor=2,
            pending_training_runtime_resize=True,
        ),
    )

    monkeypatch.setattr(session, "resolve_effective_training_setup", lambda viewer_obj: (None, params, None, None))
    monkeypatch.setattr(session, "_replace_training_renderer", lambda viewer_obj, width, height, reset_loss_debug=True: replace_calls.append((width, height)))

    changed = session.ensure_training_runtime_resolution(viewer)

    assert changed is False
    assert update_calls == []
    assert replace_calls == []
    assert viewer.s.pending_training_runtime_resize is False


def test_ensure_training_runtime_resolution_replaces_same_size_renderer_for_runtime_change(monkeypatch) -> None:
    update_calls: list[tuple[object, object, object]] = []
    replace_calls: list[tuple[int, int]] = []

    params_before = SimpleNamespace(
        adam=SimpleNamespace(__dataclass_fields__={"lr": None}, lr=1e-3),
        stability=SimpleNamespace(__dataclass_fields__={"eps": None}, eps=1e-8),
        training=SimpleNamespace(
            __dataclass_fields__={
                "max_sh_band": None,
                "train_downscale_mode": None,
                "train_auto_start_downscale": None,
                "train_downscale_base_iters": None,
                "train_downscale_iter_step": None,
                "train_downscale_max_iters": None,
                "train_downscale_factor": None,
                "train_subsample_factor": None,
            },
            max_sh_band=1,
            train_downscale_mode=1,
            train_auto_start_downscale=1,
            train_downscale_base_iters=200,
            train_downscale_iter_step=50,
            train_downscale_max_iters=30_000,
            train_downscale_factor=1,
            train_subsample_factor=1,
        ),
    )
    params_after = SimpleNamespace(
        adam=params_before.adam,
        stability=params_before.stability,
        training=SimpleNamespace(
            __dataclass_fields__=params_before.training.__dataclass_fields__,
            max_sh_band=3,
            train_downscale_mode=1,
            train_auto_start_downscale=1,
            train_downscale_base_iters=200,
            train_downscale_iter_step=50,
            train_downscale_max_iters=30_000,
            train_downscale_factor=1,
            train_subsample_factor=1,
        ),
    )
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            trainer=SimpleNamespace(
                update_hyperparams=lambda adam, stability, training: update_calls.append((adam, stability, training)),
                effective_train_render_factor=lambda: 2,
                max_training_resolution=lambda: (32, 32),
            ),
            training_renderer=SimpleNamespace(width=32, height=32),
            training_frames=[SimpleNamespace(width=64, height=64)],
            applied_training_signature=session._training_live_params_signature(params_before),
            applied_training_runtime_signature=session._training_runtime_signature(params_before),
            applied_training_runtime_factor=2,
            pending_training_runtime_resize=True,
        ),
    )

    monkeypatch.setattr(session, "resolve_effective_training_setup", lambda viewer_obj: (None, params_after, None, None))
    monkeypatch.setattr(session, "_replace_training_renderer", lambda viewer_obj, width, height, reset_loss_debug=True: replace_calls.append((width, height)))

    changed = session.ensure_training_runtime_resolution(viewer)

    assert changed is True
    assert update_calls == [(params_after.adam, params_after.stability, params_after.training)]
    assert replace_calls == [(32, 32)]
    assert viewer.s.applied_training_signature == session._training_live_params_signature(params_after)
    assert viewer.s.applied_training_runtime_signature == session._training_runtime_signature(params_after)
    assert viewer.s.pending_training_runtime_resize is False


def test_ensure_renderer_clears_replaced_renderer_resources(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    previous_renderer = SimpleNamespace(width=64, height=64, clear_scene_resources=lambda: calls.append(("clear", None)))
    new_renderer = SimpleNamespace(width=128, height=72)
    viewer = SimpleNamespace(
        device=SimpleNamespace(),
        renderer_params=lambda allow_debug_overlays: _renderer_params(),
        s=SimpleNamespace(renderer=previous_renderer, scene=None),
    )

    class _Settings:
        @classmethod
        def from_renderer_params(cls, width: int, height: int, params: object):
            del params
            return cls(width, height)

        def __init__(self, width: int, height: int) -> None:
            self.width = width
            self.height = height

        def create_renderer(self, device) -> object:
            del device
            return new_renderer

    monkeypatch.setattr(session, "GaussianRenderSettings", _Settings)
    monkeypatch.setattr(session, "_apply_debug_buffers", lambda viewer_obj, renderer: calls.append(("debug", renderer)))
    monkeypatch.setattr(session, "_invalidate", lambda viewer_obj, *targets: calls.append(("invalidate", targets)))

    result = session.ensure_renderer(viewer, "renderer", 128, 72, allow_debug_overlays=True)

    assert result is new_renderer
    assert viewer.s.renderer is new_renderer
    assert calls == [
        ("debug", new_renderer),
        ("clear", None),
        ("invalidate", ("main", "debug")),
    ]


def test_create_renderer_applies_training_sh_cap_when_trainer_exists(monkeypatch) -> None:
    new_renderer = SimpleNamespace(max_sh_band=3)
    params = SimpleNamespace(training=SimpleNamespace(max_sh_band=1))
    viewer = SimpleNamespace(
        device=SimpleNamespace(),
        renderer_params=lambda allow_debug_overlays: _renderer_params(debug=bool(allow_debug_overlays)),
        s=SimpleNamespace(trainer=SimpleNamespace()),
    )

    class _Settings:
        @classmethod
        def from_renderer_params(cls, width: int, height: int, params_obj: object):
            del width, height, params_obj
            return cls()

        def create_renderer(self, device) -> object:
            del device
            return new_renderer

    monkeypatch.setattr(session, "GaussianRenderSettings", _Settings)
    monkeypatch.setattr(session, "resolve_effective_training_setup", lambda viewer_obj: (None, params, None, None))

    result = session._create_renderer(viewer, 128, 72, allow_debug_overlays=True)

    assert result is new_renderer
    assert result.max_sh_band == 1


def test_clear_releases_renderer_scene_resources() -> None:
    calls: list[str] = []
    renderer = SimpleNamespace(clear_scene_resources=lambda: calls.append("clear"))
    viewer = SimpleNamespace(s=SimpleNamespace(training_renderer=renderer, debug_renderer=None))

    session._clear(viewer, "training_renderer")

    assert viewer.s.training_renderer is None
    assert calls == ["clear"]


def test_ensure_renderer_keeps_existing_main_renderer_when_replacement_fails(monkeypatch) -> None:
    existing_renderer = SimpleNamespace(width=64, height=64)
    viewer = SimpleNamespace(
        device=SimpleNamespace(),
        renderer_params=lambda allow_debug_overlays: _renderer_params(),
        s=SimpleNamespace(renderer=existing_renderer, scene=None),
    )

    class _FailingSettings:
        @classmethod
        def from_renderer_params(cls, width: int, height: int, params: object):
            del width, height, params
            return cls()

        def create_renderer(self, device) -> object:
            del device
            raise RuntimeError("renderer create failed")

    monkeypatch.setattr(session, "GaussianRenderSettings", _FailingSettings)

    with pytest.raises(RuntimeError, match="renderer create failed"):
        session.ensure_renderer(viewer, "renderer", 128, 72, allow_debug_overlays=True)

    assert viewer.s.renderer is existing_renderer


def test_maybe_reallocate_renderers_recycles_live_renderers_at_interval(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    current_time = session._PERIODIC_RENDERER_REALLOCATION_INTERVAL_S + 1.0
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            renderer=SimpleNamespace(width=128, height=72),
            debug_renderer=SimpleNamespace(width=96, height=54),
            trainer=SimpleNamespace(),
            training_renderer=SimpleNamespace(width=64, height=32, _render_capacity_width=80, _render_capacity_height=40),
            last_periodic_renderer_reallocation_time=0.0,
        )
    )

    monkeypatch.setattr(session, "recreate_renderer", lambda viewer_obj, width, height: calls.append(("main", (width, height))))
    monkeypatch.setattr(
        session,
        "ensure_renderer",
        lambda viewer_obj, attr, width, height, allow_debug_overlays, force_recreate=False: calls.append(("debug", (attr, width, height, allow_debug_overlays, force_recreate))),
    )
    monkeypatch.setattr(
        session,
        "_replace_training_renderer",
        lambda viewer_obj, width, height, reset_loss_debug=True: calls.append(("training", (width, height, reset_loss_debug))),
    )

    recycled = session.maybe_reallocate_renderers(viewer, 128, 72, current_time)

    assert recycled is True
    assert calls == [
        ("main", (128, 72)),
        ("debug", ("debug_renderer", 96, 54, True, True)),
        ("training", (80, 40, False)),
    ]
    assert viewer.s.last_periodic_renderer_reallocation_time == pytest.approx(current_time)


def test_rebind_renderer_preserves_refinement_history() -> None:
    calls: list[tuple[str, object]] = []
    renderer = SimpleNamespace(width=80, height=40, _render_capacity_width=80, _render_capacity_height=40)
    trainer = SimpleNamespace(
        renderer=None,
        optimizer=SimpleNamespace(renderer=None),
        training=SimpleNamespace(train_downscale_factor=1),
        state=SimpleNamespace(step=7),
        _scene_count=32,
        _refinement_splat_capacity=32,
        _max_training_resolution=lambda _step: (80, 40),
        effective_train_downscale_factor=lambda _step: 1,
        _ensure_training_buffers=lambda splat_count, batch_step_count=1: calls.append(("training_buffers", (splat_count, batch_step_count))),
        _ensure_refinement_buffers=lambda splat_count: calls.append(("refinement_buffers", splat_count)),
        _ensure_train_target_texture=lambda: calls.append(("target", None)),
        _invalidate_downscaled_target=lambda: calls.append(("invalidate", None)),
        apply_renderer_training_hparams=lambda step=None, renderer=None: calls.append(("params", (step, renderer))),
        _clear_clone_counts=lambda preserve_refinement_history=False: calls.append(("clear", preserve_refinement_history)),
    )

    session.GaussianTrainer.rebind_renderer(trainer, renderer)

    assert trainer.renderer is renderer
    assert trainer.optimizer.renderer is renderer
    assert trainer._dynamic_frame_resolution is True
    assert trainer.training.train_downscale_factor == 1
    assert calls == [
        ("params", (None, renderer)),
        ("training_buffers", (32, 1)),
        ("refinement_buffers", 32),
        ("clear", True),
        ("target", None),
        ("invalidate", None),
    ]


def test_choose_colmap_root_auto_selects_first_matching_image_folder(tmp_path: Path) -> None:
    database_path, images_root = _build_colmap_tree(
        tmp_path,
        image_names=["frame_000.png", "frame_001.png"],
        image_root_rel=Path("input"),
    )
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={}),
        s=SimpleNamespace(last_error="stale"),
    )

    session.choose_colmap_root(viewer, database_path.parents[1])

    assert viewer.ui._values["colmap_root_path"] == str(database_path.resolve().parents[1])
    assert viewer.ui._values["colmap_database_path"] == str(database_path.resolve())
    assert viewer.ui._values["colmap_images_root"] == str(images_root)
    assert viewer.ui._values["colmap_selected_camera_ids"] == (7,)
    assert viewer.ui._values["_colmap_camera_rows"] == (
        {
            "camera_id": 7,
            "model_name": "PINHOLE",
            "frame_count": 2,
            "resolution_text": "64x64",
            "focal_text": "64.00, 64.00",
            "principal_text": "32.00, 32.00",
            "distortion_text": "0, 0",
        },
    )
    assert viewer.s.last_error == ""


def test_choose_colmap_root_skips_depth_named_image_directories(tmp_path: Path) -> None:
    database_path, _ = _build_colmap_tree(
        tmp_path,
        image_names=["frame_000.png", "frame_001.png"],
        image_root_rel=Path("rgb"),
    )
    depth_like = database_path.parents[1] / "depth_images"
    depth_like.mkdir(parents=True, exist_ok=True)
    for image_name in ("frame_000.png", "frame_001.png"):
        Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(depth_like / image_name)
    viewer = SimpleNamespace(ui=SimpleNamespace(_values={}), s=SimpleNamespace(last_error="stale"))

    session.choose_colmap_root(viewer, database_path.parents[1])

    assert viewer.ui._values["colmap_images_root"].endswith("rgb")


def test_choose_colmap_root_keeps_dataset_root_for_relative_image_subdirs(tmp_path: Path) -> None:
    database_path, _ = _build_colmap_tree(
        tmp_path,
        image_names=["images/frame_000.png", "images/frame_001.png"],
        image_root_rel=Path("."),
    )
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={}),
        s=SimpleNamespace(last_error="stale"),
    )

    session.choose_colmap_root(viewer, database_path.parents[1])

    assert viewer.ui._values["colmap_images_root"] == str(database_path.resolve().parents[1])


def test_choose_colmap_root_works_without_database(tmp_path: Path) -> None:
    root, images_root = _build_colmap_tree_without_database(
        tmp_path,
        image_names=["images_4/frame_000.png", "images_4/frame_001.png"],
        image_root_rel=Path("."),
    )
    sparse_zero = root / "sparse" / "0"
    for path in tuple(sparse_zero.iterdir()):
        path.replace(root / "sparse" / path.name)
    sparse_zero.rmdir()
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={}),
        s=SimpleNamespace(last_error="stale"),
    )

    session.choose_colmap_root(viewer, root)

    assert viewer.ui._values["colmap_root_path"] == str(root)
    assert viewer.ui._values["colmap_database_path"] == ""
    assert viewer.ui._values["colmap_images_root"] == str(root)
    assert viewer.s.last_error == ""


@pytest.mark.parametrize(("model_id", "model_name"), ((4, "OPENCV"), (6, "FULL_OPENCV")))
def test_choose_colmap_root_supports_opencv_camera_models(tmp_path: Path, model_id: int, model_name: str) -> None:
    database_path, images_root = _build_colmap_tree(
        tmp_path,
        image_names=["frame_000.png"],
        image_root_rel=Path("images"),
        model_id=model_id,
    )
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={}),
        s=SimpleNamespace(last_error="stale"),
    )

    session.choose_colmap_root(viewer, database_path.parents[1])

    assert viewer.ui._values["colmap_root_path"] == str(database_path.resolve().parents[1])
    assert viewer.ui._values["colmap_database_path"] == str(database_path.resolve())
    assert viewer.ui._values["colmap_images_root"] == str(images_root)
    assert viewer.ui._values["colmap_selected_camera_ids"] == (7,)
    assert viewer.ui._values["_colmap_camera_rows"][0]["model_name"] == model_name
    assert viewer.s.last_error == ""


def test_import_colmap_dataset_clears_loaded_scene_before_loading(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            renderer=SimpleNamespace(
                clear_scene_resources=lambda: calls.append(("clear_renderer", None)),
                set_debug_grad_norm_buffer=lambda buffer: calls.append(("clear_grad_debug", buffer)),
                set_debug_splat_age_buffer=lambda buffer: calls.append(("clear_splat_age_debug", buffer)),
            ),
            trainer=SimpleNamespace(state=SimpleNamespace(step=0)),
            training_active=False,
            training_elapsed_s=0.0,
            training_resume_time=None,
            training_renderer=None,
            training_frames=[SimpleNamespace(width=8, height=8)],
            scene=SimpleNamespace(count=123),
            scene_path=Path("scene.ply"),
            colmap_root=Path("dataset/old"),
            colmap_recon=object(),
            colmap_import_progress=None,
            applied_renderer_params_training=None,
            applied_renderer_params_debug=None,
            applied_training_signature=None,
            applied_training_runtime_factor=None,
            pending_training_runtime_resize=False,
            applied_renderer_params_main=None,
            cached_training_setup_signature=None,
            cached_training_setup=None,
        ),
        c=lambda key: SimpleNamespace(value=0),
    )

    monkeypatch.setattr(session, "update_debug_frame_slider_range", lambda viewer_obj: calls.append(("update_slider", None)))
    monkeypatch.setattr(session, "_clear_cached_init_source", lambda viewer_obj: calls.append(("clear_cached_init", None)))
    monkeypatch.setattr(session, "_reset_training_visual_state", lambda viewer_obj: calls.append(("reset_training_visual", None)))
    monkeypatch.setattr(session, "_reset_loss_debug", lambda viewer_obj: calls.append(("reset_loss_debug", None)))
    monkeypatch.setattr(session, "_load_aligned_colmap_reconstruction", lambda root, auto_rotate_scene=True: calls.append(("load_recon", root, auto_rotate_scene)) or "recon")
    monkeypatch.setattr(
        session,
        "build_training_frames_from_root",
        lambda recon, images_root, selected_camera_ids=(), downscale_mode="original", downscale_max_size=None, downscale_scale=1.0: calls.append(("build_frames", recon, tuple(selected_camera_ids))) or ["frame"],
    )
    monkeypatch.setattr(session, "_create_native_dataset_textures", lambda viewer_obj, frames: calls.append(("create_textures", list(frames))) or ["tex"])
    monkeypatch.setattr(session, "_finish_import_colmap_dataset", lambda viewer_obj, **kwargs: calls.append(("finish", kwargs["recon"])))

    session.import_colmap_dataset(
        viewer,
        colmap_root=Path("dataset/new"),
        database_path=None,
        images_root=Path("dataset/new/images"),
        init_mode="pointcloud",
        custom_ply_path=None,
        image_downscale_mode="original",
        image_downscale_max_size=2048,
        image_downscale_scale=1.0,
        nn_radius_scale_coef=0.5,
        min_track_length=3,
        diffused_point_count=100000,
    )

    assert calls[:8] == [
        ("clear_grad_debug", None),
        ("clear_splat_age_debug", None),
        ("reset_training_visual", None),
        ("reset_loss_debug", None),
        ("clear_cached_init", None),
        ("update_slider", None),
        ("clear_renderer", None),
        ("load_recon", Path("dataset/new").resolve(), True),
    ]
    assert viewer.s.scene is None
    assert viewer.s.scene_path is None
    assert viewer.s.colmap_root is None
    assert viewer.s.colmap_recon is None
    assert viewer.s.training_frames == []


def test_import_colmap_from_ui_clears_loaded_scene_before_queueing(tmp_path: Path, monkeypatch) -> None:
    database_path, images_root = _build_colmap_tree(
        tmp_path,
        image_names=["frame_000.png"],
        image_root_rel=Path("images"),
    )
    calls: list[tuple[str, object]] = []
    viewer = SimpleNamespace(
        ui=SimpleNamespace(
            _values={
                "colmap_root_path": str(database_path.parents[1]),
                "colmap_database_path": str(database_path),
                "colmap_images_root": str(images_root),
                "colmap_depth_value_mode": 1,
                "colmap_init_mode": 0,
                "colmap_custom_ply_path": "",
                "colmap_image_downscale_mode": 0,
                "colmap_image_max_size": 2048,
                "colmap_image_scale": 1.0,
                "colmap_nn_radius_scale_coef": 0.5,
                "colmap_min_track_length": 5,
                "colmap_diffused_point_count": 100000,
                "colmap_pointcloud_enabled": True,
                "colmap_selected_camera_ids": (7,),
                "_colmap_camera_rows": ({"camera_id": 7, "frame_count": 1},),
                "use_target_alpha_mask": True,
            }
        ),
        s=SimpleNamespace(
            renderer=SimpleNamespace(
                clear_scene_resources=lambda: calls.append(("clear_renderer", None)),
                set_debug_grad_norm_buffer=lambda buffer: calls.append(("clear_grad_debug", buffer)),
                set_debug_splat_age_buffer=lambda buffer: calls.append(("clear_splat_age_debug", buffer)),
            ),
            trainer=None,
            training_active=False,
            training_elapsed_s=0.0,
            training_resume_time=None,
            training_renderer=None,
            training_frames=[SimpleNamespace(width=8, height=8)],
            scene=SimpleNamespace(count=123),
            scene_path=Path("scene.ply"),
            colmap_root=Path("dataset/old"),
            colmap_recon=object(),
            colmap_import_progress=None,
            applied_renderer_params_training=None,
            applied_renderer_params_debug=None,
            applied_training_signature=None,
            applied_training_runtime_factor=None,
            pending_training_runtime_resize=False,
            applied_renderer_params_main=None,
            cached_training_setup_signature=None,
            cached_training_setup=None,
            last_error="stale",
        ),
        c=lambda key: SimpleNamespace(value=0),
    )

    monkeypatch.setattr(session, "update_debug_frame_slider_range", lambda viewer_obj: calls.append(("update_slider", None)))
    monkeypatch.setattr(session, "_clear_cached_init_source", lambda viewer_obj: calls.append(("clear_cached_init", None)))
    monkeypatch.setattr(session, "_reset_training_visual_state", lambda viewer_obj: calls.append(("reset_training_visual", None)))
    monkeypatch.setattr(session, "_reset_loss_debug", lambda viewer_obj: calls.append(("reset_loss_debug", None)))

    session.import_colmap_from_ui(viewer)

    assert calls == [
        ("clear_grad_debug", None),
        ("clear_splat_age_debug", None),
        ("reset_training_visual", None),
        ("reset_loss_debug", None),
        ("clear_cached_init", None),
        ("update_slider", None),
        ("clear_renderer", None),
    ]
    assert viewer.s.scene is None
    assert viewer.s.scene_path is None
    assert viewer.s.colmap_root is None
    assert viewer.s.colmap_recon is None
    assert viewer.s.training_frames == []
    assert viewer.s.colmap_import_progress is not None
    assert viewer.s.colmap_import_progress.depth_value_mode == "z_depth"
    assert viewer.s.colmap_import_progress.min_track_length == 5
    assert viewer.s.colmap_import_progress.selected_camera_ids == (7,)
    assert viewer.s.colmap_import_progress.use_target_alpha_mask is True


def test_import_colmap_from_ui_queues_custom_mesh_mode(tmp_path: Path, monkeypatch) -> None:
    database_path, images_root = _build_colmap_tree(
        tmp_path,
        image_names=["frame_000.png"],
        image_root_rel=Path("images"),
    )
    mesh_path = tmp_path / "seed.obj"
    mesh_path.write_text("o seed\n", encoding="utf-8")
    viewer = SimpleNamespace(
        ui=SimpleNamespace(
            _values={
                "colmap_root_path": str(database_path.parents[1]),
                "colmap_database_path": str(database_path),
                "colmap_images_root": str(images_root),
                "colmap_depth_value_mode": 1,
                "colmap_init_mode": 0,
                "colmap_custom_ply_path": "",
                "colmap_custom_mesh_path": str(mesh_path),
                "colmap_image_downscale_mode": 0,
                "colmap_image_max_size": 2048,
                "colmap_image_scale": 1.0,
                "colmap_auto_rotate_scene": False,
                "colmap_nn_radius_scale_coef": 0.5,
                "colmap_min_track_length": 5,
                "colmap_diffused_point_count": 4096,
                "colmap_custom_mesh_enabled": True,
                "colmap_selected_camera_ids": (7,),
                "_colmap_camera_rows": ({"camera_id": 7, "frame_count": 1},),
                "use_target_alpha_mask": False,
            }
        ),
        s=SimpleNamespace(
            renderer=SimpleNamespace(
                clear_scene_resources=lambda: None,
                set_debug_grad_norm_buffer=lambda buffer: None,
                set_debug_splat_age_buffer=lambda buffer: None,
            ),
            trainer=None,
            training_active=False,
            training_elapsed_s=0.0,
            training_resume_time=None,
            training_renderer=None,
            training_frames=[],
            scene=None,
            scene_path=None,
            colmap_root=None,
            colmap_recon=None,
            colmap_import_progress=None,
            applied_renderer_params_training=None,
            applied_renderer_params_debug=None,
            applied_training_signature=None,
            applied_training_runtime_factor=None,
            pending_training_runtime_resize=False,
            applied_renderer_params_main=None,
            cached_training_setup_signature=None,
            cached_training_setup=None,
            last_error="",
        ),
        c=lambda key: SimpleNamespace(value=0),
    )

    monkeypatch.setattr(session, "update_debug_frame_slider_range", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_clear_cached_init_source", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_reset_training_visual_state", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_reset_loss_debug", lambda viewer_obj: None)

    session.import_colmap_from_ui(viewer)

    assert viewer.s.colmap_import_progress is not None
    assert viewer.s.colmap_import_progress.init_mode == "pointcloud"
    assert viewer.s.colmap_import_progress.auto_rotate_scene is False
    assert viewer.s.colmap_import_progress.custom_mesh_path == mesh_path.resolve()
    assert viewer.s.colmap_import_progress.diffused_point_count == 4096


def test_import_colmap_from_ui_queues_multi_source_settings(tmp_path: Path, monkeypatch) -> None:
    database_path, images_root = _build_colmap_tree(
        tmp_path,
        image_names=["frame_000.png"],
        image_root_rel=Path("images"),
    )
    ply_path = tmp_path / "seed.ply"
    ply_path.write_text("ply\n", encoding="utf-8")
    mesh_path = tmp_path / "seed.obj"
    mesh_path.write_text("o seed\n", encoding="utf-8")
    viewer = SimpleNamespace(
        ui=SimpleNamespace(
            _values={
                "colmap_root_path": str(database_path.parents[1]),
                "colmap_database_path": str(database_path),
                "colmap_images_root": str(images_root),
                "colmap_depth_value_mode": 1,
                "colmap_pointcloud_enabled": True,
                "colmap_pointcloud_nn_radius_scale_coef": 0.4,
                "colmap_diffused_enabled": True,
                "colmap_diffused_point_count": 4096,
                "colmap_diffused_diffusion_radius": 0.75,
                "colmap_diffused_nn_radius_scale_coef": 0.45,
                "colmap_custom_ply_enabled": True,
                "colmap_custom_ply_path": str(ply_path),
                "colmap_custom_ply_nn_radius_scale_coef": 1.1,
                "colmap_custom_mesh_enabled": True,
                "colmap_custom_mesh_path": str(mesh_path),
                "colmap_custom_mesh_point_count": 2048,
                "colmap_custom_mesh_nn_radius_scale_coef": 0.55,
                "colmap_fibonacci_sphere_enabled": True,
                "colmap_fibonacci_sphere_point_count": 512,
                "colmap_fibonacci_sphere_radius": 12.0,
                "colmap_fibonacci_sphere_nn_radius_scale_coef": 1.2,
                "colmap_image_downscale_mode": 0,
                "colmap_image_max_size": 2048,
                "colmap_image_scale": 1.0,
                "colmap_auto_rotate_scene": True,
                "colmap_selected_camera_ids": (),
                "_colmap_camera_rows": (),
                "use_target_alpha_mask": False,
                "compress_dataset_using_bc7": False,
                "colmap_init_mode": 0,
                "colmap_nn_radius_scale_coef": 0.5,
                "colmap_min_track_length": 3,
            }
        ),
        s=SimpleNamespace(
            renderer=SimpleNamespace(
                clear_scene_resources=lambda: None,
                set_debug_grad_norm_buffer=lambda buffer: None,
                set_debug_splat_age_buffer=lambda buffer: None,
            ),
            trainer=None,
            training_active=False,
            training_elapsed_s=0.0,
            training_resume_time=None,
            training_renderer=None,
            training_frames=[],
            scene=None,
            scene_path=None,
            colmap_root=None,
            colmap_recon=None,
            colmap_import_progress=None,
            applied_renderer_params_training=None,
            applied_renderer_params_debug=None,
            applied_training_signature=None,
            applied_training_runtime_factor=None,
            pending_training_runtime_resize=False,
            applied_renderer_params_main=None,
            cached_training_setup_signature=None,
            cached_training_setup=None,
            last_error="",
        ),
        c=lambda key: SimpleNamespace(value=0),
    )

    monkeypatch.setattr(session, "update_debug_frame_slider_range", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_clear_cached_init_source", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_reset_training_visual_state", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_reset_loss_debug", lambda viewer_obj: None)

    session.import_colmap_from_ui(viewer)

    progress = viewer.s.colmap_import_progress
    assert progress is not None
    assert progress.init_mode == "pointcloud"
    assert progress.pointcloud_enabled is True
    assert progress.pointcloud_nn_radius_scale_coef == pytest.approx(0.4)
    assert progress.diffused_enabled is True
    assert progress.diffused_point_count == 4096
    assert progress.diffused_diffusion_radius == pytest.approx(0.75)
    assert progress.custom_ply_enabled is True
    assert progress.custom_ply_path == ply_path.resolve()
    assert progress.custom_mesh_enabled is True
    assert progress.custom_mesh_path == mesh_path.resolve()
    assert progress.custom_mesh_point_count == 2048
    assert progress.fibonacci_sphere_enabled is True
    assert progress.fibonacci_sphere_point_count == 512
    assert progress.fibonacci_sphere_nn_radius_scale_coef == pytest.approx(1.2)


def test_build_initial_training_scene_combines_enabled_sources(monkeypatch) -> None:
    point_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    point_colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    ply_positions = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
    ply_scales = np.array([[1.25, -0.75, 0.5]], dtype=np.float32)
    ply_scene = GaussianScene(
        positions=ply_positions,
        scales=ply_scales,
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.ones((1,), dtype=np.float32),
        colors=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        sh_coeffs=np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32),
    )

    def _scene_from_points(positions: np.ndarray, colors: np.ndarray) -> GaussianScene:
        count = int(positions.shape[0])
        return GaussianScene(
            positions=np.asarray(positions, dtype=np.float32),
            scales=np.zeros((count, 3), dtype=np.float32),
            rotations=np.repeat(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), count, axis=0),
            opacities=np.ones((count,), dtype=np.float32),
            colors=np.asarray(colors, dtype=np.float32),
            sh_coeffs=np.repeat(np.asarray(colors, dtype=np.float32)[:, None, :], 1, axis=1),
        )

    viewer = SimpleNamespace(
        s=SimpleNamespace(
            colmap_recon=object(),
            colmap_root=Path("dataset/garden"),
            colmap_import=SimpleNamespace(
                init_mode="pointcloud",
                pointcloud_enabled=True,
                pointcloud_nn_radius_scale_coef=0.5,
                diffused_enabled=False,
                custom_ply_enabled=True,
                custom_ply_nn_radius_scale_coef=1.0,
                custom_mesh_enabled=False,
                fibonacci_sphere_enabled=False,
                min_track_length=3,
            ),
            cached_init_pointcloud_positions=point_positions,
            cached_init_pointcloud_colors=point_colors,
            cached_init_custom_ply_scene=ply_scene,
        )
    )

    monkeypatch.setattr(session, "_ensure_cached_init_source", lambda viewer_obj, init: None)
    monkeypatch.setattr(session, "_pointcloud_init_hparams_from_positions", lambda *args, **kwargs: SimpleNamespace(base_scale=0.25))
    monkeypatch.setattr(session, "initialize_scene_from_points_colors", lambda positions, colors, seed, init_hparams: _scene_from_points(positions, colors))

    scene_obj, scale_reg_reference = session._build_initial_training_scene(viewer, SimpleNamespace(seed=7), SimpleNamespace(training=SimpleNamespace(max_gaussians=0)), SimpleNamespace())

    assert scale_reg_reference is None
    assert scene_obj.count == 3
    assert np.array_equal(scene_obj.positions, np.concatenate((point_positions, ply_positions), axis=0))
    np.testing.assert_allclose(scene_obj.scales, np.concatenate((np.zeros((2, 3), dtype=np.float32), ply_scales), axis=0), atol=1e-6)


def test_import_colmap_from_ui_rejects_empty_camera_selection(tmp_path: Path) -> None:
    database_path, images_root = _build_colmap_tree(
        tmp_path,
        image_names=["frame_000.png"],
        image_root_rel=Path("images"),
    )
    viewer = SimpleNamespace(
        ui=SimpleNamespace(
            _values={
                "colmap_root_path": str(database_path.parents[1]),
                "colmap_database_path": str(database_path),
                "colmap_images_root": str(images_root),
                "colmap_depth_value_mode": 1,
                "colmap_init_mode": 0,
                "colmap_custom_ply_path": "",
                "colmap_image_downscale_mode": 0,
                "colmap_image_max_size": 2048,
                "colmap_image_scale": 1.0,
                "colmap_nn_radius_scale_coef": 0.5,
                "colmap_min_track_length": 5,
                "colmap_diffused_point_count": 100000,
                "colmap_selected_camera_ids": (),
                "_colmap_camera_rows": ({"camera_id": 7, "frame_count": 1},),
                "use_target_alpha_mask": False,
            }
        ),
        s=SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="Select at least one COLMAP camera model before importing."):
        session.import_colmap_from_ui(viewer)

def test_advance_colmap_import_processes_images_incrementally(tmp_path: Path, monkeypatch) -> None:
    _, images_root = _build_colmap_tree(
        tmp_path,
        image_names=["frame_000.png", "frame_001.png"],
        image_root_rel=Path("images"),
    )
    recon = SimpleNamespace(
        images={
            1: SimpleNamespace(name="frame_000.png", camera_id=7, q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), t_xyz=np.array([0.0, 0.0, -2.0], dtype=np.float32)),
            2: SimpleNamespace(name="frame_001.png", camera_id=8, q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), t_xyz=np.array([0.0, 0.0, -2.0], dtype=np.float32)),
        },
        cameras={
            7: SimpleNamespace(width=8, height=8, fx=64.0, fy=64.0, cx=4.0, cy=4.0),
            8: SimpleNamespace(width=8, height=8, fx=64.0, fy=64.0, cx=4.0, cy=4.0),
        },
    )
    calls: list[object] = []
    viewer = SimpleNamespace(
        toolkit=SimpleNamespace(close_colmap_import_window=lambda: calls.append("close")),
        s=SimpleNamespace(
            colmap_import_progress=ColmapImportProgress(
                dataset_root=images_root.parent,
                colmap_root=images_root.parent,
                database_path=None,
                images_root=images_root,
                init_mode="pointcloud",
                custom_ply_path=None,
                image_downscale_mode="original",
                image_downscale_max_size=1600,
                image_downscale_scale=1.0,
                nn_radius_scale_coef=0.25,
                selected_camera_ids=(7,),
            ),
        ),
    )

    monkeypatch.setattr(session, "_load_aligned_colmap_reconstruction", lambda root, auto_rotate_scene=True: recon)
    monkeypatch.setattr(session, "load_training_frame_rgba8", lambda frame: f"rgba:{Path(frame.image_path).name}")
    monkeypatch.setattr(session, "_create_native_dataset_texture_from_rgba8", lambda viewer_obj, rgba8: f"tex:{rgba8}")

    def _finish(viewer_obj, **kwargs) -> None:
        calls.append(("finish", len(kwargs["training_frames"]), list(kwargs["frame_targets_native"])))
        viewer_obj.s.colmap_import_progress = None

    monkeypatch.setattr(session, "_finish_import_colmap_dataset", _finish)

    while viewer.s.colmap_import_progress is not None:
        session.advance_colmap_import(viewer)

    assert calls == [("finish", 1, ["tex:rgba:frame_000.png"]), "close"]


def test_advance_colmap_import_applies_selected_image_downscale(tmp_path: Path, monkeypatch) -> None:
    _, images_root = _build_colmap_tree(
        tmp_path,
        image_names=["frame_000.png"],
        image_root_rel=Path("images"),
    )
    Image.fromarray(np.full((6, 12, 3), 127, dtype=np.uint8)).save(images_root / "frame_000.png")
    recon = SimpleNamespace(
        images={1: SimpleNamespace(name="frame_000.png", camera_id=7, q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), t_xyz=np.array([0.0, 0.0, -2.0], dtype=np.float32))},
        cameras={7: SimpleNamespace(width=12, height=6, fx=120.0, fy=60.0, cx=6.0, cy=3.0)},
    )
    calls: list[object] = []
    viewer = SimpleNamespace(
        toolkit=SimpleNamespace(close_colmap_import_window=lambda: calls.append("close")),
        s=SimpleNamespace(
            colmap_import_progress=ColmapImportProgress(
                dataset_root=images_root.parent,
                colmap_root=images_root.parent,
                database_path=None,
                images_root=images_root,
                init_mode="pointcloud",
                custom_ply_path=None,
                image_downscale_mode="max_size",
                image_downscale_max_size=4,
                image_downscale_scale=1.0,
                nn_radius_scale_coef=0.25,
                selected_camera_ids=(),
            ),
        ),
    )

    monkeypatch.setattr(session, "_load_aligned_colmap_reconstruction", lambda root, auto_rotate_scene=True: recon)
    monkeypatch.setattr(session, "load_training_frame_rgba8", lambda frame: calls.append((frame.width, frame.height, frame.fx, frame.fy)) or "rgba")
    monkeypatch.setattr(session, "_create_native_dataset_texture_from_rgba8", lambda viewer_obj, rgba8: calls.append(("upload", rgba8)) or "tex")

    def _finish(viewer_obj, **kwargs) -> None:
        frame = kwargs["training_frames"][0]
        calls.append((frame.width, frame.height, frame.fx, frame.fy, kwargs["frame_targets_native"]))
        viewer_obj.s.colmap_import_progress = None

    monkeypatch.setattr(session, "_finish_import_colmap_dataset", _finish)

    while viewer.s.colmap_import_progress is not None:
        session.advance_colmap_import(viewer)

    assert calls == [(4, 2, 40.0, 20.0), ("upload", "rgba"), (4, 2, 40.0, 20.0, ["tex"]), "close"]


def test_finish_import_colmap_dataset_resets_toolkit_plot_history(monkeypatch) -> None:
    recon = SimpleNamespace(points3d={1: object()})
    monkeypatch.setattr(session, "_point_tables", lambda recon_obj, min_track_length=3: (np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)))
    monkeypatch.setattr(session, "_reset_loaded_runtime", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_update_import_settings", lambda viewer_obj, **kwargs: None)
    monkeypatch.setattr(session, "apply_live_params", lambda viewer_obj: None)
    monkeypatch.setattr(session, "estimate_point_bounds", lambda xyz: xyz)
    monkeypatch.setattr(session, "initialize_training_scene", lambda viewer_obj, frame_targets_native=None: None)
    calls: list[str] = []
    viewer = SimpleNamespace(
        toolkit=SimpleNamespace(reset_plot_history=lambda: calls.append("reset")),
        ui=SimpleNamespace(_values={}),
        s=SimpleNamespace(),
        apply_camera_fit=lambda bounds: calls.append("fit"),
    )

    session._finish_import_colmap_dataset(
        viewer,
        colmap_root=Path("dataset/garden"),
        database_path=None,
        images_root=Path("dataset/garden/images"),
        init_mode="pointcloud",
        custom_ply_path=None,
        image_downscale_mode="original",
        image_downscale_max_size=1600,
        image_downscale_scale=1.0,
        nn_radius_scale_coef=0.25,
        diffused_point_count=100000,
        recon=recon,
        training_frames=[],
        frame_targets_native=None,
    )

    assert calls == ["reset", "fit"]


def test_finish_import_colmap_dataset_uses_training_camera_position_only(monkeypatch) -> None:
    recon = SimpleNamespace(points3d={1: object()})
    training_frames = [SimpleNamespace(make_camera=lambda near=0.1, far=120.0: SimpleNamespace(position=np.array([1.0, 2.0, 3.0], dtype=np.float32), target=np.array([1.0, 2.0, 4.0], dtype=np.float32), up=np.array([0.0, 1.0, 0.0], dtype=np.float32), near=near, far=far))]
    monkeypatch.setattr(session, "_point_tables", lambda recon_obj, min_track_length=3: (np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)))
    monkeypatch.setattr(session, "_reset_loaded_runtime", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_update_import_settings", lambda viewer_obj, **kwargs: None)
    monkeypatch.setattr(session, "apply_live_params", lambda viewer_obj: None)
    monkeypatch.setattr(session, "estimate_point_bounds", lambda xyz: SimpleNamespace(center=np.zeros((3,), dtype=np.float32), radius=2.0))
    monkeypatch.setattr(session, "initialize_training_scene", lambda viewer_obj, frame_targets_native=None: None)
    calls: list[object] = []
    viewer = SimpleNamespace(
        toolkit=SimpleNamespace(reset_plot_history=lambda: calls.append("reset")),
        ui=SimpleNamespace(_values={}),
        s=SimpleNamespace(fov_y=60.0, near=0.1, far=120.0),
        apply_camera_fit=lambda bounds: calls.append(("fit", bounds)),
        apply_camera_position=lambda camera, **kwargs: calls.append(("camera_position", np.asarray(camera.position, dtype=np.float32).copy(), kwargs)),
        apply_camera_pose=lambda camera, **kwargs: (_ for _ in ()).throw(AssertionError("should not apply training orientation")),
    )

    session._finish_import_colmap_dataset(
        viewer,
        colmap_root=Path("dataset/garden"),
        database_path=None,
        images_root=Path("dataset/garden/images"),
        init_mode="pointcloud",
        custom_ply_path=None,
        image_downscale_mode="original",
        image_downscale_max_size=1600,
        image_downscale_scale=1.0,
        nn_radius_scale_coef=0.25,
        diffused_point_count=100000,
        recon=recon,
        training_frames=training_frames,
        frame_targets_native=None,
    )

    assert calls[0] == "reset"
    assert calls[1][0] == "camera_position"
    np.testing.assert_allclose(calls[1][1], np.array([1.0, 2.0, 3.0], dtype=np.float32), rtol=0.0, atol=1e-6)
    assert set(calls[1][2]) == {"near", "far", "move_speed"}


def test_finish_import_colmap_dataset_falls_back_to_bounds_fit_without_training_camera(monkeypatch) -> None:
    recon = SimpleNamespace(points3d={1: object()})
    training_frames = [SimpleNamespace(make_camera=lambda near=0.1, far=120.0: (_ for _ in ()).throw(RuntimeError("bad frame")))]
    monkeypatch.setattr(session, "_point_tables", lambda recon_obj, min_track_length=3: (np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)))
    monkeypatch.setattr(session, "_reset_loaded_runtime", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_update_import_settings", lambda viewer_obj, **kwargs: None)
    monkeypatch.setattr(session, "apply_live_params", lambda viewer_obj: None)
    monkeypatch.setattr(session, "estimate_point_bounds", lambda xyz: ("bounds", xyz.shape[0]))
    monkeypatch.setattr(session, "initialize_training_scene", lambda viewer_obj, frame_targets_native=None: None)
    calls: list[object] = []
    viewer = SimpleNamespace(
        toolkit=SimpleNamespace(reset_plot_history=lambda: calls.append("reset")),
        ui=SimpleNamespace(_values={}),
        s=SimpleNamespace(fov_y=60.0, near=0.1, far=120.0),
        apply_camera_fit=lambda bounds: calls.append(("fit", bounds)),
        apply_camera_position=lambda camera, **kwargs: calls.append(("camera_position", camera, kwargs)),
    )

    session._finish_import_colmap_dataset(
        viewer,
        colmap_root=Path("dataset/garden"),
        database_path=None,
        images_root=Path("dataset/garden/images"),
        init_mode="pointcloud",
        custom_ply_path=None,
        image_downscale_mode="original",
        image_downscale_max_size=1600,
        image_downscale_scale=1.0,
        nn_radius_scale_coef=0.25,
        diffused_point_count=100000,
        recon=recon,
        training_frames=training_frames,
        frame_targets_native=None,
    )

    assert calls == ["reset", ("fit", ("bounds", 1))]


def test_finish_import_colmap_dataset_seeds_pointcloud_cached_init_source(monkeypatch) -> None:
    recon = SimpleNamespace(points3d={1: object()})
    expected_positions = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    expected_colors = np.array([[0.25, 0.5, 0.75]], dtype=np.float32)
    calls: list[object] = []
    viewer = SimpleNamespace(
        toolkit=SimpleNamespace(reset_plot_history=lambda: calls.append("reset")),
        ui=SimpleNamespace(_values={}),
        s=SimpleNamespace(
            cached_init_point_positions=None,
            cached_init_point_colors=None,
            cached_init_signature=None,
        ),
        init_params=lambda: SimpleNamespace(seed=7),
        apply_camera_fit=lambda bounds: calls.append(("fit", bounds)),
    )

    monkeypatch.setattr(session, "_point_tables", lambda recon_obj, min_track_length=3: (np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)))
    monkeypatch.setattr(session, "_reset_loaded_runtime", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_reset_training_visual_state", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_update_import_settings", lambda viewer_obj, **kwargs: setattr(viewer_obj.s, "colmap_import", SimpleNamespace(**kwargs)))
    monkeypatch.setattr(session, "_set_colmap_camera_preview", lambda viewer_obj, recon_obj, camera_ids: None)
    monkeypatch.setattr(session, "apply_live_params", lambda viewer_obj: calls.append("apply_live"))
    monkeypatch.setattr(session, "estimate_point_bounds", lambda xyz: ("bounds", xyz.shape[0]))
    monkeypatch.setattr(session, "initialize_training_scene", lambda viewer_obj, frame_targets_native=None: calls.append(("initialize", frame_targets_native)))

    def _ensure_cached(viewer_obj, init) -> None:
        calls.append(("ensure_cached", init.seed, viewer_obj.s.colmap_import.init_mode, viewer_obj.s.colmap_import.fibonacci_sphere_point_count))
        viewer_obj.s.cached_init_point_positions = expected_positions
        viewer_obj.s.cached_init_point_colors = expected_colors
        viewer_obj.s.cached_init_signature = ("cached",)

    monkeypatch.setattr(session, "_ensure_cached_init_source", _ensure_cached)

    session._finish_import_colmap_dataset(
        viewer,
        colmap_root=Path("dataset/garden"),
        database_path=None,
        images_root=Path("dataset/garden/images"),
        init_mode="pointcloud",
        custom_ply_path=None,
        image_downscale_mode="original",
        image_downscale_max_size=1600,
        image_downscale_scale=1.0,
        nn_radius_scale_coef=0.25,
        min_track_length=3,
        diffused_point_count=100000,
        fibonacci_sphere_point_count=4,
        fibonacci_sphere_radius=2.0,
        recon=recon,
        training_frames=[],
        frame_targets_native=None,
    )

    assert calls == [
        ("ensure_cached", 7, "pointcloud", 4),
        "apply_live",
        ("fit", ("bounds", 1)),
        ("initialize", None),
    ]
    np.testing.assert_allclose(viewer.s.cached_init_point_positions, expected_positions, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(viewer.s.cached_init_point_colors, expected_colors, rtol=0.0, atol=0.0)
    assert viewer.s.cached_init_signature == ("cached",)


def test_import_colmap_dataset_uses_aligned_reconstruction(monkeypatch) -> None:
    recon = object()
    frames = [SimpleNamespace(width=32, height=32, image_id=1)]
    calls: list[object] = []
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            renderer=SimpleNamespace(
                clear_scene_resources=lambda: calls.append(("clear_renderer", None)),
                set_debug_grad_norm_buffer=lambda buffer: calls.append(("clear_grad_debug", buffer)),
                set_debug_splat_age_buffer=lambda buffer: calls.append(("clear_splat_age_debug", buffer)),
            ),
            trainer=None,
            training_active=False,
            training_elapsed_s=0.0,
            training_resume_time=None,
            training_renderer=None,
            training_frames=[],
            scene=None,
            scene_path=None,
            colmap_root=None,
            colmap_recon=None,
            colmap_import_progress=None,
            applied_renderer_params_training=None,
            applied_renderer_params_debug=None,
            applied_training_signature=None,
            applied_training_runtime_factor=None,
            pending_training_runtime_resize=False,
            applied_renderer_params_main=None,
            cached_training_setup_signature=None,
            cached_training_setup=None,
        ),
        c=lambda key: SimpleNamespace(value=0),
    )

    monkeypatch.setattr(session, "update_debug_frame_slider_range", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_clear_cached_init_source", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_reset_training_visual_state", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_reset_loss_debug", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_load_aligned_colmap_reconstruction", lambda root, auto_rotate_scene=True: recon if auto_rotate_scene else (_ for _ in ()).throw(AssertionError("expected aligned reconstruction")))
    monkeypatch.setattr(
        session,
        "build_training_frames_from_root",
        lambda recon, images_root, selected_camera_ids=(), downscale_mode="original", downscale_max_size=None, downscale_scale=1.0: calls.append(("frames", recon, Path(images_root), tuple(selected_camera_ids), downscale_mode, downscale_max_size, downscale_scale)) or frames,
    )
    monkeypatch.setattr(
        session,
        "_finish_import_colmap_dataset",
        lambda viewer_obj, **kwargs: calls.append(("finish", kwargs["recon"], kwargs["training_frames"], kwargs["frame_targets_native"])),
    )
    monkeypatch.setattr(session, "_create_native_dataset_textures", lambda viewer_obj, resolved_frames: ["tex0"] if resolved_frames is frames else (_ for _ in ()).throw(AssertionError("unexpected frames")))

    session.import_colmap_dataset(
        viewer,
        colmap_root=Path("dataset/garden"),
        database_path=None,
        images_root=Path("dataset/garden/images_8"),
        init_mode="pointcloud",
        auto_rotate_scene=True,
        custom_ply_path=None,
        image_downscale_mode="original",
        image_downscale_max_size=2048,
        image_downscale_scale=1.0,
        nn_radius_scale_coef=0.5,
        diffused_point_count=100000,
    )

    assert calls[:3] == [
        ("clear_grad_debug", None),
        ("clear_splat_age_debug", None),
        ("clear_renderer", None),
    ]
    assert calls[-2:] == [
        ("frames", recon, Path("dataset/garden/images_8"), (), "original", 2048, 1.0),
        ("finish", recon, frames, ["tex0"]),
    ]


def test_import_colmap_dataset_can_skip_aligned_reconstruction(monkeypatch) -> None:
    raw_recon = object()
    frames = [SimpleNamespace(width=32, height=32, image_id=1)]
    calls: list[object] = []
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            renderer=SimpleNamespace(
                clear_scene_resources=lambda: calls.append(("clear_renderer", None)),
                set_debug_grad_norm_buffer=lambda buffer: calls.append(("clear_grad_debug", buffer)),
                set_debug_splat_age_buffer=lambda buffer: calls.append(("clear_splat_age_debug", buffer)),
            ),
            trainer=None,
            training_active=False,
            training_elapsed_s=0.0,
            training_resume_time=None,
            training_renderer=None,
            training_frames=[],
            scene=None,
            scene_path=None,
            colmap_root=None,
            colmap_recon=None,
            colmap_import_progress=None,
            applied_renderer_params_training=None,
            applied_renderer_params_debug=None,
            applied_training_signature=None,
            applied_training_runtime_factor=None,
            pending_training_runtime_resize=False,
            applied_renderer_params_main=None,
            cached_training_setup_signature=None,
            cached_training_setup=None,
        ),
        c=lambda key: SimpleNamespace(value=0),
    )

    monkeypatch.setattr(session, "update_debug_frame_slider_range", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_clear_cached_init_source", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_reset_training_visual_state", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_reset_loss_debug", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_load_aligned_colmap_reconstruction", lambda root, auto_rotate_scene=True: raw_recon if not auto_rotate_scene else (_ for _ in ()).throw(AssertionError("expected raw reconstruction")))
    monkeypatch.setattr(
        session,
        "build_training_frames_from_root",
        lambda recon, images_root, selected_camera_ids=(), downscale_mode="original", downscale_max_size=None, downscale_scale=1.0: calls.append(("frames", recon, Path(images_root), tuple(selected_camera_ids), downscale_mode, downscale_max_size, downscale_scale)) or frames,
    )
    monkeypatch.setattr(
        session,
        "_finish_import_colmap_dataset",
        lambda viewer_obj, **kwargs: calls.append(("finish", kwargs["recon"], kwargs["training_frames"], kwargs["frame_targets_native"])),
    )
    monkeypatch.setattr(session, "_create_native_dataset_textures", lambda viewer_obj, resolved_frames: ["tex0"] if resolved_frames is frames else (_ for _ in ()).throw(AssertionError("unexpected frames")))

    session.import_colmap_dataset(
        viewer,
        colmap_root=Path("dataset/garden"),
        database_path=None,
        images_root=Path("dataset/garden/images_8"),
        init_mode="pointcloud",
        auto_rotate_scene=False,
        custom_ply_path=None,
        image_downscale_mode="original",
        image_downscale_max_size=2048,
        image_downscale_scale=1.0,
        nn_radius_scale_coef=0.5,
        diffused_point_count=100000,
    )

    assert calls[-2:] == [
        ("frames", raw_recon, Path("dataset/garden/images_8"), (), "original", 2048, 1.0),
        ("finish", raw_recon, frames, ["tex0"]),
    ]


def test_advance_colmap_import_prepare_respects_auto_rotate_scene(monkeypatch) -> None:
    raw_recon = SimpleNamespace(images={})
    calls: list[object] = []
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            colmap_import_progress=ColmapImportProgress(
                dataset_root=Path("dataset"),
                colmap_root=Path("dataset"),
                database_path=None,
                images_root=Path("dataset/images"),
                init_mode="pointcloud",
                custom_ply_path=None,
                image_downscale_mode="original",
                image_downscale_max_size=2048,
                image_downscale_scale=1.0,
                nn_radius_scale_coef=0.5,
                auto_rotate_scene=False,
            )
        ),
        init_params=lambda: SimpleNamespace(seed=0),
    )

    monkeypatch.setattr(session, "_load_aligned_colmap_reconstruction", lambda root, auto_rotate_scene=True: calls.append((Path(root), auto_rotate_scene)) or raw_recon)
    monkeypatch.setattr(session, "build_depth_path_index", lambda depth_root: (_ for _ in ()).throw(AssertionError("depth index should not be built")))

    session.advance_colmap_import(viewer)

    assert calls == [(Path("dataset"), False)]
    assert viewer.s.colmap_import_progress.recon is raw_recon
    assert viewer.s.colmap_import_progress.phase == "scan_frames"


def test_colmap_import_settings_defaults_prefer_pointcloud() -> None:
    defaults = ColmapImportSettings()

    assert defaults.init_mode == "pointcloud"
    assert defaults.auto_rotate_scene is True
    assert defaults.nn_radius_scale_coef == 0.5
    assert defaults.min_track_length == 3
    assert defaults.depth_root is None
    assert defaults.selected_camera_ids == ()
    assert defaults.depth_value_mode == "z_depth"
    assert defaults.depth_point_count == 100000
    assert defaults.fibonacci_sphere_point_count == 0
    assert defaults.fibonacci_sphere_radius == 20.0
    assert defaults.use_target_alpha_mask is False


def test_refresh_cached_raster_grad_histograms_requires_explicit_request() -> None:
    calls: list[tuple[int, int, float, float]] = []
    hist = SimpleNamespace(counts=np.ones((14, 8), dtype=np.int64), param_labels=("p",) * 14)
    ranges = SimpleNamespace(min_values=np.full((14,), -1.0, dtype=np.float32), max_values=np.full((14,), 2.0, dtype=np.float32), param_labels=("p",) * 14)
    renderer = SimpleNamespace(
        cached_raster_grad_atomic_mode="float",
        compute_scene_param_histograms=lambda scene_count, *, bin_count, min_value, max_value, param_min_values=None, param_max_values=None, metrics=None: calls.append((scene_count, bin_count, min_value, max_value, metrics)) or hist,
        compute_scene_param_ranges=lambda scene_count, *, metrics=None: ranges,
    )
    metrics = object()
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={"hist_bin_count": 8, "hist_min_value": -7.0, "hist_max_value": 1.0, "_histograms_refresh_requested": True}),
        s=SimpleNamespace(
            trainer=SimpleNamespace(state=SimpleNamespace(step=3), scene=SimpleNamespace(count=32), metrics=metrics),
            training_renderer=renderer,
            cached_raster_grad_histograms=None,
            cached_raster_grad_ranges=None,
            cached_raster_grad_histogram_mode="",
            cached_raster_grad_histogram_step=-1,
            cached_raster_grad_histogram_scene_count=-1,
            cached_raster_grad_histogram_signature=None,
            cached_raster_grad_histogram_status="",
        ),
    )

    session.refresh_cached_raster_grad_histograms(viewer)
    session.refresh_cached_raster_grad_histograms(viewer)

    assert calls == [(32, 8, -7.0, 1.0, metrics)]
    assert viewer.s.cached_raster_grad_histograms is hist
    assert viewer.s.cached_raster_grad_ranges is ranges
    assert viewer.s.cached_raster_grad_histogram_signature == (3, 32, 8, -7.0, 1.0)


def test_refresh_cached_raster_grad_histograms_skips_without_request() -> None:
    calls: list[int] = []
    renderer = SimpleNamespace(
        cached_raster_grad_atomic_mode="float",
        compute_scene_param_histograms=lambda scene_count, *, bin_count, min_value, max_value, param_min_values=None, param_max_values=None, metrics=None: calls.append(scene_count),
        compute_scene_param_ranges=lambda scene_count, *, metrics=None: None,
    )
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={"hist_bin_count": 8, "hist_min_value": -7.0, "hist_max_value": 1.0, "_histograms_refresh_requested": False}),
        s=SimpleNamespace(
            trainer=SimpleNamespace(state=SimpleNamespace(step=3), scene=SimpleNamespace(count=32), metrics=object()),
            training_renderer=renderer,
            cached_raster_grad_histograms=None,
            cached_raster_grad_ranges=None,
            cached_raster_grad_histogram_mode="",
            cached_raster_grad_histogram_step=-1,
            cached_raster_grad_histogram_scene_count=-1,
            cached_raster_grad_histogram_signature=None,
            cached_raster_grad_histogram_status="",
        ),
    )

    session.refresh_cached_raster_grad_histograms(viewer)

    assert calls == []
    assert viewer.s.cached_raster_grad_histograms is None
    assert viewer.s.cached_raster_grad_histogram_signature is None


def test_refresh_cached_raster_grad_histograms_honors_manual_refresh() -> None:
    calls: list[int] = []
    renderer = SimpleNamespace(
        cached_raster_grad_atomic_mode="fixed",
        compute_scene_param_histograms=lambda scene_count, *, bin_count, min_value, max_value, param_min_values=None, param_max_values=None, metrics=None: calls.append(scene_count) or SimpleNamespace(counts=np.zeros((14, bin_count), dtype=np.int64), param_labels=(), param_groups=()),
        compute_scene_param_ranges=lambda scene_count, *, metrics=None: SimpleNamespace(min_values=np.zeros((14,), dtype=np.float32), max_values=np.zeros((14,), dtype=np.float32), param_labels=(), param_groups=()),
    )
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={"hist_bin_count": 4, "hist_min_value": -1.0, "hist_max_value": 1.0, "_histograms_refresh_requested": True}),
        s=SimpleNamespace(
            trainer=SimpleNamespace(state=SimpleNamespace(step=0), scene=SimpleNamespace(count=12), metrics=object()),
            training_renderer=renderer,
            cached_raster_grad_histograms=None,
            cached_raster_grad_ranges=None,
            cached_raster_grad_histogram_mode="",
            cached_raster_grad_histogram_step=-1,
            cached_raster_grad_histogram_scene_count=-1,
            cached_raster_grad_histogram_signature=None,
            cached_raster_grad_histogram_status="",
        ),
    )

    session.refresh_cached_raster_grad_histograms(viewer)

    assert calls == [12]
    assert viewer.ui._values["_histograms_refresh_requested"] is False


def test_refresh_cached_raster_grad_histograms_uses_scene_param_ranges_directly() -> None:
    hist = SimpleNamespace(counts=np.ones((14, 4), dtype=np.int64), param_labels=("g",) * 14, param_groups=(("g", (0,)),))
    ranges = SimpleNamespace(
        min_values=np.array([-1.0, -2.0, -0.1], dtype=np.float32),
        max_values=np.array([3.0, 4.0, 0.4], dtype=np.float32),
        param_labels=("p0", "p1", "p2"),
        param_groups=(("group", (0, 1, 2)),),
    )
    renderer = SimpleNamespace(
        cached_raster_grad_atomic_mode="float",
        compute_scene_param_histograms=lambda scene_count, *, bin_count, min_value, max_value, param_min_values=None, param_max_values=None, metrics=None: hist,
        compute_scene_param_ranges=lambda scene_count, *, metrics=None: ranges,
    )
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={"hist_bin_count": 4, "hist_min_value": -1.0, "hist_max_value": 1.0, "_histograms_refresh_requested": True}),
        s=SimpleNamespace(
            trainer=SimpleNamespace(state=SimpleNamespace(step=5), scene=SimpleNamespace(count=7), metrics=object()),
            training_renderer=renderer,
            cached_raster_grad_histograms=None,
            cached_raster_grad_ranges=None,
            cached_raster_grad_histogram_mode="",
            cached_raster_grad_histogram_step=-1,
            cached_raster_grad_histogram_scene_count=-1,
            cached_raster_grad_histogram_signature=None,
            cached_raster_grad_histogram_status="",
        ),
    )

    session.refresh_cached_raster_grad_histograms(viewer)

    np.testing.assert_allclose(viewer.s.cached_raster_grad_ranges.min_values, np.array([-1.0, -2.0, -0.1], dtype=np.float32))
    np.testing.assert_allclose(viewer.s.cached_raster_grad_ranges.max_values, np.array([3.0, 4.0, 0.4], dtype=np.float32))
    assert viewer.s.cached_raster_grad_ranges.param_labels == ("p0", "p1", "p2")


def test_refresh_cached_raster_grad_histograms_appends_refinement_distributions() -> None:
    scene_edges = np.stack(
        (
            np.linspace(0.0, 1.0, 5, dtype=np.float64),
            np.linspace(-3.0, 0.0, 5, dtype=np.float64),
            np.linspace(-2.0, -0.1, 5, dtype=np.float64),
        ),
        axis=0,
    )
    scene_hist = SimpleNamespace(
        counts=np.array([[1, 2, 3, 4], [2, 0, 1, 1], [0, 1, 2, 1]], dtype=np.int64),
        bin_edges_log10=scene_edges[0],
        bin_edges_by_param_log10=scene_edges,
        param_labels=("position.x", "scale.x", "opacity"),
        param_groups=(("position", (0,)), ("scale", (1,)), ("opacity", (2,))),
        param_value_scales=(session.PARAM_HISTOGRAM_SCALE_LINEAR, session.PARAM_HISTOGRAM_SCALE_LOG10, session.PARAM_HISTOGRAM_SCALE_LOG10),
    )
    scene_ranges = SimpleNamespace(
        min_values=np.array([-10.0, -3.0, -2.0], dtype=np.float32),
        max_values=np.array([20.0, 0.0, -0.1], dtype=np.float32),
        param_labels=("position.x", "scale.x", "opacity"),
        param_groups=(("position", (0,)), ("scale", (1,)), ("opacity", (2,))),
        param_value_scales=(session.PARAM_HISTOGRAM_SCALE_LINEAR, session.PARAM_HISTOGRAM_SCALE_LOG10, session.PARAM_HISTOGRAM_SCALE_LOG10),
    )
    refinement_hist = SimpleNamespace(
        counts=np.array([[4, 3, 2, 1], [0, 1, 0, 2]], dtype=np.int64),
        bin_edges_log10=np.linspace(-4.0, 1.0, 5, dtype=np.float64),
        param_labels=("Contribution distribution", "Refinement distribution"),
        param_groups=(("Contribution distribution", (0,)), ("Refinement distribution", (1,))),
        param_value_scales=(session.PARAM_HISTOGRAM_SCALE_LOG10, session.PARAM_HISTOGRAM_SCALE_LOG10),
    )
    refinement_ranges = SimpleNamespace(
        min_values=np.array([-3.0, -4.0], dtype=np.float32),
        max_values=np.array([0.0, 1.0], dtype=np.float32),
        param_labels=("Contribution distribution", "Refinement distribution"),
        param_groups=(("Contribution distribution", (0,)), ("Refinement distribution", (1,))),
        param_value_scales=(session.PARAM_HISTOGRAM_SCALE_LOG10, session.PARAM_HISTOGRAM_SCALE_LOG10),
    )
    hist_bounds: list[tuple[float, float]] = []
    scene_bounds: list[tuple[np.ndarray, np.ndarray]] = []
    renderer = SimpleNamespace(
        cached_raster_grad_atomic_mode="float",
        compute_scene_param_histograms=lambda scene_count, *, bin_count, min_value, max_value, param_min_values=None, param_max_values=None, metrics=None: scene_bounds.append((np.asarray(param_min_values, dtype=np.float32), np.asarray(param_max_values, dtype=np.float32))) or scene_hist,
        compute_scene_param_ranges=lambda scene_count, *, metrics=None: scene_ranges,
    )
    trainer = SimpleNamespace(
        state=SimpleNamespace(step=8),
        scene=SimpleNamespace(count=16),
        metrics=object(),
        compute_refinement_distribution_histograms=lambda scene_count, *, bin_count, min_log10, max_log10: hist_bounds.append((min_log10, max_log10)) or refinement_hist,
        compute_refinement_distribution_ranges=lambda scene_count: refinement_ranges,
    )
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={"hist_bin_count": 4, "hist_min_value": 0.0, "hist_max_value": 1.0, "_histograms_refresh_requested": True}),
        s=SimpleNamespace(
            trainer=trainer,
            training_renderer=renderer,
            cached_raster_grad_histograms=None,
            cached_raster_grad_ranges=None,
            cached_raster_grad_histogram_mode="",
            cached_raster_grad_histogram_step=-1,
            cached_raster_grad_histogram_scene_count=-1,
            cached_raster_grad_histogram_signature=None,
            cached_raster_grad_histogram_status="",
        ),
    )

    session.refresh_cached_raster_grad_histograms(viewer)

    hist = viewer.s.cached_raster_grad_histograms
    ranges = viewer.s.cached_raster_grad_ranges
    np.testing.assert_array_equal(hist.counts, np.concatenate((scene_hist.counts, refinement_hist.counts), axis=0))
    np.testing.assert_allclose(
        hist.bin_edges_by_param_log10,
        np.concatenate((scene_edges, np.stack((refinement_hist.bin_edges_log10, refinement_hist.bin_edges_log10), axis=0)), axis=0),
    )
    assert hist.param_labels == ("position.x", "scale.x", "opacity", "Contribution distribution", "Refinement distribution")
    assert hist.param_groups == (("position", (0,)), ("scale", (1,)), ("opacity", (2,)), ("Contribution distribution", (3,)), ("Refinement distribution", (4,)))
    assert hist.param_value_scales == (session.PARAM_HISTOGRAM_SCALE_LINEAR, session.PARAM_HISTOGRAM_SCALE_LOG10, session.PARAM_HISTOGRAM_SCALE_LOG10, session.PARAM_HISTOGRAM_SCALE_LOG10, session.PARAM_HISTOGRAM_SCALE_LOG10)
    np.testing.assert_allclose(scene_bounds[0][0], np.array([0.0, -3.0, -2.0], dtype=np.float32))
    np.testing.assert_allclose(scene_bounds[0][1], np.array([1.0, 0.0, -0.1], dtype=np.float32))
    assert hist_bounds == [(-4.0, 1.0)]
    np.testing.assert_allclose(ranges.min_values, np.array([-10.0, -3.0, -2.0, -3.0, -4.0], dtype=np.float32))
    np.testing.assert_allclose(ranges.max_values, np.array([20.0, 0.0, -0.1, 0.0, 1.0], dtype=np.float32))
    assert ranges.param_groups == hist.param_groups
    assert ranges.param_value_scales == hist.param_value_scales


def test_initialize_training_scene_rebinds_debug_buffers_for_new_trainer(monkeypatch) -> None:
    class _Encoder:
        def finish(self) -> str:
            return "finished"

    class _ViewportRenderer:
        def __init__(self) -> None:
            self.grad_buffer = "stale-grad"
            self.splat_age_buffer = "stale-splat-age"
            self.copy_targets: list[object] = []

        def set_debug_grad_norm_buffer(self, buffer) -> None:
            self.grad_buffer = buffer

        def set_debug_splat_age_buffer(self, buffer) -> None:
            self.splat_age_buffer = buffer

        def copy_scene_state_to(self, encoder, dst) -> None:
            del encoder
            self.copy_targets.append(dst)

    class _TrainingRenderer:
        width = 32
        height = 32
        work_buffers = {"debug_grad_norm": "new-grad"}

        def copy_scene_state_to(self, encoder, dst) -> None:
            del encoder
            dst.copy_targets.append(self)

    training_renderer = _TrainingRenderer()
    old_training_renderer = object()
    main_renderer = _ViewportRenderer()
    debug_renderer = _ViewportRenderer()
    new_trainer = SimpleNamespace(
        refinement_buffers={"splat_age": "new-splat-age"},
        effective_train_downscale_factor=lambda step=0: 1,
        effective_train_render_factor=lambda step=0: 1,
    )
    calls: list[str] = []
    viewer = SimpleNamespace(
        device=SimpleNamespace(
            create_command_encoder=lambda: _Encoder(),
            submit_command_buffer=lambda command_buffer: None,
        ),
        toolkit=SimpleNamespace(reset_plot_history=lambda: calls.append("reset_plot_history")),
        init_params=lambda: SimpleNamespace(seed=7),
        renderer_params=lambda allow_debug_overlays: SimpleNamespace(mode="debug" if allow_debug_overlays else "train"),
        training_params=lambda: object(),
        apply_camera_fit=lambda bounds: None,
        s=SimpleNamespace(
            colmap_recon=object(),
            training_frames=[SimpleNamespace(width=32, height=32)],
            colmap_import=SimpleNamespace(
                init_mode="pointcloud",
                custom_ply_path=None,
                nn_radius_scale_coef=0.5,
                diffused_point_count=100,
            ),
            trainer=SimpleNamespace(refinement_buffers={"splat_age": "old-splat-age"}),
            renderer=main_renderer,
            debug_renderer=debug_renderer,
            training_renderer=old_training_renderer,
            training_active=True,
            training_elapsed_s=12.0,
            training_resume_time=3.0,
            scene=None,
            applied_renderer_params_training=None,
            applied_training_signature=None,
            applied_training_runtime_factor=None,
            pending_training_runtime_resize=True,
            cached_raster_grad_histograms=object(),
            cached_raster_grad_ranges=object(),
            cached_raster_grad_histogram_mode="fixed",
            cached_raster_grad_histogram_step=9,
            cached_raster_grad_histogram_scene_count=99,
            cached_raster_grad_histogram_signature=("old",),
            cached_raster_grad_histogram_status="stale",
            last_error="stale",
            colmap_root=None,
        ),
    )

    monkeypatch.setattr(session, "resolve_effective_training_setup", lambda viewer_obj: (SimpleNamespace(seed=7), SimpleNamespace(training=SimpleNamespace(max_gaussians=8), adam=SimpleNamespace(tag="adam"), stability=SimpleNamespace(tag="stability")), SimpleNamespace(), SimpleNamespace(name="test")))
    monkeypatch.setattr(session, "resolve_effective_train_render_factor", lambda training, step: 1)
    monkeypatch.setattr(session, "resolve_training_resolution", lambda width, height, factor: (width, height))
    def _ensure_renderer(viewer_obj, attr, width, height, allow_debug_overlays):
        del attr, width, height, allow_debug_overlays
        calls.append(f"ensure_renderer_cleared={viewer_obj.s.training_renderer is None}")
        viewer_obj.s.training_renderer = training_renderer
        return training_renderer
    monkeypatch.setattr(session, "ensure_renderer", _ensure_renderer)
    monkeypatch.setattr(session, "_build_initial_training_scene", lambda viewer_obj, init, params, init_hparams: (SimpleNamespace(count=8), 0.25))
    monkeypatch.setattr(session, "apply_live_params", lambda viewer_obj: calls.append(f"apply_live_params_trainer_is_none={viewer_obj.s.trainer is None}"))
    monkeypatch.setattr(session, "GaussianTrainer", lambda **kwargs: new_trainer)
    monkeypatch.setattr(session, "estimate_scene_bounds", lambda scene: scene)
    monkeypatch.setattr(session, "_renderer_params_signature", lambda params: (params.mode,))
    monkeypatch.setattr(session, "_training_params_signature", lambda params: ("training",))
    monkeypatch.setattr(session, "update_debug_frame_slider_range", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_reset_loss_debug", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_invalidate", lambda viewer_obj, *targets: None)

    session.initialize_training_scene(viewer)

    assert viewer.s.trainer is new_trainer
    assert calls == ["reset_plot_history", "ensure_renderer_cleared=True", "apply_live_params_trainer_is_none=True", "reset_plot_history"]
    assert main_renderer.grad_buffer == "new-grad"
    assert main_renderer.splat_age_buffer == "new-splat-age"
    assert debug_renderer.grad_buffer == "new-grad"
    assert debug_renderer.splat_age_buffer == "new-splat-age"
    assert viewer.s.cached_raster_grad_histograms is None
    assert viewer.s.cached_raster_grad_ranges is None
    assert viewer.s.cached_raster_grad_histogram_mode == ""
    assert viewer.s.cached_raster_grad_histogram_step == -1
    assert viewer.s.cached_raster_grad_histogram_scene_count == -1
    assert viewer.s.cached_raster_grad_histogram_signature is None
    assert viewer.s.cached_raster_grad_histogram_status == ""


def test_initialize_training_scene_rebuilds_training_frames_from_colmap(monkeypatch) -> None:
    frame = SimpleNamespace(width=48, height=24, image_id=9)
    training_renderer = SimpleNamespace(copy_scene_state_to=lambda encoder, dst: None)
    main_renderer = SimpleNamespace(set_debug_grad_norm_buffer=lambda buffer: None, set_debug_splat_age_buffer=lambda buffer: None)
    debug_renderer = SimpleNamespace(set_debug_grad_norm_buffer=lambda buffer: None, set_debug_splat_age_buffer=lambda buffer: None)
    new_trainer = SimpleNamespace(effective_train_downscale_factor=lambda step=0: 1, effective_train_render_factor=lambda step=0: 1, scene=SimpleNamespace(count=8), refinement_buffers={})
    built_scene = SimpleNamespace(count=8)
    viewer = SimpleNamespace(
        device=SimpleNamespace(create_command_encoder=lambda: SimpleNamespace(finish=lambda: "finished"), submit_command_buffer=lambda command_buffer: None),
        renderer_params=lambda allow_debug_overlays: SimpleNamespace(mode="main"),
        training_params=lambda: object(),
        init_params=lambda: SimpleNamespace(seed=7),
        apply_camera_fit=lambda bounds: None,
        s=SimpleNamespace(
            colmap_recon=object(),
            training_frames=[frame],
            colmap_import=SimpleNamespace(
                images_root=Path("dataset/garden/images_8"),
                image_downscale_mode="original",
                image_downscale_max_size=2048,
                image_downscale_scale=1.0,
                init_mode="pointcloud",
                custom_ply_path=None,
                nn_radius_scale_coef=0.5,
                diffused_point_count=100,
            ),
            trainer=None,
            renderer=main_renderer,
            debug_renderer=debug_renderer,
            training_renderer=None,
            training_active=False,
            training_elapsed_s=0.0,
            training_resume_time=None,
            scene=None,
            applied_renderer_params_training=None,
            applied_training_signature=None,
            applied_training_runtime_factor=None,
            pending_training_runtime_resize=False,
            cached_raster_grad_histograms=None,
            cached_raster_grad_ranges=None,
            cached_raster_grad_histogram_mode="",
            cached_raster_grad_histogram_step=-1,
            cached_raster_grad_histogram_scene_count=-1,
            cached_raster_grad_histogram_signature=None,
            cached_raster_grad_histogram_status="",
            last_error="",
            colmap_root=Path("dataset/garden"),
        ),
    )

    monkeypatch.setattr(session, "_refresh_training_frames", lambda viewer_obj: (_ for _ in ()).throw(AssertionError("reinit should not rebuild frames when cached frames exist")))
    monkeypatch.setattr(session, "resolve_effective_training_setup", lambda viewer_obj: (SimpleNamespace(seed=7), SimpleNamespace(training=SimpleNamespace(max_gaussians=8), adam=SimpleNamespace(), stability=SimpleNamespace()), SimpleNamespace(), SimpleNamespace(name="test")))
    monkeypatch.setattr(session, "resolve_effective_train_render_factor", lambda training, step: 1)
    monkeypatch.setattr(session, "resolve_training_resolution", lambda width, height, factor: (width, height))
    monkeypatch.setattr(session, "ensure_renderer", lambda viewer_obj, attr, width, height, allow_debug_overlays: training_renderer)
    monkeypatch.setattr(session, "_build_initial_training_scene", lambda viewer_obj, init, params, init_hparams: (built_scene, 0.25))
    monkeypatch.setattr(session, "apply_live_params", lambda viewer_obj: None)
    monkeypatch.setattr(session, "GaussianTrainer", lambda **kwargs: new_trainer)
    monkeypatch.setattr(session, "estimate_scene_bounds", lambda loaded_scene: loaded_scene)
    monkeypatch.setattr(session, "_renderer_params_signature", lambda params: (params.mode,))
    monkeypatch.setattr(session, "_training_params_signature", lambda params: ("training",))
    monkeypatch.setattr(session, "update_debug_frame_slider_range", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_reset_loss_debug", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_invalidate", lambda viewer_obj, *targets: None)

    session.initialize_training_scene(viewer)

    assert viewer.s.colmap_recon is not None
    assert viewer.s.training_frames == [frame]


def test_refresh_training_frames_uses_cached_reconstruction(monkeypatch) -> None:
    recon = object()
    frame = SimpleNamespace(width=24, height=12, image_id=3)
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            colmap_recon=recon,
            colmap_root=Path("dataset/garden"),
            training_frames=[],
            colmap_import=SimpleNamespace(
                images_root=Path("dataset/garden/images_8"),
                image_downscale_mode="max_size",
                image_downscale_max_size=1024,
                image_downscale_scale=1.0,
            ),
        )
    )

    monkeypatch.setattr(session, "load_colmap_reconstruction", lambda root: (_ for _ in ()).throw(AssertionError("should not reload reconstruction")))
    monkeypatch.setattr(
        session,
        "build_training_frames_from_root",
        lambda recon_obj, images_root, selected_camera_ids=(), downscale_mode="original", downscale_max_size=None, downscale_scale=1.0: [frame] if recon_obj is recon and tuple(selected_camera_ids) == () else (_ for _ in ()).throw(AssertionError("unexpected reconstruction instance")),
    )

    session._refresh_training_frames(viewer)

    assert viewer.s.training_frames == [frame]


def test_build_initial_training_scene_uses_cached_diffused_points(monkeypatch) -> None:
    cached_positions = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32)
    cached_colors = np.array([[0.2, 0.3, 0.4], [0.4, 0.3, 0.2]], dtype=np.float32)
    built_scene = SimpleNamespace(count=2)
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            colmap_recon=object(),
            colmap_root=Path("dataset/garden"),
            colmap_import=SimpleNamespace(init_mode="pointcloud", diffused_enabled=True, diffused_nn_radius_scale_coef=0.5, min_track_length=3),
            cached_init_diffused_positions=cached_positions,
            cached_init_diffused_colors=cached_colors,
            cached_init_scene=None,
            cached_init_signature=("cached",),
        )
    )

    monkeypatch.setattr(session, "_ensure_cached_init_source", lambda viewer_obj, init: None)
    monkeypatch.setattr(
        session,
        "_diffused_pointcloud_init_hparams_from_positions",
        lambda recon, positions, init_hparams, nn_radius_scale_coef, min_track_length: SimpleNamespace(base_scale=0.25) if np.array_equal(positions, cached_positions) and int(min_track_length) == 3 else (_ for _ in ()).throw(AssertionError("cached positions not used")),
    )
    monkeypatch.setattr(session, "initialize_scene_from_points_colors", lambda positions, colors, seed, init_hparams: built_scene if np.array_equal(positions, cached_positions) and np.array_equal(colors, cached_colors) else (_ for _ in ()).throw(AssertionError("cached init data not forwarded")))

    scene, scale_reg_reference = session._build_initial_training_scene(viewer, SimpleNamespace(seed=7), SimpleNamespace(training=SimpleNamespace(max_gaussians=8)), SimpleNamespace())

    assert scene is built_scene
    assert scale_reg_reference is None


def test_ensure_cached_init_source_samples_custom_mesh(monkeypatch, tmp_path: Path) -> None:
    mesh_path = tmp_path / "seed.obj"
    mesh_path.write_text("o seed\n", encoding="utf-8")
    sampled_positions = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32)
    sampled_colors = np.array([[0.2, 0.3, 0.4], [0.4, 0.3, 0.2]], dtype=np.float32)
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            colmap_recon=object(),
            colmap_root=Path("dataset/garden"),
            colmap_import=SimpleNamespace(
                init_mode="pointcloud",
                custom_mesh_enabled=True,
                custom_mesh_path=mesh_path,
                custom_mesh_point_count=2048,
                min_track_length=3,
                fibonacci_sphere_point_count=0,
                fibonacci_sphere_radius=20.0,
            ),
            cached_init_custom_mesh_positions=None,
            cached_init_custom_mesh_colors=None,
            cached_init_scene=None,
            cached_init_signature=None,
        )
    )

    monkeypatch.setattr(
        session,
        "sample_mesh_surface_points",
        lambda mesh_arg, point_count, seed: (sampled_positions, sampled_colors)
        if Path(mesh_arg) == mesh_path and int(point_count) == 2048 and int(seed) == 11
        else (_ for _ in ()).throw(AssertionError("unexpected mesh sample request")),
    )
    session._ensure_cached_init_source(viewer, SimpleNamespace(seed=11))

    expected_positions = sampled_positions @ session._MESH_TO_COLMAP_COORDINATE_TRANSFORM[:3, :3].T
    assert np.array_equal(viewer.s.cached_init_custom_mesh_positions, expected_positions)
    assert np.array_equal(viewer.s.cached_init_custom_mesh_colors, sampled_colors)
    assert viewer.s.cached_init_signature == session._cached_init_signature(viewer, SimpleNamespace(seed=11))


def test_build_initial_training_scene_uses_cached_mesh_points(monkeypatch) -> None:
    cached_positions = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    cached_colors = np.array([[0.2, 0.3, 0.4], [0.4, 0.3, 0.2], [0.7, 0.6, 0.5]], dtype=np.float32)
    built_scene = SimpleNamespace(count=2)
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            colmap_recon=object(),
            colmap_root=Path("dataset/garden"),
            colmap_import=SimpleNamespace(init_mode="pointcloud", custom_mesh_enabled=True, custom_mesh_nn_radius_scale_coef=0.25, fibonacci_sphere_enabled=False),
            cached_init_custom_mesh_positions=cached_positions,
            cached_init_custom_mesh_colors=cached_colors,
            cached_init_scene=None,
            cached_init_signature=("cached",),
        )
    )

    monkeypatch.setattr(session, "_ensure_cached_init_source", lambda viewer_obj, init: None)
    monkeypatch.setattr(
        session,
        "_sampled_point_init_hparams_from_positions",
        lambda positions, max_gaussians, init_hparams, nn_radius_scale_coef: SimpleNamespace(base_scale=0.125)
        if np.array_equal(positions, cached_positions) and int(max_gaussians) == cached_positions.shape[0] and float(nn_radius_scale_coef) == 0.25
        else (_ for _ in ()).throw(AssertionError("unexpected mesh init hparams request")),
    )
    monkeypatch.setattr(
        session,
        "initialize_scene_from_points_colors",
        lambda positions, colors, seed, init_hparams: built_scene
        if np.array_equal(positions, cached_positions) and np.array_equal(colors, cached_colors) and int(seed) == 7 and getattr(init_hparams, "base_scale", None) == 0.125
        else (_ for _ in ()).throw(AssertionError("unexpected mesh initializer data")),
    )

    scene, scale_reg_reference = session._build_initial_training_scene(viewer, SimpleNamespace(seed=7), SimpleNamespace(training=SimpleNamespace(max_gaussians=2)), SimpleNamespace())

    assert scene is built_scene
    assert scale_reg_reference is None


def test_sampled_point_init_hparams_disables_mesh_position_jitter(monkeypatch) -> None:
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32)
    resolved = GaussianInitHyperParams(position_jitter_std=0.75, base_scale=1.5, scale_jitter_ratio=0.2, initial_opacity=0.3, color_jitter_std=0.0)

    monkeypatch.setattr(session, "resolve_points_init_hparams", lambda pts, max_gaussians, init_hparams: resolved if np.array_equal(pts, positions) and int(max_gaussians) == 2 else (_ for _ in ()).throw(AssertionError("unexpected init hparams request")))
    monkeypatch.setattr(session, "point_nn_scales", lambda pts: np.array([2.0, 4.0], dtype=np.float32) if np.array_equal(pts, positions[:2]) else (_ for _ in ()).throw(AssertionError("unexpected nn scale request")))

    result = session._sampled_point_init_hparams_from_positions(positions, 2, SimpleNamespace(), 0.25)

    assert result.position_jitter_std == 0.0
    assert result.base_scale == 0.75
    assert result.scale_jitter_ratio == 0.2
    assert result.initial_opacity == 0.3
    assert result.color_jitter_std == 0.0


def test_build_initial_training_scene_combines_fibonacci_as_separate_source(monkeypatch) -> None:
    point_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    point_colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    fibonacci_positions = np.array([[10.0, 0.0, 0.0], [10.0, 1.0, 0.0]], dtype=np.float32)
    fibonacci_colors = np.array([[0.8, 0.8, 0.8], [0.8, 0.8, 0.8]], dtype=np.float32)
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            colmap_recon=object(),
            colmap_root=Path("dataset/garden"),
            colmap_import=SimpleNamespace(
                init_mode="pointcloud",
                pointcloud_enabled=True,
                pointcloud_nn_radius_scale_coef=0.5,
                fibonacci_sphere_enabled=True,
                fibonacci_sphere_nn_radius_scale_coef=1.0,
                min_track_length=3,
            ),
            cached_init_pointcloud_positions=point_positions,
            cached_init_pointcloud_colors=point_colors,
            cached_init_fibonacci_positions=fibonacci_positions,
            cached_init_fibonacci_colors=fibonacci_colors,
            cached_init_scene=None,
            cached_init_signature=("cached",),
        )
    )

    monkeypatch.setattr(session, "_ensure_cached_init_source", lambda viewer_obj, init: None)
    point_scene = GaussianScene(
        positions=point_positions,
        scales=np.zeros((2, 3), dtype=np.float32),
        rotations=np.repeat(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), 2, axis=0),
        opacities=np.ones((2,), dtype=np.float32),
        colors=point_colors,
        sh_coeffs=point_colors[:, None, :],
    )
    fibonacci_scene = GaussianScene(
        positions=fibonacci_positions,
        scales=np.zeros((2, 3), dtype=np.float32),
        rotations=np.repeat(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), 2, axis=0),
        opacities=np.ones((2,), dtype=np.float32),
        colors=fibonacci_colors,
        sh_coeffs=fibonacci_colors[:, None, :],
    )
    monkeypatch.setattr(
        session,
        "_pointcloud_init_hparams_from_positions",
        lambda recon, positions, max_gaussians, init_hparams, nn_radius_scale_coef, min_track_length: SimpleNamespace(base_scale=0.5)
        if np.array_equal(positions, point_positions) and int(max_gaussians) == point_positions.shape[0]
        else (_ for _ in ()).throw(AssertionError("fibonacci source leaked into pointcloud initialization")),
    )
    monkeypatch.setattr(
        session,
        "_sampled_point_init_hparams_from_positions",
        lambda positions, max_gaussians, init_hparams, nn_radius_scale_coef: SimpleNamespace(base_scale=0.25)
        if np.array_equal(positions, fibonacci_positions) and int(max_gaussians) == fibonacci_positions.shape[0]
        else (_ for _ in ()).throw(AssertionError("unexpected fibonacci init request")),
    )
    monkeypatch.setattr(
        session,
        "initialize_scene_from_points_colors",
        lambda positions, colors, seed, init_hparams: point_scene
        if np.array_equal(positions, point_positions) and np.array_equal(colors, point_colors)
        else fibonacci_scene
        if np.array_equal(positions, fibonacci_positions) and np.array_equal(colors, fibonacci_colors)
        else (_ for _ in ()).throw(AssertionError("unexpected initializer data")),
    )

    scene, scale_reg_reference = session._build_initial_training_scene(viewer, SimpleNamespace(seed=7), SimpleNamespace(training=SimpleNamespace(max_gaussians=1)), SimpleNamespace())

    assert scene.count == 4
    assert np.array_equal(scene.positions, np.concatenate((point_positions, fibonacci_positions), axis=0))
    assert scale_reg_reference is None


@pytest.mark.parametrize(("source_key", "helper_name"), (("pointcloud", "_pointcloud_init_hparams_from_positions"), ("diffused", "_diffused_pointcloud_init_hparams_from_positions")))
def test_build_initial_training_scene_fibonacci_source_uses_own_nn_coef(monkeypatch, source_key: str, helper_name: str) -> None:
    base_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    base_colors = np.ones((2, 3), dtype=np.float32)
    fibonacci_positions = np.array([[10.0, 1.0, 0.0], [10.0, -1.0, 0.0], [11.0, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype=np.float32)
    fibonacci_colors = np.ones((4, 3), dtype=np.float32)

    def _viewer_with_coef(nn_radius_scale_coef: float) -> SimpleNamespace:
        import_cfg = {
            "init_mode": "pointcloud",
            "pointcloud_enabled": source_key == "pointcloud",
            "pointcloud_nn_radius_scale_coef": nn_radius_scale_coef,
            "diffused_enabled": source_key == "diffused",
            "diffused_nn_radius_scale_coef": nn_radius_scale_coef,
            "fibonacci_sphere_enabled": True,
            "fibonacci_sphere_nn_radius_scale_coef": 1.0,
            "min_track_length": 3,
        }
        return SimpleNamespace(
            s=SimpleNamespace(
                colmap_recon=object(),
                colmap_root=Path("dataset/garden"),
                colmap_import=SimpleNamespace(**import_cfg),
                cached_init_pointcloud_positions=base_positions if source_key == "pointcloud" else None,
                cached_init_pointcloud_colors=base_colors if source_key == "pointcloud" else None,
                cached_init_diffused_positions=base_positions if source_key == "diffused" else None,
                cached_init_diffused_colors=base_colors if source_key == "diffused" else None,
                cached_init_fibonacci_positions=fibonacci_positions,
                cached_init_fibonacci_colors=fibonacci_colors,
                cached_init_scene=None,
                cached_init_signature=("cached",),
            )
        )

    def _resolved_init(recon, positions, *args):
        del recon
        assert np.array_equal(positions, base_positions)
        nn_radius_scale_coef = float(args[-2])
        return SimpleNamespace(position_jitter_std=None, base_scale=nn_radius_scale_coef, scale_jitter_ratio=None, initial_opacity=None)

    monkeypatch.setattr(session, "_ensure_cached_init_source", lambda viewer_obj, init: None)
    monkeypatch.setattr(session, helper_name, _resolved_init)
    monkeypatch.setattr(
        session,
        "_sampled_point_init_hparams_from_positions",
        lambda positions, max_gaussians, init_hparams, nn_radius_scale_coef: SimpleNamespace(position_jitter_std=None, base_scale=float(nn_radius_scale_coef), scale_jitter_ratio=None, initial_opacity=None)
        if np.array_equal(positions, fibonacci_positions)
        else (_ for _ in ()).throw(AssertionError("unexpected fibonacci init request")),
    )

    low_scene, _ = session._build_initial_training_scene(_viewer_with_coef(0.1), SimpleNamespace(seed=7), SimpleNamespace(training=SimpleNamespace(max_gaussians=2)), SimpleNamespace())
    high_scene, _ = session._build_initial_training_scene(_viewer_with_coef(3.0), SimpleNamespace(seed=7), SimpleNamespace(training=SimpleNamespace(max_gaussians=2)), SimpleNamespace())

    np.testing.assert_allclose(np.exp(low_scene.scales[-4:, 0]), np.ones((4,), dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.exp(high_scene.scales[-4:, 0]), np.ones((4,), dtype=np.float32), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(low_scene.scales[-4:, :], high_scene.scales[-4:, :], rtol=0.0, atol=1e-6)


def test_apply_live_params_defers_subsample_runtime_change_until_resize(monkeypatch) -> None:
    update_calls: list[tuple[object, object, object]] = []

    def _params(train_subsample_factor: int) -> SimpleNamespace:
        return SimpleNamespace(
            adam=SimpleNamespace(__dataclass_fields__={"lr": None}, lr=1e-3),
            stability=SimpleNamespace(__dataclass_fields__={"eps": None}, eps=1e-8),
            training=SimpleNamespace(
                __dataclass_fields__={
                    "train_downscale_mode": None,
                    "train_auto_start_downscale": None,
                    "train_downscale_base_iters": None,
                    "train_downscale_iter_step": None,
                    "train_downscale_max_iters": None,
                    "train_downscale_factor": None,
                    "train_subsample_factor": None,
                },
                train_downscale_mode=1,
                train_auto_start_downscale=1,
                train_downscale_base_iters=200,
                train_downscale_iter_step=50,
                train_downscale_max_iters=30_000,
                train_downscale_factor=1,
                train_subsample_factor=train_subsample_factor,
            ),
        )

    params_before = _params(1)
    params_after = _params(2)
    viewer = SimpleNamespace(
        render_background=lambda: (0.0, 0.0, 0.0),
        renderer_params=lambda allow_debug_overlays: SimpleNamespace(__dataclass_fields__={"debug": None}, debug=bool(allow_debug_overlays)),
        training_params=lambda: SimpleNamespace(training=SimpleNamespace(sh_band=0, use_sh=False)),
        s=SimpleNamespace(
            background=None,
            renderer=None,
            training_renderer=None,
            debug_renderer=None,
            trainer=SimpleNamespace(compute_debug_grad_norm=False, update_hyperparams=lambda adam, stability, training: update_calls.append((adam, stability, training))),
            applied_renderer_params_main=None,
            applied_renderer_params_training=None,
            applied_renderer_params_debug=None,
            applied_training_signature=session._training_live_params_signature(params_before),
            applied_training_runtime_signature=session._training_runtime_signature(params_before),
            pending_training_runtime_resize=False,
        ),
    )

    monkeypatch.setattr(session, "resolve_effective_training_setup", lambda viewer_obj: (None, params_after, None, None))

    session.apply_live_params(viewer)

    assert update_calls == []
    assert viewer.s.applied_training_signature == session._training_live_params_signature(params_before)
    assert viewer.s.applied_training_runtime_signature == session._training_runtime_signature(params_before)
    assert viewer.s.pending_training_runtime_resize is True


def test_apply_live_params_syncs_renderer_sh_band(monkeypatch) -> None:
    update_calls: list[object] = []
    renderer = SimpleNamespace(sh_band=0, max_sh_band=3, use_sh=False, debug_show_grad_norm=False)
    training_renderer = SimpleNamespace(sh_band=3, max_sh_band=1, use_sh=True, debug_show_grad_norm=False)
    debug_renderer = SimpleNamespace(sh_band=0, max_sh_band=3, use_sh=False, debug_show_grad_norm=False)
    params = SimpleNamespace(
        adam=SimpleNamespace(),
        stability=SimpleNamespace(),
        training=SimpleNamespace(
            sh_band=0,
            use_sh=False,
            max_sh_band=1,
            train_downscale_mode="auto",
            train_auto_start_downscale=4,
            train_downscale_base_iters=200,
            train_downscale_iter_step=50,
            train_downscale_max_iters=30_000,
            train_downscale_factor=1,
            train_subsample_factor=1,
        ),
    )
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={"_viewport_sh_band": 3}),
        render_background=lambda: (0.0, 0.0, 0.0),
        renderer_params=lambda allow_debug_overlays: _renderer_params(debug=bool(allow_debug_overlays)),
        training_params=lambda: SimpleNamespace(training=SimpleNamespace(sh_band=0, use_sh=False)),
        s=SimpleNamespace(
            background=None,
            renderer=renderer,
            training_renderer=training_renderer,
            debug_renderer=debug_renderer,
            trainer=SimpleNamespace(compute_debug_grad_norm=False, update_hyperparams=lambda adam, stability, training: update_calls.append((adam, stability, training))),
            applied_renderer_params_main=None,
            applied_renderer_params_training=None,
            applied_renderer_params_debug=None,
            applied_training_signature=None,
            applied_training_runtime_signature=None,
            pending_training_runtime_resize=False,
        ),
    )

    monkeypatch.setattr(session, "resolve_effective_training_setup", lambda viewer_obj: (None, params, None, None))
    monkeypatch.setattr(session, "_apply_debug_buffers", lambda *_args: None)

    session.apply_live_params(viewer)

    assert viewer.s.renderer.sh_band == 3
    assert viewer.s.renderer.max_sh_band == 1
    assert viewer.s.debug_renderer.sh_band == 3
    assert viewer.s.debug_renderer.max_sh_band == 1
    assert viewer.s.training_renderer.sh_band == 0
    assert viewer.s.training_renderer.max_sh_band == 1
    assert len(update_calls) == 1


def test_apply_live_params_resets_existing_renderer_debug_mode_to_normal(monkeypatch) -> None:
    class _Params(SimpleNamespace):
        __dataclass_fields__ = {
            "debug_mode": None,
            "debug_show_ellipses": None,
            "debug_show_processed_count": None,
            "debug_show_grad_norm": None,
        }

        def renderer_kwargs(self) -> dict[str, object]:
            return {
                "debug_mode": self.debug_mode,
                "debug_show_ellipses": self.debug_show_ellipses,
                "debug_show_processed_count": self.debug_show_processed_count,
                "debug_show_grad_norm": self.debug_show_grad_norm,
            }

    renderer = SimpleNamespace(
        sh_band=0,
        debug_mode=session.GaussianRenderer.DEBUG_MODE_SPLAT_AGE,
        debug_show_ellipses=False,
        debug_show_processed_count=False,
        debug_show_grad_norm=False,
    )
    debug_renderer = SimpleNamespace(
        sh_band=0,
        debug_mode=session.GaussianRenderer.DEBUG_MODE_SPLAT_AGE,
        debug_show_ellipses=False,
        debug_show_processed_count=False,
        debug_show_grad_norm=False,
    )
    params = _Params(
        debug_mode=None,
        debug_show_ellipses=False,
        debug_show_processed_count=False,
        debug_show_grad_norm=False,
    )
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={"_viewport_sh_band": 0}),
        render_background=lambda: (0.0, 0.0, 0.0),
        renderer_params=lambda allow_debug_overlays: params,
        training_params=lambda: SimpleNamespace(training=SimpleNamespace(sh_band=0, use_sh=False)),
        s=SimpleNamespace(
            background=None,
            renderer=renderer,
            training_renderer=None,
            debug_renderer=debug_renderer,
            trainer=None,
            applied_renderer_params_main=None,
            applied_renderer_params_training=None,
            applied_renderer_params_debug=None,
        ),
    )

    monkeypatch.setattr(session, "_apply_debug_buffers", lambda *_args: None)

    session.apply_live_params(viewer)

    assert viewer.s.renderer.debug_mode == session.GaussianRenderer.DEBUG_MODE_NORMAL
    assert viewer.s.debug_renderer.debug_mode == session.GaussianRenderer.DEBUG_MODE_NORMAL


def test_apply_live_params_updates_refinement_exponents_without_renderer_signature_change(monkeypatch) -> None:
    renderer = SimpleNamespace(
        sh_band=0,
        debug_show_grad_norm=False,
        debug_refinement_grad_variance_weight_exponent=0.25,
        debug_refinement_contribution_weight_exponent=0.5,
    )
    debug_renderer = SimpleNamespace(
        sh_band=0,
        debug_show_grad_norm=False,
        debug_refinement_grad_variance_weight_exponent=0.25,
        debug_refinement_contribution_weight_exponent=0.5,
    )
    params = SimpleNamespace(
        adam=SimpleNamespace(),
        stability=SimpleNamespace(),
        training=SimpleNamespace(
            sh_band=0,
            use_sh=False,
            refinement_grad_variance_weight_exponent=1.5,
            refinement_contribution_weight_exponent=2.5,
            train_downscale_mode="auto",
            train_auto_start_downscale=4,
            train_downscale_base_iters=200,
            train_downscale_iter_step=50,
            train_downscale_max_iters=30_000,
            train_downscale_factor=1,
            train_subsample_factor=1,
        ),
    )
    renderer_signature = (True,)
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={"_viewport_sh_band": 3}),
        render_background=lambda: (0.0, 0.0, 0.0),
        renderer_params=lambda allow_debug_overlays: _renderer_params(debug=bool(allow_debug_overlays)),
        training_params=lambda: SimpleNamespace(training=SimpleNamespace(
            sh_band=0,
            use_sh=False,
            refinement_grad_variance_weight_exponent=1.5,
            refinement_contribution_weight_exponent=2.5,
        )),
        s=SimpleNamespace(
            background=None,
            renderer=renderer,
            training_renderer=None,
            debug_renderer=debug_renderer,
            trainer=None,
            applied_renderer_params_main=renderer_signature,
            applied_renderer_params_training=None,
            applied_renderer_params_debug=renderer_signature,
        ),
    )

    monkeypatch.setattr(session, "resolve_effective_training_setup", lambda viewer_obj: (None, params, None, None))
    monkeypatch.setattr(session, "_renderer_params_signature", lambda params_obj: renderer_signature)
    monkeypatch.setattr(session, "_apply_debug_buffers", lambda *_args: None)

    session.apply_live_params(viewer)

    assert viewer.s.renderer.debug_refinement_grad_variance_weight_exponent == 1.5
    assert viewer.s.renderer.debug_refinement_contribution_weight_exponent == 2.5
    assert viewer.s.debug_renderer.debug_refinement_grad_variance_weight_exponent == 1.5
    assert viewer.s.debug_renderer.debug_refinement_contribution_weight_exponent == 2.5


def test_apply_live_params_uses_viewport_sh_default_without_trainer(monkeypatch) -> None:
    renderer = SimpleNamespace(sh_band=0, debug_show_grad_norm=False)
    debug_renderer = SimpleNamespace(sh_band=0, debug_show_grad_norm=False)
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={"_viewport_sh_band": 3}),
        render_background=lambda: (0.0, 0.0, 0.0),
        renderer_params=lambda allow_debug_overlays: _renderer_params(debug=bool(allow_debug_overlays)),
        training_params=lambda: SimpleNamespace(training=SimpleNamespace(sh_band=0, use_sh=False)),
        s=SimpleNamespace(
            background=None,
            renderer=renderer,
            training_renderer=None,
            debug_renderer=debug_renderer,
            trainer=None,
            applied_renderer_params_main=None,
            applied_renderer_params_training=None,
            applied_renderer_params_debug=None,
        ),
    )

    monkeypatch.setattr(session, "_apply_debug_buffers", lambda *_args: None)

    session.apply_live_params(viewer)

    assert viewer.s.renderer.sh_band == 3
    assert viewer.s.debug_renderer.sh_band == 3
