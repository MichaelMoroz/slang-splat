from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

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


def test_set_training_active_accumulates_elapsed_time_on_pause(monkeypatch) -> None:
    viewer = _viewer()
    times = iter((10.0, 14.5))
    monkeypatch.setattr(session.time, "perf_counter", lambda: next(times))

    session.set_training_active(viewer, True)
    session.set_training_active(viewer, False)

    assert viewer.s.training_active is False
    assert viewer.s.training_resume_time is None
    assert viewer.s.training_elapsed_s == 4.5


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

    class _MainRenderer:
        def __init__(self) -> None:
            self.bound = None
            self.contribution_bound = None
            self.contribution_pixels = None

        def set_debug_grad_norm_buffer(self, buffer) -> None:
            self.bound = buffer

        def set_debug_clone_count_buffer(self, buffer) -> None:
            del buffer

        def set_debug_splat_contribution_buffer(self, buffer) -> None:
            self.contribution_bound = buffer

        def set_debug_contribution_observed_pixel_count(self, value) -> None:
            self.contribution_pixels = value

    new_renderer = SimpleNamespace(width=32, height=32, work_buffers={"debug_grad_norm": "grad_norm"})
    trainer = SimpleNamespace(
        compute_debug_grad_norm=True,
        refinement_buffers={"splat_contribution": "contrib"},
        observed_contribution_pixel_count=2048,
        effective_train_downscale_factor=lambda: 2,
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
    assert viewer.s.renderer.contribution_bound == "contrib"
    assert viewer.s.renderer.contribution_pixels == 2048
    assert calls == [
        ("copy", new_renderer),
        ("submit", "finished"),
        ("rebind", new_renderer),
        ("invalidate", ()),
        ("reset_loss_debug", None),
    ]


def test_ensure_renderer_keeps_existing_main_renderer_when_replacement_fails(monkeypatch) -> None:
    existing_renderer = SimpleNamespace(width=64, height=64)
    viewer = SimpleNamespace(
        device=SimpleNamespace(),
        renderer_params=lambda allow_debug_overlays: SimpleNamespace(),
        s=SimpleNamespace(renderer=existing_renderer, scene=None),
    )

    monkeypatch.setattr(session, "renderer_kwargs", lambda params: {})

    class _FailingSettings:
        def __init__(self, width: int, height: int, **kwargs) -> None:
            del width, height, kwargs

        def create_renderer(self, device) -> object:
            del device
            raise RuntimeError("renderer create failed")

    monkeypatch.setattr(session, "GaussianRenderSettings", _FailingSettings)

    with pytest.raises(RuntimeError, match="renderer create failed"):
        session.ensure_renderer(viewer, "renderer", 128, 72, allow_debug_overlays=True)

    assert viewer.s.renderer is existing_renderer


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
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={}),
        s=SimpleNamespace(last_error="stale"),
    )

    session.choose_colmap_root(viewer, root)

    assert viewer.ui._values["colmap_root_path"] == str(root)
    assert viewer.ui._values["colmap_database_path"] == ""
    assert viewer.ui._values["colmap_images_root"] == str(root)
    assert viewer.s.last_error == ""


def test_choose_colmap_root_rejects_unknown_camera_model(tmp_path: Path) -> None:
    database_path, _ = _build_colmap_tree(
        tmp_path,
        image_names=["frame_000.png"],
        image_root_rel=Path("images"),
        model_id=4,
    )
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={}),
        s=SimpleNamespace(last_error="stale"),
    )

    with pytest.raises(ValueError, match="Unsupported COLMAP camera model id 4"):
        session.choose_colmap_root(viewer, database_path.parents[1])


def test_import_colmap_dataset_clears_loaded_scene_before_loading(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            renderer=SimpleNamespace(
                clear_scene_resources=lambda: calls.append(("clear_renderer", None)),
                set_debug_grad_norm_buffer=lambda buffer: calls.append(("clear_grad_debug", buffer)),
                set_debug_clone_count_buffer=lambda buffer: calls.append(("clear_clone_debug", buffer)),
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
    monkeypatch.setattr(session, "_load_aligned_colmap_reconstruction", lambda root: calls.append(("load_recon", root)) or "recon")
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
        diffusion_radius=1.0,
    )

    assert calls[:8] == [
        ("clear_grad_debug", None),
        ("clear_clone_debug", None),
        ("reset_training_visual", None),
        ("reset_loss_debug", None),
        ("clear_cached_init", None),
        ("update_slider", None),
        ("clear_renderer", None),
        ("load_recon", Path("dataset/new").resolve()),
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
                "colmap_diffusion_radius": 1.0,
                "colmap_selected_camera_ids": (7,),
                "_colmap_camera_rows": ({"camera_id": 7, "frame_count": 1},),
                "use_target_alpha_mask": True,
            }
        ),
        s=SimpleNamespace(
            renderer=SimpleNamespace(
                clear_scene_resources=lambda: calls.append(("clear_renderer", None)),
                set_debug_grad_norm_buffer=lambda buffer: calls.append(("clear_grad_debug", buffer)),
                set_debug_clone_count_buffer=lambda buffer: calls.append(("clear_clone_debug", buffer)),
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
        ("clear_clone_debug", None),
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
                "colmap_diffusion_radius": 1.0,
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

    monkeypatch.setattr(session, "_load_aligned_colmap_reconstruction", lambda root: recon)
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

    monkeypatch.setattr(session, "_load_aligned_colmap_reconstruction", lambda root: recon)
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
        diffusion_radius=1.0,
        recon=recon,
        training_frames=[],
        frame_targets_native=None,
    )

    assert calls == ["reset", "fit"]


def test_finish_import_colmap_dataset_prefers_training_view_camera_fit(monkeypatch) -> None:
    recon = SimpleNamespace(points3d={1: object()})
    training_frames = [SimpleNamespace()]
    monkeypatch.setattr(session, "_point_tables", lambda recon_obj, min_track_length=3: (np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)))
    monkeypatch.setattr(session, "_reset_loaded_runtime", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_update_import_settings", lambda viewer_obj, **kwargs: None)
    monkeypatch.setattr(session, "apply_live_params", lambda viewer_obj: None)
    monkeypatch.setattr(session, "estimate_point_bounds", lambda xyz: (_ for _ in ()).throw(AssertionError("bounds fit should not be used when training-view fit succeeds")))
    monkeypatch.setattr(session, "initialize_training_scene", lambda viewer_obj, frame_targets_native=None: None)
    calls: list[object] = []
    viewer = SimpleNamespace(
        toolkit=SimpleNamespace(reset_plot_history=lambda: calls.append("reset")),
        ui=SimpleNamespace(_values={}),
        s=SimpleNamespace(),
        apply_camera_fit=lambda bounds: calls.append(("fit", bounds)),
        apply_camera_fit_to_training_views=lambda frames: calls.append(("view_fit", tuple(frames))) or True,
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
        diffusion_radius=1.0,
        recon=recon,
        training_frames=training_frames,
        frame_targets_native=None,
    )

    assert calls == ["reset", ("view_fit", tuple(training_frames))]


def test_import_colmap_dataset_uses_aligned_reconstruction(monkeypatch) -> None:
    aligned_recon = object()
    frames = [SimpleNamespace(width=32, height=32, image_id=1)]
    calls: list[object] = []
    viewer = SimpleNamespace(
        s=SimpleNamespace(
            renderer=SimpleNamespace(
                clear_scene_resources=lambda: calls.append(("clear_renderer", None)),
                set_debug_grad_norm_buffer=lambda buffer: calls.append(("clear_grad_debug", buffer)),
                set_debug_clone_count_buffer=lambda buffer: calls.append(("clear_clone_debug", buffer)),
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

    monkeypatch.setattr(session, "_load_aligned_colmap_reconstruction", lambda root: aligned_recon)
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
        custom_ply_path=None,
        image_downscale_mode="original",
        image_downscale_max_size=2048,
        image_downscale_scale=1.0,
        nn_radius_scale_coef=0.5,
        diffused_point_count=100000,
        diffusion_radius=1.0,
    )

    assert calls[:3] == [
        ("clear_grad_debug", None),
        ("clear_clone_debug", None),
        ("clear_renderer", None),
    ]
    assert calls[-2:] == [
        ("frames", aligned_recon, Path("dataset/garden/images_8"), (), "original", 2048, 1.0),
        ("finish", aligned_recon, frames, ["tex0"]),
    ]


def test_colmap_import_settings_defaults_prefer_pointcloud() -> None:
    defaults = ColmapImportSettings()

    assert defaults.init_mode == "pointcloud"
    assert defaults.nn_radius_scale_coef == 0.5
    assert defaults.min_track_length == 3
    assert defaults.depth_root is None
    assert defaults.selected_camera_ids == ()
    assert defaults.depth_value_mode == "z_depth"
    assert defaults.depth_point_count == 100000
    assert defaults.use_target_alpha_mask is False


def test_refresh_cached_raster_grad_histograms_requires_explicit_request() -> None:
    calls: list[tuple[int, int, float, float]] = []
    hist = SimpleNamespace(counts=np.ones((14, 8), dtype=np.int64), param_labels=("p",) * 14)
    ranges = SimpleNamespace(min_values=np.full((14,), -1.0, dtype=np.float32), max_values=np.full((14,), 2.0, dtype=np.float32), param_labels=("p",) * 14)
    renderer = SimpleNamespace(
        cached_raster_grad_atomic_mode="float",
        compute_cached_raster_grad_component_histograms=lambda metrics, scene_count, *, bin_count, min_log10, max_log10: calls.append((scene_count, bin_count, min_log10, max_log10)) or hist,
        compute_cached_raster_grad_component_ranges=lambda metrics, scene_count: ranges,
    )
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={"hist_bin_count": 8, "hist_min_log10": -7.0, "hist_max_log10": 1.0, "_histograms_refresh_requested": True}),
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
    session.refresh_cached_raster_grad_histograms(viewer)

    assert calls == [(32, 8, -7.0, 1.0)]
    assert viewer.s.cached_raster_grad_histograms is hist
    assert viewer.s.cached_raster_grad_ranges is ranges
    assert viewer.s.cached_raster_grad_histogram_signature == (3, "float", 32, 8, -7.0, 1.0)


def test_refresh_cached_raster_grad_histograms_skips_without_request() -> None:
    calls: list[int] = []
    renderer = SimpleNamespace(
        cached_raster_grad_atomic_mode="float",
        compute_cached_raster_grad_component_histograms=lambda metrics, scene_count, *, bin_count, min_log10, max_log10: calls.append(scene_count),
        compute_cached_raster_grad_component_ranges=lambda metrics, scene_count: None,
    )
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={"hist_bin_count": 8, "hist_min_log10": -7.0, "hist_max_log10": 1.0, "_histograms_refresh_requested": False}),
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
        compute_cached_raster_grad_component_histograms=lambda metrics, scene_count, *, bin_count, min_log10, max_log10: calls.append(scene_count) or SimpleNamespace(counts=np.zeros((14, bin_count), dtype=np.int64), param_labels=()),
        compute_cached_raster_grad_component_ranges=lambda metrics, scene_count: SimpleNamespace(min_values=np.zeros((14,), dtype=np.float32), max_values=np.zeros((14,), dtype=np.float32), param_labels=()),
    )
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={"hist_bin_count": 4, "hist_min_log10": -8.0, "hist_max_log10": 2.0, "_histograms_refresh_requested": True}),
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


def test_refresh_cached_raster_grad_histograms_appends_sh_ranges_when_available() -> None:
    hist = SimpleNamespace(counts=np.ones((14, 4), dtype=np.int64), param_labels=("g",) * 14)
    grad_ranges = SimpleNamespace(
        min_values=np.array([-1.0, -2.0], dtype=np.float32),
        max_values=np.array([3.0, 4.0], dtype=np.float32),
        param_labels=("grad0", "grad1"),
    )
    sh_ranges = SimpleNamespace(
        min_values=np.array([-0.1, -0.2, -0.3], dtype=np.float32),
        max_values=np.array([0.4, 0.5, 0.6], dtype=np.float32),
        param_labels=("sh0.r", "sh0.g", "sh0.b"),
    )
    renderer = SimpleNamespace(
        cached_raster_grad_atomic_mode="float",
        compute_cached_raster_grad_component_histograms=lambda metrics, scene_count, *, bin_count, min_log10, max_log10: hist,
        compute_cached_raster_grad_component_ranges=lambda metrics, scene_count: grad_ranges,
        compute_sh_component_ranges=lambda scene_count: sh_ranges,
    )
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={"hist_bin_count": 4, "hist_min_log10": -8.0, "hist_max_log10": 2.0, "_histograms_refresh_requested": True}),
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

    np.testing.assert_allclose(viewer.s.cached_raster_grad_ranges.min_values, np.array([-1.0, -2.0, -0.1, -0.2, -0.3], dtype=np.float32))
    np.testing.assert_allclose(viewer.s.cached_raster_grad_ranges.max_values, np.array([3.0, 4.0, 0.4, 0.5, 0.6], dtype=np.float32))
    assert viewer.s.cached_raster_grad_ranges.param_labels == ("grad0", "grad1", "sh0.r", "sh0.g", "sh0.b")


def test_initialize_training_scene_rebinds_debug_buffers_for_new_trainer(monkeypatch) -> None:
    class _Encoder:
        def finish(self) -> str:
            return "finished"

    class _ViewportRenderer:
        def __init__(self) -> None:
            self.grad_buffer = "stale-grad"
            self.clone_buffer = "stale-clone"
            self.copy_targets: list[object] = []

        def set_debug_grad_norm_buffer(self, buffer) -> None:
            self.grad_buffer = buffer

        def set_debug_clone_count_buffer(self, buffer) -> None:
            self.clone_buffer = buffer

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
        refinement_buffers={"clone_counts": "new-clone"},
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
                diffusion_radius=1.0,
            ),
            trainer=SimpleNamespace(refinement_buffers={"clone_counts": "old-clone"}),
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
    assert main_renderer.clone_buffer == "new-clone"
    assert debug_renderer.grad_buffer == "new-grad"
    assert debug_renderer.clone_buffer == "new-clone"
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
    main_renderer = SimpleNamespace(set_debug_grad_norm_buffer=lambda buffer: None, set_debug_clone_count_buffer=lambda buffer: None)
    debug_renderer = SimpleNamespace(set_debug_grad_norm_buffer=lambda buffer: None, set_debug_clone_count_buffer=lambda buffer: None)
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
                diffusion_radius=1.0,
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

    monkeypatch.setattr(session, "_load_aligned_colmap_reconstruction", lambda root: (_ for _ in ()).throw(AssertionError("should not reload reconstruction")))
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
            colmap_import=SimpleNamespace(init_mode="diffused_pointcloud", nn_radius_scale_coef=0.5, min_track_length=3),
            cached_init_point_positions=cached_positions,
            cached_init_point_colors=cached_colors,
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
    assert scale_reg_reference == 0.25


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
                    "depth_ratio_weight": None,
                },
                train_downscale_mode=1,
                train_auto_start_downscale=1,
                train_downscale_base_iters=200,
                train_downscale_iter_step=50,
                train_downscale_max_iters=30_000,
                train_downscale_factor=1,
                train_subsample_factor=train_subsample_factor,
                depth_ratio_weight=0.1,
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
    renderer = SimpleNamespace(sh_band=0, use_sh=False, debug_show_grad_norm=False)
    training_renderer = SimpleNamespace(sh_band=3, use_sh=True, debug_show_grad_norm=False)
    debug_renderer = SimpleNamespace(sh_band=0, use_sh=False, debug_show_grad_norm=False)
    params = SimpleNamespace(
        adam=SimpleNamespace(),
        stability=SimpleNamespace(),
        training=SimpleNamespace(
            sh_band=0,
            use_sh=False,
            train_downscale_mode="auto",
            train_auto_start_downscale=4,
            train_downscale_base_iters=200,
            train_downscale_iter_step=50,
            train_downscale_max_iters=30_000,
            train_downscale_factor=1,
            train_subsample_factor=1,
            depth_ratio_weight=0.1,
        ),
    )
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={"_viewport_sh_band": 3}),
        render_background=lambda: (0.0, 0.0, 0.0),
        renderer_params=lambda allow_debug_overlays: SimpleNamespace(__dataclass_fields__={"debug": None}, debug=bool(allow_debug_overlays)),
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
    monkeypatch.setattr(session, "renderer_kwargs", lambda *_args: {})
    monkeypatch.setattr(session, "_apply_debug_buffers", lambda *_args: None)

    session.apply_live_params(viewer)

    assert viewer.s.renderer.sh_band == 3
    assert viewer.s.debug_renderer.sh_band == 3
    assert viewer.s.training_renderer.sh_band == 0
    assert len(update_calls) == 1


def test_apply_live_params_uses_viewport_sh_default_without_trainer(monkeypatch) -> None:
    renderer = SimpleNamespace(sh_band=0, debug_show_grad_norm=False)
    debug_renderer = SimpleNamespace(sh_band=0, debug_show_grad_norm=False)
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={"_viewport_sh_band": 3}),
        render_background=lambda: (0.0, 0.0, 0.0),
        renderer_params=lambda allow_debug_overlays: SimpleNamespace(__dataclass_fields__={"debug": None}, debug=bool(allow_debug_overlays)),
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

    monkeypatch.setattr(session, "renderer_kwargs", lambda *_args: {})
    monkeypatch.setattr(session, "_apply_debug_buffers", lambda *_args: None)

    session.apply_live_params(viewer)

    assert viewer.s.renderer.sh_band == 3
    assert viewer.s.debug_renderer.sh_band == 3
