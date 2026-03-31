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


def _write_cameras_bin(path: Path) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<i", 7))
        handle.write(struct.pack("<i", 1))
        handle.write(struct.pack("<Q", 64))
        handle.write(struct.pack("<Q", 64))
        handle.write(struct.pack("<dddd", 64.0, 64.0, 32.0, 32.0))


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
        handle.write(struct.pack("<Q", 0))


def _write_database(path: Path, image_names: list[str]) -> None:
    with sqlite3.connect(str(path)) as conn:
        conn.execute("CREATE TABLE images (image_id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        conn.executemany("INSERT INTO images(image_id, name) VALUES (?, ?)", list(enumerate(image_names, start=1)))
        conn.commit()


def _build_colmap_tree(tmp_path: Path, *, image_names: list[str], image_root_rel: Path) -> tuple[Path, Path]:
    root = tmp_path / "scene"
    sparse = root / "sparse" / "0"
    database_path = root / "distorted" / "database.db"
    images_root = root / image_root_rel
    sparse.mkdir(parents=True)
    database_path.parent.mkdir(parents=True)
    images_root.mkdir(parents=True, exist_ok=True)
    _write_cameras_bin(sparse / "cameras.bin")
    _write_images_bin(sparse / "images.bin", image_names)
    _write_points3d_bin(sparse / "points3D.bin")
    _write_database(database_path, image_names)
    for image_name in image_names:
        image_path = (images_root / image_name).resolve()
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(image_path)
    return database_path, images_root.resolve()


def _build_colmap_tree_without_database(tmp_path: Path, *, image_names: list[str], image_root_rel: Path) -> tuple[Path, Path]:
    root = tmp_path / "scene_no_db"
    sparse = root / "sparse" / "0"
    images_root = root / image_root_rel
    sparse.mkdir(parents=True)
    images_root.mkdir(parents=True, exist_ok=True)
    _write_cameras_bin(sparse / "cameras.bin")
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

        def set_debug_grad_norm_buffer(self, buffer) -> None:
            self.bound = buffer

    new_renderer = SimpleNamespace(width=32, height=32, work_buffers={"debug_grad_norm": "grad_norm"})
    trainer = SimpleNamespace(
        compute_debug_grad_norm=True,
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
    assert viewer.s.last_error == ""


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

def test_advance_colmap_import_processes_images_incrementally(tmp_path: Path, monkeypatch) -> None:
    _, images_root = _build_colmap_tree(
        tmp_path,
        image_names=["frame_000.png", "frame_001.png"],
        image_root_rel=Path("images"),
    )
    recon = SimpleNamespace(
        images={
            1: SimpleNamespace(name="frame_000.png", camera_id=7, q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), t_xyz=np.array([0.0, 0.0, -2.0], dtype=np.float32)),
            2: SimpleNamespace(name="frame_001.png", camera_id=7, q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), t_xyz=np.array([0.0, 0.0, -2.0], dtype=np.float32)),
        },
        cameras={7: SimpleNamespace(width=8, height=8, fx=64.0, fy=64.0, cx=4.0, cy=4.0)},
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
                image_downscale_target_width=1600,
                image_downscale_scale=1.0,
                nn_radius_scale_coef=0.25,
            ),
        ),
    )

    monkeypatch.setattr(session, "load_colmap_reconstruction", lambda root: recon)
    monkeypatch.setattr(session, "_create_native_dataset_texture", lambda viewer_obj, image_path, *, target_size=None: f"tex:{Path(image_path).name}")

    def _finish(viewer_obj, **kwargs) -> None:
        calls.append(("finish", len(kwargs["training_frames"]), list(kwargs["frame_targets_native"])))
        viewer_obj.s.colmap_import_progress = None

    monkeypatch.setattr(session, "_finish_import_colmap_dataset", _finish)

    while viewer.s.colmap_import_progress is not None:
        session.advance_colmap_import(viewer)

    assert calls == [
        ("finish", 2, ["tex:frame_000.png", "tex:frame_001.png"]),
        "close",
    ]


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
                image_downscale_mode="width",
                image_downscale_target_width=4,
                image_downscale_scale=1.0,
                nn_radius_scale_coef=0.25,
            ),
        ),
    )

    monkeypatch.setattr(session, "load_colmap_reconstruction", lambda root: recon)
    monkeypatch.setattr(session, "_create_native_dataset_texture", lambda viewer_obj, image_path, *, target_size=None: calls.append((Path(image_path).name, target_size)) or "tex")

    def _finish(viewer_obj, **kwargs) -> None:
        frame = kwargs["training_frames"][0]
        calls.append((frame.width, frame.height, frame.fx, frame.fy, kwargs["frame_targets_native"]))
        viewer_obj.s.colmap_import_progress = None

    monkeypatch.setattr(session, "_finish_import_colmap_dataset", _finish)

    while viewer.s.colmap_import_progress is not None:
        session.advance_colmap_import(viewer)

    assert calls == [
        ("frame_000.png", (4, 2)),
        (4, 2, 40.0, 20.0, ["tex"]),
        "close",
    ]


def test_finish_import_colmap_dataset_resets_toolkit_plot_history(monkeypatch) -> None:
    recon = SimpleNamespace(points3d={1: object()})
    monkeypatch.setattr(session, "_point_tables", lambda recon_obj: (np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.float32)))
    monkeypatch.setattr(session, "_reset_loaded_runtime", lambda viewer_obj: None)
    monkeypatch.setattr(session, "_update_import_settings", lambda viewer_obj, **kwargs: None)
    monkeypatch.setattr(session, "apply_live_params", lambda viewer_obj: None)
    monkeypatch.setattr(session, "estimate_point_bounds", lambda xyz: xyz)
    monkeypatch.setattr(session, "initialize_training_scene", lambda viewer_obj, frame_targets_native=None: None)
    calls: list[str] = []
    viewer = SimpleNamespace(
        toolkit=SimpleNamespace(reset_plot_history=lambda: calls.append("reset")),
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
        image_downscale_target_width=1600,
        image_downscale_scale=1.0,
        nn_radius_scale_coef=0.25,
        diffused_point_count=100000,
        diffusion_radius=1.0,
        recon=recon,
        training_frames=[],
        frame_targets_native=None,
    )

    assert calls == ["reset", "fit"]


def test_colmap_import_settings_defaults_prefer_diffused_pointcloud() -> None:
    defaults = ColmapImportSettings()

    assert defaults.init_mode == "diffused_pointcloud"
    assert defaults.nn_radius_scale_coef == 0.5


def test_refresh_cached_raster_grad_histograms_caches_by_signature() -> None:
    calls: list[tuple[int, int, float, float]] = []
    hist = SimpleNamespace(counts=np.ones((14, 8), dtype=np.int64), param_labels=("p",) * 14)
    ranges = SimpleNamespace(min_values=np.full((14,), -1.0, dtype=np.float32), max_values=np.full((14,), 2.0, dtype=np.float32), param_labels=("p",) * 14)
    renderer = SimpleNamespace(
        cached_raster_grad_atomic_mode="float",
        compute_cached_raster_grad_component_histograms=lambda metrics, scene_count, *, bin_count, min_log10, max_log10: calls.append((scene_count, bin_count, min_log10, max_log10)) or hist,
        compute_cached_raster_grad_component_ranges=lambda metrics, scene_count: ranges,
    )
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={"hist_bin_count": 8, "hist_min_log10": -7.0, "hist_max_log10": 1.0, "hist_auto_refresh": True, "_histograms_refresh_requested": False}),
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


def test_refresh_cached_raster_grad_histograms_honors_manual_refresh() -> None:
    calls: list[int] = []
    renderer = SimpleNamespace(
        cached_raster_grad_atomic_mode="fixed",
        compute_cached_raster_grad_component_histograms=lambda metrics, scene_count, *, bin_count, min_log10, max_log10: calls.append(scene_count) or SimpleNamespace(counts=np.zeros((14, bin_count), dtype=np.int64), param_labels=()),
        compute_cached_raster_grad_component_ranges=lambda metrics, scene_count: SimpleNamespace(min_values=np.zeros((14,), dtype=np.float32), max_values=np.zeros((14,), dtype=np.float32), param_labels=()),
    )
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={"hist_bin_count": 4, "hist_min_log10": -8.0, "hist_max_log10": 2.0, "hist_auto_refresh": False, "_histograms_refresh_requested": True}),
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
