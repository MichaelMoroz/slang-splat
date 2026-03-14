from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.viewer import session


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
