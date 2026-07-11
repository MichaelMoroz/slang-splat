from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.scene import GaussianScene, load_gaussian_ply, save_gaussian_ply
from src.viewer import project
from src.viewer import ui
from src.viewer.config import apply_config_overlay
from src.viewer.state import ProjectState

from tests.test_viewer_ui import _dummy_renderer


def _scene(count: int = 4) -> GaussianScene:
    rotations = np.zeros((count, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    return GaussianScene(
        positions=np.linspace(0.0, 1.0, count * 3, dtype=np.float32).reshape(count, 3),
        scales=np.full((count, 3), 0.1, dtype=np.float32),
        rotations=rotations,
        opacities=np.full((count,), 0.5, dtype=np.float32),
        colors=np.full((count, 3), 0.25, dtype=np.float32),
        sh_coeffs=np.full((count, 1, 3), 0.125, dtype=np.float32),
    )


class _Trainer:
    def __init__(self, step: int = 0):
        self.state = SimpleNamespace(step=step)
        self.scene = SimpleNamespace(count=4)
        self.adam = object()
        self.stability = object()
        self.training = object()
        self.replaced_scenes: list[GaussianScene] = []
        self.hyperparam_updates = 0

    def replace_scene(self, scene: GaussianScene) -> None:
        self.replaced_scenes.append(scene)
        self.scene = SimpleNamespace(count=int(scene.count))

    def update_hyperparams(self, adam, stability, training) -> None:
        self.hyperparam_updates += 1


def _viewer(
    *,
    values: dict | None = None,
    trainer: _Trainer | None = None,
    colmap_root: Path | None = None,
    images_root: Path | None = None,
    scene: GaussianScene | None = None,
):
    viewer_ui = ui.build_ui(_dummy_renderer())
    if values:
        viewer_ui._values.update(values)
    state = SimpleNamespace(
        project=ProjectState(),
        trainer=trainer,
        colmap_root=colmap_root,
        colmap_import=SimpleNamespace(images_root=images_root),
        training_elapsed_s=12.5,
        training_active=False,
        training_resume_time=None,
        last_error="",
        scene=None,
    )
    viewer = SimpleNamespace(ui=viewer_ui, s=state)
    if scene is not None:
        viewer._export_source_scene = lambda: scene
        viewer._export_should_include_sh = lambda: True
    return viewer


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "garden"
    images = root / "images_4"
    images.mkdir(parents=True)
    (root / "sparse" / "0").mkdir(parents=True)
    (images / "0001.png").write_bytes(b"not-really-a-png")
    return root, images


# --- config snapshot round trip -------------------------------------------


def test_save_project_round_trips_config(tmp_path: Path) -> None:
    root, images = _dataset(tmp_path)
    viewer = _viewer(colmap_root=root, images_root=images, scene=_scene(), trainer=_Trainer(step=42))
    viewer.ui._values["colmap_diffused_point_count"] = 123456
    viewer.ui._values["train_iters"] = 777
    viewer.ui._values["colmap_root_path"] = str(root)
    viewer.ui._values["colmap_images_root"] = str(images)

    target = project.save_project(viewer, tmp_path / "proj.splatproj")
    project.join_pending_writes(viewer)
    manifest, project_dir = project.load_project_manifest(target)

    assert project_dir == target
    assert manifest["format"] == "splatproj"
    assert manifest["schema_version"] == project.PROJECT_SCHEMA_VERSION
    # viewer.ui / viewer.state are machine-local and must not be captured.
    assert "ui" not in manifest["config"]["viewer"]
    assert "state" not in manifest["config"]["viewer"]

    fresh = ui.build_ui(_dummy_renderer())
    assert fresh._values["colmap_diffused_point_count"] != 123456
    apply_config_overlay(fresh._values, project.project_config_overlay(manifest))
    assert fresh._values["colmap_diffused_point_count"] == 123456
    assert int(fresh._values["train_iters"]) == 777
    assert str(fresh._values["colmap_root_path"]) == str(root)


def test_save_project_writes_scene_result_and_dataset(tmp_path: Path) -> None:
    root, images = _dataset(tmp_path)
    scene = _scene(count=6)
    viewer = _viewer(colmap_root=root, images_root=images, scene=scene, trainer=_Trainer(step=30000))

    target = project.save_project(viewer, root / "slang_splat")
    project.join_pending_writes(viewer)
    manifest, _ = project.load_project_manifest(target)

    assert (target / project.PROJECT_SCENE_NAME).is_file()
    assert not (target / (project.PROJECT_SCENE_NAME + ".tmp")).exists()
    loaded = load_gaussian_ply(target / project.PROJECT_SCENE_NAME)
    assert int(loaded.count) == 6
    assert manifest["result"]["step"] == 30000
    assert manifest["result"]["splat_count"] == 6
    assert manifest["result"]["train_elapsed_s"] == pytest.approx(12.5)
    dataset = manifest["dataset"]
    assert dataset["mode"] == "reference"
    assert dataset["colmap_root_rel"] == ".."
    assert dataset["images_root_rel"] == "../images_4"
    assert dataset["fingerprint"]["files"] == 1
    assert manifest["scene"]["objects"][0]["ply"] == "scene.ply"
    assert viewer.s.project.dir == target
    assert viewer.s.project.saved_step == 30000


def test_config_forward_compat_missing_and_unknown_keys(tmp_path: Path) -> None:
    root, images = _dataset(tmp_path)
    viewer = _viewer(colmap_root=root, images_root=images, scene=_scene(), trainer=_Trainer())
    target = project.save_project(viewer, tmp_path / "proj.splatproj")
    project.join_pending_writes(viewer)

    manifest_path = target / project.PROJECT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline = int(ui.build_ui(_dummy_renderer())._values["colmap_diffused_point_count"])
    # Deleted key -> current default applies; unknown key -> ignored gracefully.
    del manifest["config"]["viewer"]["import"]["colmap_diffused_point_count"]
    manifest["config"]["training_build_args"]["nonexistent_future_knob"] = 1.25
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded, _ = project.load_project_manifest(target)
    fresh = ui.build_ui(_dummy_renderer())
    apply_config_overlay(fresh._values, project.project_config_overlay(loaded))
    assert int(fresh._values["colmap_diffused_point_count"]) == baseline
    assert "nonexistent_future_knob" not in fresh._values


def test_migrate_rejects_foreign_and_newer_manifests() -> None:
    with pytest.raises(ValueError):
        project.migrate_project_manifest({"format": "other"})
    with pytest.raises(ValueError):
        project.migrate_project_manifest({"format": "splatproj", "schema_version": project.PROJECT_SCHEMA_VERSION + 1})


# --- mandatory creation on import -----------------------------------------


def test_on_import_completed_creates_default_project(tmp_path: Path) -> None:
    root, images = _dataset(tmp_path)
    trainer = _Trainer(step=0)
    viewer = _viewer(colmap_root=root, images_root=images, trainer=trainer)

    project.on_import_completed(viewer)
    project.join_pending_writes(viewer)

    manifest_path = root / "slang_splat" / project.PROJECT_MANIFEST_NAME
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["result"]["step"] == 0
    # scene.ply appears at the first save/autosave, not at creation.
    assert not (root / "slang_splat" / project.PROJECT_SCENE_NAME).exists()
    assert viewer.s.project.dir == root / "slang_splat"
    assert viewer.ui._values["_project_active"] is True
    assert viewer.ui._values["_project_name"] == "garden"


def test_on_import_completed_prefers_pending_dir_and_keeps_active_project(tmp_path: Path) -> None:
    root, images = _dataset(tmp_path)
    viewer = _viewer(colmap_root=root, images_root=images, trainer=_Trainer())
    chosen = tmp_path / "elsewhere.splatproj"
    viewer.s.project.pending_dir = chosen

    project.on_import_completed(viewer)
    project.join_pending_writes(viewer)
    assert (chosen / project.PROJECT_MANIFEST_NAME).is_file()
    assert viewer.s.project.dir == chosen

    # A later import completion with an active project must not re-create.
    viewer.s.project.pending_dir = None
    (chosen / project.PROJECT_MANIFEST_NAME).unlink()
    project.on_import_completed(viewer)
    project.join_pending_writes(viewer)
    assert not (chosen / project.PROJECT_MANIFEST_NAME).exists()
    assert not (root / "slang_splat" / project.PROJECT_MANIFEST_NAME).exists()


def test_on_import_completed_consumes_pending_resume(tmp_path: Path) -> None:
    root, images = _dataset(tmp_path)
    trainer = _Trainer(step=0)
    viewer = _viewer(colmap_root=root, images_root=images, trainer=trainer)
    scene_ply = tmp_path / "saved.ply"
    save_gaussian_ply(scene_ply, _scene(count=5))
    viewer.s.project.dir = root / "slang_splat"
    viewer.s.project.pending_resume = {"scene_ply": scene_ply, "step": 12345, "elapsed_s": 99.0}

    project.on_import_completed(viewer)

    assert trainer.state.step == 12345
    assert len(trainer.replaced_scenes) == 1
    assert int(trainer.replaced_scenes[0].count) == 5
    assert trainer.hyperparam_updates == 1
    assert viewer.s.training_elapsed_s == pytest.approx(99.0)
    assert viewer.s.project.pending_resume is None


def test_request_colmap_import_collision_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, images = _dataset(tmp_path)
    existing = root / "slang_splat"
    existing.mkdir()
    (existing / project.PROJECT_MANIFEST_NAME).write_text("{}", encoding="utf-8")
    viewer = _viewer(colmap_root=None, images_root=None)
    viewer.ui._values["colmap_root_path"] = str(root)

    started: list[object] = []
    import src.viewer.session as session_mod

    monkeypatch.setattr(session_mod, "import_colmap_from_ui", lambda v: started.append(v))

    assert project.request_colmap_import(viewer) == "prompt"
    assert viewer.ui._values["_project_collision_open"] is True
    assert viewer.s.project.pending_import_dir == existing
    assert started == []

    project.collision_choice(viewer, "fresh")
    assert started == [viewer]
    assert viewer.s.project.pending_dir == existing
    assert viewer.ui._values["_project_collision_open"] is False


def test_request_colmap_import_without_collision_starts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, images = _dataset(tmp_path)
    viewer = _viewer(colmap_root=None, images_root=None)
    viewer.ui._values["colmap_root_path"] = str(root)

    started: list[object] = []
    import src.viewer.session as session_mod

    monkeypatch.setattr(session_mod, "import_colmap_from_ui", lambda v: started.append(v))

    assert project.request_colmap_import(viewer) == "started"
    assert started == [viewer]
    assert viewer.s.project.pending_dir == root / "slang_splat"


def test_request_colmap_import_missing_root_skips_project_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    viewer = _viewer()
    viewer.ui._values["colmap_root_path"] = str(tmp_path / "does-not-exist")
    import src.viewer.session as session_mod

    started: list[object] = []
    monkeypatch.setattr(session_mod, "import_colmap_from_ui", lambda v: started.append(v))
    assert project.request_colmap_import(viewer) == "started"
    assert started == [viewer]
    assert viewer.s.project.pending_dir is None
    assert not (tmp_path / "does-not-exist").exists()


def test_write_initial_manifest_archives_stale_scene(tmp_path: Path) -> None:
    root, images = _dataset(tmp_path)
    target = root / "slang_splat"
    target.mkdir()
    save_gaussian_ply(target / project.PROJECT_SCENE_NAME, _scene())
    viewer = _viewer(colmap_root=root, images_root=images, trainer=_Trainer())

    project.write_initial_manifest(viewer, target)
    project.join_pending_writes(viewer)

    assert not (target / project.PROJECT_SCENE_NAME).exists()
    assert (target / "scene.ply.bak").is_file()
    manifest = json.loads((target / project.PROJECT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["result"]["step"] == 0


def test_collision_choice_cancel_does_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    viewer = _viewer()
    viewer.s.project.pending_import_dir = tmp_path / "slang_splat"
    import src.viewer.session as session_mod

    monkeypatch.setattr(session_mod, "import_colmap_from_ui", lambda v: pytest.fail("import must not start"))
    project.collision_choice(viewer, "cancel")
    assert viewer.s.project.pending_import_dir is None
    assert viewer.s.project.pending_dir is None


# --- autosave ---------------------------------------------------------------


def test_maybe_autosave_fires_on_interval_crossings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    trainer = _Trainer(step=0)
    viewer = _viewer(trainer=trainer)
    viewer.s.project.dir = tmp_path / "slang_splat"
    viewer.ui._values["autosave_interval_steps"] = 100

    saves: list[int] = []

    def _fake_save(v, project_dir=None, *, autosave=False):
        assert autosave is True
        saves.append(int(trainer.state.step))
        v.s.project.saved_step = int(trainer.state.step)
        return v.s.project.dir

    monkeypatch.setattr(project, "save_project", _fake_save)

    trainer.state.step = 99
    assert project.maybe_autosave(viewer) is False
    trainer.state.step = 100
    assert project.maybe_autosave(viewer) is True
    trainer.state.step = 150
    assert project.maybe_autosave(viewer) is False
    trainer.state.step = 205
    assert project.maybe_autosave(viewer) is True
    assert saves == [100, 205]


def test_maybe_autosave_disabled_and_projectless(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    trainer = _Trainer(step=50000)
    viewer = _viewer(trainer=trainer)
    monkeypatch.setattr(project, "save_project", lambda *a, **k: pytest.fail("must not save"))

    # No active project -> never fires.
    assert project.maybe_autosave(viewer) is False
    # Interval 0 -> manual saves only.
    viewer.s.project.dir = tmp_path
    viewer.ui._values["autosave_interval_steps"] = 0
    assert project.maybe_autosave(viewer) is False


def test_maybe_autosave_defers_while_write_in_flight(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    trainer = _Trainer(step=100)
    viewer = _viewer(trainer=trainer)
    viewer.s.project.dir = tmp_path
    viewer.ui._values["autosave_interval_steps"] = 100
    viewer.s.project.write_thread = SimpleNamespace(is_alive=lambda: True)

    saves: list[int] = []
    monkeypatch.setattr(project, "save_project", lambda v, project_dir=None, *, autosave=False: saves.append(trainer.state.step))

    assert project.maybe_autosave(viewer) is False
    assert viewer.s.project.autosave_pending is True
    # Deferred, not dropped: fires at the next safe point once the writer is free
    # even though no new crossing happened.
    viewer.s.project.write_thread = None
    trainer.state.step = 130
    assert project.maybe_autosave(viewer) is True
    assert saves == [130]


def test_maybe_autosave_reanchors_after_schedule_restart(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    trainer = _Trainer(step=10)
    viewer = _viewer(trainer=trainer)
    viewer.s.project.dir = tmp_path
    viewer.s.project.saved_step = 30000
    viewer.ui._values["autosave_interval_steps"] = 100
    monkeypatch.setattr(project, "save_project", lambda *a, **k: pytest.fail("must not save yet"))

    assert project.maybe_autosave(viewer) is False
    assert viewer.s.project.saved_step == 10


def test_autosave_writes_single_slot(tmp_path: Path) -> None:
    root, images = _dataset(tmp_path)
    trainer = _Trainer(step=15000)
    viewer = _viewer(colmap_root=root, images_root=images, scene=_scene(), trainer=trainer)
    viewer.s.project.dir = root / "slang_splat"
    viewer.ui._values["autosave_interval_steps"] = 15000

    assert project.maybe_autosave(viewer) is True
    thread = viewer.s.project.write_thread
    assert thread is not None
    thread.join()

    target = root / "slang_splat"
    manifest = json.loads((target / project.PROJECT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["result"]["step"] == 15000
    assert (target / project.PROJECT_SCENE_NAME).is_file()
    # Single latest slot only: no checkpoint history directory.
    assert not (target / "checkpoints").exists()
    assert viewer.s.project.saved_step == 15000


def test_autosave_on_stop_saves_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _Trainer(step=22000)
    viewer = _viewer(trainer=trainer)
    viewer.s.project.dir = tmp_path / "slang_splat"
    viewer.s.project.saved_step = 15000
    viewer.ui._values["autosave_interval_steps"] = 15000

    saves: list[int] = []

    def _fake_save(v, project_dir=None, *, autosave=False):
        saves.append(int(trainer.state.step))
        v.s.project.saved_step = int(trainer.state.step)
        return v.s.project.dir

    monkeypatch.setattr(project, "save_project", _fake_save)
    assert project.autosave_on_stop(viewer) is True
    assert saves == [22000]
    # Nothing new -> no redundant save.
    assert project.autosave_on_stop(viewer) is False


# --- background writer ------------------------------------------------------


def test_save_project_does_not_block_on_the_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    root, images = _dataset(tmp_path)
    viewer = _viewer(colmap_root=root, images_root=images, scene=_scene(), trainer=_Trainer(step=7))
    release = threading.Event()
    real_write = project._write_scene_and_manifest

    def _blocked_write(project_dir, scene, include_sh, manifest):
        assert release.wait(timeout=30.0)
        real_write(project_dir, scene, include_sh, manifest)

    monkeypatch.setattr(project, "_write_scene_and_manifest", _blocked_write)
    completions: list[Exception | None] = []

    target = project.save_project(viewer, tmp_path / "async.splatproj", on_complete=completions.append)

    # The caller returned while the write is still blocked: nothing on disk yet.
    assert viewer.s.project.write_thread.is_alive()
    assert not (target / project.PROJECT_MANIFEST_NAME).exists()
    assert completions == []
    release.set()
    project.join_pending_writes(viewer)
    assert (target / project.PROJECT_MANIFEST_NAME).is_file()
    assert (target / project.PROJECT_SCENE_NAME).is_file()
    assert completions == [None]


def test_saves_chain_in_submission_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    root, images = _dataset(tmp_path)
    trainer = _Trainer(step=100)
    viewer = _viewer(colmap_root=root, images_root=images, scene=_scene(), trainer=trainer)
    release = threading.Event()
    real_write = project._write_scene_and_manifest
    write_steps: list[int] = []

    def _blocked_write(project_dir, scene, include_sh, manifest):
        assert release.wait(timeout=30.0)
        write_steps.append(int(manifest["result"]["step"]))
        real_write(project_dir, scene, include_sh, manifest)

    monkeypatch.setattr(project, "_write_scene_and_manifest", _blocked_write)
    target = project.save_project(viewer, tmp_path / "chain.splatproj")
    trainer.state.step = 200
    project.save_project(viewer)
    release.set()
    project.join_pending_writes(viewer)

    assert write_steps == [100, 200]
    manifest = json.loads((target / project.PROJECT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["result"]["step"] == 200


def test_save_failure_reports_error_and_keeps_previous_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, images = _dataset(tmp_path)
    viewer = _viewer(colmap_root=root, images_root=images, scene=_scene(), trainer=_Trainer(step=5))
    target = project.save_project(viewer, tmp_path / "err.splatproj")
    project.join_pending_writes(viewer)

    monkeypatch.setattr(project, "_write_scene_and_manifest", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    completions: list[Exception | None] = []
    project.save_project(viewer, on_complete=completions.append)
    project.join_pending_writes(viewer)

    assert len(completions) == 1 and isinstance(completions[0], OSError)
    assert "disk full" in viewer.s.last_error
    manifest = json.loads((target / project.PROJECT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["result"]["step"] == 5  # previous slot intact


def test_fingerprint_walk_runs_on_writer_and_is_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, images = _dataset(tmp_path)
    viewer = _viewer(colmap_root=root, images_root=images, scene=_scene(), trainer=_Trainer())
    calls: list[Path] = []
    real_fingerprint = project.dataset_fingerprint
    monkeypatch.setattr(project, "dataset_fingerprint", lambda p: calls.append(Path(p)) or real_fingerprint(p))

    target = project.save_project(viewer, root / "slang_splat")
    project.join_pending_writes(viewer)
    assert calls == [images.resolve()]
    assert viewer.s.project.fingerprint_cache[0] == str(images.resolve())

    project.save_project(viewer)
    project.join_pending_writes(viewer)
    assert calls == [images.resolve()]  # cache hit: no second walk
    manifest = json.loads((target / project.PROJECT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["dataset"]["fingerprint"]["files"] == 1


# --- open / relink ----------------------------------------------------------


def test_resolve_dataset_roots_prefers_abs_then_rel(tmp_path: Path) -> None:
    root, images = _dataset(tmp_path)
    project_dir = root / "slang_splat"
    project_dir.mkdir()
    manifest = {
        "dataset": {
            "colmap_root_abs": (tmp_path / "moved-away").as_posix(),
            "colmap_root_rel": "..",
            "images_root_abs": (tmp_path / "moved-away" / "images_4").as_posix(),
            "images_root_rel": "../images_4",
        }
    }
    roots = project.resolve_dataset_roots(manifest, project_dir)
    assert roots == (root.resolve(), images.resolve())


def test_resolve_dataset_roots_missing_everything(tmp_path: Path) -> None:
    manifest = {"dataset": {"colmap_root_abs": (tmp_path / "gone").as_posix(), "colmap_root_rel": "nowhere"}}
    assert project.resolve_dataset_roots(manifest, tmp_path / "proj") is None


def test_open_project_sets_relink_prompt_when_dataset_missing(tmp_path: Path) -> None:
    root, images = _dataset(tmp_path)
    viewer = _viewer(colmap_root=root, images_root=images, scene=_scene(), trainer=_Trainer())
    viewer.ui._values["colmap_root_path"] = str(root)
    viewer.ui._values["colmap_images_root"] = str(images)
    target = project.save_project(viewer, tmp_path / "detached.splatproj")
    project.join_pending_writes(viewer)

    manifest_path = target / project.PROJECT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"]["colmap_root_abs"] = (tmp_path / "gone").as_posix()
    manifest["dataset"]["colmap_root_rel"] = "gone"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    opened = _viewer()
    project.open_project(opened, target)
    assert opened.ui._values["_project_relink_open"] is True
    assert opened.s.project.pending_open is not None

    # Locate the moved dataset -> import proceeds with substituted roots.
    import src.viewer.session as session_mod

    started: list[object] = []
    import pytest as _pytest

    monkey = _pytest.MonkeyPatch()
    try:
        monkey.setattr(session_mod, "import_colmap_from_ui", lambda v: started.append(v))
        project.relink_choice(opened, "locate", located_root=root)
    finally:
        monkey.undo()
    assert started == [opened]
    assert opened.ui._values["colmap_root_path"] == str(root.resolve())
    assert opened.ui._values["colmap_images_root"] == str(images.resolve())
    assert opened.s.project.pending_open is None
    assert opened.s.project.dir == target


def test_open_colmap_project_resumes_from_saved_scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, images = _dataset(tmp_path)
    viewer = _viewer(colmap_root=root, images_root=images, scene=_scene(count=7), trainer=_Trainer(step=30000))
    viewer.ui._values["colmap_root_path"] = str(root)
    viewer.ui._values["colmap_images_root"] = str(images)
    target = project.save_project(viewer, root / "slang_splat")
    project.join_pending_writes(viewer)

    opened = _viewer()
    import src.viewer.session as session_mod

    started: list[object] = []
    monkeypatch.setattr(session_mod, "import_colmap_from_ui", lambda v: started.append(v))
    project.open_project(opened, target / project.PROJECT_MANIFEST_NAME)

    assert started == [opened]
    resume = opened.s.project.pending_resume
    assert resume is not None
    assert resume["step"] == 30000
    assert Path(resume["scene_ply"]) == target / project.PROJECT_SCENE_NAME
    # The scene parse was prefetched at open time, concurrent with the import.
    assert int(resume["loader"].result().count) == 7
    assert opened.s.project.saved_step == 30000
    assert opened.ui._values["_project_active"] is True


class _FakeLoader:
    def __init__(self, scene: GaussianScene | None = None, error: Exception | None = None, done: bool = True):
        self.path = Path("fake.ply")
        self.scene = scene
        self.error = error
        self.finished = done

    def done(self) -> bool:
        return self.finished

    def result(self) -> GaussianScene:
        if self.error is not None:
            raise self.error
        return self.scene


def test_on_import_completed_defers_resume_until_parse_finishes(tmp_path: Path) -> None:
    trainer = _Trainer(step=0)
    viewer = _viewer(trainer=trainer)
    viewer.s.project.dir = tmp_path / "slang_splat"
    loader = _FakeLoader(scene=_scene(count=3), done=False)
    viewer.s.project.pending_resume = {"scene_ply": tmp_path / "scene.ply", "step": 500, "elapsed_s": 42.0, "loader": loader}

    project.on_import_completed(viewer)
    # Parse still running: nothing applied, nothing blocked.
    assert viewer.s.project.pending_resume is not None
    assert trainer.replaced_scenes == []
    assert project.project_open_in_progress(viewer) is True

    loader.finished = True
    project.advance_project_open(viewer)
    assert trainer.state.step == 500
    assert len(trainer.replaced_scenes) == 1
    assert viewer.s.training_elapsed_s == pytest.approx(42.0)
    assert viewer.s.project.pending_resume is None
    assert project.project_open_in_progress(viewer) is False


def test_maybe_autosave_skips_while_resume_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _Trainer(step=50000)
    viewer = _viewer(trainer=trainer)
    viewer.s.project.dir = tmp_path
    viewer.ui._values["autosave_interval_steps"] = 100
    viewer.s.project.pending_resume = {"scene_ply": tmp_path / "scene.ply", "step": 1, "loader": _FakeLoader(done=False)}
    monkeypatch.setattr(project, "save_project", lambda *a, **k: pytest.fail("must not save mid-open"))
    assert project.maybe_autosave(viewer) is False


def test_open_ply_project_loads_via_pump(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    root, images = _dataset(tmp_path)
    viewer = _viewer(colmap_root=root, images_root=images, scene=_scene(count=9), trainer=_Trainer(step=3))
    viewer.ui._values["source"] = "ply"
    target = project.save_project(viewer, tmp_path / "plyproj.splatproj")
    project.join_pending_writes(viewer)

    opened = _viewer()
    import src.viewer.session as session_mod

    loads: list[tuple[Path, int]] = []
    monkeypatch.setattr(session_mod, "load_scene", lambda v, path, scene=None: loads.append((Path(path), int(scene.count))))
    project.open_project(opened, target)
    pending = opened.s.project.pending_scene_load
    assert pending is not None
    assert loads == []  # nothing loaded synchronously

    for _ in range(500):
        if pending["loader"].done():
            break
        time.sleep(0.01)
    project.advance_project_open(opened)
    assert loads == [(target / project.PROJECT_SCENE_NAME, 9)]
    assert opened.s.project.pending_scene_load is None
    assert opened.s.project.dir == target
    assert opened.ui._values["_project_active"] is True


# --- headless ---------------------------------------------------------------


def test_load_headless_project_maps_resume_keys(tmp_path: Path) -> None:
    root, images = _dataset(tmp_path)
    viewer = _viewer(colmap_root=root, images_root=images, scene=_scene(), trainer=_Trainer(step=30000))
    viewer.ui._values["colmap_root_path"] = str(root)
    viewer.ui._values["colmap_images_root"] = str(images)
    target = project.save_project(viewer, root / "slang_splat")
    project.join_pending_writes(viewer)

    config, project_dir, manifest = project.load_headless_project(target)
    run = config["viewer"]["run"]
    assert project_dir == target
    assert Path(run["resume_ply"]) == target / project.PROJECT_SCENE_NAME
    assert int(run["train_start_step"]) == 30000
    assert Path(run["colmap_root_path"]) == root.resolve()
    assert Path(run["colmap_images_root"]) == images.resolve()
    assert int(run["autosave_interval_steps"]) == 15000


def test_is_project_path(tmp_path: Path) -> None:
    assert project.is_project_path(tmp_path / "foo" / "project.json")
    assert project.is_project_path(tmp_path / "name.splatproj")
    assert not project.is_project_path(tmp_path / "run_config.json")
    plain_dir = tmp_path / "plain"
    plain_dir.mkdir()
    assert not project.is_project_path(plain_dir)
    (plain_dir / "project.json").write_text("{}", encoding="utf-8")
    assert project.is_project_path(plain_dir)


def test_project_display_name() -> None:
    assert project.project_display_name(Path("C:/data/garden/slang_splat")) == "garden"
    assert project.project_display_name(Path("C:/projects/hotel.splatproj")) == "hotel"
