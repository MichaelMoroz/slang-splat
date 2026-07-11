"""Splat project save/open/autosave (doc/SplatProjectDesign.md).

A project is a directory holding ``project.json`` (config snapshot + dataset
reference + result) next to ``scene.ply`` (the latest saved splats — also the
single autosave slot). Every COLMAP import creates a project, by default at
``<colmap_root>/slang_splat/``; autosave is step-driven and overwrites the
latest slot every ``viewer.run.autosave_interval_steps`` training steps.

This module is UI-framework-free: dialogs live in ``app.py`` callbacks and the
modal drawing lives in ``ui.py``; both talk to this module through plain
functions and ``viewer.ui._values`` keys (``_project_*``).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any

from ..app.training_controls import TRAINING_BUILD_ARG_UI_KEYS
from ..repo_defaults import deep_merge_config, json_value, load_defaults
from ..scene import GaussianScene, load_gaussian_ply, save_gaussian_ply
from .config import exported_run_values

PROJECT_FORMAT = "splatproj"
PROJECT_SCHEMA_VERSION = 1
PROJECT_MANIFEST_NAME = "project.json"
PROJECT_SCENE_NAME = "scene.ply"
PROJECT_DIR_SUFFIX = ".splatproj"
DEFAULT_PROJECT_DIRNAME = "slang_splat"
DEFAULT_AUTOSAVE_INTERVAL_STEPS = 15000

_app_version_cache: str | None = None


def default_project_dir(colmap_root: str | Path) -> Path:
    return Path(colmap_root).expanduser() / DEFAULT_PROJECT_DIRNAME


def project_manifest_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.name.lower() == PROJECT_MANIFEST_NAME else resolved / PROJECT_MANIFEST_NAME


def is_project_path(path: str | Path) -> bool:
    resolved = Path(path)
    if resolved.name.lower() == PROJECT_MANIFEST_NAME or resolved.suffix.lower() == PROJECT_DIR_SUFFIX:
        return True
    return resolved.is_dir() and (resolved / PROJECT_MANIFEST_NAME).is_file()


def ensure_writable_dir(path: str | Path) -> bool:
    target = Path(path)
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / f".write_probe_{os.getpid()}"
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _app_version() -> str:
    global _app_version_cache
    if _app_version_cache is None:
        try:
            import subprocess

            _app_version_cache = subprocess.run(
                ("git", "describe", "--always", "--dirty"),
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
        except Exception:
            _app_version_cache = ""
    return _app_version_cache


def _viewer_values(viewer: object) -> dict[str, object]:
    values = getattr(getattr(viewer, "ui", None), "_values", None)
    return values if isinstance(values, dict) else {}


def _set_ui_value(viewer: object, key: str, value: object) -> None:
    values = getattr(getattr(viewer, "ui", None), "_values", None)
    if isinstance(values, dict):
        values[key] = value


def _project_state(viewer: object) -> object | None:
    return getattr(getattr(viewer, "s", None), "project", None)


def _section(data: dict[str, Any] | None, *keys: str) -> dict[str, Any]:
    node: object = data
    for key in keys:
        if not isinstance(node, dict):
            return {}
        node = node.get(key, {})
    return node if isinstance(node, dict) else {}


def _path_or_none(value: object) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text == "" else Path(text).expanduser()


# --- config snapshot ------------------------------------------------------


def capture_project_config(values: dict[str, object], *, existing_run: dict[str, Any] | None = None) -> dict[str, Any]:
    """Project config snapshot: the defaults.json shape minus viewer.ui/state.

    Uses the same three exporters as ``Update Defaults`` so one code path
    serves defaults, run configs, and projects.
    """
    from .ui import export_repo_defaults_from_ui_values  # deferred: ui pulls imgui

    exported = export_repo_defaults_from_ui_values(values)
    viewer_export = exported.get("viewer", {})
    training = {
        build_arg: values[control_key]
        for build_arg, control_key in TRAINING_BUILD_ARG_UI_KEYS.items()
        if control_key in values
    }
    return {
        "training_build_args": json_value(training),
        "renderer": exported.get("renderer", {}),
        "viewer": {
            "controls": viewer_export.get("controls", {}),
            "import": viewer_export.get("import", {}),
            "run": json_value(exported_run_values(values, existing_run=existing_run)),
        },
    }


def project_config_overlay(manifest: dict[str, Any]) -> dict[str, Any]:
    return deep_merge_config(load_defaults(), _section(manifest, "config"))


# --- dataset block --------------------------------------------------------


def dataset_fingerprint(images_root: str | Path) -> dict[str, Any] | None:
    """Cheap dataset identity: metadata scan only, no content reads."""
    root = Path(images_root)
    if not root.is_dir():
        return None
    entries: list[tuple[str, int]] = []
    total_bytes = 0
    newest_mtime = 0.0
    for dirpath, _dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        for filename in filenames:
            try:
                stat = (base / filename).stat()
            except OSError:
                continue
            entries.append(((base / filename).relative_to(root).as_posix(), int(stat.st_size)))
            total_bytes += int(stat.st_size)
            newest_mtime = max(newest_mtime, float(stat.st_mtime))
    entries.sort()
    digest = hashlib.sha1()
    for name, size in entries:
        digest.update(f"{name}:{size}\n".encode("utf-8"))
    return {
        "files": len(entries),
        "bytes": total_bytes,
        "newest_mtime": int(newest_mtime),
        "sample": digest.hexdigest(),
    }


def _relative_or_none(target: Path, base: Path) -> str | None:
    try:
        return Path(os.path.relpath(target, base)).as_posix()
    except ValueError:  # different drives on Windows
        return None


def _dataset_block(viewer: object, project_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    """Dataset block + the images root still needing a fingerprint (walked on the
    writer thread — the first-save directory scan must not stall the caller)."""
    colmap_root = getattr(getattr(viewer, "s", None), "colmap_root", None)
    if colmap_root is None:
        return None, None
    colmap_root = Path(colmap_root).resolve()
    images_root = getattr(getattr(getattr(viewer, "s", None), "colmap_import", None), "images_root", None)
    images_root = colmap_root if images_root is None else Path(images_root).resolve()
    base = Path(project_dir).resolve()
    fingerprint = None
    cache = getattr(_project_state(viewer), "fingerprint_cache", None)
    if isinstance(cache, tuple) and len(cache) == 2 and cache[0] == str(images_root):
        fingerprint = cache[1]
    block = {
        "mode": "reference",
        "colmap_root_abs": colmap_root.as_posix(),
        "colmap_root_rel": _relative_or_none(colmap_root, base),
        "images_root_abs": images_root.as_posix(),
        "images_root_rel": _relative_or_none(images_root, base),
        "fingerprint": fingerprint,
    }
    return block, None if fingerprint is not None else images_root


def resolve_dataset_roots(manifest: dict[str, Any], project_dir: Path) -> tuple[Path, Path] | None:
    """abs -> project-relative -> None (caller prompts). Returns (colmap_root, images_root)."""
    dataset = _section(manifest, "dataset")
    base = Path(project_dir)
    colmap_root = None
    for candidate in (_path_or_none(dataset.get("colmap_root_abs")), _joined_rel(base, dataset.get("colmap_root_rel"))):
        if candidate is not None and candidate.is_dir():
            colmap_root = candidate
            break
    if colmap_root is None:
        return None
    stored_abs = _path_or_none(dataset.get("colmap_root_abs"))
    images_candidates = [
        _path_or_none(dataset.get("images_root_abs")),
        _joined_rel(base, dataset.get("images_root_rel")),
    ]
    if stored_abs is not None:
        images_candidates.append(_substitute_prefix(_path_or_none(dataset.get("images_root_abs")), stored_abs, colmap_root))
    for candidate in images_candidates:
        if candidate is not None and candidate.is_dir():
            return colmap_root.resolve(), candidate.resolve()
    return None


def _joined_rel(base: Path, rel: object) -> Path | None:
    if rel is None or str(rel).strip() == "":
        return None
    return (base / str(rel)).resolve()


def _substitute_prefix(path: Path | None, old_prefix: Path | None, new_prefix: Path) -> Path | None:
    if path is None or old_prefix is None:
        return None
    try:
        return new_prefix / path.relative_to(old_prefix)
    except ValueError:
        return None


# --- manifest io ----------------------------------------------------------


def migrate_project_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("format") != PROJECT_FORMAT:
        raise ValueError("Not a splat project manifest (missing format: splatproj).")
    version = int(manifest.get("schema_version", 1))
    if version > PROJECT_SCHEMA_VERSION:
        raise ValueError(f"Project schema v{version} is newer than this app supports (v{PROJECT_SCHEMA_VERSION}).")
    # v1: no migrations yet; chain vN -> vN+1 rewrites here as the schema grows.
    return manifest


def load_project_manifest(path: str | Path) -> tuple[dict[str, Any], Path]:
    manifest_path = project_manifest_path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Project manifest root must be a JSON object: {manifest_path}")
    return migrate_project_manifest(manifest), manifest_path.parent


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False)
        handle.write("\n")
    os.replace(tmp, path)


def _existing_manifest_run(project_dir: Path) -> dict[str, Any] | None:
    manifest_path = project_dir / PROJECT_MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        run = _section(manifest, "config", "viewer", "run")
        return run or None
    except Exception:
        return None


def _existing_manifest_created(project_dir: Path) -> str:
    manifest_path = project_dir / PROJECT_MANIFEST_NAME
    if not manifest_path.is_file():
        return ""
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            return str(json.load(handle).get("created", ""))
    except Exception:
        return ""


def _build_manifest(
    *,
    config: dict[str, Any],
    dataset: dict[str, Any] | None,
    result: dict[str, Any] | None,
    created: str,
) -> dict[str, Any]:
    now = _now_iso()
    return {
        "format": PROJECT_FORMAT,
        "schema_version": PROJECT_SCHEMA_VERSION,
        "app_version": _app_version(),
        "created": created or now,
        "modified": now,
        "config": config,
        "dataset": dataset,
        "scene": {"objects": [{"id": 0, "name": "scene", "ply": PROJECT_SCENE_NAME, "training": None}]},
        "result": result,
    }


# --- result / elapsed -----------------------------------------------------


def _training_elapsed_seconds(viewer: object) -> float:
    state = getattr(viewer, "s", None)
    elapsed = float(getattr(state, "training_elapsed_s", 0.0))
    resume_time = getattr(state, "training_resume_time", None)
    if bool(getattr(state, "training_active", False)) and resume_time is not None:
        import time

        elapsed += max(time.perf_counter() - float(resume_time), 0.0)
    return elapsed


def _resolve_sh_band(trainer: object, step: int) -> int | None:
    try:
        from ..training import resolve_sh_band

        return int(resolve_sh_band(trainer.training, step))
    except Exception:
        return None


def _result_block(viewer: object, splat_count: int | None) -> dict[str, Any]:
    trainer = getattr(getattr(viewer, "s", None), "trainer", None)
    step = int(getattr(getattr(trainer, "state", None), "step", 0)) if trainer is not None else 0
    if splat_count is None:
        splat_count = int(getattr(getattr(trainer, "scene", None), "count", 0)) if trainer is not None else 0
    return {
        "step": step,
        "train_elapsed_s": round(_training_elapsed_seconds(viewer), 3),
        "splat_count": int(splat_count),
        "sh_band": None if trainer is None else _resolve_sh_band(trainer, step),
    }


# --- save -----------------------------------------------------------------


def _write_busy(ps: object) -> bool:
    thread = getattr(ps, "write_thread", None)
    return thread is not None and thread.is_alive()


def _join_write(ps: object) -> None:
    thread = getattr(ps, "write_thread", None)
    if thread is not None and thread.is_alive():
        thread.join()
    if ps is not None:
        ps.write_thread = None


def _write_scene_and_manifest(project_dir: Path, scene: GaussianScene | None, include_sh: bool, manifest: dict[str, Any]) -> None:
    if scene is not None:
        tmp = project_dir / (PROJECT_SCENE_NAME + ".tmp")
        save_gaussian_ply(tmp, scene, include_sh=include_sh)
        os.replace(tmp, project_dir / PROJECT_SCENE_NAME)
    # Manifest last: a crash mid-write leaves the previous manifest pointing at a
    # scene that is at least as new as result.step claims — never the reverse.
    _write_json_atomic(project_dir / PROJECT_MANIFEST_NAME, manifest)


def _capture_scene_for_save(viewer: object):
    """GPU readback on the caller thread; returns (finish, include_sh, count).

    ``finish`` is a CPU-only zero-arg callable producing the GaussianScene on the
    writer thread (None when nothing is loaded — manifest-only save). Mirrors the
    source priority of ``ViewerCore._export_source_scene``.
    """
    state = getattr(viewer, "s", None)
    include_sh_fn = getattr(viewer, "_export_should_include_sh", None)

    def _include_sh() -> bool:
        return bool(include_sh_fn()) if callable(include_sh_fn) else True

    trainer = getattr(state, "trainer", None)
    if trainer is not None and callable(getattr(trainer, "capture_live_scene", None)):
        return trainer.capture_live_scene(), _include_sh(), int(getattr(getattr(trainer, "scene", None), "count", 0))
    renderer = getattr(state, "renderer", None)
    if renderer is not None and int(getattr(renderer, "_scene_count", 0)) > 0 and callable(getattr(renderer, "capture_live_scene", None)):
        return renderer.capture_live_scene(), _include_sh(), int(renderer._scene_count)
    export_scene = getattr(viewer, "_export_source_scene", None)
    if callable(export_scene):
        try:
            scene = export_scene()
        except Exception:
            return None, True, None  # nothing loaded yet: manifest-only save
        return (lambda: scene), _include_sh(), int(scene.count)
    return None, True, None


def _run_deferred_save(
    *,
    previous_thread: object | None,
    project_dir: Path,
    finish_scene,
    include_sh: bool,
    manifest: dict[str, Any],
    fingerprint_root: Path | None,
    ps: object | None,
    state: object | None,
    on_complete,
) -> None:
    error: Exception | None = None
    try:
        # Chain instead of making the caller wait: writes stay ordered and the
        # render thread never blocks on a busy writer.
        if previous_thread is not None and previous_thread.is_alive():
            previous_thread.join()
        scene = None if finish_scene is None else finish_scene()
        if fingerprint_root is not None and isinstance(manifest.get("dataset"), dict):
            fingerprint = dataset_fingerprint(fingerprint_root)
            manifest["dataset"]["fingerprint"] = fingerprint
            if ps is not None:
                ps.fingerprint_cache = (str(fingerprint_root), fingerprint)
        _write_scene_and_manifest(project_dir, scene, include_sh, manifest)
    except Exception as exc:  # noqa: BLE001 - a failed save must not kill the writer chain
        error = exc
        message = f"Project save failed: {exc}"
        print(message, flush=True)
        if state is not None:
            state.last_error = message
    if callable(on_complete):
        try:
            on_complete(error)
        except Exception:
            pass


def save_project(viewer: object, project_dir: str | Path | None = None, *, autosave: bool = False, on_complete=None) -> Path:
    """Write the latest slot (project.json + scene.ply) to the project dir.

    Only the GPU readback and the config/dataset snapshot run on the caller
    thread; scene conversion, PLY serialization, the first-save fingerprint
    walk, and all file writes (tmp + atomic replace) run on a background writer
    thread. Concurrent saves chain in submission order. ``on_complete(error)``
    fires on the writer thread once the files are on disk.
    """
    ps = _project_state(viewer)
    target = Path(project_dir).expanduser() if project_dir is not None else getattr(ps, "dir", None)
    if target is None:
        raise RuntimeError("No project directory is set. Use Save Project As…")
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)

    values = _viewer_values(viewer)
    config = capture_project_config(values, existing_run=_existing_manifest_run(target))
    dataset, fingerprint_root = _dataset_block(viewer, target)
    finish_scene, include_sh, splat_count = _capture_scene_for_save(viewer)

    created = getattr(ps, "created", "") or _existing_manifest_created(target)
    result = _result_block(viewer, splat_count)
    manifest = _build_manifest(config=config, dataset=dataset, result=result, created=created)

    _spawn_writer(
        viewer,
        project_dir=target,
        finish_scene=finish_scene,
        include_sh=include_sh,
        manifest=manifest,
        fingerprint_root=fingerprint_root,
        on_complete=on_complete,
    )
    if ps is not None:
        ps.dir = target
        ps.created = manifest["created"]
        ps.saved_step = int(result["step"])
    _sync_project_ui(viewer)
    return target


def _spawn_writer(
    viewer: object,
    *,
    project_dir: Path,
    finish_scene,
    include_sh: bool,
    manifest: dict[str, Any],
    fingerprint_root: Path | None,
    on_complete=None,
) -> None:
    ps = _project_state(viewer)
    thread = threading.Thread(
        target=_run_deferred_save,
        kwargs=dict(
            previous_thread=getattr(ps, "write_thread", None),
            project_dir=project_dir,
            finish_scene=finish_scene,
            include_sh=include_sh,
            manifest=manifest,
            fingerprint_root=fingerprint_root,
            ps=ps,
            state=getattr(viewer, "s", None),
            on_complete=on_complete,
        ),
        name="splatproj-save",
        daemon=False,
    )
    if ps is not None:
        ps.write_thread = thread
    thread.start()


def write_initial_manifest(viewer: object, project_dir: str | Path) -> Path:
    """Mandatory-creation write at import completion: manifest only, no scene.

    ``scene.ply`` first appears at the first save/autosave so import completion
    stays cheap; opening a never-saved project just re-imports and re-inits.
    """
    ps = _project_state(viewer)
    target = Path(project_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    stale_scene = target / PROJECT_SCENE_NAME
    if stale_scene.is_file():
        # The project restarts at step 0; a leftover scene.ply would otherwise be
        # resumed against the fresh manifest. Archive instead of delete.
        os.replace(stale_scene, stale_scene.with_suffix(".ply.bak"))
    config = capture_project_config(_viewer_values(viewer))
    dataset, fingerprint_root = _dataset_block(viewer, target)
    manifest = _build_manifest(
        config=config,
        dataset=dataset,
        result=_result_block(viewer, None),
        created="",
    )
    # Same deferred writer as saves: the fingerprint walk of a large images root
    # must not stall the import-completion frame.
    _spawn_writer(
        viewer,
        project_dir=target,
        finish_scene=None,
        include_sh=True,
        manifest=manifest,
        fingerprint_root=fingerprint_root,
    )
    if ps is not None:
        ps.dir = target
        ps.created = manifest["created"]
        ps.saved_step = int(manifest["result"]["step"])
    _sync_project_ui(viewer)
    print(f"Project created: {target}")
    return target


# --- autosave -------------------------------------------------------------


def _autosave_interval(viewer: object) -> int:
    values = _viewer_values(viewer)
    try:
        return int(values.get("autosave_interval_steps", DEFAULT_AUTOSAVE_INTERVAL_STEPS))
    except (TypeError, ValueError):
        return DEFAULT_AUTOSAVE_INTERVAL_STEPS


def maybe_autosave(viewer: object) -> bool:
    """Step-driven autosave; called at the between-batches safe point.

    Fires when trainer.state.step crosses a multiple of the interval. A save
    deferred by an in-flight write is retried at the next safe point, not
    dropped — the next scheduled crossing may be a full interval away.
    """
    ps = _project_state(viewer)
    if ps is None or getattr(ps, "dir", None) is None:
        return False
    if getattr(ps, "pending_resume", None) is not None:
        # A project open is still applying; saving now would capture the
        # pre-resume init scene against the resumed manifest.
        return False
    trainer = getattr(getattr(viewer, "s", None), "trainer", None)
    if trainer is None:
        return False
    interval = _autosave_interval(viewer)
    if interval <= 0:
        return False
    step = int(getattr(getattr(trainer, "state", None), "step", 0))
    saved_step = int(getattr(ps, "saved_step", 0))
    if step < saved_step:
        # The schedule restarted below the last save (Reload / Reinitialize
        # Gaussians): re-anchor so autosave fires at the next crossing from here.
        ps.saved_step = saved_step = step
    due = bool(getattr(ps, "autosave_pending", False)) or (step > saved_step and step // interval > saved_step // interval)
    if not due:
        return False
    if _write_busy(ps):
        ps.autosave_pending = True
        return False
    ps.autosave_pending = False
    try:
        save_project(viewer, autosave=True)
    except Exception as exc:
        message = f"Autosave failed: {exc}"
        print(message, flush=True)
        state = getattr(viewer, "s", None)
        if state is not None:
            state.last_error = message
        return False
    return True


def join_pending_writes(viewer: object) -> None:
    """Block until an in-flight background autosave write has finished."""
    ps = _project_state(viewer)
    if ps is not None:
        _join_write(ps)


def autosave_on_stop(viewer: object) -> bool:
    """Stop/pause boundary save so a stopped run never rolls back on reopen."""
    ps = _project_state(viewer)
    if ps is None or getattr(ps, "dir", None) is None:
        return False
    trainer = getattr(getattr(viewer, "s", None), "trainer", None)
    if trainer is None or _autosave_interval(viewer) <= 0:
        return False
    step = int(getattr(getattr(trainer, "state", None), "step", 0))
    if step <= int(getattr(ps, "saved_step", 0)):
        return False
    ps.autosave_pending = False
    try:
        # Background like every save: the writer chains behind any in-flight
        # autosave, so stopping training never blocks the UI on file IO.
        save_project(viewer, autosave=True)
    except Exception as exc:
        message = f"Autosave on stop failed: {exc}"
        print(message, flush=True)
        state = getattr(viewer, "s", None)
        if state is not None:
            state.last_error = message
        return False
    return True


# --- background scene loading ----------------------------------------------


class _SceneLoader:
    """Parses a scene PLY on a background thread (read-only: safe as daemon).

    Started at project-open time so the parse overlaps the dataset import;
    the per-frame pump (`advance_project_open`) applies the result without
    ever blocking the render thread on the parse.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.scene: GaussianScene | None = None
        self.error: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="splatproj-scene-load", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self.scene = load_gaussian_ply(self.path)
        except Exception as exc:  # noqa: BLE001 - surfaced via .error at apply time
            self.error = exc

    def done(self) -> bool:
        return not self._thread.is_alive()

    def result(self) -> GaussianScene:
        self._thread.join()
        if self.error is not None:
            raise self.error
        return self.scene


def _try_apply_pending_resume(viewer: object) -> bool:
    """Apply a pending project resume once the trainer and the parsed scene exist."""
    state = getattr(viewer, "s", None)
    ps = _project_state(viewer)
    resume = None if ps is None else getattr(ps, "pending_resume", None)
    if resume is None:
        return False
    trainer = getattr(state, "trainer", None)
    if trainer is None:
        return False
    loader = resume.get("loader")
    if loader is not None and not loader.done():
        return False
    ps.pending_resume = None
    try:
        scene = loader.result() if loader is not None else load_gaussian_ply(resume["scene_ply"])
        trainer.state.step = int(resume.get("step", 0))
        trainer.replace_scene(scene)
        trainer.update_hyperparams(trainer.adam, trainer.stability, trainer.training)
        state.training_elapsed_s = float(resume.get("elapsed_s", 0.0))
        print(f"Project resume: {scene.count:,} splats at step {int(trainer.state.step)}", flush=True)
    except Exception as exc:
        message = f"Project resume failed: {exc}"
        print(message, flush=True)
        state.last_error = message
    return True


def advance_project_open(viewer: object) -> None:
    """Per-frame pump for asynchronous project opens (GUI render loop).

    Applies a finished background scene parse: either the scene-only open
    (pending_scene_load -> load_scene + activate) or the trainer resume
    (pending_resume -> replace_scene).
    """
    ps = _project_state(viewer)
    if ps is None:
        return
    pending_load = getattr(ps, "pending_scene_load", None)
    if pending_load is not None and pending_load["loader"].done():
        from . import session  # deferred

        ps.pending_scene_load = None
        loader = pending_load["loader"]
        state = getattr(viewer, "s", None)
        try:
            session.load_scene(viewer, loader.path, scene=loader.result())
            _activate_project(viewer, pending_load["project_dir"], pending_load["manifest"])
        except Exception as exc:
            message = f"Project open failed: {exc}"
            print(message, flush=True)
            if state is not None:
                state.last_error = message
    _try_apply_pending_resume(viewer)


def project_open_in_progress(viewer: object) -> bool:
    ps = _project_state(viewer)
    if ps is None:
        return False
    return getattr(ps, "pending_scene_load", None) is not None or getattr(ps, "pending_resume", None) is not None


# --- import hooks (mandatory creation + resume) ----------------------------


def request_colmap_import(viewer: object) -> str:
    """GUI Load-COLMAP entry: resolve the project location before importing.

    Returns "started" (import triggered), "prompt" (collision modal opened),
    or "unwritable" (caller must supply a location via dialog).
    """
    ps = _project_state(viewer)
    root_text = str(_viewer_values(viewer).get("colmap_root_path", "") or "").strip()
    root = Path(root_text).expanduser() if root_text else None
    # A missing root fails import validation with its own clear error; don't
    # create a stray slang_splat/ tree under a mistyped path first.
    target = default_project_dir(root) if root is not None and root.is_dir() else None
    if target is not None and ps is not None:
        active = getattr(ps, "dir", None)
        same_project = active is not None and Path(active).resolve() == target.resolve()
        if (target / PROJECT_MANIFEST_NAME).is_file() and not same_project:
            ps.pending_import_dir = target
            _set_ui_value(viewer, "_project_collision_open", True)
            _set_ui_value(viewer, "_project_collision_message", f"A project already exists at\n{target}")
            return "prompt"
        if not ensure_writable_dir(target):
            return "unwritable"
    return start_colmap_import(viewer, target)


def start_colmap_import(viewer: object, project_dir: str | Path | None) -> str:
    from . import session  # deferred: session imports this module

    ps = _project_state(viewer)
    if ps is not None:
        ps.pending_dir = None if project_dir is None else Path(project_dir).expanduser()
        ps.pending_resume = None
    session.import_colmap_from_ui(viewer)
    return "started"


def collision_choice(viewer: object, choice: str, *, chosen_dir: str | Path | None = None) -> None:
    _set_ui_value(viewer, "_project_collision_open", False)
    ps = _project_state(viewer)
    target = None if ps is None else getattr(ps, "pending_import_dir", None)
    if ps is not None:
        ps.pending_import_dir = None
    if choice == "cancel" or target is None:
        return
    if choice == "open":
        open_project(viewer, Path(target) / PROJECT_MANIFEST_NAME)
    elif choice == "fresh":
        start_colmap_import(viewer, target)
    elif choice == "choose" and chosen_dir is not None:
        start_colmap_import(viewer, chosen_dir)


def on_import_completed(viewer: object) -> None:
    """Import-completion hook (trainer just constructed): resume, then create.

    Shared by GUI and headless because both funnel through
    ``session._finish_import_colmap_dataset``.
    """
    state = getattr(viewer, "s", None)
    ps = _project_state(viewer)
    if ps is None:
        return
    trainer = getattr(state, "trainer", None)
    pending_elapsed = getattr(ps, "pending_elapsed_s", None)
    ps.pending_elapsed_s = None
    if pending_elapsed is not None and getattr(ps, "pending_resume", None) is None and trainer is not None:
        # Headless reopen resumes via run keys; restore the elapsed counter the
        # import reset (the GUI open path restores it through pending_resume).
        state.training_elapsed_s = float(pending_elapsed)
    # Applies inline when the open-time scene prefetch already finished (the
    # common case — the parse overlapped the import); otherwise the pending
    # resume stays and the per-frame pump applies it without blocking.
    _try_apply_pending_resume(viewer)
    target = getattr(ps, "pending_dir", None)
    ps.pending_dir = None
    if target is None and getattr(ps, "dir", None) is None:
        colmap_root = getattr(state, "colmap_root", None)
        if colmap_root is not None:
            target = default_project_dir(colmap_root)
    if target is not None:
        try:
            write_initial_manifest(viewer, target)
        except Exception as exc:
            message = f"Project creation failed at {target}: {exc}"
            print(message, flush=True)
            state.last_error = message
    _sync_project_ui(viewer)


# --- open -----------------------------------------------------------------


def open_project(viewer: object, path: str | Path) -> None:
    ps = _project_state(viewer)
    if ps is not None:
        # A new open supersedes any in-flight one; orphaned loader threads are
        # read-only daemons and finish harmlessly.
        ps.pending_resume = None
        ps.pending_scene_load = None
    manifest, project_dir = load_project_manifest(path)
    overlay = project_config_overlay(manifest)
    run = _section(overlay, "viewer", "run")
    source = str(run.get("source", "colmap") or "colmap").strip().lower()
    if source != "colmap":
        _open_ply_project(viewer, manifest, project_dir, overlay)
        return
    roots = resolve_dataset_roots(manifest, project_dir)
    if roots is None:
        ps = _project_state(viewer)
        if ps is not None:
            ps.pending_open = (manifest, project_dir)
        stored = _section(manifest, "dataset").get("colmap_root_abs", "")
        _set_ui_value(viewer, "_project_relink_open", True)
        _set_ui_value(viewer, "_project_relink_message", f"Dataset not found at\n{stored}")
        return
    _open_colmap_project(viewer, manifest, project_dir, roots)


def relink_choice(viewer: object, choice: str, *, located_root: str | Path | None = None) -> None:
    _set_ui_value(viewer, "_project_relink_open", False)
    ps = _project_state(viewer)
    pending = None if ps is None else getattr(ps, "pending_open", None)
    if pending is None:
        return
    manifest, project_dir = pending
    if choice == "cancel":
        ps.pending_open = None
        return
    if choice == "open_anyway":
        ps.pending_open = None
        _open_ply_project(viewer, manifest, project_dir, project_config_overlay(manifest))
        return
    if choice == "locate" and located_root is not None:
        located = Path(located_root).expanduser()
        substituted = dict(manifest)
        dataset = dict(_section(manifest, "dataset"))
        old_root = _path_or_none(dataset.get("colmap_root_abs"))
        dataset["colmap_root_abs"] = located.as_posix()
        old_images = _path_or_none(dataset.get("images_root_abs"))
        relocated_images = _substitute_prefix(old_images, old_root, located)
        if relocated_images is not None:
            dataset["images_root_abs"] = relocated_images.as_posix()
        substituted["dataset"] = dataset
        roots = resolve_dataset_roots(substituted, project_dir)
        if roots is None:
            _set_ui_value(viewer, "_project_relink_open", True)
            _set_ui_value(viewer, "_project_relink_message", f"Dataset not found at\n{located}")
            return
        ps.pending_open = None
        _open_colmap_project(viewer, substituted, project_dir, roots)
        return
    # Unknown choice or missing folder: keep the pending open alive.
    _set_ui_value(viewer, "_project_relink_open", True)


def _activate_project(viewer: object, project_dir: Path, manifest: dict[str, Any]) -> None:
    ps = _project_state(viewer)
    result = manifest.get("result") or {}
    if ps is not None:
        ps.dir = Path(project_dir)
        ps.created = str(manifest.get("created", ""))
        ps.saved_step = int(result.get("step", 0) or 0)
        ps.pending_import_dir = None
        ps.fingerprint_cache = None
    _sync_project_ui(viewer)


def _apply_overlay(viewer: object, overlay: dict[str, Any]) -> None:
    from .config import apply_config_overlay

    values = getattr(getattr(viewer, "ui", None), "_values", None)
    if isinstance(values, dict):
        apply_config_overlay(values, overlay)


def deactivate_project(viewer: object) -> None:
    ps = _project_state(viewer)
    if ps is None:
        return
    ps.dir = None
    ps.created = ""
    ps.saved_step = 0
    ps.pending_resume = None
    ps.pending_scene_load = None
    ps.fingerprint_cache = None
    _sync_project_ui(viewer)


def _open_ply_project(viewer: object, manifest: dict[str, Any], project_dir: Path, overlay: dict[str, Any]) -> None:
    _apply_overlay(viewer, overlay)
    scene_ply = Path(project_dir) / PROJECT_SCENE_NAME
    if not scene_ply.is_file():
        run = _section(overlay, "viewer", "run")
        ply_path = _path_or_none(run.get("ply_path"))
        if ply_path is None or not ply_path.is_file():
            raise FileNotFoundError(f"Project has no saved scene: {scene_ply}")
        scene_ply = ply_path
    ps = _project_state(viewer)
    if ps is None:
        return
    # Parse on a background thread; the per-frame pump loads the scene (which
    # deactivates any active project) and then activates this one.
    ps.pending_scene_load = {
        "loader": _SceneLoader(scene_ply),
        "manifest": manifest,
        "project_dir": Path(project_dir),
    }


def _open_colmap_project(viewer: object, manifest: dict[str, Any], project_dir: Path, roots: tuple[Path, Path]) -> None:
    from . import session  # deferred

    overlay = project_config_overlay(manifest)
    _apply_overlay(viewer, overlay)
    colmap_root, images_root = roots
    values = _viewer_values(viewer)
    values["colmap_root_path"] = str(colmap_root)
    values["colmap_images_root"] = str(images_root)
    stored_root = _path_or_none(_section(manifest, "dataset").get("colmap_root_abs"))
    if stored_root is not None and stored_root.resolve() != colmap_root.resolve():
        # The dataset moved: rewrite other dataset-derived paths by prefix substitution.
        for key in ("colmap_depth_root", "colmap_alpha_mask_root", "colmap_custom_ply_path", "colmap_custom_mesh_path", "colmap_database_path"):
            relocated = _substitute_prefix(_path_or_none(values.get(key)), stored_root, colmap_root)
            if relocated is not None and relocated.exists():
                values[key] = str(relocated)
    _activate_project(viewer, project_dir, manifest)
    ps = _project_state(viewer)
    scene_ply = Path(project_dir) / PROJECT_SCENE_NAME
    result = manifest.get("result") or {}
    if ps is not None and scene_ply.is_file():
        ps.pending_resume = {
            "scene_ply": scene_ply,
            "step": int(result.get("step", 0) or 0),
            "elapsed_s": float(result.get("train_elapsed_s", 0.0) or 0.0),
            # Start parsing now: the PLY parse runs concurrently with the dataset
            # import so the resume applies without a render-thread stall.
            "loader": _SceneLoader(scene_ply),
        }
    session.import_colmap_from_ui(viewer)


# --- headless -------------------------------------------------------------


def load_headless_project(config_path: str | Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Map a .splatproj manifest onto the headless config schema.

    ``result`` maps onto the existing ``resume_ply``/``train_start_step`` keys;
    explicit values in the stored run section win over the mapping.
    """
    manifest, project_dir = load_project_manifest(config_path)
    overlay = project_config_overlay(manifest)
    run = overlay.setdefault("viewer", {}).setdefault("run", {})
    roots = resolve_dataset_roots(manifest, project_dir)
    if roots is not None:
        run["colmap_root_path"] = str(roots[0])
        run["colmap_images_root"] = str(roots[1])
    result = manifest.get("result") or {}
    scene_ply = project_dir / PROJECT_SCENE_NAME
    if scene_ply.is_file():
        if not run.get("resume_ply"):
            run["resume_ply"] = str(scene_ply)
        if not int(run.get("train_start_step", 0) or 0):
            run["train_start_step"] = int(result.get("step", 0) or 0)
    return overlay, project_dir, manifest


def activate_project(viewer: object, project_dir: str | Path, manifest: dict[str, Any]) -> None:
    _activate_project(viewer, Path(project_dir), manifest)
    ps = _project_state(viewer)
    if ps is not None:
        result = manifest.get("result") or {}
        ps.pending_elapsed_s = float(result.get("train_elapsed_s", 0.0) or 0.0)


# --- ui sync --------------------------------------------------------------


def project_display_name(project_dir: str | Path) -> str:
    """Human name for a project: the dataset folder for default-location projects."""
    resolved = Path(project_dir)
    if resolved.name == DEFAULT_PROJECT_DIRNAME and resolved.parent.name:
        return resolved.parent.name
    return resolved.stem


def _sync_project_ui(viewer: object) -> None:
    ps = _project_state(viewer)
    active = ps is not None and getattr(ps, "dir", None) is not None
    name = project_display_name(ps.dir) if active else ""
    _set_ui_value(viewer, "_project_active", bool(active))
    _set_ui_value(viewer, "_project_name", name)
    _set_ui_value(viewer, "_project_dir", str(ps.dir) if active else "")
    set_title = getattr(viewer, "set_project_window_title", None)
    if callable(set_title):
        set_title(name if active else None)
