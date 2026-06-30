from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import slangpy as spy

from .. import create_default_device, device_type_from_name
from ..app.shared import save_snapshot
from ..repo_defaults import load_config
from ..renderer import GaussianRenderSettings
from ..scene import load_gaussian_ply, save_gaussian_ply
from . import frame_capture, presenter, presenter_state, session
from .app import ViewerCore, _initial_renderer_params, _precompile_runtime_shaders
from .buffer_debug import collect_resource_debug_snapshot, write_resource_debug_log
from .config import apply_config_overlay
from .state import DEFAULT_MAX_PREPASS_MEMORY_MB, LOSS_DEBUG_OPTIONS, ViewerState
from .ui import build_ui

_DEFAULT_RENDER_WIDTH = 1280
_DEFAULT_RENDER_HEIGHT = 720
_STATS_FIELDS = (
    "step",
    "splats",
    "last_loss",
    "avg_loss",
    "last_mse",
    "avg_mse",
    "last_ssim",
    "avg_ssim",
    "last_psnr",
    "avg_psnr",
)


@dataclass(frozen=True, slots=True)
class ScheduledAction:
    step: int
    action: str
    config: dict[str, Any]


class _HeadlessToolkitState:
    def __init__(self) -> None:
        self.fps_history: list[float] = []
        self.frame_time_history: list[float] = []
        self.step_history: list[int] = []
        self.loss_history: list[float] = []
        self.ssim_history: list[float] = []
        self.psnr_history: list[float] = []
        self.photometric_step_history: list[int] = []
        self.photometric_loss_history: list[float] = []

    def append_training_plot_sample(self, step: int, _time_s: float, loss: float, ssim: float, psnr: float) -> None:
        self.step_history.append(int(step))
        self.loss_history.append(float(loss))
        self.ssim_history.append(float(ssim))
        self.psnr_history.append(float(psnr))

    def append_photometric_plot_sample(self, step: int, _time_s: float, loss: float) -> None:
        self.photometric_step_history.append(int(step))
        self.photometric_loss_history.append(float(loss))

    def clear_photometric_plot_history(self) -> None:
        self.photometric_step_history.clear()
        self.photometric_loss_history.clear()


class _HeadlessToolkit:
    def __init__(self, viewer: "HeadlessViewer") -> None:
        self._viewer = viewer
        self.tk = _HeadlessToolkitState()
        self.callbacks = SimpleNamespace()

    def close_colmap_import_window(self) -> None:
        return None

    def reset_plot_history(self) -> None:
        self.tk.step_history.clear()
        self.tk.loss_history.clear()
        self.tk.ssim_history.clear()
        self.tk.psnr_history.clear()

    def shutdown(self) -> None:
        return None

    def viewport_size(self) -> tuple[int, int]:
        renderer = getattr(self._viewer.s, "renderer", None)
        return (
            int(getattr(renderer, "width", _DEFAULT_RENDER_WIDTH)),
            int(getattr(renderer, "height", _DEFAULT_RENDER_HEIGHT)),
        )


class HeadlessViewer(ViewerCore):
    def __init__(
        self,
        *,
        device: spy.Device,
        config: dict[str, Any],
        width: int = _DEFAULT_RENDER_WIDTH,
        height: int = _DEFAULT_RENDER_HEIGHT,
    ) -> None:
        viewer_cfg = config.get("viewer", {}) if isinstance(config.get("viewer", {}), dict) else {}
        state_cfg = viewer_cfg.get("state", {}) if isinstance(viewer_cfg.get("state", {}), dict) else {}
        self._device = device
        self.loss_debug_view_options = LOSS_DEBUG_OPTIONS
        self.s = ViewerState(max_prepass_memory_mb=int(state_cfg.get("max_prepass_memory_mb", DEFAULT_MAX_PREPASS_MEMORY_MB)))
        self.s.list_capacity_multiplier = int(state_cfg.get("list_capacity_multiplier", self.s.list_capacity_multiplier))
        self.s.background = spy.float3(*tuple(float(v) for v in state_cfg.get("background", (0.0, 0.0, 0.0))))
        _precompile_runtime_shaders(self.device)
        self.s.renderer = GaussianRenderSettings.from_renderer_params(width, height, _initial_renderer_params(self.s)).create_renderer(self.device)
        self.ui = build_ui(self.s.renderer)
        apply_config_overlay(self.ui._values, config)
        self.s.list_capacity_multiplier = int(self.ui._values.get("list_capacity_multiplier", self.s.list_capacity_multiplier))
        self.s.max_prepass_memory_mb = int(self.ui._values.get("max_prepass_memory_mb", self.s.max_prepass_memory_mb))
        self.s.move_speed = float(self.ui._values.get("move_speed", self.s.move_speed))
        self.s.fov_y = float(self.ui._values.get("fov", self.s.fov_y))
        self.toolkit = _HeadlessToolkit(self)
        session.create_debug_shaders(self)

    @property
    def device(self) -> spy.Device:
        return self._device

    def _run_action(self, action, *, close_colmap_import: bool = False) -> None:
        action()
        self.s.last_error = ""
        if close_colmap_import:
            self.toolkit.close_colmap_import_window()

    def update_camera(self, _dt: float) -> None:
        return None

    def shutdown(self) -> None:
        self.toolkit.shutdown()
        wait = getattr(self.device, "wait", None)
        if callable(wait):
            wait()


def _section(config: dict[str, Any], *keys: str) -> dict[str, Any]:
    node: object = config
    for key in keys:
        if not isinstance(node, dict):
            return {}
        node = node.get(key, {})
    return node if isinstance(node, dict) else {}


def _run_config(config: dict[str, Any]) -> dict[str, Any]:
    return _section(config, "viewer", "run")


def _path_or_none(value: object) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text == "" else Path(text)


def _run_path(run: dict[str, Any], key: str) -> Path | None:
    return _path_or_none(run.get(key))


def _graphics_api(config: dict[str, Any]) -> str:
    value = _section(config, "viewer", "ui").get("graphics_api", "vulkan")
    return "dx12" if str(value).strip().lower() in {"dx12", "d3d12"} else "vulkan"


def _events_from_schedule(schedule: object, train_iters: int) -> tuple[ScheduledAction, ...]:
    if not isinstance(schedule, list):
        return ()
    actions: list[ScheduledAction] = []
    for raw in schedule:
        if not isinstance(raw, dict):
            continue
        action = str(raw.get("do", "")).strip()
        if action == "":
            continue
        steps: set[int] = set()
        at = raw.get("at")
        if isinstance(at, list):
            steps.update(int(step) for step in at)
        elif at is not None:
            steps.add(int(at))
        every = raw.get("every")
        if every is not None:
            interval = max(int(every), 1)
            steps.update(range(interval, max(int(train_iters), 0) + 1, interval))
        for step in sorted(step for step in steps if 0 <= int(step) <= max(int(train_iters), 0)):
            actions.append(ScheduledAction(step=int(step), action=action, config=dict(raw)))
    return tuple(sorted(actions, key=lambda item: (item.step, item.action)))


def _next_event_step(events: tuple[ScheduledAction, ...], current_step: int) -> int | None:
    for event in events:
        if event.step > int(current_step):
            return int(event.step)
    return None


def _events_at(events: tuple[ScheduledAction, ...], step: int) -> tuple[ScheduledAction, ...]:
    return tuple(event for event in events if int(event.step) == int(step))


def _event_output_dir(event: ScheduledAction, run: dict[str, Any], default_name: str) -> Path:
    value = event.config.get("output_dir", run.get("capture_output_dir"))
    return Path(value) if value is not None else Path("outputs") / "headless" / default_name


def _stats_cfg(run: dict[str, Any]) -> dict[str, Any]:
    stats = run.get("stats", {})
    return stats if isinstance(stats, dict) else {}


def _metric_value(state: object, key: str) -> float:
    value = float(getattr(state, key, float("nan")))
    return value if np.isfinite(value) else float("nan")


def _stats_row(viewer: HeadlessViewer) -> dict[str, object]:
    trainer = viewer.s.trainer
    state = getattr(trainer, "state", SimpleNamespace(step=0))
    return {
        "step": int(getattr(state, "step", 0)),
        "splats": int(getattr(getattr(trainer, "scene", None), "count", getattr(getattr(viewer.s, "scene", None), "count", 0))),
        "last_loss": _metric_value(state, "last_loss"),
        "avg_loss": _metric_value(state, "avg_loss"),
        "last_mse": _metric_value(state, "last_mse"),
        "avg_mse": _metric_value(state, "avg_mse"),
        "last_ssim": _metric_value(state, "last_ssim"),
        "avg_ssim": _metric_value(state, "avg_ssim"),
        "last_psnr": _metric_value(state, "last_psnr"),
        "avg_psnr": _metric_value(state, "avg_psnr"),
    }


def _open_stats_writer(stats: dict[str, Any]) -> tuple[Any | None, csv.DictWriter | None]:
    output = _path_or_none(stats.get("output"))
    if output is None:
        return None, None
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = output.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=_STATS_FIELDS)
    writer.writeheader()
    return handle, writer


def _emit_stats(viewer: HeadlessViewer, stats: dict[str, Any], writer: csv.DictWriter | None, step: int, *, force: bool = False) -> bool:
    interval = int(stats.get("log_interval", 0) or 0)
    if not force and (interval <= 0 or int(step) % interval != 0):
        return False
    row = _stats_row(viewer)
    print(
        "step={step} splats={splats} loss={avg_loss:.6g} psnr={avg_psnr:.3f}dB ssim={avg_ssim:.5f}".format(
            **row
        )
    )
    if writer is not None:
        writer.writerow(row)
    return True


def _write_per_view_metrics(viewer: HeadlessViewer, stats: dict[str, Any]) -> Path | None:
    output = _path_or_none(stats.get("per_view_output") or stats.get("per_view_metrics_output"))
    trainer = viewer.s.trainer
    if output is None or trainer is None or not hasattr(trainer, "frame_metrics_snapshot"):
        return None
    snapshot = trainer.frame_metrics_snapshot()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("frame_index", "image_name", "loss", "mse", "ssim", "psnr", "visited")
    frames = tuple(getattr(viewer.s, "training_frames", ()))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        loss = np.asarray(snapshot.get("loss", ()), dtype=np.float64)
        mse = np.asarray(snapshot.get("mse", ()), dtype=np.float64)
        ssim = np.asarray(snapshot.get("ssim", ()), dtype=np.float64)
        psnr = np.asarray(snapshot.get("psnr", ()), dtype=np.float64)
        visited = np.asarray(snapshot.get("visited", ()), dtype=np.int64)
        count = max(loss.size, mse.size, ssim.size, psnr.size, visited.size)
        for frame_index in range(count):
            frame = frames[frame_index] if frame_index < len(frames) else None
            writer.writerow(
                {
                    "frame_index": frame_index,
                    "image_name": Path(getattr(frame, "image_path", f"frame_{frame_index}")).name,
                    "loss": float(loss[frame_index]) if frame_index < loss.size else float("nan"),
                    "mse": float(mse[frame_index]) if frame_index < mse.size else float("nan"),
                    "ssim": float(ssim[frame_index]) if frame_index < ssim.size else float("nan"),
                    "psnr": float(psnr[frame_index]) if frame_index < psnr.size else float("nan"),
                    "visited": int(visited[frame_index]) if frame_index < visited.size else 0,
                }
            )
    return output


def _capture_summary(viewer: HeadlessViewer) -> frame_capture.PythonFrameCaptureSummary:
    return presenter._python_frame_capture_summary(viewer)


def _run_with_captures(viewer: HeadlessViewer, run: dict[str, Any], events: tuple[ScheduledAction, ...], step: int, action: Callable[[], int]) -> int:
    capture_events = tuple(event for event in events if event.action in {"capture_python", "capture_renderdoc"})
    renderdoc_events = tuple(event for event in capture_events if event.action == "capture_renderdoc")
    python_events = tuple(event for event in capture_events if event.action == "capture_python")

    def _submit() -> int:
        return int(action())

    wrapped: Callable[[], int] = _submit
    if python_events:
        event = python_events[0]
        directory = _event_output_dir(event, run, "python_capture")

        def _python_wrapped(inner: Callable[[], int] = wrapped) -> int:
            result = {"steps": 0}
            frame_capture.capture_python_frame(
                lambda: result.update(steps=inner()),
                frame_index=step,
                directory=directory,
                summary_provider=lambda: _capture_summary(viewer),
            )
            return int(result["steps"])

        wrapped = _python_wrapped
    if not renderdoc_events:
        return wrapped()
    event = renderdoc_events[0]
    capture = frame_capture.begin_renderdoc_frame_capture(
        device=viewer.device,
        window=None,
        frame_index=step,
        directory=_event_output_dir(event, run, "renderdoc"),
    )
    try:
        return wrapped()
    finally:
        frame_capture.end_renderdoc_frame_capture(capture)


def _capture_resources(viewer: HeadlessViewer, event: ScheduledAction, run: dict[str, Any]) -> Path:
    snapshot = collect_resource_debug_snapshot(viewer, include_process_vram=True)
    return write_resource_debug_log(snapshot, directory=_event_output_dir(event, run, "resources"))


def _active_renderer(viewer: HeadlessViewer):
    return viewer.s.training_renderer if viewer.s.trainer is not None and viewer.s.training_renderer is not None else viewer.s.renderer


def _texture_active_region(texture: object, width: int, height: int) -> np.ndarray:
    return np.asarray(texture.to_numpy())[: int(height), : int(width)].copy()


def _render_plain_snapshot(viewer: HeadlessViewer, path: Path, width: int, height: int) -> None:
    renderer = _active_renderer(viewer)
    if renderer is None:
        raise RuntimeError("No renderer is available for a headless render snapshot.")
    set_resolution = getattr(renderer, "set_render_resolution", None)
    if callable(set_resolution):
        set_resolution(int(width), int(height))
    tex, _stats = renderer.render_to_texture(viewer.camera(), background=viewer.s.background)
    save_snapshot(path, _texture_active_region(tex, int(renderer.width), int(renderer.height)))


def _render_debug_snapshot(viewer: HeadlessViewer, path: Path, width: int, height: int, view_name: str) -> None:
    labels = tuple(key for key, _label in LOSS_DEBUG_OPTIONS)
    if view_name not in labels:
        raise ValueError(f"Unknown debug view: {view_name}")
    viewer.ui._values["loss_debug_view"] = labels.index(view_name)
    encoder = viewer.device.create_command_encoder()
    tex = presenter._render_debug_view(viewer, encoder, int(width), int(height), int(getattr(viewer.s, "render_frame_index", 0)))
    viewer.device.submit_command_buffer(encoder.finish())
    save_snapshot(path, _texture_active_region(tex, int(width), int(height)))


def _render_snapshots(viewer: HeadlessViewer, render_cfg: object, *, step: int | None = None) -> tuple[Path, ...]:
    cfg = render_cfg if isinstance(render_cfg, dict) else {}
    if not cfg:
        return ()
    output_dir = Path(cfg.get("output_dir", Path("outputs") / "headless" / "renders"))
    output_dir.mkdir(parents=True, exist_ok=True)
    width = int(cfg.get("width", _DEFAULT_RENDER_WIDTH))
    height = int(cfg.get("height", _DEFAULT_RENDER_HEIGHT))
    views = cfg.get("views", ("rendered",))
    if isinstance(views, str):
        views = (views,)
    suffix = "final" if step is None else f"step_{int(step):06d}"
    paths: list[Path] = []
    for view_name in tuple(str(view) for view in views):
        path = output_dir / f"{suffix}_{view_name}.png"
        if view_name in {"render", "main", "plain"}:
            _render_plain_snapshot(viewer, path, width, height)
        elif view_name == "rendered" and viewer.s.trainer is None:
            _render_plain_snapshot(viewer, path, width, height)
        else:
            _render_debug_snapshot(viewer, path, width, height, view_name)
        paths.append(path)
    return tuple(paths)


def _run_post_step_events(viewer: HeadlessViewer, run: dict[str, Any], events: tuple[ScheduledAction, ...], step: int) -> None:
    for event in events:
        if event.action in {"capture_python", "capture_renderdoc"}:
            continue
        if event.action in {"capture_buffers", "capture_buffer", "capture_resources"}:
            path = _capture_resources(viewer, event, run)
            print(f"Resource capture: {path}")
        elif event.action == "render":
            for path in _render_snapshots(viewer, event.config, step=step):
                print(f"Render snapshot: {path}")
        else:
            raise ValueError(f"Unsupported headless scheduled action: {event.action}")


def _run_source_stage(viewer: HeadlessViewer, run: dict[str, Any]) -> None:
    source = str(run.get("source", "colmap") or "colmap").strip().lower()
    if source == "colmap":
        session.import_colmap_from_ui(viewer)
        while viewer.s.colmap_import_progress is not None:
            session.advance_colmap_import(viewer)
        return
    if source == "ply":
        ply_path = _run_path(run, "ply_path")
        if ply_path is None:
            raise ValueError("viewer.run.ply_path is required when source is 'ply'.")
        session.load_scene(viewer, ply_path)
        return
    raise ValueError(f"Unsupported headless source: {source}")


def _run_photometric_stage(viewer: HeadlessViewer, run: dict[str, Any]) -> None:
    photometric = run.get("photometric")
    cfg = photometric if isinstance(photometric, dict) else {}
    steps = int(cfg.get("steps", 0) or 0)
    if steps <= 0:
        return
    session.set_photometric_active(viewer, True)
    while bool(getattr(viewer.s, "photometric_prepare_pending_active", False)):
        session.advance_photometric_initialization(viewer)
    trainer = getattr(viewer.s, "photometric_trainer", None)
    while trainer is not None and int(getattr(trainer.state, "step", 0)) < steps:
        remaining = steps - int(getattr(trainer.state, "step", 0))
        presenter_state._run_photometric_batch(viewer, max_steps=remaining)
        trainer = getattr(viewer.s, "photometric_trainer", None)
    session.set_photometric_active(viewer, False)


def _run_training_stage(viewer: HeadlessViewer, run: dict[str, Any]) -> None:
    train_iters = int(run.get("train_iters", 0) or 0)
    if train_iters <= 0:
        return
    stats = _stats_cfg(run)
    events = _events_from_schedule(run.get("schedule", ()), train_iters)
    stats_handle, stats_writer = _open_stats_writer(stats)
    last_stats_step: int | None = None
    try:
        session.set_training_active(viewer, True)
        while viewer.s.trainer is not None and int(viewer.s.trainer.state.step) < train_iters:
            session.apply_live_params(viewer)
            if bool(getattr(viewer.s, "pending_training_runtime_resize", False)):
                session.ensure_training_runtime_resolution(viewer)
            step = int(viewer.s.trainer.state.step)
            budget = min(train_iters - step, presenter_state._training_steps_per_frame(viewer))
            next_step = _next_event_step(events, step)
            if next_step is not None:
                budget = min(budget, next_step - step)
            target_step = step + budget
            due = _events_at(events, target_step) if target_step == next_step else ()
            executed = _run_with_captures(
                viewer,
                run,
                due,
                target_step,
                lambda: presenter_state._run_training_batch(viewer, max_steps=budget),
            )
            if executed <= 0:
                raise RuntimeError("Training batch made no progress.")
            if bool(getattr(viewer.s, "training_runtime_factor_changed", False)):
                session.ensure_training_runtime_resolution(viewer)
                viewer.s.training_runtime_factor_changed = False
            step = int(viewer.s.trainer.state.step)
            _run_post_step_events(viewer, run, _events_at(events, step), step)
            if _emit_stats(viewer, stats, stats_writer, step):
                last_stats_step = step
        final_step = int(getattr(viewer.s.trainer.state, "step", 0)) if viewer.s.trainer is not None else train_iters
        if final_step != last_stats_step:
            _emit_stats(viewer, stats, stats_writer, final_step, force=True)
    finally:
        if stats_handle is not None:
            stats_handle.close()
        session.set_training_active(viewer, False)


def _run_metrics_stage(viewer: HeadlessViewer, run: dict[str, Any]) -> None:
    metrics = run.get("metrics")
    if metrics is None:
        return
    cfg = metrics if isinstance(metrics, dict) else {}
    scene_ply = _path_or_none(cfg.get("scene_ply"))
    if scene_ply is not None:
        if viewer.s.trainer is None:
            raise RuntimeError("metrics.scene_ply requires an initialized COLMAP trainer.")
        viewer.s.trainer.replace_scene(load_gaussian_ply(scene_ply))
    session.start_dataset_metrics_logging(viewer, output_path=_path_or_none(cfg.get("output")))
    while viewer.s.dataset_metrics_task is not None:
        session.advance_dataset_metrics(viewer)


def _run_export_stage(viewer: HeadlessViewer, run: dict[str, Any]) -> Path | None:
    output_ply = _run_path(run, "output_ply")
    if output_ply is None:
        return None
    scene = viewer._export_source_scene()
    include_sh = viewer._export_should_include_sh()
    saved_path = save_gaussian_ply(output_ply, scene, include_sh=include_sh)
    print(f"Exported scene: {saved_path} ({scene.count:,} splats)")
    return saved_path


def run_headless(viewer: HeadlessViewer, config: dict[str, Any]) -> int:
    run = _run_config(config)
    session.apply_live_params(viewer)
    _run_source_stage(viewer, run)
    _run_photometric_stage(viewer, run)
    if bool(run.get("reinitialize", False)):
        session.reinitialize_training_scene(viewer)
    _run_training_stage(viewer, run)
    _run_metrics_stage(viewer, run)
    _write_per_view_metrics(viewer, _stats_cfg(run))
    _render_snapshots(viewer, run.get("render"))
    _run_export_stage(viewer, run)
    return 0


def run_headless_from_config(config_path: str | Path, *, graphics_api: str | None = None) -> int:
    config = load_config(config_path)
    api_name = _graphics_api(config) if graphics_api is None else str(graphics_api)
    device = create_default_device(device_type=device_type_from_name(api_name), enable_debug_layers=False)
    viewer = HeadlessViewer(device=device, config=config)
    try:
        return run_headless(viewer, config)
    finally:
        viewer.shutdown()
