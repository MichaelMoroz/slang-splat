from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import time

from src.viewer import presenter
from src.viewer.state import ColmapImportProgress


class _DummyEncoder:
    def __init__(self) -> None:
        self.blit_calls: list[tuple[object, object]] = []
        self.clear_calls: list[tuple[object, list[float]]] = []

    def blit(self, dst: object, src: object) -> None:
        self.blit_calls.append((dst, src))

    def clear_texture_float(self, texture: object, clear_value: list[float]) -> None:
        self.clear_calls.append((texture, clear_value))

    def finish(self) -> str:
        return "finished"


class _DummyRenderer:
    def __init__(self, width: int = 640, height: int = 360) -> None:
        self.width = width
        self.height = height

    def render_to_texture(self, camera: object, background: object, read_stats: bool, command_encoder: object) -> tuple[object, dict[str, int | bool | float]]:
        return object(), {"generated_entries": 1, "written_entries": 2, "overflow": False}


class _DummyTrainer:
    def __init__(self) -> None:
        self.state = SimpleNamespace(step=0, last_loss=0.0, avg_loss=0.0, last_mse=0.0, avg_mse=0.0, last_psnr=float("inf"), avg_psnr=float("inf"), last_density_loss=0.0, avg_density_loss=0.0, last_frame_index=0, last_instability="")
        self.scene = SimpleNamespace(count=4)
        self.training = SimpleNamespace(
            train_downscale_mode=1,
            train_auto_start_downscale=1,
            train_downscale_max_iters=30000,
            lr_schedule_enabled=True,
            lr_schedule_start_lr=1e-3,
            lr_schedule_end_lr=1e-4,
            lr_schedule_steps=30000,
            maintenance_interval=200,
            maintenance_growth_ratio=0.02,
            maintenance_growth_start_step=500,
            maintenance_alpha_cull_threshold=1e-2,
            maintenance_contribution_cull_threshold=0.001,
            maintenance_contribution_cull_decay=0.95,
            density_regularizer=0.05,
            max_allowed_density=12.0,
            max_gaussians=1000000,
        )
        self.step_calls = 0
        self.step_batch_calls: list[int] = []

    def step(self) -> None:
        self.step_calls += 1

    def step_batch(self, steps: int) -> int:
        self.step_batch_calls.append(int(steps))
        self.step_calls += int(steps)
        return int(steps)

    def make_frame_camera(self, frame_index: int, width: int, height: int) -> tuple[int, int, int]:
        return (frame_index, width, height)

    def effective_train_downscale_factor(self) -> int:
        return 1

    def current_base_lr(self) -> float:
        return 1e-3

    def get_frame_target_texture(self, frame_index: int, native_resolution: bool = True, encoder: object | None = None) -> str:
        return f"target_tex_{frame_index}_{native_resolution}"


class _CaptureKernel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def dispatch(self, *, thread_count: object, vars: dict[str, object], command_encoder: object) -> None:
        self.calls.append({"thread_count": thread_count, "vars": vars, "command_encoder": command_encoder})


def _control(value: object) -> SimpleNamespace:
    return SimpleNamespace(value=value)


def _text() -> SimpleNamespace:
    return SimpleNamespace(text="")


def _viewer(loss_debug: bool) -> SimpleNamespace:
    trainer = _DummyTrainer()
    controls = {
        "loss_debug": _control(loss_debug),
        "loss_debug_frame": _control(0),
        "loss_debug_view": _control(0),
        "loss_debug_abs_scale": _control(1.0),
        "images_subdir": _control(0),
        "training_steps_per_frame": _control(3),
        "train_downscale_factor": _control(1),
        "lr_schedule_enabled": _control(True),
        "lr_schedule_start_lr": _control(1e-3),
        "lr_schedule_end_lr": _control(1e-4),
        "lr_schedule_steps": _control(30000),
        "maintenance_interval": _control(200),
        "maintenance_growth_ratio": _control(0.02),
        "maintenance_growth_start_step": _control(500),
        "maintenance_alpha_cull_threshold": _control(1e-2),
        "maintenance_contribution_cull_threshold": _control(0.001),
        "maintenance_contribution_cull_decay": _control(0.95),
        "max_gaussians": _control(1000000),
        "train_downscale_mode": _control(1),
        "train_auto_start_downscale": _control(1),
        "train_downscale_max_iters": _control(30000),
    }
    texts = {key: _text() for key in ("fps", "images_subdir", "loss_debug_view", "loss_debug_frame", "path", "scene_stats", "render_stats", "training", "training_time", "training_iters_avg", "training_loss", "training_mse", "training_density", "training_psnr", "training_instability", "training_resolution", "training_downscale", "training_schedule", "training_maintenance", "colmap_import_status", "colmap_import_current", "histogram_status", "error")}
    viewer = SimpleNamespace()
    viewer.device = SimpleNamespace()
    viewer.toolkit = SimpleNamespace(viewport_size=lambda: (640, 360))
    viewer.loss_debug_view_options = (("rendered", "Rendered"), ("target", "Target"), ("abs_diff", "Abs Diff"))
    viewer.image_subdir_options = ("images_8",)
    viewer.ui = SimpleNamespace(controls=controls, texts=texts, _values={"show_histograms": False, "_histogram_payload": None, "_histogram_range_payload": None}, _texts={key: value.text for key, value in texts.items()})
    viewer.c = lambda key: viewer.ui.controls[key]
    viewer.t = lambda key: viewer.ui.texts[key]
    viewer.camera = lambda: "camera"
    viewer.update_camera = lambda dt: None
    viewer.s = SimpleNamespace(
        fps_smooth=60.0,
        last_time=time.perf_counter(),
        renderer=_DummyRenderer(),
        debug_renderer=None,
        scene=SimpleNamespace(count=4),
        stats={},
        scene_path=None,
        colmap_root=Path("dataset"),
        training_frames=[SimpleNamespace(image_path=Path("frame.png"), width=640, height=360)],
        trainer=trainer,
        training_active=True,
        training_renderer=_DummyRenderer(),
        background=(0.0, 0.0, 0.0),
        training_elapsed_s=0.0,
        training_resume_time=None,
        last_render_exception="",
        last_error="",
        last_training_batch_steps=0,
        viewport_texture=None,
        colmap_import_progress=None,
        cached_raster_grad_ranges=None,
    )
    return viewer


def test_render_frame_uses_debug_branch_when_visual_loss_debug_enabled(monkeypatch):
    viewer = _viewer(loss_debug=True)
    render_context = SimpleNamespace(surface_texture=SimpleNamespace(width=640, height=360), command_encoder=_DummyEncoder())
    calls: list[str] = []

    monkeypatch.setattr(presenter.session, "apply_live_params", lambda viewer_obj: calls.append("apply"))
    monkeypatch.setattr(presenter.session, "ensure_training_runtime_resolution", lambda viewer_obj: calls.append("train_resize"))
    monkeypatch.setattr(presenter.session, "recreate_renderer", lambda viewer_obj, width, height: calls.append("resize"))
    monkeypatch.setattr(presenter.session, "update_debug_frame_slider_range", lambda viewer_obj: None)
    monkeypatch.setattr(presenter, "_render_debug_view", lambda viewer_obj, encoder, width, height: calls.append("debug") or "debug_tex")
    monkeypatch.setattr(presenter, "_render_main_view", lambda viewer_obj, encoder: calls.append("main") or "main_tex")
    monkeypatch.setattr(presenter, "update_ui_text", lambda viewer_obj, dt: calls.append("ui"))

    presenter.render_frame(viewer, render_context)

    assert viewer.s.trainer.step_calls == 3
    assert viewer.s.viewport_texture == "debug_tex"
    assert calls == ["apply", "debug", "ui"]


def test_render_frame_uses_main_branch_when_visual_loss_debug_disabled(monkeypatch):
    viewer = _viewer(loss_debug=False)
    render_context = SimpleNamespace(surface_texture=SimpleNamespace(width=640, height=360), command_encoder=_DummyEncoder())
    calls: list[str] = []

    monkeypatch.setattr(presenter.session, "apply_live_params", lambda viewer_obj: calls.append("apply"))
    monkeypatch.setattr(presenter.session, "ensure_training_runtime_resolution", lambda viewer_obj: calls.append("train_resize"))
    monkeypatch.setattr(presenter.session, "recreate_renderer", lambda viewer_obj, width, height: calls.append("resize"))
    monkeypatch.setattr(presenter.session, "update_debug_frame_slider_range", lambda viewer_obj: None)
    monkeypatch.setattr(presenter, "_render_debug_view", lambda viewer_obj, encoder, width, height: calls.append("debug") or "debug_tex")
    monkeypatch.setattr(presenter, "_render_main_view", lambda viewer_obj, encoder: calls.append("main") or "main_tex")
    monkeypatch.setattr(presenter, "update_ui_text", lambda viewer_obj, dt: calls.append("ui"))

    presenter.render_frame(viewer, render_context)

    assert viewer.s.trainer.step_calls == 3
    assert viewer.s.viewport_texture == "main_tex"
    assert calls == ["apply", "main", "ui"]


def test_render_frame_runs_configured_training_batch(monkeypatch):
    viewer = _viewer(loss_debug=False)
    viewer.c("training_steps_per_frame").value = 3
    render_context = SimpleNamespace(surface_texture=SimpleNamespace(width=640, height=360), command_encoder=_DummyEncoder())
    calls: list[str] = []

    monkeypatch.setattr(presenter.session, "apply_live_params", lambda viewer_obj: calls.append("apply"))
    monkeypatch.setattr(presenter.session, "ensure_training_runtime_resolution", lambda viewer_obj: calls.append("train_resize"))
    monkeypatch.setattr(presenter.session, "recreate_renderer", lambda viewer_obj, width, height: calls.append("resize"))
    monkeypatch.setattr(presenter.session, "update_debug_frame_slider_range", lambda viewer_obj: None)
    monkeypatch.setattr(presenter, "_render_debug_view", lambda viewer_obj, encoder, width, height: calls.append("debug") or "debug_tex")
    monkeypatch.setattr(presenter, "_render_main_view", lambda viewer_obj, encoder: calls.append("main") or "main_tex")
    monkeypatch.setattr(presenter, "update_ui_text", lambda viewer_obj, dt: calls.append("ui"))

    presenter.render_frame(viewer, render_context)

    assert viewer.s.trainer.step_calls == 3
    assert viewer.s.trainer.step_batch_calls == [3]
    assert viewer.s.last_training_batch_steps == 3
    assert calls == ["apply", "main", "ui"]


def test_render_frame_refreshes_histograms_when_window_open(monkeypatch):
    viewer = _viewer(loss_debug=False)
    viewer.ui._values["show_histograms"] = True
    render_context = SimpleNamespace(surface_texture=SimpleNamespace(width=640, height=360), command_encoder=_DummyEncoder())
    calls: list[str] = []

    monkeypatch.setattr(presenter.session, "apply_live_params", lambda viewer_obj: calls.append("apply"))
    monkeypatch.setattr(presenter.session, "ensure_training_runtime_resolution", lambda viewer_obj: calls.append("train_resize"))
    monkeypatch.setattr(presenter.session, "recreate_renderer", lambda viewer_obj, width, height: calls.append("resize"))
    monkeypatch.setattr(presenter.session, "refresh_cached_raster_grad_histograms", lambda viewer_obj: calls.append("hist"))
    monkeypatch.setattr(presenter, "_render_main_view", lambda viewer_obj, encoder: calls.append("main") or "main_tex")
    monkeypatch.setattr(presenter, "update_ui_text", lambda viewer_obj, dt: calls.append("ui"))

    presenter.render_frame(viewer, render_context)

    assert calls == ["apply", "main", "hist", "ui"]


def test_render_frame_handles_resize_failure_without_raising(monkeypatch):
    viewer = _viewer(loss_debug=False)
    viewer.s.renderer.width = 320
    viewer.s.renderer.height = 180
    render_context = SimpleNamespace(surface_texture=SimpleNamespace(width=640, height=360), command_encoder=_DummyEncoder())
    calls: list[str] = []

    monkeypatch.setattr(presenter.session, "apply_live_params", lambda viewer_obj: calls.append("apply"))
    monkeypatch.setattr(presenter.session, "ensure_training_runtime_resolution", lambda viewer_obj: calls.append("train_resize"))
    monkeypatch.setattr(presenter.session, "recreate_renderer", lambda viewer_obj, width, height: (_ for _ in ()).throw(RuntimeError("resize boom")))
    monkeypatch.setattr(presenter, "_render_main_view", lambda viewer_obj, encoder: calls.append("main") or "main_tex")
    monkeypatch.setattr(presenter, "update_ui_text", lambda viewer_obj, dt: calls.append("ui"))

    presenter.render_frame(viewer, render_context)

    assert viewer.s.training_active is False
    assert viewer.s.last_error == "resize boom"
    assert viewer.s.last_render_exception == "resize boom"
    assert render_context.command_encoder.clear_calls[-1] == (render_context.surface_texture, [0.0, 0.0, 0.0, 1.0])
    assert calls == ["apply", "ui"]


def test_render_frame_handles_live_param_failure_without_raising(monkeypatch):
    viewer = _viewer(loss_debug=False)
    render_context = SimpleNamespace(surface_texture=SimpleNamespace(width=640, height=360), command_encoder=_DummyEncoder())
    calls: list[str] = []

    monkeypatch.setattr(presenter.session, "apply_live_params", lambda viewer_obj: (_ for _ in ()).throw(RuntimeError("live params boom")))
    monkeypatch.setattr(presenter.session, "advance_colmap_import", lambda viewer_obj: calls.append("import"))
    monkeypatch.setattr(presenter, "_render_main_view", lambda viewer_obj, encoder: calls.append("main") or "main_tex")
    monkeypatch.setattr(presenter, "update_ui_text", lambda viewer_obj, dt: calls.append("ui"))

    presenter.render_frame(viewer, render_context)

    assert viewer.s.training_active is False
    assert viewer.s.last_error == "live params boom"
    assert viewer.s.last_render_exception == "live params boom"
    assert render_context.command_encoder.clear_calls == [(render_context.surface_texture, [0.0, 0.0, 0.0, 1.0])]
    assert calls == ["ui"]


def test_render_frame_handles_import_failure_without_raising(monkeypatch):
    viewer = _viewer(loss_debug=False)
    render_context = SimpleNamespace(surface_texture=SimpleNamespace(width=640, height=360), command_encoder=_DummyEncoder())
    calls: list[str] = []

    monkeypatch.setattr(presenter.session, "apply_live_params", lambda viewer_obj: calls.append("apply"))
    monkeypatch.setattr(presenter.session, "advance_colmap_import", lambda viewer_obj: (_ for _ in ()).throw(RuntimeError("import boom")))
    monkeypatch.setattr(presenter, "_render_main_view", lambda viewer_obj, encoder: calls.append("main") or "main_tex")
    monkeypatch.setattr(presenter, "update_ui_text", lambda viewer_obj, dt: calls.append("ui"))

    presenter.render_frame(viewer, render_context)

    assert viewer.s.training_active is False
    assert viewer.s.last_error == "import boom"
    assert viewer.s.last_render_exception == "import boom"
    assert render_context.command_encoder.clear_calls == [(render_context.surface_texture, [0.0, 0.0, 0.0, 1.0])]
    assert calls == ["apply", "ui"]


def test_update_ui_text_reports_training_schedule_and_maintenance() -> None:
    viewer = _viewer(loss_debug=False)

    presenter.update_ui_text(viewer, 1.0 / 60.0)

    assert viewer.t("training_schedule").text == "LR Schedule: cosine 1.00e-03 -> 1.00e-04 | steps=30,000 | current=1.00e-03"
    assert viewer.t("training_maintenance").text == "Maintenance: every 200 | growth=0.00% now | target=2.00% after 500 | alpha<1.00e-02 or contrib<0.001% culled | decay=95.00%/pass | max=1,000,000"


def test_render_frame_recovers_missing_main_renderer_by_recreating_it(monkeypatch):
    viewer = _viewer(loss_debug=False)
    viewer.s.renderer = None
    render_context = SimpleNamespace(surface_texture=SimpleNamespace(width=640, height=360), command_encoder=_DummyEncoder())
    calls: list[object] = []
    replacement_renderer = _DummyRenderer()

    monkeypatch.setattr(presenter.session, "apply_live_params", lambda viewer_obj: calls.append("apply"))
    monkeypatch.setattr(presenter.session, "advance_colmap_import", lambda viewer_obj: calls.append("import"))
    monkeypatch.setattr(presenter.session, "recreate_renderer", lambda viewer_obj, width, height: calls.append(("resize", width, height)) or setattr(viewer_obj.s, "renderer", replacement_renderer))
    monkeypatch.setattr(presenter, "_render_main_view", lambda viewer_obj, encoder: calls.append("main") or "main_tex")
    monkeypatch.setattr(presenter, "update_ui_text", lambda viewer_obj, dt: calls.append("ui"))

    presenter.render_frame(viewer, render_context)

    assert viewer.s.renderer is replacement_renderer
    assert viewer.s.viewport_texture == "main_tex"
    assert calls == ["apply", "import", ("resize", 640, 360), "main", "ui"]


def test_render_frame_consumes_pending_reinitialize_before_live_params(monkeypatch):
    viewer = _viewer(loss_debug=False)
    viewer.s.training_active = False
    viewer.s.pending_training_reinitialize = True
    render_context = SimpleNamespace(surface_texture=SimpleNamespace(width=640, height=360), command_encoder=_DummyEncoder())
    calls: list[str] = []

    monkeypatch.setattr(presenter.session, "initialize_training_scene", lambda viewer_obj: calls.append("init"))
    monkeypatch.setattr(presenter.session, "apply_live_params", lambda viewer_obj: calls.append("apply"))
    monkeypatch.setattr(presenter.session, "advance_colmap_import", lambda viewer_obj: calls.append("import"))
    monkeypatch.setattr(presenter, "_render_main_view", lambda viewer_obj, encoder: calls.append("main") or "main_tex")
    monkeypatch.setattr(presenter, "update_ui_text", lambda viewer_obj, dt: calls.append("ui"))

    presenter.render_frame(viewer, render_context)

    assert viewer.s.pending_training_reinitialize is False
    assert viewer.s.trainer.step_calls == 0
    assert calls == ["init", "apply", "import", "main", "ui"]


def test_render_frame_resizes_main_renderer_from_viewport_size(monkeypatch):
    viewer = _viewer(loss_debug=False)
    viewer.toolkit.viewport_size = lambda: (480, 270)
    viewer.s.renderer.width = 640
    viewer.s.renderer.height = 360
    render_context = SimpleNamespace(surface_texture=SimpleNamespace(width=1280, height=720), command_encoder=_DummyEncoder())
    calls: list[object] = []

    monkeypatch.setattr(presenter.session, "apply_live_params", lambda viewer_obj: calls.append("apply"))
    monkeypatch.setattr(presenter.session, "advance_colmap_import", lambda viewer_obj: calls.append("import"))
    monkeypatch.setattr(presenter.session, "recreate_renderer", lambda viewer_obj, width, height: calls.append(("resize", width, height)) or setattr(viewer_obj.s.renderer, "width", width) or setattr(viewer_obj.s.renderer, "height", height))
    monkeypatch.setattr(presenter, "_render_main_view", lambda viewer_obj, encoder: calls.append("main") or "main_tex")
    monkeypatch.setattr(presenter, "update_ui_text", lambda viewer_obj, dt: calls.append("ui"))

    presenter.render_frame(viewer, render_context)

    assert viewer.s.viewport_texture == "main_tex"
    assert ("resize", 480, 270) in calls


def test_update_ui_text_uses_permutation_averages() -> None:
    viewer = _viewer(loss_debug=False)
    viewer.s.trainer.state.avg_loss = 1.25
    viewer.s.trainer.state.avg_mse = 2.5e-3
    viewer.s.trainer.state.avg_density_loss = 6.5e-3
    viewer.s.trainer.state.avg_psnr = 26.75
    viewer.s.trainer.state.step = 120
    viewer.s.training_elapsed_s = 30.0

    presenter.update_ui_text(viewer, 1.0 / 60.0)

    assert viewer.t("training_time").text == "Time: 00:30"
    assert viewer.t("training_iters_avg").text == "Avg it/s: 4.00"
    assert viewer.t("training_loss").text == "Loss Avg: 1.250000e+00"
    assert viewer.t("training_mse").text == "MSE Avg: 2.500000e-03"
    assert viewer.t("training_density").text == "Density Avg: 6.500000e-03"
    assert viewer.t("training_psnr").text == "PSNR Avg: 26.750 dB"
    assert viewer.t("training_resolution").text == "Train Res: 640x360 (N=1)"


def test_update_ui_text_populates_colmap_import_progress_fields() -> None:
    viewer = _viewer(loss_debug=False)
    viewer.s.colmap_import_progress = ColmapImportProgress(
        dataset_root=Path("dataset"),
        colmap_root=Path("dataset"),
        database_path=None,
        images_root=Path("dataset/images"),
        init_mode="pointcloud",
        custom_ply_path=None,
        image_downscale_mode="original",
        image_downscale_target_width=1600,
        image_downscale_scale=1.0,
        nn_radius_scale_coef=0.25,
        phase="load_textures",
        current=3,
        total=8,
        current_name="frame_003.png",
    )

    presenter.update_ui_text(viewer, 1.0 / 60.0)

    assert viewer.ui._values["_colmap_import_active"] is True
    assert viewer.ui._values["_colmap_import_fraction"] == 3.0 / 8.0
    assert viewer.t("colmap_import_status").text == "Loading images: 3/8"
    assert viewer.t("colmap_import_current").text == "frame_003.png"


def test_render_debug_source_uses_overlay_renderer_for_rendered_view(monkeypatch) -> None:
    viewer = _viewer(loss_debug=True)
    viewer.c("loss_debug_view").value = 0
    overlay_renderer = _DummyRenderer()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(presenter.session, "ensure_renderer", lambda viewer_obj, attr, width, height, allow_debug_overlays: calls.append(("ensure", allow_debug_overlays)) or overlay_renderer)
    monkeypatch.setattr(presenter.session, "sync_scene_from_training_renderer", lambda viewer_obj, dst_renderer, target: calls.append(("sync", target)))

    source_tex, stats, width, height = presenter._render_debug_source(viewer, _DummyEncoder(), 0)

    assert width == 640
    assert height == 360
    assert stats["generated_entries"] == 1
    assert calls == [("ensure", True), ("sync", "debug")]


def test_render_debug_source_uses_training_renderer_for_abs_diff(monkeypatch) -> None:
    viewer = _viewer(loss_debug=True)
    viewer.c("loss_debug_view").value = 2
    calls: list[str] = []

    monkeypatch.setattr(presenter.session, "ensure_renderer", lambda *args, **kwargs: calls.append("ensure"))
    monkeypatch.setattr(presenter.session, "sync_scene_from_training_renderer", lambda *args, **kwargs: calls.append("sync"))

    source_tex, stats, width, height = presenter._render_debug_source(viewer, _DummyEncoder(), 0)

    assert width == 640
    assert height == 360
    assert stats["written_entries"] == 2
    assert calls == []


def test_dispatch_debug_abs_diff_uses_runtime_ui_scale(monkeypatch) -> None:
    viewer = _viewer(loss_debug=True)
    viewer.c("loss_debug_abs_scale").value = 3.5
    viewer.s.debug_abs_diff_kernel = _CaptureKernel()
    encoder = _DummyEncoder()
    output = SimpleNamespace(width=640, height=360)

    monkeypatch.setattr(presenter, "_ensure_texture", lambda viewer_obj, attr, width, height: output)

    result = presenter._dispatch_debug_abs_diff(viewer, encoder, "rendered_tex", "target_tex", 640, 360)

    assert result is output
    assert len(viewer.s.debug_abs_diff_kernel.calls) == 1
    assert viewer.s.debug_abs_diff_kernel.calls[0]["vars"]["g_DebugDiffScale"] == 3.5


def test_dispatch_viewport_present_uses_present_kernel(monkeypatch) -> None:
    viewer = _viewer(loss_debug=False)
    viewer.s.debug_letterbox_kernel = _CaptureKernel()
    encoder = _DummyEncoder()
    output = SimpleNamespace(width=640, height=360)

    monkeypatch.setattr(presenter, "_ensure_texture", lambda viewer_obj, attr, width, height: output)

    result = presenter._dispatch_viewport_present(viewer, encoder, "source_tex", 320, 180, 640, 360)

    assert result is output
    assert len(viewer.s.debug_letterbox_kernel.calls) == 1
    vars = viewer.s.debug_letterbox_kernel.calls[0]["vars"]
    assert vars["g_LetterboxSource"] == "source_tex"
    assert vars["g_LetterboxSourceWidth"] == 320
    assert vars["g_LetterboxSourceHeight"] == 180
    assert vars["g_LetterboxOutputWidth"] == 640
    assert vars["g_LetterboxOutputHeight"] == 360
