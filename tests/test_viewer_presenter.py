from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import slangpy as spy

from src.viewer import presenter
from src.viewer import session as viewer_session
from src.viewer.state import ColmapImportProgress


class _DummyEncoder:
    def __init__(self) -> None:
        self.clear_calls: list[tuple[object, list[float]]] = []

    def clear_texture_float(self, texture: object, clear_value: list[float]) -> None:
        self.clear_calls.append((texture, clear_value))

    def finish(self) -> str:
        return "finished"


class _DummyRenderer:
    def __init__(self, width: int = 640, height: int = 360) -> None:
        self.width = width
        self.height = height
        self.sh_band = 0
        self.training_forward_calls: list[dict[str, object]] = []

    def render_to_texture(self, camera: object, background: object, read_stats: bool, command_encoder: object) -> tuple[object, dict[str, int | bool | float]]:
        return object(), {"generated_entries": 1, "written_entries": 2, "overflow": False}

    def render_training_forward_to_texture(self, camera: object, background: object, read_stats: bool, command_encoder: object, **kwargs) -> tuple[object, dict[str, int | bool | float]]:
        self.training_forward_calls.append({"camera": camera, "background": background, "read_stats": read_stats, "command_encoder": command_encoder, **kwargs})
        return "training_preview_tex", {"generated_entries": 1, "written_entries": 2, "overflow": False}


class _DummyTrainer:
    def __init__(self) -> None:
        self.state = SimpleNamespace(step=0, last_loss=0.0, avg_loss=0.0, last_mse=0.0, avg_mse=0.0, last_psnr=float("inf"), avg_psnr=float("inf"), avg_density_loss=0.0, last_frame_index=0, last_instability="")
        self.scene = SimpleNamespace(count=4)
        self.training = SimpleNamespace(
            camera_min_dist=0.1,
            train_downscale_mode=1,
            train_auto_start_downscale=1,
            train_subsample_factor=1,
            train_downscale_max_iters=30000,
            lr_schedule_enabled=True,
            lr_schedule_start_lr=0.005,
            lr_schedule_stage1_lr=0.002,
            lr_schedule_stage2_lr=0.001,
            lr_schedule_end_lr=1.5e-4,
            lr_schedule_steps=30000,
            lr_schedule_stage1_step=3000,
            lr_schedule_stage2_step=14000,
            lr_pos_mul=1.0,
            lr_pos_stage1_mul=0.75,
            lr_pos_stage2_mul=0.2,
            lr_pos_stage3_mul=0.2,
            lr_sh_mul=0.05,
            lr_sh_stage1_mul=0.05,
            lr_sh_stage2_mul=0.05,
            lr_sh_stage3_mul=0.05,
            refinement_interval=200,
            refinement_growth_ratio=0.05,
            refinement_growth_start_step=500,
            refinement_alpha_cull_threshold=1e-2,
            refinement_min_contribution=512,
            refinement_min_contribution_decay=0.995,
            refinement_opacity_mul=1.0,
            refinement_use_compact_split=True,
            refinement_solve_opacity=True,
            refinement_split_beta=0.28,
            refinement_loss_weight=0.25,
            refinement_target_edge_weight=0.75,
            density_regularizer=0.02,
            depth_ratio_weight=1.0,
            depth_ratio_stage1_weight=0.05,
            depth_ratio_stage2_weight=0.01,
            depth_ratio_stage3_weight=0.001,
            sorting_order_dithering=0.5,
            sorting_order_dithering_stage1=0.2,
            sorting_order_dithering_stage2=0.05,
            sorting_order_dithering_stage3=0.01,
            depth_ratio_grad_min=0.0,
            depth_ratio_grad_max=0.1,
            position_random_step_noise_stage1_lr=466666.6666666667,
            position_random_step_noise_stage2_lr=416666.6666666667,
            position_random_step_noise_stage3_lr=0.0,
            sh_band=0,
            sh_band_stage1=1,
            sh_band_stage2=1,
            sh_band_stage3=1,
            max_allowed_density=12.0,
            max_gaussians=1000000,
        )
        self.step_calls = 0
        self.step_batch_calls: list[int] = []
        self.training_resolution_calls: list[tuple[int, int]] = []
        self.sample_vars_calls: list[tuple[int, int, int | None]] = []
        self.background_seed_calls: list[int | None] = []
        self.hparam_calls: list[tuple[int | None, object]] = []
        self.sort_calls: list[tuple[int, int, object]] = []
        self.target_calls: list[tuple[int, bool]] = []
        self.subsample_factor = 1

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

    def effective_train_subsample_factor(self, frame_index: int = 0, step: int | None = None) -> int:
        return int(self.subsample_factor)

    def effective_train_render_factor(self) -> int:
        return 1

    def training_resolution(self, frame_index: int = 0, step: int | None = None) -> tuple[int, int]:
        self.training_resolution_calls.append((int(frame_index), int(step or 0)))
        return (320, 180)

    def frame_size(self, frame_index: int) -> tuple[int, int]:
        return (640, 360)

    def current_base_lr(self) -> float:
        return 0.005

    def get_frame_target_texture(self, frame_index: int, native_resolution: bool = True, encoder: object | None = None) -> str:
        self.target_calls.append((int(frame_index), bool(native_resolution)))
        return f"target_tex_{frame_index}_{native_resolution}"

    def training_background(self) -> np.ndarray:
        return np.asarray([0.25, 0.5, 0.75], dtype=np.float32)

    def training_background_seed(self, seed_index: int | None = None) -> int:
        self.background_seed_calls.append(None if seed_index is None else int(seed_index))
        return 1000 + int(seed_index or 0)

    def training_sample_vars(self, frame_index: int, step: int | None = None, sample_seed_step: int | None = None) -> dict[str, object]:
        self.sample_vars_calls.append((int(frame_index), int(step or 0), None if sample_seed_step is None else int(sample_seed_step)))
        return {
            "g_TrainingSubsample": {
                "enabled": np.uint32(1 if self.subsample_factor > 1 else 0),
                "factor": np.uint32(2),
                "nativeWidth": np.uint32(640),
                "nativeHeight": np.uint32(360),
                "frameIndex": np.uint32(frame_index),
                "stepIndex": np.uint32(0 if sample_seed_step is None else sample_seed_step),
            }
        }

    def apply_renderer_training_hparams(self, step: int | None = None, renderer: object | None = None) -> None:
        self.hparam_calls.append((None if step is None else int(step), renderer))
        if renderer is not None:
            renderer.sh_band = 2

    def sorting_dither(self, frame_index: int, step: int, camera: object) -> SimpleNamespace:
        self.sort_calls.append((int(frame_index), int(step), camera))
        return SimpleNamespace(position=np.asarray([1.0, 2.0, 3.0], dtype=np.float32), sigma=0.125, seed=555)

    def frame_metrics_snapshot(self) -> dict[str, np.ndarray]:
        return {
            "loss": np.asarray([0.25], dtype=np.float64),
            "mse": np.asarray([0.125], dtype=np.float64),
            "psnr": np.asarray([32.5], dtype=np.float64),
            "visited": np.asarray([True], dtype=bool),
        }


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
        "debug_mode": _control(0 if loss_debug else 1),
        "loss_debug_frame": _control(0),
        "loss_debug_view": _control(0),
        "loss_debug_abs_scale": _control(1.0),
        "ssim_c2": _control(9e-4),
        "images_subdir": _control(0),
        "training_steps_per_frame": _control(3),
        "train_downscale_factor": _control(1),
        "lr_schedule_enabled": _control(True),
        "lr_schedule_start_lr": _control(0.005),
        "lr_pos_mul": _control(1.0),
        "lr_pos_stage1_mul": _control(0.75),
        "lr_pos_stage2_mul": _control(0.2),
        "lr_pos_stage3_mul": _control(0.2),
        "lr_sh_mul": _control(0.05),
        "lr_sh_stage1_mul": _control(0.05),
        "lr_sh_stage2_mul": _control(0.05),
        "lr_sh_stage3_mul": _control(0.05),
        "depth_ratio_weight": _control(1.0),
        "sorting_order_dithering": _control(0.5),
        "sorting_order_dithering_stage1": _control(0.2),
        "sorting_order_dithering_stage2": _control(0.05),
        "sorting_order_dithering_stage3": _control(0.01),
        "position_random_step_noise_lr": _control(5e5),
        "sh_band": _control(0),
        "lr_schedule_stage1_lr": _control(0.002),
        "lr_schedule_stage2_lr": _control(0.001),
        "lr_schedule_end_lr": _control(1.5e-4),
        "lr_schedule_steps": _control(30000),
        "lr_schedule_stage1_step": _control(3000),
        "lr_schedule_stage2_step": _control(14000),
        "refinement_interval": _control(200),
        "refinement_growth_ratio": _control(0.05),
        "refinement_growth_start_step": _control(500),
        "refinement_alpha_cull_threshold": _control(1e-2),
        "refinement_min_contribution": _control(512),
        "refinement_min_contribution_decay": _control(0.995),
        "refinement_opacity_mul": _control(1.0),
        "refinement_clone_scale_mul": _control(1.0),
        "refinement_use_compact_split": _control(True),
        "refinement_solve_opacity": _control(True),
        "refinement_split_beta": _control(0.28),
        "refinement_momentum_weight_exponent": _control(1.5),
        "refinement_loss_weight": _control(0.25),
        "refinement_target_edge_weight": _control(0.75),
        "depth_ratio_stage1_weight": _control(0.05),
        "depth_ratio_stage2_weight": _control(0.01),
        "depth_ratio_stage3_weight": _control(0.001),
        "position_random_step_noise_stage1_lr": _control(466666.6666666667),
        "position_random_step_noise_stage2_lr": _control(416666.6666666667),
        "position_random_step_noise_stage3_lr": _control(0.0),
        "sh_band_stage1": _control(1),
        "sh_band_stage2": _control(1),
        "sh_band_stage3": _control(1),
        "max_gaussians": _control(1000000),
        "train_downscale_mode": _control(1),
        "train_subsample_factor": _control(0),
        "train_auto_start_downscale": _control(1),
        "train_downscale_max_iters": _control(30000),
    }
    texts = {key: _text() for key in ("fps", "images_subdir", "loss_debug_view", "loss_debug_frame", "loss_debug_psnr", "path", "scene_stats", "render_stats", "training", "training_time", "training_iters_avg", "training_loss", "training_mse", "training_density", "training_psnr", "training_instability", "training_resolution", "training_downscale", "training_schedule", "training_schedule_values", "training_refinement", "colmap_import_status", "colmap_import_current", "histogram_status", "error")}
    viewer = SimpleNamespace()
    viewer.device = SimpleNamespace()
    viewer.toolkit = SimpleNamespace(viewport_size=lambda: (640, 360))
    viewer.loss_debug_view_options = (("rendered", "Rendered"), ("target", "Target"), ("abs_diff", "Abs Diff"), ("dssim", "DSSIM"), ("rendered_edges", "Rendered Edges"), ("target_edges", "Target Edges"))
    viewer.ui = SimpleNamespace(controls=controls, texts=texts, _values={"show_histograms": False, "_histogram_payload": None, "_histogram_range_payload": None, "show_training_cameras": bool(loss_debug)}, _texts={key: value.text for key, value in texts.items()})
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
        training_frames=[SimpleNamespace(image_path=Path("frame.png"), width=640, height=360, fx=525.0, fy=520.0, cx=320.0, cy=180.0)],
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
        debug_target_texture=None,
        debug_dssim_features_kernel=None,
        debug_dssim_compose_kernel=None,
        debug_dssim_blur=None,
        debug_dssim_resolution=None,
        debug_dssim_moments=None,
        debug_dssim_blurred_moments=None,
        debug_target_sample_kernel=None,
        colmap_import_progress=None,
        cached_raster_grad_ranges=None,
        render_frame_index=0,
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
    monkeypatch.setattr(presenter, "_render_debug_view", lambda viewer_obj, encoder, width, height, render_frame_index: calls.append(f"debug:{render_frame_index}") or "debug_tex")
    monkeypatch.setattr(presenter, "_render_main_view", lambda viewer_obj, encoder: calls.append("main") or "main_tex")
    monkeypatch.setattr(presenter, "update_ui_text", lambda viewer_obj, dt: calls.append("ui"))

    presenter.render_frame(viewer, render_context)

    assert viewer.s.trainer.step_calls == 3
    assert viewer.s.viewport_texture == "debug_tex"
    assert viewer.s.render_frame_index == 1
    assert calls == ["apply", "debug:0", "ui"]


def test_render_frame_uses_main_branch_when_visual_loss_debug_disabled(monkeypatch):
    viewer = _viewer(loss_debug=False)
    render_context = SimpleNamespace(surface_texture=SimpleNamespace(width=640, height=360), command_encoder=_DummyEncoder())
    calls: list[str] = []

    monkeypatch.setattr(presenter.session, "apply_live_params", lambda viewer_obj: calls.append("apply"))
    monkeypatch.setattr(presenter.session, "ensure_training_runtime_resolution", lambda viewer_obj: calls.append("train_resize"))
    monkeypatch.setattr(presenter.session, "recreate_renderer", lambda viewer_obj, width, height: calls.append("resize"))
    monkeypatch.setattr(presenter.session, "update_debug_frame_slider_range", lambda viewer_obj: None)
    monkeypatch.setattr(presenter, "_render_debug_view", lambda viewer_obj, encoder, width, height, render_frame_index: calls.append("debug") or "debug_tex")
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
    monkeypatch.setattr(presenter, "_render_debug_view", lambda viewer_obj, encoder, width, height, render_frame_index: calls.append("debug") or "debug_tex")
    monkeypatch.setattr(presenter, "_render_main_view", lambda viewer_obj, encoder: calls.append("main") or "main_tex")
    monkeypatch.setattr(presenter, "update_ui_text", lambda viewer_obj, dt: calls.append("ui"))

    presenter.render_frame(viewer, render_context)

    assert viewer.s.trainer.step_calls == 3
    assert viewer.s.trainer.step_batch_calls == [3]
    assert viewer.s.last_training_batch_steps == 3
    assert calls == ["apply", "main", "ui"]


def test_render_frame_refreshes_histograms_when_requested(monkeypatch):
    viewer = _viewer(loss_debug=False)
    viewer.ui._values["_histograms_refresh_requested"] = True
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


def test_render_frame_skips_histogram_refresh_without_request(monkeypatch):
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

    assert calls == ["apply", "main", "ui"]


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


def test_update_ui_text_reports_training_schedule_and_refinement() -> None:
    viewer = _viewer(loss_debug=False)

    presenter.update_ui_text(viewer, 1.0 / 60.0)

    assert viewer.t("training_schedule").text == "LR Schedule: 5.00e-03@0 -> 2.00e-03@3,000 -> 1.00e-03@14,000 -> 1.50e-04@30,000 | current=5.00e-03"
    assert viewer.t("training_schedule_values").text == "Current Values: step=0 | Stage 0 | lr=5.00e-03 | pos=1.00x | shlr=0.05x | depth=1.00e+00 | cspace=0.5 | hi=0 | lo=0 | dither=0.5 | noise=5.00e+05 | sh=SH0"
    assert viewer.t("training_refinement").text == "Refinement: every 200 | growth=0.00% now | target=5.00% after 500 | alpha<1.00e-02 or min contrib<512 | decay=99.50%/pass | alpha mul=1.00x | clone scale=1.00x | max=1,000,000"
    assert viewer.t("loss_debug_psnr").text == "PSNR: 32.50 dB"
    assert viewer.ui._values["_training_view_overlay_segments"] == ()
    assert viewer.ui._values["_training_views_rows"] == (
        {
            "frame_index": 0,
            "image_name": "frame.png",
            "resolution": "640x360",
            "fx": 525.0,
            "fy": 520.0,
            "cx": 320.0,
            "cy": 180.0,
            "camera_min_dist": 0.1,
            "loss": 0.25,
            "psnr": 32.5,
            "visited": True,
            "is_last": True,
        },
    )


def test_update_ui_text_skips_resource_debug_snapshot_when_closed(monkeypatch) -> None:
    viewer = _viewer(loss_debug=False)
    calls: list[object] = []
    monkeypatch.setattr(presenter, "collect_resource_debug_snapshot", lambda viewer_obj, **_kwargs: calls.append(viewer_obj) or object())

    presenter.update_ui_text(viewer, 1.0 / 60.0)

    assert calls == []
    assert "_resource_debug_snapshot" not in viewer.ui._values


def test_update_ui_text_throttles_resource_debug_snapshot(monkeypatch) -> None:
    viewer = _viewer(loss_debug=False)
    viewer.ui._values.update({"show_resource_debug": True, "_resource_debug_snapshot": None, "_resource_debug_next_update": 0.0})
    snapshots = ["snapshot_a", "snapshot_b"]
    calls: list[bool] = []

    def collect(_viewer_obj, *, include_process_vram: bool = False):
        calls.append(bool(include_process_vram))
        return snapshots[min(len(calls) - 1, len(snapshots) - 1)]

    monkeypatch.setattr(presenter, "collect_resource_debug_snapshot", collect)

    presenter.update_ui_text(viewer, 1.0 / 60.0)
    presenter.update_ui_text(viewer, 1.0 / 60.0)
    viewer.ui._values["_resource_debug_refresh_requested"] = True
    viewer.ui._values["_resource_debug_process_vram_requested"] = True
    presenter.update_ui_text(viewer, 1.0 / 60.0)

    assert calls == [False, True]
    assert viewer.ui._values["_resource_debug_snapshot"] == "snapshot_b"
    assert viewer.ui._values["_resource_debug_refresh_requested"] is False
    assert viewer.ui._values["_resource_debug_process_vram_requested"] is False


def test_update_ui_text_previews_current_schedule_values_without_trainer() -> None:
    viewer = _viewer(loss_debug=False)
    viewer.s.trainer = None
    viewer.s.training_active = False

    presenter.update_ui_text(viewer, 1.0 / 60.0)

    assert viewer.t("training_schedule_values").text == "Current Values: step=0 | Stage 0 | lr=5.00e-03 | pos=1.00x | shlr=0.05x | depth=1.00e+00 | cspace=0.5 | hi=0 | lo=0 | dither=0.5 | noise=5.00e+05 | sh=SH0"


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

    monkeypatch.setattr(presenter.session, "reinitialize_training_scene", lambda viewer_obj: calls.append("init"))
    monkeypatch.setattr(presenter.session, "apply_live_params", lambda viewer_obj: calls.append("apply"))
    monkeypatch.setattr(presenter.session, "advance_colmap_import", lambda viewer_obj: calls.append("import"))
    monkeypatch.setattr(presenter, "_render_main_view", lambda viewer_obj, encoder: calls.append("main") or "main_tex")
    monkeypatch.setattr(presenter, "update_ui_text", lambda viewer_obj, dt: calls.append("ui"))

    presenter.render_frame(viewer, render_context)

    assert viewer.s.pending_training_reinitialize is False
    assert viewer.s.trainer.step_calls == 0
    assert calls == ["init", "apply", "import", "main", "ui"]


def test_render_frame_skips_training_batch_when_runtime_resize_is_applied(monkeypatch):
    viewer = _viewer(loss_debug=False)
    render_context = SimpleNamespace(surface_texture=SimpleNamespace(width=640, height=360), command_encoder=_DummyEncoder())
    calls: list[str] = []

    def _apply_live_params(viewer_obj) -> None:
        viewer_obj.s.pending_training_runtime_resize = True
        calls.append("apply")

    monkeypatch.setattr(presenter.session, "apply_live_params", _apply_live_params)
    monkeypatch.setattr(presenter.session, "advance_colmap_import", lambda viewer_obj: calls.append("import"))
    monkeypatch.setattr(presenter.session, "ensure_training_runtime_resolution", lambda viewer_obj: calls.append("train_resize") or True)
    monkeypatch.setattr(presenter, "_render_main_view", lambda viewer_obj, encoder: calls.append("main") or "main_tex")
    monkeypatch.setattr(presenter, "update_ui_text", lambda viewer_obj, dt: calls.append("ui"))

    presenter.render_frame(viewer, render_context)

    assert viewer.s.trainer.step_calls == 0
    assert viewer.s.trainer.step_batch_calls == []
    assert viewer.s.last_training_batch_steps == 0
    assert calls == ["apply", "import", "train_resize", "main", "ui"]


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
    assert viewer.t("loss_debug_psnr").text == "PSNR: 32.50 dB"
    assert viewer.t("training_resolution").text == "Train Res: 640x360 (N=1)"


def test_update_ui_text_populates_colmap_import_progress_fields() -> None:
    viewer = _viewer(loss_debug=False)
    viewer.s.colmap_import_progress = ColmapImportProgress(
        dataset_root=Path("dataset"),
        colmap_root=Path("dataset"),
        database_path=None,
        images_root=Path("dataset/images"),
        init_mode="pointcloud",
        compress_dataset_using_bc7=False,
        custom_ply_path=None,
        image_downscale_mode="original",
        image_downscale_max_size=1600,
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
    viewer.s.trainer.state.step = 17
    overlay_renderer = _DummyRenderer()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(presenter.session, "ensure_renderer", lambda viewer_obj, attr, width, height, allow_debug_overlays: calls.append(("ensure", allow_debug_overlays)) or overlay_renderer)
    monkeypatch.setattr(presenter.session, "sync_scene_from_training_renderer", lambda viewer_obj, dst_renderer, target: calls.append(("sync", target)))

    source_tex, stats, width, height, sample_vars = presenter._render_debug_source(viewer, _DummyEncoder(), 0, 42)

    assert source_tex == "training_preview_tex"
    assert width == 320
    assert height == 180
    assert stats["generated_entries"] == 1
    assert calls == [("ensure", True), ("sync", "debug")]
    assert viewer.s.trainer.training_resolution_calls == [(0, 17)]
    assert viewer.s.trainer.sample_vars_calls == [(0, 17, 42)]
    assert viewer.s.trainer.background_seed_calls == [42]
    assert viewer.s.trainer.hparam_calls == [(17, overlay_renderer)]
    assert viewer.s.trainer.sort_calls == [(0, 17, (0, 320, 180))]
    assert sample_vars["g_TrainingSubsample"]["stepIndex"] == np.uint32(42)
    render_call = overlay_renderer.training_forward_calls[0]
    assert render_call["camera"] == (0, 320, 180)
    assert render_call["training_native_camera"] == (0, 640, 360)
    assert render_call["training_background_seed"] == 1042
    np.testing.assert_allclose(render_call["sort_camera_position"], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    assert render_call["sort_camera_dither_sigma"] == 0.125
    assert render_call["sort_camera_dither_seed"] == 555


def test_render_debug_source_uses_overlay_renderer_for_abs_diff(monkeypatch) -> None:
    viewer = _viewer(loss_debug=True)
    viewer.c("loss_debug_view").value = 2
    overlay_renderer = _DummyRenderer()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(presenter.session, "ensure_renderer", lambda viewer_obj, attr, width, height, allow_debug_overlays: calls.append(("ensure", allow_debug_overlays)) or overlay_renderer)
    monkeypatch.setattr(presenter.session, "sync_scene_from_training_renderer", lambda viewer_obj, dst_renderer, target: calls.append(("sync", target)))

    source_tex, stats, width, height, _ = presenter._render_debug_source(viewer, _DummyEncoder(), 0, 7)

    assert source_tex == "training_preview_tex"
    assert width == 320
    assert height == 180
    assert stats["written_entries"] == 2
    assert calls == [("ensure", True), ("sync", "debug")]


def test_render_debug_target_uses_downscaled_target_without_subsampling() -> None:
    viewer = _viewer(loss_debug=True)
    encoder = _DummyEncoder()
    sample_vars = {"g_TrainingSubsample": {"enabled": np.uint32(0), "factor": np.uint32(1)}}

    target = presenter._render_debug_target(viewer, encoder, 0, 320, 180, 5, sample_vars)

    assert target == "target_tex_0_False"
    assert viewer.s.trainer.target_calls == [(0, False)]


def test_render_debug_target_samples_native_target_with_render_frame_seed(monkeypatch) -> None:
    viewer = _viewer(loss_debug=True)
    viewer.s.trainer.subsample_factor = 2
    viewer.s.debug_target_sample_kernel = _CaptureKernel()
    output = SimpleNamespace(width=320, height=180)
    sample_vars = viewer.s.trainer.training_sample_vars(0, 9, sample_seed_step=77)

    monkeypatch.setattr(presenter, "_ensure_texture", lambda viewer_obj, attr, width, height: output)

    target = presenter._render_debug_target(viewer, _DummyEncoder(), 0, 320, 180, 9, sample_vars)

    assert target is output
    assert viewer.s.trainer.target_calls == [(0, True)]
    vars = viewer.s.debug_target_sample_kernel.calls[0]["vars"]
    assert vars["g_SourceTarget"] == "target_tex_0_True"
    assert vars["g_DownscaledTarget"] is output
    assert vars["g_TargetWidth"] == 320
    assert vars["g_TargetHeight"] == 180
    assert vars["g_TrainingSubsample"]["stepIndex"] == np.uint32(77)


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
    vars = viewer.s.debug_abs_diff_kernel.calls[0]["vars"]
    assert vars["g_DebugDiffScale"] == 3.5
    assert vars["g_DebugRenderedIsLinear"] == 1


def test_dispatch_debug_edge_filter_uses_source_texture(monkeypatch) -> None:
    viewer = _viewer(loss_debug=True)
    viewer.s.debug_edge_kernel = _CaptureKernel()
    encoder = _DummyEncoder()
    output = SimpleNamespace(width=640, height=360)

    monkeypatch.setattr(presenter, "_ensure_texture", lambda viewer_obj, attr, width, height: output)

    result = presenter._dispatch_debug_edge_filter(viewer, encoder, "source_tex", 640, 360)

    assert result is output
    assert len(viewer.s.debug_edge_kernel.calls) == 1
    vars = viewer.s.debug_edge_kernel.calls[0]["vars"]
    assert vars["g_DebugRendered"] == "source_tex"
    assert vars["g_DebugWidth"] == 640
    assert vars["g_DebugHeight"] == 360
    assert vars["g_DebugSourceIsLinear"] == 0


def test_dispatch_debug_dssim_runs_features_blur_and_compose(monkeypatch) -> None:
    viewer = _viewer(loss_debug=True)
    encoder = _DummyEncoder()
    blur_calls: list[tuple[object, object, int]] = []

    class _DummyBlur:
        def blur(self, encoder_obj, input_buffer, output_buffer, channel_count):
            blur_calls.append((input_buffer, output_buffer, channel_count))

    output = SimpleNamespace(width=640, height=360)
    monkeypatch.setattr(presenter, "_ensure_texture", lambda viewer_obj, attr, width, height: output)
    monkeypatch.setattr(presenter, "_ensure_debug_dssim_runtime", lambda viewer_obj, width, height: setattr(viewer_obj.s, "debug_dssim_blur", _DummyBlur()) or setattr(viewer_obj.s, "debug_dssim_moments", "moments") or setattr(viewer_obj.s, "debug_dssim_blurred_moments", "blurred"))
    viewer.s.debug_dssim_features_kernel = _CaptureKernel()
    viewer.s.debug_dssim_compose_kernel = _CaptureKernel()

    result = presenter._dispatch_debug_dssim(viewer, encoder, "rendered_tex", "target_tex", 640, 360)

    assert result is output
    assert blur_calls == [("moments", "blurred", 15)]
    assert viewer.s.debug_dssim_features_kernel.calls[0]["vars"]["g_DebugRendered"] == "rendered_tex"
    assert viewer.s.debug_dssim_features_kernel.calls[0]["vars"]["g_DebugTarget"] == "target_tex"
    assert viewer.s.debug_dssim_compose_kernel.calls[0]["vars"]["g_DebugTarget"] == "target_tex"
    assert viewer.s.debug_dssim_compose_kernel.calls[0]["vars"]["g_SSIMC2"] == 9e-4


def test_render_debug_view_routes_edge_modes(monkeypatch) -> None:
    viewer = _viewer(loss_debug=True)
    encoder = _DummyEncoder()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(presenter, "_render_debug_source", lambda viewer_obj, enc, frame_idx, render_frame_index: ("rendered_tex", {"generated_entries": 1}, 640, 360, {"g_TrainingSubsample": {"enabled": np.uint32(0)}}))
    monkeypatch.setattr(viewer.s.trainer, "get_frame_target_texture", lambda frame_idx, native_resolution=True, encoder=None: "target_tex")
    monkeypatch.setattr(presenter, "_dispatch_debug_abs_diff", lambda viewer_obj, enc, rendered_tex, target_tex, width, height, *, rendered_is_linear=True: calls.append(("abs_diff", rendered_tex, target_tex, width, height, rendered_is_linear)) or "abs_diff_tex")
    monkeypatch.setattr(presenter, "_dispatch_debug_dssim", lambda viewer_obj, enc, rendered_tex, target_tex, width, height: calls.append(("dssim", rendered_tex, target_tex, width, height)) or "dssim_tex")
    monkeypatch.setattr(presenter, "_dispatch_debug_edge_filter", lambda viewer_obj, enc, source_tex, width, height, *, source_is_linear=False: calls.append(("edge", source_tex, width, height, source_is_linear)) or f"edge_{source_tex}")
    monkeypatch.setattr(presenter, "_dispatch_training_debug_present", lambda viewer_obj, enc, source_tex, source_width, source_height, output_width, output_height, *, source_is_linear=False, apply_loss_colorspace=False: calls.append(("present", source_tex, source_width, source_height, output_width, output_height, source_is_linear, apply_loss_colorspace)) or "present_tex")

    viewer.c("loss_debug_view").value = 3
    assert presenter._render_debug_view(viewer, encoder, 800, 600, 123) == "present_tex"
    viewer.c("loss_debug_view").value = 4
    assert presenter._render_debug_view(viewer, encoder, 800, 600, 123) == "present_tex"
    viewer.c("loss_debug_view").value = 5
    assert presenter._render_debug_view(viewer, encoder, 800, 600, 123) == "present_tex"

    assert calls == [
        ("dssim", "rendered_tex", "target_tex", 640, 360),
        ("present", "dssim_tex", 640, 360, 800, 600, False, False),
        ("edge", "rendered_tex", 640, 360, True),
        ("present", "edge_rendered_tex", 640, 360, 800, 600, False, False),
        ("edge", "target_tex", 640, 360, False),
        ("present", "edge_target_tex", 640, 360, 800, 600, False, False),
    ]


def test_render_debug_view_presents_rendered_as_linear_and_target_as_display(monkeypatch) -> None:
    viewer = _viewer(loss_debug=True)
    encoder = _DummyEncoder()
    calls: list[tuple[str, object, bool, bool]] = []

    monkeypatch.setattr(presenter, "_render_debug_source", lambda viewer_obj, enc, frame_idx, render_frame_index: ("rendered_tex", {"generated_entries": 1}, 640, 360, {"g_TrainingSubsample": {"enabled": np.uint32(0)}}))
    monkeypatch.setattr(viewer.s.trainer, "get_frame_target_texture", lambda frame_idx, native_resolution=True, encoder=None: "target_tex")
    monkeypatch.setattr(presenter, "_dispatch_training_debug_present", lambda viewer_obj, enc, source_tex, source_width, source_height, output_width, output_height, *, source_is_linear=False, apply_loss_colorspace=False: calls.append(("present", source_tex, source_is_linear, apply_loss_colorspace)) or "present_tex")

    viewer.c("loss_debug_view").value = 0
    assert presenter._render_debug_view(viewer, encoder, 800, 600, 123) == "present_tex"
    viewer.c("loss_debug_view").value = 1
    assert presenter._render_debug_view(viewer, encoder, 800, 600, 123) == "present_tex"

    assert calls == [
        ("present", "rendered_tex", True, True),
        ("present", "target_tex", False, True),
    ]


def test_render_debug_view_skips_target_work_for_rendered_mode(monkeypatch) -> None:
    viewer = _viewer(loss_debug=True)
    encoder = _DummyEncoder()
    calls: list[tuple[str, object, bool, bool]] = []

    monkeypatch.setattr(presenter, "_render_debug_source", lambda viewer_obj, enc, frame_idx, render_frame_index: ("rendered_tex", {"generated_entries": 1}, 640, 360, {"g_TrainingSubsample": {"enabled": np.uint32(0)}}))
    monkeypatch.setattr(presenter, "_render_debug_target", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rendered view should not render the target path")))
    monkeypatch.setattr(presenter, "_dispatch_training_debug_present", lambda viewer_obj, enc, source_tex, source_width, source_height, output_width, output_height, *, source_is_linear=False, apply_loss_colorspace=False: calls.append(("present", source_tex, source_is_linear, apply_loss_colorspace)) or "present_tex")

    viewer.c("loss_debug_view").value = 0
    assert presenter._render_debug_view(viewer, encoder, 800, 600, 123) == "present_tex"

    assert calls == [("present", "rendered_tex", True, True)]


def test_dispatch_training_debug_present_passes_colorspace_mod(monkeypatch) -> None:
    viewer = _viewer(loss_debug=True)
    viewer.s.debug_letterbox_kernel = _CaptureKernel()
    viewer.s.trainer.state.step = 17
    encoder = _DummyEncoder()
    output = SimpleNamespace(width=640, height=360)

    monkeypatch.setattr(presenter, "_ensure_texture", lambda viewer_obj, attr, width, height: output)

    result = presenter._dispatch_training_debug_present(viewer, encoder, "source_tex", 320, 180, 640, 360, source_is_linear=True, apply_loss_colorspace=True)

    assert result is output
    vars = viewer.s.debug_letterbox_kernel.calls[0]["vars"]
    assert vars["g_LetterboxSource"] == "source_tex"
    assert vars["g_LetterboxSourceIsLinear"] == 1
    assert vars["g_LetterboxApplyLossColorspace"] == 1
    assert np.isclose(vars["g_ColorspaceMod"], presenter.resolve_colorspace_mod(viewer.s.trainer.training, 17))


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
    assert vars["g_LetterboxSourceIsLinear"] == 0

    presenter._dispatch_viewport_present(viewer, encoder, "linear_source_tex", 320, 180, 640, 360, source_is_linear=True)
    assert viewer.s.debug_letterbox_kernel.calls[1]["vars"]["g_LetterboxSourceIsLinear"] == 1


def test_dispatch_viewport_present_zero_stays_zero_and_output_is_finite(device) -> None:
    viewer = SimpleNamespace(
        device=device,
        s=SimpleNamespace(debug_present_texture=None, debug_letterbox_kernel=None),
    )
    viewer_session.create_debug_shaders(viewer)
    source = device.create_texture(
        format=spy.Format.rgba32_float,
        width=2,
        height=2,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
    )
    source.copy_from_numpy(
        np.array(
            [
                [[0.0, 0.0, 0.0, 1.0], [np.nan, 0.0, -0.5, 1.0]],
                [[0.25, 0.5, 1.0, 1.0], [4.0, 2.0, 0.125, 1.0]],
            ],
            dtype=np.float32,
        )
    )

    encoder = device.create_command_encoder()
    output = presenter._dispatch_viewport_present(viewer, encoder, source, 2, 2, 2, 2)
    device.submit_command_buffer(encoder.finish())
    device.wait()

    image = np.asarray(output.to_numpy(), dtype=np.float32)
    assert np.all(np.isfinite(image[..., :3]))
    np.testing.assert_allclose(image[0, 0, :3], np.zeros((3,), dtype=np.float32), rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(image[0, 1, :3], np.zeros((3,), dtype=np.float32), rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(image[..., 3], np.ones((2, 2), dtype=np.float32), rtol=0.0, atol=1e-7)


def test_dispatch_viewport_present_keeps_existing_display_transform(device) -> None:
    viewer = SimpleNamespace(
        device=device,
        s=SimpleNamespace(debug_present_texture=None, debug_letterbox_kernel=None),
    )
    viewer_session.create_debug_shaders(viewer)
    source = device.create_texture(
        format=spy.Format.rgba32_float,
        width=1,
        height=1,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
    )
    source.copy_from_numpy(np.array([[[0.25, 0.5, 1.0, 1.0]]], dtype=np.float32))

    encoder = device.create_command_encoder()
    output = presenter._dispatch_viewport_present(viewer, encoder, source, 1, 1, 1, 1, source_is_linear=False)
    device.submit_command_buffer(encoder.finish())
    device.wait()

    image = np.asarray(output.to_numpy(), dtype=np.float32)
    expected = np.array([0.5370987, 0.735357, 1.0], dtype=np.float32)
    np.testing.assert_allclose(image[0, 0, :3], expected, rtol=0.0, atol=1e-5)
