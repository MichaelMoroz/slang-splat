from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import time

from src.viewer import presenter


class _DummyEncoder:
    def __init__(self) -> None:
        self.blit_calls: list[tuple[object, object]] = []
        self.clear_calls: list[tuple[object, list[float]]] = []

    def blit(self, dst: object, src: object) -> None:
        self.blit_calls.append((dst, src))

    def clear_texture_float(self, texture: object, clear_value: list[float]) -> None:
        self.clear_calls.append((texture, clear_value))


class _DummyRenderer:
    def __init__(self, width: int = 640, height: int = 360) -> None:
        self.width = width
        self.height = height


class _DummyTrainer:
    def __init__(self) -> None:
        self.state = SimpleNamespace(step=0, last_loss=0.0, avg_loss=0.0, last_mse=0.0, last_frame_index=0, last_instability="")
        self.scene = SimpleNamespace(count=4)
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1


def _control(value: object) -> SimpleNamespace:
    return SimpleNamespace(value=value)


def _text() -> SimpleNamespace:
    return SimpleNamespace(text="")


def _viewer(loss_debug: bool) -> SimpleNamespace:
    trainer = _DummyTrainer()
    controls = {
        "loss_debug": _control(loss_debug),
        "loss_debug_frame": _control(0),
        "loss_debug_view": _control(2),
        "images_subdir": _control(0),
    }
    texts = {key: _text() for key in ("fps", "images_subdir", "loss_debug_view", "loss_debug_frame", "path", "scene_stats", "render_stats", "training", "training_loss", "training_mse", "training_instability", "error")}
    viewer = SimpleNamespace()
    viewer.device = SimpleNamespace()
    viewer.loss_debug_view_options = (("rendered", "Rendered"), ("target", "Target"), ("abs_diff", "Abs Diff"))
    viewer.image_subdir_options = ("images_8",)
    viewer.ui = SimpleNamespace(controls=controls, texts=texts)
    viewer.c = lambda key: viewer.ui.controls[key]
    viewer.t = lambda key: viewer.ui.texts[key]
    viewer.camera = lambda: "camera"
    viewer.update_camera = lambda dt: None
    viewer.s = SimpleNamespace(
        fps_smooth=60.0,
        last_time=time.perf_counter(),
        renderer=_DummyRenderer(),
        scene=SimpleNamespace(count=4),
        stats={},
        scene_path=None,
        colmap_root=Path("dataset"),
        training_frames=[SimpleNamespace(image_path=Path("frame.png"))],
        trainer=trainer,
        training_active=True,
        training_renderer=object(),
        background=(0.0, 0.0, 0.0),
        last_render_exception="",
        last_error="",
    )
    return viewer


def test_render_frame_uses_debug_branch_when_visual_loss_debug_enabled(monkeypatch):
    viewer = _viewer(loss_debug=True)
    render_context = SimpleNamespace(surface_texture=SimpleNamespace(width=640, height=360), command_encoder=_DummyEncoder())
    calls: list[str] = []

    monkeypatch.setattr(presenter.session, "apply_live_params", lambda viewer_obj: calls.append("apply"))
    monkeypatch.setattr(presenter.session, "recreate_renderer", lambda viewer_obj, width, height: calls.append("resize"))
    monkeypatch.setattr(presenter.session, "update_debug_frame_slider_range", lambda viewer_obj: None)
    monkeypatch.setattr(presenter, "_render_debug_view", lambda viewer_obj, image, encoder, width, height: calls.append("debug"))
    monkeypatch.setattr(presenter, "_render_main_view", lambda viewer_obj, image, encoder: calls.append("main"))
    monkeypatch.setattr(presenter, "update_ui_text", lambda viewer_obj, dt: calls.append("ui"))

    presenter.render_frame(viewer, render_context)

    assert viewer.s.trainer.step_calls == 1
    assert calls == ["apply", "debug", "ui"]


def test_render_frame_uses_main_branch_when_visual_loss_debug_disabled(monkeypatch):
    viewer = _viewer(loss_debug=False)
    render_context = SimpleNamespace(surface_texture=SimpleNamespace(width=640, height=360), command_encoder=_DummyEncoder())
    calls: list[str] = []

    monkeypatch.setattr(presenter.session, "apply_live_params", lambda viewer_obj: calls.append("apply"))
    monkeypatch.setattr(presenter.session, "recreate_renderer", lambda viewer_obj, width, height: calls.append("resize"))
    monkeypatch.setattr(presenter.session, "update_debug_frame_slider_range", lambda viewer_obj: None)
    monkeypatch.setattr(presenter, "_render_debug_view", lambda viewer_obj, image, encoder, width, height: calls.append("debug"))
    monkeypatch.setattr(presenter, "_render_main_view", lambda viewer_obj, image, encoder: calls.append("main"))
    monkeypatch.setattr(presenter, "update_ui_text", lambda viewer_obj, dt: calls.append("ui"))

    presenter.render_frame(viewer, render_context)

    assert viewer.s.trainer.step_calls == 1
    assert calls == ["apply", "main", "ui"]
