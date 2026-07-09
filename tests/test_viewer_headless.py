from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.app.training_controls import TRAINING_BUILD_ARG_UI_KEYS
from src.viewer import app, headless, presenter_state, session
from src.viewer.config import apply_config_overlay


def test_apply_config_overlay_flattens_headless_sections() -> None:
    values: dict[str, object] = {}
    apply_config_overlay(
        values,
        {
            "training_build_args": {"base_lr": 0.123},
            "renderer": {
                "sort_splats_by": "z_depth",
                "debug_mode": "grad_norm",
                "debug_splat_age_range": [2.0, 9.0],
                "cached_raster_grad_atomic_mode": "fixed",
            },
            "viewer": {
                "controls": {"move_speed": 4.5},
                "ui": {"viewport_sh_band": 2},
                "run": {
                    "colmap_root_path": "dataset/garden",
                    "colmap_selected_camera_ids": [7, 11],
                    "train_iters": 12,
                    "output_ply": None,
                },
            },
        },
    )

    assert values[TRAINING_BUILD_ARG_UI_KEYS["base_lr"]] == pytest.approx(0.123)
    assert values["sort_splats_by"] == 1
    assert values["debug_mode"] > 0
    assert values["debug_splat_age_min"] == pytest.approx(2.0)
    assert values["debug_splat_age_max"] == pytest.approx(9.0)
    assert values["cached_raster_grad_atomic_mode"] == 1
    assert values["move_speed"] == pytest.approx(4.5)
    assert values["_viewport_sh_band"] == 2
    assert values["colmap_root_path"] == "dataset/garden"
    assert values["colmap_selected_camera_ids"] == (7, 11)
    assert values["train_iters"] == 12
    assert values["output_ply"] == ""


def test_training_batch_clamps_to_max_steps() -> None:
    calls: list[int] = []

    class _Trainer:
        state = SimpleNamespace(step=0)

        def effective_train_render_factor(self) -> int:
            return 1

        def step_batch(self, steps: int) -> int:
            calls.append(int(steps))
            return int(steps)

    viewer = SimpleNamespace(
        c=lambda key: SimpleNamespace(value=8 if key == "training_steps_per_frame" else 0),
        s=SimpleNamespace(training_active=True, trainer=_Trainer(), last_time=1.0, last_interaction_time=0.0, training_runtime_factor_changed=False, last_training_batch_steps=0),
    )

    assert presenter_state._run_training_batch(viewer, max_steps=3) == 3
    assert calls == [3]
    assert viewer.s.last_training_batch_steps == 3


def test_photometric_batch_clamps_to_max_steps() -> None:
    calls: list[int] = []
    trainer = SimpleNamespace(train_step=lambda: calls.append(1))
    viewer = SimpleNamespace(
        ui=SimpleNamespace(_values={"photometric_steps_per_frame": 8}),
        s=SimpleNamespace(photometric_active=True, photometric_trainer=trainer),
    )

    assert presenter_state._run_photometric_batch(viewer, max_steps=2) == 2
    assert len(calls) == 2


def test_headless_training_stage_clamps_to_next_scheduled_step(monkeypatch) -> None:
    batches: list[int] = []
    post_steps: list[int] = []
    trainer = SimpleNamespace(state=SimpleNamespace(step=0))
    viewer = SimpleNamespace(
        c=lambda key: SimpleNamespace(value=8 if key == "training_steps_per_frame" else 0),
        s=SimpleNamespace(trainer=trainer, last_time=1.0, last_interaction_time=0.0),
    )
    run = {
        "train_iters": 10,
        "schedule": [
            {"at": 3, "do": "render"},
            {"every": 4, "do": "capture_buffers"},
        ],
    }

    monkeypatch.setattr(session, "set_training_active", lambda viewer_obj, active: None)
    monkeypatch.setattr(session, "apply_live_params", lambda viewer_obj: None)
    monkeypatch.setattr(session, "ensure_training_runtime_resolution", lambda viewer_obj: False)

    def _batch(viewer_obj, max_steps=None):
        step_count = int(max_steps)
        batches.append(step_count)
        viewer_obj.s.trainer.state.step += step_count
        return step_count

    monkeypatch.setattr(presenter_state, "_run_training_batch", _batch)
    monkeypatch.setattr(headless, "_run_post_step_events", lambda viewer_obj, run_cfg, events, step: post_steps.append(int(step)) if events else None)
    monkeypatch.setattr(headless, "_emit_stats", lambda *args, **kwargs: None)

    headless._run_training_stage(viewer, run)

    assert batches == [3, 1, 4, 2]
    assert post_steps == [3, 4, 8]
    assert trainer.state.step == 10


def test_plain_snapshot_crops_renderer_capacity_texture(monkeypatch, tmp_path: Path) -> None:
    class _Texture:
        def to_numpy(self):
            pixels = np.zeros((720, 1280, 4), dtype=np.float32)
            pixels[359, 639, 0] = 1.0
            pixels[360, 640, 0] = 0.5
            return pixels

    class _Renderer:
        width = 1280
        height = 720

        def set_render_resolution(self, width: int, height: int) -> None:
            self.width = int(width)
            self.height = int(height)

        def render_to_texture(self, camera, *, background=None):
            return _Texture(), None

    renderer = _Renderer()
    saved: list[np.ndarray] = []
    viewer = SimpleNamespace(
        camera=lambda: object(),
        s=SimpleNamespace(training_renderer=None, renderer=renderer, trainer=None, background=None),
    )

    monkeypatch.setattr(headless, "save_snapshot", lambda _path, rgba: saved.append(np.asarray(rgba).copy()))

    headless._render_plain_snapshot(viewer, tmp_path / "render.png", 640, 360)

    assert saved[0].shape == (360, 640, 4)
    assert saved[0][-1, -1, 0] == pytest.approx(1.0)


def test_debug_snapshot_crops_texture_to_requested_size(monkeypatch, tmp_path: Path) -> None:
    class _Texture:
        def to_numpy(self):
            pixels = np.zeros((480, 640, 4), dtype=np.float32)
            pixels[99, 199, 1] = 1.0
            pixels[100, 200, 1] = 0.5
            return pixels

    class _Encoder:
        def finish(self):
            return object()

    class _Device:
        def create_command_encoder(self):
            return _Encoder()

        def submit_command_buffer(self, _command_buffer) -> None:
            pass

    saved: list[np.ndarray] = []
    viewer = SimpleNamespace(
        device=_Device(),
        ui=SimpleNamespace(_values={}),
        s=SimpleNamespace(render_frame_index=0),
    )

    def _render_debug_view(viewer_obj, _encoder, output_width: int, output_height: int, _frame_index: int):
        assert viewer_obj is viewer
        assert (output_width, output_height) == (200, 100)
        return _Texture()

    monkeypatch.setattr(headless.presenter, "_render_debug_view", _render_debug_view)
    monkeypatch.setattr(headless, "save_snapshot", lambda _path, rgba: saved.append(np.asarray(rgba).copy()))

    headless._render_debug_snapshot(viewer, tmp_path / "debug.png", 200, 100, "rendered")

    assert saved[0].shape == (100, 200, 4)
    assert saved[0][-1, -1, 1] == pytest.approx(1.0)


def test_dataset_metrics_output_path_accepts_file_and_directory(tmp_path: Path) -> None:
    exact = session._dataset_metrics_output_path(Path("garden"), tmp_path / "metrics.txt")
    nested = session._dataset_metrics_output_path(Path("garden"), tmp_path / "metrics_dir")

    assert exact == tmp_path / "metrics.txt"
    assert nested.parent == tmp_path / "metrics_dir"
    assert nested.name.startswith("garden_")
    assert nested.suffix == ".txt"


def test_app_main_dispatches_headless_config(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[Path, str | None]] = []
    cfg = tmp_path / "run.json"
    cfg.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(headless, "run_headless_from_config", lambda path, graphics_api=None: calls.append((Path(path), graphics_api)) or 17)

    assert app.main(["--headless", "--config", str(cfg), "--graphics-api", "dx12"]) == 17
    assert calls == [(cfg, "dx12")]
