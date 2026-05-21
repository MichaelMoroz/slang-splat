from __future__ import annotations

from pathlib import Path

import numpy as np

from src.scene import ColmapFrame
from src.training import GaussianTrainer, TrainingHyperParams, TrainingState
from src.training.gaussian_trainer import _FrameMetricBookkeeper


def test_frame_metric_bookkeeper_averages_unique_frames_only() -> None:
    tracker = _FrameMetricBookkeeper.create(3)

    tracker.update(2, loss=9.0, mse=0.09, ssim=0.1, psnr=10.0)
    tracker.update(0, loss=3.0, mse=0.03, ssim=0.3, psnr=20.0)

    assert np.isclose(tracker.mean("loss"), 6.0)
    assert np.isclose(tracker.mean("mse"), 0.06)
    assert np.isclose(tracker.mean("ssim"), 0.2)
    assert np.isclose(tracker.mean("psnr"), 15.0)


def test_frame_metric_bookkeeper_replaces_existing_frame_value() -> None:
    tracker = _FrameMetricBookkeeper.create(2)

    tracker.update(0, loss=1.0, mse=0.1, ssim=0.1, psnr=10.0)
    tracker.update(1, loss=5.0, mse=0.5, ssim=0.5, psnr=30.0)
    tracker.update(0, loss=9.0, mse=0.9, ssim=0.9, psnr=50.0)

    assert np.isclose(tracker.mean("loss"), 7.0)
    assert np.isclose(tracker.mean("mse"), 0.7)
    assert np.isclose(tracker.mean("ssim"), 0.7)
    assert np.isclose(tracker.mean("psnr"), 40.0)


def test_frame_metric_bookkeeper_reset_clears_stored_frame_values() -> None:
    tracker = _FrameMetricBookkeeper.create(2)
    tracker.update(0, loss=1.0, mse=0.1, ssim=0.1, psnr=10.0)
    tracker.reset()
    tracker.update(1, loss=5.0, mse=0.5, ssim=0.5, psnr=30.0)

    assert np.isclose(tracker.mean("loss"), 5.0)
    assert np.isclose(tracker.mean("mse"), 0.5)
    assert np.isclose(tracker.mean("ssim"), 0.5)
    assert np.isclose(tracker.mean("psnr"), 30.0)


def _frame(width: int, height: int) -> ColmapFrame:
    return ColmapFrame(
        image_id=int(width + height),
        image_path=Path("synthetic.png"),
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        t_xyz=np.zeros((3,), dtype=np.float32),
        fx=float(width),
        fy=float(width),
        cx=float(width) * 0.5,
        cy=float(height) * 0.5,
        width=int(width),
        height=int(height),
    )


def test_trainer_max_training_resolution_uses_all_frames() -> None:
    trainer = object.__new__(GaussianTrainer)
    trainer.frames = [_frame(7692, 5098), _frame(7965, 5275), _frame(7533, 5037)]
    trainer.training = TrainingHyperParams(train_downscale_mode=4, train_subsample_factor=1)
    trainer.state = TrainingState()
    trainer._training_resolution_summary_cache_key = None
    trainer._training_resolution_summary_cache = None

    assert trainer.max_training_resolution(0) == (1992, 1319)
    assert trainer.training_resolutions_vary(0)


def test_trainer_resolution_summary_cache_reuses_all_frame_scan() -> None:
    trainer = object.__new__(GaussianTrainer)
    trainer.frames = [_frame(2048, 1024), _frame(1024, 768), _frame(2048, 1024)]
    trainer.training = TrainingHyperParams(train_downscale_mode=2, train_subsample_factor=1)
    trainer.state = TrainingState()
    trainer._training_resolution_summary_cache_key = None
    trainer._training_resolution_summary_cache = None
    calls: list[tuple[int, int]] = []

    trainer.effective_train_downscale_factor = lambda step=None: 2

    def _training_resolution(frame_index: int = 0, step: int | None = None) -> tuple[int, int]:
        calls.append((int(frame_index), int(step or 0)))
        return ((1024, 512), (512, 384), (1024, 512))[int(frame_index)]

    trainer.training_resolution = _training_resolution

    assert trainer.max_training_resolution(0) == (1024, 512)
    assert trainer.training_resolutions_vary(0) is True
    assert calls == [(0, 0), (1, 0), (2, 0)]


def test_trainer_restores_uniform_training_resolution_after_viewport_resize() -> None:
    trainer = object.__new__(GaussianTrainer)
    calls: list[object] = []

    class _Renderer:
        def __init__(self) -> None:
            self.width = 640
            self.height = 360

        def ensure_render_capacity(self, width: int, height: int) -> bool:
            calls.append(("capacity", int(width), int(height)))
            return False

        def set_render_resolution(self, width: int, height: int) -> bool:
            self.width = int(width)
            self.height = int(height)
            calls.append(("resolution", self.width, self.height))
            return True

    trainer.renderer = _Renderer()
    trainer._dynamic_frame_resolution = False
    trainer.training_resolution = lambda frame_index=0, step=None: (320, 180)
    trainer._max_training_resolution = lambda step=None: (320, 180)
    trainer._invalidate_downscaled_target = lambda: calls.append("invalidate")
    trainer._refinement_camera_signature = "stale"

    trainer._ensure_frame_render_resolution(0, 0)

    assert trainer.renderer.width == 320
    assert trainer.renderer.height == 180
    assert calls == [("capacity", 320, 180), ("resolution", 320, 180), "invalidate"]
    assert trainer._refinement_camera_signature is None
