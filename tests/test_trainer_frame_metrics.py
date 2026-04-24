from __future__ import annotations

from pathlib import Path

import numpy as np

from src.scene import ColmapFrame
from src.training import GaussianTrainer, TrainingHyperParams, TrainingState
from src.training.gaussian_trainer import _FrameMetricBookkeeper


def test_frame_metric_bookkeeper_averages_unique_frames_only() -> None:
    tracker = _FrameMetricBookkeeper.create(3)

    tracker.update(2, loss=9.0, mse=0.09, psnr=10.0)
    tracker.update(0, loss=3.0, mse=0.03, psnr=20.0)

    assert np.isclose(tracker.mean("loss"), 6.0)
    assert np.isclose(tracker.mean("mse"), 0.06)
    assert np.isclose(tracker.mean("psnr"), 15.0)


def test_frame_metric_bookkeeper_replaces_existing_frame_value() -> None:
    tracker = _FrameMetricBookkeeper.create(2)

    tracker.update(0, loss=1.0, mse=0.1, psnr=10.0)
    tracker.update(1, loss=5.0, mse=0.5, psnr=30.0)
    tracker.update(0, loss=9.0, mse=0.9, psnr=50.0)

    assert np.isclose(tracker.mean("loss"), 7.0)
    assert np.isclose(tracker.mean("mse"), 0.7)
    assert np.isclose(tracker.mean("psnr"), 40.0)


def test_frame_metric_bookkeeper_reset_clears_stored_frame_values() -> None:
    tracker = _FrameMetricBookkeeper.create(2)
    tracker.update(0, loss=1.0, mse=0.1, psnr=10.0)
    tracker.reset()
    tracker.update(1, loss=5.0, mse=0.5, psnr=30.0)

    assert np.isclose(tracker.mean("loss"), 5.0)
    assert np.isclose(tracker.mean("mse"), 0.5)
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

    assert trainer.max_training_resolution(0) == (1992, 1319)
    assert trainer.training_resolutions_vary(0)
