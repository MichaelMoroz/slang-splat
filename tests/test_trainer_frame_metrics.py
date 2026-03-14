from __future__ import annotations

import numpy as np

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
