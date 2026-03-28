from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.colmap import CameraInfo, _decode_intrinsics
from train.dataset import _camera_tensor


def test_decode_intrinsics_preserves_radial_distortion_and_principal_point() -> None:
    fx, fy, cx, cy, k1, k2 = _decode_intrinsics(4, np.array([1000.0, 900.0, 320.0, 240.0, 0.12, -0.03, 0.0, 0.0], dtype=np.float64))

    assert fx == 1000.0
    assert fy == 900.0
    assert cx == 320.0
    assert cy == 240.0
    assert k1 == 0.12
    assert k2 == -0.03


def test_camera_tensor_scales_intrinsics_and_keeps_distortion() -> None:
    info = CameraInfo(
        uid=1,
        rotation_w2c=np.eye(3, dtype=np.float32),
        translation_w2c=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        fx=500.0,
        fy=450.0,
        cx=200.0,
        cy=150.0,
        k1=0.07,
        k2=-0.02,
        fov_x=0.0,
        fov_y=0.0,
        width=400,
        height=300,
        image_path=Path("frame.png"),
        image_name="frame",
    )

    camera = _camera_tensor(info, (800, 600), near=0.1, far=100.0, device=torch.device("cpu"))

    assert float(camera[7].item()) == 1000.0
    assert float(camera[8].item()) == 900.0
    assert float(camera[9].item()) == 400.0
    assert float(camera[10].item()) == 300.0
    assert float(camera[13].item()) == pytest.approx(0.07)
    assert float(camera[14].item()) == pytest.approx(-0.02)
