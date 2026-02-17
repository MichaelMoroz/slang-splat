from __future__ import annotations

import numpy as np

from src.renderer import Camera


def test_look_at_camera_defaults_to_center_principal_point():
    camera = Camera.look_at(position=(0.0, 0.0, -3.0), target=(0.0, 0.0, 0.0), fov_y_degrees=60.0)
    fx, fy = camera.focal_pixels_xy(640, 480)
    cx, cy = camera.principal_point(640, 480)
    assert np.isclose(fx, fy)
    assert np.isclose(cx, 320.0)
    assert np.isclose(cy, 240.0)
    gpu = camera.gpu_params(640, 480)
    assert "focalPixels" in gpu
    assert "principalPoint" in gpu


def test_colmap_camera_uses_explicit_intrinsics_and_extrinsics():
    camera = Camera.from_colmap(
        q_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        t_xyz=np.array([0.0, 0.0, 2.0], dtype=np.float32),
        fx=500.0,
        fy=450.0,
        cx=321.0,
        cy=241.0,
        near=0.1,
        far=20.0,
    )
    fx, fy = camera.focal_pixels_xy(999, 777)
    cx, cy = camera.principal_point(999, 777)
    assert np.isclose(fx, 500.0)
    assert np.isclose(fy, 450.0)
    assert np.isclose(cx, 321.0)
    assert np.isclose(cy, 241.0)
    assert np.allclose(camera.position, np.array([0.0, 0.0, -2.0], dtype=np.float32), atol=1e-5)
    right, up, forward = camera.basis()
    np.testing.assert_allclose(right, np.array([1.0, 0.0, 0.0], dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(up, np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(forward, np.array([0.0, 0.0, 1.0], dtype=np.float32), atol=1e-5)
