from __future__ import annotations

import numpy as np

from src.renderer import Camera, PROJECTION_MODEL_EQUIRECTANGULAR, PROJECTION_MODEL_FISHEYE


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


def test_distorted_screen_ray_roundtrips_through_projection():
    camera = Camera.look_at(
        position=(0.0, 0.0, -3.0),
        target=(0.0, 0.0, 0.0),
        near=0.1,
        far=20.0,
        distortion_k1=0.08,
        distortion_k2=-0.02,
    )
    screen = np.array([517.5, 201.25], dtype=np.float32)
    ray = camera.screen_to_world_ray(screen, 640, 480)
    point = camera.position + ray * np.float32(5.0)
    projected, ok = camera.project_world_to_screen(point, 640, 480)

    assert ok
    np.testing.assert_allclose(projected, screen, rtol=0.0, atol=1e-4)


def test_full_opencv_distorted_screen_ray_roundtrips_through_projection():
    camera = Camera.look_at(
        position=(0.0, 0.0, -3.0),
        target=(0.0, 0.0, 0.0),
        near=0.1,
        far=20.0,
        distortion_k1=0.08,
        distortion_k2=-0.02,
        distortion_p1=0.003,
        distortion_p2=-0.002,
        distortion_k3=0.004,
        distortion_k4=0.001,
        distortion_k5=-0.0005,
        distortion_k6=0.0002,
    )
    screen = np.array([517.5, 201.25], dtype=np.float32)
    ray = camera.screen_to_world_ray(screen, 640, 480)
    point = camera.position + ray * np.float32(5.0)
    projected, ok = camera.project_world_to_screen(point, 640, 480)

    assert ok
    np.testing.assert_allclose(projected, screen, rtol=0.0, atol=1e-4)


def _fisheye_camera(**distortion) -> Camera:
    return Camera(
        position=np.zeros((3,), dtype=np.float32),
        target=np.array((0.0, 0.0, 1.0), dtype=np.float32),
        up=np.array((0.0, 1.0, 0.0), dtype=np.float32),
        fx=200.0,
        fy=200.0,
        cx=320.0,
        cy=240.0,
        projection_model=PROJECTION_MODEL_FISHEYE,
        **distortion,
    )


def test_fisheye_equidistant_projection_maps_known_angles():
    camera = _fisheye_camera()
    width, height = 640, 480

    # Zero coefficients reduce KB4 to the pure equidistant mapping r = f * theta.
    cases = (
        ((0.0, 0.0, 1.0), (320.0, 240.0)),
        ((1.0, 0.0, 0.0), (320.0 + 200.0 * np.pi / 2.0, 240.0)),
        ((0.0, 1.0, 0.0), (320.0, 240.0 + 200.0 * np.pi / 2.0)),
        ((1.0, 0.0, -1.0), (320.0 + 200.0 * 0.75 * np.pi, 240.0)),
    )
    for camera_pos, expected_screen in cases:
        screen, ok = camera.project_camera_to_screen(np.asarray(camera_pos, dtype=np.float32), width, height)
        assert ok
        np.testing.assert_allclose(screen, np.asarray(expected_screen, dtype=np.float32), rtol=0.0, atol=1e-4)

    # Directly behind the optical axis the azimuth is undefined.
    _, ok = camera.project_camera_to_screen(np.array((0.0, 0.0, -1.0), dtype=np.float32), width, height)
    assert not ok


def test_fisheye_distorted_screen_ray_roundtrips_through_projection():
    camera = _fisheye_camera(distortion_k1=0.03, distortion_k2=-0.008, distortion_k3=0.002, distortion_k4=-0.0004)
    width, height = 640, 480

    for screen in ((320.0, 240.0), (517.5, 201.25), (80.25, 400.5), (600.0, 30.0)):
        screen_pos = np.asarray(screen, dtype=np.float32)
        ray = camera.screen_to_world_ray(screen_pos, width, height)
        point = camera.position + ray * np.float32(5.0)
        projected, ok = camera.project_world_to_screen(point, width, height)
        assert ok
        np.testing.assert_allclose(projected, screen_pos, rtol=0.0, atol=1e-3)


def test_fisheye_projection_roundtrips_beyond_180_degrees():
    camera = _fisheye_camera(distortion_k1=0.02, distortion_k2=-0.004)
    width, height = 640, 480

    # theta = 130 degrees: well behind the camera plane, valid for a >180 lens.
    theta = np.deg2rad(130.0)
    direction = np.array((np.sin(theta), 0.0, np.cos(theta)), dtype=np.float32)
    screen, ok = camera.project_world_to_screen(camera.position + direction * np.float32(4.0), width, height)
    assert ok
    ray = camera.screen_to_world_ray(screen, width, height)
    np.testing.assert_allclose(ray, direction, rtol=0.0, atol=1e-4)
    assert camera.gpu_params(width, height)["projectionModel"] == np.uint32(PROJECTION_MODEL_FISHEYE)


def test_equirectangular_projection_maps_full_sphere_and_roundtrips():
    camera = Camera.look_at(
        position=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, 1.0),
        projection_model=PROJECTION_MODEL_EQUIRECTANGULAR,
    )
    width, height = 400, 200

    cases = (
        ((0.0, 0.0, 1.0), (200.0, 100.0)),
        ((1.0, 0.0, 0.0), (300.0, 100.0)),
        ((0.0, -1.0, 0.0), (200.0, 0.0)),
        ((0.0, 0.0, -1.0), (0.0, 100.0)),
    )
    for world_pos, expected_screen in cases:
        screen, ok = camera.project_world_to_screen(np.asarray(world_pos, dtype=np.float32), width, height)
        assert ok
        np.testing.assert_allclose(screen, np.asarray(expected_screen, dtype=np.float32), rtol=0.0, atol=1e-5)

    sample_screen = np.array((123.5, 45.25), dtype=np.float32)
    ray = camera.screen_to_world_ray(sample_screen, width, height)
    point = camera.position + ray * np.float32(7.0)
    projected, ok = camera.project_world_to_screen(point, width, height)

    assert ok
    np.testing.assert_allclose(projected, sample_screen, rtol=0.0, atol=1e-4)
    assert camera.gpu_params(width, height)["projectionModel"] == np.uint32(PROJECTION_MODEL_EQUIRECTANGULAR)
