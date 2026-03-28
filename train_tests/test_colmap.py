from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_COLMAP_PATH = Path(__file__).resolve().parents[1] / "train" / "colmap.py"
_SPEC = importlib.util.spec_from_file_location("train_colmap_for_tests", _COLMAP_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_COLMAP = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _COLMAP
_SPEC.loader.exec_module(_COLMAP)

CameraInfo = _COLMAP.CameraInfo
estimate_scene_up_rotation = _COLMAP.estimate_scene_up_rotation
rotate_scene = _COLMAP.rotate_scene


def _camera(rotation_w2c: np.ndarray, uid: int = 0) -> CameraInfo:
    return CameraInfo(
        uid=uid,
        rotation_w2c=np.asarray(rotation_w2c, dtype=np.float32),
        translation_w2c=np.zeros((3,), dtype=np.float32),
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        k1=0.0,
        k2=0.0,
        fov_x=1.0,
        fov_y=1.0,
        width=100,
        height=100,
        image_path=Path("image.png"),
        image_name=f"cam_{uid}",
    )


def test_estimate_scene_up_rotation_aligns_average_camera_up_to_positive_y() -> None:
    tilt = np.array(
        (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
        ),
        dtype=np.float32,
    )
    cameras = [_camera(tilt, 0), _camera(tilt, 1)]

    rotation = estimate_scene_up_rotation(cameras)
    rotated_cameras, _ = rotate_scene(cameras, np.zeros((0, 3), dtype=np.float32), rotation)
    average_up = np.stack([camera.rotation_w2c.T @ np.array((0.0, 1.0, 0.0), dtype=np.float32) for camera in rotated_cameras], axis=0).mean(axis=0)

    np.testing.assert_allclose(average_up / np.linalg.norm(average_up), np.array((0.0, 1.0, 0.0), dtype=np.float32), atol=1e-5)


def test_rotate_scene_rotates_point_cloud_and_preserves_world_to_camera_translation() -> None:
    rotation = np.array(
        (
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    camera = CameraInfo(
        uid=1,
        rotation_w2c=np.eye(3, dtype=np.float32),
        translation_w2c=np.array((3.0, 4.0, 5.0), dtype=np.float32),
        fx=100.0,
        fy=100.0,
        cx=50.0,
        cy=50.0,
        k1=0.0,
        k2=0.0,
        fov_x=1.0,
        fov_y=1.0,
        width=100,
        height=100,
        image_path=Path("image.png"),
        image_name="cam",
    )
    xyz = np.array(((1.0, 0.0, 0.0), (0.0, 2.0, 0.0)), dtype=np.float32)

    rotated_cameras, rotated_xyz = rotate_scene([camera], xyz, rotation)

    np.testing.assert_allclose(rotated_xyz, np.array(((0.0, 1.0, 0.0), (-2.0, 0.0, 0.0)), dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(rotated_cameras[0].rotation_w2c, rotation.T, atol=1e-6)
    np.testing.assert_allclose(rotated_cameras[0].translation_w2c, camera.translation_w2c, atol=1e-6)
