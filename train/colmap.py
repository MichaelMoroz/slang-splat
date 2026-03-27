from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData


@dataclass(frozen=True)
class CameraInfo:
    uid: int
    rotation_w2c: np.ndarray
    translation_w2c: np.ndarray
    fov_x: float
    fov_y: float
    width: int
    height: int
    image_path: Path
    image_name: str


CAMERA_MODELS = {
    0: 3,
    1: 4,
    2: 4,
    3: 5,
    4: 8,
    5: 8,
    6: 12,
    7: 5,
    8: 4,
    9: 5,
    10: 12,
}


def focal2fov(focal: float, pixels: int) -> float:
    return 2.0 * np.arctan(pixels / (2.0 * focal))


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float32,
    )


def _normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), float(eps))


def _rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = _normalize(source)
    target = _normalize(target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1e-6:
        return np.eye(3, dtype=np.float32)
    if dot < -1.0 + 1e-6:
        axis = np.cross(source, np.array((1.0, 0.0, 0.0), dtype=np.float32))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(source, np.array((0.0, 0.0, 1.0), dtype=np.float32))
        axis = _normalize(axis)
        outer = np.outer(axis, axis)
        return (2.0 * outer - np.eye(3, dtype=np.float32)).astype(np.float32)
    axis = np.cross(source, target).astype(np.float32)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis /= axis_norm
    skew = np.array(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        ),
        dtype=np.float32,
    )
    angle = float(np.arccos(dot))
    sin_angle = float(np.sin(angle))
    cos_angle = float(np.cos(angle))
    return (np.eye(3, dtype=np.float32) + sin_angle * skew + (1.0 - cos_angle) * (skew @ skew)).astype(np.float32)


def estimate_scene_up_rotation(cameras: list[CameraInfo], target_up: np.ndarray | None = None) -> np.ndarray:
    target = np.asarray((0.0, 1.0, 0.0) if target_up is None else target_up, dtype=np.float32)
    if not cameras:
        return np.eye(3, dtype=np.float32)
    up_vectors = np.stack([camera.rotation_w2c.T @ np.array((0.0, 1.0, 0.0), dtype=np.float32) for camera in cameras], axis=0)
    mean_up = up_vectors.mean(axis=0, dtype=np.float32)
    if float(np.linalg.norm(mean_up)) < 1e-6:
        return np.eye(3, dtype=np.float32)
    return _rotation_between_vectors(mean_up, target)


def rotate_scene(cameras: list[CameraInfo], xyz: np.ndarray, rotation: np.ndarray) -> tuple[list[CameraInfo], np.ndarray]:
    world_rotation = np.asarray(rotation, dtype=np.float32)
    rotated_xyz = (np.asarray(xyz, dtype=np.float32) @ world_rotation.T).astype(np.float32, copy=False)
    rotated_cameras = [
        CameraInfo(
            uid=camera.uid,
            rotation_w2c=(camera.rotation_w2c @ world_rotation.T).astype(np.float32, copy=False),
            translation_w2c=np.asarray(camera.translation_w2c, dtype=np.float32),
            fov_x=camera.fov_x,
            fov_y=camera.fov_y,
            width=camera.width,
            height=camera.height,
            image_path=camera.image_path,
            image_name=camera.image_name,
        )
        for camera in cameras
    ]
    return rotated_cameras, rotated_xyz


def _read(handle, fmt: str):
    return struct.unpack("<" + fmt, handle.read(struct.calcsize("<" + fmt)))


def read_intrinsics_binary(path: str | Path) -> dict[int, dict[str, object]]:
    intrinsics = {}
    with open(path, "rb") as handle:
        for _ in range(_read(handle, "Q")[0]):
            camera_id, model_id, width, height = _read(handle, "iiQQ")
            params = np.array(_read(handle, "d" * CAMERA_MODELS[model_id]), dtype=np.float64)
            intrinsics[camera_id] = {"width": int(width), "height": int(height), "model_id": int(model_id), "params": params}
    return intrinsics


def read_extrinsics_binary(path: str | Path) -> dict[int, dict[str, object]]:
    extrinsics = {}
    with open(path, "rb") as handle:
        for _ in range(_read(handle, "Q")[0]):
            image_id = _read(handle, "i")[0]
            qvec, tvec = np.array(_read(handle, "dddd")), np.array(_read(handle, "ddd"))
            camera_id = _read(handle, "i")[0]
            name = bytearray()
            while (char := handle.read(1)) != b"\x00":
                name.extend(char)
            handle.seek(_read(handle, "Q")[0] * 24, os.SEEK_CUR)
            extrinsics[image_id] = {"id": image_id, "qvec": qvec, "tvec": tvec, "camera_id": camera_id, "name": name.decode("utf-8")}
    return extrinsics


def read_points3d_binary(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    xyz, rgb = [], []
    with open(path, "rb") as handle:
        for _ in range(_read(handle, "Q")[0]):
            _read(handle, "Q")
            xyz.append(_read(handle, "ddd"))
            rgb.append(_read(handle, "BBB"))
            _read(handle, "d")
            handle.seek(_read(handle, "Q")[0] * 8, os.SEEK_CUR)
    return np.asarray(xyz, dtype=np.float32), np.asarray(rgb, dtype=np.uint8)


def _read_ply_points(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    vertex = PlyData.read(str(path))["vertex"]
    xyz = np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=1).astype(np.float32)
    rgb = np.stack((vertex["red"], vertex["green"], vertex["blue"]), axis=1).astype(np.uint8)
    return xyz, rgb


def load_colmap_cameras(scene_path: str | Path, image_dir: str | Path) -> list[CameraInfo]:
    scene_path, image_root = Path(scene_path), Path(image_dir)
    if not image_root.is_absolute():
        image_root = scene_path / image_root
    extrinsics = read_extrinsics_binary(scene_path / "sparse" / "0" / "images.bin")
    intrinsics = read_intrinsics_binary(scene_path / "sparse" / "0" / "cameras.bin")
    cameras = []
    for extr in extrinsics.values():
        intr = intrinsics[int(extr["camera_id"])]
        fx, fy = (float(intr["params"][0]),) * 2 if int(intr["model_id"]) == 0 else map(float, intr["params"][:2])
        name = Path(str(extr["name"]))
        cameras.append(
            CameraInfo(
                uid=int(extr["id"]),
                rotation_w2c=qvec2rotmat(np.asarray(extr["qvec"], dtype=np.float64)),
                translation_w2c=np.asarray(extr["tvec"], dtype=np.float32),
                fov_x=float(focal2fov(fx, int(intr["width"]))),
                fov_y=float(focal2fov(fy, int(intr["height"]))),
                width=int(intr["width"]),
                height=int(intr["height"]),
                image_path=image_root / name,
                image_name=name.stem,
            )
        )
    return sorted(cameras, key=lambda item: item.image_name)


def load_point_cloud(scene_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    scene_path = Path(scene_path)
    ply_path = scene_path / "sparse" / "0" / "points3D.ply"
    return _read_ply_points(ply_path) if ply_path.exists() else read_points3d_binary(scene_path / "sparse" / "0" / "points3D.bin")
