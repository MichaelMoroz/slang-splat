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
