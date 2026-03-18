from __future__ import annotations

from pathlib import Path
import struct

import numpy as np

from .colmap_types import ColmapCamera, ColmapImage, ColmapPoint3D, ColmapReconstruction

COLMAP_SIMPLE_PINHOLE_MODEL_ID = 0
COLMAP_PINHOLE_MODEL_ID = 1
COLMAP_SIMPLE_RADIAL_MODEL_ID = 2
COLMAP_RADIAL_MODEL_ID = 3
U64 = struct.Struct("<Q")
I32 = struct.Struct("<i")


def _read(handle, fmt: struct.Struct) -> int:
    return int(fmt.unpack(handle.read(fmt.size))[0])


def _read_f64_array(handle, count: int) -> tuple[float, ...]:
    return tuple(float(v) for v in struct.unpack("<" + ("d" * count), handle.read(8 * count)))


def _read_string(handle) -> str:
    data = bytearray()
    while (chunk := handle.read(1)) not in {b"", b"\x00"}:
        data.extend(chunk)
    if chunk == b"":
        raise ValueError("Unexpected EOF while reading null-terminated string from COLMAP images.bin.")
    return data.decode("utf-8")


def _camera_params_count(model_id: int) -> int:
    if model_id == COLMAP_SIMPLE_PINHOLE_MODEL_ID: return 3
    if model_id == COLMAP_PINHOLE_MODEL_ID: return 4
    if model_id == COLMAP_SIMPLE_RADIAL_MODEL_ID: return 4
    if model_id == COLMAP_RADIAL_MODEL_ID: return 5
    raise ValueError(
        f"Unsupported COLMAP camera model id {model_id}. "
        "Supported models are SIMPLE_PINHOLE (0), PINHOLE (1), SIMPLE_RADIAL (2), and RADIAL (3)."
    )


def _camera_intrinsics(model_id: int, params: tuple[float, ...]) -> tuple[float, float, float, float, float, float]:
    if model_id == COLMAP_SIMPLE_PINHOLE_MODEL_ID: return params[0], params[0], params[1], params[2], 0.0, 0.0
    if model_id == COLMAP_PINHOLE_MODEL_ID: return params[0], params[1], params[2], params[3], 0.0, 0.0
    if model_id == COLMAP_SIMPLE_RADIAL_MODEL_ID: return params[0], params[0], params[1], params[2], params[3], 0.0
    if model_id == COLMAP_RADIAL_MODEL_ID: return params[0], params[0], params[1], params[2], params[3], params[4]
    raise ValueError(f"Unsupported COLMAP camera model id {model_id}.")


def _load_cameras_bin(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with path.open("rb") as handle:
        for _ in range(_read(handle, U64)):
            camera_id, model_id = _read(handle, I32), _read(handle, I32)
            width, height = _read(handle, U64), _read(handle, U64)
            params = _read_f64_array(handle, _camera_params_count(model_id))
            fx, fy, cx, cy, k1, k2 = _camera_intrinsics(model_id, params)
            cameras[camera_id] = ColmapCamera(camera_id, model_id, width, height, fx, fy, cx, cy, k1, k2)
    return cameras


def _load_images_bin(path: Path) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    point_dtype = np.dtype([("xy", "<f8", (2,)), ("point_id", "<i8")])
    with path.open("rb") as handle:
        for _ in range(_read(handle, U64)):
            image_id = _read(handle, I32)
            q_wxyz = np.asarray(_read_f64_array(handle, 4), dtype=np.float32)
            t_xyz = np.asarray(_read_f64_array(handle, 3), dtype=np.float32)
            camera_id = _read(handle, I32)
            name = _read_string(handle)
            points2d = np.frombuffer(handle.read(_read(handle, U64) * point_dtype.itemsize), dtype=point_dtype)
            images[image_id] = ColmapImage(
                image_id=image_id,
                q_wxyz=q_wxyz,
                t_xyz=t_xyz,
                camera_id=camera_id,
                name=name,
                points2d_xy=points2d["xy"].astype(np.float32, copy=True),
                points2d_point3d_ids=points2d["point_id"].astype(np.int64, copy=True),
            )
    return images


def _load_points3d_bin(path: Path) -> dict[int, ColmapPoint3D]:
    points: dict[int, ColmapPoint3D] = {}
    with path.open("rb") as handle:
        for _ in range(_read(handle, U64)):
            point_id = _read(handle, U64)
            xyz = np.asarray(_read_f64_array(handle, 3), dtype=np.float32)
            rgb = np.frombuffer(handle.read(3), dtype=np.uint8).astype(np.float32) / 255.0
            error = float(struct.unpack("<d", handle.read(8))[0])
            handle.seek(_read(handle, U64) * 8, 1)
            points[point_id] = ColmapPoint3D(point_id=point_id, xyz=xyz, rgb=rgb, error=error)
    return points


def load_colmap_reconstruction(root: Path, sparse_subdir: str = "sparse/0") -> ColmapReconstruction:
    root_path = Path(root).resolve()
    sparse_dir = (root_path / sparse_subdir).resolve()
    cameras_path, images_path, points_path = (sparse_dir / name for name in ("cameras.bin", "images.bin", "points3D.bin"))
    for file_path in (cameras_path, images_path, points_path):
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required COLMAP file: {file_path}")
    return ColmapReconstruction(
        root=root_path,
        sparse_dir=sparse_dir,
        cameras=_load_cameras_bin(cameras_path),
        images=_load_images_bin(images_path),
        points3d=_load_points3d_bin(points_path),
    )
