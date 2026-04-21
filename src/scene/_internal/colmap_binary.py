from __future__ import annotations

from pathlib import Path
import struct

import numpy as np

from .colmap_types import ColmapCamera, ColmapImage, ColmapPoint3D, ColmapReconstruction

COLMAP_SIMPLE_PINHOLE_MODEL_ID = 0
COLMAP_PINHOLE_MODEL_ID = 1
COLMAP_SIMPLE_RADIAL_MODEL_ID = 2
COLMAP_RADIAL_MODEL_ID = 3
COLMAP_CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": COLMAP_SIMPLE_PINHOLE_MODEL_ID,
    "PINHOLE": COLMAP_PINHOLE_MODEL_ID,
    "SIMPLE_RADIAL": COLMAP_SIMPLE_RADIAL_MODEL_ID,
    "RADIAL": COLMAP_RADIAL_MODEL_ID,
}
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
            track_length = _read(handle, U64)
            handle.seek(track_length * 8, 1)
            points[point_id] = ColmapPoint3D(point_id=point_id, xyz=xyz, rgb=rgb, error=error, track_length=track_length)
    return points


def _iter_colmap_text_lines(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            yield line


def _load_cameras_txt(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    for line in _iter_colmap_text_lines(path):
        tokens = line.split()
        if len(tokens) < 5:
            raise ValueError(f"Malformed COLMAP cameras.txt line: {line}")
        camera_id = int(tokens[0])
        model_name = tokens[1]
        model_id = COLMAP_CAMERA_MODEL_IDS.get(model_name)
        if model_id is None:
            raise ValueError(
                f"Unsupported COLMAP camera model {model_name}. "
                "Supported models are SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, and RADIAL."
            )
        width = int(tokens[2])
        height = int(tokens[3])
        params = tuple(float(value) for value in tokens[4:])
        expected_param_count = _camera_params_count(model_id)
        if len(params) != expected_param_count:
            raise ValueError(
                f"COLMAP camera {camera_id} expected {expected_param_count} params for {model_name}, got {len(params)}."
            )
        fx, fy, cx, cy, k1, k2 = _camera_intrinsics(model_id, params)
        cameras[camera_id] = ColmapCamera(camera_id, model_id, width, height, fx, fy, cx, cy, k1, k2)
    return cameras


def _load_images_txt(path: Path) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    with path.open("r", encoding="utf-8") as handle:
        while True:
            header = handle.readline()
            if header == "":
                break
            header = header.strip()
            if not header or header.startswith("#"):
                continue
            points_line = handle.readline()
            if points_line == "":
                raise ValueError("Unexpected EOF while reading COLMAP images.txt point observations.")
            tokens = header.split()
            if len(tokens) < 10:
                raise ValueError(f"Malformed COLMAP images.txt line: {header}")
            image_id = int(tokens[0])
            q_wxyz = np.asarray([float(value) for value in tokens[1:5]], dtype=np.float32)
            t_xyz = np.asarray([float(value) for value in tokens[5:8]], dtype=np.float32)
            camera_id = int(tokens[8])
            name = " ".join(tokens[9:])
            point_tokens = points_line.strip().split()
            if point_tokens:
                if len(point_tokens) % 3 != 0:
                    raise ValueError(f"Malformed COLMAP images.txt observation line: {points_line.strip()}")
                point_triplets = np.asarray([float(value) for value in point_tokens], dtype=np.float64).reshape(-1, 3)
                points2d_xy = np.ascontiguousarray(point_triplets[:, :2], dtype=np.float32)
                points2d_point3d_ids = np.ascontiguousarray(point_triplets[:, 2], dtype=np.int64)
            else:
                points2d_xy = np.zeros((0, 2), dtype=np.float32)
                points2d_point3d_ids = np.zeros((0,), dtype=np.int64)
            images[image_id] = ColmapImage(
                image_id=image_id,
                q_wxyz=q_wxyz,
                t_xyz=t_xyz,
                camera_id=camera_id,
                name=name,
                points2d_xy=points2d_xy,
                points2d_point3d_ids=points2d_point3d_ids,
            )
    return images


def _load_points3d_txt(path: Path) -> dict[int, ColmapPoint3D]:
    points: dict[int, ColmapPoint3D] = {}
    for line in _iter_colmap_text_lines(path):
        tokens = line.split()
        if len(tokens) < 8:
            raise ValueError(f"Malformed COLMAP points3D.txt line: {line}")
        point_id = int(tokens[0])
        xyz = np.asarray([float(value) for value in tokens[1:4]], dtype=np.float32)
        rgb = np.asarray([int(value) for value in tokens[4:7]], dtype=np.float32) / 255.0
        error = float(tokens[7])
        track_tokens = tokens[8:]
        if len(track_tokens) % 2 != 0:
            raise ValueError(f"Malformed COLMAP points3D.txt track line: {line}")
        track_length = len(track_tokens) // 2
        points[point_id] = ColmapPoint3D(point_id=point_id, xyz=xyz, rgb=rgb, error=error, track_length=track_length)
    return points


def _resolve_colmap_sparse_paths(sparse_dir: Path) -> tuple[Path, Path, Path, str]:
    binary_paths = tuple((sparse_dir / name).resolve() for name in ("cameras.bin", "images.bin", "points3D.bin"))
    if all(path.exists() for path in binary_paths):
        return binary_paths[0], binary_paths[1], binary_paths[2], "bin"
    text_paths = tuple((sparse_dir / name).resolve() for name in ("cameras.txt", "images.txt", "points3D.txt"))
    if all(path.exists() for path in text_paths):
        return text_paths[0], text_paths[1], text_paths[2], "txt"
    missing = [str(path) for path in (*binary_paths, *text_paths) if not path.exists()]
    raise FileNotFoundError(
        "Missing required COLMAP sparse files. Expected either binary or text export under "
        f"{sparse_dir}. Missing candidates: {', '.join(missing)}"
    )


def load_colmap_reconstruction(root: Path, sparse_subdir: str = "sparse/0") -> ColmapReconstruction:
    root_path = Path(root).resolve()
    sparse_dir = (root_path / sparse_subdir).resolve()
    cameras_path, images_path, points_path, format_kind = _resolve_colmap_sparse_paths(sparse_dir)
    return ColmapReconstruction(
        root=root_path,
        sparse_dir=sparse_dir,
        cameras=_load_cameras_bin(cameras_path) if format_kind == "bin" else _load_cameras_txt(cameras_path),
        images=_load_images_bin(images_path) if format_kind == "bin" else _load_images_txt(images_path),
        points3d=_load_points3d_bin(points_path) if format_kind == "bin" else _load_points3d_txt(points_path),
    )
