from __future__ import annotations

from pathlib import Path
import struct

import numpy as np

from .colmap_types import ColmapCamera, ColmapImage, ColmapPoint3D, ColmapReconstruction

COLMAP_SIMPLE_PINHOLE_MODEL_ID = 0
COLMAP_PINHOLE_MODEL_ID = 1
COLMAP_SIMPLE_RADIAL_MODEL_ID = 2
COLMAP_RADIAL_MODEL_ID = 3
COLMAP_OPENCV_MODEL_ID = 4
COLMAP_FULL_OPENCV_MODEL_ID = 6
COLMAP_CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": COLMAP_SIMPLE_PINHOLE_MODEL_ID,
    "PINHOLE": COLMAP_PINHOLE_MODEL_ID,
    "SIMPLE_RADIAL": COLMAP_SIMPLE_RADIAL_MODEL_ID,
    "RADIAL": COLMAP_RADIAL_MODEL_ID,
    "OPENCV": COLMAP_OPENCV_MODEL_ID,
    "FULL_OPENCV": COLMAP_FULL_OPENCV_MODEL_ID,
}
COLMAP_CAMERA_MODEL_PARAM_COUNTS = {
    COLMAP_SIMPLE_PINHOLE_MODEL_ID: 3,
    COLMAP_PINHOLE_MODEL_ID: 4,
    COLMAP_SIMPLE_RADIAL_MODEL_ID: 4,
    COLMAP_RADIAL_MODEL_ID: 5,
    COLMAP_OPENCV_MODEL_ID: 8,
    COLMAP_FULL_OPENCV_MODEL_ID: 12,
}
COLMAP_SUPPORTED_CAMERA_MODEL_NAMES = tuple(COLMAP_CAMERA_MODEL_IDS)
U64 = struct.Struct("<Q")
I32 = struct.Struct("<i")
COLMAP_DEFAULT_SPARSE_SUBDIR = "sparse/0"


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
    count = COLMAP_CAMERA_MODEL_PARAM_COUNTS.get(int(model_id))
    if count is not None: return count
    raise ValueError(
        f"Unsupported COLMAP camera model id {model_id}. "
        f"Supported models are {', '.join(COLMAP_SUPPORTED_CAMERA_MODEL_NAMES)}."
    )


def _camera_intrinsics(model_id: int, params: tuple[float, ...]) -> tuple[float, float, float, float, float, float, float, float, float, float, float, float]:
    if model_id == COLMAP_SIMPLE_PINHOLE_MODEL_ID: return params[0], params[0], params[1], params[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if model_id == COLMAP_PINHOLE_MODEL_ID: return params[0], params[1], params[2], params[3], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if model_id == COLMAP_SIMPLE_RADIAL_MODEL_ID: return params[0], params[0], params[1], params[2], params[3], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if model_id == COLMAP_RADIAL_MODEL_ID: return params[0], params[0], params[1], params[2], params[3], params[4], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if model_id == COLMAP_OPENCV_MODEL_ID: return params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], 0.0, 0.0, 0.0, 0.0
    if model_id == COLMAP_FULL_OPENCV_MODEL_ID: return params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11]
    raise ValueError(f"Unsupported COLMAP camera model id {model_id}.")


def _load_cameras_bin(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with path.open("rb") as handle:
        for _ in range(_read(handle, U64)):
            camera_id, model_id = _read(handle, I32), _read(handle, I32)
            width, height = _read(handle, U64), _read(handle, U64)
            params = _read_f64_array(handle, _camera_params_count(model_id))
            fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = _camera_intrinsics(model_id, params)
            cameras[camera_id] = ColmapCamera(camera_id, model_id, width, height, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6)
    return cameras


def _load_images_bin(path: Path, *, load_observations: bool = True) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    point_dtype = np.dtype([("xy", "<f8", (2,)), ("point_id", "<i8")])
    empty_xy = np.zeros((0, 2), dtype=np.float32)
    empty_ids = np.zeros((0,), dtype=np.int64)
    with path.open("rb") as handle:
        for _ in range(_read(handle, U64)):
            image_id = _read(handle, I32)
            q_wxyz = np.asarray(_read_f64_array(handle, 4), dtype=np.float32)
            t_xyz = np.asarray(_read_f64_array(handle, 3), dtype=np.float32)
            camera_id = _read(handle, I32)
            name = _read_string(handle)
            point_count = _read(handle, U64)
            if not load_observations:
                # Skip past the observation block without materializing it.
                handle.seek(point_count * point_dtype.itemsize, 1)
                points2d_xy, points2d_point3d_ids = empty_xy, empty_ids
            else:
                points2d = np.frombuffer(handle.read(point_count * point_dtype.itemsize), dtype=point_dtype)
                points2d_xy = points2d["xy"].astype(np.float32, copy=True)
                points2d_point3d_ids = points2d["point_id"].astype(np.int64, copy=True)
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
                f"Supported models are {', '.join(COLMAP_SUPPORTED_CAMERA_MODEL_NAMES)}."
            )
        width = int(tokens[2])
        height = int(tokens[3])
        params = tuple(float(value) for value in tokens[4:])
        expected_param_count = _camera_params_count(model_id)
        if len(params) != expected_param_count:
            raise ValueError(
                f"COLMAP camera {camera_id} expected {expected_param_count} params for {model_name}, got {len(params)}."
            )
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = _camera_intrinsics(model_id, params)
        cameras[camera_id] = ColmapCamera(camera_id, model_id, width, height, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6)
    return cameras


def _load_images_txt(path: Path, *, load_observations: bool = True) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    empty_xy = np.zeros((0, 2), dtype=np.float32)
    empty_ids = np.zeros((0,), dtype=np.int64)
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
            # Per-image 2D observations are only needed for overlays / photometric
            # compensation; skip the (potentially huge) parse when not requested.
            if not load_observations:
                points2d_xy, points2d_point3d_ids = empty_xy, empty_ids
            else:
                point_tokens = points_line.split()
                if len(point_tokens) % 3 != 0:
                    raise ValueError(f"Malformed COLMAP images.txt observation line: {points_line.strip()}")
                if point_tokens:
                    point_triplets = np.array(point_tokens, dtype=np.float64).reshape(-1, 3)
                    points2d_xy = np.ascontiguousarray(point_triplets[:, :2], dtype=np.float32)
                    points2d_point3d_ids = np.ascontiguousarray(point_triplets[:, 2], dtype=np.int64)
                else:
                    points2d_xy, points2d_point3d_ids = empty_xy, empty_ids
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
        xyz = np.array(tokens[1:4], dtype=np.float32)
        rgb = np.array(tokens[4:7], dtype=np.float32) / 255.0
        error = float(tokens[7])
        track_token_count = len(tokens) - 8
        if track_token_count % 2 != 0:
            raise ValueError(f"Malformed COLMAP points3D.txt track line: {line}")
        track_length = track_token_count // 2
        points[point_id] = ColmapPoint3D(point_id=point_id, xyz=xyz, rgb=rgb, error=error, track_length=track_length)
    return points


def _resolve_colmap_sparse_paths(*sparse_dirs: Path) -> tuple[Path, Path, Path, str]:
    for sparse_dir in sparse_dirs:
        sparse_path = Path(sparse_dir).resolve()
        for format_kind in ("bin", "txt"):
            names = tuple(f"{name}.{format_kind}" for name in ("cameras", "images", "points3D"))
            paths = tuple((sparse_path / name).resolve() for name in names)
            if all(path.exists() for path in paths):
                return paths[0], paths[1], paths[2], format_kind
    raise FileNotFoundError(f"Missing required COLMAP sparse files under: {', '.join(str(Path(path).resolve()) for path in sparse_dirs)}")


def load_colmap_reconstruction(
    root: Path,
    sparse_subdir: str = COLMAP_DEFAULT_SPARSE_SUBDIR,
    *,
    load_points3d: bool = True,
    load_observations: bool = True,
) -> ColmapReconstruction:
    """Load a COLMAP reconstruction.

    ``load_points3d`` and ``load_observations`` can be disabled to skip the
    expensive 3D point cloud and per-image 2D observation parsing when only camera
    poses are needed (e.g. the dataset-selection preview), which is the bulk of the
    parse cost for large reconstructions.
    """
    root_path = Path(root).resolve()
    sparse_subdir_text = str(sparse_subdir)
    candidates = ((root_path / sparse_subdir_text).resolve() if sparse_subdir_text else root_path,)
    if sparse_subdir_text.replace("\\", "/").strip("/") == COLMAP_DEFAULT_SPARSE_SUBDIR:
        child_sparse_dirs = tuple((path / "sparse").resolve() for path in sorted(root_path.iterdir()) if path.is_dir()) if root_path.exists() else ()
        candidates = tuple(dict.fromkeys((*candidates, (root_path / "sparse").resolve(), root_path, *child_sparse_dirs)))
    cameras_path, images_path, points_path, format_kind = _resolve_colmap_sparse_paths(*candidates)
    sparse_dir = cameras_path.parent
    if not load_points3d:
        points3d: dict[int, ColmapPoint3D] = {}
    elif format_kind == "bin":
        points3d = _load_points3d_bin(points_path)
    else:
        points3d = _load_points3d_txt(points_path)
    return ColmapReconstruction(
        root=root_path,
        sparse_dir=sparse_dir,
        cameras=_load_cameras_bin(cameras_path) if format_kind == "bin" else _load_cameras_txt(cameras_path),
        images=(
            _load_images_bin(images_path, load_observations=load_observations)
            if format_kind == "bin"
            else _load_images_txt(images_path, load_observations=load_observations)
        ),
        points3d=points3d,
    )


def count_colmap_points3d(sparse_dir: Path) -> tuple[int, int]:
    """Fast ``(total_points, tracked_points_min2)`` count without parsing coordinates.

    Streams ``points3D.{txt,bin}`` and only inspects track lengths, so it is cheap
    enough to run during the dataset-selection preview for very large point clouds.
    """
    sparse_path = Path(sparse_dir).resolve()
    txt_path = sparse_path / "points3D.txt"
    bin_path = sparse_path / "points3D.bin"
    if txt_path.exists():
        total = 0
        tracked = 0
        with txt_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                total += 1
                # track_length = (token_count - 8) / 2; >= 2 observations.
                if stripped.count(" ") >= 11:
                    tracked += 1
        return total, tracked
    if bin_path.exists():
        # Per record: id(u64) + xyz(3*f64) + rgb(3*u8) + error(f64) = 43 bytes,
        # then track_length(u64), then track_length * (2*i32) bytes.
        with bin_path.open("rb") as handle:
            count = _read(handle, U64)
            tracked = 0
            for _ in range(count):
                handle.seek(43, 1)
                track_length = _read(handle, U64)
                if track_length >= 2:
                    tracked += 1
                handle.seek(int(track_length) * 8, 1)
        return int(count), int(tracked)
    return 0, 0
