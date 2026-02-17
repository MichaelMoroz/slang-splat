from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..renderer.camera import Camera
from .gaussian_scene import GaussianScene

COLMAP_SIMPLE_PINHOLE_MODEL_ID = 0
COLMAP_PINHOLE_MODEL_ID = 1


@dataclass(slots=True)
class ColmapCamera:
    camera_id: int
    model_id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(slots=True)
class ColmapImage:
    image_id: int
    q_wxyz: np.ndarray
    t_xyz: np.ndarray
    camera_id: int
    name: str
    points2d_xy: np.ndarray
    points2d_point3d_ids: np.ndarray


@dataclass(slots=True)
class ColmapPoint3D:
    point_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float


@dataclass(slots=True)
class ColmapReconstruction:
    root: Path
    sparse_dir: Path
    cameras: dict[int, ColmapCamera]
    images: dict[int, ColmapImage]
    points3d: dict[int, ColmapPoint3D]


@dataclass(slots=True)
class ColmapFrame:
    image_id: int
    image_path: Path
    q_wxyz: np.ndarray
    t_xyz: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def make_camera(self, near: float = 0.1, far: float = 120.0) -> Camera:
        return Camera.from_colmap(
            q_wxyz=self.q_wxyz,
            t_xyz=self.t_xyz,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            near=near,
            far=far,
        )


@dataclass(slots=True)
class GaussianInitHyperParams:
    position_jitter_std: float = 0.01
    base_scale: float = 0.03
    scale_jitter_ratio: float = 0.2
    initial_opacity: float = 0.5
    color_jitter_std: float = 0.0


def _read_u64(handle) -> int:
    return int(struct.unpack("<Q", handle.read(8))[0])


def _read_i32(handle) -> int:
    return int(struct.unpack("<i", handle.read(4))[0])


def _read_f64_array(handle, count: int) -> tuple[float, ...]:
    return tuple(float(v) for v in struct.unpack("<" + ("d" * count), handle.read(8 * count)))


def _read_null_terminated_string(handle) -> str:
    data = bytearray()
    while True:
        chunk = handle.read(1)
        if chunk == b"":
            raise ValueError("Unexpected EOF while reading null-terminated string from COLMAP images.bin.")
        if chunk == b"\x00":
            return data.decode("utf-8")
        data.extend(chunk)


def _camera_params_count(model_id: int) -> int:
    if model_id == COLMAP_SIMPLE_PINHOLE_MODEL_ID:
        return 3
    if model_id == COLMAP_PINHOLE_MODEL_ID:
        return 4
    raise ValueError(
        f"Unsupported COLMAP camera model id {model_id}. "
        "Only SIMPLE_PINHOLE (0) and PINHOLE (1) are currently supported."
    )


def _load_cameras_bin(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with path.open("rb") as handle:
        count = _read_u64(handle)
        for _ in range(count):
            camera_id = _read_i32(handle)
            model_id = _read_i32(handle)
            width = _read_u64(handle)
            height = _read_u64(handle)
            params = _read_f64_array(handle, _camera_params_count(model_id))
            if model_id == COLMAP_SIMPLE_PINHOLE_MODEL_ID:
                f, cx, cy = params
                fx = f
                fy = f
            else:
                fx, fy, cx, cy = params
            cameras[camera_id] = ColmapCamera(
                camera_id=camera_id,
                model_id=model_id,
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )
    return cameras


def _load_images_bin(path: Path) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    with path.open("rb") as handle:
        count = _read_u64(handle)
        for _ in range(count):
            image_id = _read_i32(handle)
            q_wxyz = np.asarray(_read_f64_array(handle, 4), dtype=np.float32)
            t_xyz = np.asarray(_read_f64_array(handle, 3), dtype=np.float32)
            camera_id = _read_i32(handle)
            name = _read_null_terminated_string(handle)
            points2d_count = _read_u64(handle)
            points2d_xy = np.zeros((points2d_count, 2), dtype=np.float32)
            points2d_point3d_ids = np.zeros((points2d_count,), dtype=np.int64)
            for i in range(points2d_count):
                x = float(struct.unpack("<d", handle.read(8))[0])
                y = float(struct.unpack("<d", handle.read(8))[0])
                point3d_id = int(struct.unpack("<q", handle.read(8))[0])
                points2d_xy[i, 0] = x
                points2d_xy[i, 1] = y
                points2d_point3d_ids[i] = point3d_id
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
        count = _read_u64(handle)
        for _ in range(count):
            point_id = _read_u64(handle)
            xyz = np.asarray(_read_f64_array(handle, 3), dtype=np.float32)
            rgb = np.frombuffer(handle.read(3), dtype=np.uint8).copy()
            error = float(struct.unpack("<d", handle.read(8))[0])
            track_len = _read_u64(handle)
            handle.seek(track_len * 8, 1)
            points[point_id] = ColmapPoint3D(
                point_id=point_id,
                xyz=xyz,
                rgb=rgb.astype(np.float32) / 255.0,
                error=error,
            )
    return points


def load_colmap_reconstruction(root: Path, sparse_subdir: str = "sparse/0") -> ColmapReconstruction:
    root_path = Path(root).resolve()
    sparse_dir = (root_path / sparse_subdir).resolve()
    cameras_path = sparse_dir / "cameras.bin"
    images_path = sparse_dir / "images.bin"
    points_path = sparse_dir / "points3D.bin"
    for file_path in (cameras_path, images_path, points_path):
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required COLMAP file: {file_path}")
    cameras = _load_cameras_bin(cameras_path)
    images = _load_images_bin(images_path)
    points = _load_points3d_bin(points_path)
    return ColmapReconstruction(
        root=root_path,
        sparse_dir=sparse_dir,
        cameras=cameras,
        images=images,
        points3d=points,
    )


def build_training_frames(recon: ColmapReconstruction, images_subdir: str = "images_4") -> list[ColmapFrame]:
    images_root = (recon.root / images_subdir).resolve()
    if not images_root.exists():
        raise FileNotFoundError(f"COLMAP image directory does not exist: {images_root}")
    frames: list[ColmapFrame] = []
    for image_id in sorted(recon.images.keys()):
        image = recon.images[image_id]
        camera = recon.cameras.get(image.camera_id)
        if camera is None:
            continue
        image_path = (images_root / image.name).resolve()
        if not image_path.exists():
            continue
        with Image.open(image_path) as pil_image:
            width, height = pil_image.size
        sx = float(width) / float(camera.width)
        sy = float(height) / float(camera.height)
        frames.append(
            ColmapFrame(
                image_id=image.image_id,
                image_path=image_path,
                q_wxyz=image.q_wxyz.astype(np.float32),
                t_xyz=image.t_xyz.astype(np.float32),
                fx=float(camera.fx) * sx,
                fy=float(camera.fy) * sy,
                cx=float(camera.cx) * sx,
                cy=float(camera.cy) * sy,
                width=int(width),
                height=int(height),
            )
        )
    if not frames:
        raise RuntimeError(f"No training frames were found in {images_root}.")
    return frames


def _random_unit_quaternions(rng: np.random.Generator, count: int) -> np.ndarray:
    q = rng.normal(size=(count, 4)).astype(np.float32)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
    return q


def initialize_scene_from_colmap_points(
    recon: ColmapReconstruction,
    max_gaussians: int,
    seed: int,
    init_hparams: GaussianInitHyperParams | None = None,
) -> GaussianScene:
    params = init_hparams if init_hparams is not None else GaussianInitHyperParams()
    if not recon.points3d:
        raise RuntimeError("COLMAP reconstruction has no 3D points.")
    rng = np.random.default_rng(int(seed))
    points = list(recon.points3d.values())
    point_count = len(points)
    chosen_count = point_count if max_gaussians <= 0 else max(int(max_gaussians), 1)
    sample_with_replacement = chosen_count > point_count
    indices = rng.choice(point_count, size=chosen_count, replace=sample_with_replacement)
    positions = np.stack([points[int(i)].xyz for i in indices], axis=0).astype(np.float32)
    colors = np.stack([points[int(i)].rgb for i in indices], axis=0).astype(np.float32)
    if float(params.position_jitter_std) > 0.0:
        positions += rng.normal(scale=float(params.position_jitter_std), size=positions.shape).astype(np.float32)
    if float(params.color_jitter_std) > 0.0:
        colors = np.clip(
            colors + rng.normal(scale=float(params.color_jitter_std), size=colors.shape).astype(np.float32),
            0.0,
            1.0,
        )

    base_scale = max(float(params.base_scale), 1e-4)
    scale_jitter_ratio = max(float(params.scale_jitter_ratio), 0.0)
    scales = np.full((chosen_count, 3), base_scale, dtype=np.float32)
    if scale_jitter_ratio > 0.0:
        jitter = rng.uniform(1.0 - scale_jitter_ratio, 1.0 + scale_jitter_ratio, size=(chosen_count, 3)).astype(
            np.float32
        )
        scales = np.clip(scales * jitter, 1e-4, 10.0)

    rotations = _random_unit_quaternions(rng, chosen_count)
    opacities = np.full((chosen_count,), float(np.clip(params.initial_opacity, 1e-4, 0.9999)), dtype=np.float32)
    sh_coeffs = np.zeros((chosen_count, 1, 3), dtype=np.float32)
    return GaussianScene(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
        sh_coeffs=sh_coeffs,
    )
