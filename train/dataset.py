from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .colmap import CameraInfo, load_colmap_cameras, load_point_cloud


def _srgb_to_linear_rgb(image: torch.Tensor) -> torch.Tensor:
    return torch.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055).pow(2.4))


@dataclass(frozen=True)
class CameraSample:
    image_name: str
    image_path: Path
    image_size: tuple[int, int]
    camera_params: torch.Tensor
    image: torch.Tensor | None


@dataclass(frozen=True)
class SceneData:
    scene_path: Path
    point_xyz: np.ndarray
    point_rgb: np.ndarray
    train_cameras: list[CameraSample]
    test_cameras: list[CameraSample]
    background: tuple[float, float, float]
    extent_radius: float


def _rotmat_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    m, trace = matrix.astype(np.float64), float(np.trace(matrix))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        quat = np.array((0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s), dtype=np.float32)
    else:
        idx = int(np.argmax(np.diag(m)))
        vals = (
            (2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]), lambda s: ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)),
            (2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]), lambda s: ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)),
            (2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]), lambda s: ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)),
        )
        quat = np.array(vals[idx][1](vals[idx][0]), dtype=np.float32)
    return quat / np.linalg.norm(quat)


def _camera_tensor(info: CameraInfo, image_size: tuple[int, int], near: float, far: float, device: torch.device) -> torch.Tensor:
    width, height = image_size
    quat = _rotmat_to_quat_wxyz(info.rotation_w2c)
    fx, fy = width / (2.0 * np.tan(info.fov_x * 0.5)), height / (2.0 * np.tan(info.fov_y * 0.5))
    return torch.tensor((quat[0], quat[1], quat[2], quat[3], *info.translation_w2c, fx, fy, width * 0.5, height * 0.5, near, far, 0.0, 0.0), dtype=torch.float32, device=device)


def _load_image_tensor(path: Path, device: torch.device, preload_cuda: bool) -> torch.Tensor:
    image = torch.from_numpy(np.array(Image.open(path).convert("RGB"), dtype=np.uint8, copy=True))
    return image.to(device=device) if preload_cuda else image


def _image_to_linear_float(image: torch.Tensor) -> torch.Tensor:
    return _srgb_to_linear_rgb(image.to(dtype=torch.float32) / 255.0)


def probe_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.size[0]), int(image.size[1])


def _build_samples(infos: list[CameraInfo], device: torch.device, preload_cuda: bool, near: float, far: float) -> list[CameraSample]:
    if not infos:
        return []
    ref_width, ref_height = probe_image_size(infos[0].image_path)
    sx, sy = ref_width / infos[0].width, ref_height / infos[0].height
    return [
        CameraSample(
            image_name=info.image_name,
            image_path=info.image_path,
            image_size=(max(int(round(info.width * sx)), 1), max(int(round(info.height * sy)), 1)),
            camera_params=_camera_tensor(info, (max(int(round(info.width * sx)), 1), max(int(round(info.height * sy)), 1)), near, far, device),
            image=_load_image_tensor(info.image_path, device, preload_cuda) if preload_cuda else None,
        )
        for info in infos
    ]


def _compute_extent_radius(cameras: list[CameraInfo]) -> float:
    centers = np.hstack([(-(c.rotation_w2c.T @ c.translation_w2c)).reshape(3, 1) for c in cameras])
    center = centers.mean(axis=1, keepdims=True)
    return float(np.linalg.norm(centers - center, axis=0).max() * 1.1)


def load_colmap_scene(scene_path: str | Path, image_dir: str = "images_4", eval_split: bool = True, llff_hold: int = 8, preload_cuda: bool = True, device: str | torch.device = "cuda", near: float = 0.0, far: float = 1000.0, white_background: bool = False) -> SceneData:
    device, scene_path = torch.device(device), Path(scene_path).resolve()
    cameras = load_colmap_cameras(scene_path, image_dir)
    train_infos = [cam for i, cam in enumerate(cameras) if not eval_split or i % llff_hold != 0]
    test_infos = [] if not eval_split else [cam for i, cam in enumerate(cameras) if i % llff_hold == 0]
    xyz, rgb = load_point_cloud(scene_path)
    return SceneData(scene_path, xyz, rgb, _build_samples(train_infos, device, preload_cuda, near, far), _build_samples(test_infos, device, preload_cuda, near, far), (1.0, 1.0, 1.0) if white_background else (0.0, 0.0, 0.0), _compute_extent_radius(train_infos))
