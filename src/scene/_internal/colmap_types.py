from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...renderer.camera import Camera


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
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0


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
    track_length: int = 0


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
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    k5: float = 0.0
    k6: float = 0.0

    def make_camera(self, near: float = 0.1, far: float = 120.0) -> Camera:
        return Camera.from_colmap(
            q_wxyz=self.q_wxyz,
            t_xyz=self.t_xyz,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            distortion_k1=self.k1,
            distortion_k2=self.k2,
            distortion_p1=self.p1,
            distortion_p2=self.p2,
            distortion_k3=self.k3,
            distortion_k4=self.k4,
            distortion_k5=self.k5,
            distortion_k6=self.k6,
            near=near,
            far=far,
        )


@dataclass(slots=True)
class GaussianInitHyperParams:
    position_jitter_std: float | None = None
    base_scale: float | None = None
    scale_jitter_ratio: float | None = None
    initial_opacity: float | None = None
    color_jitter_std: float | None = None


def point_tables(recon: ColmapReconstruction, min_track_length: int = 0) -> tuple[np.ndarray, np.ndarray]:
    xyz = getattr(recon, "point_xyz_table", None)
    rgb = getattr(recon, "point_rgb_table", None)
    track_lengths = getattr(recon, "point_track_length_table", None)
    min_track = max(int(min_track_length), 0)
    if xyz is not None and rgb is not None:
        xyz_arr = np.ascontiguousarray(xyz, dtype=np.float32)
        rgb_arr = np.ascontiguousarray(rgb, dtype=np.float32)
        if track_lengths is None or min_track <= 0:
            return xyz_arr, rgb_arr
        track_arr = np.asarray(track_lengths, dtype=np.int32).reshape(-1)
        mask = track_arr >= min_track
        return np.ascontiguousarray(xyz_arr[mask], dtype=np.float32), np.ascontiguousarray(rgb_arr[mask], dtype=np.float32)
    if not recon.points3d:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    points = tuple(point for point in recon.points3d.values() if int(getattr(point, "track_length", 0)) >= min_track)
    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return (
        np.ascontiguousarray(np.stack([point.xyz for point in points], axis=0), dtype=np.float32),
        np.ascontiguousarray(np.stack([point.rgb for point in points], axis=0), dtype=np.float32),
    )
