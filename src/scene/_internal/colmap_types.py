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
    position_jitter_std: float | None = None
    base_scale: float | None = None
    scale_jitter_ratio: float | None = None
    initial_opacity: float | None = None
    color_jitter_std: float | None = None


def point_tables(recon: ColmapReconstruction) -> tuple[np.ndarray, np.ndarray]:
    xyz = getattr(recon, "point_xyz_table", None)
    rgb = getattr(recon, "point_rgb_table", None)
    if xyz is not None and rgb is not None:
        return np.ascontiguousarray(xyz, dtype=np.float32), np.ascontiguousarray(rgb, dtype=np.float32)
    if not recon.points3d:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    points = tuple(recon.points3d.values())
    return (
        np.ascontiguousarray(np.stack([point.xyz for point in points], axis=0), dtype=np.float32),
        np.ascontiguousarray(np.stack([point.rgb for point in points], axis=0), dtype=np.float32),
    )
