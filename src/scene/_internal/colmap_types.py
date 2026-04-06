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
    k1: float = 0.0
    k2: float = 0.0

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
            near=near,
            far=far,
        )


@dataclass(slots=True)
class FrameSamplingGraph:
    frame_count: int
    neighbor_indices: tuple[np.ndarray, ...]
    neighbor_weights: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        self.frame_count = max(int(self.frame_count), 0)
        if len(self.neighbor_indices) != self.frame_count or len(self.neighbor_weights) != self.frame_count:
            raise ValueError("FrameSamplingGraph adjacency must match frame_count.")
        self.neighbor_indices = tuple(np.ascontiguousarray(np.asarray(indices, dtype=np.int32).reshape(-1)) for indices in self.neighbor_indices)
        self.neighbor_weights = tuple(np.ascontiguousarray(np.asarray(weights, dtype=np.float32).reshape(-1)) for weights in self.neighbor_weights)
        for indices, weights in zip(self.neighbor_indices, self.neighbor_weights, strict=True):
            if indices.shape != weights.shape:
                raise ValueError("FrameSamplingGraph neighbor index and weight arrays must match.")

    @classmethod
    def empty(cls, frame_count: int) -> "FrameSamplingGraph":
        count = max(int(frame_count), 0)
        empty_i = np.zeros((0,), dtype=np.int32)
        empty_w = np.zeros((0,), dtype=np.float32)
        return cls(count, tuple(empty_i.copy() for _ in range(count)), tuple(empty_w.copy() for _ in range(count)))


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
