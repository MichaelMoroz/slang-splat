from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...renderer.camera import PROJECTION_MODEL_EQUIRECTANGULAR, PROJECTION_MODEL_FISHEYE, PROJECTION_MODEL_PINHOLE, Camera

COLMAP_SIMPLE_PINHOLE_MODEL_ID = 0
COLMAP_PINHOLE_MODEL_ID = 1
COLMAP_SIMPLE_RADIAL_MODEL_ID = 2
COLMAP_RADIAL_MODEL_ID = 3
COLMAP_OPENCV_MODEL_ID = 4
COLMAP_OPENCV_FISHEYE_MODEL_ID = 5
COLMAP_FULL_OPENCV_MODEL_ID = 6
COLMAP_SIMPLE_RADIAL_FISHEYE_MODEL_ID = 8
COLMAP_RADIAL_FISHEYE_MODEL_ID = 9
COLMAP_SIMPLE_FISHEYE_MODEL_ID = 14
COLMAP_FISHEYE_MODEL_ID = 15
COLMAP_EQUIRECTANGULAR_MODEL_ID = 17
COLMAP_CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": COLMAP_SIMPLE_PINHOLE_MODEL_ID,
    "PINHOLE": COLMAP_PINHOLE_MODEL_ID,
    "SIMPLE_RADIAL": COLMAP_SIMPLE_RADIAL_MODEL_ID,
    "RADIAL": COLMAP_RADIAL_MODEL_ID,
    "OPENCV": COLMAP_OPENCV_MODEL_ID,
    "OPENCV_FISHEYE": COLMAP_OPENCV_FISHEYE_MODEL_ID,
    "FULL_OPENCV": COLMAP_FULL_OPENCV_MODEL_ID,
    "SIMPLE_RADIAL_FISHEYE": COLMAP_SIMPLE_RADIAL_FISHEYE_MODEL_ID,
    "RADIAL_FISHEYE": COLMAP_RADIAL_FISHEYE_MODEL_ID,
    "SIMPLE_FISHEYE": COLMAP_SIMPLE_FISHEYE_MODEL_ID,
    "FISHEYE": COLMAP_FISHEYE_MODEL_ID,
    "EQUIRECTANGULAR": COLMAP_EQUIRECTANGULAR_MODEL_ID,
}
COLMAP_CAMERA_MODEL_NAMES = {model_id: name for name, model_id in COLMAP_CAMERA_MODEL_IDS.items()}
COLMAP_CAMERA_MODEL_PARAM_COUNTS = {
    COLMAP_SIMPLE_PINHOLE_MODEL_ID: 3,
    COLMAP_PINHOLE_MODEL_ID: 4,
    COLMAP_SIMPLE_RADIAL_MODEL_ID: 4,
    COLMAP_RADIAL_MODEL_ID: 5,
    COLMAP_OPENCV_MODEL_ID: 8,
    COLMAP_OPENCV_FISHEYE_MODEL_ID: 8,
    COLMAP_FULL_OPENCV_MODEL_ID: 12,
    COLMAP_SIMPLE_RADIAL_FISHEYE_MODEL_ID: 4,
    COLMAP_RADIAL_FISHEYE_MODEL_ID: 5,
    COLMAP_SIMPLE_FISHEYE_MODEL_ID: 3,
    COLMAP_FISHEYE_MODEL_ID: 4,
    COLMAP_EQUIRECTANGULAR_MODEL_ID: 2,
}
COLMAP_FISHEYE_MODEL_IDS = (
    COLMAP_OPENCV_FISHEYE_MODEL_ID,
    COLMAP_SIMPLE_RADIAL_FISHEYE_MODEL_ID,
    COLMAP_RADIAL_FISHEYE_MODEL_ID,
    COLMAP_SIMPLE_FISHEYE_MODEL_ID,
    COLMAP_FISHEYE_MODEL_ID,
)
COLMAP_SUPPORTED_CAMERA_MODEL_NAMES = tuple(COLMAP_CAMERA_MODEL_IDS)


def colmap_projection_model(model_id: int) -> int:
    if int(model_id) == COLMAP_EQUIRECTANGULAR_MODEL_ID:
        return PROJECTION_MODEL_EQUIRECTANGULAR
    if int(model_id) in COLMAP_FISHEYE_MODEL_IDS:
        return PROJECTION_MODEL_FISHEYE
    return PROJECTION_MODEL_PINHOLE


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
    camera_id: int | None = None
    model_id: int = COLMAP_PINHOLE_MODEL_ID
    # Full lens FOV in degrees for the fisheye image-circle alpha mask (0 = no mask).
    # Circular fisheyes record black pixels outside the image circle; the mask zeroes
    # target alpha there so Skip Mask / alpha-target training ignores those pixels.
    fisheye_mask_fov_degrees: float = 0.0

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
            projection_model=colmap_projection_model(self.model_id),
        )


@dataclass(slots=True)
class GaussianInitHyperParams:
    position_jitter_std: float | None = None
    base_scale: float | None = None
    scale_jitter_ratio: float | None = None
    initial_opacity: float | None = None
    color_jitter_std: float | None = None
    neighbor_anisotropy_strength: float | None = None
    neighbor_count: int | None = None


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
