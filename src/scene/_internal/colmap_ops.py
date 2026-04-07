from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ..gaussian_scene import GaussianScene
from ..sh_utils import SUPPORTED_SH_COEFF_COUNT, rgb_to_sh0
from .colmap_types import ColmapCamera, ColmapFrame, ColmapImage, ColmapReconstruction, GaussianInitHyperParams, point_tables

INIT_BASE_SCALE_SPACING_RATIO = 0.25
INIT_JITTER_SPACING_RATIO = 1.0 / np.sqrt(12.0)
INIT_REPLACEMENT_JITTER_BOOST = 1.5
INIT_SCALE_JITTER_BASE = 0.03
INIT_SCALE_JITTER_VARIABILITY = 0.10
INIT_SCALE_JITTER_MIN = 0.01
INIT_SCALE_JITTER_MAX = 0.16
INIT_OPACITY_BASE = 0.22
INIT_OPACITY_MIN = 0.10
INIT_OPACITY_MAX = 0.35
_MIN_SCALE = 1e-4
_MAX_SCALE = 1e4
TRAINING_FRAME_LOAD_THREADS = 16
DEPTH_INIT_MAX_CORRESPONDENCES = 128
DEPTH_INIT_CLOSEST_CORRESPONDENCES = 32
DEPTH_INIT_MIN_CORRESPONDENCES = 16
DEPTH_INIT_MIN_INLIERS = 12
DEPTH_INIT_RIDGE_LAMBDA = 1e-4
DEPTH_INIT_MAD_SCALE = 3.5
DEPTH_INIT_DISTANCE_FLOOR = 1e-4
DEPTH_INIT_VALUE_DISTANCE = "distance"
DEPTH_INIT_VALUE_Z_DEPTH = "z_depth"


@dataclass(slots=True)
class DepthInitFramePayload:
    frame: ColmapFrame
    rgba8: np.ndarray
    depth_map: np.ndarray
    camera_id: int
    fit_features: np.ndarray
    fit_targets: np.ndarray


def _camera_to_world_pose(q_wxyz: np.ndarray, t_xyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float32).reshape(4)
    t = np.asarray(t_xyz, dtype=np.float32).reshape(3)
    rot_wc = Rotation.from_quat(np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)).as_matrix().astype(np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rot_wc.T
    pose[:3, 3] = -(rot_wc.T @ t)
    return pose


def _orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(rotation, dtype=np.float64).reshape(3, 3), full_matrices=False)
    ortho = u @ vh
    if np.linalg.det(ortho) < 0.0:
        u[:, -1] *= -1.0
        ortho = u @ vh
    return ortho.astype(np.float32)


def _world_to_camera_from_pose(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose_arr = np.asarray(pose, dtype=np.float32).reshape(4, 4)
    rot_cw = _orthonormalize_rotation(pose_arr[:3, :3])
    center = pose_arr[:3, 3].astype(np.float32, copy=False)
    rot_wc = rot_cw.T
    t_xyz = -(rot_wc @ center)
    return rot_wc.astype(np.float32, copy=False), t_xyz.astype(np.float32, copy=False)


def _rotation_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(np.asarray(rotation, dtype=np.float64).reshape(3, 3)).as_quat().astype(np.float32)
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def rescale_poses_to_unit_cube(poses: np.ndarray, transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(poses, dtype=np.float32)[:, :3, 3]
    max_extent = float(np.max(np.abs(positions))) if positions.size > 0 else 0.0
    if not np.isfinite(max_extent) or max_extent <= 1e-8:
        return poses.astype(np.float32, copy=False), transform.astype(np.float32, copy=False)
    scale = np.float32(1.0 / max_extent)
    scale_transform = np.eye(4, dtype=np.float32)
    scale_transform[:3, :3] *= scale
    return scale_transform @ poses, scale_transform @ transform


def transform_poses_pca(poses: np.ndarray, rescale: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Aligns the scene by assuming that most movement happened parallel to the ground plane during capture.
    Adapted from Zip-NeRF (https://github.com/jonbarron/camp_zipnerf)
    """
    poses_arr = np.asarray(poses, dtype=np.float32).reshape(-1, 4, 4).copy()
    if poses_arr.shape[0] == 0:
        return poses_arr, np.eye(4, dtype=np.float32)

    colmap2opengl = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    poses_arr = poses_arr @ colmap2opengl

    positions = poses_arr[:, :3, 3]
    mean_position = np.mean(positions, axis=0, dtype=np.float32)
    displacements = positions - mean_position
    cov = displacements.T @ displacements
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
    eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]].astype(np.float32)

    rotation = eigvecs.T.astype(np.float32)
    if np.linalg.det(rotation) < 0.0:
        rotation = np.diag([1.0, 1.0, -1.0]).astype(np.float32) @ rotation

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = rotation @ (-mean_position)
    poses_arr = transform @ poses_arr

    if float(np.mean(poses_arr[:, 2, 1], dtype=np.float32)) < 0.0:
        flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        poses_arr = flip @ poses_arr
        transform = flip @ transform

    if rescale:
        poses_arr, transform = rescale_poses_to_unit_cube(poses_arr, transform)

    aligned2colmap = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    poses_arr = aligned2colmap @ poses_arr
    transform = aligned2colmap @ transform
    poses_arr = poses_arr @ np.linalg.inv(colmap2opengl).astype(np.float32)
    return poses_arr.astype(np.float32, copy=False), transform.astype(np.float32, copy=False)


def transform_colmap_reconstruction_pca(recon: ColmapReconstruction, rescale: bool = False) -> tuple[ColmapReconstruction, np.ndarray]:
    image_items = sorted(recon.images.items())
    if not image_items:
        return recon, np.eye(4, dtype=np.float32)

    poses = np.stack([_camera_to_world_pose(image.q_wxyz, image.t_xyz) for _, image in image_items], axis=0)
    transformed_poses, transform = transform_poses_pca(poses, rescale=rescale)

    transformed_images = {
        image_id: replace(
            image,
            q_wxyz=_rotation_to_quaternion_wxyz(_world_to_camera_from_pose(transformed_pose)[0]),
            t_xyz=_world_to_camera_from_pose(transformed_pose)[1],
        )
        for (image_id, image), transformed_pose in zip(image_items, transformed_poses, strict=False)
    }

    transformed_points = {}
    for point_id, point in recon.points3d.items():
        xyz_h = np.concatenate((np.asarray(point.xyz, dtype=np.float32), np.array([1.0], dtype=np.float32)))
        transformed_xyz = (transform @ xyz_h)[:3].astype(np.float32, copy=False)
        transformed_points[point_id] = replace(point, xyz=transformed_xyz)

    transformed_recon = ColmapReconstruction(
        root=recon.root,
        sparse_dir=recon.sparse_dir,
        cameras=recon.cameras,
        images=transformed_images,
        points3d=transformed_points,
    )
    return transformed_recon, transform


def _colmap_point_positions(recon: ColmapReconstruction) -> np.ndarray:
    points = point_tables(recon)[0]
    return points[np.isfinite(points).all(axis=1)]


def _subsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    return points if points.shape[0] <= max_points else points[np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)]


def _estimate_point_spacing(points: np.ndarray) -> tuple[float, float]:
    if points.shape[0] <= 1:
        return 1.0, 0.15
    sample = _subsample_points(points, 2048)
    dist2 = np.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=2, dtype=np.float32)
    np.fill_diagonal(dist2, np.inf)
    nearest = np.sqrt(np.min(dist2, axis=1))
    nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
    if nearest.size == 0:
        return 1.0, 0.15
    spacing = float(np.median(nearest))
    variability = (float(np.percentile(nearest, 75.0)) - float(np.percentile(nearest, 25.0))) / max(spacing, 1e-6)
    return max(spacing, 1e-4), float(np.clip(variability, 0.0, 1.0))


def suggest_colmap_init_hparams(recon: ColmapReconstruction, max_gaussians: int) -> GaussianInitHyperParams:
    points = _colmap_point_positions(recon)
    if points.shape[0] == 0:
        raise RuntimeError("COLMAP reconstruction has no finite 3D points.")
    point_count = int(points.shape[0])
    chosen_count = point_count if max_gaussians <= 0 else max(int(max_gaussians), 1)
    spacing, variability = _estimate_point_spacing(points)
    density_scale = float((point_count / max(chosen_count, 1)) ** (1.0 / 3.0))
    target_spacing = max(spacing * density_scale, 1e-4)
    replacement_factor = INIT_REPLACEMENT_JITTER_BOOST if chosen_count > point_count else 1.0
    return GaussianInitHyperParams(
        position_jitter_std=float(np.clip(INIT_JITTER_SPACING_RATIO * target_spacing * replacement_factor, 0.0, 10.0)),
        base_scale=float(np.clip(INIT_BASE_SCALE_SPACING_RATIO * target_spacing, _MIN_SCALE, 10.0)),
        scale_jitter_ratio=float(np.clip(INIT_SCALE_JITTER_BASE + INIT_SCALE_JITTER_VARIABILITY * variability, INIT_SCALE_JITTER_MIN, INIT_SCALE_JITTER_MAX)),
        initial_opacity=float(np.clip(INIT_OPACITY_BASE * np.sqrt(density_scale), INIT_OPACITY_MIN, INIT_OPACITY_MAX)),
        color_jitter_std=0.0,
    )


def resolve_colmap_init_hparams(recon: ColmapReconstruction, max_gaussians: int, init_hparams: GaussianInitHyperParams | None = None) -> GaussianInitHyperParams:
    suggested = suggest_colmap_init_hparams(recon, max_gaussians)
    if init_hparams is None:
        return suggested
    return GaussianInitHyperParams(**{name: getattr(suggested, name) if getattr(init_hparams, name) is None else float(getattr(init_hparams, name)) for name in ("position_jitter_std", "base_scale", "scale_jitter_ratio", "initial_opacity", "color_jitter_std")})


def resolve_training_frame_image_size(
    width: int,
    height: int,
    *,
    downscale_mode: str = "original",
    downscale_max_size: int | None = None,
    downscale_scale: float = 1.0,
) -> tuple[int, int]:
    src_width = max(int(width), 1)
    src_height = max(int(height), 1)
    mode = str(downscale_mode).strip().lower()
    if mode == "original":
        return src_width, src_height
    if mode == "max_size":
        if downscale_max_size is None:
            raise ValueError("downscale_max_size is required when downscale_mode='max_size'.")
        src_max_size = max(src_width, src_height)
        target_max_size = min(max(int(downscale_max_size), 1), src_max_size)
        if target_max_size >= src_max_size:
            return src_width, src_height
        scale = float(target_max_size) / float(src_max_size)
        target_width = max(1, min(src_width, int(round(src_width * scale))))
        target_height = max(1, min(src_height, int(round(src_height * scale))))
        return target_width, target_height
    if mode == "scale":
        factor = float(np.clip(downscale_scale, 1e-6, 1.0))
        if factor >= 1.0:
            return src_width, src_height
        target_width = max(1, min(src_width, int(round(src_width * factor))))
        target_height = max(1, min(src_height, int(round(src_height * factor))))
        return target_width, target_height
    raise ValueError(f"Unsupported image downscale mode: {downscale_mode}")


def _build_training_frame(task: tuple[int, object, object, Path, str, int | None, float]) -> ColmapFrame:
    image_id, image, camera, image_path, downscale_mode, downscale_max_size, downscale_scale = task
    with Image.open(image_path) as pil_image:
        src_width, src_height = pil_image.size
    width, height = resolve_training_frame_image_size(
        src_width,
        src_height,
        downscale_mode=downscale_mode,
        downscale_max_size=downscale_max_size,
        downscale_scale=downscale_scale,
    )
    sx, sy = float(width) / float(camera.width), float(height) / float(camera.height)
    return ColmapFrame(
        image_id,
        image_path,
        image.q_wxyz.astype(np.float32),
        image.t_xyz.astype(np.float32),
        float(camera.fx) * sx,
        float(camera.fy) * sy,
        float(camera.cx) * sx,
        float(camera.cy) * sy,
        int(width),
        int(height),
        float(getattr(camera, "k1", 0.0)),
        float(getattr(camera, "k2", 0.0)),
    )


def build_training_frames_from_root(
    recon: ColmapReconstruction,
    images_root: Path,
    *,
    downscale_mode: str = "original",
    downscale_max_size: int | None = None,
    downscale_scale: float = 1.0,
) -> list[ColmapFrame]:
    images_root = Path(images_root).resolve()
    if not images_root.exists():
        raise FileNotFoundError(f"COLMAP image directory does not exist: {images_root}")
    tasks = []
    for image_id, image in sorted(recon.images.items()):
        image_path, camera = (images_root / image.name).resolve(), recon.cameras.get(image.camera_id)
        if camera is None or not image_path.exists(): continue
        tasks.append((image_id, image, camera, image_path, downscale_mode, downscale_max_size, downscale_scale))
    frames: list[ColmapFrame] = []
    if tasks:
        with ThreadPoolExecutor(max_workers=TRAINING_FRAME_LOAD_THREADS, thread_name_prefix="colmap-frame") as executor:
            frames = list(executor.map(_build_training_frame, tasks))
    if not frames:
        raise RuntimeError(f"No training frames were found in {images_root}.")
    return frames


def build_training_frames(recon: ColmapReconstruction, images_subdir: str = "images_4") -> list[ColmapFrame]:
    return build_training_frames_from_root(recon, recon.root / images_subdir)


def load_rgba8_image(image_path: Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(Path(image_path).resolve()) as pil_image:
        rgba_image = pil_image.convert("RGBA")
        if target_size is not None:
            resolved_size = (max(int(target_size[0]), 1), max(int(target_size[1]), 1))
            if rgba_image.size != resolved_size:
                rgba_image = rgba_image.resize(resolved_size, Image.Resampling.LANCZOS)
        return np.array(rgba_image, dtype=np.uint8, order="C", copy=True)


def load_training_frame_rgba8(frame: ColmapFrame) -> np.ndarray:
    return load_rgba8_image(frame.image_path, target_size=(max(int(frame.width), 1), max(int(frame.height), 1)))


def build_depth_path_index(depth_root: Path) -> dict[str, Path]:
    root = Path(depth_root).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Depth directory does not exist: {root}")
    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        key = str(relative.with_suffix("")).replace("\\", "/").lower()
        index[key] = path.resolve()
    return index


def match_depth_path(images_root: Path, image_path: Path, depth_index: dict[str, Path]) -> Path | None:
    relative = Path(image_path).resolve().relative_to(Path(images_root).resolve())
    return depth_index.get(str(relative.with_suffix("")).replace("\\", "/").lower())


def load_depth_u16_image(image_path: Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(Path(image_path).resolve()) as pil_image:
        depth_image = pil_image
        if target_size is not None:
            resolved_size = (max(int(target_size[0]), 1), max(int(target_size[1]), 1))
            if depth_image.size != resolved_size:
                depth_image = depth_image.resize(resolved_size, Image.Resampling.NEAREST)
        depth = np.array(depth_image, dtype=np.float32, order="C", copy=True)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise RuntimeError(f"Depth image must be scalar: {image_path}")
    return np.ascontiguousarray(depth, dtype=np.float32)


def _depth_sample_linear(depth_map: np.ndarray, xy: np.ndarray) -> float:
    depth = np.asarray(depth_map, dtype=np.float32)
    x, y = np.asarray(xy, dtype=np.float32).reshape(2)
    width = depth.shape[1]
    height = depth.shape[0]
    if width <= 0 or height <= 0:
        return float("nan")
    x = float(np.clip(x, 0.0, max(width - 1, 0)))
    y = float(np.clip(y, 0.0, max(height - 1, 0)))
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    tx = np.float32(x - x0)
    ty = np.float32(y - y0)
    top = np.float32(1.0 - tx) * depth[y0, x0] + tx * depth[y0, x1]
    bottom = np.float32(1.0 - tx) * depth[y1, x0] + tx * depth[y1, x1]
    return float(np.float32(1.0 - ty) * top + ty * bottom)


def _depth_affine_features(raw_depth: float) -> np.ndarray:
    return np.asarray((1.0, float(raw_depth)), dtype=np.float32)


def _ridge_affine_fit(features: np.ndarray, targets: np.ndarray, ridge_lambda: float = DEPTH_INIT_RIDGE_LAMBDA) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64).reshape(-1, 2)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    xtx = x.T @ x
    xtx[0, 0] += float(max(ridge_lambda, 0.0))
    xtx[1, 1] += float(max(ridge_lambda, 0.0))
    xty = x.T @ y
    return np.asarray(np.linalg.solve(xtx, xty), dtype=np.float32)


def _robust_ridge_fit(features: np.ndarray, targets: np.ndarray) -> np.ndarray | None:
    x = np.asarray(features, dtype=np.float32).reshape(-1, 2)
    y = np.asarray(targets, dtype=np.float32).reshape(-1)
    if x.shape[0] < DEPTH_INIT_MIN_CORRESPONDENCES or x.shape[0] != y.size:
        return None
    coeffs = _ridge_affine_fit(x, y)
    residuals = y - x @ coeffs
    center = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - center)))
    robust_sigma = max(1.4826 * mad, 1e-6)
    inliers = np.abs(residuals - center) <= DEPTH_INIT_MAD_SCALE * robust_sigma
    if int(np.count_nonzero(inliers)) >= DEPTH_INIT_MIN_INLIERS:
        refined = _ridge_affine_fit(x[inliers], y[inliers])
        if np.all(np.isfinite(refined)):
            return refined
    return coeffs if np.all(np.isfinite(coeffs)) else None
def collect_depth_distance_remap_samples(
    recon: ColmapReconstruction,
    image: ColmapImage,
    frame: ColmapFrame,
    camera: ColmapCamera,
    depth_map: np.ndarray,
    depth_value_mode: str = DEPTH_INIT_VALUE_Z_DEPTH,
) -> tuple[np.ndarray, np.ndarray]:
    del camera
    point_ids = np.asarray(image.points2d_point3d_ids, dtype=np.int64).reshape(-1)
    valid_ids = np.flatnonzero(point_ids > 0)
    if valid_ids.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    camera_obj = frame.make_camera()
    depth_mode = str(depth_value_mode).strip().lower()
    candidates: list[tuple[float, np.ndarray, float]] = []
    for point_idx in valid_ids:
        point = recon.points3d.get(int(point_ids[point_idx]))
        if point is None:
            continue
        point_camera = np.asarray(camera_obj.world_point_to_camera(point.xyz), dtype=np.float32).reshape(3)
        point_distance = float(np.linalg.norm(point_camera))
        if not np.isfinite(point_distance) or point_distance <= 0.0:
            continue
        xy, valid = camera_obj.project_world_to_screen(np.asarray(point.xyz, dtype=np.float32), int(frame.width), int(frame.height))
        if not valid:
            continue
        if float(xy[0]) < 0.0 or float(xy[1]) < 0.0 or float(xy[0]) > float(frame.width - 1) or float(xy[1]) > float(frame.height - 1):
            continue
        raw_depth = _depth_sample_linear(depth_map, xy)
        if not np.isfinite(raw_depth) or raw_depth <= 0.0:
            continue
        target = float(point_camera[2]) if depth_mode == DEPTH_INIT_VALUE_Z_DEPTH else point_distance
        if not np.isfinite(target) or target <= 0.0:
            continue
        candidates.append((point_distance, _depth_affine_features(raw_depth), target))
        if len(candidates) >= DEPTH_INIT_MAX_CORRESPONDENCES:
            break
    if len(candidates) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    selected = sorted(candidates, key=lambda item: item[0])[:DEPTH_INIT_CLOSEST_CORRESPONDENCES]
    return (
        np.ascontiguousarray(np.stack([item[1] for item in selected], axis=0), dtype=np.float32),
        np.ascontiguousarray(np.asarray([item[2] for item in selected], dtype=np.float32), dtype=np.float32),
    )


def fit_depth_distance_remap(
    recon: ColmapReconstruction,
    image: ColmapImage,
    frame: ColmapFrame,
    camera: ColmapCamera,
    depth_map: np.ndarray,
    depth_value_mode: str = DEPTH_INIT_VALUE_Z_DEPTH,
) -> np.ndarray | None:
    features, targets = collect_depth_distance_remap_samples(recon, image, frame, camera, depth_map, depth_value_mode)
    return _robust_ridge_fit(features, targets)


def fit_depth_distance_remap_for_payload(payload: DepthInitFramePayload) -> np.ndarray | None:
    if payload.fit_features.size == 0 or payload.fit_targets.size == 0:
        return None
    coeffs = _robust_ridge_fit(
        np.asarray(payload.fit_features, dtype=np.float32).reshape(-1, 2),
        np.asarray(payload.fit_targets, dtype=np.float32).reshape(-1),
    )
    return None if coeffs is None else np.asarray(coeffs, dtype=np.float32)


def _predict_depth_distance_map(frame: ColmapFrame, depth_map: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    raw_depth = np.asarray(depth_map, dtype=np.float32)
    del frame
    return np.ascontiguousarray(np.float32(coeffs[0]) + np.float32(coeffs[1]) * raw_depth, dtype=np.float32)


def build_depth_init_frame_payload(
    recon: ColmapReconstruction,
    image: ColmapImage,
    camera: ColmapCamera,
    frame: ColmapFrame,
    depth_path: Path,
    depth_value_mode: str = DEPTH_INIT_VALUE_Z_DEPTH,
) -> DepthInitFramePayload | None:
    rgba8 = load_training_frame_rgba8(frame)
    depth_map = load_depth_u16_image(depth_path, target_size=(int(frame.width), int(frame.height)))
    if not np.any(np.isfinite(depth_map) & (depth_map > 0.0)):
        return None
    fit_features, fit_targets = collect_depth_distance_remap_samples(recon, image, frame, camera, depth_map, depth_value_mode)
    return DepthInitFramePayload(
        frame=frame,
        rgba8=rgba8,
        depth_map=depth_map,
        camera_id=int(image.camera_id),
        fit_features=np.asarray(fit_features, dtype=np.float32),
        fit_targets=np.asarray(fit_targets, dtype=np.float32),
    )


def load_training_frame_rgba8_with_depth_payload(task: tuple[ColmapReconstruction, ColmapImage, ColmapCamera, ColmapFrame, Path | None, str]) -> tuple[np.ndarray, DepthInitFramePayload | None]:
    recon, image, camera, frame, depth_path, depth_value_mode = task
    rgba8 = load_training_frame_rgba8(frame)
    if depth_path is None:
        return rgba8, None
    depth_map = load_depth_u16_image(depth_path, target_size=(int(frame.width), int(frame.height)))
    if not np.any(np.isfinite(depth_map) & (depth_map > 0.0)):
        return rgba8, None
    fit_features, fit_targets = collect_depth_distance_remap_samples(recon, image, frame, camera, depth_map, depth_value_mode)
    return rgba8, DepthInitFramePayload(
        frame=frame,
        rgba8=rgba8,
        depth_map=depth_map,
        camera_id=int(image.camera_id),
        fit_features=np.asarray(fit_features, dtype=np.float32),
        fit_targets=np.asarray(fit_targets, dtype=np.float32),
    )


def generate_depth_init_points(
    payloads: list[DepthInitFramePayload],
    total_point_count: int,
    seed: int,
    depth_value_mode: str = DEPTH_INIT_VALUE_Z_DEPTH,
) -> tuple[np.ndarray, np.ndarray]:
    usable = [payload for payload in payloads if payload is not None]
    if len(usable) == 0:
        raise RuntimeError("Depth initialization found no usable RGB/depth pairs.")
    depth_mode = str(depth_value_mode).strip().lower()
    calibrated_payloads: list[tuple[DepthInitFramePayload, np.ndarray, np.ndarray, int]] = []
    for payload in usable:
        coeffs = fit_depth_distance_remap_for_payload(payload)
        if coeffs is None:
            continue
        predicted = _predict_depth_distance_map(payload.frame, payload.depth_map, coeffs)
        valid_mask = np.isfinite(payload.depth_map) & (payload.depth_map > 0.0) & np.isfinite(predicted) & (predicted > DEPTH_INIT_DISTANCE_FLOOR)
        valid_count = int(np.count_nonzero(valid_mask))
        if valid_count <= 0:
            continue
        calibrated_payloads.append((payload, predicted, np.asarray(valid_mask, dtype=bool), valid_count))
    if len(calibrated_payloads) == 0:
        raise RuntimeError("Depth initialization found no calibrated RGB/depth pairs after per-pose fitting.")
    total_budget = max(int(total_point_count), 1)
    valid_counts = np.asarray([valid_count for _, _, _, valid_count in calibrated_payloads], dtype=np.int64)
    total_valid = int(np.sum(valid_counts))
    if total_valid <= 0:
        raise RuntimeError("Depth initialization found no valid calibrated depth pixels.")
    raw_alloc = np.asarray(valid_counts, dtype=np.float64) * (float(total_budget) / float(total_valid))
    counts = np.minimum(np.floor(raw_alloc).astype(np.int64), valid_counts)
    remainder = min(total_budget - int(np.sum(counts)), int(np.sum(valid_counts - counts)))
    if remainder > 0:
        order = np.argsort(-(raw_alloc - counts))
        for index in order:
            if remainder <= 0:
                break
            if counts[index] >= valid_counts[index]:
                continue
            counts[index] += 1
            remainder -= 1
    if int(np.sum(counts)) <= 0:
        counts[np.argmax(valid_counts)] = 1
    rng = np.random.default_rng(int(seed))
    positions: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    for (payload, predicted, valid_mask, _), sample_count in zip(calibrated_payloads, counts, strict=False):
        count = int(sample_count)
        if count <= 0:
            continue
        valid_indices = np.flatnonzero(valid_mask.reshape(-1))
        if valid_indices.size == 0:
            continue
        chosen = rng.choice(valid_indices, size=min(count, valid_indices.size), replace=False)
        ys, xs = np.unravel_index(chosen, valid_mask.shape)
        camera = payload.frame.make_camera()
        for x, y in zip(xs.astype(np.int32), ys.astype(np.int32), strict=False):
            depth = max(float(predicted[int(y), int(x)]), DEPTH_INIT_DISTANCE_FLOOR)
            screen_pos = np.array([float(x) + 0.5, float(y) + 0.5], dtype=np.float32)
            if depth_mode == DEPTH_INIT_VALUE_Z_DEPTH:
                world = camera.screen_to_world(screen_pos, depth, int(payload.frame.width), int(payload.frame.height))
            else:
                ray = camera.screen_to_world_ray(screen_pos, int(payload.frame.width), int(payload.frame.height))
                world = np.asarray(camera.position, dtype=np.float32) + np.asarray(ray, dtype=np.float32) * np.float32(depth)
            positions.append(np.asarray(world, dtype=np.float32).reshape(3))
            colors.append(np.asarray(payload.rgba8[int(y), int(x), :3], dtype=np.float32) / np.float32(255.0))
    if len(positions) == 0:
        raise RuntimeError("Depth initialization could not sample any calibrated points.")
    return np.ascontiguousarray(np.stack(positions, axis=0), dtype=np.float32), np.ascontiguousarray(np.stack(colors, axis=0), dtype=np.float32)


def _random_unit_quaternions(rng: np.random.Generator, count: int) -> np.ndarray:
    q = rng.normal(size=(count, 4)).astype(np.float32)
    return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)


def _identity_quaternions(count: int) -> np.ndarray:
    return np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (max(int(count), 0), 1))


def point_nn_scales(points: np.ndarray) -> np.ndarray:
    pts = np.ascontiguousarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if pts.shape[0] == 1:
        return np.full((1,), 1e-4, dtype=np.float32)
    dists, _ = cKDTree(pts).query(pts, k=2, workers=-1)
    nearest = np.asarray(dists[:, 1], dtype=np.float32)
    return np.clip(nearest, 1e-4, 1e4).astype(np.float32)


def _build_scene_from_positions_colors(
    positions: np.ndarray,
    colors: np.ndarray,
    seed: int,
    init_hparams: GaussianInitHyperParams | None = None,
) -> GaussianScene:
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    colors = np.ascontiguousarray(colors, dtype=np.float32)
    if positions.shape[0] == 0 or colors.shape[0] != positions.shape[0]:
        raise RuntimeError("Gaussian initialization requires non-empty, matched position/color tables.")
    rng = np.random.default_rng(int(seed))
    if init_hparams is not None and init_hparams.position_jitter_std is not None and float(init_hparams.position_jitter_std) > 0.0:
        positions += rng.normal(0.0, float(init_hparams.position_jitter_std), size=positions.shape).astype(np.float32)
    scales_1d = point_nn_scales(positions)
    if init_hparams is not None and init_hparams.base_scale is not None:
        median_scale = float(np.median(scales_1d)) if scales_1d.size > 0 else 1.0
        scales_1d = scales_1d * (float(max(init_hparams.base_scale, _MIN_SCALE)) / max(median_scale, 1e-6))
    scales = np.repeat(scales_1d[:, None], 3, axis=1).astype(np.float32)
    if init_hparams is not None and init_hparams.scale_jitter_ratio is not None and float(init_hparams.scale_jitter_ratio) > 0.0:
        lo = max(1.0 - float(init_hparams.scale_jitter_ratio), _MIN_SCALE)
        hi = 1.0 + float(init_hparams.scale_jitter_ratio)
        scales *= rng.uniform(lo, hi, size=scales.shape).astype(np.float32)
    scales = np.log(np.clip(scales, _MIN_SCALE, _MAX_SCALE)).astype(np.float32)
    opacity = 0.1 if init_hparams is None or init_hparams.initial_opacity is None else float(np.clip(init_hparams.initial_opacity, 1e-4, 0.9999))
    count = int(positions.shape[0])
    return GaussianScene(
        positions=positions,
        scales=scales,
        rotations=_identity_quaternions(count),
        opacities=np.full((count,), opacity, dtype=np.float32),
        colors=colors,
        sh_coeffs=np.pad(rgb_to_sh0(colors)[:, None, :], ((0, 0), (0, SUPPORTED_SH_COEFF_COUNT - 1), (0, 0))).astype(np.float32, copy=False),
    )


def sample_colmap_diffused_points(
    recon: ColmapReconstruction,
    point_count: int,
    diffusion_radius: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    xyz, rgb = point_tables(recon)
    if xyz.shape[0] == 0:
        raise RuntimeError("COLMAP reconstruction has no 3D points.")
    count = max(int(point_count), 1)
    radius = max(float(diffusion_radius), 0.0)
    rng = np.random.default_rng(int(seed))
    base_indices = rng.integers(0, xyz.shape[0], size=count, dtype=np.int64)
    positions = np.ascontiguousarray(xyz[base_indices], dtype=np.float32)
    colors = np.ascontiguousarray(rgb[base_indices], dtype=np.float32)
    if radius <= 0.0:
        return positions, colors
    original_nn = point_nn_scales(xyz)
    local_radius = np.ascontiguousarray(original_nn[base_indices] * np.float32(radius), dtype=np.float32)
    positions += rng.normal(0.0, 1.0, size=positions.shape).astype(np.float32) * local_radius[:, None]
    return positions, colors


def initialize_scene_from_colmap_points(recon: ColmapReconstruction, max_gaussians: int, seed: int, init_hparams: GaussianInitHyperParams | None = None) -> GaussianScene:
    xyz, rgb = point_tables(recon)
    if xyz.shape[0] == 0:
        raise RuntimeError("COLMAP reconstruction has no 3D points.")
    chosen_count = xyz.shape[0] if max_gaussians <= 0 else min(max(int(max_gaussians), 1), xyz.shape[0])
    return _build_scene_from_positions_colors(xyz[:chosen_count].copy(), rgb[:chosen_count].copy(), seed, init_hparams)


def initialize_scene_from_points_colors(
    positions: np.ndarray,
    colors: np.ndarray,
    seed: int,
    init_hparams: GaussianInitHyperParams | None = None,
) -> GaussianScene:
    return _build_scene_from_positions_colors(
        np.ascontiguousarray(positions, dtype=np.float32).copy(),
        np.ascontiguousarray(colors, dtype=np.float32).copy(),
        seed,
        init_hparams,
    )


def initialize_scene_from_colmap_diffused_points(
    recon: ColmapReconstruction,
    point_count: int,
    diffusion_radius: float,
    seed: int,
    init_hparams: GaussianInitHyperParams | None = None,
) -> GaussianScene:
    positions, colors = sample_colmap_diffused_points(recon, point_count, diffusion_radius, seed)
    return _build_scene_from_positions_colors(positions, colors, seed, init_hparams)
