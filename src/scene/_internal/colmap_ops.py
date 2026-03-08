from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..gaussian_scene import GaussianScene
from .colmap_types import ColmapFrame, ColmapReconstruction, GaussianInitHyperParams, point_tables

DEFAULT_RENDER_RADIUS_SCALE = 2.6
INIT_TARGET_RADIUS_RATIO = 0.20
INIT_BASE_SCALE_SPACING_RATIO = INIT_TARGET_RADIUS_RATIO / DEFAULT_RENDER_RADIUS_SCALE
INIT_JITTER_SPACING_RATIO = 1.0 / np.sqrt(12.0)
INIT_REPLACEMENT_JITTER_BOOST = 1.5
INIT_SCALE_JITTER_BASE = 0.03
INIT_SCALE_JITTER_VARIABILITY = 0.10
INIT_SCALE_JITTER_MIN = 0.01
INIT_SCALE_JITTER_MAX = 0.16
INIT_OPACITY_BASE = 0.22
INIT_OPACITY_MIN = 0.10
INIT_OPACITY_MAX = 0.35
_colmap_point_positions = lambda recon: (lambda points: points[np.isfinite(points).all(axis=1)])(point_tables(recon)[0])
_subsample_points = lambda points, max_points: points if points.shape[0] <= max_points else points[np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)]


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
        base_scale=float(np.clip(INIT_BASE_SCALE_SPACING_RATIO * target_spacing, 1e-4, 10.0)),
        scale_jitter_ratio=float(np.clip(INIT_SCALE_JITTER_BASE + INIT_SCALE_JITTER_VARIABILITY * variability, INIT_SCALE_JITTER_MIN, INIT_SCALE_JITTER_MAX)),
        initial_opacity=float(np.clip(INIT_OPACITY_BASE * np.sqrt(density_scale), INIT_OPACITY_MIN, INIT_OPACITY_MAX)),
        color_jitter_std=0.0,
    )


def resolve_colmap_init_hparams(recon: ColmapReconstruction, max_gaussians: int, init_hparams: GaussianInitHyperParams | None = None) -> GaussianInitHyperParams:
    suggested = suggest_colmap_init_hparams(recon, max_gaussians)
    if init_hparams is None:
        return suggested
    return GaussianInitHyperParams(**{name: getattr(suggested, name) if getattr(init_hparams, name) is None else float(getattr(init_hparams, name)) for name in ("position_jitter_std", "base_scale", "scale_jitter_ratio", "initial_opacity", "color_jitter_std")})


def build_training_frames(recon: ColmapReconstruction, images_subdir: str = "images_4") -> list[ColmapFrame]:
    images_root = (recon.root / images_subdir).resolve()
    if not images_root.exists():
        raise FileNotFoundError(f"COLMAP image directory does not exist: {images_root}")
    frames = []
    for image_id, image in sorted(recon.images.items()):
        image_path, camera = (images_root / image.name).resolve(), recon.cameras.get(image.camera_id)
        if camera is None or not image_path.exists(): continue
        with Image.open(image_path) as pil_image: width, height = pil_image.size
        sx, sy = float(width) / float(camera.width), float(height) / float(camera.height)
        frames.append(ColmapFrame(image_id, image_path, image.q_wxyz.astype(np.float32), image.t_xyz.astype(np.float32), float(camera.fx) * sx, float(camera.fy) * sy, float(camera.cx) * sx, float(camera.cy) * sy, int(width), int(height)))
    if not frames:
        raise RuntimeError(f"No training frames were found in {images_root}.")
    return frames


_random_unit_quaternions = lambda rng, count: (lambda q: q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8))(rng.normal(size=(count, 4)).astype(np.float32))


def initialize_scene_from_colmap_points(recon: ColmapReconstruction, max_gaussians: int, seed: int, init_hparams: GaussianInitHyperParams | None = None) -> GaussianScene:
    params = resolve_colmap_init_hparams(recon, max_gaussians, init_hparams)
    xyz, rgb = point_tables(recon)
    if xyz.shape[0] == 0:
        raise RuntimeError("COLMAP reconstruction has no 3D points.")
    rng = np.random.default_rng(int(seed))
    chosen_count = xyz.shape[0] if max_gaussians <= 0 else max(int(max_gaussians), 1)
    indices = rng.choice(xyz.shape[0], size=chosen_count, replace=chosen_count > xyz.shape[0])
    positions, colors = xyz[indices].copy(), rgb[indices].copy()
    if float(params.position_jitter_std) > 0.0:
        positions += rng.normal(scale=float(params.position_jitter_std), size=positions.shape).astype(np.float32)
    if float(params.color_jitter_std) > 0.0:
        colors = np.clip(colors + rng.normal(scale=float(params.color_jitter_std), size=colors.shape).astype(np.float32), 0.0, 1.0)
    base_scale = max(float(params.base_scale), 1e-4)
    scales = np.full((chosen_count, 3), base_scale, dtype=np.float32)
    if float(params.scale_jitter_ratio) > 0.0:
        scales = np.clip(
            scales * rng.uniform(1.0 - float(params.scale_jitter_ratio), 1.0 + float(params.scale_jitter_ratio), size=(chosen_count, 3)).astype(np.float32),
            1e-4,
            10.0,
        )
    return GaussianScene(
        positions=positions,
        scales=scales,
        rotations=_random_unit_quaternions(rng, chosen_count),
        opacities=np.full((chosen_count,), float(np.clip(params.initial_opacity, 1e-4, 0.9999)), dtype=np.float32),
        colors=colors,
        sh_coeffs=np.zeros((chosen_count, 1, 3), dtype=np.float32),
    )
