from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image
import slangpy as spy

from ..common import clamp_float, clamp_int
from ..scene import GaussianInitHyperParams, GaussianScene
from ..training import AdamHyperParams, StabilityHyperParams, TrainingHyperParams, resolve_training_profile

EPS = 1e-8
MIN_SCENE_RADIUS = 1.0
SCENE_CORE_QUANTILE = 0.7
SCENE_CORE_LIMIT = 2048
CAMERA_DISTANCE_SCALE = 1.35
CAMERA_NEAR_RATIO = 0.0015
CAMERA_FAR_RADIUS_SCALE = 4.0
CAMERA_MIN_FAR = 80.0
MOVE_SPEED_RADIUS_SCALE = 0.15
MOVE_SPEED_MIN = 0.25
GAUSSIAN_SIGMA_SUPPORT = 3.0
_LR_LIMITS = (0.1, 10.0)
_CLAMP_LIMITS = {
    "grad_component_clip": (1e-5, 1e6),
    "grad_norm_clip": (1e-5, 1e6),
    "max_update": (1e-8, 10.0),
    "min_scale": (1e-8, 1e3),
    "max_scale": (1e-8, 1e4),
    "max_anisotropy": (1.0, 1e4),
    "min_opacity": (0.0, 1.0),
    "max_opacity": (0.0, 1.0),
    "position_abs_max": (1e-3, 1e9),
    "loss_grad_clip": (1e-5, 1e6),
}


@dataclass(frozen=True, slots=True)
class RendererParams:
    radius_scale: float = 1.0; alpha_cutoff: float = 1.0 / 255.0
    max_anisotropy: float = 32.0
    transmittance_threshold: float = 0.005; list_capacity_multiplier: int = 64
    max_prepass_memory_mb: int = 4096; cached_raster_grad_atomic_mode: str = "fixed"; cached_raster_grad_fixed_ro_local_range: float = 0.01; cached_raster_grad_fixed_scale_range: float = 0.01
    cached_raster_grad_fixed_quat_range: float = 0.01; cached_raster_grad_fixed_color_range: float = 0.2; cached_raster_grad_fixed_opacity_range: float = 0.2
    debug_mode: str | None = None; debug_grad_norm_threshold: float = 2e-4; debug_ellipse_thickness_px: float = 2.0
    debug_clone_count_range: tuple[float, float] = (0.0, 16.0); debug_density_range: tuple[float, float] = (0.0, 20.0); debug_depth_mean_range: tuple[float, float] = (0.0, 10.0); debug_depth_std_range: tuple[float, float] = (0.0, 0.5)
    debug_show_ellipses: bool = False; debug_show_processed_count: bool = False; debug_show_grad_norm: bool = False


@dataclass(frozen=True, slots=True)
class InitParams:
    hparams: GaussianInitHyperParams; seed: int


@dataclass(frozen=True, slots=True)
class AppTrainingParams:
    adam: AdamHyperParams; stability: StabilityHyperParams; training: TrainingHyperParams


@dataclass(frozen=True, slots=True)
class SceneBounds:
    center: np.ndarray; radius: float


@dataclass(frozen=True, slots=True)
class CameraFit:
    position: spy.float3; near: float; far: float; move_speed: float


def _filtered_rows(values: np.ndarray) -> np.ndarray:
    rows = np.ascontiguousarray(values, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    return rows[np.isfinite(rows[:, :3]).all(axis=1), :3]


def _fit_vector(values: np.ndarray | None, size: int, fill: float = 0.0) -> np.ndarray:
    if values is None:
        return np.full((size,), fill, dtype=np.float32)
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    padded = np.pad(flat[:size], (0, max(size - flat.size, 0)), constant_values=fill)
    return padded[:size].astype(np.float32)


def _weighted_bounds(points: np.ndarray, extents: np.ndarray | None = None, weights: np.ndarray | None = None) -> SceneBounds:
    pts = _filtered_rows(points)
    if pts.shape[0] == 0:
        return SceneBounds(center=np.zeros((3,), dtype=np.float32), radius=MIN_SCENE_RADIUS)
    ext = _fit_vector(extents, pts.shape[0])
    if weights is None:
        core, core_ext, center = pts, ext, np.mean(pts, axis=0, dtype=np.float32)
    else:
        core_w = np.clip(_fit_vector(weights, pts.shape[0], 1.0), 1e-3, 1.0)
        core_mask = core_w > np.quantile(core_w, SCENE_CORE_QUANTILE)
        if np.count_nonzero(core_mask) > SCENE_CORE_LIMIT:
            pts, ext, core_w = pts[core_mask], ext[core_mask], core_w[core_mask]
        total_w = float(np.sum(core_w))
        center = (np.sum(pts * core_w[:, None], axis=0) / max(total_w, EPS) if total_w > EPS else np.mean(pts, axis=0, dtype=np.float32)).astype(np.float32)
        core, core_ext = pts, ext
    effective = np.linalg.norm(core - center[None, :], axis=1) + core_ext
    q_lo, q_hi = np.percentile(core, (5.0, 95.0), axis=0)
    radius = max(float(np.percentile(effective, 90.0)), float(0.5 * np.linalg.norm((q_hi - q_lo).astype(np.float32))), MIN_SCENE_RADIUS)
    return SceneBounds(center=center.astype(np.float32), radius=radius)


def estimate_scene_bounds(scene: GaussianScene) -> SceneBounds:
    support_extents = GAUSSIAN_SIGMA_SUPPORT * np.max(np.exp(np.asarray(scene.scales, dtype=np.float32)), axis=1)
    return _weighted_bounds(scene.positions, extents=2.0 * support_extents, weights=scene.opacities)


def estimate_point_bounds(points: np.ndarray) -> SceneBounds:
    return _weighted_bounds(points)


def fit_camera(bounds: SceneBounds, fov_y_degrees: float) -> CameraFit:
    radius = max(float(bounds.radius), MIN_SCENE_RADIUS)
    distance = max(radius / max(float(np.tan(0.5 * np.deg2rad(float(fov_y_degrees)))), 1e-4) * 0.95, radius * CAMERA_DISTANCE_SCALE, MIN_SCENE_RADIUS)
    position = bounds.center + np.array([0.0, 0.0, -distance], dtype=np.float32)
    return CameraFit(position=spy.float3(*position.tolist()), near=max(0.01, distance * CAMERA_NEAR_RATIO), far=max(distance + radius * CAMERA_FAR_RADIUS_SCALE, CAMERA_MIN_FAR), move_speed=max(MOVE_SPEED_MIN, radius * MOVE_SPEED_RADIUS_SCALE))


def build_init_params(
    position_jitter_std: float | None,
    base_scale: float | None,
    scale_jitter_ratio: float | None,
    initial_opacity: float | None,
    seed: int,
) -> InitParams:
    return InitParams(
        hparams=GaussianInitHyperParams(
            position_jitter_std=None if position_jitter_std is None else clamp_float(position_jitter_std, 0.0, 10.0),
            base_scale=None if base_scale is None else clamp_float(base_scale, 1e-8, 1e3),
            scale_jitter_ratio=None if scale_jitter_ratio is None else clamp_float(scale_jitter_ratio, 0.0, 10.0),
            initial_opacity=None if initial_opacity is None else clamp_float(initial_opacity, 0.0, 1.0),
            color_jitter_std=0.0,
        ),
        seed=clamp_int(seed, 0, 1_000_000_000),
    )


def build_training_params(
    *,
    background: tuple[float, float, float] | np.ndarray,
    base_lr: float,
    lr_pos_mul: float,
    lr_scale_mul: float,
    lr_rot_mul: float,
    lr_color_mul: float,
    lr_opacity_mul: float,
    beta1: float,
    beta2: float,
    grad_clip: float,
    grad_norm_clip: float,
    max_update: float,
    min_scale: float,
    max_scale: float,
    max_anisotropy: float,
    min_opacity: float,
    max_opacity: float,
    position_abs_max: float,
    near: float,
    far: float,
    scale_l2_weight: float,
    scale_abs_reg_weight: float,
    opacity_reg_weight: float,
    depth_ratio_weight: float = 0.05,
    density_regularizer: float = 0.05,
    max_allowed_density: float = 4.5,
    max_gaussians: int,
    random_background: bool = True,
    lr_schedule_enabled: bool = True,
    lr_schedule_start_lr: float | None = None,
    lr_schedule_end_lr: float | None = None,
    lr_schedule_steps: int = 30_000,
    maintenance_interval: int = 200,
    maintenance_growth_ratio: float = 0.02,
    maintenance_growth_start_step: int = 2_000,
    maintenance_alpha_cull_threshold: float = 1e-2,
    train_downscale_mode: int = 1,
    train_auto_start_downscale: int = 16,
    train_downscale_base_iters: int = 200,
    train_downscale_iter_step: int = 200,
    train_downscale_max_iters: int = 30_000,
    train_downscale_factor: int = 1,
) -> AppTrainingParams:
    base_lr = clamp_float(base_lr, 1e-8, 1.0)
    adam = AdamHyperParams(
        **{
            name: base_lr * clamp_float(value, *_LR_LIMITS)
            for name, value in {
                "position_lr": lr_pos_mul,
                "scale_lr": lr_scale_mul,
                "rotation_lr": lr_rot_mul,
                "color_lr": lr_color_mul,
                "opacity_lr": lr_opacity_mul,
            }.items()
        },
        beta1=clamp_float(beta1, 0.0, 0.99999),
        beta2=clamp_float(beta2, 0.0, 0.999999),
    )
    stability = StabilityHyperParams(
        **{
            name: clamp_float(value, *limits)
            for (name, limits), value in zip(
                _CLAMP_LIMITS.items(),
                (
                    grad_clip,
                    grad_norm_clip,
                    max_update,
                    min_scale,
                    max_scale,
                    max_anisotropy,
                    min_opacity,
                    max_opacity,
                    position_abs_max,
                    grad_clip,
                ),
            )
        }
    )
    training = TrainingHyperParams(
        background=tuple(float(v) for v in np.asarray(background, dtype=np.float32).reshape(3)),
        near=clamp_float(near, 1e-6, 1e4),
        far=clamp_float(far, 1e-5, 1e6),
        random_background=bool(random_background),
        scale_l2_weight=clamp_float(scale_l2_weight, 0.0, 1e4),
        scale_abs_reg_weight=clamp_float(scale_abs_reg_weight, 0.0, 1e4),
        opacity_reg_weight=clamp_float(opacity_reg_weight, 0.0, 1e4),
        depth_ratio_weight=clamp_float(depth_ratio_weight, 0.0, 1e4),
        density_regularizer=clamp_float(density_regularizer, 0.0, 1e4),
        max_allowed_density=clamp_float(max_allowed_density, 0.0, 1e6),
        lr_schedule_enabled=bool(lr_schedule_enabled),
        lr_schedule_start_lr=base_lr if lr_schedule_start_lr is None else clamp_float(lr_schedule_start_lr, 1e-8, 1.0),
        lr_schedule_end_lr=max(base_lr * 0.1, 1e-8) if lr_schedule_end_lr is None else clamp_float(lr_schedule_end_lr, 1e-8, 1.0),
        lr_schedule_steps=clamp_int(lr_schedule_steps, 1, 1_000_000_000),
        maintenance_interval=clamp_int(maintenance_interval, 1, 1_000_000_000),
        maintenance_growth_ratio=clamp_float(maintenance_growth_ratio, 0.0, 10.0),
        maintenance_growth_start_step=clamp_int(maintenance_growth_start_step, 0, 1_000_000_000),
        maintenance_alpha_cull_threshold=clamp_float(maintenance_alpha_cull_threshold, 1e-8, 1.0),
        max_gaussians=clamp_int(max_gaussians, 0, 10_000_000),
        train_downscale_mode=clamp_int(train_downscale_mode, 0, 16),
        train_auto_start_downscale=clamp_int(train_auto_start_downscale, 1, 16),
        train_downscale_base_iters=clamp_int(train_downscale_base_iters, 1, 1_000_000_000),
        train_downscale_iter_step=clamp_int(train_downscale_iter_step, 0, 1_000_000_000),
        train_downscale_max_iters=clamp_int(train_downscale_max_iters, 1, 1_000_000_000),
        train_downscale_factor=clamp_int(train_downscale_factor, 1, 16),
    )
    stability.max_scale = max(stability.max_scale, stability.min_scale)
    stability.max_opacity = max(stability.max_opacity, stability.min_opacity)
    if training.far <= training.near:
        training.far = training.near + 1e-3
    return AppTrainingParams(adam=adam, stability=stability, training=training)


def apply_training_profile(
    params: AppTrainingParams,
    profile_name: str | None,
    *,
    dataset_root: Path | None = None,
    images_subdir: str | None = None,
) -> tuple[AppTrainingParams, object]:
    profile = resolve_training_profile(profile_name, dataset_root=dataset_root, images_subdir=images_subdir)
    return (
        AppTrainingParams(
            adam=replace(params.adam, **profile.adam_overrides),
            stability=replace(params.stability, **profile.stability_overrides),
            training=replace(params.training, **profile.training_overrides),
        ),
        profile,
    )


def renderer_kwargs(params: RendererParams) -> dict[str, object]:
    kwargs = {name: getattr(params, name) for name in RendererParams.__dataclass_fields__}
    if kwargs.get("debug_mode") is None:
        del kwargs["debug_mode"]
    return kwargs


def save_snapshot(path: Path, rgba: np.ndarray, flip_y: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(np.asarray(rgba, dtype=np.float32)[:, :, :3], 0.0, 1.0)
    Image.fromarray((255.0 * (np.flipud(rgb) if flip_y else rgb) + 0.5).astype(np.uint8), mode="RGB").save(path)
