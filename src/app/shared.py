from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image
import slangpy as spy

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
clamp_float = lambda value, lo, hi: float(np.clip(float(value), float(lo), float(hi)))
clamp_int = lambda value, lo, hi: int(np.clip(int(value), int(lo), int(hi)))


@dataclass(frozen=True, slots=True)
class RendererParams:
    radius_scale: float = 2.6; alpha_cutoff: float = 1.0 / 255.0; max_splat_steps: int = 32768
    transmittance_threshold: float = 0.005; sampled5_safety_scale: float = 1.0; list_capacity_multiplier: int = 64
    max_prepass_memory_mb: int = 4096; debug_show_ellipses: bool = False; debug_show_processed_count: bool = False; debug_show_grad_norm: bool = False


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


_filtered_rows = lambda values: (lambda rows: np.zeros((0, 3), dtype=np.float32) if rows.ndim != 2 or rows.shape[1] < 3 else rows[np.isfinite(rows[:, :3]).all(axis=1), :3])(np.ascontiguousarray(values, dtype=np.float32))
_fit_vector = lambda values, size, fill=0.0: np.full((size,), fill, dtype=np.float32) if values is None else np.pad(np.asarray(values, dtype=np.float32).reshape(-1)[:size], (0, max(size - np.asarray(values).size, 0)), constant_values=fill)[:size].astype(np.float32)


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


estimate_scene_bounds = lambda scene: _weighted_bounds(scene.positions, extents=2.0 * np.max(np.asarray(scene.scales, dtype=np.float32), axis=1), weights=scene.opacities)
estimate_point_bounds = lambda points: _weighted_bounds(points)


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
    epsilon: float,
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
    opacity_reg_weight: float,
    lambda_dssim: float,
    mcmc_position_noise_enabled: bool,
    mcmc_position_noise_scale: float,
    mcmc_opacity_gate_sharpness: float,
    mcmc_opacity_gate_center: float,
    max_gaussians: int,
    densify_from_iter: int,
    densify_until_iter: int,
    densification_interval: int,
    densify_grad_threshold: float,
    percent_dense: float,
    prune_min_opacity: float,
    screen_size_prune_threshold: float,
    world_size_prune_ratio: float,
    opacity_reset_interval: int,
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
        epsilon=clamp_float(epsilon, 1e-12, 1e-2),
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
        scale_l2_weight=clamp_float(scale_l2_weight, 0.0, 1e4),
        opacity_reg_weight=clamp_float(opacity_reg_weight, 0.0, 1e4),
        lambda_dssim=clamp_float(lambda_dssim, 0.0, 1.0),
        mcmc_position_noise_enabled=bool(mcmc_position_noise_enabled),
        mcmc_position_noise_scale=clamp_float(mcmc_position_noise_scale, 0.0, 1e8),
        mcmc_opacity_gate_sharpness=clamp_float(mcmc_opacity_gate_sharpness, 0.0, 1e6),
        mcmc_opacity_gate_center=clamp_float(mcmc_opacity_gate_center, 0.0, 1.0),
        max_gaussians=clamp_int(max_gaussians, 0, 10_000_000),
        densify_from_iter=clamp_int(densify_from_iter, 0, 10_000_000),
        densify_until_iter=clamp_int(densify_until_iter, 0, 10_000_000),
        densification_interval=clamp_int(densification_interval, 1, 10_000_000),
        densify_grad_threshold=clamp_float(densify_grad_threshold, 0.0, 1e6),
        percent_dense=clamp_float(percent_dense, 0.0, 1.0),
        prune_min_opacity=clamp_float(prune_min_opacity, 0.0, 1.0),
        screen_size_prune_threshold=clamp_float(screen_size_prune_threshold, 0.0, 1e6),
        world_size_prune_ratio=clamp_float(world_size_prune_ratio, 0.0, 1e6),
        opacity_reset_interval=clamp_int(opacity_reset_interval, 0, 10_000_000),
    )
    stability.max_scale = max(stability.max_scale, stability.min_scale)
    stability.max_opacity = max(stability.max_opacity, stability.min_opacity)
    if training.far <= training.near:
        training.far = training.near + 1e-3
    training.densify_until_iter = max(training.densify_until_iter, training.densify_from_iter)
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


renderer_kwargs = lambda params: {name: getattr(params, name) for name in RendererParams.__dataclass_fields__}


def save_snapshot(path: Path, rgba: np.ndarray, flip_y: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(np.asarray(rgba, dtype=np.float32)[:, :, :3], 0.0, 1.0)
    Image.fromarray((255.0 * (np.flipud(rgb) if flip_y else rgb) + 0.5).astype(np.uint8), mode="RGB").save(path)
