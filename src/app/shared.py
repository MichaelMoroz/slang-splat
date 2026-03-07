from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..scene import GaussianInitHyperParams, GaussianScene
from ..training import AdamHyperParams, StabilityHyperParams, TrainingHyperParams

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


def clamp_float(value: float, lo: float, hi: float) -> float:
    return float(np.clip(float(value), float(lo), float(hi)))


def clamp_int(value: int, lo: int, hi: int) -> int:
    return int(np.clip(int(value), int(lo), int(hi)))


@dataclass(frozen=True, slots=True)
class RendererParams:
    radius_scale: float = 2.6
    alpha_cutoff: float = 1.0 / 255.0
    max_splat_steps: int = 32768
    transmittance_threshold: float = 0.005
    sampled5_safety_scale: float = 1.0
    list_capacity_multiplier: int = 64
    max_prepass_memory_mb: int = 4096
    debug_show_ellipses: bool = False
    debug_show_processed_count: bool = False


@dataclass(frozen=True, slots=True)
class InitParams:
    hparams: GaussianInitHyperParams
    gaussian_count: int
    seed: int


@dataclass(frozen=True, slots=True)
class AppTrainingParams:
    adam: AdamHyperParams
    stability: StabilityHyperParams
    training: TrainingHyperParams


@dataclass(frozen=True, slots=True)
class SceneBounds:
    center: np.ndarray
    radius: float


@dataclass(frozen=True, slots=True)
class CameraFit:
    position: np.ndarray
    near: float
    far: float
    move_speed: float


def _filtered_rows(values: np.ndarray) -> np.ndarray:
    rows = np.ascontiguousarray(values, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    return rows[np.isfinite(rows[:, :3]).all(axis=1), :3]


def _weighted_bounds(points: np.ndarray, extents: np.ndarray | None = None, weights: np.ndarray | None = None) -> SceneBounds:
    pts = _filtered_rows(points)
    if pts.shape[0] == 0:
        return SceneBounds(center=np.zeros((3,), dtype=np.float32), radius=MIN_SCENE_RADIUS)
    if weights is None:
        center = np.mean(pts, axis=0, dtype=np.float32)
        core = pts
        core_extents = np.zeros((pts.shape[0],), dtype=np.float32) if extents is None else np.asarray(extents, dtype=np.float32)
    else:
        w = np.clip(np.asarray(weights, dtype=np.float32).reshape(-1), 1e-3, 1.0)
        w = w[: pts.shape[0]]
        core_mask = w > np.quantile(w, SCENE_CORE_QUANTILE)
        if np.count_nonzero(core_mask) > SCENE_CORE_LIMIT:
            core, w = pts[core_mask], w[core_mask]
            core_extents = np.zeros((core.shape[0],), dtype=np.float32) if extents is None else np.asarray(extents, dtype=np.float32)[core_mask]
        else:
            core, core_extents = pts, np.zeros((pts.shape[0],), dtype=np.float32) if extents is None else np.asarray(extents, dtype=np.float32)
        center = (
            np.sum(core * w[:, None], axis=0) / max(float(np.sum(w)), EPS)
            if np.sum(w) > EPS
            else np.mean(core, axis=0, dtype=np.float32)
        ).astype(np.float32)
    rel = core - center[None, :]
    dist = np.linalg.norm(rel, axis=1)
    effective = dist + np.asarray(core_extents, dtype=np.float32).reshape(-1)
    q_lo = np.percentile(core, 5.0, axis=0)
    q_hi = np.percentile(core, 95.0, axis=0)
    quant_extent = 0.5 * np.linalg.norm((q_hi - q_lo).astype(np.float32))
    radius = max(float(np.percentile(effective, 90.0)), float(quant_extent), MIN_SCENE_RADIUS)
    return SceneBounds(center=center.astype(np.float32), radius=radius)


def estimate_scene_bounds(scene: GaussianScene) -> SceneBounds:
    scales = np.max(np.asarray(scene.scales, dtype=np.float32), axis=1)
    return _weighted_bounds(scene.positions, extents=2.0 * scales, weights=scene.opacities)


def estimate_point_bounds(points: np.ndarray) -> SceneBounds:
    return _weighted_bounds(points)


def fit_camera(bounds: SceneBounds, fov_y_degrees: float) -> CameraFit:
    radius = max(float(bounds.radius), MIN_SCENE_RADIUS)
    fit_distance = radius / max(float(np.tan(0.5 * np.deg2rad(float(fov_y_degrees)))), 1e-4)
    distance = max(fit_distance * 0.95, radius * CAMERA_DISTANCE_SCALE, MIN_SCENE_RADIUS)
    return CameraFit(
        position=bounds.center + np.array([0.0, 0.0, -distance], dtype=np.float32),
        near=max(0.01, distance * CAMERA_NEAR_RATIO),
        far=max(distance + radius * CAMERA_FAR_RADIUS_SCALE, CAMERA_MIN_FAR),
        move_speed=max(MOVE_SPEED_MIN, radius * MOVE_SPEED_RADIUS_SCALE),
    )


def build_init_params(
    position_jitter_std: float,
    base_scale: float,
    scale_jitter_ratio: float,
    initial_opacity: float,
    gaussian_count: int,
    seed: int,
) -> InitParams:
    return InitParams(
        hparams=GaussianInitHyperParams(
            position_jitter_std=clamp_float(position_jitter_std, 0.0, 10.0),
            base_scale=clamp_float(base_scale, 1e-8, 1e3),
            scale_jitter_ratio=clamp_float(scale_jitter_ratio, 0.0, 10.0),
            initial_opacity=clamp_float(initial_opacity, 0.0, 1.0),
            color_jitter_std=0.0,
        ),
        gaussian_count=clamp_int(gaussian_count, 1, 10_000_000),
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
    mcmc_position_noise_enabled: bool,
    mcmc_position_noise_scale: float,
    mcmc_opacity_gate_sharpness: float,
    mcmc_opacity_gate_center: float,
    low_quality_reinit_enabled: bool,
    ema_decay: float = 0.95,
) -> AppTrainingParams:
    base_lr = clamp_float(base_lr, 1e-8, 1.0)
    adam = AdamHyperParams(
        position_lr=base_lr * clamp_float(lr_pos_mul, 0.1, 10.0),
        scale_lr=base_lr * clamp_float(lr_scale_mul, 0.1, 10.0),
        rotation_lr=base_lr * clamp_float(lr_rot_mul, 0.1, 10.0),
        color_lr=base_lr * clamp_float(lr_color_mul, 0.1, 10.0),
        opacity_lr=base_lr * clamp_float(lr_opacity_mul, 0.1, 10.0),
        beta1=clamp_float(beta1, 0.0, 0.99999),
        beta2=clamp_float(beta2, 0.0, 0.999999),
        epsilon=clamp_float(epsilon, 1e-12, 1e-2),
    )
    stability = StabilityHyperParams(
        grad_component_clip=clamp_float(grad_clip, 1e-5, 1e6),
        grad_norm_clip=clamp_float(grad_norm_clip, 1e-5, 1e6),
        max_update=clamp_float(max_update, 1e-8, 10.0),
        min_scale=clamp_float(min_scale, 1e-8, 1e3),
        max_scale=clamp_float(max_scale, 1e-8, 1e4),
        max_anisotropy=clamp_float(max_anisotropy, 1.0, 1e4),
        min_opacity=clamp_float(min_opacity, 0.0, 1.0),
        max_opacity=clamp_float(max_opacity, 0.0, 1.0),
        position_abs_max=clamp_float(position_abs_max, 1e-3, 1e9),
        loss_grad_clip=clamp_float(grad_clip, 1e-5, 1e6),
    )
    training = TrainingHyperParams(
        background=tuple(float(v) for v in np.asarray(background, dtype=np.float32).reshape(3)),
        near=clamp_float(near, 1e-6, 1e4),
        far=clamp_float(far, 1e-5, 1e6),
        ema_decay=float(ema_decay),
        scale_l2_weight=clamp_float(scale_l2_weight, 0.0, 1e4),
        mcmc_position_noise_enabled=bool(mcmc_position_noise_enabled),
        mcmc_position_noise_scale=clamp_float(mcmc_position_noise_scale, 0.0, 1e4),
        mcmc_opacity_gate_sharpness=clamp_float(mcmc_opacity_gate_sharpness, 0.0, 1e6),
        mcmc_opacity_gate_center=clamp_float(mcmc_opacity_gate_center, 0.0, 1.0),
        low_quality_reinit_enabled=bool(low_quality_reinit_enabled),
    )
    if stability.max_scale < stability.min_scale:
        stability.max_scale = stability.min_scale
    if stability.max_opacity < stability.min_opacity:
        stability.max_opacity = stability.min_opacity
    if training.far <= training.near:
        training.far = training.near + 1e-3
    return AppTrainingParams(adam=adam, stability=stability, training=training)


def renderer_kwargs(params: RendererParams) -> dict[str, object]:
    return {
        "radius_scale": float(params.radius_scale),
        "alpha_cutoff": float(params.alpha_cutoff),
        "max_splat_steps": int(params.max_splat_steps),
        "transmittance_threshold": float(params.transmittance_threshold),
        "sampled5_safety_scale": float(params.sampled5_safety_scale),
        "list_capacity_multiplier": int(params.list_capacity_multiplier),
        "max_prepass_memory_mb": int(params.max_prepass_memory_mb),
        "debug_show_ellipses": bool(params.debug_show_ellipses),
        "debug_show_processed_count": bool(params.debug_show_processed_count),
    }


def save_snapshot(path: Path, rgba: np.ndarray, flip_y: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(np.asarray(rgba, dtype=np.float32)[:, :, :3], 0.0, 1.0)
    if flip_y:
        rgb = np.flipud(rgb)
    Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(path)
