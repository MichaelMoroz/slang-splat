from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image
import slangpy as spy

from ..repo_defaults import renderer_defaults
from ..utility import clamp_float, clamp_int
from ..scene import GaussianInitHyperParams, GaussianScene
from ..training.defaults import DEFAULT_REFINEMENT_CLONE_SCALE_MUL, DEFAULT_REFINEMENT_MOMENTUM_WEIGHT_EXPONENT, DEFAULT_REFINEMENT_SPLIT_BETA, TRAINING_BUILD_ARG_DEFAULTS
from ..training import AdamHyperParams, DEFAULT_DEBUG_CONTRIBUTION_RANGE_PERCENT, DEFAULT_REFINEMENT_MIN_CONTRIBUTION, DEFAULT_REFINEMENT_MIN_CONTRIBUTION_DECAY, StabilityHyperParams, TRAIN_BACKGROUND_MODE_CUSTOM, TRAIN_BACKGROUND_MODE_RANDOM, TRAIN_SUBSAMPLE_MAX_FACTOR, TrainingHyperParams, resolve_depth_ratio_grad_band, resolve_training_profile

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
_CLAMP_LIMITS = {
    "grad_component_clip": (1e-5, 1e6),
    "grad_norm_clip": (1e-5, 1e6),
    "max_update": (1e-8, 10.0),
    "max_scale": (1e-8, 1e4),
    "max_anisotropy": (1.0, 1e4),
    "min_opacity": (0.0, 1.0),
    "max_opacity": (0.0, 1.0),
    "position_abs_max": (1e-3, 1e9),
    "loss_grad_clip": (1e-5, 1e6),
}
_RENDERER_DEFAULTS = renderer_defaults()


@dataclass(frozen=True, slots=True)
class RendererParams:
    radius_scale: float = float(_RENDERER_DEFAULTS["radius_scale"]); alpha_cutoff: float = float(_RENDERER_DEFAULTS["alpha_cutoff"])
    max_anisotropy: float = float(_RENDERER_DEFAULTS["max_anisotropy"])
    transmittance_threshold: float = float(_RENDERER_DEFAULTS["transmittance_threshold"]); list_capacity_multiplier: int = int(_RENDERER_DEFAULTS["list_capacity_multiplier"])
    max_prepass_memory_mb: int = int(_RENDERER_DEFAULTS["max_prepass_memory_mb"]); cached_raster_grad_atomic_mode: str = str(_RENDERER_DEFAULTS["cached_raster_grad_atomic_mode"]); cached_raster_grad_fixed_ro_local_range: float = float(_RENDERER_DEFAULTS["cached_raster_grad_fixed_ro_local_range"]); cached_raster_grad_fixed_scale_range: float = float(_RENDERER_DEFAULTS["cached_raster_grad_fixed_scale_range"])
    cached_raster_grad_fixed_quat_range: float = float(_RENDERER_DEFAULTS["cached_raster_grad_fixed_quat_range"]); cached_raster_grad_fixed_color_range: float = float(_RENDERER_DEFAULTS["cached_raster_grad_fixed_color_range"]); cached_raster_grad_fixed_opacity_range: float = float(_RENDERER_DEFAULTS["cached_raster_grad_fixed_opacity_range"])
    debug_mode: str | None = _RENDERER_DEFAULTS["debug_mode"]; debug_grad_norm_threshold: float = float(_RENDERER_DEFAULTS["debug_grad_norm_threshold"]); debug_ellipse_thickness_px: float = float(_RENDERER_DEFAULTS["debug_ellipse_thickness_px"])
    debug_gaussian_scale_multiplier: float = float(_RENDERER_DEFAULTS["debug_gaussian_scale_multiplier"]); debug_min_opacity: float = float(_RENDERER_DEFAULTS["debug_min_opacity"]); debug_opacity_multiplier: float = float(_RENDERER_DEFAULTS["debug_opacity_multiplier"]); debug_ellipse_scale_multiplier: float = float(_RENDERER_DEFAULTS["debug_ellipse_scale_multiplier"])
    debug_splat_age_range: tuple[float, float] = tuple(float(v) for v in _RENDERER_DEFAULTS["debug_splat_age_range"]); debug_density_range: tuple[float, float] = tuple(float(v) for v in _RENDERER_DEFAULTS["debug_density_range"]); debug_contribution_range: tuple[float, float] = tuple(float(v) for v in _RENDERER_DEFAULTS["debug_contribution_range"]); debug_adam_momentum_range: tuple[float, float] = tuple(float(v) for v in _RENDERER_DEFAULTS["debug_adam_momentum_range"]); debug_depth_mean_range: tuple[float, float] = tuple(float(v) for v in _RENDERER_DEFAULTS["debug_depth_mean_range"]); debug_depth_std_range: tuple[float, float] = tuple(float(v) for v in _RENDERER_DEFAULTS["debug_depth_std_range"])
    debug_depth_local_mismatch_range: tuple[float, float] = tuple(float(v) for v in _RENDERER_DEFAULTS["debug_depth_local_mismatch_range"]); debug_depth_local_mismatch_smooth_radius: float = float(_RENDERER_DEFAULTS["debug_depth_local_mismatch_smooth_radius"]); debug_depth_local_mismatch_reject_radius: float = float(_RENDERER_DEFAULTS["debug_depth_local_mismatch_reject_radius"]); debug_sh_coeff_index: int = int(_RENDERER_DEFAULTS["debug_sh_coeff_index"])
    debug_show_ellipses: bool = bool(_RENDERER_DEFAULTS["debug_show_ellipses"]); debug_show_processed_count: bool = bool(_RENDERER_DEFAULTS["debug_show_processed_count"]); debug_show_grad_norm: bool = bool(_RENDERER_DEFAULTS["debug_show_grad_norm"])


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
    base_lr: float = TRAINING_BUILD_ARG_DEFAULTS["base_lr"],
    lr_pos_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_pos_mul"],
    lr_scale_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_scale_mul"],
    lr_rot_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_rot_mul"],
    lr_color_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_color_mul"],
    lr_opacity_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_opacity_mul"],
    beta1: float = TRAINING_BUILD_ARG_DEFAULTS["beta1"],
    beta2: float = TRAINING_BUILD_ARG_DEFAULTS["beta2"],
    grad_clip: float = TRAINING_BUILD_ARG_DEFAULTS["grad_clip"],
    grad_norm_clip: float = TRAINING_BUILD_ARG_DEFAULTS["grad_norm_clip"],
    max_update: float = TRAINING_BUILD_ARG_DEFAULTS["max_update"],
    max_scale: float = TRAINING_BUILD_ARG_DEFAULTS["max_scale"],
    max_anisotropy: float = TRAINING_BUILD_ARG_DEFAULTS["max_anisotropy"],
    min_opacity: float = TRAINING_BUILD_ARG_DEFAULTS["min_opacity"],
    max_opacity: float = TRAINING_BUILD_ARG_DEFAULTS["max_opacity"],
    position_abs_max: float = TRAINING_BUILD_ARG_DEFAULTS["position_abs_max"],
    camera_min_dist: float = TRAINING_BUILD_ARG_DEFAULTS["camera_min_dist"],
    scale_l2_weight: float = TRAINING_BUILD_ARG_DEFAULTS["scale_l2_weight"],
    scale_abs_reg_weight: float = TRAINING_BUILD_ARG_DEFAULTS["scale_abs_reg_weight"],
    opacity_reg_weight: float = TRAINING_BUILD_ARG_DEFAULTS["opacity_reg_weight"],
    sh1_reg_weight: float = TRAINING_BUILD_ARG_DEFAULTS["sh1_reg_weight"],
    density_regularizer: float = TRAINING_BUILD_ARG_DEFAULTS["density_regularizer"],
    color_non_negative_reg: float = TRAINING_BUILD_ARG_DEFAULTS["color_non_negative_reg"],
    depth_ratio_weight: float = TRAINING_BUILD_ARG_DEFAULTS["depth_ratio_weight"],
    max_visible_angle_deg: float = TRAINING_BUILD_ARG_DEFAULTS["max_visible_angle_deg"],
    sorting_order_dithering: float = TRAINING_BUILD_ARG_DEFAULTS["sorting_order_dithering"],
    sorting_order_dithering_stage1: float = TRAINING_BUILD_ARG_DEFAULTS["sorting_order_dithering_stage1"],
    sorting_order_dithering_stage2: float = TRAINING_BUILD_ARG_DEFAULTS["sorting_order_dithering_stage2"],
    sorting_order_dithering_stage3: float = TRAINING_BUILD_ARG_DEFAULTS["sorting_order_dithering_stage3"],
    colorspace_mod: float = TRAINING_BUILD_ARG_DEFAULTS["colorspace_mod"],
    ssim_weight: float = TRAINING_BUILD_ARG_DEFAULTS["ssim_weight"],
    ssim_c2: float = TRAINING_BUILD_ARG_DEFAULTS["ssim_c2"],
    refinement_loss_weight: float = TRAINING_BUILD_ARG_DEFAULTS["refinement_loss_weight"],
    refinement_target_edge_weight: float = TRAINING_BUILD_ARG_DEFAULTS["refinement_target_edge_weight"],
    depth_ratio_grad_min: float = TRAINING_BUILD_ARG_DEFAULTS["depth_ratio_grad_min"],
    depth_ratio_grad_max: float = TRAINING_BUILD_ARG_DEFAULTS["depth_ratio_grad_max"],
    max_allowed_density_start: float = TRAINING_BUILD_ARG_DEFAULTS["max_allowed_density_start"],
    max_allowed_density: float = TRAINING_BUILD_ARG_DEFAULTS["max_allowed_density"],
    lr_pos_stage1_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_pos_stage1_mul"],
    lr_pos_stage2_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_pos_stage2_mul"],
    lr_pos_stage3_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_pos_stage3_mul"],
    lr_sh_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_sh_mul"],
    lr_sh_stage1_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_sh_stage1_mul"],
    lr_sh_stage2_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_sh_stage2_mul"],
    lr_sh_stage3_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_sh_stage3_mul"],
    position_random_step_noise_lr: float = TRAINING_BUILD_ARG_DEFAULTS["position_random_step_noise_lr"],
    position_random_step_opacity_gate_center: float = TRAINING_BUILD_ARG_DEFAULTS["position_random_step_opacity_gate_center"],
    position_random_step_opacity_gate_sharpness: float = TRAINING_BUILD_ARG_DEFAULTS["position_random_step_opacity_gate_sharpness"],
    max_gaussians: int = TRAINING_BUILD_ARG_DEFAULTS["max_gaussians"],
    background_mode: int = TRAINING_BUILD_ARG_DEFAULTS["background_mode"],
    use_target_alpha_mask: bool = TRAINING_BUILD_ARG_DEFAULTS["use_target_alpha_mask"],
    use_sh: bool = TRAINING_BUILD_ARG_DEFAULTS["use_sh"],
    sh_band: int | None = None,
    lr_schedule_enabled: bool = TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_enabled"],
    lr_schedule_start_lr: float = TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_start_lr"],
    lr_schedule_stage1_lr: float = TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_stage1_lr"],
    lr_schedule_stage2_lr: float = TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_stage2_lr"],
    lr_schedule_end_lr: float = TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_end_lr"],
    lr_schedule_steps: int = TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_steps"],
    lr_schedule_stage1_step: int = TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_stage1_step"],
    lr_schedule_stage2_step: int = TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_stage2_step"],
    refinement_interval: int = TRAINING_BUILD_ARG_DEFAULTS["refinement_interval"],
    refinement_growth_ratio: float = TRAINING_BUILD_ARG_DEFAULTS["refinement_growth_ratio"],
    refinement_growth_start_step: int = TRAINING_BUILD_ARG_DEFAULTS["refinement_growth_start_step"],
    refinement_alpha_cull_threshold: float = TRAINING_BUILD_ARG_DEFAULTS["refinement_alpha_cull_threshold"],
    refinement_min_contribution: int = DEFAULT_REFINEMENT_MIN_CONTRIBUTION,
    refinement_min_contribution_decay: float = DEFAULT_REFINEMENT_MIN_CONTRIBUTION_DECAY,
    refinement_opacity_mul: float = TRAINING_BUILD_ARG_DEFAULTS["refinement_opacity_mul"],
    refinement_sample_radius: float = TRAINING_BUILD_ARG_DEFAULTS["refinement_sample_radius"],
    refinement_clone_scale_mul: float = DEFAULT_REFINEMENT_CLONE_SCALE_MUL,
    refinement_use_compact_split: bool = bool(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_use_compact_split", False)),
    refinement_solve_opacity: bool = bool(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_solve_opacity", False)),
    refinement_split_beta: float = DEFAULT_REFINEMENT_SPLIT_BETA,
    refinement_momentum_weight_exponent: float = DEFAULT_REFINEMENT_MOMENTUM_WEIGHT_EXPONENT,
    depth_ratio_stage1_weight: float = TRAINING_BUILD_ARG_DEFAULTS["depth_ratio_stage1_weight"],
    depth_ratio_stage2_weight: float = TRAINING_BUILD_ARG_DEFAULTS["depth_ratio_stage2_weight"],
    depth_ratio_stage3_weight: float = TRAINING_BUILD_ARG_DEFAULTS["depth_ratio_stage3_weight"],
    colorspace_mod_stage1: float = TRAINING_BUILD_ARG_DEFAULTS["colorspace_mod_stage1"],
    colorspace_mod_stage2: float = TRAINING_BUILD_ARG_DEFAULTS["colorspace_mod_stage2"],
    colorspace_mod_stage3: float = TRAINING_BUILD_ARG_DEFAULTS["colorspace_mod_stage3"],
    ssim_weight_stage1: float = TRAINING_BUILD_ARG_DEFAULTS["ssim_weight_stage1"],
    ssim_weight_stage2: float = TRAINING_BUILD_ARG_DEFAULTS["ssim_weight_stage2"],
    ssim_weight_stage3: float = TRAINING_BUILD_ARG_DEFAULTS["ssim_weight_stage3"],
    max_visible_angle_deg_stage1: float = TRAINING_BUILD_ARG_DEFAULTS["max_visible_angle_deg_stage1"],
    max_visible_angle_deg_stage2: float = TRAINING_BUILD_ARG_DEFAULTS["max_visible_angle_deg_stage2"],
    max_visible_angle_deg_stage3: float = TRAINING_BUILD_ARG_DEFAULTS["max_visible_angle_deg_stage3"],
    position_random_step_noise_stage1_lr: float = TRAINING_BUILD_ARG_DEFAULTS["position_random_step_noise_stage1_lr"],
    position_random_step_noise_stage2_lr: float = TRAINING_BUILD_ARG_DEFAULTS["position_random_step_noise_stage2_lr"],
    position_random_step_noise_stage3_lr: float = TRAINING_BUILD_ARG_DEFAULTS["position_random_step_noise_stage3_lr"],
    use_sh_stage1: bool = TRAINING_BUILD_ARG_DEFAULTS["use_sh_stage1"],
    use_sh_stage2: bool = TRAINING_BUILD_ARG_DEFAULTS["use_sh_stage2"],
    use_sh_stage3: bool = TRAINING_BUILD_ARG_DEFAULTS["use_sh_stage3"],
    sh_band_stage1: int | None = None,
    sh_band_stage2: int | None = None,
    sh_band_stage3: int | None = None,
    train_downscale_mode: int = 1,
    train_auto_start_downscale: int = 16,
    train_downscale_base_iters: int = 200,
    train_downscale_iter_step: int = 200,
    train_downscale_max_iters: int = 30_000,
    train_downscale_factor: int = 1,
    train_subsample_factor: int = 0,
) -> AppTrainingParams:
    resolved_sh_band = clamp_int(3 if sh_band is None and bool(use_sh) else (0 if sh_band is None else sh_band), 0, 3)
    resolved_sh_band_stage1 = clamp_int(1 if sh_band_stage1 is None and bool(use_sh_stage1) else (0 if sh_band_stage1 is None else sh_band_stage1), 0, 3)
    resolved_sh_band_stage2 = clamp_int(2 if sh_band_stage2 is None and bool(use_sh_stage2) else (0 if sh_band_stage2 is None else sh_band_stage2), 0, 3)
    resolved_sh_band_stage3 = clamp_int(3 if sh_band_stage3 is None and bool(use_sh_stage3) else (0 if sh_band_stage3 is None else sh_band_stage3), 0, 3)
    base_lr = clamp_float(base_lr, 1e-8, 1.0)
    adam = AdamHyperParams(
        **{
            name: max(float(value), 0.0) * base_lr
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
    resolved_depth_ratio_grad_min, resolved_depth_ratio_grad_max = resolve_depth_ratio_grad_band(
        clamp_float(depth_ratio_grad_min, 0.0, 1e6),
        clamp_float(depth_ratio_grad_max, 0.0, 1e6),
    )
    training = TrainingHyperParams(
        background=tuple(float(v) for v in np.asarray(background, dtype=np.float32).reshape(3)),
        camera_min_dist=clamp_float(camera_min_dist, 0.0, 1e6),
        background_mode=TRAIN_BACKGROUND_MODE_RANDOM if clamp_int(background_mode, TRAIN_BACKGROUND_MODE_CUSTOM, TRAIN_BACKGROUND_MODE_RANDOM) == TRAIN_BACKGROUND_MODE_RANDOM else TRAIN_BACKGROUND_MODE_CUSTOM,
        use_target_alpha_mask=bool(use_target_alpha_mask),
        use_sh=resolved_sh_band > 0,
        sh_band=resolved_sh_band,
        scale_l2_weight=clamp_float(scale_l2_weight, 0.0, 1e4),
        scale_abs_reg_weight=clamp_float(scale_abs_reg_weight, 0.0, 1e4),
        sh1_reg_weight=clamp_float(sh1_reg_weight, 0.0, 1e4),
        opacity_reg_weight=clamp_float(opacity_reg_weight, 0.0, 1e4),
        density_regularizer=clamp_float(density_regularizer, 0.0, 1e4),
        color_non_negative_reg=clamp_float(color_non_negative_reg, 0.0, 1e4),
        depth_ratio_weight=clamp_float(depth_ratio_weight, 0.0, 1e4),
        max_visible_angle_deg=clamp_float(max_visible_angle_deg, 1e-8, 89.999),
        sorting_order_dithering=clamp_float(sorting_order_dithering, 0.0, 1.0),
        sorting_order_dithering_stage1=clamp_float(sorting_order_dithering_stage1, 0.0, 1.0),
        sorting_order_dithering_stage2=clamp_float(sorting_order_dithering_stage2, 0.0, 1.0),
        sorting_order_dithering_stage3=clamp_float(sorting_order_dithering_stage3, 0.0, 1.0),
        colorspace_mod=clamp_float(colorspace_mod, 1e-8, 8.0),
        ssim_weight=clamp_float(ssim_weight, 0.0, 1.0),
        ssim_c2=clamp_float(ssim_c2, 1e-8, 1.0),
        refinement_loss_weight=clamp_float(refinement_loss_weight, 0.0, 1e4),
        refinement_target_edge_weight=clamp_float(refinement_target_edge_weight, 0.0, 1e4),
        depth_ratio_grad_min=resolved_depth_ratio_grad_min,
        depth_ratio_grad_max=resolved_depth_ratio_grad_max,
        max_allowed_density_start=clamp_float(max_allowed_density_start, 0.0, 1e6),
        max_allowed_density=clamp_float(max_allowed_density, 0.0, 1e6),
        lr_pos_mul=max(float(lr_pos_mul), 0.0),
        lr_pos_stage1_mul=max(float(lr_pos_stage1_mul), 0.0),
        lr_pos_stage2_mul=max(float(lr_pos_stage2_mul), 0.0),
        lr_pos_stage3_mul=max(float(lr_pos_stage3_mul), 0.0),
        lr_sh_mul=max(float(lr_sh_mul), 0.0),
        lr_sh_stage1_mul=max(float(lr_sh_stage1_mul), 0.0),
        lr_sh_stage2_mul=max(float(lr_sh_stage2_mul), 0.0),
        lr_sh_stage3_mul=max(float(lr_sh_stage3_mul), 0.0),
        position_random_step_noise_lr=clamp_float(position_random_step_noise_lr, 0.0, 1e12),
        position_random_step_opacity_gate_center=clamp_float(position_random_step_opacity_gate_center, 0.0, 1.0),
        position_random_step_opacity_gate_sharpness=clamp_float(position_random_step_opacity_gate_sharpness, 0.0, 1e6),
        lr_schedule_enabled=bool(lr_schedule_enabled),
        lr_schedule_start_lr=clamp_float(lr_schedule_start_lr, 1e-8, 1.0),
        lr_schedule_stage1_lr=clamp_float(lr_schedule_stage1_lr, 1e-8, 1.0),
        lr_schedule_stage2_lr=clamp_float(lr_schedule_stage2_lr, 1e-8, 1.0),
        lr_schedule_end_lr=clamp_float(lr_schedule_end_lr, 1e-8, 1.0),
        lr_schedule_steps=clamp_int(lr_schedule_steps, 1, 1_000_000_000),
        lr_schedule_stage1_step=clamp_int(lr_schedule_stage1_step, 0, 1_000_000_000),
        lr_schedule_stage2_step=clamp_int(lr_schedule_stage2_step, 0, 1_000_000_000),
        refinement_interval=clamp_int(refinement_interval, 1, 1_000_000_000),
        refinement_growth_ratio=clamp_float(refinement_growth_ratio, 0.0, 10.0),
        refinement_growth_start_step=clamp_int(refinement_growth_start_step, 0, 1_000_000_000),
        refinement_alpha_cull_threshold=clamp_float(refinement_alpha_cull_threshold, 1e-8, 1.0),
        refinement_min_contribution=clamp_int(refinement_min_contribution, 0, 1_000_000_000),
        refinement_min_contribution_decay=clamp_float(refinement_min_contribution_decay, 0.0, 1.0),
        refinement_opacity_mul=clamp_float(refinement_opacity_mul, 0.0, 1.0),
        refinement_sample_radius=clamp_float(refinement_sample_radius, 0.0, 1e6),
        refinement_clone_scale_mul=clamp_float(refinement_clone_scale_mul, 0.0, 1e6),
        refinement_use_compact_split=bool(refinement_use_compact_split),
        refinement_solve_opacity=bool(refinement_solve_opacity),
        refinement_split_beta=clamp_float(refinement_split_beta, 0.0, 1.0),
        refinement_momentum_weight_exponent=clamp_float(refinement_momentum_weight_exponent, 0.0, 16.0),
        depth_ratio_stage1_weight=clamp_float(depth_ratio_stage1_weight, 0.0, 1e4),
        depth_ratio_stage2_weight=clamp_float(depth_ratio_stage2_weight, 0.0, 1e4),
        depth_ratio_stage3_weight=clamp_float(depth_ratio_stage3_weight, 0.0, 1e4),
        colorspace_mod_stage1=clamp_float(colorspace_mod_stage1, 1e-8, 8.0),
        colorspace_mod_stage2=clamp_float(colorspace_mod_stage2, 1e-8, 8.0),
        colorspace_mod_stage3=clamp_float(colorspace_mod_stage3, 1e-8, 8.0),
        ssim_weight_stage1=clamp_float(ssim_weight_stage1, 0.0, 1.0),
        ssim_weight_stage2=clamp_float(ssim_weight_stage2, 0.0, 1.0),
        ssim_weight_stage3=clamp_float(ssim_weight_stage3, 0.0, 1.0),
        max_visible_angle_deg_stage1=clamp_float(max_visible_angle_deg_stage1, 1e-8, 89.999),
        max_visible_angle_deg_stage2=clamp_float(max_visible_angle_deg_stage2, 1e-8, 89.999),
        max_visible_angle_deg_stage3=clamp_float(max_visible_angle_deg_stage3, 1e-8, 89.999),
        position_random_step_noise_stage1_lr=clamp_float(position_random_step_noise_stage1_lr, 0.0, 1e12),
        position_random_step_noise_stage2_lr=clamp_float(position_random_step_noise_stage2_lr, 0.0, 1e12),
        position_random_step_noise_stage3_lr=clamp_float(position_random_step_noise_stage3_lr, 0.0, 1e12),
        use_sh_stage1=resolved_sh_band_stage1 > 0,
        use_sh_stage2=resolved_sh_band_stage2 > 0,
        use_sh_stage3=resolved_sh_band_stage3 > 0,
        sh_band_stage1=resolved_sh_band_stage1,
        sh_band_stage2=resolved_sh_band_stage2,
        sh_band_stage3=resolved_sh_band_stage3,
        max_gaussians=clamp_int(max_gaussians, 0, 10_000_000),
        train_downscale_mode=clamp_int(train_downscale_mode, 0, 16),
        train_auto_start_downscale=clamp_int(train_auto_start_downscale, 1, 16),
        train_downscale_base_iters=clamp_int(train_downscale_base_iters, 1, 1_000_000_000),
        train_downscale_iter_step=clamp_int(train_downscale_iter_step, 0, 1_000_000_000),
        train_downscale_max_iters=clamp_int(train_downscale_max_iters, 1, 1_000_000_000),
        train_downscale_factor=clamp_int(train_downscale_factor, 1, 16),
        train_subsample_factor=clamp_int(train_subsample_factor, 0, TRAIN_SUBSAMPLE_MAX_FACTOR),
    )
    stability.max_opacity = max(stability.max_opacity, stability.min_opacity)
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
