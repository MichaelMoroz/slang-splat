from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image
import slangpy as spy

from ..renderer.render_params import RendererParams
from ..utility import clamp_float, clamp_int
from ..scene import GaussianInitHyperParams, GaussianScene
from ..training.alpha_modes import resolve_target_alpha_mode
from ..training.defaults import DEFAULT_MAX_OPACITY_STAGE0, DEFAULT_MAX_OPACITY_STAGE1, DEFAULT_MAX_OPACITY_STAGE2, DEFAULT_MAX_OPACITY_STAGE3, DEFAULT_MAX_OPACITY_STAGE4, DEFAULT_REFINEMENT_CLONE_SCALE_MUL, DEFAULT_REFINEMENT_CONTRIBUTION_WEIGHT_EXPONENT, DEFAULT_REFINEMENT_GRAD_VARIANCE_WEIGHT_EXPONENT, DEFAULT_REFINEMENT_SPLIT_BETA, TRAINING_BUILD_ARG_DEFAULTS
from ..training import AdamHyperParams, DEFAULT_DEBUG_CONTRIBUTION_RANGE, DEFAULT_REFINEMENT_MIN_CONTRIBUTION, DEFAULT_REFINEMENT_MIN_CONTRIBUTION_DECAY, StabilityHyperParams, TRAIN_BACKGROUND_MODE_CUSTOM, TRAIN_BACKGROUND_MODE_RANDOM, TRAIN_SUBSAMPLE_MAX_FACTOR, TrainingHyperParams, resolve_training_profile

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
    lr_scale_stage1_mul: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("lr_scale_stage1_mul", TRAINING_BUILD_ARG_DEFAULTS["lr_scale_mul"])),
    lr_scale_stage2_mul: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("lr_scale_stage2_mul", TRAINING_BUILD_ARG_DEFAULTS["lr_scale_mul"])),
    lr_scale_stage3_mul: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("lr_scale_stage3_mul", TRAINING_BUILD_ARG_DEFAULTS["lr_scale_mul"])),
    lr_scale_stage4_mul: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("lr_scale_stage4_mul", TRAINING_BUILD_ARG_DEFAULTS.get("lr_scale_stage3_mul", TRAINING_BUILD_ARG_DEFAULTS["lr_scale_mul"]))),
    lr_rot_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_rot_mul"],
    lr_rot_stage1_mul: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("lr_rot_stage1_mul", TRAINING_BUILD_ARG_DEFAULTS["lr_rot_mul"])),
    lr_rot_stage2_mul: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("lr_rot_stage2_mul", TRAINING_BUILD_ARG_DEFAULTS["lr_rot_mul"])),
    lr_rot_stage3_mul: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("lr_rot_stage3_mul", TRAINING_BUILD_ARG_DEFAULTS["lr_rot_mul"])),
    lr_rot_stage4_mul: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("lr_rot_stage4_mul", TRAINING_BUILD_ARG_DEFAULTS.get("lr_rot_stage3_mul", TRAINING_BUILD_ARG_DEFAULTS["lr_rot_mul"]))),
    lr_color_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_color_mul"],
    lr_color_stage1_mul: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("lr_color_stage1_mul", TRAINING_BUILD_ARG_DEFAULTS["lr_color_mul"])),
    lr_color_stage2_mul: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("lr_color_stage2_mul", TRAINING_BUILD_ARG_DEFAULTS["lr_color_mul"])),
    lr_color_stage3_mul: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("lr_color_stage3_mul", TRAINING_BUILD_ARG_DEFAULTS["lr_color_mul"])),
    lr_color_stage4_mul: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("lr_color_stage4_mul", TRAINING_BUILD_ARG_DEFAULTS.get("lr_color_stage3_mul", TRAINING_BUILD_ARG_DEFAULTS["lr_color_mul"]))),
    lr_opacity_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_opacity_mul"],
    lr_opacity_stage1_mul: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("lr_opacity_stage1_mul", TRAINING_BUILD_ARG_DEFAULTS["lr_opacity_mul"])),
    lr_opacity_stage2_mul: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("lr_opacity_stage2_mul", TRAINING_BUILD_ARG_DEFAULTS["lr_opacity_mul"])),
    lr_opacity_stage3_mul: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("lr_opacity_stage3_mul", TRAINING_BUILD_ARG_DEFAULTS["lr_opacity_mul"])),
    lr_opacity_stage4_mul: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("lr_opacity_stage4_mul", TRAINING_BUILD_ARG_DEFAULTS.get("lr_opacity_stage3_mul", TRAINING_BUILD_ARG_DEFAULTS["lr_opacity_mul"]))),
    beta1: float = TRAINING_BUILD_ARG_DEFAULTS["beta1"],
    beta2: float = TRAINING_BUILD_ARG_DEFAULTS["beta2"],
    grad_clip: float = TRAINING_BUILD_ARG_DEFAULTS["grad_clip"],
    max_update: float = TRAINING_BUILD_ARG_DEFAULTS["max_update"],
    max_anisotropy: float = TRAINING_BUILD_ARG_DEFAULTS["max_anisotropy"],
    min_opacity: float = TRAINING_BUILD_ARG_DEFAULTS["min_opacity"],
    max_opacity: float = TRAINING_BUILD_ARG_DEFAULTS["max_opacity"],
    max_opacity_stage0: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("max_opacity_stage0", DEFAULT_MAX_OPACITY_STAGE0)),
    max_opacity_stage1: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("max_opacity_stage1", DEFAULT_MAX_OPACITY_STAGE1)),
    max_opacity_stage2: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("max_opacity_stage2", DEFAULT_MAX_OPACITY_STAGE2)),
    max_opacity_stage3: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("max_opacity_stage3", DEFAULT_MAX_OPACITY_STAGE3)),
    max_opacity_stage4: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("max_opacity_stage4", DEFAULT_MAX_OPACITY_STAGE4)),
    position_abs_max: float = TRAINING_BUILD_ARG_DEFAULTS["position_abs_max"],
    camera_min_dist: float = TRAINING_BUILD_ARG_DEFAULTS["camera_min_dist"],
    scale_l2_weight: float = TRAINING_BUILD_ARG_DEFAULTS["scale_l2_weight"],
    scale_abs_reg_weight: float = TRAINING_BUILD_ARG_DEFAULTS["scale_abs_reg_weight"],
    opacity_reg_weight: float = TRAINING_BUILD_ARG_DEFAULTS["opacity_reg_weight"],
    opacity_reg_weight_stage1: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("opacity_reg_weight_stage1", TRAINING_BUILD_ARG_DEFAULTS["opacity_reg_weight"])),
    opacity_reg_weight_stage2: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("opacity_reg_weight_stage2", TRAINING_BUILD_ARG_DEFAULTS["opacity_reg_weight"])),
    opacity_reg_weight_stage3: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("opacity_reg_weight_stage3", TRAINING_BUILD_ARG_DEFAULTS["opacity_reg_weight"])),
    opacity_reg_weight_stage4: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("opacity_reg_weight_stage4", TRAINING_BUILD_ARG_DEFAULTS.get("opacity_reg_weight_stage3", TRAINING_BUILD_ARG_DEFAULTS["opacity_reg_weight"]))),
    sh1_reg_weight: float = TRAINING_BUILD_ARG_DEFAULTS["sh1_reg_weight"],
    position_push_away_from_camera_step: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("position_push_away_from_camera_step", 0.0)),
    position_push_away_from_camera_step_stage1: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("position_push_away_from_camera_step_stage1", TRAINING_BUILD_ARG_DEFAULTS.get("position_push_away_from_camera_step", 0.0))),
    position_push_away_from_camera_step_stage2: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("position_push_away_from_camera_step_stage2", TRAINING_BUILD_ARG_DEFAULTS.get("position_push_away_from_camera_step", 0.0))),
    position_push_away_from_camera_step_stage3: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("position_push_away_from_camera_step_stage3", TRAINING_BUILD_ARG_DEFAULTS.get("position_push_away_from_camera_step", 0.0))),
    position_push_away_from_camera_step_stage4: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("position_push_away_from_camera_step_stage4", TRAINING_BUILD_ARG_DEFAULTS.get("position_push_away_from_camera_step_stage3", TRAINING_BUILD_ARG_DEFAULTS.get("position_push_away_from_camera_step", 0.0)))) ,
    density_regularizer: float = TRAINING_BUILD_ARG_DEFAULTS["density_regularizer"],
    max_visible_angle_deg: float = TRAINING_BUILD_ARG_DEFAULTS["max_visible_angle_deg"],
    sorting_order_dithering: float = TRAINING_BUILD_ARG_DEFAULTS["sorting_order_dithering"],
    sorting_order_dithering_stage1: float = TRAINING_BUILD_ARG_DEFAULTS["sorting_order_dithering_stage1"],
    sorting_order_dithering_stage2: float = TRAINING_BUILD_ARG_DEFAULTS["sorting_order_dithering_stage2"],
    sorting_order_dithering_stage3: float = TRAINING_BUILD_ARG_DEFAULTS["sorting_order_dithering_stage3"],
    sorting_order_dithering_stage4: float = TRAINING_BUILD_ARG_DEFAULTS["sorting_order_dithering_stage4"],
    colorspace_mod: float = TRAINING_BUILD_ARG_DEFAULTS["colorspace_mod"],
    ssim_weight: float = TRAINING_BUILD_ARG_DEFAULTS["ssim_weight"],
    ssim_c2: float = TRAINING_BUILD_ARG_DEFAULTS["ssim_c2"],
    refinement_loss_weight: float = TRAINING_BUILD_ARG_DEFAULTS["refinement_loss_weight"],
    refinement_target_edge_weight: float = TRAINING_BUILD_ARG_DEFAULTS["refinement_target_edge_weight"],
    refinement_min_screen_radius_px: float = TRAINING_BUILD_ARG_DEFAULTS["refinement_min_screen_radius_px"],
    max_allowed_density_start: float = TRAINING_BUILD_ARG_DEFAULTS["max_allowed_density_start"],
    max_allowed_density: float = TRAINING_BUILD_ARG_DEFAULTS["max_allowed_density"],
    lr_pos_stage1_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_pos_stage1_mul"],
    lr_pos_stage2_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_pos_stage2_mul"],
    lr_pos_stage3_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_pos_stage3_mul"],
    lr_pos_stage4_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_pos_stage4_mul"],
    lr_sh_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_sh_mul"],
    lr_sh_stage1_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_sh_stage1_mul"],
    lr_sh_stage2_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_sh_stage2_mul"],
    lr_sh_stage3_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_sh_stage3_mul"],
    lr_sh_stage4_mul: float = TRAINING_BUILD_ARG_DEFAULTS["lr_sh_stage4_mul"],
    position_random_step_noise_lr: float = TRAINING_BUILD_ARG_DEFAULTS["position_random_step_noise_lr"],
    position_random_step_opacity_gate_center: float = TRAINING_BUILD_ARG_DEFAULTS["position_random_step_opacity_gate_center"],
    position_random_step_opacity_gate_sharpness: float = TRAINING_BUILD_ARG_DEFAULTS["position_random_step_opacity_gate_sharpness"],
    max_gaussians: int = TRAINING_BUILD_ARG_DEFAULTS["max_gaussians"],
    background_mode: int = TRAINING_BUILD_ARG_DEFAULTS["background_mode"],
    target_alpha_mode: int | None = TRAINING_BUILD_ARG_DEFAULTS.get("target_alpha_mode", None),
    use_target_alpha_mask: bool = TRAINING_BUILD_ARG_DEFAULTS["use_target_alpha_mask"],
    target_alpha_threshold: float = TRAINING_BUILD_ARG_DEFAULTS["target_alpha_threshold"],
    use_sh: bool = TRAINING_BUILD_ARG_DEFAULTS["use_sh"],
    sh_band: int | None = None,
    max_sh_band: int | None = None,
    lr_schedule_enabled: bool = TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_enabled"],
    lr_schedule_start_lr: float = TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_start_lr"],
    lr_schedule_stage1_lr: float = TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_stage1_lr"],
    lr_schedule_stage2_lr: float = TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_stage2_lr"],
    lr_schedule_stage3_lr: float = TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_stage3_lr"],
    lr_schedule_end_lr: float = TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_end_lr"],
    lr_schedule_steps: int = TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_steps"],
    lr_schedule_stage1_step: int = TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_stage1_step"],
    lr_schedule_stage2_step: int = TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_stage2_step"],
    lr_schedule_stage3_step: int = TRAINING_BUILD_ARG_DEFAULTS["lr_schedule_stage3_step"],
    refinement_interval: int = TRAINING_BUILD_ARG_DEFAULTS["refinement_interval"],
    refinement_growth_start_step: int = TRAINING_BUILD_ARG_DEFAULTS["refinement_growth_start_step"],
    refinement_target_splat_ratio: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_target_splat_ratio", 0.10)),
    refinement_target_splat_ratio_stage1: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_target_splat_ratio_stage1", 0.20)),
    refinement_target_splat_ratio_stage2: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_target_splat_ratio_stage2", 0.50)),
    refinement_target_splat_ratio_stage3: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_target_splat_ratio_stage3", 1.00)),
    refinement_target_splat_ratio_stage4: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_target_splat_ratio_stage4", TRAINING_BUILD_ARG_DEFAULTS.get("refinement_target_splat_ratio_stage3", 1.00))),
    refinement_max_growth_per_step: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_max_growth_per_step", 0.15)),
    refinement_max_prune_per_step: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_max_prune_per_step", 0.15)),
    refinement_alpha_cull_threshold: float = TRAINING_BUILD_ARG_DEFAULTS["refinement_alpha_cull_threshold"],
    refinement_min_contribution: float = DEFAULT_REFINEMENT_MIN_CONTRIBUTION,
    refinement_min_contribution_decay: float = DEFAULT_REFINEMENT_MIN_CONTRIBUTION_DECAY,
    refinement_ema_pose_count_decay: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_ema_pose_count_decay", 0.25)),
    refinement_viewed_fraction_zero_threshold: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_viewed_fraction_zero_threshold", 0.66)),
    refinement_prune_lowest_contribution_ratio: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_prune_lowest_contribution_ratio", 0.0)),
    refinement_prune_lowest_contribution_ratio_stage1: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_prune_lowest_contribution_ratio_stage1", TRAINING_BUILD_ARG_DEFAULTS.get("refinement_prune_lowest_contribution_ratio", 0.0))),
    refinement_prune_lowest_contribution_ratio_stage2: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_prune_lowest_contribution_ratio_stage2", TRAINING_BUILD_ARG_DEFAULTS.get("refinement_prune_lowest_contribution_ratio", 0.0))),
    refinement_prune_lowest_contribution_ratio_stage3: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_prune_lowest_contribution_ratio_stage3", TRAINING_BUILD_ARG_DEFAULTS.get("refinement_prune_lowest_contribution_ratio", 0.0))),
    refinement_prune_lowest_contribution_ratio_stage4: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_prune_lowest_contribution_ratio_stage4", TRAINING_BUILD_ARG_DEFAULTS.get("refinement_prune_lowest_contribution_ratio_stage3", TRAINING_BUILD_ARG_DEFAULTS.get("refinement_prune_lowest_contribution_ratio", 0.0)))) ,
    refinement_opacity_mul: float = TRAINING_BUILD_ARG_DEFAULTS["refinement_opacity_mul"],
    refinement_sample_radius: float = TRAINING_BUILD_ARG_DEFAULTS["refinement_sample_radius"],
    refinement_clone_scale_mul: float = DEFAULT_REFINEMENT_CLONE_SCALE_MUL,
    refinement_use_compact_split: bool = bool(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_use_compact_split", False)),
    refinement_solve_opacity: bool = bool(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_solve_opacity", False)),
    refinement_split_beta: float = DEFAULT_REFINEMENT_SPLIT_BETA,
    refinement_grad_variance_weight_exponent: float | None = None,
    refinement_contribution_weight_exponent: float | None = None,
    refinement_contribution_area_exponent: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_contribution_area_exponent", 0.5)),
    refinement_contribution_view_count_exponent: float = float(TRAINING_BUILD_ARG_DEFAULTS.get("refinement_contribution_view_count_exponent", 0.5)),
    refinement_momentum_weight_exponent: float | None = None,
    colorspace_mod_stage1: float = TRAINING_BUILD_ARG_DEFAULTS["colorspace_mod_stage1"],
    colorspace_mod_stage2: float = TRAINING_BUILD_ARG_DEFAULTS["colorspace_mod_stage2"],
    colorspace_mod_stage3: float = TRAINING_BUILD_ARG_DEFAULTS["colorspace_mod_stage3"],
    colorspace_mod_stage4: float = TRAINING_BUILD_ARG_DEFAULTS["colorspace_mod_stage4"],
    ssim_weight_stage1: float = TRAINING_BUILD_ARG_DEFAULTS["ssim_weight_stage1"],
    ssim_weight_stage2: float = TRAINING_BUILD_ARG_DEFAULTS["ssim_weight_stage2"],
    ssim_weight_stage3: float = TRAINING_BUILD_ARG_DEFAULTS["ssim_weight_stage3"],
    ssim_weight_stage4: float = TRAINING_BUILD_ARG_DEFAULTS["ssim_weight_stage4"],
    max_visible_angle_deg_stage1: float = TRAINING_BUILD_ARG_DEFAULTS["max_visible_angle_deg_stage1"],
    max_visible_angle_deg_stage2: float = TRAINING_BUILD_ARG_DEFAULTS["max_visible_angle_deg_stage2"],
    max_visible_angle_deg_stage3: float = TRAINING_BUILD_ARG_DEFAULTS["max_visible_angle_deg_stage3"],
    max_visible_angle_deg_stage4: float = TRAINING_BUILD_ARG_DEFAULTS["max_visible_angle_deg_stage4"],
    refinement_min_screen_radius_px_stage1: float = TRAINING_BUILD_ARG_DEFAULTS["refinement_min_screen_radius_px_stage1"],
    refinement_min_screen_radius_px_stage2: float = TRAINING_BUILD_ARG_DEFAULTS["refinement_min_screen_radius_px_stage2"],
    refinement_min_screen_radius_px_stage3: float = TRAINING_BUILD_ARG_DEFAULTS["refinement_min_screen_radius_px_stage3"],
    refinement_min_screen_radius_px_stage4: float = TRAINING_BUILD_ARG_DEFAULTS["refinement_min_screen_radius_px_stage4"],
    position_random_step_noise_stage1_lr: float = TRAINING_BUILD_ARG_DEFAULTS["position_random_step_noise_stage1_lr"],
    position_random_step_noise_stage2_lr: float = TRAINING_BUILD_ARG_DEFAULTS["position_random_step_noise_stage2_lr"],
    position_random_step_noise_stage3_lr: float = TRAINING_BUILD_ARG_DEFAULTS["position_random_step_noise_stage3_lr"],
    position_random_step_noise_stage4_lr: float = TRAINING_BUILD_ARG_DEFAULTS["position_random_step_noise_stage4_lr"],
    use_sh_stage1: bool = TRAINING_BUILD_ARG_DEFAULTS["use_sh_stage1"],
    use_sh_stage2: bool = TRAINING_BUILD_ARG_DEFAULTS["use_sh_stage2"],
    use_sh_stage3: bool = TRAINING_BUILD_ARG_DEFAULTS["use_sh_stage3"],
    use_sh_stage4: bool = TRAINING_BUILD_ARG_DEFAULTS["use_sh_stage4"],
    sh_band_stage1: int | None = None,
    sh_band_stage2: int | None = None,
    sh_band_stage3: int | None = None,
    sh_band_stage4: int | None = None,
    train_downscale_mode: int = 1,
    train_auto_start_downscale: int = 16,
    train_downscale_base_iters: int = 200,
    train_downscale_iter_step: int = 200,
    train_downscale_max_iters: int = 30_000,
    train_downscale_factor: int = 1,
    train_subsample_factor: int = 0,
) -> AppTrainingParams:
    resolved_sh_band = 3 if sh_band is None and bool(use_sh) else (0 if sh_band is None else int(sh_band))
    resolved_max_sh_band = 3 if max_sh_band is None else int(max_sh_band)
    resolved_sh_band_stage1 = 1 if sh_band_stage1 is None and bool(use_sh_stage1) else (0 if sh_band_stage1 is None else int(sh_band_stage1))
    resolved_sh_band_stage2 = 2 if sh_band_stage2 is None and bool(use_sh_stage2) else (0 if sh_band_stage2 is None else int(sh_band_stage2))
    resolved_sh_band_stage3 = 3 if sh_band_stage3 is None and bool(use_sh_stage3) else (0 if sh_band_stage3 is None else int(sh_band_stage3))
    resolved_sh_band_stage4 = 3 if sh_band_stage4 is None and bool(use_sh_stage4) else (0 if sh_band_stage4 is None else int(sh_band_stage4))
    resolved_target_alpha_mode = resolve_target_alpha_mode(target_alpha_mode, legacy_use_target_alpha_mask=bool(use_target_alpha_mask))
    resolved_refinement_variance_exponent = (
        DEFAULT_REFINEMENT_GRAD_VARIANCE_WEIGHT_EXPONENT
        if refinement_grad_variance_weight_exponent is None and refinement_momentum_weight_exponent is None
        else (refinement_momentum_weight_exponent if refinement_grad_variance_weight_exponent is None else refinement_grad_variance_weight_exponent)
    )
    resolved_refinement_contribution_exponent = DEFAULT_REFINEMENT_CONTRIBUTION_WEIGHT_EXPONENT if refinement_contribution_weight_exponent is None else refinement_contribution_weight_exponent
    base_lr = float(base_lr)
    adam = AdamHyperParams(
        **{
            name: float(value) * base_lr
            for name, value in {
                "position_lr": lr_pos_mul,
                "scale_lr": lr_scale_mul,
                "rotation_lr": lr_rot_mul,
                "color_lr": lr_color_mul,
                "opacity_lr": lr_opacity_mul,
            }.items()
        },
        beta1=float(beta1),
        beta2=float(beta2),
    )
    stability = StabilityHyperParams(
        grad_component_clip=float(grad_clip),
        max_update=float(max_update),
        max_anisotropy=float(max_anisotropy),
        min_opacity=float(min_opacity),
        max_opacity=float(max_opacity),
        position_abs_max=float(position_abs_max),
        loss_grad_clip=float(grad_clip),
    )
    training = TrainingHyperParams(
        background=tuple(float(v) for v in np.asarray(background, dtype=np.float32).reshape(3)),
        camera_min_dist=float(camera_min_dist),
        background_mode=TRAIN_BACKGROUND_MODE_RANDOM if clamp_int(background_mode, TRAIN_BACKGROUND_MODE_CUSTOM, TRAIN_BACKGROUND_MODE_RANDOM) == TRAIN_BACKGROUND_MODE_RANDOM else TRAIN_BACKGROUND_MODE_CUSTOM,
        target_alpha_mode=resolved_target_alpha_mode,
        use_target_alpha_mask=bool(use_target_alpha_mask),
        target_alpha_threshold=float(target_alpha_threshold),
        use_sh=resolved_sh_band > 0,
        sh_band=resolved_sh_band,
        max_sh_band=resolved_max_sh_band,
        scale_l2_weight=float(scale_l2_weight),
        scale_abs_reg_weight=float(scale_abs_reg_weight),
        sh1_reg_weight=float(sh1_reg_weight),
        max_opacity=float(max_opacity),
        max_opacity_stage0=float(max_opacity_stage0),
        max_opacity_stage1=float(max_opacity_stage1),
        max_opacity_stage2=float(max_opacity_stage2),
        max_opacity_stage3=float(max_opacity_stage3),
        max_opacity_stage4=float(max_opacity_stage4),
        opacity_reg_weight=float(opacity_reg_weight),
        opacity_reg_weight_stage1=float(opacity_reg_weight_stage1),
        opacity_reg_weight_stage2=float(opacity_reg_weight_stage2),
        opacity_reg_weight_stage3=float(opacity_reg_weight_stage3),
        opacity_reg_weight_stage4=float(opacity_reg_weight_stage4),
        position_push_away_from_camera_step=float(position_push_away_from_camera_step),
        position_push_away_from_camera_step_stage1=float(position_push_away_from_camera_step_stage1),
        position_push_away_from_camera_step_stage2=float(position_push_away_from_camera_step_stage2),
        position_push_away_from_camera_step_stage3=float(position_push_away_from_camera_step_stage3),
        position_push_away_from_camera_step_stage4=float(position_push_away_from_camera_step_stage4),
        density_regularizer=float(density_regularizer),
        max_visible_angle_deg=float(max_visible_angle_deg),
        sorting_order_dithering=float(sorting_order_dithering),
        sorting_order_dithering_stage1=float(sorting_order_dithering_stage1),
        sorting_order_dithering_stage2=float(sorting_order_dithering_stage2),
        sorting_order_dithering_stage3=float(sorting_order_dithering_stage3),
        sorting_order_dithering_stage4=float(sorting_order_dithering_stage4),
        colorspace_mod=float(colorspace_mod),
        ssim_weight=float(ssim_weight),
        ssim_c2=float(ssim_c2),
        refinement_loss_weight=float(refinement_loss_weight),
        refinement_target_edge_weight=float(refinement_target_edge_weight),
        refinement_min_screen_radius_px=float(refinement_min_screen_radius_px),
        max_allowed_density_start=float(max_allowed_density_start),
        max_allowed_density=float(max_allowed_density),
        lr_pos_mul=float(lr_pos_mul),
        lr_pos_stage1_mul=float(lr_pos_stage1_mul),
        lr_pos_stage2_mul=float(lr_pos_stage2_mul),
        lr_pos_stage3_mul=float(lr_pos_stage3_mul),
        lr_pos_stage4_mul=float(lr_pos_stage4_mul),
        lr_scale_mul=float(lr_scale_mul),
        lr_scale_stage1_mul=float(lr_scale_stage1_mul),
        lr_scale_stage2_mul=float(lr_scale_stage2_mul),
        lr_scale_stage3_mul=float(lr_scale_stage3_mul),
        lr_scale_stage4_mul=float(lr_scale_stage4_mul),
        lr_rot_mul=float(lr_rot_mul),
        lr_rot_stage1_mul=float(lr_rot_stage1_mul),
        lr_rot_stage2_mul=float(lr_rot_stage2_mul),
        lr_rot_stage3_mul=float(lr_rot_stage3_mul),
        lr_rot_stage4_mul=float(lr_rot_stage4_mul),
        lr_color_mul=float(lr_color_mul),
        lr_color_stage1_mul=float(lr_color_stage1_mul),
        lr_color_stage2_mul=float(lr_color_stage2_mul),
        lr_color_stage3_mul=float(lr_color_stage3_mul),
        lr_color_stage4_mul=float(lr_color_stage4_mul),
        lr_opacity_mul=float(lr_opacity_mul),
        lr_opacity_stage1_mul=float(lr_opacity_stage1_mul),
        lr_opacity_stage2_mul=float(lr_opacity_stage2_mul),
        lr_opacity_stage3_mul=float(lr_opacity_stage3_mul),
        lr_opacity_stage4_mul=float(lr_opacity_stage4_mul),
        lr_sh_mul=float(lr_sh_mul),
        lr_sh_stage1_mul=float(lr_sh_stage1_mul),
        lr_sh_stage2_mul=float(lr_sh_stage2_mul),
        lr_sh_stage3_mul=float(lr_sh_stage3_mul),
        lr_sh_stage4_mul=float(lr_sh_stage4_mul),
        position_random_step_noise_lr=float(position_random_step_noise_lr),
        position_random_step_opacity_gate_center=float(position_random_step_opacity_gate_center),
        position_random_step_opacity_gate_sharpness=float(position_random_step_opacity_gate_sharpness),
        lr_schedule_enabled=bool(lr_schedule_enabled),
        lr_schedule_start_lr=float(lr_schedule_start_lr),
        lr_schedule_stage1_lr=float(lr_schedule_stage1_lr),
        lr_schedule_stage2_lr=float(lr_schedule_stage2_lr),
        lr_schedule_stage3_lr=float(lr_schedule_stage3_lr),
        lr_schedule_end_lr=float(lr_schedule_end_lr),
        lr_schedule_steps=int(lr_schedule_steps),
        lr_schedule_stage1_step=int(lr_schedule_stage1_step),
        lr_schedule_stage2_step=int(lr_schedule_stage2_step),
        lr_schedule_stage3_step=int(lr_schedule_stage3_step),
        refinement_interval=int(refinement_interval),
        refinement_growth_start_step=int(refinement_growth_start_step),
        refinement_target_splat_ratio=float(refinement_target_splat_ratio),
        refinement_target_splat_ratio_stage1=float(refinement_target_splat_ratio_stage1),
        refinement_target_splat_ratio_stage2=float(refinement_target_splat_ratio_stage2),
        refinement_target_splat_ratio_stage3=float(refinement_target_splat_ratio_stage3),
        refinement_target_splat_ratio_stage4=float(refinement_target_splat_ratio_stage4),
        refinement_max_growth_per_step=float(refinement_max_growth_per_step),
        refinement_max_prune_per_step=float(refinement_max_prune_per_step),
        refinement_alpha_cull_threshold=float(refinement_alpha_cull_threshold),
        refinement_min_contribution=float(refinement_min_contribution),
        refinement_min_contribution_decay=float(refinement_min_contribution_decay),
        refinement_ema_pose_count_decay=float(refinement_ema_pose_count_decay),
        refinement_viewed_fraction_zero_threshold=float(refinement_viewed_fraction_zero_threshold),
        refinement_prune_lowest_contribution_ratio=float(refinement_prune_lowest_contribution_ratio),
        refinement_prune_lowest_contribution_ratio_stage1=float(refinement_prune_lowest_contribution_ratio_stage1),
        refinement_prune_lowest_contribution_ratio_stage2=float(refinement_prune_lowest_contribution_ratio_stage2),
        refinement_prune_lowest_contribution_ratio_stage3=float(refinement_prune_lowest_contribution_ratio_stage3),
        refinement_prune_lowest_contribution_ratio_stage4=float(refinement_prune_lowest_contribution_ratio_stage4),
        refinement_opacity_mul=float(refinement_opacity_mul),
        refinement_sample_radius=float(refinement_sample_radius),
        refinement_clone_scale_mul=float(refinement_clone_scale_mul),
        refinement_use_compact_split=bool(refinement_use_compact_split),
        refinement_solve_opacity=bool(refinement_solve_opacity),
        refinement_split_beta=float(refinement_split_beta),
        refinement_grad_variance_weight_exponent=float(resolved_refinement_variance_exponent),
        refinement_contribution_weight_exponent=float(resolved_refinement_contribution_exponent),
        refinement_contribution_area_exponent=float(refinement_contribution_area_exponent),
        refinement_contribution_view_count_exponent=float(refinement_contribution_view_count_exponent),
        colorspace_mod_stage1=float(colorspace_mod_stage1),
        colorspace_mod_stage2=float(colorspace_mod_stage2),
        colorspace_mod_stage3=float(colorspace_mod_stage3),
        colorspace_mod_stage4=float(colorspace_mod_stage4),
        ssim_weight_stage1=float(ssim_weight_stage1),
        ssim_weight_stage2=float(ssim_weight_stage2),
        ssim_weight_stage3=float(ssim_weight_stage3),
        ssim_weight_stage4=float(ssim_weight_stage4),
        max_visible_angle_deg_stage1=float(max_visible_angle_deg_stage1),
        max_visible_angle_deg_stage2=float(max_visible_angle_deg_stage2),
        max_visible_angle_deg_stage3=float(max_visible_angle_deg_stage3),
        max_visible_angle_deg_stage4=float(max_visible_angle_deg_stage4),
        refinement_min_screen_radius_px_stage1=float(refinement_min_screen_radius_px_stage1),
        refinement_min_screen_radius_px_stage2=float(refinement_min_screen_radius_px_stage2),
        refinement_min_screen_radius_px_stage3=float(refinement_min_screen_radius_px_stage3),
        refinement_min_screen_radius_px_stage4=float(refinement_min_screen_radius_px_stage4),
        position_random_step_noise_stage1_lr=float(position_random_step_noise_stage1_lr),
        position_random_step_noise_stage2_lr=float(position_random_step_noise_stage2_lr),
        position_random_step_noise_stage3_lr=float(position_random_step_noise_stage3_lr),
        position_random_step_noise_stage4_lr=float(position_random_step_noise_stage4_lr),
        use_sh_stage1=resolved_sh_band_stage1 > 0,
        use_sh_stage2=resolved_sh_band_stage2 > 0,
        use_sh_stage3=resolved_sh_band_stage3 > 0,
        use_sh_stage4=resolved_sh_band_stage4 > 0,
        sh_band_stage1=resolved_sh_band_stage1,
        sh_band_stage2=resolved_sh_band_stage2,
        sh_band_stage3=resolved_sh_band_stage3,
        sh_band_stage4=resolved_sh_band_stage4,
        max_gaussians=int(max_gaussians),
        train_downscale_mode=int(train_downscale_mode),
        train_auto_start_downscale=int(train_auto_start_downscale),
        train_downscale_base_iters=int(train_downscale_base_iters),
        train_downscale_iter_step=int(train_downscale_iter_step),
        train_downscale_max_iters=int(train_downscale_max_iters),
        train_downscale_factor=int(train_downscale_factor),
        train_subsample_factor=int(train_subsample_factor),
    )
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
def save_snapshot(path: Path, rgba: np.ndarray, flip_y: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(np.asarray(rgba, dtype=np.float32)[:, :, :3], 0.0, 1.0)
    Image.fromarray((255.0 * (np.flipud(rgb) if flip_y else rgb) + 0.5).astype(np.uint8), mode="RGB").save(path)
