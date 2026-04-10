from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import slangpy as spy

from ..utility import RW_BUFFER_USAGE, SHADER_ROOT, alloc_buffer, alloc_texture_2d, buffer_to_numpy, clamp_index, debug_region, dispatch, grow_capacity, load_compute_items, thread_count_1d, thread_count_2d
from ..filter import SeparableGaussianBlur
from ..metrics import Metrics, psnr_from_mse
from ..renderer import Camera, GaussianRenderer
from ..scene import ColmapFrame, GaussianInitHyperParams, GaussianScene, SUPPORTED_SH_COEFF_COUNT, pad_sh_coeffs, rgb_to_sh0, sh_coeffs_to_display_colors
from ..scene._internal.colmap_ops import TRAINING_FRAME_LOAD_THREADS, load_training_frame_rgba8
from .adam import AdamOptimizer, AdamRuntimeHyperParams
from .optimizer import GaussianOptimizer
from .schedule import DEFAULT_REFINEMENT_MIN_CONTRIBUTION_DECAY, resolve_base_learning_rate, resolve_clone_probability_threshold, resolve_depth_ratio_weight, resolve_effective_refinement_interval, resolve_learning_rate_scale, resolve_position_lr_mul, resolve_position_random_step_noise_lr, resolve_refinement_min_contribution_percent, resolve_max_allowed_density, resolve_sh_band, should_run_refinement_step

TRAIN_DOWNSCALE_MODE_AUTO = 0
TRAIN_DOWNSCALE_MAX_FACTOR = 16
TRAIN_SUBSAMPLE_MODE_AUTO = 0
TRAIN_SUBSAMPLE_MAX_FACTOR = 4
TRAIN_SUBSAMPLE_TARGET_MAX_SIDE = 1000
TRAIN_BACKGROUND_MODE_CUSTOM = 0
TRAIN_BACKGROUND_MODE_RANDOM = 1
SPLAT_CONTRIBUTION_FIXED_SCALE = 256.0
DEFAULT_REFINEMENT_MIN_CONTRIBUTION_PERCENT = 1e-05
DEFAULT_DEBUG_CONTRIBUTION_RANGE_PERCENT = (0.001, 1.0)
DEPTH_RATIO_GRAD_MIN_BAND_WIDTH = 1e-4
DEFAULT_DEPTH_RATIO_GRAD_MIN = 0.0
DEFAULT_DEPTH_RATIO_GRAD_MAX = 0.1
DEFAULT_SSIM_WEIGHT = 0.2
DEFAULT_SSIM_C1 = 1e-4
DEFAULT_SSIM_C2 = 9e-4
_REFINEMENT_HASH_INIT = 0x9E3779B9
_REFINEMENT_HASH_MIX = 0x85EBCA6B


def _hash_u32_scalar(value: int) -> np.uint32:
    x = int(value) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return np.uint32(x)


def _refinement_hash_combine(seed: int, value: int) -> np.uint32:
    mixed = (int(value) * _REFINEMENT_HASH_MIX + _REFINEMENT_HASH_INIT) & 0xFFFFFFFF
    return _hash_u32_scalar((int(seed) ^ mixed) & 0xFFFFFFFF)


def _refinement_camera_hash(camera_id: int) -> np.uint32:
    return _refinement_hash_combine(_REFINEMENT_HASH_INIT, int(camera_id))


def _u32_bits_to_f32(value: int) -> np.float32:
    return np.asarray([np.uint32(value)], dtype=np.uint32).view(np.float32)[0]


def contribution_percent_from_fixed_count(contribution_fixed: float | np.ndarray, observed_pixel_count: float) -> float | np.ndarray:
    pixels = max(float(observed_pixel_count), 1.0)
    contribution = np.asarray(contribution_fixed, dtype=np.float64) * (100.0 / (SPLAT_CONTRIBUTION_FIXED_SCALE * pixels))
    if contribution.ndim == 0:
        return float(contribution)
    return contribution


def contribution_fixed_count_from_percent(contribution_percent: float, observed_pixel_count: float) -> int:
    percent = max(float(contribution_percent), 0.0)
    pixels = max(float(observed_pixel_count), 0.0)
    if percent <= 0.0 or pixels <= 0.0:
        return 0
    return max(int(round((percent * 0.01) * pixels * SPLAT_CONTRIBUTION_FIXED_SCALE)), 0)


def resolve_training_resolution(width: int, height: int, downscale_factor: int) -> tuple[int, int]:
    factor = max(int(downscale_factor), 1)
    native_width = max(int(width), 1)
    native_height = max(int(height), 1)
    return (native_width + factor - 1) // factor, (native_height + factor - 1) // factor


def resolve_effective_train_downscale_factor(training_hparams: "TrainingHyperParams", step: int) -> int:
    resolved_step = max(int(step), 0)
    mode = int(training_hparams.train_downscale_mode)
    if mode != TRAIN_DOWNSCALE_MODE_AUTO:
        return min(max(mode, 1), TRAIN_DOWNSCALE_MAX_FACTOR)
    start_factor = min(max(int(training_hparams.train_auto_start_downscale), 1), TRAIN_DOWNSCALE_MAX_FACTOR)
    base_iters = max(int(training_hparams.train_downscale_base_iters), 1)
    iter_step = max(int(training_hparams.train_downscale_iter_step), 0)
    elapsed = 0
    for factor in range(start_factor, 0, -1):
        duration = max(base_iters + (start_factor - factor) * iter_step, 1)
        next_elapsed = elapsed + duration
        if resolved_step < next_elapsed:
            return factor
        elapsed = next_elapsed
    return 1


def resolve_auto_train_subsample_factor(width: int, height: int, downscale_factor: int = 1) -> int:
    native_max_side = max(int(width), int(height), 1)
    base_factor = max(int(downscale_factor), 1)
    resolved_factor = 1
    for factor in range(1, TRAIN_SUBSAMPLE_MAX_FACTOR + 1):
        max_side = (native_max_side + base_factor * factor - 1) // (base_factor * factor)
        if max_side <= TRAIN_SUBSAMPLE_TARGET_MAX_SIDE: break
        resolved_factor = factor
    return resolved_factor


def resolve_train_subsample_factor(training_hparams: "TrainingHyperParams", width: int | None = None, height: int | None = None, step: int = 0) -> int:
    mode = int(training_hparams.train_subsample_factor)
    if mode != TRAIN_SUBSAMPLE_MODE_AUTO:
        return min(max(mode, 1), TRAIN_SUBSAMPLE_MAX_FACTOR)
    downscale_factor = resolve_effective_train_downscale_factor(training_hparams, step)
    return 1 if width is None or height is None else resolve_auto_train_subsample_factor(width, height, downscale_factor)


def resolve_effective_train_render_factor(training_hparams: "TrainingHyperParams", step: int, width: int | None = None, height: int | None = None) -> int:
    return max(resolve_effective_train_downscale_factor(training_hparams, step) * resolve_train_subsample_factor(training_hparams, width, height, step), 1)


def resolve_depth_ratio_grad_band(grad_min: float, grad_max: float) -> tuple[float, float]:
    resolved_min = max(float(grad_min), 0.0)
    resolved_max = max(float(grad_max), resolved_min + DEPTH_RATIO_GRAD_MIN_BAND_WIDTH)
    return resolved_min, resolved_max


@dataclass(slots=True)
class _SceneCountProxy:
    count: int


@dataclass(slots=True)
class _FrameMetricBookkeeper:
    loss: np.ndarray
    mse: np.ndarray
    psnr: np.ndarray
    visited: np.ndarray

    @classmethod
    def create(cls, frame_count: int) -> "_FrameMetricBookkeeper":
        count = max(int(frame_count), 1)
        return cls(
            loss=np.full((count,), np.nan, dtype=np.float64),
            mse=np.full((count,), np.nan, dtype=np.float64),
            psnr=np.full((count,), np.nan, dtype=np.float64),
            visited=np.zeros((count,), dtype=bool),
        )

    def reset(self) -> None:
        self.loss.fill(np.nan)
        self.mse.fill(np.nan)
        self.psnr.fill(np.nan)
        self.visited.fill(False)

    def update(self, frame_index: int, loss: float, mse: float, psnr: float) -> None:
        idx = int(frame_index)
        self.loss[idx] = float(loss)
        self.mse[idx] = float(mse)
        self.psnr[idx] = float(psnr)
        self.visited[idx] = True

    def mean(self, name: str) -> float:
        values = getattr(self, name)
        valid = self.visited & np.isfinite(values)
        return float(np.mean(values[valid], dtype=np.float64)) if np.any(valid) else float("nan")


@dataclass(slots=True)
class AdamHyperParams:
    position_lr: float = 1e-3; scale_lr: float = 2.5e-4; rotation_lr: float = 1e-3; color_lr: float = 1e-3
    opacity_lr: float = 1e-3; beta1: float = 0.9; beta2: float = 0.999


@dataclass(slots=True)
class StabilityHyperParams:
    grad_component_clip: float = 10.0; grad_norm_clip: float = 10.0; max_update: float = 0.05
    max_scale: float = 3.0; max_anisotropy: float = 32.0; min_opacity: float = 1e-4; max_opacity: float = 0.9999
    position_abs_max: float = 1e4; huge_value: float = 1e8; loss_grad_clip: float = 10.0


@dataclass(slots=True)
class TrainingHyperParams:
    background: tuple[float, float, float] = (1.0, 1.0, 1.0); near: float = 0.1; far: float = 120.0
    background_mode: int = TRAIN_BACKGROUND_MODE_RANDOM; use_target_alpha_mask: bool = False; use_sh: bool = False; sh_band: int = 0
    scale_l2_weight: float = 0.0; scale_abs_reg_weight: float = 0.01; sh1_reg_weight: float = 0.01; opacity_reg_weight: float = 0.01; density_regularizer: float = 0.02; color_non_negative_reg: float = 0.01; depth_ratio_weight: float = 1.0; ssim_weight: float = DEFAULT_SSIM_WEIGHT; ssim_c1: float = DEFAULT_SSIM_C1; ssim_c2: float = DEFAULT_SSIM_C2; max_allowed_density_start: float = 5.0; max_allowed_density: float = 12.0
    refinement_loss_weight: float = 0.25; refinement_target_edge_weight: float = 0.75
    depth_ratio_grad_min: float = 0.0; depth_ratio_grad_max: float = 0.1
    lr_pos_mul: float = 1.0; lr_pos_stage1_mul: float = 0.75; lr_pos_stage2_mul: float = 0.2; lr_pos_stage3_mul: float = 0.2
    lr_sh_mul: float = 0.05; lr_sh_stage1_mul: float = 0.05; lr_sh_stage2_mul: float = 0.05; lr_sh_stage3_mul: float = 0.05
    position_random_step_noise_lr: float = 5e5; position_random_step_opacity_gate_center: float = 0.005; position_random_step_opacity_gate_sharpness: float = 100.0
    lr_schedule_enabled: bool = True; lr_schedule_start_lr: float = 0.005; lr_schedule_stage1_lr: float = 0.002; lr_schedule_stage2_lr: float = 0.001; lr_schedule_end_lr: float = 1.5e-4; lr_schedule_steps: int = 30_000; lr_schedule_stage1_step: int = 3000; lr_schedule_stage2_step: int = 14_000
    refinement_interval: int = 200; refinement_growth_ratio: float = 0.05; refinement_growth_start_step: int = 500; refinement_alpha_cull_threshold: float = 1e-2; refinement_min_contribution_percent: float = DEFAULT_REFINEMENT_MIN_CONTRIBUTION_PERCENT; refinement_min_contribution_decay: float = DEFAULT_REFINEMENT_MIN_CONTRIBUTION_DECAY; refinement_opacity_mul: float = 1.0
    depth_ratio_stage1_weight: float = 0.05; depth_ratio_stage2_weight: float = 0.01; depth_ratio_stage3_weight: float = 0.001
    position_random_step_noise_stage1_lr: float = 466666.6666666667; position_random_step_noise_stage2_lr: float = 416666.6666666667; position_random_step_noise_stage3_lr: float = 0.0
    use_sh_stage1: bool = True; use_sh_stage2: bool = True; use_sh_stage3: bool = True
    sh_band_stage1: int = 1; sh_band_stage2: int = 1; sh_band_stage3: int = 1
    max_gaussians: int = 1_000_000; train_downscale_mode: int = 1; train_auto_start_downscale: int = 16
    train_downscale_base_iters: int = 200; train_downscale_iter_step: int = 50; train_downscale_max_iters: int = 30_000
    train_downscale_factor: int = 1; train_subsample_factor: int = 0

    def __post_init__(self) -> None:
        self.lr_schedule_enabled = bool(self.lr_schedule_enabled)
        self.lr_schedule_start_lr = max(float(self.lr_schedule_start_lr), 1e-8)
        self.lr_schedule_end_lr = max(float(self.lr_schedule_end_lr), 1e-8)
        background = np.asarray(self.background, dtype=np.float32).reshape(3)
        self.background = tuple(float(v) for v in np.clip(background, 0.0, 1.0))
        self.background_mode = TRAIN_BACKGROUND_MODE_RANDOM if int(self.background_mode) == TRAIN_BACKGROUND_MODE_RANDOM else TRAIN_BACKGROUND_MODE_CUSTOM
        self.use_target_alpha_mask = bool(self.use_target_alpha_mask)
        self.sh_band = min(max(int(self.sh_band), 0), 3) if int(self.sh_band) != 0 else (3 if bool(self.use_sh) else 0)
        self.use_sh = self.sh_band > 0
        self.lr_schedule_steps = max(int(self.lr_schedule_steps), 1)
        self.lr_schedule_stage1_step = min(max(int(self.lr_schedule_stage1_step), 0), self.lr_schedule_steps)
        self.lr_schedule_stage2_step = min(max(int(self.lr_schedule_stage2_step), self.lr_schedule_stage1_step), self.lr_schedule_steps)
        self.lr_schedule_stage1_lr = max(float(self.lr_schedule_stage1_lr), 1e-8)
        self.lr_schedule_stage2_lr = max(float(self.lr_schedule_stage2_lr), 1e-8)
        self.lr_pos_mul = max(float(self.lr_pos_mul), 1e-8)
        self.lr_pos_stage1_mul = max(float(self.lr_pos_stage1_mul), 1e-8)
        self.lr_pos_stage2_mul = max(float(self.lr_pos_stage2_mul), 1e-8)
        self.lr_pos_stage3_mul = max(float(self.lr_pos_stage3_mul), 1e-8)
        self.lr_sh_mul = max(float(self.lr_sh_mul), 1e-8)
        self.lr_sh_stage1_mul = max(float(self.lr_sh_stage1_mul), 1e-8)
        self.lr_sh_stage2_mul = max(float(self.lr_sh_stage2_mul), 1e-8)
        self.lr_sh_stage3_mul = max(float(self.lr_sh_stage3_mul), 1e-8)
        self.refinement_interval = max(int(self.refinement_interval), 1)
        self.refinement_growth_ratio = max(float(self.refinement_growth_ratio), 0.0)
        self.refinement_growth_start_step = max(int(self.refinement_growth_start_step), 0)
        self.refinement_alpha_cull_threshold = min(max(float(self.refinement_alpha_cull_threshold), 1e-8), 1.0)
        self.refinement_min_contribution_percent = min(max(float(self.refinement_min_contribution_percent), 0.0), 100.0)
        self.refinement_min_contribution_decay = min(max(float(self.refinement_min_contribution_decay), 0.0), 1.0)
        self.refinement_opacity_mul = min(max(float(self.refinement_opacity_mul), 0.0), 1.0)
        self.refinement_loss_weight = max(float(self.refinement_loss_weight), 0.0)
        self.refinement_target_edge_weight = max(float(self.refinement_target_edge_weight), 0.0)
        self.sh1_reg_weight = max(float(self.sh1_reg_weight), 0.0)
        self.density_regularizer = max(float(self.density_regularizer), 0.0)
        self.color_non_negative_reg = max(float(self.color_non_negative_reg), 0.0)
        self.depth_ratio_weight = max(float(self.depth_ratio_weight), 0.0)
        self.ssim_weight = min(max(float(self.ssim_weight), 0.0), 1.0)
        self.ssim_c1 = max(float(self.ssim_c1), 1e-8)
        self.ssim_c2 = max(float(self.ssim_c2), 1e-8)
        self.depth_ratio_grad_min, self.depth_ratio_grad_max = resolve_depth_ratio_grad_band(self.depth_ratio_grad_min, self.depth_ratio_grad_max)
        self.depth_ratio_stage1_weight = max(float(self.depth_ratio_stage1_weight), 0.0)
        self.depth_ratio_stage2_weight = max(float(self.depth_ratio_stage2_weight), 0.0)
        self.depth_ratio_stage3_weight = max(float(self.depth_ratio_stage3_weight), 0.0)
        self.max_allowed_density_start = max(float(self.max_allowed_density_start), 0.0)
        self.max_allowed_density = max(float(self.max_allowed_density), 0.0)
        self.max_allowed_density = max(self.max_allowed_density, self.max_allowed_density_start)
        self.position_random_step_noise_lr = max(float(self.position_random_step_noise_lr), 0.0)
        self.position_random_step_noise_stage1_lr = max(float(self.position_random_step_noise_stage1_lr), 0.0)
        self.position_random_step_noise_stage2_lr = max(float(self.position_random_step_noise_stage2_lr), 0.0)
        self.position_random_step_noise_stage3_lr = max(float(self.position_random_step_noise_stage3_lr), 0.0)
        self.position_random_step_opacity_gate_center = min(max(float(self.position_random_step_opacity_gate_center), 0.0), 1.0)
        self.position_random_step_opacity_gate_sharpness = max(float(self.position_random_step_opacity_gate_sharpness), 0.0)
        self.sh_band_stage1 = min(max(int(self.sh_band_stage1), 0), 3) if int(self.sh_band_stage1) != 3 else (0 if not bool(self.use_sh_stage1) else 3)
        self.sh_band_stage2 = min(max(int(self.sh_band_stage2), 0), 3) if int(self.sh_band_stage2) != 3 else (0 if not bool(self.use_sh_stage2) else 3)
        self.sh_band_stage3 = min(max(int(self.sh_band_stage3), 0), 3) if int(self.sh_band_stage3) != 3 else (0 if not bool(self.use_sh_stage3) else 3)
        self.use_sh_stage1 = self.sh_band_stage1 > 0
        self.use_sh_stage2 = self.sh_band_stage2 > 0
        self.use_sh_stage3 = self.sh_band_stage3 > 0
        mode = int(self.train_downscale_mode)
        legacy_factor = min(max(int(self.train_downscale_factor), 1), TRAIN_DOWNSCALE_MAX_FACTOR)
        if mode == 1 and legacy_factor != 1:
            mode = legacy_factor
        self.train_downscale_mode = TRAIN_DOWNSCALE_MODE_AUTO if mode == TRAIN_DOWNSCALE_MODE_AUTO else min(max(mode, 1), TRAIN_DOWNSCALE_MAX_FACTOR)
        self.train_auto_start_downscale = min(max(int(self.train_auto_start_downscale), 1), TRAIN_DOWNSCALE_MAX_FACTOR)
        self.train_downscale_base_iters = max(int(self.train_downscale_base_iters), 1)
        self.train_downscale_iter_step = max(int(self.train_downscale_iter_step), 0)
        self.train_downscale_max_iters = max(int(self.train_downscale_max_iters), 1)
        self.train_subsample_factor = TRAIN_SUBSAMPLE_MODE_AUTO if int(self.train_subsample_factor) == TRAIN_SUBSAMPLE_MODE_AUTO else min(max(int(self.train_subsample_factor), 1), TRAIN_SUBSAMPLE_MAX_FACTOR)
        self.train_downscale_factor = resolve_effective_train_downscale_factor(self, 0)


@dataclass(slots=True)
class TrainingState:
    step: int = 0; last_loss: float = float("nan"); avg_loss: float = float("nan"); last_mse: float = float("nan"); avg_mse: float = float("nan"); last_psnr: float = float("nan"); avg_psnr: float = float("nan")
    avg_density_loss: float = float("nan")
    last_frame_index: int = -1; last_instability: str = ""; last_base_lr: float = float("nan")


class GaussianTrainer:
    _LOSS_SLOT_TOTAL = 0
    _LOSS_SLOT_MSE = 1
    _LOSS_SLOT_DENSITY = 2
    _BATCH_STEP_INFO_STRIDE = 4
    _U32_BYTES = 4
    _FLOAT4_BYTES = 16
    _SSIM_FEATURE_CHANNELS = 15
    _PREPASS_CAPACITY_SYNC_INTERVAL = 32
    _REFINEMENT_CAMERA_ROW_COUNT = 6
    _KERNEL_ENTRIES = {
        "downscale_target": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csResampleDownscaledTargetNearest"),
        "clear_loss": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csClearLossBuffer"),
        "ssim_features": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csComputeSSIMFeatures"),
        "loss_forward": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csComputeBlendedLossForward"),
        "ssim_blurred_grads": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csComputeSSIMBlurredGradients"),
        "loss_backward": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csComputeBlendedLossBackward"),
        "cache_step_info": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csCacheTrainingStepInfo"),
        "position_random_step": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csApplyPositionRandomSteps"),
        "clear_clone_counts": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csClearCloneCounts"),
        "clear_refinement_counters": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csClearRefinementCounters"),
        "clamp_refinement_min_screen_size": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csClampRefinementMinScreenSize"),
        "prepare_refinement_counts": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csPrepareRefinementCounts"),
        "rewrite_refinement_splats": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csRewriteRefinementSplats"),
    }

    def _pixel_thread_count(self) -> spy.uint3:
        return thread_count_2d(self.renderer.width, self.renderer.height)

    def _training_background(self) -> np.ndarray:
        return np.asarray(self.training.background, dtype=np.float32).reshape(3)

    def _training_background_seed(self, step: int | None = None) -> int:
        resolved_step = self.state.step if step is None else int(step)
        return int(_refinement_hash_combine(self._seed + 0x9E3779B9, max(resolved_step, 0)))

    def _frame(self, frame_index: int) -> ColmapFrame:
        return self.frames[clamp_index(frame_index, len(self.frames))]

    def make_frame_camera(self, frame_index: int, width: int, height: int) -> Camera:
        return self._make_frame_camera(self._frame(frame_index), int(width), int(height))

    def frame_size(self, frame_index: int) -> tuple[int, int]:
        frame = self._frame(frame_index)
        return int(frame.width), int(frame.height)

    def effective_train_downscale_factor(self, step: int | None = None) -> int:
        resolved_step = self.state.step if step is None else int(step)
        return resolve_effective_train_downscale_factor(self.training, resolved_step)

    def effective_train_subsample_factor(self, frame_index: int = 0, step: int | None = None) -> int:
        resolved_step = self.state.step if step is None else int(step)
        width, height = self.frame_size(frame_index)
        return resolve_train_subsample_factor(self.training, width, height, resolved_step)

    def effective_train_render_factor(self, step: int | None = None, frame_index: int = 0) -> int:
        resolved_step = self.state.step if step is None else int(step)
        width, height = self.frame_size(frame_index)
        return resolve_effective_train_render_factor(self.training, resolved_step, width, height)

    def training_resolution(self, frame_index: int = 0, step: int | None = None) -> tuple[int, int]:
        width, height = self.frame_size(frame_index)
        return resolve_training_resolution(width, height, self.effective_train_render_factor(step, frame_index))

    def current_base_lr(self, step: int | None = None) -> float:
        resolved_step = self.state.step if step is None else int(step)
        return resolve_base_learning_rate(self.training, resolved_step)

    def frame_metrics_snapshot(self) -> dict[str, np.ndarray]:
        return {
            "loss": np.asarray(self._frame_metrics.loss, dtype=np.float64).copy(),
            "mse": np.asarray(self._frame_metrics.mse, dtype=np.float64).copy(),
            "psnr": np.asarray(self._frame_metrics.psnr, dtype=np.float64).copy(),
            "visited": np.asarray(self._frame_metrics.visited, dtype=bool).copy(),
        }

    def effective_refinement_interval(self) -> int:
        return resolve_effective_refinement_interval(self.training, len(self.frames))

    def refinement_due(self, step: int | None = None) -> bool:
        resolved_step = self.state.step if step is None else int(step)
        return should_run_refinement_step(self.training, resolved_step, len(self.frames))

    def clone_probability_threshold(self, splat_count: int | None = None, width: int | None = None, height: int | None = None, step: int | None = None) -> float:
        resolved_splats = self._scene_count if splat_count is None else int(splat_count)
        resolved_width = self.renderer.width if width is None else int(width)
        resolved_height = self.renderer.height if height is None else int(height)
        resolved_step = self.state.step if step is None else int(step)
        return resolve_clone_probability_threshold(self.training, resolved_splats, resolved_width * resolved_height, resolved_step, len(self.frames))

    def get_frame_target_texture(
        self,
        frame_index: int,
        native_resolution: bool = True,
        encoder: spy.CommandEncoder | None = None,
    ) -> spy.Texture:
        frame_index = clamp_index(frame_index, len(self.frames))
        if native_resolution:
            return self._frame_targets_native[frame_index]
        self._ensure_train_target_texture()
        if encoder is None:
            local_encoder = self.device.create_command_encoder()
            self._refresh_train_target(local_encoder, frame_index)
            self.device.submit_command_buffer(local_encoder.finish())
            self.device.wait()
        else:
            self._refresh_train_target(encoder, frame_index)
        return self._require_train_target_texture()

    def _adam_runtime_hparams(self) -> AdamRuntimeHyperParams:
        return AdamRuntimeHyperParams(
            grad_component_clip=float(self.stability.grad_component_clip),
            grad_norm_clip=float(self.stability.grad_norm_clip),
            max_update=float(self.stability.max_update),
            huge_value=float(self.stability.huge_value),
        )

    def _apply_renderer_training_hparams(self, step: int | None = None) -> None:
        resolved_step = 0 if step is None and not hasattr(self, "state") else self.state.step if step is None else int(step)
        self.renderer.sh_band = resolve_sh_band(self.training, resolved_step)

    def _dispatch(self, kernel: str, encoder: spy.CommandEncoder, thread_count: spy.uint3, vars: dict[str, object]) -> None:
        dispatch(
            kernel=self._kernels[kernel],
            thread_count=thread_count,
            vars=vars,
            command_encoder=encoder,
            debug_label=f"Trainer::{kernel}",
            debug_color_index=40 + len(kernel),
        )

    def _native_frame_camera(self, frame_index: int) -> Camera:
        width, height = self.frame_size(frame_index)
        return self.make_frame_camera(frame_index, width, height)

    def _training_sample_vars(self, frame_index: int, step: int | None = None) -> dict[str, object]:
        resolved_step = self.state.step if step is None else int(step)
        frame = self._frame(frame_index)
        subsample_factor = self.effective_train_subsample_factor(frame_index, resolved_step)
        return {
            "g_TrainingSubsample": {
                "enabled": np.uint32(1 if subsample_factor > 1 else 0),
                "factor": np.uint32(self.effective_train_render_factor(resolved_step, frame_index)),
                "nativeWidth": np.uint32(max(int(frame.width), 1)),
                "nativeHeight": np.uint32(max(int(frame.height), 1)),
                "frameIndex": np.uint32(max(int(frame_index), 0)),
                "stepIndex": np.uint32(max(resolved_step, 0)),
            }
        }

    def _dispatch_raster_training_forward(
        self,
        encoder: spy.CommandEncoder,
        frame_camera: Camera,
        background: np.ndarray,
        step: int | None = None,
        frame_index: int = 0,
        native_camera: Camera | None = None,
    ) -> None:
        resolved_step = self.state.step if step is None else int(step)
        self.renderer.rasterize_training_forward_current_scene(
            encoder=encoder,
            camera=frame_camera,
            background=background,
            clone_counts_buffer=self._refinement_buffers["clone_counts"],
            splat_contribution_buffer=self._refinement_buffers["splat_contribution"],
            clone_select_probability=self.clone_probability_threshold(step=resolved_step),
            clone_seed=self._seed + resolved_step,
            training_background_mode=int(self.training.background_mode),
            training_background_seed=self._training_background_seed(resolved_step),
            training_native_camera=self._native_frame_camera(frame_index) if native_camera is None else native_camera,
            training_sample_vars=self._training_sample_vars(frame_index, resolved_step),
        )

    def _dispatch_raster_backward(
        self,
        encoder: spy.CommandEncoder,
        frame_camera: Camera,
        background: np.ndarray,
        step: int | None = None,
        frame_index: int = 0,
        native_camera: Camera | None = None,
    ) -> None:
        resolved_step = self.state.step if step is None else int(step)
        self.renderer.clear_raster_grads_current_scene(encoder)
        self.renderer.rasterize_backward_current_scene(
            encoder=encoder,
            camera=frame_camera,
            background=background,
            output_grad=self.renderer.output_grad_buffer,
            grad_scale=1.0,
            regularizer_grad=self.renderer.work_buffers["training_regularizer_grad"],
            clone_counts_buffer=self._refinement_buffers["clone_counts"],
            clone_select_probability=self.clone_probability_threshold(step=resolved_step),
            clone_seed=self._seed + resolved_step,
            training_background_mode=int(self.training.background_mode),
            training_background_seed=self._training_background_seed(resolved_step),
            training_native_camera=self._native_frame_camera(frame_index) if native_camera is None else native_camera,
            training_sample_vars=self._training_sample_vars(frame_index, resolved_step),
            refinement_loss_weight=float(self.training.refinement_loss_weight),
            refinement_target_edge_weight=float(self.training.refinement_target_edge_weight),
        )

    def _read_loss_metrics(self) -> tuple[float, float, float]:
        values = buffer_to_numpy(self._buffers["loss"], np.float32)
        total = float(values[self._LOSS_SLOT_TOTAL]) if values.size > self._LOSS_SLOT_TOTAL else float("nan")
        mse = float(values[self._LOSS_SLOT_MSE]) if values.size > self._LOSS_SLOT_MSE else float("nan")
        density = float(values[self._LOSS_SLOT_DENSITY]) if values.size > self._LOSS_SLOT_DENSITY else float("nan")
        return total, mse, density

    def _read_batch_step_metrics(self, step_count: int) -> np.ndarray:
        count = max(int(step_count), 0)
        values = buffer_to_numpy(self._buffers["batch_step_info"], np.float32)
        if count <= 0:
            return np.zeros((0, self._BATCH_STEP_INFO_STRIDE), dtype=np.float32)
        return np.asarray(values[: count * self._BATCH_STEP_INFO_STRIDE], dtype=np.float32).reshape(count, self._BATCH_STEP_INFO_STRIDE).copy()

    def _read_refinement_counter(self, name: str) -> int:
        return int(buffer_to_numpy(self._refinement_buffers[name], np.uint32)[0])

    def _refinement_vars(self, *, dst_splat_count: int = 0, append_splat_count: int = 0, survivor_count: int = 0) -> dict[str, object]:
        refinement_threshold = resolve_refinement_min_contribution_percent(self.training, max(self.state.step - 1, 0), len(self.frames))
        return {
            "g_SrcSplatParams": self.renderer.scene_buffers["splat_params"],
            "g_SrcAdamMoments": self.adam_optimizer.buffers["adam_moments"],
            "g_DstSplatParams": self._refinement_buffers["dst_splat_params"],
            "g_DstAdamMoments": self._refinement_buffers["dst_adam_moments"],
            "g_AppendParams": self._refinement_buffers["append_params"],
            "g_CloneCounts": self._refinement_buffers["clone_counts"],
            "g_SplatContribution": self._refinement_buffers["splat_contribution"],
            "g_TotalCloneCounter": self._refinement_buffers["total_clone_counter"],
            "g_AppendCounter": self._refinement_buffers["append_counter"],
            "g_RefinementCameraRows": self._refinement_buffers["camera_rows"],
            "g_SrcSplatCount": int(self._scene_count),
            "g_DstSplatCount": int(max(dst_splat_count, 1)),
            "g_AppendSplatCount": int(max(append_splat_count, 1)),
            "g_SurvivorCount": int(max(survivor_count, 0)),
            "g_RefinementSeed": np.uint32(self._seed + self.state.step),
            "g_RefinementCameraCount": int(len(self.frames)),
            "g_RefinementAlphaCullThreshold": float(self.training.refinement_alpha_cull_threshold),
            "g_RefinementMinContributionThreshold": np.uint32(contribution_fixed_count_from_percent(refinement_threshold, self._observed_contribution_pixel_count)),
            "g_RefinementOpacityMul": float(self.training.refinement_opacity_mul),
            "g_RefinementRadiusScale": float(max(self.renderer.radius_scale, 1e-8)),
        }

    def update_hyperparams(self, adam_hparams: AdamHyperParams, stability_hparams: StabilityHyperParams, training_hparams: TrainingHyperParams) -> None:
        self.adam = adam_hparams
        self.stability = stability_hparams
        self.training = training_hparams
        self._apply_renderer_training_hparams()
        self.training.train_downscale_factor = self.effective_train_downscale_factor(self.state.step)
        self.state.last_base_lr = self.current_base_lr(self.state.step)
        self.adam_optimizer.update_hyperparams(self.adam, self._adam_runtime_hparams())
        self.optimizer.update_hyperparams(self.adam, self.stability)
        self._ensure_refinement_buffers(self._scene_count)
        self._refinement_camera_signature = None
        self._invalidate_downscaled_target()

    def _dispatch_adam_step(self, encoder: spy.CommandEncoder, frame_camera: Camera | None = None) -> None:
        self._dispatch_optimizer_step(encoder, self.state.step + 1, frame_camera)

    def _position_random_step_vars(self, step_index: int) -> dict[str, object]:
        position_lr_mul_scale = resolve_position_lr_mul(self.training, int(step_index)) / max(float(self.training.lr_pos_mul), 1e-8)
        return {
            "g_PositionRandomStepParams": self.renderer.scene_buffers["splat_params"],
            "g_PositionRandomStepSplatCount": int(self._scene_count),
            "g_PositionRandomStepSeed": np.uint32(self._seed + int(step_index)),
            "g_PositionRandomStepNoiseLr": float(resolve_position_random_step_noise_lr(self.training, int(step_index))),
            "g_PositionRandomStepPositionLr": float(self.adam.position_lr * resolve_learning_rate_scale(self.training, int(step_index)) * position_lr_mul_scale),
            "g_PositionRandomStepOpacityGateCenter": float(self.training.position_random_step_opacity_gate_center),
            "g_PositionRandomStepOpacityGateSharpness": float(self.training.position_random_step_opacity_gate_sharpness),
        }

    def _dispatch_position_random_steps(self, encoder: spy.CommandEncoder, step_index: int) -> None:
        if self._scene_count <= 0 or resolve_position_random_step_noise_lr(self.training, int(step_index)) <= 0.0:
            return
        self._dispatch(
            "position_random_step",
            encoder,
            thread_count_1d(self._scene_count),
            self._position_random_step_vars(step_index),
        )

    def __init__(
        self,
        device: spy.Device,
        renderer: GaussianRenderer,
        scene: GaussianScene | None,
        frames: list[ColmapFrame],
        adam_hparams: AdamHyperParams | None = None,
        stability_hparams: StabilityHyperParams | None = None,
        training_hparams: TrainingHyperParams | None = None,
        seed: int = 0,
        scene_count: int | None = None,
        upload_initial_scene: bool = True,
        init_point_positions: np.ndarray | None = None,
        init_point_colors: np.ndarray | None = None,
        init_point_positions_buffer: spy.Buffer | None = None,
        init_point_colors_buffer: spy.Buffer | None = None,
        init_point_count: int = 0,
        scale_reg_reference: float | None = None,
        frame_targets_native: list[spy.Texture] | None = None,
    ) -> None:
        if not frames:
            raise ValueError("Training requires at least one COLMAP frame.")
        if scene is None and scene_count is None:
            raise ValueError("GaussianTrainer requires either scene or scene_count.")
        self.device, self.renderer, self.frames = device, renderer, frames
        self._seed = int(seed)
        self._scene_count = int(scene.count if scene is not None else max(int(scene_count if scene_count is not None else 0), 1))
        self.scene = _SceneCountProxy(self._scene_count)
        self.adam = AdamHyperParams() if adam_hparams is None else adam_hparams
        self.stability = StabilityHyperParams() if stability_hparams is None else stability_hparams
        self.training = TrainingHyperParams() if training_hparams is None else training_hparams
        self._apply_renderer_training_hparams()
        self.metrics = Metrics(self.device)
        self.adam_optimizer = AdamOptimizer(self.device, self.adam, self._adam_runtime_hparams())
        self.optimizer = GaussianOptimizer(self.device, self.renderer, self.adam, self.stability)
        self.compute_debug_grad_norm = False
        self.state = TrainingState()
        self.state.last_base_lr = self.current_base_lr(0)
        self._frame_metrics = _FrameMetricBookkeeper.create(len(self.frames))
        self._kernels = load_compute_items(
            self.device,
            {
                name: ("kernel", shader_path, entry)
                for name, (shader_path, entry) in self._KERNEL_ENTRIES.items()
            },
        )
        self._buffers: dict[str, spy.Buffer] = {}
        self._refinement_buffers: dict[str, spy.Buffer] = {}
        self._splat_capacity = 0
        self._batch_step_capacity = 0
        self._refinement_splat_capacity = 0
        self._refinement_append_capacity = 0
        self._refinement_output_capacity = 0
        self._refinement_camera_capacity = 0
        self._refinement_camera_signature: tuple[int, int, float, float, int] | None = None
        self._scale_reg_reference = float(max(scale_reg_reference, 1e-8)) if scale_reg_reference is not None else self._estimate_scale_reg_reference(scene)
        self._init_point_count = 0
        self._init_point_positions_cpu: np.ndarray | None = None
        self._init_point_colors_cpu: np.ndarray | None = None
        self._frame_targets_native: list[spy.Texture] = []
        self._train_target_texture: spy.Texture | None = None
        self._downscaled_target_key: tuple[int, int, int, int] | None = None
        self._ssim_blur: SeparableGaussianBlur | None = None
        self._ssim_resolution: tuple[int, int] | None = None
        self._observed_contribution_pixel_count = 0
        self._frame_rng = np.random.default_rng(self._seed)
        self._frame_order = np.zeros((len(self.frames),), dtype=np.int32)
        self._frame_cursor = len(self.frames)
        if upload_initial_scene and scene is not None:
            self.renderer.set_scene(scene)
        else:
            self.renderer.bind_scene_count(self._scene_count)
        self._ensure_training_buffers(self._scene_count, 1)
        self._ensure_refinement_buffers(self._scene_count)
        self._zero_optimizer_moments()
        if frame_targets_native is None:
            self._create_dataset_textures()
        else:
            if len(frame_targets_native) != len(self.frames):
                raise ValueError("frame_targets_native must match the number of training frames.")
            self._frame_targets_native = list(frame_targets_native)
        self._bind_or_upload_init_pointcloud(init_point_positions, init_point_colors, init_point_positions_buffer, init_point_colors_buffer, init_point_count)
        self._clear_clone_counts()
        self._reset_frame_order()

    def _estimate_scale_reg_reference(self, scene: GaussianScene | None) -> float:
        if scene is None or scene.count <= 0:
            return 1.0
        scales = np.asarray(scene.scales, dtype=np.float32)
        finite = np.isfinite(scales).all(axis=1)
        return 1.0 if not np.any(finite) else float(np.exp(np.mean(scales[finite], dtype=np.float64)))

    def _ensure_training_buffers(self, splat_count: int, batch_step_count: int = 1) -> None:
        count = max(int(splat_count), 1)
        required_batch_steps = max(int(batch_step_count), 1)
        if self._buffers and count <= self._splat_capacity and required_batch_steps <= self._batch_step_capacity: return
        self._splat_capacity = grow_capacity(count, self._splat_capacity)
        self._batch_step_capacity = grow_capacity(required_batch_steps, self._batch_step_capacity)
        self._buffers.setdefault("loss", alloc_buffer(self.device, size=16, usage=RW_BUFFER_USAGE))
        self._buffers["batch_step_info"] = alloc_buffer(
            self.device,
            size=self._batch_step_capacity * self._BATCH_STEP_INFO_STRIDE * 4,
            usage=RW_BUFFER_USAGE,
        )

    def _ensure_ssim_buffers(self) -> None:
        width = max(int(self.renderer.width), 1)
        height = max(int(self.renderer.height), 1)
        resolution = (width, height)
        if self._ssim_resolution == resolution and self._ssim_blur is not None and all(name in self._buffers for name in ("ssim_moments", "ssim_blurred_moments", "ssim_blurred_feature_grads", "ssim_feature_grads")):
            return
        self._ssim_blur = SeparableGaussianBlur(self.device, width=width, height=height)
        self._buffers["ssim_moments"] = self._ssim_blur.make_buffer(self._SSIM_FEATURE_CHANNELS)
        self._buffers["ssim_blurred_moments"] = self._ssim_blur.make_buffer(self._SSIM_FEATURE_CHANNELS)
        self._buffers["ssim_blurred_feature_grads"] = self._ssim_blur.make_buffer(self._SSIM_FEATURE_CHANNELS)
        self._buffers["ssim_feature_grads"] = self._ssim_blur.make_buffer(self._SSIM_FEATURE_CHANNELS)
        self._ssim_resolution = resolution

    def _expected_refinement_append_count(self, splat_count: int) -> int:
        return max(int(np.ceil(max(int(splat_count), 1) * max(float(self.training.refinement_growth_ratio), 0.0))), 1)

    def _ensure_refinement_buffers(self, splat_count: int, append_count: int | None = None) -> None:
        required_splats = max(int(splat_count), 1)
        required_append = self._expected_refinement_append_count(required_splats) if append_count is None else max(int(append_count), 1)
        required_output = max(required_splats + required_append, 1)
        required_camera_count = max(len(self.frames), 1)
        grow_splats = required_splats > self._refinement_splat_capacity
        grow_append = required_append > self._refinement_append_capacity
        grow_output = required_output > self._refinement_output_capacity
        grow_cameras = required_camera_count > self._refinement_camera_capacity
        if self._refinement_buffers and not grow_splats and not grow_append and not grow_output and not grow_cameras:
            return
        packed_param_bytes = self.renderer.TRAINABLE_PARAM_COUNT * self._U32_BYTES
        if "total_clone_counter" not in self._refinement_buffers:
            self._refinement_buffers["total_clone_counter"] = alloc_buffer(self.device, size=self._U32_BYTES, usage=RW_BUFFER_USAGE)
        if "append_counter" not in self._refinement_buffers:
            self._refinement_buffers["append_counter"] = alloc_buffer(self.device, size=self._U32_BYTES, usage=RW_BUFFER_USAGE)
        if grow_splats or "clone_counts" not in self._refinement_buffers:
            self._refinement_splat_capacity = grow_capacity(required_splats, self._refinement_splat_capacity)
            self._refinement_buffers["clone_counts"] = alloc_buffer(self.device, size=self._refinement_splat_capacity * self._U32_BYTES, usage=RW_BUFFER_USAGE)
            self._refinement_buffers["splat_contribution"] = alloc_buffer(self.device, size=self._refinement_splat_capacity * self._U32_BYTES, usage=RW_BUFFER_USAGE)
        elif "splat_contribution" not in self._refinement_buffers:
            self._refinement_buffers["splat_contribution"] = alloc_buffer(self.device, size=self._refinement_splat_capacity * self._U32_BYTES, usage=RW_BUFFER_USAGE)
        if grow_append or "append_params" not in self._refinement_buffers:
            self._refinement_append_capacity = grow_capacity(required_append, self._refinement_append_capacity)
            self._refinement_buffers["append_params"] = alloc_buffer(self.device, size=self._refinement_append_capacity * packed_param_bytes, usage=RW_BUFFER_USAGE)
        if grow_output or "dst_splat_params" not in self._refinement_buffers:
            self._refinement_output_capacity = grow_capacity(required_output, self._refinement_output_capacity)
            self._refinement_buffers["dst_splat_params"] = alloc_buffer(self.device, size=self._refinement_output_capacity * packed_param_bytes, usage=RW_BUFFER_USAGE)
            self._refinement_buffers["dst_adam_moments"] = alloc_buffer(
                self.device,
                size=self._refinement_output_capacity * self.renderer.TRAINABLE_PARAM_COUNT * self._U32_BYTES * 2,
                usage=RW_BUFFER_USAGE,
            )
        elif "dst_adam_moments" not in self._refinement_buffers:
            self._refinement_buffers["dst_adam_moments"] = alloc_buffer(
                self.device,
                size=max(self._refinement_output_capacity, 1) * self.renderer.TRAINABLE_PARAM_COUNT * self._U32_BYTES * 2,
                usage=RW_BUFFER_USAGE,
            )
        if grow_cameras or "camera_rows" not in self._refinement_buffers:
            self._refinement_camera_capacity = grow_capacity(required_camera_count, self._refinement_camera_capacity)
            self._refinement_buffers["camera_rows"] = alloc_buffer(
                self.device,
                size=self._refinement_camera_capacity * self._REFINEMENT_CAMERA_ROW_COUNT * self._FLOAT4_BYTES,
                usage=RW_BUFFER_USAGE,
            )
            self._refinement_camera_signature = None

    def _refinement_camera_rows(self) -> np.ndarray:
        rows = np.zeros((max(len(self.frames), 1) * self._REFINEMENT_CAMERA_ROW_COUNT, 4), dtype=np.float32)
        for frame_index in range(len(self.frames)):
            frame = self.frames[frame_index]
            width = int(frame.width)
            height = int(frame.height)
            camera = self.make_frame_camera(frame_index, width, height)
            right, up, forward = camera.basis()
            fx, fy = camera.focal_pixels_xy(width, height)
            cx, cy = camera.principal_point(width, height)
            k1, k2 = camera.distortion_coeffs()
            camera_hash = _u32_bits_to_f32(int(_refinement_camera_hash(frame.image_id)))
            base = frame_index * self._REFINEMENT_CAMERA_ROW_COUNT
            rows[base + 0] = np.array([width, height, camera.position[0], camera.position[1]], dtype=np.float32)
            rows[base + 1] = np.array([camera.position[2], fx, fy, cx], dtype=np.float32)
            rows[base + 2] = np.array([cy, camera.near, camera.far, k1], dtype=np.float32)
            rows[base + 3] = np.array([k2, right[0], right[1], right[2]], dtype=np.float32)
            rows[base + 4] = np.array([up[0], up[1], up[2], forward[0]], dtype=np.float32)
            rows[base + 5] = np.array([forward[1], forward[2], camera_hash, 0.0], dtype=np.float32)
        return rows

    def _refinement_camera_signature_value(self) -> tuple[object, ...]:
        signature: list[object] = [
            int(self.renderer.width),
            int(self.renderer.height),
            round(float(self.training.near), 8),
            round(float(self.training.far), 8),
            int(len(self.frames)),
        ]
        for frame in self.frames:
            signature.extend(
                (
                    int(frame.image_id),
                    int(frame.width),
                    int(frame.height),
                    round(float(frame.fx), 6),
                    round(float(frame.fy), 6),
                    round(float(frame.cx), 6),
                    round(float(frame.cy), 6),
                    round(float(frame.k1), 8),
                    round(float(frame.k2), 8),
                    *(round(float(v), 8) for v in np.asarray(frame.q_wxyz, dtype=np.float32).reshape(4)),
                    *(round(float(v), 8) for v in np.asarray(frame.t_xyz, dtype=np.float32).reshape(3)),
                )
            )
        return tuple(signature)

    def _refresh_refinement_camera_buffer(self) -> None:
        signature = self._refinement_camera_signature_value()
        if self._refinement_camera_signature == signature:
            return
        self._ensure_refinement_buffers(self._scene_count)
        self._refinement_buffers["camera_rows"].copy_from_numpy(self._refinement_camera_rows())
        self._refinement_camera_signature = signature

    def _clear_clone_counts(self) -> None:
        enc = self.device.create_command_encoder()
        self._dispatch("clear_clone_counts", enc, spy.uint3(max(self._scene_count, 1), 1, 1), self._refinement_vars())
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()
        self._observed_contribution_pixel_count = 0

    def _run_refinement(self) -> None:
        self._refresh_refinement_camera_buffer()
        enc = self.device.create_command_encoder()
        self._dispatch("clamp_refinement_min_screen_size", enc, spy.uint3(max(self._scene_count, 1), 1, 1), self._refinement_vars())
        self._dispatch("clear_refinement_counters", enc, spy.uint3(1, 1, 1), self._refinement_vars())
        self._dispatch("prepare_refinement_counts", enc, spy.uint3(max(self._scene_count, 1), 1, 1), self._refinement_vars())
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()

        survivor_count = self._read_refinement_counter("append_counter")
        clone_total = self._read_refinement_counter("total_clone_counter")
        max_gaussians = max(int(self.training.max_gaussians), 0)
        clone_headroom = max(max_gaussians - survivor_count, 0) if max_gaussians > 0 else clone_total
        capped_clone_total = min(int(clone_total), int(clone_headroom))
        next_count = int(survivor_count + capped_clone_total)
        if next_count <= 0:
            self._clear_clone_counts()
            return

        self._ensure_refinement_buffers(self._scene_count, capped_clone_total)
        vars = self._refinement_vars(dst_splat_count=next_count, append_splat_count=capped_clone_total, survivor_count=survivor_count)
        enc = self.device.create_command_encoder()
        self._dispatch("clear_refinement_counters", enc, spy.uint3(1, 1, 1), vars)
        self._dispatch("rewrite_refinement_splats", enc, spy.uint3(max(self._scene_count, 1), 1, 1), vars)
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()

        self.renderer.bind_scene_count(next_count)
        self._scene_count = next_count
        self.scene.count = next_count
        self._ensure_training_buffers(self._scene_count, 1)
        self.adam_optimizer.ensure_moment_capacity(self._scene_count * self.renderer.TRAINABLE_PARAM_COUNT)
        copy_enc = self.device.create_command_encoder()
        copy_enc.copy_buffer(
            self.renderer.scene_buffers["splat_params"],
            0,
            self._refinement_buffers["dst_splat_params"],
            0,
            self._scene_count * self.renderer.TRAINABLE_PARAM_COUNT * self._U32_BYTES,
        )
        copy_enc.copy_buffer(
            self.adam_optimizer.buffers["adam_moments"],
            0,
            self._refinement_buffers["dst_adam_moments"],
            0,
            self._scene_count * self.renderer.TRAINABLE_PARAM_COUNT * self._U32_BYTES * 2,
        )
        self.device.submit_command_buffer(copy_enc.finish())
        self.device.wait()
        self._ensure_refinement_buffers(self._scene_count)
        self._clear_clone_counts()

    def _zero_optimizer_moments(self) -> None:
        self.adam_optimizer.zero_moments(self._scene_count * self.renderer.TRAINABLE_PARAM_COUNT)

    def _reset_frame_order(self) -> None:
        self._frame_order = self._frame_rng.permutation(len(self.frames)).astype(np.int32)
        self._frame_cursor = 0

    def _next_frame_index(self) -> int:
        if self._frame_cursor >= len(self.frames):
            self._reset_frame_order()
        frame_index = int(self._frame_order[self._frame_cursor])
        self._frame_cursor += 1
        return frame_index

    def _bind_or_upload_init_pointcloud(
        self,
        positions: np.ndarray | None,
        colors: np.ndarray | None,
        positions_buffer: spy.Buffer | None,
        colors_buffer: spy.Buffer | None,
        point_count: int,
    ) -> None:
        if positions_buffer is not None and colors_buffer is not None and int(point_count) > 0:
            self._init_point_count = int(point_count)
            self._init_point_positions_cpu = np.asarray(positions_buffer.to_numpy(), dtype=np.float32)[: self._init_point_count, :3].copy()
            self._init_point_colors_cpu = np.asarray(colors_buffer.to_numpy(), dtype=np.float32)[: self._init_point_count, :3].copy()
            return
        if positions is None or colors is None:
            return
        pos = np.ascontiguousarray(positions, dtype=np.float32)
        col = np.ascontiguousarray(colors, dtype=np.float32)
        if pos.shape[0] != col.shape[0]:
            raise ValueError("init_point_positions and init_point_colors must have matching row count.")
        self._init_point_count = int(pos.shape[0])
        self._init_point_positions_cpu = pos[:, :3].copy()
        self._init_point_colors_cpu = col[:, :3].copy()

    def _make_frame_camera(self, frame: ColmapFrame, width: int, height: int) -> Camera:
        camera = frame.make_camera(near=float(self.training.near), far=float(self.training.far))
        frame_width, frame_height = max(int(frame.width), 1), max(int(frame.height), 1)
        if int(width) == frame_width and int(height) == frame_height:
            return camera
        scale = {"fx": float(width) / float(frame_width), "fy": float(height) / float(frame_height), "cx": float(width) / float(frame_width), "cy": float(height) / float(frame_height)}
        for name, mul in scale.items():
            value = getattr(camera, name)
            if value is not None:
                setattr(camera, name, float(value) * mul)
        return camera

    def _frame_target_rgba8(self, frame: ColmapFrame) -> np.ndarray:
        return load_training_frame_rgba8(frame)

    def _create_gpu_texture(self, rgba8: np.ndarray) -> spy.Texture:
        tex = alloc_texture_2d(
            self.device,
            format=spy.Format.rgba8_unorm_srgb,
            width=int(rgba8.shape[1]),
            height=int(rgba8.shape[0]),
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
        )
        tex.copy_from_numpy(np.ascontiguousarray(rgba8, dtype=np.uint8))
        return tex

    def _create_dataset_textures(self) -> None:
        self._frame_targets_native = []
        with ThreadPoolExecutor(max_workers=TRAINING_FRAME_LOAD_THREADS, thread_name_prefix="trainer-target") as executor:
            for rgba8 in executor.map(load_training_frame_rgba8, self.frames):
                self._frame_targets_native.append(self._create_gpu_texture(rgba8))

    @property
    def observed_contribution_pixel_count(self) -> int:
        return int(max(self._observed_contribution_pixel_count, 0))

    def _require_train_target_texture(self) -> spy.Texture:
        if self._train_target_texture is None:
            raise RuntimeError("Training target texture is not initialized.")
        return self._train_target_texture

    def _invalidate_downscaled_target(self) -> None:
        self._downscaled_target_key = None

    def _ensure_train_target_texture(self) -> None:
        width = max(int(self.renderer.width), 1)
        height = max(int(self.renderer.height), 1)
        texture = self._train_target_texture
        if texture is not None and int(texture.width) == width and int(texture.height) == height:
            return
        self._train_target_texture = alloc_texture_2d(
            self.device,
            format=spy.Format.rgba32_float,
            width=width,
            height=height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        )
        self._invalidate_downscaled_target()

    def _downscale_vars(self, frame_index: int) -> dict[str, object]:
        frame = self._frame(frame_index)
        factor = self.effective_train_render_factor()
        return {
            "g_DownscaleFactor": int(factor),
            "g_SourceWidth": int(frame.width),
            "g_SourceHeight": int(frame.height),
            "g_TargetWidth": int(self.renderer.width),
            "g_TargetHeight": int(self.renderer.height),
        }

    def _dispatch_downscale_target(self, encoder: spy.CommandEncoder, frame_index: int) -> None:
        self._ensure_train_target_texture()
        self._dispatch(
            "downscale_target",
            encoder,
            self._pixel_thread_count(),
            {
                "g_SourceTarget": self._frame_targets_native[frame_index],
                "g_DownscaledTarget": self._require_train_target_texture(),
                **self._downscale_vars(frame_index),
            },
        )

    def _refresh_train_target(self, encoder: spy.CommandEncoder, frame_index: int) -> None:
        factor = self.effective_train_render_factor()
        key = (
            int(frame_index),
            int(factor),
            int(self.renderer.width),
            int(self.renderer.height),
        )
        if self._downscaled_target_key == key:
            return
        self._dispatch_downscale_target(encoder, frame_index)
        self._downscaled_target_key = key

    def _loss_vars(self, frame_index: int, step: int | None = None) -> dict[str, object]:
        resolved_step = self.state.step if step is None else int(step)
        return {
            "g_Width": int(self.renderer.width),
            "g_Height": int(self.renderer.height),
            "g_InvPixelCount": 1.0 / float(max(self.renderer.width * self.renderer.height, 1)),
            "g_LossGradClip": float(self.stability.loss_grad_clip),
            "g_HugeValue": float(self.stability.huge_value),
            "g_UseTargetAlphaMask": int(bool(self.training.use_target_alpha_mask)),
            "g_DensityRegularizer": float(self.training.density_regularizer),
            "g_DepthRatioWeight": float(resolve_depth_ratio_weight(self.training, resolved_step)),
            "g_SSIMWeight": float(self.training.ssim_weight),
            "g_SSIMC1": float(self.training.ssim_c1),
            "g_SSIMC2": float(self.training.ssim_c2),
            "g_RefinementLossWeight": float(self.training.refinement_loss_weight),
            "g_RefinementTargetEdgeWeight": float(self.training.refinement_target_edge_weight),
            "g_DepthRatioGradMin": float(self.training.depth_ratio_grad_min),
            "g_DepthRatioGradMax": float(self.training.depth_ratio_grad_max),
            "g_MaxAllowedDensity": float(resolve_max_allowed_density(self.training, resolved_step)),
            **self._training_sample_vars(frame_index, resolved_step),
        }

    def _ssim_vars(self) -> dict[str, object]:
        self._ensure_ssim_buffers()
        return {
            "g_SSIMMoments": self._buffers["ssim_moments"],
            "g_SSIMBlurredMoments": self._buffers["ssim_blurred_moments"],
            "g_SSIMBlurredFeatureGrads": self._buffers["ssim_blurred_feature_grads"],
            "g_SSIMFeatureGrads": self._buffers["ssim_feature_grads"],
        }

    def _dispatch_ssim_feature_extraction(self, encoder: spy.CommandEncoder, target_texture: spy.Texture, step: int | None = None, frame_index: int = 0) -> None:
        self._dispatch(
            "ssim_features",
            encoder,
            self._pixel_thread_count(),
            {
                "g_Rendered": self.renderer.output_texture,
                "g_Target": target_texture,
                **self._ssim_vars(),
                **self._loss_vars(frame_index, step),
            },
        )

    def _dispatch_ssim_blur(self, encoder: spy.CommandEncoder, input_name: str, output_name: str) -> None:
        self._ensure_ssim_buffers()
        if self._ssim_blur is None:
            raise RuntimeError("SSIM blur utility is not initialized.")
        self._ssim_blur.blur(encoder, self._buffers[input_name], self._buffers[output_name], self._SSIM_FEATURE_CHANNELS)

    def _dispatch_ssim_blurred_gradients(self, encoder: spy.CommandEncoder, target_texture: spy.Texture, step: int | None = None, frame_index: int = 0) -> None:
        self._dispatch(
            "ssim_blurred_grads",
            encoder,
            self._pixel_thread_count(),
            {
                "g_Target": target_texture,
                **self._ssim_vars(),
                **self._loss_vars(frame_index, step),
            },
        )

    def _dispatch_loss_forward(self, encoder: spy.CommandEncoder, target_texture: spy.Texture, step: int | None = None, frame_index: int = 0) -> None:
        shared = {"g_OutputGrad": self.renderer.output_grad_buffer, "g_RegularizerGrad": self.renderer.work_buffers["training_regularizer_grad"], "g_LossBuffer": self._buffers["loss"], "g_TrainingRgbLoss": self.renderer.work_buffers["training_rgb_loss"], "g_TrainingRgbLossTotal": self.renderer.work_buffers["training_rgb_loss_total"], "g_TrainingTargetEdge": self.renderer.work_buffers["training_target_edge"], "g_TrainingTargetEdgeTotal": self.renderer.work_buffers["training_target_edge_total"], **self._loss_vars(frame_index, step)}
        self._dispatch("clear_loss", encoder, self._pixel_thread_count(), shared)
        self._dispatch_ssim_feature_extraction(encoder, target_texture, step, frame_index)
        self._dispatch_ssim_blur(encoder, "ssim_moments", "ssim_blurred_moments")
        self._dispatch(
            "loss_forward",
            encoder,
            self._pixel_thread_count(),
            {
                "g_Rendered": self.renderer.output_texture,
                "g_RenderedDepthStats": self.renderer.training_depth_stats_texture,
                "g_RenderedDensity": self.renderer.work_buffers["training_density"],
                "g_Target": target_texture,
                "g_OutputGrad": self.renderer.output_grad_buffer,
                "g_RegularizerGrad": self.renderer.work_buffers["training_regularizer_grad"],
                "g_LossBuffer": self._buffers["loss"],
                "g_TrainingRgbLoss": self.renderer.work_buffers["training_rgb_loss"],
                "g_TrainingRgbLossTotal": self.renderer.work_buffers["training_rgb_loss_total"],
                "g_TrainingTargetEdge": self.renderer.work_buffers["training_target_edge"],
                "g_TrainingTargetEdgeTotal": self.renderer.work_buffers["training_target_edge_total"],
                **self._ssim_vars(),
                **self._loss_vars(frame_index, step),
            },
        )

    def _dispatch_loss_backward(self, encoder: spy.CommandEncoder, target_texture: spy.Texture, step: int | None = None, frame_index: int = 0) -> None:
        self._dispatch_ssim_blurred_gradients(encoder, target_texture, step, frame_index)
        self._dispatch_ssim_blur(encoder, "ssim_blurred_feature_grads", "ssim_feature_grads")
        self._dispatch(
            "loss_backward",
            encoder,
            self._pixel_thread_count(),
            {
                "g_Rendered": self.renderer.output_texture,
                "g_RenderedDepthStats": self.renderer.training_depth_stats_texture,
                "g_RenderedDensity": self.renderer.work_buffers["training_density"],
                "g_Target": target_texture,
                "g_OutputGrad": self.renderer.output_grad_buffer,
                "g_RegularizerGrad": self.renderer.work_buffers["training_regularizer_grad"],
                "g_LossBuffer": self._buffers["loss"],
                **self._ssim_vars(),
                **self._loss_vars(frame_index, step),
            },
        )

    def _dispatch_training_forward(
        self,
        encoder: spy.CommandEncoder,
        frame_camera: Camera,
        background: np.ndarray,
        target_texture: spy.Texture,
        step: int | None = None,
        frame_index: int = 0,
        native_camera: Camera | None = None,
    ) -> None:
        with debug_region(encoder, "Trainer Forward", 50):
            self._dispatch_raster_training_forward(encoder, frame_camera, background, step, frame_index, native_camera)
            self._dispatch_loss_forward(encoder, target_texture, step, frame_index)

    def _dispatch_training_backward(
        self,
        encoder: spy.CommandEncoder,
        frame_camera: Camera,
        background: np.ndarray,
        target_texture: spy.Texture,
        step: int | None = None,
        frame_index: int = 0,
        native_camera: Camera | None = None,
    ) -> None:
        with debug_region(encoder, "Trainer Backward", 51):
            self._dispatch_loss_backward(encoder, target_texture, step, frame_index)
            self._dispatch_raster_backward(encoder, frame_camera, background, step, frame_index, native_camera)

    def _dispatch_cache_step_info(self, encoder: spy.CommandEncoder, batch_step_index: int) -> None:
        self._dispatch(
            "cache_step_info",
            encoder,
            spy.uint3(1, 1, 1),
            {
                "g_LossBuffer": self._buffers["loss"],
                "g_BatchStepInfo": self._buffers["batch_step_info"],
                "g_BatchStepIndex": int(batch_step_index),
            },
        )

    def _dispatch_optimizer_step(self, encoder: spy.CommandEncoder, step_index: int, frame_camera: Camera | None = None) -> None:
        with debug_region(encoder, "Trainer Optimizer", 52):
            self.optimizer.update_step(step_index, self.training)
            self.optimizer.dispatch_regularizers(
                encoder,
                scene_buffers=self.renderer.scene_buffers,
                work_buffers=self.renderer.work_buffers,
                loss_buffer=self._buffers["loss"],
                splat_count=self._scene_count,
                training_hparams=self.training,
                scale_reg_reference=self._scale_reg_reference,
                color_non_negative_seed=self._seed + int(step_index),
            )
            self.adam_optimizer.dispatch_step(
                encoder,
                params_buffer=self.renderer.scene_buffers["splat_params"],
                grads_buffer=self.renderer.work_buffers["param_grads"],
                element_count=self._scene_count,
                packed_param_count=self._scene_count * self.renderer.TRAINABLE_PARAM_COUNT,
                param_group_size=self._scene_count,
                param_settings=self.optimizer.param_settings,
                param_settings_count=self.optimizer.param_settings_count,
                step_index=int(step_index),
                debug_element_grad_norm_buffer=self.renderer.work_buffers["debug_grad_norm"] if self.compute_debug_grad_norm else None,
            )
            self.optimizer.dispatch_projection(
                encoder,
                scene_buffers=self.renderer.scene_buffers,
                work_buffers=self.renderer.work_buffers,
                splat_count=self._scene_count,
                training_hparams=self.training,
                scale_reg_reference=self._scale_reg_reference,
                frame_camera=frame_camera,
                width=self.renderer.width if frame_camera is not None else None,
                height=self.renderer.height if frame_camera is not None else None,
            )

    def initialize_scene_from_pointcloud(self, splat_count: int, init_hparams: GaussianInitHyperParams, seed: int) -> None:
        if self._init_point_positions_cpu is None or self._init_point_colors_cpu is None or self._init_point_count <= 0:
            raise RuntimeError("Pointcloud initializer buffers are not available.")
        from ..scene._internal.colmap_ops import point_nn_scales

        count = min(max(int(splat_count), 1), int(self._init_point_count))
        positions = np.ascontiguousarray(self._init_point_positions_cpu[:count], dtype=np.float32)
        colors = np.ascontiguousarray(self._init_point_colors_cpu[:count], dtype=np.float32)
        scales = np.log(np.repeat(point_nn_scales(positions)[:, None], 3, axis=1).astype(np.float32))
        rotations = np.zeros((count, 4), dtype=np.float32)
        rotations[:, 0] = 1.0
        scene = GaussianScene(
            positions=positions,
            scales=scales,
            rotations=rotations,
            opacities=np.full((count,), float(np.clip(init_hparams.initial_opacity, 1e-4, 0.9999)), dtype=np.float32),
            colors=colors,
            sh_coeffs=np.pad(rgb_to_sh0(colors)[:, None, :], ((0, 0), (0, SUPPORTED_SH_COEFF_COUNT - 1), (0, 0))).astype(np.float32, copy=False),
        )
        self._scene_count, self.scene = count, _SceneCountProxy(count)
        self.renderer.set_scene(scene)
        self._ensure_training_buffers(self._scene_count, 1)
        self._ensure_refinement_buffers(self._scene_count)
        self._refinement_camera_signature = None
        self._scale_reg_reference = float(max(np.exp(np.median(scales[:, 0])), 1e-8))
        self._zero_optimizer_moments()
        self.state = TrainingState()
        self.state.last_base_lr = self.current_base_lr(0)
        self._frame_metrics.reset()
        self._frame_rng = np.random.default_rng(int(seed))
        self._clear_clone_counts()
        self._reset_frame_order()
        self._invalidate_downscaled_target()

    def rebind_renderer(self, renderer: GaussianRenderer) -> None:
        self.renderer = renderer
        self.optimizer.renderer = renderer
        self.training.train_downscale_factor = self.effective_train_downscale_factor(self.state.step)
        self._ensure_training_buffers(self._scene_count, 1)
        self._ensure_refinement_buffers(self._scene_count)
        self._refinement_camera_signature = None
        self._clear_clone_counts()
        self._ensure_train_target_texture()
        self._invalidate_downscaled_target()

    def read_live_scene(self) -> GaussianScene:
        groups = self.renderer.read_scene_groups(self._scene_count)
        color_alpha = np.asarray(groups["color_alpha"], dtype=np.float32)
        sh_coeffs = pad_sh_coeffs(groups["sh_coeffs"], SUPPORTED_SH_COEFF_COUNT)
        colors = sh_coeffs_to_display_colors(sh_coeffs)
        opacities = np.reciprocal(1.0 + np.exp(-color_alpha[:, 3])).astype(np.float32, copy=False)
        return GaussianScene(
            positions=np.asarray(groups["positions"][:, :3], dtype=np.float32),
            scales=np.asarray(groups["scales"][:, :3], dtype=np.float32),
            rotations=np.asarray(groups["rotations"], dtype=np.float32),
            opacities=np.asarray(opacities, dtype=np.float32),
            colors=np.asarray(colors, dtype=np.float32),
            sh_coeffs=sh_coeffs,
        )

    def _maybe_sync_prepass_capacity(self, frame_camera: Camera, training_step: int) -> None:
        interval = max(int(self._PREPASS_CAPACITY_SYNC_INTERVAL), 1)
        if int(training_step) % interval != 0:
            return
        self.renderer.sync_prepass_capacity_for_current_scene(frame_camera)

    def step_batch(self, step_count: int) -> int:
        requested = max(int(step_count), 0)
        if requested <= 0:
            return 0

        first_factor = self.effective_train_downscale_factor(self.state.step)
        batch_steps = 0
        while batch_steps < requested and self.effective_train_downscale_factor(self.state.step + batch_steps) == first_factor:
            if batch_steps > 0 and self.refinement_due(self.state.step + batch_steps):
                break
            batch_steps += 1
        if batch_steps <= 0:
            return 0

        self.training.train_downscale_factor = first_factor
        self._ensure_training_buffers(self._scene_count, batch_steps)
        frame_indices: list[int] = []
        enc = self.device.create_command_encoder()
        for batch_index in range(batch_steps):
            background = self._training_background()
            training_step = self.state.step + batch_index
            frame_index = self._next_frame_index()
            frame_indices.append(frame_index)
            frame_camera = self.make_frame_camera(frame_index, self.renderer.width, self.renderer.height)
            native_camera = self._native_frame_camera(frame_index)
            self._apply_renderer_training_hparams(training_step)
            self._maybe_sync_prepass_capacity(frame_camera, training_step)
            self.renderer.record_prepass_for_current_scene(enc, frame_camera)
            target_texture = self.get_frame_target_texture(frame_index, native_resolution=self.effective_train_subsample_factor(frame_index) > 1, encoder=enc)
            self._dispatch_training_forward(enc, frame_camera, background, target_texture, training_step, frame_index, native_camera)
            self._dispatch_training_backward(enc, frame_camera, background, target_texture, training_step, frame_index, native_camera)
            self._dispatch_optimizer_step(enc, self.state.step + batch_index + 1, frame_camera)
            self._dispatch_position_random_steps(enc, self.state.step + batch_index + 1)
            self._dispatch_cache_step_info(enc, batch_index)
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()
        self._observed_contribution_pixel_count += batch_steps * max(int(self.renderer.width) * int(self.renderer.height), 0)

        step_metrics = self._read_batch_step_metrics(batch_steps)
        had_nonfinite = False
        for batch_index, frame_index in enumerate(frame_indices):
            loss = float(step_metrics[batch_index, self._LOSS_SLOT_TOTAL])
            image_mse = float(step_metrics[batch_index, self._LOSS_SLOT_MSE])
            had_nonfinite = had_nonfinite or not np.isfinite(loss)
            self.state.step += 1
            self.state.last_base_lr = self.current_base_lr(self.state.step)
            self.state.last_frame_index = frame_index
            self.state.last_loss = loss
            self.state.last_mse = image_mse
            self.state.last_psnr = float(psnr_from_mse(image_mse))
            self._frame_metrics.update(frame_index, self.state.last_loss, self.state.last_mse, self.state.last_psnr)
        if had_nonfinite:
            self.state.last_instability = "Non-finite loss after batched ADAM; moments reset."
            self._zero_optimizer_moments()
        else:
            self.state.last_instability = ""
        self.state.avg_loss = self._frame_metrics.mean("loss")
        self.state.avg_mse = self._frame_metrics.mean("mse")
        self.state.avg_psnr = self._frame_metrics.mean("psnr")
        self.state.avg_density_loss = float(np.mean(step_metrics[:, self._LOSS_SLOT_DENSITY], dtype=np.float64)) if batch_steps > 0 else float("nan")
        self.training.train_downscale_factor = self.effective_train_downscale_factor(self.state.step)
        if self.refinement_due(self.state.step):
            self._run_refinement()
        return batch_steps

    def step(self) -> float:
        self.step_batch(1)
        return float(self.state.last_loss)

    @property
    def refinement_buffers(self) -> dict[str, spy.Buffer]:
        return self._refinement_buffers
