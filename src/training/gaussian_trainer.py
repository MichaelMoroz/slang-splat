from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import slangpy as spy

from ..common import SHADER_ROOT, buffer_to_numpy, clamp_index, debug_region, dispatch, thread_count_2d
from ..metrics import Metrics, psnr_from_mse
from ..renderer import Camera, GaussianRenderer
from ..scene import ColmapFrame, GaussianInitHyperParams, GaussianScene
from .adam import AdamOptimizer, AdamRuntimeHyperParams
from .optimizer import GaussianOptimizer
from .schedule import resolve_clone_probability_threshold, resolve_cosine_base_learning_rate, should_run_maintenance_step

TRAIN_DOWNSCALE_MODE_AUTO = 0
TRAIN_DOWNSCALE_MAX_FACTOR = 16
_SH_C0 = 0.28209479177387814


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
    grad_component_clip: float = 10.0; grad_norm_clip: float = 10.0; max_update: float = 0.05; min_scale: float = 1e-3
    max_scale: float = 3.0; max_anisotropy: float = 10.0; min_opacity: float = 1e-4; max_opacity: float = 0.9999
    position_abs_max: float = 1e4; huge_value: float = 1e8; min_inv_scale: float = 1e-6; loss_grad_clip: float = 10.0


@dataclass(slots=True)
class TrainingHyperParams:
    background: tuple[float, float, float] = (0.0, 0.0, 0.0); near: float = 0.1; far: float = 120.0
    random_background: bool = False
    scale_l2_weight: float = 0.0; scale_abs_reg_weight: float = 0.01; opacity_reg_weight: float = 0.01; depth_ratio_weight: float = 0.005
    lr_schedule_enabled: bool = True; lr_schedule_start_lr: float = 1e-3; lr_schedule_end_lr: float = 1e-4; lr_schedule_steps: int = 30_000
    maintenance_interval: int = 200; maintenance_growth_ratio: float = 0.05; maintenance_alpha_cull_threshold: float = 1e-2
    max_gaussians: int = 2_000_000; train_downscale_mode: int = 1; train_auto_start_downscale: int = 16
    train_downscale_base_iters: int = 200; train_downscale_iter_step: int = 50; train_downscale_max_iters: int = 30_000
    train_downscale_factor: int = 1

    def __post_init__(self) -> None:
        self.lr_schedule_enabled = bool(self.lr_schedule_enabled)
        self.lr_schedule_start_lr = max(float(self.lr_schedule_start_lr), 1e-8)
        self.lr_schedule_end_lr = max(float(self.lr_schedule_end_lr), 1e-8)
        self.random_background = bool(self.random_background)
        self.lr_schedule_steps = max(int(self.lr_schedule_steps), 1)
        self.maintenance_interval = max(int(self.maintenance_interval), 1)
        self.maintenance_growth_ratio = max(float(self.maintenance_growth_ratio), 0.0)
        self.maintenance_alpha_cull_threshold = min(max(float(self.maintenance_alpha_cull_threshold), 1e-8), 1.0)
        self.depth_ratio_weight = max(float(self.depth_ratio_weight), 0.0)
        mode = int(self.train_downscale_mode)
        legacy_factor = min(max(int(self.train_downscale_factor), 1), TRAIN_DOWNSCALE_MAX_FACTOR)
        if mode == 1 and legacy_factor != 1:
            mode = legacy_factor
        self.train_downscale_mode = TRAIN_DOWNSCALE_MODE_AUTO if mode == TRAIN_DOWNSCALE_MODE_AUTO else min(max(mode, 1), TRAIN_DOWNSCALE_MAX_FACTOR)
        self.train_auto_start_downscale = min(max(int(self.train_auto_start_downscale), 1), TRAIN_DOWNSCALE_MAX_FACTOR)
        self.train_downscale_base_iters = max(int(self.train_downscale_base_iters), 1)
        self.train_downscale_iter_step = max(int(self.train_downscale_iter_step), 0)
        self.train_downscale_max_iters = max(int(self.train_downscale_max_iters), 1)
        self.train_downscale_factor = resolve_effective_train_downscale_factor(self, 0)


@dataclass(slots=True)
class TrainingState:
    step: int = 0; last_loss: float = float("nan"); avg_loss: float = float("nan"); last_mse: float = float("nan"); avg_mse: float = float("nan"); last_psnr: float = float("nan"); avg_psnr: float = float("nan")
    last_depth_ratio_loss: float = float("nan"); avg_depth_ratio_loss: float = float("nan")
    last_frame_index: int = -1; last_instability: str = ""; last_base_lr: float = float("nan")


class GaussianTrainer:
    _LOSS_SLOT_TOTAL = 0
    _LOSS_SLOT_MSE = 1
    _LOSS_SLOT_DEPTH_RATIO = 2
    _BATCH_STEP_INFO_STRIDE = 4
    _U32_BYTES = 4
    _KERNEL_ENTRIES = {
        "downscale_target": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csResampleDownscaledTargetNearest"),
        "clear_loss": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csClearLossBuffer"),
        "loss_forward": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csComputeL1LossForward"),
        "finalize_depth_ratio_loss": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csFinalizeDepthRatioLoss"),
        "loss_backward": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csComputeL1LossBackward"),
        "cache_step_info": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csCacheTrainingStepInfo"),
        "clear_clone_counts": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csClearCloneCounts"),
        "clear_maintenance_counters": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csClearMaintenanceCounters"),
        "prepare_maintenance_counts": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csPrepareMaintenanceCounts"),
        "rewrite_maintenance_splats": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csRewriteMaintenanceSplats"),
    }

    def _pixel_thread_count(self) -> spy.uint3:
        return thread_count_2d(self.renderer.width, self.renderer.height)

    def _training_background(self) -> np.ndarray:
        if not bool(self.training.random_background):
            return np.asarray(self.training.background, dtype=np.float32).reshape(3)
        return np.asarray(self._background_rng.random(3), dtype=np.float32)

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

    def training_resolution(self, frame_index: int = 0, step: int | None = None) -> tuple[int, int]:
        width, height = self.frame_size(frame_index)
        return resolve_training_resolution(width, height, self.effective_train_downscale_factor(step))

    def current_base_lr(self, step: int | None = None) -> float:
        resolved_step = self.state.step if step is None else int(step)
        return resolve_cosine_base_learning_rate(self.training, resolved_step)

    def maintenance_due(self, step: int | None = None) -> bool:
        resolved_step = self.state.step if step is None else int(step)
        return should_run_maintenance_step(self.training, resolved_step)

    def clone_probability_threshold(self, splat_count: int | None = None, width: int | None = None, height: int | None = None) -> float:
        resolved_splats = self._scene_count if splat_count is None else int(splat_count)
        resolved_width = self.renderer.width if width is None else int(width)
        resolved_height = self.renderer.height if height is None else int(height)
        return resolve_clone_probability_threshold(self.training, resolved_splats, resolved_width * resolved_height)

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

    def _dispatch(self, kernel: str, encoder: spy.CommandEncoder, thread_count: spy.uint3, vars: dict[str, object]) -> None:
        dispatch(
            kernel=self._kernels[kernel],
            thread_count=thread_count,
            vars=vars,
            command_encoder=encoder,
            debug_label=f"Trainer::{kernel}",
            debug_color_index=40 + len(kernel),
        )

    def _dispatch_raster_training_forward(self, encoder: spy.CommandEncoder, frame_camera: Camera, background: np.ndarray) -> None:
        self.renderer.rasterize_training_forward_current_scene(
            encoder=encoder,
            camera=frame_camera,
            background=background,
            clone_counts_buffer=self._maintenance_buffers["clone_counts"],
            clone_select_probability=self.clone_probability_threshold(),
            clone_seed=self._seed + self.state.step,
        )

    def _dispatch_raster_backward(self, encoder: spy.CommandEncoder, frame_camera: Camera, background: np.ndarray) -> None:
        self.renderer.clear_raster_grads_current_scene(encoder)
        self.renderer.rasterize_backward_current_scene(
            encoder=encoder,
            camera=frame_camera,
            background=background,
            output_grad=self.renderer.output_grad_buffer,
            grad_scale=1.0,
            depth_ratio_grad=self.renderer.work_buffers["training_depth_ratio_grad"],
        )

    def _read_loss_metrics(self) -> tuple[float, float, float]:
        values = buffer_to_numpy(self._buffers["loss"], np.float32)
        total = float(values[self._LOSS_SLOT_TOTAL]) if values.size > self._LOSS_SLOT_TOTAL else float("nan")
        mse = float(values[self._LOSS_SLOT_MSE]) if values.size > self._LOSS_SLOT_MSE else float("nan")
        depth_ratio = float(values[self._LOSS_SLOT_DEPTH_RATIO]) if values.size > self._LOSS_SLOT_DEPTH_RATIO else float("nan")
        return total, mse, depth_ratio

    def _read_batch_step_metrics(self, step_count: int) -> np.ndarray:
        count = max(int(step_count), 0)
        values = buffer_to_numpy(self._buffers["batch_step_info"], np.float32)
        if count <= 0:
            return np.zeros((0, self._BATCH_STEP_INFO_STRIDE), dtype=np.float32)
        return np.asarray(values[: count * self._BATCH_STEP_INFO_STRIDE], dtype=np.float32).reshape(count, self._BATCH_STEP_INFO_STRIDE).copy()

    def _read_maintenance_counter(self, name: str) -> int:
        return int(buffer_to_numpy(self._maintenance_buffers[name], np.uint32)[0])

    def _maintenance_vars(self, *, dst_splat_count: int = 0, append_splat_count: int = 0, survivor_count: int = 0) -> dict[str, object]:
        return {
            "g_SrcSplatParams": self.renderer.scene_buffers["splat_params"],
            "g_SrcAdamMoments": self.adam_optimizer.buffers["adam_moments"],
            "g_DstSplatParams": self._maintenance_buffers["dst_splat_params"],
            "g_DstAdamMoments": self._maintenance_buffers["dst_adam_moments"],
            "g_AppendParams": self._maintenance_buffers["append_params"],
            "g_CloneCounts": self._maintenance_buffers["clone_counts"],
            "g_TotalCloneCounter": self._maintenance_buffers["total_clone_counter"],
            "g_AppendCounter": self._maintenance_buffers["append_counter"],
            "g_SrcSplatCount": int(self._scene_count),
            "g_DstSplatCount": int(max(dst_splat_count, 1)),
            "g_AppendSplatCount": int(max(append_splat_count, 1)),
            "g_SurvivorCount": int(max(survivor_count, 0)),
            "g_MaintenanceSeed": np.uint32(self._seed + self.state.step),
            "g_MaintenanceAlphaCullThreshold": float(self.training.maintenance_alpha_cull_threshold),
        }

    def update_hyperparams(self, adam_hparams: AdamHyperParams, stability_hparams: StabilityHyperParams, training_hparams: TrainingHyperParams) -> None:
        self.adam = adam_hparams
        self.stability = stability_hparams
        self.training = training_hparams
        self.training.train_downscale_factor = self.effective_train_downscale_factor(self.state.step)
        self.state.last_base_lr = self.current_base_lr(self.state.step)
        self.adam_optimizer.update_hyperparams(self.adam, self._adam_runtime_hparams())
        self.optimizer.update_hyperparams(self.adam, self.stability)
        self._ensure_maintenance_buffers(self._scene_count)
        self._invalidate_downscaled_target()

    def _dispatch_adam_step(self, encoder: spy.CommandEncoder) -> None:
        self._dispatch_optimizer_step(encoder, self.state.step + 1)

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
        self.metrics = Metrics(self.device)
        self.adam_optimizer = AdamOptimizer(self.device, self.adam, self._adam_runtime_hparams())
        self.optimizer = GaussianOptimizer(self.device, self.renderer, self.adam, self.stability)
        self.compute_debug_grad_norm = False
        self.state = TrainingState()
        self.state.last_base_lr = self.current_base_lr(0)
        self._frame_metrics = _FrameMetricBookkeeper.create(len(self.frames))
        self._kernels = {
            name: self.device.create_compute_kernel(self.device.load_program(str(shader_path), [entry]))
            for name, (shader_path, entry) in self._KERNEL_ENTRIES.items()
        }
        self._buffers: dict[str, spy.Buffer] = {}
        self._maintenance_buffers: dict[str, spy.Buffer] = {}
        self._splat_capacity = 0
        self._batch_step_capacity = 0
        self._maintenance_splat_capacity = 0
        self._maintenance_append_capacity = 0
        self._maintenance_output_capacity = 0
        self._scale_reg_reference = float(max(scale_reg_reference, 1e-8)) if scale_reg_reference is not None else self._estimate_scale_reg_reference(scene)
        self._init_point_positions_buffer: spy.Buffer | None = None
        self._init_point_colors_buffer: spy.Buffer | None = None
        self._init_point_count = 0
        self._init_point_positions_cpu: np.ndarray | None = None
        self._init_point_colors_cpu: np.ndarray | None = None
        self._frame_targets_native: list[spy.Texture] = []
        self._train_target_texture: spy.Texture | None = None
        self._downscaled_target_key: tuple[int, int, int, int] | None = None
        self._frame_rng = np.random.default_rng(self._seed)
        self._background_rng = np.random.default_rng(self._seed + 0x9E3779B9)
        self._frame_order = np.zeros((len(self.frames),), dtype=np.int32)
        self._frame_cursor = len(self.frames)
        if upload_initial_scene and scene is not None:
            self.renderer.set_scene(scene)
        else:
            self.renderer.bind_scene_count(self._scene_count)
        self._ensure_training_buffers(self._scene_count, 1)
        self._ensure_maintenance_buffers(self._scene_count)
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
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
        self._splat_capacity = max(count, max(self._splat_capacity, 1) + max(self._splat_capacity, 1) // 2)
        self._batch_step_capacity = max(required_batch_steps, max(self._batch_step_capacity, 1) + max(self._batch_step_capacity, 1) // 2)
        self._buffers.setdefault("loss", self.device.create_buffer(size=16, usage=usage))
        self._buffers.setdefault("depth_ratio_stats", self.device.create_buffer(size=16, usage=usage))
        self._buffers["batch_step_info"] = self.device.create_buffer(size=self._batch_step_capacity * self._BATCH_STEP_INFO_STRIDE * 4, usage=usage)

    def _expected_maintenance_append_count(self, splat_count: int) -> int:
        return max(int(np.ceil(max(int(splat_count), 1) * max(float(self.training.maintenance_growth_ratio), 0.0))), 1)

    def _ensure_maintenance_buffers(self, splat_count: int, append_count: int | None = None) -> None:
        required_splats = max(int(splat_count), 1)
        required_append = self._expected_maintenance_append_count(required_splats) if append_count is None else max(int(append_count), 1)
        required_output = max(required_splats + required_append, 1)
        grow_splats = required_splats > self._maintenance_splat_capacity
        grow_append = required_append > self._maintenance_append_capacity
        grow_output = required_output > self._maintenance_output_capacity
        if self._maintenance_buffers and not grow_splats and not grow_append and not grow_output:
            return
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
        packed_param_bytes = self.renderer.TRAINABLE_PARAM_COUNT * self._U32_BYTES
        if "total_clone_counter" not in self._maintenance_buffers:
            self._maintenance_buffers["total_clone_counter"] = self.device.create_buffer(size=self._U32_BYTES, usage=usage)
        if "append_counter" not in self._maintenance_buffers:
            self._maintenance_buffers["append_counter"] = self.device.create_buffer(size=self._U32_BYTES, usage=usage)
        if grow_splats or "clone_counts" not in self._maintenance_buffers:
            self._maintenance_splat_capacity = max(required_splats, max(self._maintenance_splat_capacity, 1) + max(self._maintenance_splat_capacity, 1) // 2)
            self._maintenance_buffers["clone_counts"] = self.device.create_buffer(size=self._maintenance_splat_capacity * self._U32_BYTES, usage=usage)
        if grow_append or "append_params" not in self._maintenance_buffers:
            self._maintenance_append_capacity = max(required_append, max(self._maintenance_append_capacity, 1) + max(self._maintenance_append_capacity, 1) // 2)
            self._maintenance_buffers["append_params"] = self.device.create_buffer(size=self._maintenance_append_capacity * packed_param_bytes, usage=usage)
        if grow_output or "dst_splat_params" not in self._maintenance_buffers:
            self._maintenance_output_capacity = max(required_output, max(self._maintenance_output_capacity, 1) + max(self._maintenance_output_capacity, 1) // 2)
            self._maintenance_buffers["dst_splat_params"] = self.device.create_buffer(size=self._maintenance_output_capacity * packed_param_bytes, usage=usage)
            self._maintenance_buffers["dst_adam_moments"] = self.device.create_buffer(size=self._maintenance_output_capacity * self.renderer.TRAINABLE_PARAM_COUNT * self._U32_BYTES * 2, usage=usage)
        elif "dst_adam_moments" not in self._maintenance_buffers:
            self._maintenance_buffers["dst_adam_moments"] = self.device.create_buffer(size=max(self._maintenance_output_capacity, 1) * self.renderer.TRAINABLE_PARAM_COUNT * self._U32_BYTES * 2, usage=usage)

    def _clear_clone_counts(self) -> None:
        enc = self.device.create_command_encoder()
        self._dispatch("clear_clone_counts", enc, spy.uint3(max(self._scene_count, 1), 1, 1), self._maintenance_vars())
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()

    def _run_maintenance(self) -> None:
        enc = self.device.create_command_encoder()
        self._dispatch("clear_maintenance_counters", enc, spy.uint3(1, 1, 1), self._maintenance_vars())
        self._dispatch("prepare_maintenance_counts", enc, spy.uint3(max(self._scene_count, 1), 1, 1), self._maintenance_vars())
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()

        survivor_count = self._read_maintenance_counter("append_counter")
        clone_total = self._read_maintenance_counter("total_clone_counter")
        max_gaussians = max(int(self.training.max_gaussians), 0)
        clone_headroom = max(max_gaussians - survivor_count, 0) if max_gaussians > 0 else clone_total
        capped_clone_total = min(int(clone_total), int(clone_headroom))
        next_count = int(survivor_count + capped_clone_total)
        if next_count <= 0:
            self._clear_clone_counts()
            return

        self._ensure_maintenance_buffers(self._scene_count, capped_clone_total)
        vars = self._maintenance_vars(dst_splat_count=next_count, append_splat_count=capped_clone_total, survivor_count=survivor_count)
        enc = self.device.create_command_encoder()
        self._dispatch("clear_maintenance_counters", enc, spy.uint3(1, 1, 1), vars)
        self._dispatch("rewrite_maintenance_splats", enc, spy.uint3(max(self._scene_count, 1), 1, 1), vars)
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
            self._maintenance_buffers["dst_splat_params"],
            0,
            self._scene_count * self.renderer.TRAINABLE_PARAM_COUNT * self._U32_BYTES,
        )
        copy_enc.copy_buffer(
            self.adam_optimizer.buffers["adam_moments"],
            0,
            self._maintenance_buffers["dst_adam_moments"],
            0,
            self._scene_count * self.renderer.TRAINABLE_PARAM_COUNT * self._U32_BYTES * 2,
        )
        self.device.submit_command_buffer(copy_enc.finish())
        self.device.wait()
        self._ensure_maintenance_buffers(self._scene_count)
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

    def _create_init_point_buffer(self, points: np.ndarray) -> spy.Buffer:
        packed = np.ascontiguousarray(points, dtype=np.float32)
        if packed.ndim != 2 or packed.shape[1] < 3:
            raise ValueError("Point table must be [N, >=3] float array.")
        packed4 = np.pad(packed[:, :3], ((0, 0), (0, 1)))
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
        buffer = self.device.create_buffer(size=max(packed4.shape[0], 1) * 16, usage=usage)
        if packed4.shape[0] > 0:
            buffer.copy_from_numpy(packed4)
        return buffer

    def _bind_or_upload_init_pointcloud(
        self,
        positions: np.ndarray | None,
        colors: np.ndarray | None,
        positions_buffer: spy.Buffer | None,
        colors_buffer: spy.Buffer | None,
        point_count: int,
    ) -> None:
        if positions_buffer is not None and colors_buffer is not None and int(point_count) > 0:
            self._init_point_positions_buffer, self._init_point_colors_buffer, self._init_point_count = positions_buffer, colors_buffer, int(point_count)
            self._init_point_positions_cpu = np.asarray(positions_buffer.to_numpy(), dtype=np.float32)[: self._init_point_count, :3].copy()
            self._init_point_colors_cpu = np.asarray(colors_buffer.to_numpy(), dtype=np.float32)[: self._init_point_count, :3].copy()
            return
        if positions is None or colors is None:
            return
        pos = np.ascontiguousarray(positions, dtype=np.float32)
        col = np.ascontiguousarray(colors, dtype=np.float32)
        if pos.shape[0] != col.shape[0]:
            raise ValueError("init_point_positions and init_point_colors must have matching row count.")
        self._init_point_positions_buffer = self._create_init_point_buffer(pos)
        self._init_point_colors_buffer = self._create_init_point_buffer(col)
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

    def _create_gpu_texture(self, image: Image.Image) -> spy.Texture:
        rgba8 = np.array(image.convert("RGBA"), dtype=np.uint8, order="C", copy=True)
        tex = self.device.create_texture(format=spy.Format.rgba8_unorm_srgb, width=int(rgba8.shape[1]), height=int(rgba8.shape[0]), usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination)
        tex.copy_from_numpy(np.ascontiguousarray(rgba8, dtype=np.uint8))
        return tex

    def _create_dataset_textures(self) -> None:
        self._frame_targets_native = []
        for frame in self.frames:
            with Image.open(frame.image_path) as pil_image:
                self._frame_targets_native.append(self._create_gpu_texture(pil_image))

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
        self._train_target_texture = self.device.create_texture(
            format=spy.Format.rgba32_float,
            width=width,
            height=height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        )
        self._invalidate_downscaled_target()

    def _downscale_vars(self, frame_index: int) -> dict[str, object]:
        frame = self._frame(frame_index)
        factor = self.effective_train_downscale_factor()
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
        factor = self.effective_train_downscale_factor()
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

    def _loss_vars(self) -> dict[str, object]:
        return {
            "g_Width": int(self.renderer.width),
            "g_Height": int(self.renderer.height),
            "g_InvPixelCount": 1.0 / float(max(self.renderer.width * self.renderer.height, 1)),
            "g_LossGradClip": float(self.stability.loss_grad_clip),
            "g_HugeValue": float(self.stability.huge_value),
            "g_DepthRatioWeight": float(self.training.depth_ratio_weight),
        }

    def _dispatch_loss_forward(self, encoder: spy.CommandEncoder, target_texture: spy.Texture) -> None:
        shared = {"g_OutputGrad": self.renderer.output_grad_buffer, "g_DepthRatioGrad": self.renderer.work_buffers["training_depth_ratio_grad"], "g_LossBuffer": self._buffers["loss"], "g_DepthRatioStats": self._buffers["depth_ratio_stats"], **self._loss_vars()}
        self._dispatch("clear_loss", encoder, self._pixel_thread_count(), shared)
        self._dispatch(
            "loss_forward",
            encoder,
            self._pixel_thread_count(),
            {
                "g_Rendered": self.renderer.output_texture,
                "g_RenderedDepthRatio": self.renderer.work_buffers["training_depth_ratio"],
                "g_Target": target_texture,
                "g_OutputGrad": self.renderer.output_grad_buffer,
                "g_DepthRatioGrad": self.renderer.work_buffers["training_depth_ratio_grad"],
                "g_LossBuffer": self._buffers["loss"],
                "g_DepthRatioStats": self._buffers["depth_ratio_stats"],
                **self._loss_vars(),
            },
        )
        self._dispatch(
            "finalize_depth_ratio_loss",
            encoder,
            spy.uint3(1, 1, 1),
            {
                "g_LossBuffer": self._buffers["loss"],
                "g_DepthRatioStats": self._buffers["depth_ratio_stats"],
                **self._loss_vars(),
            },
        )

    def _dispatch_loss_backward(self, encoder: spy.CommandEncoder, target_texture: spy.Texture) -> None:
        self._dispatch(
            "loss_backward",
            encoder,
            self._pixel_thread_count(),
            {
                "g_Rendered": self.renderer.output_texture,
                "g_RenderedDepthRatio": self.renderer.work_buffers["training_depth_ratio"],
                "g_Target": target_texture,
                "g_OutputGrad": self.renderer.output_grad_buffer,
                "g_DepthRatioGrad": self.renderer.work_buffers["training_depth_ratio_grad"],
                "g_LossBuffer": self._buffers["loss"],
                "g_DepthRatioStats": self._buffers["depth_ratio_stats"],
                **self._loss_vars(),
            },
        )

    def _dispatch_training_forward(self, encoder: spy.CommandEncoder, frame_camera: Camera, background: np.ndarray, target_texture: spy.Texture) -> None:
        with debug_region(encoder, "Trainer Forward", 50):
            self._dispatch_raster_training_forward(encoder, frame_camera, background)
            self._dispatch_loss_forward(encoder, target_texture)

    def _dispatch_training_backward(self, encoder: spy.CommandEncoder, frame_camera: Camera, background: np.ndarray, target_texture: spy.Texture) -> None:
        with debug_region(encoder, "Trainer Backward", 51):
            self._dispatch_loss_backward(encoder, target_texture)
            self._dispatch_raster_backward(encoder, frame_camera, background)

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

    def _dispatch_optimizer_step(self, encoder: spy.CommandEncoder, step_index: int) -> None:
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
            sh_coeffs=np.zeros((count, 1, 3), dtype=np.float32),
        )
        self._scene_count, self.scene = count, _SceneCountProxy(count)
        self.renderer.set_scene(scene)
        self._ensure_training_buffers(self._scene_count, 1)
        self._ensure_maintenance_buffers(self._scene_count)
        self._scale_reg_reference = float(max(np.exp(np.median(scales[:, 0])), 1e-8))
        self._zero_optimizer_moments()
        self.state = TrainingState()
        self.state.last_base_lr = self.current_base_lr(0)
        self._frame_metrics.reset()
        self._frame_rng = np.random.default_rng(int(seed))
        self._background_rng = np.random.default_rng(int(seed) + 0x9E3779B9)
        self._clear_clone_counts()
        self._reset_frame_order()
        self._invalidate_downscaled_target()

    def rebind_renderer(self, renderer: GaussianRenderer) -> None:
        self.renderer = renderer
        self.optimizer.renderer = renderer
        self.training.train_downscale_factor = self.effective_train_downscale_factor(self.state.step)
        self._ensure_training_buffers(self._scene_count, 1)
        self._ensure_maintenance_buffers(self._scene_count)
        self._clear_clone_counts()
        self._ensure_train_target_texture()
        self._invalidate_downscaled_target()

    def scale_histogram(self, *, bin_count: int = 64, min_log10: float = -6.0, max_log10: float = 1.0):
        return self.metrics.compute_scale_histogram(
            self.renderer.scene_buffers["splat_params"],
            self._scene_count,
            bin_count=bin_count,
            min_log10=min_log10,
            max_log10=max_log10,
        )

    def read_live_scene(self) -> GaussianScene:
        groups = self.renderer.read_scene_groups(self._scene_count)
        color_alpha = np.asarray(groups["color_alpha"], dtype=np.float32)
        colors = np.clip(color_alpha[:, :3], 0.0, 1.0)
        opacities = np.reciprocal(1.0 + np.exp(-color_alpha[:, 3])).astype(np.float32, copy=False)
        sh_coeffs = np.zeros((self._scene_count, 1, 3), dtype=np.float32)
        sh_coeffs[:, 0, :] = ((colors - 0.5) / _SH_C0).astype(np.float32, copy=False)
        return GaussianScene(
            positions=np.asarray(groups["positions"][:, :3], dtype=np.float32),
            scales=np.asarray(groups["scales"][:, :3], dtype=np.float32),
            rotations=np.asarray(groups["rotations"], dtype=np.float32),
            opacities=np.asarray(opacities, dtype=np.float32),
            colors=np.asarray(colors, dtype=np.float32),
            sh_coeffs=sh_coeffs,
        )

    def anisotropy_histogram(self, *, bin_count: int = 64, min_log10: float = 0.0, max_log10: float = 2.0):
        return self.metrics.compute_anisotropy_histogram(
            self.renderer.scene_buffers["splat_params"],
            self._scene_count,
            bin_count=bin_count,
            min_log10=min_log10,
            max_log10=max_log10,
        )

    def step_batch(self, step_count: int) -> int:
        requested = max(int(step_count), 0)
        if requested <= 0:
            return 0

        first_factor = self.effective_train_downscale_factor(self.state.step)
        batch_steps = 0
        while batch_steps < requested and self.effective_train_downscale_factor(self.state.step + batch_steps) == first_factor:
            if batch_steps > 0 and self.maintenance_due(self.state.step + batch_steps):
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
            frame_index = self._next_frame_index()
            frame_indices.append(frame_index)
            frame_camera = self.make_frame_camera(frame_index, self.renderer.width, self.renderer.height)
            self.renderer.record_prepass_for_current_scene(enc, frame_camera)
            target_texture = self.get_frame_target_texture(frame_index, native_resolution=False, encoder=enc)
            self._dispatch_training_forward(enc, frame_camera, background, target_texture)
            self._dispatch_training_backward(enc, frame_camera, background, target_texture)
            self._dispatch_optimizer_step(enc, self.state.step + batch_index + 1)
            self._dispatch_cache_step_info(enc, batch_index)
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()

        step_metrics = self._read_batch_step_metrics(batch_steps)
        had_nonfinite = False
        for batch_index, frame_index in enumerate(frame_indices):
            loss = float(step_metrics[batch_index, self._LOSS_SLOT_TOTAL])
            image_mse = float(step_metrics[batch_index, self._LOSS_SLOT_MSE])
            depth_ratio_loss = float(step_metrics[batch_index, self._LOSS_SLOT_DEPTH_RATIO])
            had_nonfinite = had_nonfinite or not np.isfinite(loss)
            self.state.step += 1
            self.state.last_base_lr = self.current_base_lr(self.state.step)
            self.state.last_frame_index = frame_index
            self.state.last_loss = loss
            self.state.last_mse = image_mse
            self.state.last_depth_ratio_loss = depth_ratio_loss
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
        self.state.avg_depth_ratio_loss = float(np.mean(step_metrics[:, self._LOSS_SLOT_DEPTH_RATIO], dtype=np.float64)) if batch_steps > 0 else float("nan")
        self.training.train_downscale_factor = self.effective_train_downscale_factor(self.state.step)
        if self.maintenance_due(self.state.step):
            self._background_rng = np.random.default_rng(self._seed + 0x9E3779B9)
            self._run_maintenance()
        return batch_steps

    def step(self) -> float:
        self.step_batch(1)
        return float(self.state.last_loss)

    @property
    def maintenance_buffers(self) -> dict[str, spy.Buffer]:
        return self._maintenance_buffers
