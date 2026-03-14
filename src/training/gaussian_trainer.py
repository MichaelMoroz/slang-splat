from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import slangpy as spy

from ..common import SHADER_ROOT, buffer_to_numpy, clamp_index, thread_count_2d
from ..metrics import Metrics, psnr_from_mse
from ..renderer import Camera, GaussianRenderer
from ..scene import ColmapFrame, GaussianInitHyperParams, GaussianScene
from .adam import AdamOptimizer, AdamRuntimeHyperParams
from .optimizer import GaussianOptimizer


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
    scale_l2_weight: float = 0.0; scale_abs_reg_weight: float = 0.01; opacity_reg_weight: float = 0.01
    max_gaussians: int = 5_900_000


@dataclass(slots=True)
class TrainingState:
    step: int = 0; last_loss: float = float("nan"); avg_loss: float = float("nan"); last_mse: float = float("nan"); avg_mse: float = float("nan"); last_psnr: float = float("nan"); avg_psnr: float = float("nan")
    last_frame_index: int = -1; last_instability: str = ""


class GaussianTrainer:
    _LOSS_SLOT_TOTAL = 0
    _LOSS_SLOT_MSE = 1
    _KERNEL_ENTRIES = {
        "clear_loss_grad": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csClearLossAndGradTex"),
        "loss_grad": (Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang"), "csComputeL1LossGrad"),
    }

    def _pixel_thread_count(self) -> spy.uint3:
        return thread_count_2d(self.renderer.width, self.renderer.height)

    def _frame(self, frame_index: int) -> ColmapFrame:
        return self.frames[clamp_index(frame_index, len(self.frames))]

    def make_frame_camera(self, frame_index: int, width: int, height: int) -> Camera:
        return self._make_frame_camera(self._frame(frame_index), int(width), int(height))

    def frame_size(self, frame_index: int) -> tuple[int, int]:
        frame = self._frame(frame_index)
        return int(frame.width), int(frame.height)

    def get_frame_target_texture(self, frame_index: int, native_resolution: bool = True) -> spy.Texture:
        return self._frame_targets_train[clamp_index(frame_index, len(self.frames))]

    def _adam_runtime_hparams(self) -> AdamRuntimeHyperParams:
        return AdamRuntimeHyperParams(
            grad_component_clip=float(self.stability.grad_component_clip),
            grad_norm_clip=float(self.stability.grad_norm_clip),
            max_update=float(self.stability.max_update),
            huge_value=float(self.stability.huge_value),
        )

    def _dispatch(self, kernel: str, encoder: spy.CommandEncoder, thread_count: spy.uint3, vars: dict[str, object]) -> None:
        self._kernels[kernel].dispatch(thread_count=thread_count, vars=vars, command_encoder=encoder)

    def _dispatch_raster_forward_backward(self, encoder: spy.CommandEncoder, frame_camera: Camera, background: np.ndarray) -> None:
        self.renderer.clear_raster_grads_current_scene(encoder)
        self.renderer.rasterize_forward_backward_current_scene(
            encoder=encoder,
            camera=frame_camera,
            background=background,
            output_grad=self.renderer.output_grad_buffer,
        )

    def _read_loss_metrics(self) -> tuple[float, float]:
        values = buffer_to_numpy(self._buffers["loss"], np.float32)
        total = float(values[self._LOSS_SLOT_TOTAL]) if values.size > self._LOSS_SLOT_TOTAL else float("nan")
        mse = float(values[self._LOSS_SLOT_MSE]) if values.size > self._LOSS_SLOT_MSE else float("nan")
        return total, mse

    def update_hyperparams(self, adam_hparams: AdamHyperParams, stability_hparams: StabilityHyperParams, training_hparams: TrainingHyperParams) -> None:
        self.adam = adam_hparams
        self.stability = stability_hparams
        self.training = training_hparams
        self.adam_optimizer.update_hyperparams(self.adam, self._adam_runtime_hparams())
        self.optimizer.update_hyperparams(self.adam, self.stability)

    def _dispatch_adam_step(self, encoder: spy.CommandEncoder) -> None:
        self._dispatch_optimizer_step(encoder)

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
        self._frame_metrics = _FrameMetricBookkeeper.create(len(self.frames))
        self._kernels = {
            name: self.device.create_compute_kernel(self.device.load_program(str(shader_path), [entry]))
            for name, (shader_path, entry) in self._KERNEL_ENTRIES.items()
        }
        self._buffers: dict[str, spy.Buffer] = {}
        self._splat_capacity = 0
        self._scale_reg_reference = float(max(scale_reg_reference, 1e-8)) if scale_reg_reference is not None else self._estimate_scale_reg_reference(scene)
        self._init_point_positions_buffer: spy.Buffer | None = None
        self._init_point_colors_buffer: spy.Buffer | None = None
        self._init_point_count = 0
        self._init_point_positions_cpu: np.ndarray | None = None
        self._init_point_colors_cpu: np.ndarray | None = None
        self._frame_targets_train: list[spy.Texture] = []
        self._frame_rng = np.random.default_rng(self._seed)
        self._frame_order = np.zeros((len(self.frames),), dtype=np.int32)
        self._frame_cursor = len(self.frames)
        if upload_initial_scene and scene is not None:
            self.renderer.set_scene(scene)
        else:
            self.renderer.bind_scene_count(self._scene_count)
        self._ensure_training_buffers(self._scene_count)
        self._zero_optimizer_moments()
        self._create_dataset_textures()
        self._bind_or_upload_init_pointcloud(init_point_positions, init_point_colors, init_point_positions_buffer, init_point_colors_buffer, init_point_count)
        self._reset_frame_order()

    def _estimate_scale_reg_reference(self, scene: GaussianScene | None) -> float:
        if scene is None or scene.count <= 0:
            return 1.0
        scales = np.asarray(scene.scales, dtype=np.float32)
        finite = np.isfinite(scales).all(axis=1)
        return 1.0 if not np.any(finite) else float(np.exp(np.mean(np.log(np.maximum(scales[finite], 1e-8)), dtype=np.float64)))

    def _ensure_training_buffers(self, splat_count: int) -> None:
        count = max(int(splat_count), 1)
        if self._buffers and count <= self._splat_capacity: return
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
        self._splat_capacity = max(count, max(self._splat_capacity, 1) + max(self._splat_capacity, 1) // 2)
        self._buffers.setdefault("loss", self.device.create_buffer(size=8, usage=usage))

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
        train_size = (int(self.renderer.width), int(self.renderer.height))
        self._frame_targets_train = []
        for frame in self.frames:
            with Image.open(frame.image_path) as pil_image:
                native = pil_image.convert("RGB")
                resized = native if native.size == train_size else native.resize(train_size, resample=Image.Resampling.BILINEAR)
                self._frame_targets_train.append(self._create_gpu_texture(resized))

    def _loss_vars(self) -> dict[str, object]:
        return {
            "g_Width": int(self.renderer.width),
            "g_Height": int(self.renderer.height),
            "g_InvPixelCount": 1.0 / float(max(self.renderer.width * self.renderer.height, 1)),
            "g_LossGradClip": float(self.stability.loss_grad_clip),
            "g_HugeValue": float(self.stability.huge_value),
        }

    def _dispatch_loss_grad(self, encoder: spy.CommandEncoder, target_texture: spy.Texture) -> None:
        shared = {"g_OutputGrad": self.renderer.output_grad_buffer, "g_LossBuffer": self._buffers["loss"], **self._loss_vars()}
        self._dispatch("clear_loss_grad", encoder, self._pixel_thread_count(), shared)
        self._dispatch(
            "loss_grad",
            encoder,
            self._pixel_thread_count(),
            {
                "g_Rendered": self.renderer.output_texture,
                "g_Target": target_texture,
                "g_OutputGrad": self.renderer.output_grad_buffer,
                "g_LossBuffer": self._buffers["loss"],
                **self._loss_vars(),
            },
        )

    def _dispatch_optimizer_step(self, encoder: spy.CommandEncoder) -> None:
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
            step_index=self.state.step + 1,
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
        scales = np.repeat(point_nn_scales(positions)[:, None], 3, axis=1).astype(np.float32)
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
        self._ensure_training_buffers(self._scene_count)
        self._scale_reg_reference = float(max(np.median(scales[:, 0]), 1e-8))
        self._zero_optimizer_moments()
        self.state = TrainingState()
        self._frame_metrics.reset()
        self._frame_rng = np.random.default_rng(int(seed))
        self._reset_frame_order()

    def scale_histogram(self, *, bin_count: int = 64, min_log10: float = -6.0, max_log10: float = 1.0):
        return self.metrics.compute_scale_histogram(
            self.renderer.scene_buffers["splat_params"],
            self._scene_count,
            bin_count=bin_count,
            min_log10=min_log10,
            max_log10=max_log10,
        )

    def anisotropy_histogram(self, *, bin_count: int = 64, min_log10: float = 0.0, max_log10: float = 2.0):
        return self.metrics.compute_anisotropy_histogram(
            self.renderer.scene_buffers["splat_params"],
            self._scene_count,
            bin_count=bin_count,
            min_log10=min_log10,
            max_log10=max_log10,
        )

    def step(self) -> float:
        frame_index = self._next_frame_index()
        frame_camera = self.make_frame_camera(frame_index, self.renderer.width, self.renderer.height)
        background = np.asarray(self.training.background, dtype=np.float32).reshape(3)
        self.renderer.execute_prepass_for_current_scene(frame_camera, sync_counts=False)
        enc = self.device.create_command_encoder()
        self.renderer.rasterize_current_scene(enc, frame_camera, background)
        self._dispatch_loss_grad(enc, self.get_frame_target_texture(frame_index, native_resolution=False))
        self._dispatch_raster_forward_backward(enc, frame_camera, background)
        self._dispatch_optimizer_step(enc)
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()

        loss, image_mse = self._read_loss_metrics()
        if not np.isfinite(loss):
            self.state.last_instability = "Non-finite loss after ADAM; moments reset."
            self._zero_optimizer_moments()
        else:
            self.state.last_instability = ""

        self.state.step += 1
        self.state.last_frame_index = frame_index
        self.state.last_loss = float(loss)
        self.state.last_mse = float(image_mse)
        self.state.last_psnr = float(psnr_from_mse(image_mse))
        self._frame_metrics.update(frame_index, self.state.last_loss, self.state.last_mse, self.state.last_psnr)
        self.state.avg_loss = self._frame_metrics.mean("loss")
        self.state.avg_mse = self._frame_metrics.mean("mse")
        self.state.avg_psnr = self._frame_metrics.mean("psnr")
        return float(loss)
