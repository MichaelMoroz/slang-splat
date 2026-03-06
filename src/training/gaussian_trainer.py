from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import slangpy as spy

from ..common import SHADER_ROOT
from ..renderer import Camera, GaussianRenderer
from ..scene import ColmapFrame, GaussianInitHyperParams, GaussianScene


@dataclass(slots=True)
class _SceneCountProxy:
    count: int


@dataclass(slots=True)
class AdamHyperParams:
    position_lr: float = 1e-3
    scale_lr: float = 2.5e-4
    rotation_lr: float = 1e-3
    color_lr: float = 1e-3
    opacity_lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8


@dataclass(slots=True)
class StabilityHyperParams:
    grad_component_clip: float = 10.0
    grad_norm_clip: float = 10.0
    max_update: float = 0.05
    min_scale: float = 1e-3
    max_scale: float = 3.0
    min_opacity: float = 1e-4
    max_opacity: float = 0.9999
    position_abs_max: float = 1e4
    huge_value: float = 1e8
    min_inv_scale: float = 1e-6
    loss_grad_clip: float = 10.0


@dataclass(slots=True)
class TrainingHyperParams:
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    near: float = 0.1
    far: float = 120.0
    ema_decay: float = 0.95
    psnr_reference_decay: float = 0.995
    psnr_decay: float = 0.985
    scale_l2_weight: float = 1e-3
    scale_aniso_weight: float = 1e-3
    mcmc_position_noise_enabled: bool = True
    mcmc_position_noise_scale: float = 1.0
    mcmc_opacity_gate_sharpness: float = 100.0
    mcmc_opacity_gate_center: float = 0.995
    low_quality_reinit_enabled: bool = True


@dataclass(slots=True)
class TrainingState:
    step: int = 0
    last_loss: float = float("nan")
    ema_loss: float = float("nan")
    last_mse: float = float("nan")
    ema_signal_max: float = float("nan")
    last_psnr: float = float("nan")
    ema_psnr: float = float("nan")
    last_frame_index: int = -1
    last_instability: str = ""


class GaussianTrainer:
    _LOSS_SLOT_TOTAL = 0
    _LOSS_SLOT_MSE = 1
    _PSNR_MSE_FLOOR = 1e-12

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
    ) -> None:
        if not frames:
            raise ValueError("Training requires at least one COLMAP frame.")
        if scene is None and scene_count is None:
            raise ValueError("GaussianTrainer requires either scene or scene_count.")
        self.device = device
        self.renderer = renderer
        self._scene_count = int(scene.count if scene is not None else max(int(scene_count or 0), 1))
        self.scene = scene if scene is not None else _SceneCountProxy(self._scene_count)
        self.frames = frames
        self.adam = adam_hparams if adam_hparams is not None else AdamHyperParams()
        self.stability = stability_hparams if stability_hparams is not None else StabilityHyperParams()
        self.training = training_hparams if training_hparams is not None else TrainingHyperParams()
        self.state = TrainingState()
        self._rng = np.random.default_rng(int(seed))
        self._shader_path = Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang")
        self._kernels: dict[str, spy.ComputeKernel] = {}
        self._buffers: dict[str, spy.Buffer] = {}
        self._splat_capacity = 0
        self._init_point_positions_buffer: spy.Buffer | None = None
        self._init_point_colors_buffer: spy.Buffer | None = None
        self._init_point_count = 0
        self._frame_targets_train: list[spy.Texture] = []
        self._frame_targets_native: list[spy.Texture] = []
        if upload_initial_scene and scene is not None:
            self.renderer.set_scene(scene)
        else:
            self.renderer.bind_scene_count(self._scene_count)
        self._create_shaders()
        self._create_training_buffers()
        self._create_dataset_textures()
        self._bind_or_upload_init_pointcloud(
            positions=init_point_positions,
            colors=init_point_colors,
            positions_buffer=init_point_positions_buffer,
            colors_buffer=init_point_colors_buffer,
            point_count=init_point_count,
        )

    def _create_shaders(self) -> None:
        load_program = self.device.load_program
        self._kernels = {
            "clear_loss_grad": self.device.create_compute_kernel(
                load_program(str(self._shader_path), ["csClearLossAndGradTex"])
            ),
            "mse_loss_grad": self.device.create_compute_kernel(
                load_program(str(self._shader_path), ["csComputeMSELossGrad"])
            ),
            "adam_step": self.device.create_compute_kernel(load_program(str(self._shader_path), ["csAdamStepFused"])),
            "mark_low_quality_splats": self.device.create_compute_kernel(
                load_program(str(self._shader_path), ["csMarkLowQualitySplats"])
            ),
            "resample_low_quality_splats": self.device.create_compute_kernel(
                load_program(str(self._shader_path), ["csResampleLowQualitySplatsRandom"])
            ),
            "init_from_pointcloud": self.device.create_compute_kernel(
                load_program(str(self._shader_path), ["csInitializeGaussiansFromPointCloud"])
            ),
        }

    def _create_training_buffers(self) -> None:
        self._ensure_training_buffers(self._scene_count)
        self._zero_optimizer_moments()

    def _ensure_training_buffers(self, splat_count: int) -> None:
        count = max(int(splat_count), 1)
        if self._buffers and count <= self._splat_capacity:
            return
        usage_rw = (
            spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access
            | spy.BufferUsage.copy_source
            | spy.BufferUsage.copy_destination
        )
        old_capacity = max(self._splat_capacity, 1)
        new_capacity = max(count, old_capacity + old_capacity // 2)
        if "loss" not in self._buffers:
            self._buffers["loss"] = self.device.create_buffer(size=8, usage=usage_rw)
        if "signal_max" not in self._buffers:
            self._buffers["signal_max"] = self.device.create_buffer(size=4, usage=usage_rw)
        self._buffers["low_quality_flags"] = self.device.create_buffer(size=new_capacity * 4, usage=usage_rw)
        for name in (
            "adam_m_pos",
            "adam_v_pos",
            "adam_m_scale",
            "adam_v_scale",
            "adam_m_quat",
            "adam_v_quat",
            "adam_m_color_alpha",
            "adam_v_color_alpha",
        ):
            self._buffers[name] = self.device.create_buffer(size=new_capacity * 16, usage=usage_rw)
        self._splat_capacity = new_capacity

    def _zero_optimizer_moments(self) -> None:
        count = max(int(self._scene_count), 1)
        zeros = np.zeros((count, 4), dtype=np.float32)
        for name in (
            "adam_m_pos",
            "adam_v_pos",
            "adam_m_scale",
            "adam_v_scale",
            "adam_m_quat",
            "adam_v_quat",
            "adam_m_color_alpha",
            "adam_v_color_alpha",
        ):
            self._buffers[name].copy_from_numpy(zeros)

    def _pack_point_table(self, points: np.ndarray) -> np.ndarray:
        pts = np.ascontiguousarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("Point table must be [N, >=3] float array.")
        packed = np.zeros((pts.shape[0], 4), dtype=np.float32)
        packed[:, :3] = pts[:, :3]
        return packed

    def _create_init_point_buffer(self, points: np.ndarray) -> spy.Buffer:
        packed = self._pack_point_table(points)
        usage = (
            spy.BufferUsage.shader_resource
            | spy.BufferUsage.copy_source
            | spy.BufferUsage.copy_destination
        )
        buffer = self.device.create_buffer(size=max(packed.shape[0], 1) * 16, usage=usage)
        if packed.shape[0] > 0:
            buffer.copy_from_numpy(packed)
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
            self._init_point_positions_buffer = positions_buffer
            self._init_point_colors_buffer = colors_buffer
            self._init_point_count = int(point_count)
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

    def has_pointcloud_initializer(self) -> bool:
        return (
            self._init_point_positions_buffer is not None
            and self._init_point_colors_buffer is not None
            and self._init_point_count > 0
        )

    def _make_frame_camera(self, frame: ColmapFrame, width: int, height: int) -> Camera:
        camera = frame.make_camera(near=float(self.training.near), far=float(self.training.far))
        frame_width = max(int(frame.width), 1)
        frame_height = max(int(frame.height), 1)
        same_size = int(width) == frame_width and int(height) == frame_height
        if same_size:
            return camera
        sx = float(width) / float(frame_width)
        sy = float(height) / float(frame_height)
        if camera.fx is not None:
            camera.fx = float(camera.fx) * sx
        if camera.fy is not None:
            camera.fy = float(camera.fy) * sy
        if camera.cx is not None:
            camera.cx = float(camera.cx) * sx
        if camera.cy is not None:
            camera.cy = float(camera.cy) * sy
        return camera

    def _to_rgba8(self, image: Image.Image) -> np.ndarray:
        return np.array(image.convert("RGBA"), dtype=np.uint8, order="C", copy=True)

    def _create_gpu_texture(self, rgba8: np.ndarray) -> spy.Texture:
        height, width = rgba8.shape[:2]
        tex = self.device.create_texture(
            format=spy.Format.rgba8_unorm,
            width=int(width),
            height=int(height),
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
        )
        tex.copy_from_numpy(np.ascontiguousarray(rgba8, dtype=np.uint8))
        return tex

    def _create_dataset_textures(self) -> None:
        self._frame_targets_train = []
        self._frame_targets_native = []
        train_width = int(self.renderer.width)
        train_height = int(self.renderer.height)
        for frame in self.frames:
            with Image.open(frame.image_path) as pil_image:
                image = pil_image.convert("RGB")
                self._frame_targets_native.append(self._create_gpu_texture(self._to_rgba8(image)))
                image_train = image
                if image.size != (train_width, train_height):
                    image_train = image.resize((train_width, train_height), resample=Image.Resampling.BILINEAR)
                self._frame_targets_train.append(self._create_gpu_texture(self._to_rgba8(image_train)))

    def update_hyperparams(
        self,
        adam_hparams: AdamHyperParams,
        stability_hparams: StabilityHyperParams,
        training_hparams: TrainingHyperParams,
    ) -> None:
        self.adam = adam_hparams
        self.stability = stability_hparams
        self.training = training_hparams

    def frame_count(self) -> int:
        return len(self.frames)

    def frame_size(self, frame_index: int) -> tuple[int, int]:
        idx = int(np.clip(frame_index, 0, len(self.frames) - 1))
        frame = self.frames[idx]
        return int(frame.width), int(frame.height)

    def get_frame_target_texture(self, frame_index: int, native_resolution: bool = True) -> spy.Texture:
        idx = int(np.clip(frame_index, 0, len(self.frames) - 1))
        if native_resolution:
            return self._frame_targets_native[idx]
        return self._frame_targets_train[idx]

    def make_frame_camera(self, frame_index: int, width: int, height: int) -> Camera:
        idx = int(np.clip(frame_index, 0, len(self.frames) - 1))
        frame = self.frames[idx]
        return self._make_frame_camera(frame, int(width), int(height))

    def _common_vars(self) -> dict[str, object]:
        return {
            "g_Width": int(self.renderer.width),
            "g_Height": int(self.renderer.height),
            "g_SplatCount": int(self._scene_count),
            "g_LowQualityReinitEnabled": np.uint32(1 if self.training.low_quality_reinit_enabled else 0),
            "g_InvPixelCount": 1.0 / float(max(self.renderer.width * self.renderer.height, 1)),
            "g_LossGradClip": float(self.stability.loss_grad_clip),
            "g_ScaleL2Weight": float(max(self.training.scale_l2_weight, 0.0)),
            "g_ScaleAnisoWeight": float(max(self.training.scale_aniso_weight, 0.0)),
            "g_Adam": {
                "positionLR": float(self.adam.position_lr),
                "scaleLR": float(self.adam.scale_lr),
                "rotationLR": float(self.adam.rotation_lr),
                "colorLR": float(self.adam.color_lr),
                "opacityLR": float(self.adam.opacity_lr),
                "beta1": float(self.adam.beta1),
                "beta2": float(self.adam.beta2),
                "epsilon": float(self.adam.epsilon),
                "stepIndex": int(self.state.step + 1),
            },
            "g_Stability": {
                "gradComponentClip": float(self.stability.grad_component_clip),
                "gradNormClip": float(self.stability.grad_norm_clip),
                "maxUpdate": float(self.stability.max_update),
                "minScale": float(self.stability.min_scale),
                "maxScale": float(self.stability.max_scale),
                "minOpacity": float(self.stability.min_opacity),
                "maxOpacity": float(self.stability.max_opacity),
                "positionAbsMax": float(self.stability.position_abs_max),
                "hugeValue": float(self.stability.huge_value),
            },
            "g_MCMC": {
                "enabled": np.uint32(1 if self.training.mcmc_position_noise_enabled else 0),
                "positionNoiseScale": float(max(self.training.mcmc_position_noise_scale, 0.0)),
                "opacityGateSharpness": float(max(self.training.mcmc_opacity_gate_sharpness, 0.0)),
                "opacityGateCenter": float(np.clip(self.training.mcmc_opacity_gate_center, 0.0, 1.0)),
            },
        }

    def _dispatch_loss_grad(
        self,
        encoder: spy.CommandEncoder,
        target_texture: spy.Texture,
    ) -> None:
        vars_common = self._common_vars()
        self._kernels["clear_loss_grad"].dispatch(
            thread_count=spy.uint3(self.renderer.width, self.renderer.height, 1),
            vars={
                "g_OutputGrad": self.renderer.output_grad_texture,
                "g_LossBuffer": self._buffers["loss"],
                "g_SignalMaxBuffer": self._buffers["signal_max"],
                **vars_common,
            },
            command_encoder=encoder,
        )
        self._kernels["mse_loss_grad"].dispatch(
            thread_count=spy.uint3(self.renderer.width, self.renderer.height, 1),
            vars={
                "g_Rendered": self.renderer.output_texture,
                "g_Target": target_texture,
                "g_OutputGrad": self.renderer.output_grad_texture,
                "g_LossBuffer": self._buffers["loss"],
                "g_SignalMaxBuffer": self._buffers["signal_max"],
                **vars_common,
            },
            command_encoder=encoder,
        )

    def _dispatch_raster_forward_backward(
        self,
        encoder: spy.CommandEncoder,
        frame_camera,
        background: np.ndarray,
    ) -> None:
        self.renderer.clear_raster_grads_current_scene(encoder)
        self.renderer.rasterize_forward_backward_current_scene(
            encoder=encoder,
            camera=frame_camera,
            background=background,
            output_grad=self.renderer.output_grad_texture,
        )

    def _dispatch_adam_step(self, encoder: spy.CommandEncoder) -> None:
        vars_common = self._common_vars()
        self._kernels["adam_step"].dispatch(
            thread_count=spy.uint3(self._scene_count, 1, 1),
            vars={
                "g_GradPositions": self.renderer.work_buffers["grad_positions"],
                "g_GradScales": self.renderer.work_buffers["grad_scales"],
                "g_GradRotations": self.renderer.work_buffers["grad_rotations"],
                "g_GradColorAlpha": self.renderer.work_buffers["grad_color_alpha"],
                "g_PositionsRW": self.renderer.scene_buffers["positions"],
                "g_ScalesRW": self.renderer.scene_buffers["scales"],
                "g_RotationsRW": self.renderer.scene_buffers["rotations"],
                "g_ColorAlphaRW": self.renderer.scene_buffers["color_alpha"],
                "g_AdamMPos": self._buffers["adam_m_pos"],
                "g_AdamVPos": self._buffers["adam_v_pos"],
                "g_AdamMScale": self._buffers["adam_m_scale"],
                "g_AdamVScale": self._buffers["adam_v_scale"],
                "g_AdamMQuat": self._buffers["adam_m_quat"],
                "g_AdamVQuat": self._buffers["adam_v_quat"],
                "g_AdamMColorAlpha": self._buffers["adam_m_color_alpha"],
                "g_AdamVColorAlpha": self._buffers["adam_v_color_alpha"],
                **vars_common,
            },
            command_encoder=encoder,
        )

    def _dispatch_mark_low_quality_splats(self, encoder: spy.CommandEncoder) -> None:
        vars_common = self._common_vars()
        self._kernels["mark_low_quality_splats"].dispatch(
            thread_count=spy.uint3(self._scene_count, 1, 1),
            vars={
                "g_ScalesRW": self.renderer.scene_buffers["scales"],
                "g_ColorAlphaRW": self.renderer.scene_buffers["color_alpha"],
                "g_LowQualityFlags": self._buffers["low_quality_flags"],
                **vars_common,
            },
            command_encoder=encoder,
        )

    def _dispatch_resample_low_quality_splats(self, encoder: spy.CommandEncoder) -> None:
        vars_common = self._common_vars()
        self._kernels["resample_low_quality_splats"].dispatch(
            thread_count=spy.uint3(self._scene_count, 1, 1),
            vars={
                "g_PositionsRW": self.renderer.scene_buffers["positions"],
                "g_ScalesRW": self.renderer.scene_buffers["scales"],
                "g_RotationsRW": self.renderer.scene_buffers["rotations"],
                "g_ColorAlphaRW": self.renderer.scene_buffers["color_alpha"],
                "g_AdamMPos": self._buffers["adam_m_pos"],
                "g_AdamVPos": self._buffers["adam_v_pos"],
                "g_AdamMScale": self._buffers["adam_m_scale"],
                "g_AdamVScale": self._buffers["adam_v_scale"],
                "g_AdamMQuat": self._buffers["adam_m_quat"],
                "g_AdamVQuat": self._buffers["adam_v_quat"],
                "g_AdamMColorAlpha": self._buffers["adam_m_color_alpha"],
                "g_AdamVColorAlpha": self._buffers["adam_v_color_alpha"],
                "g_LowQualityFlags": self._buffers["low_quality_flags"],
                **vars_common,
            },
            command_encoder=encoder,
        )

    def initialize_scene_from_pointcloud(
        self,
        splat_count: int,
        init_hparams: GaussianInitHyperParams,
        seed: int,
    ) -> None:
        if not self.has_pointcloud_initializer():
            raise RuntimeError("Pointcloud initializer buffers are not available.")
        self._scene_count = max(int(splat_count), 1)
        self.scene = _SceneCountProxy(self._scene_count)
        self.renderer.bind_scene_count(self._scene_count)
        self._ensure_training_buffers(self._scene_count)
        vars_common = self._common_vars()
        enc = self.device.create_command_encoder()
        self._kernels["init_from_pointcloud"].dispatch(
            thread_count=spy.uint3(self._scene_count, 1, 1),
            vars={
                "g_PositionsRW": self.renderer.scene_buffers["positions"],
                "g_ScalesRW": self.renderer.scene_buffers["scales"],
                "g_RotationsRW": self.renderer.scene_buffers["rotations"],
                "g_ColorAlphaRW": self.renderer.scene_buffers["color_alpha"],
                "g_AdamMPos": self._buffers["adam_m_pos"],
                "g_AdamVPos": self._buffers["adam_v_pos"],
                "g_AdamMScale": self._buffers["adam_m_scale"],
                "g_AdamVScale": self._buffers["adam_v_scale"],
                "g_AdamMQuat": self._buffers["adam_m_quat"],
                "g_AdamVQuat": self._buffers["adam_v_quat"],
                "g_AdamMColorAlpha": self._buffers["adam_m_color_alpha"],
                "g_AdamVColorAlpha": self._buffers["adam_v_color_alpha"],
                "g_InitPointPositions": self._init_point_positions_buffer,
                "g_InitPointColors": self._init_point_colors_buffer,
                "g_InitPointCount": int(self._init_point_count),
                "g_InitSeed": int(seed),
                "g_InitPositionJitterStd": float(max(init_hparams.position_jitter_std, 0.0)),
                "g_InitBaseScale": float(max(init_hparams.base_scale, 1e-4)),
                "g_InitScaleJitterRatio": float(max(init_hparams.scale_jitter_ratio, 0.0)),
                "g_InitInitialOpacity": float(np.clip(init_hparams.initial_opacity, 0.0, 1.0)),
                "g_InitColorJitterStd": float(max(init_hparams.color_jitter_std, 0.0)),
                **vars_common,
            },
            command_encoder=enc,
        )
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()
        self.state.step = 0
        self.state.last_loss = float("nan")
        self.state.ema_loss = float("nan")
        self.state.last_mse = float("nan")
        self.state.ema_signal_max = float("nan")
        self.state.last_psnr = float("nan")
        self.state.ema_psnr = float("nan")
        self.state.last_frame_index = -1
        self.state.last_instability = ""

    def _read_loss_metrics(self) -> tuple[float, float, float]:
        loss_raw = self._buffers["loss"].to_numpy()
        loss_values = np.frombuffer(loss_raw.tobytes(), dtype=np.float32)
        signal_raw = self._buffers["signal_max"].to_numpy()
        signal_values = np.frombuffer(signal_raw.tobytes(), dtype=np.uint32)
        total_loss = float(loss_values[self._LOSS_SLOT_TOTAL]) if loss_values.size > self._LOSS_SLOT_TOTAL else float("nan")
        mse = float(loss_values[self._LOSS_SLOT_MSE]) if loss_values.size > self._LOSS_SLOT_MSE else float("nan")
        signal_max = float(signal_values.view(np.float32)[0]) if signal_values.size else float("nan")
        return total_loss, mse, signal_max

    def _update_psnr_state(self, mse: float, signal_max: float) -> None:
        self.state.last_mse = float(mse)
        ref_decay = float(np.clip(self.training.psnr_reference_decay, 0.0, 0.999999))
        psnr_decay = float(np.clip(self.training.psnr_decay, 0.0, 0.999999))
        if np.isfinite(signal_max) and signal_max > 0.0:
            if not np.isfinite(self.state.ema_signal_max):
                self.state.ema_signal_max = float(signal_max)
            else:
                self.state.ema_signal_max = ref_decay * self.state.ema_signal_max + (1.0 - ref_decay) * float(signal_max)
        psnr = float("nan")
        if np.isfinite(mse) and mse >= 0.0 and np.isfinite(self.state.ema_signal_max) and self.state.ema_signal_max > 0.0:
            mse_safe = max(float(mse), self._PSNR_MSE_FLOOR)
            psnr = 20.0 * float(np.log10(self.state.ema_signal_max)) - 10.0 * float(np.log10(mse_safe))
        self.state.last_psnr = psnr
        if np.isfinite(psnr):
            if not np.isfinite(self.state.ema_psnr):
                self.state.ema_psnr = psnr
            else:
                self.state.ema_psnr = psnr_decay * self.state.ema_psnr + (1.0 - psnr_decay) * psnr

    def step(self) -> float:
        frame_index = int(self._rng.integers(0, len(self.frames)))
        frame_camera = self.make_frame_camera(frame_index, self.renderer.width, self.renderer.height)
        background = np.asarray(self.training.background, dtype=np.float32).reshape(3)
        target_texture = self.get_frame_target_texture(frame_index, native_resolution=False)
        self.renderer.execute_prepass_for_current_scene(frame_camera, sync_counts=False)

        enc = self.device.create_command_encoder()
        self.renderer.rasterize_current_scene(enc, frame_camera, background)
        self._dispatch_loss_grad(enc, target_texture)
        self._dispatch_raster_forward_backward(enc, frame_camera, background)
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()

        image_loss, image_mse, signal_max = self._read_loss_metrics()
        loss = image_loss
        if not np.isfinite(image_loss):
            self.state.last_instability = "Non-finite loss; ADAM step skipped and moments reset."
            self._zero_optimizer_moments()
            loss = float("inf")
        else:
            self.state.last_instability = ""
            enc_opt = self.device.create_command_encoder()
            self._dispatch_adam_step(enc_opt)
            self._dispatch_mark_low_quality_splats(enc_opt)
            self._dispatch_resample_low_quality_splats(enc_opt)
            self.device.submit_command_buffer(enc_opt.finish())
            self.device.wait()
            loss, _, _ = self._read_loss_metrics()
            if not np.isfinite(loss):
                self.state.last_instability = "Non-finite regularized loss after ADAM; using image loss."
                loss = image_loss

        self.state.step += 1
        self.state.last_frame_index = frame_index
        self.state.last_loss = float(loss)
        self._update_psnr_state(image_mse, signal_max)
        if not np.isfinite(self.state.ema_loss):
            self.state.ema_loss = float(loss)
        else:
            decay = float(np.clip(self.training.ema_decay, 0.0, 0.99999))
            self.state.ema_loss = decay * self.state.ema_loss + (1.0 - decay) * float(loss)
        return float(loss)
