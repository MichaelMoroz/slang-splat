from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import slangpy as spy

from ..common import SHADER_ROOT
from ..filter import SeparableGaussianBlur
from ..renderer import Camera, GaussianRenderer
from ..scene import ColmapFrame, GaussianInitHyperParams, GaussianScene


@dataclass(slots=True)
class _SceneCountProxy:
    count: int


@dataclass(slots=True)
class _RollingMetricWindow:
    size: int
    values: deque[float]

    def push(self, value: float) -> None:
        if np.isfinite(value):
            if len(self.values) == self.size:
                self.values.popleft()
            self.values.append(float(value))

    def mean(self) -> float:
        return float(np.fromiter(self.values, dtype=np.float64).mean()) if self.values else float("nan")


@dataclass(slots=True)
class AdamHyperParams:
    position_lr: float = 1e-3; scale_lr: float = 2.5e-4; rotation_lr: float = 1e-3; color_lr: float = 1e-3
    opacity_lr: float = 1e-3; beta1: float = 0.9; beta2: float = 0.999; epsilon: float = 1e-8


@dataclass(slots=True)
class StabilityHyperParams:
    grad_component_clip: float = 10.0; grad_norm_clip: float = 10.0; max_update: float = 0.05; min_scale: float = 1e-3
    max_scale: float = 3.0; max_anisotropy: float = 3.0; min_opacity: float = 1e-4; max_opacity: float = 0.9999
    position_abs_max: float = 1e4; huge_value: float = 1e8; min_inv_scale: float = 1e-6; loss_grad_clip: float = 10.0


@dataclass(slots=True)
class TrainingHyperParams:
    background: tuple[float, float, float] = (0.0, 0.0, 0.0); near: float = 0.1; far: float = 120.0; scale_l2_weight: float = 1e-3; mcmc_position_noise_enabled: bool = True
    mcmc_position_noise_scale: float = 1.0; mcmc_opacity_gate_sharpness: float = 100.0; mcmc_opacity_gate_center: float = 0.995; low_quality_reinit_enabled: bool = True; lambda_dssim: float = 0.2


@dataclass(slots=True)
class TrainingState:
    step: int = 0; last_loss: float = float("nan"); avg_loss: float = float("nan"); last_mse: float = float("nan")
    avg_signal_max: float = float("nan"); last_psnr: float = float("nan"); avg_psnr: float = float("nan"); last_frame_index: int = -1
    last_instability: str = ""


class GaussianTrainer:
    _LOSS_SLOT_TOTAL = 0
    _LOSS_SLOT_MSE = 1
    _PSNR_MSE_FLOOR = 1e-12
    _ADAM_BUFFER_NAMES = ("adam_m_pos", "adam_v_pos", "adam_m_scale", "adam_v_scale", "adam_m_quat", "adam_v_quat", "adam_m_color_alpha", "adam_v_color_alpha")
    _LOSS_TEXTURE_NAMES = (
        "render_sq",
        "target_sq",
        "cross",
        "mu_render",
        "mu_target",
        "blur_render_sq",
        "blur_target_sq",
        "blur_cross",
        "grad_mu",
        "grad_xx",
        "grad_xy",
        "grad_mu_blur",
        "grad_xx_blur",
        "grad_xy_blur",
    )
    _ADAM_SHADER_VARS = {
        "adam_m_pos": "g_AdamMPos",
        "adam_v_pos": "g_AdamVPos",
        "adam_m_scale": "g_AdamMScale",
        "adam_v_scale": "g_AdamVScale",
        "adam_m_quat": "g_AdamMQuat",
        "adam_v_quat": "g_AdamVQuat",
        "adam_m_color_alpha": "g_AdamMColorAlpha",
        "adam_v_color_alpha": "g_AdamVColorAlpha",
    }
    _SCENE_RW_SHADER_VARS = {"positions": "g_PositionsRW", "scales": "g_ScalesRW", "rotations": "g_RotationsRW", "color_alpha": "g_ColorAlphaRW"}
    _SCENE_GRAD_SHADER_VARS = {"grad_positions": "g_GradPositions", "grad_scales": "g_GradScales", "grad_rotations": "g_GradRotations", "grad_color_alpha": "g_GradColorAlpha"}
    _KERNEL_ENTRIES = {
        "clear_loss_grad": "csClearLossAndGradTex",
        "pack_ssim_aux": "csPackSSIMAux",
        "loss_grad": "csComputeMixedLossGrad",
        "compose_loss_grad": "csComposeMixedLossOutputGrad",
        "adam_step": "csAdamStepFused",
        "mark_low_quality_splats": "csMarkLowQualitySplats",
        "resample_low_quality_splats": "csResampleLowQualitySplatsRandom",
        "init_from_pointcloud": "csInitializeGaussiansFromPointCloud",
    }
    _buffer_vars = staticmethod(lambda mapping, source: {shader_name: source[name] for name, shader_name in mapping.items()})
    _dispatch = lambda self, kernel, encoder, thread_count, vars: self._kernels[kernel].dispatch(thread_count=thread_count, vars=vars, command_encoder=encoder)
    _scene_thread_count = lambda self: spy.uint3(self._scene_count, 1, 1)
    _pixel_thread_count = lambda self: spy.uint3(self.renderer.width, self.renderer.height, 1)
    _scene_rw_vars = lambda self: self._buffer_vars(self._SCENE_RW_SHADER_VARS, self.renderer.scene_buffers)
    _scene_grad_vars = lambda self: self._buffer_vars(self._SCENE_GRAD_SHADER_VARS, self.renderer.work_buffers)
    _frame = lambda self, frame_index: self.frames[int(np.clip(frame_index, 0, len(self.frames) - 1))]
    _adam_shader_vars = lambda self: self._buffer_vars(self._ADAM_SHADER_VARS, self._buffers)
    update_hyperparams = lambda self, adam_hparams, stability_hparams, training_hparams: setattr(self, "adam", adam_hparams) or setattr(self, "stability", stability_hparams) or setattr(self, "training", training_hparams)
    make_frame_camera = lambda self, frame_index, width, height: self._make_frame_camera(self._frame(frame_index), int(width), int(height))
    _zero_optimizer_moments = lambda self: [self._buffers[name].copy_from_numpy(np.zeros((max(int(self._scene_count), 1), 4), dtype=np.float32)) for name in self._ADAM_BUFFER_NAMES]
    frame_size = lambda self, frame_index: (int(self._frame(frame_index).width), int(self._frame(frame_index).height))
    get_frame_target_texture = lambda self, frame_index, native_resolution=True: (self._frame_targets_native if native_resolution else self._frame_targets_train)[int(np.clip(frame_index, 0, len(self.frames) - 1))]
    _dispatch_raster_forward_backward = lambda self, encoder, frame_camera, background: (self.renderer.clear_raster_grads_current_scene(encoder), self.renderer.rasterize_forward_backward_current_scene(encoder=encoder, camera=frame_camera, background=background, output_grad=self.renderer.output_grad_texture))
    _dispatch_adam_step = lambda self, encoder: self._dispatch("adam_step", encoder, self._scene_thread_count(), {**self._scene_grad_vars(), **self._scene_rw_vars(), **self._adam_shader_vars(), **self._common_vars()})
    _dispatch_mark_low_quality_splats = lambda self, encoder: self._dispatch("mark_low_quality_splats", encoder, self._scene_thread_count(), {"g_ScalesRW": self.renderer.scene_buffers["scales"], "g_ColorAlphaRW": self.renderer.scene_buffers["color_alpha"], "g_LowQualityFlags": self._buffers["low_quality_flags"], **self._common_vars()})
    _dispatch_resample_low_quality_splats = lambda self, encoder: self._dispatch("resample_low_quality_splats", encoder, self._scene_thread_count(), {**self._scene_rw_vars(), **self._adam_shader_vars(), "g_LowQualityFlags": self._buffers["low_quality_flags"], **self._common_vars()})
    _read_loss_metrics = lambda self: (lambda loss_values, signal_values: (float(loss_values[self._LOSS_SLOT_TOTAL]) if loss_values.size > self._LOSS_SLOT_TOTAL else float("nan"), float(loss_values[self._LOSS_SLOT_MSE]) if loss_values.size > self._LOSS_SLOT_MSE else float("nan"), float(signal_values.view(np.float32)[0]) if signal_values.size else float("nan")))(np.frombuffer(self._buffers["loss"].to_numpy().tobytes(), dtype=np.float32), np.frombuffer(self._buffers["signal_max"].to_numpy().tobytes(), dtype=np.uint32))

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
        pick = lambda value, fallback: fallback if value is None else value
        self.device, self.renderer, self.frames = device, renderer, frames
        self._seed = int(seed)
        self._scene_count = int(scene.count if scene is not None else max(int(scene_count if scene_count is not None else 0), 1))
        self.scene = scene if scene is not None else _SceneCountProxy(self._scene_count)
        self.adam, self.stability, self.training = pick(adam_hparams, AdamHyperParams()), pick(stability_hparams, StabilityHyperParams()), pick(training_hparams, TrainingHyperParams())
        self.state = TrainingState()
        self._metric_window_size = max(len(self.frames), 1)
        self._loss_window = _RollingMetricWindow(size=self._metric_window_size, values=deque())
        self._frame_signal_max = np.full((len(self.frames),), np.nan, dtype=np.float32)
        self._frame_psnr = np.full((len(self.frames),), np.nan, dtype=np.float32)
        self._shader_path = Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang")
        self._kernels = {name: self.device.create_compute_kernel(self.device.load_program(str(self._shader_path), [entry])) for name, entry in self._KERNEL_ENTRIES.items()}
        self._buffers: dict[str, spy.Buffer] = {}
        self._splat_capacity = 0
        self._scale_reg_reference = float(max(scale_reg_reference, 1e-8)) if scale_reg_reference is not None else self._estimate_scale_reg_reference(scene)
        self._init_point_positions_buffer: spy.Buffer | None = None
        self._init_point_colors_buffer: spy.Buffer | None = None
        self._init_point_count = 0
        self._frame_targets_train: list[spy.Texture] = []
        self._frame_targets_native: list[spy.Texture] = []
        self._loss_textures: dict[str, spy.Texture] = {}
        self._blur = SeparableGaussianBlur(self.device, self.renderer.width, self.renderer.height)
        self._frame_rng = np.random.default_rng(self._seed)
        self._frame_order = np.zeros((len(self.frames),), dtype=np.int32)
        self._frame_cursor = len(self.frames)
        if upload_initial_scene and scene is not None:
            self.renderer.set_scene(scene)
        else:
            self.renderer.bind_scene_count(self._scene_count)
        self._ensure_training_buffers(self._scene_count)
        self._ensure_loss_textures()
        self._zero_optimizer_moments()
        self._create_dataset_textures()
        self._bind_or_upload_init_pointcloud(init_point_positions, init_point_colors, init_point_positions_buffer, init_point_colors_buffer, init_point_count)
        self._reset_frame_order()

    def _ensure_training_buffers(self, splat_count: int) -> None:
        count = max(int(splat_count), 1)
        if self._buffers and count <= self._splat_capacity:
            return
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
        new_capacity = max(count, max(self._splat_capacity, 1) + max(self._splat_capacity, 1) // 2)
        self._buffers.setdefault("loss", self.device.create_buffer(size=8, usage=usage))
        self._buffers.setdefault("signal_max", self.device.create_buffer(size=4, usage=usage))
        self._buffers["low_quality_flags"] = self.device.create_buffer(size=new_capacity * 4, usage=usage)
        for name in self._ADAM_BUFFER_NAMES:
            self._buffers[name] = self.device.create_buffer(size=new_capacity * 16, usage=usage)
        self._splat_capacity = new_capacity

    def _estimate_scale_reg_reference(self, scene: GaussianScene | None) -> float:
        if scene is None or scene.count <= 0:
            return 1.0
        scales = np.asarray(scene.scales, dtype=np.float32)
        finite = np.isfinite(scales).all(axis=1)
        return 1.0 if not np.any(finite) else float(np.exp(np.mean(np.log(np.maximum(scales[finite], 1e-8)), dtype=np.float64)))

    def _ensure_loss_textures(self) -> None:
        if self._loss_textures:
            return
        self._loss_textures = {name: self._blur.make_texture() for name in self._LOSS_TEXTURE_NAMES}

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
            return
        if positions is None or colors is None:
            return
        pos, col = np.ascontiguousarray(positions, dtype=np.float32), np.ascontiguousarray(colors, dtype=np.float32)
        if pos.shape[0] != col.shape[0]:
            raise ValueError("init_point_positions and init_point_colors must have matching row count.")
        self._init_point_positions_buffer, self._init_point_colors_buffer, self._init_point_count = self._create_init_point_buffer(pos), self._create_init_point_buffer(col), int(pos.shape[0])

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
        self._frame_targets_train, self._frame_targets_native = [], []
        for frame in self.frames:
            with Image.open(frame.image_path) as pil_image:
                native = pil_image.convert("RGB")
                self._frame_targets_native.append(self._create_gpu_texture(native))
                self._frame_targets_train.append(self._create_gpu_texture(native if native.size == train_size else native.resize(train_size, resample=Image.Resampling.BILINEAR)))

    def _reset_metric_windows(self) -> None:
        self._loss_window.values.clear()
        self._frame_signal_max.fill(np.nan)
        self._frame_psnr.fill(np.nan)

    def _mean_finite(self, values: np.ndarray) -> float:
        finite = np.isfinite(values)
        return float(np.mean(values[finite], dtype=np.float64)) if np.any(finite) else float("nan")

    def _psnr_from_metrics(self, mse: float, signal_max: float) -> float:
        return (
            20.0 * float(np.log10(signal_max)) - 10.0 * float(np.log10(max(float(mse), self._PSNR_MSE_FLOOR)))
            if np.isfinite(mse) and mse >= 0.0 and np.isfinite(signal_max) and signal_max > 0.0
            else float("nan")
        )

    def _dispatch_loss_grad(self, encoder: spy.CommandEncoder, target_texture: spy.Texture) -> None:
        common = self._common_vars()
        shared = {"g_OutputGrad": self.renderer.output_grad_texture, "g_LossBuffer": self._buffers["loss"], "g_SignalMaxBuffer": self._buffers["signal_max"], **common}
        self._dispatch("clear_loss_grad", encoder, self._pixel_thread_count(), shared)
        self._dispatch(
            "pack_ssim_aux",
            encoder,
            self._pixel_thread_count(),
            {
                "g_Rendered": self.renderer.output_texture,
                "g_Target": target_texture,
                "g_RenderedSq": self._loss_textures["render_sq"],
                "g_TargetSq": self._loss_textures["target_sq"],
                "g_RenderTarget": self._loss_textures["cross"],
                **common,
            },
        )
        self._blur.blur(encoder, self.renderer.output_texture, self._loss_textures["mu_render"])
        self._blur.blur(encoder, target_texture, self._loss_textures["mu_target"])
        self._blur.blur(encoder, self._loss_textures["render_sq"], self._loss_textures["blur_render_sq"])
        self._blur.blur(encoder, self._loss_textures["target_sq"], self._loss_textures["blur_target_sq"])
        self._blur.blur(encoder, self._loss_textures["cross"], self._loss_textures["blur_cross"])
        self._dispatch(
            "loss_grad",
            encoder,
            self._pixel_thread_count(),
            {
                "g_Rendered": self.renderer.output_texture,
                "g_Target": target_texture,
                "g_OutputGrad": self.renderer.output_grad_texture,
                "g_LossBuffer": self._buffers["loss"],
                "g_SignalMaxBuffer": self._buffers["signal_max"],
                "g_MuRendered": self._loss_textures["mu_render"],
                "g_MuTarget": self._loss_textures["mu_target"],
                "g_BlurRenderedSq": self._loss_textures["blur_render_sq"],
                "g_BlurTargetSq": self._loss_textures["blur_target_sq"],
                "g_BlurRenderTarget": self._loss_textures["blur_cross"],
                "g_GradMuRendered": self._loss_textures["grad_mu"],
                "g_GradBlurRenderedSq": self._loss_textures["grad_xx"],
                "g_GradBlurRenderTarget": self._loss_textures["grad_xy"],
                **common,
            },
        )
        self._blur.blur(encoder, self._loss_textures["grad_mu"], self._loss_textures["grad_mu_blur"])
        self._blur.blur(encoder, self._loss_textures["grad_xx"], self._loss_textures["grad_xx_blur"])
        self._blur.blur(encoder, self._loss_textures["grad_xy"], self._loss_textures["grad_xy_blur"])
        self._dispatch(
            "compose_loss_grad",
            encoder,
            self._pixel_thread_count(),
            {
                "g_Rendered": self.renderer.output_texture,
                "g_Target": target_texture,
                "g_OutputGrad": self.renderer.output_grad_texture,
                "g_BlurredGradMuRendered": self._loss_textures["grad_mu_blur"],
                "g_BlurredGradRenderedSq": self._loss_textures["grad_xx_blur"],
                "g_BlurredGradRenderTarget": self._loss_textures["grad_xy_blur"],
                **common,
            },
        )

    def _common_vars(self) -> dict[str, object]:
        return {
            "g_Width": int(self.renderer.width),
            "g_Height": int(self.renderer.height),
            "g_SplatCount": int(self._scene_count),
            "g_LowQualityReinitEnabled": np.uint32(1 if self.training.low_quality_reinit_enabled else 0),
            "g_InvPixelCount": 1.0 / float(max(self.renderer.width * self.renderer.height, 1)),
            "g_LossGradClip": float(self.stability.loss_grad_clip),
            "g_ScaleL2Weight": float(max(self.training.scale_l2_weight, 0.0)),
            "g_ScaleRegReference": float(max(self._scale_reg_reference, 1e-8)),
            "g_LambdaDSSIM": float(np.clip(self.training.lambda_dssim, 0.0, 1.0)),
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
                "maxAnisotropy": float(max(self.stability.max_anisotropy, 1.0)),
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

    def initialize_scene_from_pointcloud(self, splat_count: int, init_hparams: GaussianInitHyperParams, seed: int) -> None:
        if self._init_point_positions_buffer is None or self._init_point_colors_buffer is None or self._init_point_count <= 0:
            raise RuntimeError("Pointcloud initializer buffers are not available.")
        self._scene_count, self.scene = max(int(splat_count), 1), _SceneCountProxy(max(int(splat_count), 1))
        self.renderer.bind_scene_count(self._scene_count)
        self._ensure_training_buffers(self._scene_count)
        self._scale_reg_reference = float(max(init_hparams.base_scale, 1e-8))
        enc = self.device.create_command_encoder()
        self._dispatch(
            "init_from_pointcloud",
            enc,
            self._scene_thread_count(),
            {
                **self._scene_rw_vars(),
                **self._adam_shader_vars(),
                "g_InitPointPositions": self._init_point_positions_buffer,
                "g_InitPointColors": self._init_point_colors_buffer,
                "g_InitPointCount": int(self._init_point_count),
                "g_InitSeed": int(seed),
                "g_InitPositionJitterStd": float(max(init_hparams.position_jitter_std, 0.0)),
                "g_InitBaseScale": float(max(init_hparams.base_scale, 1e-4)),
                "g_InitScaleJitterRatio": float(max(init_hparams.scale_jitter_ratio, 0.0)),
                "g_InitInitialOpacity": float(np.clip(init_hparams.initial_opacity, 0.0, 1.0)),
                "g_InitColorJitterStd": float(max(init_hparams.color_jitter_std, 0.0)),
                **self._common_vars(),
            },
        )
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()
        self.state = TrainingState()
        self._reset_metric_windows()
        self._frame_rng = np.random.default_rng(self._seed)
        self._reset_frame_order()

    def _update_psnr_state(self, frame_index: int, mse: float, signal_max: float) -> None:
        self.state.last_mse = float(mse)
        idx = int(np.clip(frame_index, 0, len(self.frames) - 1))
        self._frame_signal_max[idx] = float(signal_max) if np.isfinite(signal_max) and signal_max > 0.0 else np.nan
        self.state.avg_signal_max = self._mean_finite(self._frame_signal_max)
        self.state.last_psnr = self._psnr_from_metrics(mse, float(self._frame_signal_max[idx]))
        self._frame_psnr[idx] = float(self.state.last_psnr) if np.isfinite(self.state.last_psnr) else np.nan
        self.state.avg_psnr = self._mean_finite(self._frame_psnr)

    def step(self) -> float:
        frame_index = self._next_frame_index()
        frame_camera = self.make_frame_camera(frame_index, self.renderer.width, self.renderer.height)
        background = np.asarray(self.training.background, dtype=np.float32).reshape(3)
        self.renderer.execute_prepass_for_current_scene(frame_camera, sync_counts=False)
        enc = self.device.create_command_encoder()
        self.renderer.rasterize_current_scene(enc, frame_camera, background)
        self._dispatch_loss_grad(enc, self.get_frame_target_texture(frame_index, native_resolution=False))
        self._dispatch_raster_forward_backward(enc, frame_camera, background)
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()

        image_loss, image_mse, signal_max = self._read_loss_metrics()
        if not np.isfinite(image_loss):
            self.state.last_instability, loss = "Non-finite loss; ADAM step skipped and moments reset.", float("inf")
            self._zero_optimizer_moments()
        else:
            self.state.last_instability = ""
            enc_opt = self.device.create_command_encoder()
            self._dispatch_adam_step(enc_opt)
            self._dispatch_mark_low_quality_splats(enc_opt)
            self._dispatch_resample_low_quality_splats(enc_opt)
            self.device.submit_command_buffer(enc_opt.finish())
            self.device.wait()
            loss = self._read_loss_metrics()[0]
            if not np.isfinite(loss):
                self.state.last_instability, loss = "Non-finite regularized loss after ADAM; using image loss.", image_loss

        self.state.step += 1
        self.state.last_frame_index = frame_index
        self.state.last_loss = float(loss)
        self._loss_window.push(loss)
        self.state.avg_loss = self._loss_window.mean()
        self._update_psnr_state(frame_index, image_mse, signal_max)
        return float(loss)
