from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import slangpy as spy

from ..common import SHADER_ROOT
from ..renderer import Camera, GaussianRenderer
from ..scene import ColmapFrame, GaussianScene


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
    target_flip_y: bool = False
    ema_decay: float = 0.95
    mcmc_position_noise_enabled: bool = True
    mcmc_position_noise_scale: float = 1.0
    mcmc_opacity_gate_sharpness: float = 100.0
    mcmc_opacity_gate_center: float = 0.995


@dataclass(slots=True)
class TrainingState:
    step: int = 0
    last_loss: float = float("nan")
    ema_loss: float = float("nan")
    last_frame_index: int = -1
    last_instability: str = ""


class GaussianTrainer:
    def __init__(
        self,
        device: spy.Device,
        renderer: GaussianRenderer,
        scene: GaussianScene,
        frames: list[ColmapFrame],
        adam_hparams: AdamHyperParams | None = None,
        stability_hparams: StabilityHyperParams | None = None,
        training_hparams: TrainingHyperParams | None = None,
        seed: int = 0,
    ) -> None:
        if not frames:
            raise ValueError("Training requires at least one COLMAP frame.")
        self.device = device
        self.renderer = renderer
        self.scene = scene
        self.frames = frames
        self.adam = adam_hparams if adam_hparams is not None else AdamHyperParams()
        self.stability = stability_hparams if stability_hparams is not None else StabilityHyperParams()
        self.training = training_hparams if training_hparams is not None else TrainingHyperParams()
        self.state = TrainingState()
        self._rng = np.random.default_rng(int(seed))
        self._shader_path = Path(SHADER_ROOT / "renderer" / "gaussian_training_stage.slang")
        self._kernels: dict[str, spy.ComputeKernel] = {}
        self._buffers: dict[str, spy.Buffer] = {}
        self._frame_targets_train: list[spy.Texture] = []
        self._frame_targets_native: list[spy.Texture] = []
        self._target_flip_cached = bool(self.training.target_flip_y)
        self.renderer.set_scene(self.scene)
        self._create_shaders()
        self._create_training_buffers()
        self._create_dataset_textures()

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
        }

    def _create_training_buffers(self) -> None:
        count = max(int(self.scene.count), 1)
        usage_rw = (
            spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access
            | spy.BufferUsage.copy_source
            | spy.BufferUsage.copy_destination
        )
        self._buffers["loss"] = self.device.create_buffer(size=4, usage=usage_rw)
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
            self._buffers[name] = self.device.create_buffer(size=count * 16, usage=usage_rw)
        self._zero_optimizer_moments()

    def _zero_optimizer_moments(self) -> None:
        count = max(int(self.scene.count), 1)
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

    def _to_rgba(self, rgb: np.ndarray) -> np.ndarray:
        height, width = rgb.shape[:2]
        alpha = np.ones((height, width, 1), dtype=np.float32)
        return np.concatenate((rgb.astype(np.float32), alpha), axis=2).astype(np.float32)

    def _create_gpu_texture(self, rgba: np.ndarray) -> spy.Texture:
        height, width = rgba.shape[:2]
        tex = self.device.create_texture(
            format=spy.Format.rgba32_float,
            width=int(width),
            height=int(height),
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
        )
        tex.copy_from_numpy(np.ascontiguousarray(rgba, dtype=np.float32))
        return tex

    def _create_dataset_textures(self) -> None:
        flip_y = bool(self.training.target_flip_y)
        self._frame_targets_train = []
        self._frame_targets_native = []
        train_width = int(self.renderer.width)
        train_height = int(self.renderer.height)
        for frame in self.frames:
            with Image.open(frame.image_path) as pil_image:
                image = pil_image.convert("RGB")
                native_rgb = np.asarray(image, dtype=np.float32) / 255.0
                if flip_y:
                    native_rgb = np.flipud(native_rgb).copy()
                native_rgba = self._to_rgba(native_rgb)
                self._frame_targets_native.append(self._create_gpu_texture(native_rgba))

                if image.size != (train_width, train_height):
                    image_train = image.resize((train_width, train_height), resample=Image.Resampling.BILINEAR)
                    train_rgb = np.asarray(image_train, dtype=np.float32) / 255.0
                    if flip_y:
                        train_rgb = np.flipud(train_rgb).copy()
                else:
                    train_rgb = native_rgb
                train_rgba = self._to_rgba(train_rgb)
                self._frame_targets_train.append(self._create_gpu_texture(train_rgba))
        self._target_flip_cached = flip_y

    def _refresh_dataset_if_needed(self) -> None:
        if bool(self.training.target_flip_y) != self._target_flip_cached:
            self._create_dataset_textures()

    def update_hyperparams(
        self,
        adam_hparams: AdamHyperParams,
        stability_hparams: StabilityHyperParams,
        training_hparams: TrainingHyperParams,
    ) -> None:
        old_flip = bool(self.training.target_flip_y)
        self.adam = adam_hparams
        self.stability = stability_hparams
        self.training = training_hparams
        if bool(self.training.target_flip_y) != old_flip:
            self._create_dataset_textures()

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
            "g_SplatCount": int(self.scene.count),
            "g_InvPixelCount": 1.0 / float(max(self.renderer.width * self.renderer.height, 1)),
            "g_LossGradClip": float(self.stability.loss_grad_clip),
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
                **vars_common,
            },
            command_encoder=encoder,
        )

    def _dispatch_raster_backward(
        self,
        encoder: spy.CommandEncoder,
        frame_camera,
        background: np.ndarray,
    ) -> None:
        self.renderer.clear_raster_grads_current_scene(encoder)
        self.renderer.rasterize_backward_current_scene(
            encoder=encoder,
            camera=frame_camera,
            background=background,
            output_grad=self.renderer.output_grad_texture,
        )

    def _dispatch_adam_step(self, encoder: spy.CommandEncoder) -> None:
        vars_common = self._common_vars()
        self._kernels["adam_step"].dispatch(
            thread_count=spy.uint3(self.scene.count, 1, 1),
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

    def _read_loss_value(self) -> float:
        raw = self._buffers["loss"].to_numpy()
        values = np.frombuffer(raw.tobytes(), dtype=np.float32)
        return float(values[0]) if values.size else float("nan")

    def step(self) -> float:
        self._refresh_dataset_if_needed()
        frame_index = int(self._rng.integers(0, len(self.frames)))
        frame_camera = self.make_frame_camera(frame_index, self.renderer.width, self.renderer.height)
        background = np.asarray(self.training.background, dtype=np.float32).reshape(3)
        target_texture = self.get_frame_target_texture(frame_index, native_resolution=False)
        self.renderer.execute_prepass_for_current_scene(frame_camera, sync_counts=False)

        enc = self.device.create_command_encoder()
        self.renderer.rasterize_current_scene(enc, frame_camera, background)
        self._dispatch_loss_grad(enc, target_texture)
        self._dispatch_raster_backward(enc, frame_camera, background)
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()

        loss = self._read_loss_value()
        if not np.isfinite(loss):
            self.state.last_instability = "Non-finite loss; ADAM step skipped and moments reset."
            self._zero_optimizer_moments()
            loss = float("inf")
        else:
            self.state.last_instability = ""
            enc_opt = self.device.create_command_encoder()
            self._dispatch_adam_step(enc_opt)
            self.device.submit_command_buffer(enc_opt.finish())

        self.state.step += 1
        self.state.last_frame_index = frame_index
        self.state.last_loss = float(loss)
        if not np.isfinite(self.state.ema_loss):
            self.state.ema_loss = float(loss)
        else:
            decay = float(np.clip(self.training.ema_decay, 0.0, 0.99999))
            self.state.ema_loss = decay * self.state.ema_loss + (1.0 - decay) * float(loss)
        return float(loss)
