from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import slangpy as spy

from ..common import SHADER_ROOT
from ..renderer import GaussianRenderer
from ..scene import ColmapFrame, GaussianScene


@dataclass(slots=True)
class AdamHyperParams:
    learning_rate: float = 1e-3
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
    target_flip_y: bool = True
    ema_decay: float = 0.95


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
        self._textures: dict[str, spy.Texture] = {}
        self.renderer.set_scene(self.scene)
        self._create_shaders()
        self._create_training_buffers()

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
        if "target" not in self._textures or self._textures["target"].width != self.renderer.width or self._textures[
            "target"
        ].height != self.renderer.height:
            self._textures["target"] = self.device.create_texture(
                format=spy.Format.rgba32_float,
                width=self.renderer.width,
                height=self.renderer.height,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
            )
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

    def _load_target_image(self, path: Path) -> np.ndarray:
        with Image.open(path) as pil_image:
            image = pil_image.convert("RGB")
            if image.size != (self.renderer.width, self.renderer.height):
                image = image.resize((self.renderer.width, self.renderer.height), resample=Image.Resampling.BILINEAR)
            target_rgb = np.asarray(image, dtype=np.float32) / 255.0
        if self.training.target_flip_y:
            target_rgb = np.flipud(target_rgb).copy()
        alpha = np.ones((self.renderer.height, self.renderer.width, 1), dtype=np.float32)
        return np.concatenate((target_rgb, alpha), axis=2).astype(np.float32)

    def _upload_target_image(self, target_rgba: np.ndarray) -> None:
        self._textures["target"].copy_from_numpy(np.ascontiguousarray(target_rgba, dtype=np.float32))

    def _common_vars(self, camera_position: np.ndarray) -> dict[str, object]:
        return {
            "g_Width": int(self.renderer.width),
            "g_Height": int(self.renderer.height),
            "g_SplatCount": int(self.scene.count),
            "g_RadiusScale": float(self.renderer.radius_scale),
            "g_InvPixelCount": 1.0 / float(max(self.renderer.width * self.renderer.height, 1)),
            "g_LossGradClip": float(self.stability.loss_grad_clip),
            "g_CamPos": spy.float3(*camera_position.astype(np.float32).tolist()),
            "g_Adam": {
                "learningRate": float(self.adam.learning_rate),
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
                "minInvScale": float(self.stability.min_inv_scale),
            },
        }

    def _dispatch_loss_grad(self, encoder: spy.CommandEncoder, camera_position: np.ndarray) -> None:
        vars_common = self._common_vars(camera_position)
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
                "g_Target": self._textures["target"],
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

    def _dispatch_adam_step(self, encoder: spy.CommandEncoder, camera_position: np.ndarray) -> None:
        vars_common = self._common_vars(camera_position)
        self._kernels["adam_step"].dispatch(
            thread_count=spy.uint3(self.scene.count, 1, 1),
            vars={
                "g_SplatInvScale": self.renderer.work_buffers["splat_inv_scale"],
                "g_SplatQuat": self.renderer.work_buffers["splat_quat"],
                "g_GradSplatPosLocal": self.renderer.work_buffers["grad_splat_pos_local"],
                "g_GradSplatInvScale": self.renderer.work_buffers["grad_splat_inv_scale"],
                "g_GradSplatQuat": self.renderer.work_buffers["grad_splat_quat"],
                "g_GradScreenColorAlpha": self.renderer.work_buffers["grad_screen_color_alpha"],
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
        frame_index = int(self._rng.integers(0, len(self.frames)))
        frame = self.frames[frame_index]
        frame_camera = frame.make_camera(near=float(self.training.near), far=float(self.training.far))
        background = np.asarray(self.training.background, dtype=np.float32).reshape(3)
        target_rgba = self._load_target_image(frame.image_path)
        self._upload_target_image(target_rgba)
        self.renderer.execute_prepass_for_current_scene(frame_camera, sync_counts=False)

        enc = self.device.create_command_encoder()
        self.renderer.rasterize_current_scene(enc, frame_camera, background)
        self._dispatch_loss_grad(enc, frame_camera.position)
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
            self._dispatch_adam_step(enc_opt, frame_camera.position)
            self.device.submit_command_buffer(enc_opt.finish())
            self.device.wait()

        self.state.step += 1
        self.state.last_frame_index = frame_index
        self.state.last_loss = float(loss)
        if not np.isfinite(self.state.ema_loss):
            self.state.ema_loss = float(loss)
        else:
            decay = float(np.clip(self.training.ema_decay, 0.0, 0.99999))
            self.state.ema_loss = decay * self.state.ema_loss + (1.0 - decay) * float(loss)
        return float(loss)
