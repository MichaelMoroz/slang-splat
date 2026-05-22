from __future__ import annotations

from typing import Any

import numpy as np
import slangpy as spy
import math

from ..utility import RO_BUFFER_USAGE, SHADER_ROOT, alloc_buffer, defer_resource_release, dispatch, load_compute_kernels, thread_count_1d
from ..renderer import Camera, GaussianRenderer
from .schedule import resolve_color_lr_mul, resolve_learning_rate_scale, resolve_max_opacity, resolve_max_visible_angle_deg, resolve_opacity_lr_mul, resolve_opacity_reg_weight, resolve_position_lr_mul, resolve_position_push_away_from_camera_step, resolve_rotation_lr_mul, resolve_scale_lr_mul, resolve_sh_lr_mul


def _nonzero_denominator(value: float) -> float:
    return value if value != 0.0 else 1.0


class GaussianOptimizer:
    _PARAM_SETTINGS_U32_WIDTH = 7
    _threads = staticmethod(thread_count_1d)

    def __init__(self, device: spy.Device, renderer: GaussianRenderer, adam_hparams: Any, stability_hparams: Any) -> None:
        self.device = device
        self.renderer = renderer
        self.adam = adam_hparams
        self.stability = stability_hparams
        self._buffers: dict[str, spy.Buffer] = {}
        self._param_settings_count = 0
        self._kernels = self._create_kernels()
        self._ensure_static_buffers()
        self._upload_param_settings()

    def _create_kernels(self) -> dict[str, spy.ComputeKernel]:
        return load_compute_kernels(
            self.device,
            SHADER_ROOT / "utility" / "optimizer" / "gaussian_optimizer_stage.slang",
            {
                "project_params": "csProjectGaussianParams",
            },
        )

    def _stability_vars(self) -> dict[str, object]:
        return {
            "g_Stability": {
                "gradComponentClip": float(self.stability.grad_component_clip),
                "maxUpdate": float(self.stability.max_update),
                "maxAnisotropy": float(self.stability.max_anisotropy),
                "minOpacity": float(self.stability.min_opacity),
                "maxOpacity": float(self.stability.max_opacity),
                "positionAbsMax": float(self.stability.position_abs_max),
                "hugeValue": float(self.stability.huge_value),
            },
        }

    def _ensure_static_buffers(self) -> None:
        count = int(self.renderer.packed_trainable_param_count)
        if "param_settings" in self._buffers and self._param_settings_count == count:
            return
        defer_resource_release(self._buffers.get("param_settings"))
        self._buffers["param_settings"] = alloc_buffer(
            self.device,
            name="gaussian_optimizer.param_settings",
            size=count * self._PARAM_SETTINGS_U32_WIDTH * 4,
            usage=RO_BUFFER_USAGE,
        )
        self._param_settings_count = count

    @staticmethod
    def _raw_opacity_from_alpha(alpha: float) -> float:
        alpha_clamped = float(np.clip(alpha, 1e-6, 1.0 - 1e-6))
        return float(np.log(alpha_clamped) - np.log1p(-alpha_clamped))

    @staticmethod
    def _float_bits(values: Any) -> np.ndarray:
        return np.asarray(values, dtype=np.float32).view(np.uint32)

    def _param_settings(self, lr_scale: float = 1.0, lr_mul_scales: tuple[float, float, float, float, float, float] | None = None, training_hparams: Any | None = None, scale_reg_reference: float = 1.0, max_opacity: float | None = None) -> np.ndarray:
        count = int(self.renderer.packed_trainable_param_count)
        settings = np.zeros((count, self._PARAM_SETTINGS_U32_WIDTH), dtype=np.uint32)
        position_ids = np.asarray(self.renderer.PARAM_POSITION_IDS, dtype=np.intp)
        scale_ids = np.asarray(self.renderer.PARAM_SCALE_IDS, dtype=np.intp)
        rotation_ids = np.asarray(self.renderer.PARAM_ROTATION_IDS, dtype=np.intp)
        raw_opacity_id = int(self.renderer.packed_raw_opacity_param_id)
        color_ids = np.arange(self.renderer.PARAM_ROTATION_IDS[-1] + 1, raw_opacity_id, dtype=np.intp)
        sh0_ids = color_ids[:3]
        higher_sh_ids = color_ids[3:]
        scale = float(lr_scale)
        position_scale, scale_scale, rotation_scale, color_scale, opacity_scale, sh_scale = (float(v) for v in (lr_mul_scales or (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)))
        scale_ref_value = float(np.log(max(float(scale_reg_reference), 1e-8)))
        scale_l2_weight = 0.0 if training_hparams is None else float(training_hparams.scale_l2_weight) * (2.0 / 3.0)
        stored_sh_coeff_count = int(self.renderer.stored_sh_coeff_count)
        higher_sh_component_count = (stored_sh_coeff_count - 1) * 3 if stored_sh_coeff_count > 1 else 0
        sh1_weight = 0.0 if training_hparams is None or higher_sh_component_count <= 0 else float(training_hparams.sh1_reg_weight) / float(higher_sh_component_count)
        resolved_max_opacity = float(self.stability.max_opacity if max_opacity is None else max_opacity)

        lrs = np.full((count,), float(self.adam.opacity_lr), dtype=np.float32)
        lrs[position_ids] = float(self.adam.position_lr)
        lrs[scale_ids] = float(self.adam.scale_lr)
        lrs[rotation_ids] = float(self.adam.rotation_lr)
        lrs[color_ids] = float(self.adam.color_lr)

        param_scales = np.ones((count,), dtype=np.float32)
        param_scales[position_ids] = position_scale
        param_scales[scale_ids] = scale_scale
        param_scales[rotation_ids] = rotation_scale
        param_scales[sh0_ids] = color_scale
        param_scales[higher_sh_ids] = sh_scale
        param_scales[raw_opacity_id] = opacity_scale

        value_mins = np.full((count,), -float(self.stability.huge_value), dtype=np.float32)
        value_maxs = -value_mins
        value_mins[position_ids] = -float(self.stability.position_abs_max)
        value_maxs[position_ids] = float(self.stability.position_abs_max)
        value_mins[raw_opacity_id] = self._raw_opacity_from_alpha(float(self.stability.min_opacity))
        value_maxs[raw_opacity_id] = self._raw_opacity_from_alpha(resolved_max_opacity)

        settings[:, 0] = self._float_bits(lrs * scale * param_scales)
        settings[:, 1] = self._float_bits(float(self.stability.grad_component_clip))
        settings[:, 2] = self._float_bits(value_mins)
        settings[:, 3] = self._float_bits(value_maxs)
        if scale_l2_weight != 0.0:
            settings[scale_ids, 4] = self._float_bits(scale_ref_value)
            settings[scale_ids, 5] = self._float_bits(scale_l2_weight)
        if sh1_weight != 0.0 and len(higher_sh_ids) > 0:
            settings[higher_sh_ids, 6] = self._float_bits(sh1_weight)
        return settings

    def _upload_param_settings(self, lr_scale: float = 1.0, lr_mul_scales: tuple[float, float, float, float, float, float] | None = None, training_hparams: Any | None = None, scale_reg_reference: float = 1.0, max_opacity: float | None = None) -> None:
        self._ensure_static_buffers()
        resolved_scales = lr_mul_scales or (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        self._buffers["param_settings"].copy_from_numpy(self._param_settings(lr_scale, resolved_scales, training_hparams=training_hparams, scale_reg_reference=scale_reg_reference, max_opacity=max_opacity))

    def update_hyperparams(self, adam_hparams: Any, stability_hparams: Any) -> None:
        self.adam = adam_hparams
        self.stability = stability_hparams
        self._upload_param_settings()

    def rebind_renderer(self, renderer: GaussianRenderer) -> None:
        self.renderer = renderer
        self._upload_param_settings()

    def update_step(self, step_index: int, training_hparams: Any, scale_reg_reference: float) -> None:
        lr_scale = resolve_learning_rate_scale(training_hparams, int(step_index))
        color_lr_mul_scale = resolve_color_lr_mul(training_hparams, int(step_index)) / _nonzero_denominator(float(getattr(training_hparams, "lr_color_mul", 1.0)))
        max_opacity = resolve_max_opacity(training_hparams, int(step_index))
        self._upload_param_settings(
            lr_scale,
            (
            resolve_position_lr_mul(training_hparams, int(step_index)) / _nonzero_denominator(float(getattr(training_hparams, "lr_pos_mul", 1.0))),
            resolve_scale_lr_mul(training_hparams, int(step_index)) / _nonzero_denominator(float(getattr(training_hparams, "lr_scale_mul", 1.0))),
            resolve_rotation_lr_mul(training_hparams, int(step_index)) / _nonzero_denominator(float(getattr(training_hparams, "lr_rot_mul", 1.0))),
            color_lr_mul_scale,
            resolve_opacity_lr_mul(training_hparams, int(step_index)) / _nonzero_denominator(float(getattr(training_hparams, "lr_opacity_mul", 1.0))),
            color_lr_mul_scale * resolve_sh_lr_mul(training_hparams, int(step_index)),
            ),
            training_hparams=training_hparams,
            scale_reg_reference=scale_reg_reference,
            max_opacity=max_opacity,
        )

    @property
    def param_settings(self) -> spy.Buffer:
        return self._buffers["param_settings"]

    @property
    def param_settings_count(self) -> int:
        return int(self.renderer.packed_trainable_param_count)

    def dispatch_projection(
        self,
        encoder: spy.CommandEncoder,
        *,
        scene_buffers: dict[str, spy.Buffer],
        splat_count: int,
        training_hparams: Any,
        frame_camera: Camera | None = None,
        width: int | None = None,
        height: int | None = None,
        step_index: int = 0,
        splat_contribution_buffer: spy.Buffer | None = None,
    ) -> None:
        camera_vars: dict[str, object]
        if frame_camera is None or width is None or height is None:
            camera_vars = {
                "g_CurrentCamera": {
                    "viewport": spy.float2(1.0, 1.0),
                    "camPos": spy.float3(0.0, 0.0, 0.0),
                    "camBasis": spy.float3x3(np.eye(3, dtype=np.float32)),
                    "focalPixels": spy.float2(1.0, 1.0),
                    "principalPoint": spy.float2(0.0, 0.0),
                    "nearDepth": 0.0,
                    "farDepth": 0.0,
                    "projDistortionK1K2P1P2": spy.float4(0.0, 0.0, 0.0, 0.0),
                    "projDistortionK3K4K5K6": spy.float4(0.0, 0.0, 0.0, 0.0),
                    "minCameraDistance": 0.0,
                },
                "g_EnableCurrentCameraScreenScaleCap": np.uint32(0),
            }
        else:
            camera_vars = {
                "g_CurrentCamera": {
                    **frame_camera.gpu_params(int(width), int(height)),
                },
                "g_EnableCurrentCameraScreenScaleCap": np.uint32(1),
            }
        camera_position = np.zeros((3,), dtype=np.float32) if frame_camera is None else np.asarray(frame_camera.position, dtype=np.float32).reshape(3)
        push_step = 0.0 if frame_camera is None else float(resolve_position_push_away_from_camera_step(training_hparams, int(step_index)))
        opacity_reg_weight = float(resolve_opacity_reg_weight(training_hparams, int(step_index)))
        dispatch(
            kernel=self._kernels["project_params"],
            thread_count=self._threads(splat_count),
            vars={
                "g_SplatParamsRW": scene_buffers["splat_params"],
                **camera_vars,
                "g_SplatCount": int(splat_count),
                "g_OptimizerParamSettings": self.param_settings,
                "g_GaussianOptimizerRegularization": spy.float3(float(training_hparams.scale_abs_reg_weight), opacity_reg_weight, push_step),
                "g_GaussianOptimizerRegularizationCameraPosition": spy.float3(*camera_position.tolist()),
                "g_GaussianOptimizerSplatContributionInfo": self.renderer.work_buffers["training_splat_contribution"] if splat_contribution_buffer is None else splat_contribution_buffer,
                "g_MaxViewAngleSlope": float(math.tan(math.radians(float(resolve_max_visible_angle_deg(training_hparams, int(step_index)))))),
                "g_RendererRadiusScale": float(self.renderer.radius_scale),
                "g_SHBand": np.uint32(int(self.renderer.sh_band)),
                "g_SHNonNegativeStepIndex": np.uint32(int(step_index)),
                "g_StoredPackedParamCount": np.uint32(int(self.renderer.packed_trainable_param_count)),
                **self._stability_vars(),
            },
            command_encoder=encoder,
            debug_label="Gaussian Post Step",
            debug_color_index=71,
        )
