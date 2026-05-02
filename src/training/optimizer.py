from __future__ import annotations

from typing import Any

import numpy as np
import slangpy as spy
import math

from ..utility import RO_BUFFER_USAGE, SHADER_ROOT, alloc_buffer, defer_resource_release, dispatch, load_compute_kernels, thread_count_1d
from ..renderer import Camera, GaussianRenderer
from .schedule import resolve_color_lr_mul, resolve_learning_rate_scale, resolve_max_visible_angle_deg, resolve_opacity_lr_mul, resolve_position_lr_mul, resolve_position_push_away_from_camera_step, resolve_rotation_lr_mul, resolve_scale_lr_mul, resolve_sh_lr_mul


class GaussianOptimizer:
    _GROUPS = ((0, 3), (3, 3), (6, 4), (10, 12), (22, 1))
    _PARAM_SETTINGS_U32_WIDTH = 8
    _threads = staticmethod(thread_count_1d)

    def __init__(self, device: spy.Device, renderer: GaussianRenderer, adam_hparams: Any, stability_hparams: Any) -> None:
        self.device = device
        self.renderer = renderer
        self.adam = adam_hparams
        self.stability = stability_hparams
        self._uploaded_lr_scale = float("nan")
        self._uploaded_lr_mul_scales = (float("nan"),) * 6
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

    def _lr_for_param(self, param_id: int) -> float:
        if param_id in self.renderer.PARAM_POSITION_IDS: return float(self.adam.position_lr)
        if param_id in self.renderer.PARAM_SCALE_IDS: return float(self.adam.scale_lr)
        if param_id in self.renderer.PARAM_ROTATION_IDS: return float(self.adam.rotation_lr)
        if param_id in self.renderer.PARAM_COLOR_IDS: return float(self.adam.color_lr)
        return float(self.adam.opacity_lr)

    def _value_min_for_param(self, param_id: int) -> float:
        if param_id in self.renderer.PARAM_POSITION_IDS: return -float(self.stability.position_abs_max)
        if param_id == self.renderer.packed_raw_opacity_param_id: return self._raw_opacity_from_alpha(float(self.stability.min_opacity))
        return -float(self.stability.huge_value)

    def _value_max_for_param(self, param_id: int) -> float:
        if param_id in self.renderer.PARAM_POSITION_IDS: return float(self.stability.position_abs_max)
        if param_id in self.renderer.PARAM_SCALE_IDS: return float(np.log(max(self.stability.max_scale, 1e-8)))
        if param_id == self.renderer.packed_raw_opacity_param_id: return self._raw_opacity_from_alpha(float(self.stability.max_opacity))
        return float(self.stability.huge_value)

    def _group_for_param(self, param_id: int) -> tuple[int, int]:
        color_group_size = min(self.renderer.stored_sh_coeff_count * 3, 12)
        groups = ((0, 3), (3, 3), (6, 4), (10, color_group_size), (self.renderer.packed_raw_opacity_param_id, 1))
        for group_start, group_size in groups:
            if group_start <= int(param_id) < group_start + group_size: return int(group_start), int(group_size)
        return int(param_id), 1

    def _param_settings(self, lr_scale: float = 1.0, lr_mul_scales: tuple[float, float, float, float, float, float] | None = None) -> np.ndarray:
        settings = np.zeros((self.renderer.packed_trainable_param_count, self._PARAM_SETTINGS_U32_WIDTH), dtype=np.uint32)
        scale = max(float(lr_scale), 0.0)
        position_scale, scale_scale, rotation_scale, color_scale, opacity_scale, sh_scale = tuple(max(float(v), 0.0) for v in (lr_mul_scales or (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)))
        for param_id in range(self.renderer.packed_trainable_param_count):
            group_start, group_size = self._group_for_param(param_id)
            if param_id in self.renderer.PARAM_POSITION_IDS:
                param_scale = position_scale
            elif param_id in self.renderer.PARAM_SCALE_IDS:
                param_scale = scale_scale
            elif param_id in self.renderer.PARAM_ROTATION_IDS:
                param_scale = rotation_scale
            elif param_id in self.renderer.PARAM_SH_IDS and param_id not in self.renderer.PARAM_SH0_IDS:
                param_scale = sh_scale
            elif param_id in self.renderer.PARAM_COLOR_IDS:
                param_scale = color_scale
            elif param_id == self.renderer.packed_raw_opacity_param_id:
                param_scale = opacity_scale
            else:
                param_scale = 1.0
            settings[param_id, 0] = np.asarray([self._lr_for_param(param_id) * scale * param_scale], dtype=np.float32).view(np.uint32)[0]
            settings[param_id, 1] = np.asarray([float(self.stability.grad_component_clip)], dtype=np.float32).view(np.uint32)[0]
            settings[param_id, 2] = np.asarray([float(self.stability.grad_norm_clip)], dtype=np.float32).view(np.uint32)[0]
            settings[param_id, 3] = np.asarray([self._value_min_for_param(param_id)], dtype=np.float32).view(np.uint32)[0]
            settings[param_id, 4] = np.asarray([self._value_max_for_param(param_id)], dtype=np.float32).view(np.uint32)[0]
            settings[param_id, 5] = np.uint32(group_start)
            settings[param_id, 6] = np.uint32(group_size)
        return settings

    def _upload_param_settings(self, lr_scale: float = 1.0, lr_mul_scales: tuple[float, float, float, float, float, float] | None = None) -> None:
        self._ensure_static_buffers()
        resolved_scales = lr_mul_scales or (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        self._buffers["param_settings"].copy_from_numpy(self._param_settings(lr_scale, resolved_scales))
        self._uploaded_lr_scale = float(lr_scale)
        self._uploaded_lr_mul_scales = tuple(float(v) for v in resolved_scales)

    def update_hyperparams(self, adam_hparams: Any, stability_hparams: Any) -> None:
        self.adam = adam_hparams
        self.stability = stability_hparams
        self._upload_param_settings()

    def rebind_renderer(self, renderer: GaussianRenderer) -> None:
        self.renderer = renderer
        self._upload_param_settings()

    def update_step(self, step_index: int, training_hparams: Any) -> None:
        lr_scale = resolve_learning_rate_scale(training_hparams, int(step_index))
        color_lr_mul_scale = resolve_color_lr_mul(training_hparams, int(step_index)) / max(float(getattr(training_hparams, "lr_color_mul", 1.0)), 1e-8)
        lr_mul_scales = (
            resolve_position_lr_mul(training_hparams, int(step_index)) / max(float(getattr(training_hparams, "lr_pos_mul", 1.0)), 1e-8),
            resolve_scale_lr_mul(training_hparams, int(step_index)) / max(float(getattr(training_hparams, "lr_scale_mul", 1.0)), 1e-8),
            resolve_rotation_lr_mul(training_hparams, int(step_index)) / max(float(getattr(training_hparams, "lr_rot_mul", 1.0)), 1e-8),
            color_lr_mul_scale,
            resolve_opacity_lr_mul(training_hparams, int(step_index)) / max(float(getattr(training_hparams, "lr_opacity_mul", 1.0)), 1e-8),
            color_lr_mul_scale * resolve_sh_lr_mul(training_hparams, int(step_index)),
        )
        if (
            np.isfinite(self._uploaded_lr_scale)
            and all(np.isfinite(v) for v in self._uploaded_lr_mul_scales)
            and abs(self._uploaded_lr_scale - lr_scale) <= 1e-12
            and all(abs(a - b) <= 1e-12 for a, b in zip(self._uploaded_lr_mul_scales, lr_mul_scales))
        ):
            return
        self._upload_param_settings(lr_scale, lr_mul_scales)

    def _projection_vars(self, splat_count: int, training_hparams: Any, step_index: int, max_visible_angle_deg: float | None = None) -> dict[str, object]:
        resolved_max_visible_angle_deg = resolve_max_visible_angle_deg(training_hparams, 0) if max_visible_angle_deg is None else max_visible_angle_deg
        return {
            "g_SplatCount": int(splat_count),
            "g_MaxViewAngleSlope": float(max(math.tan(math.radians(min(max(float(resolved_max_visible_angle_deg), 1e-8), 89.999))), 1e-8)),
            "g_RendererRadiusScale": float(self.renderer.radius_scale),
            "g_SHBand": np.uint32(int(self.renderer.sh_band)),
            "g_SHNonNegativeStepIndex": np.uint32(int(step_index)),
            "g_StoredPackedParamCount": np.uint32(int(self.renderer.packed_trainable_param_count)),
            "g_Stability": {
                "gradComponentClip": float(self.stability.grad_component_clip),
                "gradNormClip": float(self.stability.grad_norm_clip),
                "maxUpdate": float(self.stability.max_update),
                "maxScale": float(self.stability.max_scale),
                "maxAnisotropy": float(max(self.stability.max_anisotropy, 1.0)),
                "minOpacity": float(self.stability.min_opacity),
                "maxOpacity": float(self.stability.max_opacity),
                "positionAbsMax": float(self.stability.position_abs_max),
                "hugeValue": float(self.stability.huge_value),
            },
        }

    def regularization_vars(self, training_hparams: Any, scale_reg_reference: float, frame_camera: Camera | None = None, step_index: int = 0, splat_contribution_buffer: spy.Buffer | None = None) -> dict[str, object]:
        camera_position = np.zeros((3,), dtype=np.float32) if frame_camera is None else np.asarray(frame_camera.position, dtype=np.float32).reshape(3)
        return {
            "g_OptimizerRegularization": {
                "scaleL2Weight": float(max(training_hparams.scale_l2_weight, 0.0)),
                "scaleAbsWeight": float(max(training_hparams.scale_abs_reg_weight, 0.0)),
                "sh1Weight": float(max(training_hparams.sh1_reg_weight, 0.0)),
                "opacityWeight": float(max(training_hparams.opacity_reg_weight, 0.0)),
                "positionPushAwayFromCameraStep": float(resolve_position_push_away_from_camera_step(training_hparams, int(step_index))),
                "scaleReference": float(max(scale_reg_reference, 1e-8)),
            },
            "g_OptimizerRegularizationCameraPosition": spy.float3(*camera_position.tolist()),
            "g_OptimizerRegularizationHasCamera": np.uint32(0 if frame_camera is None else 1),
            "g_OptimizerStoredPackedParamCount": np.uint32(int(self.renderer.packed_trainable_param_count)),
            "g_OptimizerSplatContributionInfo": self.renderer.work_buffers["training_splat_contribution"] if splat_contribution_buffer is None else splat_contribution_buffer,
        }

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
        work_buffers: dict[str, spy.Buffer],
        splat_count: int,
        training_hparams: Any,
        scale_reg_reference: float,
        frame_camera: Camera | None = None,
        width: int | None = None,
        height: int | None = None,
        step_index: int = 0,
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
        dispatch(
            kernel=self._kernels["project_params"],
            thread_count=self._threads(splat_count),
            vars={
                "g_SplatParamsRW": scene_buffers["splat_params"],
                **camera_vars,
                **self._projection_vars(
                    splat_count,
                    training_hparams,
                    int(step_index),
                    max_visible_angle_deg=resolve_max_visible_angle_deg(training_hparams, int(step_index)),
                ),
            },
            command_encoder=encoder,
            debug_label="Gaussian Param Projection",
            debug_color_index=71,
        )
