from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import slangpy as spy

from ..common import SHADER_ROOT, debug_region, thread_count_1d
from ..renderer import GaussianRenderer
from .schedule import resolve_learning_rate_scale


class GaussianOptimizer:
    _GROUPS = ((0, 3), (3, 3), (6, 4), (10, 4))
    _PARAM_SETTINGS_U32_WIDTH = 8
    _RO_BUFFER_USAGE = spy.BufferUsage.shader_resource | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
    _threads = staticmethod(thread_count_1d)

    def __init__(self, device: spy.Device, renderer: GaussianRenderer, adam_hparams: Any, stability_hparams: Any) -> None:
        self.device = device
        self.renderer = renderer
        self.adam = adam_hparams
        self.stability = stability_hparams
        self._uploaded_lr_scale = float("nan")
        self._buffers: dict[str, spy.Buffer] = {}
        self._kernels = self._create_kernels()
        self._ensure_static_buffers()
        self._upload_param_settings()

    def _create_kernels(self) -> dict[str, spy.ComputeKernel]:
        shader_path = Path(SHADER_ROOT / "utility" / "optimizer" / "gaussian_optimizer_stage.slang")
        entries = {
            "accumulate_regularizers": "csAccumulateRegularizationGrads",
            "project_params": "csProjectGaussianParams",
        }
        return {
            name: self.device.create_compute_kernel(self.device.load_program(str(shader_path), [entry]))
            for name, entry in entries.items()
        }

    def _create_buffer(self, size: int) -> spy.Buffer:
        return self.device.create_buffer(size=max(int(size), 4), usage=self._RO_BUFFER_USAGE)

    def _ensure_static_buffers(self) -> None:
        if "param_settings" in self._buffers: return
        self._buffers["param_settings"] = self._create_buffer(self.renderer.TRAINABLE_PARAM_COUNT * self._PARAM_SETTINGS_U32_WIDTH * 4)

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
        if param_id in self.renderer.PARAM_SCALE_IDS: return float(np.log(max(self.stability.min_scale, 1e-8)))
        if param_id in self.renderer.PARAM_COLOR_IDS: return 0.0
        if param_id == self.renderer.PARAM_RAW_OPACITY_ID: return self._raw_opacity_from_alpha(float(self.stability.min_opacity))
        return -float(self.stability.huge_value)

    def _value_max_for_param(self, param_id: int) -> float:
        if param_id in self.renderer.PARAM_POSITION_IDS: return float(self.stability.position_abs_max)
        if param_id in self.renderer.PARAM_SCALE_IDS: return float(np.log(max(self.stability.max_scale, 1e-8)))
        if param_id in self.renderer.PARAM_COLOR_IDS: return 1.0
        if param_id == self.renderer.PARAM_RAW_OPACITY_ID: return self._raw_opacity_from_alpha(float(self.stability.max_opacity))
        return float(self.stability.huge_value)

    def _group_for_param(self, param_id: int) -> tuple[int, int]:
        for group_start, group_size in self._GROUPS:
            if group_start <= int(param_id) < group_start + group_size: return int(group_start), int(group_size)
        return int(param_id), 1

    def _param_settings(self, lr_scale: float = 1.0) -> np.ndarray:
        settings = np.zeros((self.renderer.TRAINABLE_PARAM_COUNT, self._PARAM_SETTINGS_U32_WIDTH), dtype=np.uint32)
        scale = max(float(lr_scale), 0.0)
        for param_id in range(self.renderer.TRAINABLE_PARAM_COUNT):
            group_start, group_size = self._group_for_param(param_id)
            settings[param_id, 0] = np.asarray([self._lr_for_param(param_id) * scale], dtype=np.float32).view(np.uint32)[0]
            settings[param_id, 1] = np.asarray([float(self.stability.grad_component_clip)], dtype=np.float32).view(np.uint32)[0]
            settings[param_id, 2] = np.asarray([float(self.stability.grad_norm_clip)], dtype=np.float32).view(np.uint32)[0]
            settings[param_id, 3] = np.asarray([self._value_min_for_param(param_id)], dtype=np.float32).view(np.uint32)[0]
            settings[param_id, 4] = np.asarray([self._value_max_for_param(param_id)], dtype=np.float32).view(np.uint32)[0]
            settings[param_id, 5] = np.uint32(group_start)
            settings[param_id, 6] = np.uint32(group_size)
        return settings

    def _upload_param_settings(self, lr_scale: float = 1.0) -> None:
        self._ensure_static_buffers()
        self._buffers["param_settings"].copy_from_numpy(self._param_settings(lr_scale))
        self._uploaded_lr_scale = float(lr_scale)

    def update_hyperparams(self, adam_hparams: Any, stability_hparams: Any) -> None:
        self.adam = adam_hparams
        self.stability = stability_hparams
        self._upload_param_settings()

    def update_step(self, step_index: int, training_hparams: Any) -> None:
        lr_scale = resolve_learning_rate_scale(training_hparams, int(step_index))
        if np.isfinite(self._uploaded_lr_scale) and abs(self._uploaded_lr_scale - lr_scale) <= 1e-12:
            return
        self._upload_param_settings(lr_scale)

    def _vars(self, splat_count: int, training_hparams: Any, scale_reg_reference: float) -> dict[str, object]:
        return {
            "g_SplatCount": int(splat_count),
            "g_ScaleL2Weight": float(max(training_hparams.scale_l2_weight, 0.0)),
            "g_ScaleAbsRegWeight": float(max(training_hparams.scale_abs_reg_weight, 0.0)),
            "g_OpacityRegWeight": float(max(training_hparams.opacity_reg_weight, 0.0)),
            "g_ScaleRegReference": float(max(scale_reg_reference, 1e-8)),
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
        }

    def dispatch_regularizers(
        self,
        encoder: spy.CommandEncoder,
        *,
        scene_buffers: dict[str, spy.Buffer],
        work_buffers: dict[str, spy.Buffer],
        loss_buffer: spy.Buffer,
        splat_count: int,
        training_hparams: Any,
        scale_reg_reference: float,
    ) -> None:
        with debug_region(encoder, "Gaussian Regularizers", 70):
            self._kernels["accumulate_regularizers"].dispatch(
                thread_count=self._threads(splat_count),
                vars={
                    "g_LossBuffer": loss_buffer,
                    "g_ParamGrads": work_buffers["param_grads"],
                    "g_SplatParamsRW": scene_buffers["splat_params"],
                    **self._vars(splat_count, training_hparams, scale_reg_reference),
                },
                command_encoder=encoder,
            )

    @property
    def param_settings(self) -> spy.Buffer:
        return self._buffers["param_settings"]

    @property
    def param_settings_count(self) -> int:
        return int(self.renderer.TRAINABLE_PARAM_COUNT)

    def dispatch_projection(
        self,
        encoder: spy.CommandEncoder,
        *,
        scene_buffers: dict[str, spy.Buffer],
        work_buffers: dict[str, spy.Buffer],
        splat_count: int,
        training_hparams: Any,
        scale_reg_reference: float,
    ) -> None:
        with debug_region(encoder, "Gaussian Param Projection", 71):
            self._kernels["project_params"].dispatch(
                thread_count=self._threads(splat_count),
                vars={
                    "g_ParamGrads": work_buffers["param_grads"],
                    "g_SplatParamsRW": scene_buffers["splat_params"],
                    **self._vars(splat_count, training_hparams, scale_reg_reference),
                },
                command_encoder=encoder,
            )
