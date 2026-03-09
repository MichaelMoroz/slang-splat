from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import slangpy as spy

from ..common import SHADER_ROOT
from ..renderer import GaussianRenderer


class GaussianOptimizer:
    _GROUPS = ((0, 3), (3, 3), (6, 4), (10, 4))
    _PARAM_SETTINGS_U32_WIDTH = 8
    _RW_BUFFER_USAGE = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
    _RO_BUFFER_USAGE = spy.BufferUsage.shader_resource | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
    _clip_threads = staticmethod(lambda count: spy.uint3(int(count), 1, 1))
    _param_threads = staticmethod(lambda count, param_count: spy.uint3(int(count) * int(param_count), 1, 1))

    def __init__(self, device: spy.Device, renderer: GaussianRenderer, adam_hparams: Any, stability_hparams: Any) -> None:
        self.device = device
        self.renderer = renderer
        self.adam = adam_hparams
        self.stability = stability_hparams
        self._capacity = 0
        self._buffers: dict[str, spy.Buffer] = {}
        self._kernels = self._create_kernels()
        self._ensure_static_buffers()
        self._upload_param_settings()

    def _create_kernels(self) -> dict[str, spy.ComputeKernel]:
        entries = {
            "accumulate_regularizers": (Path(SHADER_ROOT / "utility" / "optimizer" / "gaussian_optimizer_stage.slang"), "csAccumulateRegularizationGrads"),
            "clip_grads": (Path(SHADER_ROOT / "utility" / "optimizer" / "optimizer.slang"), "csClipPackedParamGrads"),
            "adam_step": (Path(SHADER_ROOT / "utility" / "optimizer" / "optimizer.slang"), "csAdamStepPacked"),
            "project_params": (Path(SHADER_ROOT / "utility" / "optimizer" / "gaussian_optimizer_stage.slang"), "csProjectGaussianParams"),
        }
        return {
            name: self.device.create_compute_kernel(self.device.load_program(str(shader_path), [entry]))
            for name, (shader_path, entry) in entries.items()
        }

    def _create_buffer(self, size: int, usage: spy.BufferUsage) -> spy.Buffer:
        return self.device.create_buffer(size=max(int(size), 4), usage=usage)

    def _ensure_static_buffers(self) -> None:
        if "param_settings" not in self._buffers:
            self._buffers["param_settings"] = self._create_buffer(self.renderer.TRAINABLE_PARAM_COUNT * self._PARAM_SETTINGS_U32_WIDTH * 4, self._RO_BUFFER_USAGE)

    def _ensure_state_buffers(self, splat_count: int) -> None:
        count = max(int(splat_count), 1)
        if self._capacity >= count and "adam_moments" in self._buffers:
            return
        self._capacity = max(count, max(self._capacity, 1) + max(self._capacity, 1) // 2)
        self._buffers["adam_moments"] = self._create_buffer(self._capacity * self.renderer.TRAINABLE_PARAM_COUNT * 8, self._RW_BUFFER_USAGE)

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
        if param_id in self.renderer.PARAM_SCALE_IDS: return float(self.stability.min_scale)
        if param_id in self.renderer.PARAM_COLOR_IDS: return 0.0
        if param_id == self.renderer.PARAM_RAW_OPACITY_ID: return self._raw_opacity_from_alpha(float(self.stability.min_opacity))
        return -float(self.stability.huge_value)

    def _value_max_for_param(self, param_id: int) -> float:
        if param_id in self.renderer.PARAM_POSITION_IDS: return float(self.stability.position_abs_max)
        if param_id in self.renderer.PARAM_SCALE_IDS: return float(self.stability.max_scale)
        if param_id in self.renderer.PARAM_COLOR_IDS: return 1.0
        if param_id == self.renderer.PARAM_RAW_OPACITY_ID: return self._raw_opacity_from_alpha(float(self.stability.max_opacity))
        return float(self.stability.huge_value)

    def _group_for_param(self, param_id: int) -> tuple[int, int]:
        for group_start, group_size in self._GROUPS:
            if group_start <= int(param_id) < group_start + group_size:
                return int(group_start), int(group_size)
        return int(param_id), 1

    def _param_settings(self) -> np.ndarray:
        settings = np.zeros((self.renderer.TRAINABLE_PARAM_COUNT, self._PARAM_SETTINGS_U32_WIDTH), dtype=np.uint32)
        for param_id in range(self.renderer.TRAINABLE_PARAM_COUNT):
            group_start, group_size = self._group_for_param(param_id)
            settings[param_id, 0] = np.asarray([self._lr_for_param(param_id)], dtype=np.float32).view(np.uint32)[0]
            settings[param_id, 1] = np.asarray([float(self.stability.grad_component_clip)], dtype=np.float32).view(np.uint32)[0]
            settings[param_id, 2] = np.asarray([float(self.stability.grad_norm_clip)], dtype=np.float32).view(np.uint32)[0]
            settings[param_id, 3] = np.asarray([self._value_min_for_param(param_id)], dtype=np.float32).view(np.uint32)[0]
            settings[param_id, 4] = np.asarray([self._value_max_for_param(param_id)], dtype=np.float32).view(np.uint32)[0]
            settings[param_id, 5] = np.uint32(group_start)
            settings[param_id, 6] = np.uint32(group_size)
        return settings

    def _upload_param_settings(self) -> None:
        self._ensure_static_buffers()
        self._buffers["param_settings"].copy_from_numpy(self._param_settings())

    def update_hyperparams(self, adam_hparams: Any, stability_hparams: Any) -> None:
        self.adam = adam_hparams
        self.stability = stability_hparams
        self._upload_param_settings()

    def zero_moments(self, splat_count: int) -> None:
        self._ensure_state_buffers(splat_count)
        zeros = np.zeros((self._capacity * self.renderer.TRAINABLE_PARAM_COUNT, 2), dtype=np.float32)
        self._buffers["adam_moments"].copy_from_numpy(zeros)

    def _buffer_shader_vars(self) -> dict[str, object]:
        return {
            "g_OptimizerParamSettings": self._buffers["param_settings"],
            "g_OptimizerAdamMoments": self._buffers["adam_moments"],
        }

    @staticmethod
    def _packed_shader_vars(scene_buffers: dict[str, spy.Buffer], work_buffers: dict[str, spy.Buffer]) -> dict[str, object]:
        return {"g_OptimizerParams": scene_buffers["splat_params"], "g_OptimizerGrads": work_buffers["param_grads"]}

    def _gaussian_optimizer_vars(self, splat_count: int, training_hparams: Any, scale_reg_reference: float) -> dict[str, object]:
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

    def _optimizer_module_vars(self, splat_count: int, step_index: int) -> dict[str, object]:
        return {
            "g_OptimizerAdam": {
                "beta1": float(self.adam.beta1),
                "beta2": float(self.adam.beta2),
                "stepIndex": int(step_index),
            },
            "g_OptimizerStability": {
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
            "g_OptimizerSplatCount": int(splat_count),
            "g_OptimizerParamCount": int(splat_count * self.renderer.TRAINABLE_PARAM_COUNT),
            "g_OptimizerParamGroupSize": int(splat_count),
            "g_OptimizerParamSettingsCount": int(self.renderer.TRAINABLE_PARAM_COUNT),
        }

    def dispatch_step(
        self,
        encoder: spy.CommandEncoder,
        *,
        scene_buffers: dict[str, spy.Buffer],
        work_buffers: dict[str, spy.Buffer],
        loss_buffer: spy.Buffer,
        splat_count: int,
        training_hparams: Any,
        scale_reg_reference: float,
        step_index: int,
    ) -> None:
        count = max(int(splat_count), 1)
        self._ensure_state_buffers(count)
        shared_buffers = self._buffer_shader_vars()
        optimizer_vars = self._optimizer_module_vars(count, step_index)
        gaussian_vars = self._gaussian_optimizer_vars(count, training_hparams, scale_reg_reference)
        packed_vars = self._packed_shader_vars(scene_buffers, work_buffers)
        gaussian_common = {
            "g_LossBuffer": loss_buffer,
            "g_ParamGrads": work_buffers["param_grads"],
            "g_SplatParamsRW": scene_buffers["splat_params"],
            **gaussian_vars,
        }
        self._kernels["accumulate_regularizers"].dispatch(thread_count=self._clip_threads(count), vars=gaussian_common, command_encoder=encoder)
        self._kernels["clip_grads"].dispatch(thread_count=self._clip_threads(count), vars={**packed_vars, **shared_buffers, **optimizer_vars}, command_encoder=encoder)
        self._kernels["adam_step"].dispatch(thread_count=self._param_threads(count, self.renderer.TRAINABLE_PARAM_COUNT), vars={**packed_vars, **shared_buffers, **optimizer_vars}, command_encoder=encoder)
        self._kernels["project_params"].dispatch(thread_count=self._clip_threads(count), vars=gaussian_common, command_encoder=encoder)

    @property
    def buffers(self) -> dict[str, spy.Buffer]:
        return self._buffers
