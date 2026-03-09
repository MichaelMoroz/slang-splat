from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import slangpy as spy

from ..common import SHADER_ROOT
from ..renderer import GaussianRenderer


class GaussianOptimizer:
    _GROUPS = ((0, 3), (3, 3), (6, 4), (10, 4))
    _ADAM_BUFFER_NAMES = ("adam_m", "adam_v")
    _TABLE_BUFFER_NAMES = (
        "param_lrs",
        "param_grad_clip_abs",
        "param_grad_norm_clip",
        "param_value_min",
        "param_value_max",
        "param_group_starts",
        "param_group_sizes",
    )
    _TABLE_SHADER_VARS = {
        "param_lrs": "g_OptimizerParamLRs",
        "param_grad_clip_abs": "g_OptimizerParamGradClipAbs",
        "param_grad_norm_clip": "g_OptimizerParamGradNormClip",
        "param_value_min": "g_OptimizerParamValueMin",
        "param_value_max": "g_OptimizerParamValueMax",
        "param_group_starts": "g_OptimizerParamGroupStarts",
        "param_group_sizes": "g_OptimizerParamGroupSizes",
    }
    _ADAM_SHADER_VARS = {"adam_m": "g_OptimizerAdamM", "adam_v": "g_OptimizerAdamV"}
    _PACKED_SHADER_VARS = {"g_OptimizerParams": "splat_params", "g_OptimizerGrads": "param_grads"}
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
        self._upload_tables()

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
        if all(name in self._buffers for name in self._TABLE_BUFFER_NAMES):
            return
        param_count = self.renderer.TRAINABLE_PARAM_COUNT
        self._buffers["param_lrs"] = self._create_buffer(param_count * 4, self._RO_BUFFER_USAGE)
        self._buffers["param_grad_clip_abs"] = self._create_buffer(param_count * 4, self._RO_BUFFER_USAGE)
        self._buffers["param_grad_norm_clip"] = self._create_buffer(param_count * 4, self._RO_BUFFER_USAGE)
        self._buffers["param_value_min"] = self._create_buffer(param_count * 4, self._RO_BUFFER_USAGE)
        self._buffers["param_value_max"] = self._create_buffer(param_count * 4, self._RO_BUFFER_USAGE)
        self._buffers["param_group_starts"] = self._create_buffer(len(self._GROUPS) * 4, self._RO_BUFFER_USAGE)
        self._buffers["param_group_sizes"] = self._create_buffer(len(self._GROUPS) * 4, self._RO_BUFFER_USAGE)

    def _ensure_state_buffers(self, splat_count: int) -> None:
        count = max(int(splat_count), 1)
        if self._capacity >= count and all(name in self._buffers for name in self._ADAM_BUFFER_NAMES):
            return
        self._capacity = max(count, max(self._capacity, 1) + max(self._capacity, 1) // 2)
        buffer_size = self._capacity * self.renderer.TRAINABLE_PARAM_COUNT * 4
        for name in self._ADAM_BUFFER_NAMES:
            self._buffers[name] = self._create_buffer(buffer_size, self._RW_BUFFER_USAGE)

    @staticmethod
    def _raw_opacity_from_alpha(alpha: float) -> float:
        alpha_clamped = float(np.clip(alpha, 1e-6, 1.0 - 1e-6))
        return float(np.log(alpha_clamped) - np.log1p(-alpha_clamped))

    def _param_lrs(self) -> np.ndarray:
        lrs = np.zeros((self.renderer.TRAINABLE_PARAM_COUNT,), dtype=np.float32)
        for param_id in self.renderer.PARAM_POSITION_IDS:
            lrs[param_id] = float(self.adam.position_lr)
        for param_id in self.renderer.PARAM_SCALE_IDS:
            lrs[param_id] = float(self.adam.scale_lr)
        for param_id in self.renderer.PARAM_ROTATION_IDS:
            lrs[param_id] = float(self.adam.rotation_lr)
        for param_id in self.renderer.PARAM_COLOR_IDS:
            lrs[param_id] = float(self.adam.color_lr)
        lrs[self.renderer.PARAM_RAW_OPACITY_ID] = float(self.adam.opacity_lr)
        return lrs

    def _param_grad_clip_abs(self) -> np.ndarray:
        return np.full((self.renderer.TRAINABLE_PARAM_COUNT,), float(self.stability.grad_component_clip), dtype=np.float32)

    def _param_grad_norm_clip(self) -> np.ndarray:
        return np.full((self.renderer.TRAINABLE_PARAM_COUNT,), float(self.stability.grad_norm_clip), dtype=np.float32)

    def _param_value_min(self) -> np.ndarray:
        mins = np.full((self.renderer.TRAINABLE_PARAM_COUNT,), -float(self.stability.huge_value), dtype=np.float32)
        for param_id in self.renderer.PARAM_POSITION_IDS:
            mins[param_id] = -float(self.stability.position_abs_max)
        for param_id in self.renderer.PARAM_SCALE_IDS:
            mins[param_id] = float(self.stability.min_scale)
        for param_id in self.renderer.PARAM_COLOR_IDS:
            mins[param_id] = 0.0
        mins[self.renderer.PARAM_RAW_OPACITY_ID] = self._raw_opacity_from_alpha(float(self.stability.min_opacity))
        return mins

    def _param_value_max(self) -> np.ndarray:
        maxs = np.full((self.renderer.TRAINABLE_PARAM_COUNT,), float(self.stability.huge_value), dtype=np.float32)
        for param_id in self.renderer.PARAM_POSITION_IDS:
            maxs[param_id] = float(self.stability.position_abs_max)
        for param_id in self.renderer.PARAM_SCALE_IDS:
            maxs[param_id] = float(self.stability.max_scale)
        for param_id in self.renderer.PARAM_COLOR_IDS:
            maxs[param_id] = 1.0
        maxs[self.renderer.PARAM_RAW_OPACITY_ID] = self._raw_opacity_from_alpha(float(self.stability.max_opacity))
        return maxs

    def _param_group_starts(self) -> np.ndarray:
        return np.asarray([group[0] for group in self._GROUPS], dtype=np.uint32)

    def _param_group_sizes(self) -> np.ndarray:
        return np.asarray([group[1] for group in self._GROUPS], dtype=np.uint32)

    def _upload_tables(self) -> None:
        self._ensure_static_buffers()
        self._buffers["param_lrs"].copy_from_numpy(self._param_lrs())
        self._buffers["param_grad_clip_abs"].copy_from_numpy(self._param_grad_clip_abs())
        self._buffers["param_grad_norm_clip"].copy_from_numpy(self._param_grad_norm_clip())
        self._buffers["param_value_min"].copy_from_numpy(self._param_value_min())
        self._buffers["param_value_max"].copy_from_numpy(self._param_value_max())
        self._buffers["param_group_starts"].copy_from_numpy(self._param_group_starts())
        self._buffers["param_group_sizes"].copy_from_numpy(self._param_group_sizes())

    def update_hyperparams(self, adam_hparams: Any, stability_hparams: Any) -> None:
        self.adam = adam_hparams
        self.stability = stability_hparams
        self._upload_tables()

    def zero_moments(self, splat_count: int) -> None:
        self._ensure_state_buffers(splat_count)
        zeros = np.zeros((max(int(splat_count), 1) * self.renderer.TRAINABLE_PARAM_COUNT,), dtype=np.float32)
        for name in self._ADAM_BUFFER_NAMES:
            self._buffers[name].copy_from_numpy(np.pad(zeros, (0, max(self._capacity * self.renderer.TRAINABLE_PARAM_COUNT - zeros.size, 0))))

    def _buffer_shader_vars(self) -> dict[str, object]:
        table_vars = {shader_name: self._buffers[name] for name, shader_name in self._TABLE_SHADER_VARS.items()}
        adam_vars = {shader_name: self._buffers[name] for name, shader_name in self._ADAM_SHADER_VARS.items()}
        return {**table_vars, **adam_vars}

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
            "g_OptimizerParamGroupCount": int(len(self._GROUPS)),
            "g_OptimizerParamGroupSize": int(splat_count),
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
        self._kernels["project_params"].dispatch(thread_count=self._clip_threads(count), vars={**gaussian_common}, command_encoder=encoder)

    @property
    def buffers(self) -> dict[str, spy.Buffer]:
        return self._buffers
