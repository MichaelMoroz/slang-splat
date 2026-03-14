from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import slangpy as spy

from ..common import SHADER_ROOT, debug_region, thread_count_1d


@dataclass(slots=True)
class AdamRuntimeHyperParams:
    grad_component_clip: float = 10.0
    grad_norm_clip: float = 10.0
    max_update: float = 0.05
    huge_value: float = 1e8


class AdamOptimizer:
    _RW_BUFFER_USAGE = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination

    _clip_threads = staticmethod(thread_count_1d)
    _grad_norm_threads = staticmethod(thread_count_1d)
    _param_threads = staticmethod(thread_count_1d)

    def __init__(self, device: spy.Device, adam_hparams: Any, runtime_hparams: AdamRuntimeHyperParams) -> None:
        self.device = device
        self.adam = adam_hparams
        self.runtime = runtime_hparams
        self._capacity = 0
        self._buffers: dict[str, spy.Buffer] = {}
        self._kernels = self._create_kernels()

    def _create_kernels(self) -> dict[str, spy.ComputeKernel]:
        shader_path = Path(SHADER_ROOT / "utility" / "optimizer" / "optimizer.slang")
        entries = {
            "compute_grad_norms": "csComputePackedElementGradNorms",
            "clip_grads": "csClipPackedParamGrads",
            "adam_step": "csAdamStepPacked",
        }
        return {
            name: self.device.create_compute_kernel(self.device.load_program(str(shader_path), [entry]))
            for name, entry in entries.items()
        }

    def _create_buffer(self, size: int) -> spy.Buffer:
        return self.device.create_buffer(size=max(int(size), 4), usage=self._RW_BUFFER_USAGE)

    def _ensure_state_buffers(self, packed_param_count: int) -> None:
        count = max(int(packed_param_count), 1)
        if self._capacity >= count and "adam_moments" in self._buffers: return
        self._capacity = max(count, max(self._capacity, 1) + max(self._capacity, 1) // 2)
        self._buffers["adam_moments"] = self._create_buffer(self._capacity * 8)

    def update_hyperparams(self, adam_hparams: Any, runtime_hparams: AdamRuntimeHyperParams) -> None:
        self.adam = adam_hparams
        self.runtime = runtime_hparams

    def zero_moments(self, packed_param_count: int) -> None:
        self._ensure_state_buffers(packed_param_count)
        self._buffers["adam_moments"].copy_from_numpy(np.zeros((self._capacity, 2), dtype=np.float32))

    def _buffer_shader_vars(self) -> dict[str, object]:
        return {"g_OptimizerAdamMoments": self._buffers["adam_moments"]}

    @staticmethod
    def _packed_shader_vars(params_buffer: spy.Buffer, grads_buffer: spy.Buffer, param_settings: spy.Buffer) -> dict[str, object]:
        return {
            "g_OptimizerParams": params_buffer,
            "g_OptimizerGrads": grads_buffer,
            "g_OptimizerParamSettings": param_settings,
        }

    def _optimizer_vars(self, element_count: int, packed_param_count: int, param_group_size: int, param_settings_count: int, step_index: int) -> dict[str, object]:
        return {
            "g_OptimizerAdam": {
                "beta1": float(self.adam.beta1),
                "beta2": float(self.adam.beta2),
                "stepIndex": int(step_index),
            },
            "g_OptimizerRuntime": {
                "gradComponentClip": float(self.runtime.grad_component_clip),
                "gradNormClip": float(self.runtime.grad_norm_clip),
                "maxUpdate": float(self.runtime.max_update),
                "hugeValue": float(self.runtime.huge_value),
            },
            "g_OptimizerElementCount": int(element_count),
            "g_OptimizerParamCount": int(packed_param_count),
            "g_OptimizerParamGroupSize": int(param_group_size),
            "g_OptimizerParamSettingsCount": int(param_settings_count),
        }

    def dispatch_step(
        self,
        encoder: spy.CommandEncoder,
        *,
        params_buffer: spy.Buffer,
        grads_buffer: spy.Buffer,
        element_count: int,
        packed_param_count: int,
        param_group_size: int,
        param_settings: spy.Buffer,
        param_settings_count: int,
        step_index: int,
        debug_element_grad_norm_buffer: spy.Buffer | None = None,
    ) -> None:
        count = max(int(packed_param_count), 1)
        self._ensure_state_buffers(count)
        vars = {
            **self._packed_shader_vars(params_buffer, grads_buffer, param_settings),
            **self._buffer_shader_vars(),
            **self._optimizer_vars(element_count, count, param_group_size, param_settings_count, step_index),
        }
        with debug_region(encoder, "Adam Clip Grads", 60):
            self._kernels["clip_grads"].dispatch(thread_count=self._clip_threads(element_count), vars=vars, command_encoder=encoder)
        if debug_element_grad_norm_buffer is not None:
            with debug_region(encoder, "Adam Compute Grad Norms", 61):
                self._kernels["compute_grad_norms"].dispatch(
                    thread_count=self._grad_norm_threads(element_count),
                    vars={**vars, "g_OptimizerElementGradNorms": debug_element_grad_norm_buffer},
                    command_encoder=encoder,
                )
        with debug_region(encoder, "Adam Step", 62):
            self._kernels["adam_step"].dispatch(thread_count=self._param_threads(count), vars=vars, command_encoder=encoder)

    @property
    def buffers(self) -> dict[str, spy.Buffer]:
        return self._buffers
