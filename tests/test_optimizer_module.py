from __future__ import annotations

from pathlib import Path

import numpy as np
import slangpy as spy

_SETTINGS_U32_WIDTH = 8


def test_generic_packed_adam_converges_on_quadratic(device):
    grad_shader_path = Path(__file__).with_name("optimizer_test_stage.slang")
    adam_shader_path = Path("shaders/utility/optimizer/optimizer.slang")
    grad_kernel = device.create_compute_kernel(device.load_program(str(grad_shader_path), ["csComputeQuadraticGrad"]))
    adam_kernel = device.create_compute_kernel(device.load_program(str(adam_shader_path), ["csAdamStepPacked"]))
    param_count = 32
    rng = np.random.default_rng(7)
    targets = rng.normal(0.0, 0.5, size=(param_count,)).astype(np.float32)
    params_init = (targets + rng.normal(0.0, 1.25, size=(param_count,))).astype(np.float32)
    lrs = np.full((param_count,), 0.05, dtype=np.float32)
    zeros = np.zeros((param_count,), dtype=np.float32)
    usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
    params = device.create_buffer(size=param_count * 4, usage=usage)
    grads = device.create_buffer(size=param_count * 4, usage=usage)
    adam_moments = device.create_buffer(size=param_count * 8, usage=usage)
    targets_buf = device.create_buffer(size=param_count * 4, usage=usage)
    settings_buf = device.create_buffer(size=param_count * _SETTINGS_U32_WIDTH * 4, usage=usage)
    params.copy_from_numpy(params_init)
    grads.copy_from_numpy(zeros)
    adam_moments.copy_from_numpy(np.zeros((param_count, 2), dtype=np.float32))
    targets_buf.copy_from_numpy(targets)
    settings = np.zeros((param_count, _SETTINGS_U32_WIDTH), dtype=np.uint32)
    settings[:, 0] = lrs.view(np.uint32)
    settings[:, 1] = np.full((param_count,), 1e6, dtype=np.float32).view(np.uint32)
    settings[:, 2] = np.full((param_count,), 1e6, dtype=np.float32).view(np.uint32)
    settings[:, 3] = np.full((param_count,), -1e6, dtype=np.float32).view(np.uint32)
    settings[:, 4] = np.full((param_count,), 1e6, dtype=np.float32).view(np.uint32)
    settings[:, 5] = np.arange(param_count, dtype=np.uint32)
    settings[:, 6] = 1
    settings_buf.copy_from_numpy(settings)

    common_vars = {
        "g_ParamCount": int(param_count),
        "g_Targets": targets_buf,
        "g_Params": params,
        "g_Grads": grads,
        "g_Stability": {
            "gradComponentClip": 1e6,
            "gradNormClip": 1e6,
            "maxUpdate": 1.0,
            "minScale": 0.0,
            "maxScale": 0.0,
            "maxAnisotropy": 1.0,
            "minOpacity": 0.0,
            "maxOpacity": 0.0,
            "positionAbsMax": 1e6,
            "hugeValue": 1e6,
        },
        "g_OptimizerParamCount": int(param_count),
        "g_OptimizerParamGroupSize": 1,
        "g_OptimizerParamSettingsCount": int(param_count),
        "g_OptimizerParamSettings": settings_buf,
        "g_OptimizerParams": params,
        "g_OptimizerGrads": grads,
        "g_OptimizerAdamMoments": adam_moments,
        "g_OptimizerStability": {
            "gradComponentClip": 1e6,
            "gradNormClip": 1e6,
            "maxUpdate": 1.0,
            "minScale": 0.0,
            "maxScale": 0.0,
            "maxAnisotropy": 1.0,
            "minOpacity": 0.0,
            "maxOpacity": 0.0,
            "positionAbsMax": 1e6,
            "hugeValue": 1e6,
        },
    }
    thread_count = spy.uint3(param_count, 1, 1)
    start_error = float(np.mean(np.abs(params_init - targets), dtype=np.float64))
    for step in range(1, 201):
        enc = device.create_command_encoder()
        grad_kernel.dispatch(
            thread_count=thread_count,
            vars={**common_vars, "g_Adam": {"beta1": 0.9, "beta2": 0.999, "stepIndex": int(step)}},
            command_encoder=enc,
        )
        adam_kernel.dispatch(
            thread_count=thread_count,
            vars={**common_vars, "g_OptimizerAdam": {"beta1": 0.9, "beta2": 0.999, "stepIndex": int(step)}},
            command_encoder=enc,
        )
        device.submit_command_buffer(enc.finish())
    device.wait()

    final_params = np.frombuffer(params.to_numpy().tobytes(), dtype=np.float32)[:param_count].copy()
    final_error = float(np.mean(np.abs(final_params - targets), dtype=np.float64))

    assert np.all(np.isfinite(final_params))
    assert final_error < 1e-3
    assert final_error < start_error * 1e-3
