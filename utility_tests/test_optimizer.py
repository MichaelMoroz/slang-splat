from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utility.optimizer import AdamHyperParams, GenericAdamOptimizer, ParamGroupConfig


def test_generic_adam_converges_on_quadratic() -> None:
    target = torch.linspace(-1.0, 1.0, 32, dtype=torch.float32)
    param = torch.nn.Parameter(target + 2.0)
    optimizer = GenericAdamOptimizer([(param, ParamGroupConfig(name="p", lr=0.05))], hyperparams=AdamHyperParams(eps=1e-8))
    start_error = float((param.detach() - target).abs().mean())
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = 0.5 * (param - target).square().mean()
        loss.backward()
        optimizer.step()
    final_error = float((param.detach() - target).abs().mean())
    assert torch.all(torch.isfinite(param))
    assert final_error < 1e-3
    assert final_error < start_error * 1e-3


def test_generic_adam_uses_per_group_learning_rates() -> None:
    slow = torch.nn.Parameter(torch.tensor([[1.0]], dtype=torch.float32))
    fast = torch.nn.Parameter(torch.tensor([[1.0]], dtype=torch.float32))
    optimizer = GenericAdamOptimizer(
        [
            (slow, ParamGroupConfig(name="slow", lr=0.01)),
            (fast, ParamGroupConfig(name="fast", lr=0.1)),
        ]
    )
    slow.grad = torch.ones_like(slow)
    fast.grad = torch.ones_like(fast)
    optimizer.step()
    assert abs(float(fast.item()) - 1.0) > abs(float(slow.item()) - 1.0)


@pytest.mark.parametrize("width", [3, 4])
def test_generic_adam_clips_grouped_grad_norms(width: int) -> None:
    param = torch.nn.Parameter(torch.zeros((2, width), dtype=torch.float32))
    optimizer = GenericAdamOptimizer([(param, ParamGroupConfig(name="g", lr=1.0, grad_norm_clip=1.0))], hyperparams=AdamHyperParams(eps=1e-8))
    param.grad = torch.full_like(param, 4.0)
    clipped = optimizer._transform_grad(param.grad, param, optimizer.param_groups[0])
    norms = torch.linalg.vector_norm(clipped, dim=-1)
    assert torch.all(norms <= 1.0 + 1e-6)


def test_generic_adam_sanitizes_nan_and_inf_grads() -> None:
    param = torch.nn.Parameter(torch.tensor([[1.0, -1.0]], dtype=torch.float32))
    optimizer = GenericAdamOptimizer([(param, ParamGroupConfig(name="g", lr=0.1, grad_component_clip=1.0))])
    param.grad = torch.tensor([[float("nan"), float("inf")]], dtype=torch.float32)
    optimizer.step()
    assert torch.all(torch.isfinite(param))
