from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch


GradTransform = Callable[[torch.Tensor, torch.nn.Parameter], torch.Tensor]
ParamProject = Callable[[torch.nn.Parameter], None]


@dataclass(slots=True)
class AdamHyperParams:
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-15


@dataclass(slots=True)
class ParamGroupConfig:
    name: str
    lr: float
    grad_component_clip: float = float("inf")
    grad_norm_clip: float = float("inf")
    max_update: float = float("inf")
    grad_transform: GradTransform | None = None
    project_param: ParamProject | None = None


class GenericAdamOptimizer:
    def __init__(self, params: list[tuple[torch.nn.Parameter, ParamGroupConfig]], hyperparams: AdamHyperParams | None = None) -> None:
        self.hyperparams = AdamHyperParams() if hyperparams is None else hyperparams
        self.param_groups: list[dict[str, object]] = []
        self.state: dict[torch.nn.Parameter, dict[str, object]] = {}
        self._step_index = 0
        for param, config in params:
            group = {
                "params": [param],
                "name": config.name,
                "lr": float(config.lr),
                "grad_component_clip": float(config.grad_component_clip),
                "grad_norm_clip": float(config.grad_norm_clip),
                "max_update": float(config.max_update),
                "grad_transform": config.grad_transform,
                "project_param": config.project_param,
            }
            self.param_groups.append(group)
            self.state[param] = self._new_state(param)

    @staticmethod
    def _new_state(param: torch.nn.Parameter) -> dict[str, object]:
        return {
            "step": 0,
            "exp_avg": torch.zeros_like(param),
            "exp_avg_sq": torch.zeros_like(param),
        }

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            param = group["params"][0]
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.zero_()

    def set_group_lr(self, name: str, lr: float) -> None:
        next(group for group in self.param_groups if group["name"] == name)["lr"] = float(lr)

    def step(self) -> None:
        self._step_index += 1
        beta1 = float(self.hyperparams.beta1)
        beta2 = float(self.hyperparams.beta2)
        eps = float(self.hyperparams.eps)
        with torch.no_grad():
            for group in self.param_groups:
                param = group["params"][0]
                grad = param.grad
                if grad is None:
                    if callable(group["project_param"]):
                        group["project_param"](param)
                    continue
                grad = self._transform_grad(grad, param, group)
                state = self.state.setdefault(param, self._new_state(param))
                state["step"] = int(state.get("step", 0)) + 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1 ** int(state["step"])
                bias_correction2 = 1.0 - beta2 ** int(state["step"])
                m_hat = exp_avg / max(bias_correction1, torch.finfo(exp_avg.dtype).tiny)
                v_hat = exp_avg_sq / max(bias_correction2, torch.finfo(exp_avg_sq.dtype).tiny)
                update = float(group["lr"]) * m_hat / (torch.sqrt(torch.clamp_min(v_hat, 0.0)) + eps)
                max_update = float(group["max_update"])
                if math.isfinite(max_update) and max_update > 0.0:
                    update = update.clamp(min=-max_update, max=max_update)
                param.add_(-torch.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0))
                if callable(group["project_param"]):
                    group["project_param"](param)

    def _transform_grad(self, grad: torch.Tensor, param: torch.nn.Parameter, group: dict[str, object]) -> torch.Tensor:
        transformed = torch.nan_to_num(grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        if callable(group["grad_transform"]):
            transformed = group["grad_transform"](transformed, param)
        component_clip = float(group["grad_component_clip"])
        if math.isfinite(component_clip) and component_clip > 0.0:
            transformed = transformed.clamp(min=-component_clip, max=component_clip)
        norm_clip = float(group["grad_norm_clip"])
        if math.isfinite(norm_clip) and norm_clip > 0.0 and transformed.ndim >= 2 and transformed.shape[-1] > 1:
            norms = torch.linalg.vector_norm(transformed, dim=-1, keepdim=True)
            scale = torch.clamp(norm_clip / norms.clamp_min(torch.finfo(transformed.dtype).tiny), max=1.0)
            transformed = transformed * scale
        return torch.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)

    def remap_parameter(self, name: str, new_param: torch.nn.Parameter, old_count: int, pre_remove_count: int, keep_indices: torch.Tensor) -> None:
        group = next(group for group in self.param_groups if group["name"] == name)
        old_param = group["params"][0]
        old_state = self.state.pop(old_param, self._new_state(old_param))
        new_state: dict[str, object] = {}
        keep_indices = keep_indices.to(device=new_param.device, dtype=torch.long)
        for key, value in old_state.items():
            if torch.is_tensor(value) and tuple(value.shape) == tuple(old_param.shape):
                expanded = value.new_zeros((pre_remove_count, *value.shape[1:]))
                expanded[:old_count].copy_(value)
                new_state[key] = expanded.index_select(0, keep_indices.to(device=value.device)).contiguous()
            elif torch.is_tensor(value):
                new_state[key] = value.clone()
            else:
                new_state[key] = value
        group["params"] = [new_param]
        self.state[new_param] = new_state
