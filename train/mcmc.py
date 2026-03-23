from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from module import SplattingContext, render_gaussian_splats

from .dataset import CameraSample, SceneData, _image_to_linear_float, _load_image_tensor
from .losses import get_expon_lr_func, inverse_sigmoid, psnr, rgb_to_nchw, ssim, training_loss

_PARAM_COUNT = 14
_N_MAX = 51
_DEFAULT_INIT_SCALE_SPACING_RATIO = 0.25
_DEFAULT_INIT_OPACITY = 0.5
_MIN_SCALE = 1e-4
_SANITIZE_MIN_OPACITY = 1e-6
_POSITION_LIMIT_MULTIPLIER = 4.0
_SCALE_LIMIT_MULTIPLIER = 0.5
_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


def _build_rotation(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), dim=1),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), dim=1),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), dim=1),
        ),
        dim=1,
    )


def _build_scaling_rotation(scale: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    return _build_rotation(quat) @ torch.diag_embed(scale)


def _quat_rotate_vec(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    xyz = quat[:, 1:4]
    twice_cross = 2.0 * torch.cross(xyz, vec, dim=1)
    return vec + quat[:, :1] * twice_cross + torch.cross(xyz, twice_cross, dim=1)


def _quat_unrotate_vec(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    inv = quat.clone()
    inv[:, 1:4].neg_()
    return _quat_rotate_vec(inv, vec)


def _fused_densify_impl(
    xyz: torch.Tensor,
    color: torch.Tensor,
    opacity_logits: torch.Tensor,
    log_scale: torch.Tensor,
    rotation: torch.Tensor,
    binoms: torch.Tensor,
    dead: torch.Tensor,
    relocate_sampled: torch.Tensor,
    relocate_ratio: torch.Tensor,
    add_sampled: torch.Tensor,
    add_ratio: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out_xyz = xyz
    out_color = color
    out_opacity = opacity_logits
    out_log_scale = log_scale
    out_rotation = rotation
    zero = torch.empty((0,), device=xyz.device, dtype=torch.int64)

    if relocate_sampled.numel() > 0 and dead.numel() > 0:
        relocate_opacity, relocate_scale = _torch_relocation(
            torch.sigmoid(out_opacity[relocate_sampled, 0]),
            torch.exp(out_log_scale[relocate_sampled]),
            relocate_ratio[:, 0] + 1,
            binoms,
        )
        relocate_opacity_logits = inverse_sigmoid(relocate_opacity).unsqueeze(1)
        relocate_log_scale = torch.log(relocate_scale.clamp_min(1e-4))
        out_xyz = out_xyz.clone()
        out_color = out_color.clone()
        out_opacity = out_opacity.clone()
        out_log_scale = out_log_scale.clone()
        out_rotation = out_rotation.clone()
        out_xyz[dead] = out_xyz[relocate_sampled]
        out_color[dead] = out_color[relocate_sampled]
        out_opacity[dead] = relocate_opacity_logits
        out_log_scale[dead] = relocate_log_scale
        out_rotation[dead] = out_rotation[relocate_sampled]
        out_opacity[relocate_sampled] = relocate_opacity_logits
        out_log_scale[relocate_sampled] = relocate_log_scale
        zero = torch.unique(torch.cat((zero, dead.to(dtype=zero.dtype), relocate_sampled.to(dtype=zero.dtype))))

    if add_sampled.numel() > 0:
        add_opacity, add_scale = _torch_relocation(
            torch.sigmoid(out_opacity[add_sampled, 0]),
            torch.exp(out_log_scale[add_sampled]),
            add_ratio[:, 0] + 1,
            binoms,
        )
        add_opacity_logits = inverse_sigmoid(add_opacity).unsqueeze(1)
        add_log_scale = torch.log(add_scale.clamp_min(1e-4))
        current = out_xyz.shape[0]
        out_xyz = torch.cat((out_xyz, out_xyz[add_sampled]), dim=0)
        out_color = torch.cat((out_color, out_color[add_sampled]), dim=0)
        out_opacity = torch.cat((out_opacity, add_opacity_logits), dim=0)
        out_log_scale = torch.cat((out_log_scale, add_log_scale), dim=0)
        out_rotation = torch.cat((out_rotation, out_rotation[add_sampled]), dim=0)
        out_opacity = out_opacity.clone()
        out_log_scale = out_log_scale.clone()
        out_opacity[add_sampled] = add_opacity_logits
        out_log_scale[add_sampled] = add_log_scale
        new_ids = torch.arange(current, out_xyz.shape[0], device=xyz.device, dtype=zero.dtype)
        zero = torch.unique(torch.cat((zero, add_sampled.to(dtype=zero.dtype), new_ids)))

    return out_xyz, out_color, out_opacity, out_log_scale, out_rotation, zero


def _fused_sanitize_noise_impl(
    xyz: torch.Tensor,
    color: torch.Tensor,
    opacity_logits: torch.Tensor,
    log_scale: torch.Tensor,
    rotation: torch.Tensor,
    prev_xyz: torch.Tensor,
    prev_color: torch.Tensor,
    prev_opacity: torch.Tensor,
    prev_log_scale: torch.Tensor,
    prev_rotation: torch.Tensor,
    scene_extent: torch.Tensor,
    max_anisotropy: torch.Tensor,
    xyz_lr: torch.Tensor,
    noise_lr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_position = (scene_extent * _POSITION_LIMIT_MULTIPLIER).clamp_min(1.0)
    max_scale = (scene_extent * _SCALE_LIMIT_MULTIPLIER).clamp_min(0.05)
    max_anisotropy = max_anisotropy.clamp_min(1.0)
    identity = rotation.new_tensor(_IDENTITY_QUAT).expand(rotation.shape[0], -1)

    invalid_xyz = (~torch.isfinite(xyz)).sum()
    invalid_scale = (~torch.isfinite(log_scale)).sum()
    invalid_rot = (~torch.isfinite(rotation)).sum()
    invalid_opacity = (~torch.isfinite(opacity_logits)).sum()

    safe_xyz = torch.where(torch.isfinite(xyz), xyz, prev_xyz)
    safe_color = torch.where(torch.isfinite(color), color, prev_color).clamp_(0.0, 1.0)
    safe_opacity_logits = torch.where(torch.isfinite(opacity_logits), opacity_logits, prev_opacity)
    safe_opacity = torch.sigmoid(safe_opacity_logits).clamp_(_SANITIZE_MIN_OPACITY, 1.0 - torch.finfo(safe_opacity_logits.dtype).eps)
    safe_log_scale = torch.where(torch.isfinite(log_scale), log_scale, prev_log_scale)
    safe_scale = torch.exp(safe_log_scale).clamp_min_(_MIN_SCALE)
    safe_scale = torch.minimum(safe_scale, max_scale)
    safe_scale = torch.maximum(safe_scale, safe_scale.max(dim=1, keepdim=True).values / max_anisotropy)
    safe_rotation = torch.where(torch.isfinite(rotation), rotation, prev_rotation)
    rot_norm = torch.linalg.vector_norm(safe_rotation, dim=1, keepdim=True)
    prev_rot_norm = torch.linalg.vector_norm(prev_rotation, dim=1, keepdim=True)
    safe_rotation = torch.where(
        rot_norm > 1e-8,
        safe_rotation / rot_norm.clamp_min(1e-8),
        torch.where(prev_rot_norm > 1e-8, prev_rotation / prev_rot_norm.clamp_min(1e-8), identity),
    )

    clamped_scale = (safe_scale >= max_scale).sum()
    clamped_pos = (safe_xyz.abs() >= max_position).sum()
    safe_xyz = safe_xyz.clamp(-max_position, max_position)

    sigma = torch.sigmoid(100.0 * (0.005 - safe_opacity))
    base_noise = torch.randn_like(safe_xyz) * sigma * noise_lr * xyz_lr
    local_noise = _quat_unrotate_vec(safe_rotation, base_noise) * safe_scale.square()
    noise = _quat_rotate_vec(safe_rotation, local_noise)
    safe_xyz = safe_xyz + torch.nan_to_num(noise, nan=0.0, posinf=0.0, neginf=0.0)

    return (
        safe_xyz,
        safe_color,
        inverse_sigmoid(safe_opacity),
        torch.log(safe_scale),
        safe_rotation,
        invalid_xyz,
        invalid_scale,
        invalid_rot,
        invalid_opacity,
        clamped_scale,
        clamped_pos,
    )


def _estimate_initial_log_scales(points: torch.Tensor, spacing_ratio: float, chunk: int = 1024, reference_limit: int = 4096, progress: Callable[[int, int], None] | None = None) -> torch.Tensor:
    count, nearest = int(points.shape[0]), torch.empty((int(points.shape[0]),), device=points.device, dtype=points.dtype)
    ref_count = min(count, max(int(reference_limit), 1))
    refs, exact = (points, True) if ref_count >= count else (points[torch.randperm(count, device=points.device)[:ref_count]], False)
    for start in range(0, count, chunk):
        end, dist = min(start + chunk, count), torch.cdist(points[start : min(start + chunk, count)], refs)
        if progress is not None:
            progress(start, count)
        if exact:
            dist[torch.arange(end - start, device=points.device), torch.arange(start, end, device=points.device)] = float("inf")
            nearest[start:end] = dist.min(dim=1).values.clamp_min_(1e-6)
        elif ref_count == 1:
            nearest[start:end] = dist[:, 0].clamp_min_(1e-6)
        else:
            best = torch.topk(dist, k=2, dim=1, largest=False).values
            nearest[start:end] = torch.where(best[:, 0] > 1e-8, best[:, 0], best[:, 1]).clamp_min_(1e-6)
    if progress is not None:
        progress(count, count)
    return torch.log(nearest.mul(float(spacing_ratio)).unsqueeze(1).repeat(1, 3))


def _torch_relocation(opacity_old: torch.Tensor, scale_old: torch.Tensor, n: torch.Tensor, binoms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = n.clamp_(1, _N_MAX - 1)
    opacity_new, denom = 1.0 - (1.0 - opacity_old).pow(1.0 / n.to(dtype=opacity_old.dtype)), torch.zeros_like(opacity_old)
    for i in range(1, _N_MAX):
        active = n >= i
        if bool(active.any()):
            cur, acc = opacity_new[active], torch.zeros_like(opacity_new[active])
            for k in range(i):
                acc += float(binoms[i - 1, k]) * (((-1.0) ** k) / math.sqrt(k + 1.0)) * cur.pow(k + 1)
            denom[active] += acc
    denom = torch.nan_to_num(denom, nan=1.0, posinf=1.0, neginf=1.0).clamp_min_(torch.finfo(denom.dtype).eps)
    opacity_new = torch.nan_to_num(opacity_new, nan=0.005, posinf=1.0, neginf=0.005).clamp_(0.005, 1.0 - torch.finfo(opacity_new.dtype).eps)
    scale_new = torch.nan_to_num((opacity_old / denom).unsqueeze(1) * scale_old, nan=1e-4, posinf=1e4, neginf=1e-4).clamp_min_(1e-4)
    return opacity_new, scale_new


def _expand_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(ref)
    slices = tuple(slice(0, min(a, b)) for a, b in zip(src.shape, ref.shape))
    out[slices] = src[slices]
    return out


@dataclass
class MCMCConfig:
    iterations: int = 30000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    lambda_dssim: float = 0.2
    noise_lr: float = 5e5
    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    densification_interval: int = 100
    densify_from_iter: int = 500
    densify_until_iter: int = 25000
    cap_max: int = 500000
    random_background: bool = False
    init_points: int = 50000
    init_scale_spacing_ratio: float = _DEFAULT_INIT_SCALE_SPACING_RATIO
    init_scale_multiplier: float = 1.0
    init_opacity: float = _DEFAULT_INIT_OPACITY
    eval_interval: int = 250
    seed: int = 0


@dataclass
class TrainingMetrics:
    iteration: int
    loss: float
    l1: float
    test_psnr: float | None
    test_ssim: float | None
    point_count: int


@dataclass
class ScalarStats:
    mean_abs: float
    max_abs: float
    l2: float


@dataclass
class SplatStats:
    opacity_min: float
    opacity_mean: float
    opacity_max: float
    scale_min: float
    scale_mean: float
    scale_max: float
    anisotropy_mean: float
    anisotropy_max: float
    position_min: tuple[float, float, float]
    position_max: tuple[float, float, float]


@dataclass
class TrainingStepStats(TrainingMetrics):
    xyz_lr: float
    elapsed_ms: float
    frame_index: int
    camera_name: str
    background: tuple[float, float, float]
    mse: float
    train_psnr: float
    grad_stats: dict[str, ScalarStats]
    splat_stats: SplatStats


class RGBGaussianModel:
    _ATTRS = {"xyz": "_xyz", "color": "_color", "opacity": "_opacity", "scaling": "_log_scale", "rotation": "_rotation"}

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._xyz = nn.Parameter(torch.empty((0, 3), device=device))
        self._color = nn.Parameter(torch.empty((0, 3), device=device))
        self._log_scale = nn.Parameter(torch.empty((0, 3), device=device))
        self._rotation = nn.Parameter(torch.empty((0, 4), device=device))
        self._opacity = nn.Parameter(torch.empty((0, 1), device=device))
        self.optimizer: torch.optim.Adam | None = None
        self.xyz_scheduler = None
        self._binoms = torch.tensor([[math.comb(n, k) if k <= n else 0 for k in range(_N_MAX)] for n in range(_N_MAX)], device=device, dtype=torch.float32)

    @property
    def scaling(self) -> torch.Tensor:
        return torch.exp(self._log_scale)

    @property
    def rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self._rotation, dim=1)

    @property
    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    @property
    def count(self) -> int:
        return int(self._xyz.shape[0])

    def create_from_scene(self, scene: SceneData, init_points: int, init_opacity: float, init_scale_spacing_ratio: float, status: Callable[[str], None] | None = None) -> None:
        if status is not None:
            status("Uploading point cloud")
        xyz = torch.as_tensor(scene.point_xyz, dtype=torch.float32, device=self.device)
        color = torch.as_tensor(scene.point_rgb / 255.0, dtype=torch.float32, device=self.device)
        if 0 < init_points < int(xyz.shape[0]):
            if status is not None:
                status(f"Sampling initial points ({init_points:,})")
            keep = torch.randperm(int(xyz.shape[0]), device=self.device)[:init_points]
            xyz, color = xyz[keep], color[keep]
        if status is not None:
            status(f"Estimating initial scales (0/{xyz.shape[0]:,})")
        log_scale = _estimate_initial_log_scales(xyz, init_scale_spacing_ratio, progress=None if status is None else lambda done, total: status(f"Estimating initial scales ({done:,}/{total:,})"))
        rotation = torch.zeros((int(xyz.shape[0]), 4), device=self.device, dtype=torch.float32)
        rotation[:, 0] = 1.0
        self._xyz = nn.Parameter(xyz)
        self._color = nn.Parameter(color)
        self._log_scale = nn.Parameter(log_scale)
        self._rotation = nn.Parameter(rotation)
        self._opacity = nn.Parameter(inverse_sigmoid(torch.full((int(xyz.shape[0]), 1), float(init_opacity), device=self.device)))

    def scale_init(self, multiplier: float) -> None:
        if multiplier != 1.0:
            self._log_scale.data.add_(math.log(multiplier))

    def setup_training(self, cfg: MCMCConfig, spatial_lr_scale: float) -> None:
        self.optimizer = torch.optim.Adam(
            [
                {"params": [self._xyz], "lr": cfg.position_lr_init * spatial_lr_scale, "name": "xyz"},
                {"params": [self._color], "lr": cfg.feature_lr, "name": "color"},
                {"params": [self._opacity], "lr": cfg.opacity_lr, "name": "opacity"},
                {"params": [self._log_scale], "lr": cfg.scaling_lr, "name": "scaling"},
                {"params": [self._rotation], "lr": cfg.rotation_lr, "name": "rotation"},
            ],
            lr=0.0,
            eps=1e-15,
        )
        self.xyz_scheduler = get_expon_lr_func(cfg.position_lr_init * spatial_lr_scale, cfg.position_lr_final * spatial_lr_scale, lr_delay_mult=cfg.position_lr_delay_mult, max_steps=cfg.position_lr_max_steps)

    def update_learning_rate(self, iteration: int) -> float:
        lr = float(self.xyz_scheduler(iteration))
        next(group for group in self.optimizer.param_groups if group["name"] == "xyz")["lr"] = lr
        return lr

    def splats(self) -> torch.Tensor:
        splats = torch.empty((_PARAM_COUNT, self.count), device=self.device, dtype=torch.float32)
        splats[:3], splats[3:6], splats[6:10], splats[10:13], splats[13] = self._xyz.mT, self._log_scale.mT, self.rotation.mT, self._color.mT, self.opacity[:, 0]
        return splats

    def _replace(self, name: str, tensor: torch.Tensor, zero: torch.Tensor, preserve_grad: bool) -> None:
        old = getattr(self, self._ATTRS[name])
        group = next(group for group in self.optimizer.param_groups if group["name"] == name)
        state = self.optimizer.state.pop(old)
        for key in ("exp_avg", "exp_avg_sq"):
            state[key] = _expand_like(state[key], tensor)
            state[key][zero] = 0
        param = nn.Parameter(tensor)
        if preserve_grad and old.grad is not None:
            param.grad = _expand_like(old.grad, tensor)
            param.grad[zero] = 0
        group["params"][0] = param
        self.optimizer.state[param] = state
        setattr(self, self._ATTRS[name], param)

    def _replace_all(self, xyz: torch.Tensor, color: torch.Tensor, opacity: torch.Tensor, log_scale: torch.Tensor, rotation: torch.Tensor, zero: torch.Tensor, preserve_grad: bool) -> None:
        for name, tensor in (("xyz", xyz), ("color", color), ("opacity", opacity), ("scaling", log_scale), ("rotation", rotation)):
            self._replace(name, tensor, zero, preserve_grad)

    def _sample_alive(self, probs: torch.Tensor, num: int, alive: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.nan_to_num(probs.detach(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)
        total = probs.sum()
        if not torch.isfinite(total) or float(total) <= 0.0:
            probs = torch.ones_like(probs)
            total = probs.sum()
        sampled = torch.multinomial(probs / total, num, replacement=True)
        if alive is not None:
            sampled = alive[sampled]
        return sampled, torch.bincount(sampled, minlength=self.count).unsqueeze(1)

    def _updated_params(self, idxs: torch.Tensor, ratio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        opacity, scale = _torch_relocation(self.opacity[idxs, 0], self.scaling[idxs], ratio[idxs, 0] + 1, self._binoms)
        return self._xyz[idxs], self._color[idxs], inverse_sigmoid(opacity).unsqueeze(1), torch.log(scale.clamp_min(1e-4)), self._rotation[idxs]

    def relocate_gs(self, dead_mask: torch.Tensor, preserve_grad: bool = False) -> None:
        if not bool(dead_mask.any()):
            return
        dead, alive = dead_mask.nonzero(as_tuple=True)[0], (~dead_mask).nonzero(as_tuple=True)[0]
        if alive.numel() == 0:
            return
        sampled, ratio = self._sample_alive(self.opacity[alive, 0], int(dead.numel()), alive)
        xyz, color, opacity, log_scale, rotation = [tensor.detach().clone() for tensor in (self._xyz, self._color, self._opacity, self._log_scale, self._rotation)]
        new_xyz, new_color, new_opacity, new_log_scale, new_rotation = self._updated_params(sampled, ratio)
        xyz[dead], color[dead], opacity[dead], log_scale[dead], rotation[dead] = new_xyz, new_color, new_opacity, new_log_scale, new_rotation
        opacity[sampled], log_scale[sampled] = new_opacity, new_log_scale
        self._replace_all(xyz, color, opacity, log_scale, rotation, torch.unique(torch.cat((dead, sampled))), preserve_grad)

    def add_new_gs(self, cap_max: int, preserve_grad: bool = False) -> int:
        current, num_new = self.count, max(0, min(cap_max, int(1.05 * self.count)) - self.count)
        if num_new <= 0:
            return 0
        sampled, ratio = self._sample_alive(self.opacity[:, 0], num_new)
        new_xyz, new_color, new_opacity, new_log_scale, new_rotation = self._updated_params(sampled, ratio)
        xyz, color = torch.cat((self._xyz.detach(), new_xyz), dim=0), torch.cat((self._color.detach(), new_color), dim=0)
        opacity, log_scale = torch.cat((self._opacity.detach(), new_opacity), dim=0), torch.cat((self._log_scale.detach(), new_log_scale), dim=0)
        rotation = torch.cat((self._rotation.detach(), new_rotation), dim=0)
        opacity[sampled], log_scale[sampled] = new_opacity, new_log_scale
        new_ids = torch.arange(current, int(xyz.shape[0]), device=xyz.device, dtype=sampled.dtype)
        self._replace_all(xyz, color, opacity, log_scale, rotation, torch.unique(torch.cat((sampled, new_ids))), preserve_grad)
        return num_new

    def densify_gs(self, dead_mask: torch.Tensor, cap_max: int, preserve_grad: bool = False) -> int:
        dead = dead_mask.nonzero(as_tuple=True)[0]
        relocate_sampled = torch.empty((0,), device=self.device, dtype=torch.long)
        relocate_ratio = torch.empty((0, 1), device=self.device, dtype=self._opacity.dtype)
        densify_probs = self.opacity[:, 0].detach()

        if dead.numel() > 0:
            alive = (~dead_mask).nonzero(as_tuple=True)[0]
            if alive.numel() > 0:
                relocate_sampled, relocate_ratio = self._sample_alive(self.opacity[alive, 0], int(dead.numel()), alive)
                relocate_opacity, _ = _torch_relocation(
                    self.opacity[relocate_sampled, 0],
                    self.scaling[relocate_sampled],
                    relocate_ratio[relocate_sampled, 0] + 1,
                    self._binoms,
                )
                densify_probs = densify_probs.clone()
                densify_probs[dead] = relocate_opacity
                densify_probs[relocate_sampled] = relocate_opacity

        current = self.count
        num_new = max(0, min(cap_max, int(1.05 * current)) - current)
        add_sampled = torch.empty((0,), device=self.device, dtype=torch.long)
        add_ratio = torch.empty((0, 1), device=self.device, dtype=self._opacity.dtype)
        if num_new > 0:
            add_sampled, add_ratio = self._sample_alive(densify_probs, num_new)
        if relocate_sampled.numel() == 0 and add_sampled.numel() == 0:
            return 0

        xyz, color, opacity, log_scale, rotation, zero = _fused_densify_impl(
            self._xyz.detach(),
            self._color.detach(),
            self._opacity.detach(),
            self._log_scale.detach(),
            self._rotation.detach(),
            self._binoms,
            dead,
            relocate_sampled,
            relocate_ratio[relocate_sampled],
            add_sampled,
            add_ratio[add_sampled],
        )
        self._replace_all(xyz, color, opacity, log_scale, rotation, zero, preserve_grad)
        return int(add_sampled.numel())

    def add_noise(self, xyz_lr: float, noise_lr: float) -> None:
        rotation = self.rotation
        scale_sq = self.scaling.square()
        sigma = torch.sigmoid(100.0 * (0.005 - self.opacity))
        base_noise = torch.randn_like(self._xyz) * sigma * noise_lr * xyz_lr
        local_noise = _quat_unrotate_vec(rotation, base_noise) * scale_sq
        noise = _quat_rotate_vec(rotation, local_noise)
        self._xyz.data.add_(torch.nan_to_num(noise, nan=0.0, posinf=0.0, neginf=0.0))

    def sanitize_after_step(self, prev: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], scene_extent: float, max_anisotropy: float) -> str:
        prev_xyz, prev_color, prev_opacity, prev_log_scale, prev_rotation = prev
        max_position = max(float(scene_extent) * _POSITION_LIMIT_MULTIPLIER, 1.0)
        max_scale = max(float(scene_extent) * _SCALE_LIMIT_MULTIPLIER, 0.05)
        max_anisotropy = max(float(max_anisotropy), 1.0)
        identity = self._rotation.new_tensor(_IDENTITY_QUAT).expand(self._rotation.shape[0], -1)

        with torch.no_grad():
            invalid_xyz = int((~torch.isfinite(self._xyz)).sum().item())
            invalid_scale = int((~torch.isfinite(self._log_scale)).sum().item())
            invalid_rot = int((~torch.isfinite(self._rotation)).sum().item())
            invalid_opacity = int((~torch.isfinite(self._opacity)).sum().item())
            xyz = torch.where(torch.isfinite(self._xyz), self._xyz, prev_xyz)
            color = torch.where(torch.isfinite(self._color), self._color, prev_color).clamp_(0.0, 1.0)
            opacity_logits = torch.where(torch.isfinite(self._opacity), self._opacity, prev_opacity)
            opacity = torch.sigmoid(opacity_logits).clamp_(_SANITIZE_MIN_OPACITY, 1.0 - torch.finfo(opacity_logits.dtype).eps)
            log_scale = torch.where(torch.isfinite(self._log_scale), self._log_scale, prev_log_scale)
            scale = torch.exp(log_scale).clamp_(_MIN_SCALE, max_scale)
            scale = torch.maximum(scale, scale.max(dim=1, keepdim=True).values / max_anisotropy)
            rotation = torch.where(torch.isfinite(self._rotation), self._rotation, prev_rotation)
            rot_norm = torch.linalg.vector_norm(rotation, dim=1, keepdim=True)
            prev_rot_norm = torch.linalg.vector_norm(prev_rotation, dim=1, keepdim=True)
            rotation = torch.where(rot_norm > 1e-8, rotation / rot_norm.clamp_min(1e-8), torch.where(prev_rot_norm > 1e-8, prev_rotation / prev_rot_norm.clamp_min(1e-8), identity))
            clamped_scale = int((scale >= max_scale).sum().item())
            clamped_pos = int((xyz.abs() >= max_position).sum().item())
            xyz.clamp_(-max_position, max_position)
            self._xyz.data.copy_(xyz)
            self._color.data.copy_(color)
            self._opacity.data.copy_(inverse_sigmoid(opacity))
            self._log_scale.data.copy_(torch.log(scale))
            self._rotation.data.copy_(rotation)

        if invalid_xyz or invalid_scale or invalid_rot or invalid_opacity or clamped_scale or clamped_pos:
            return (
                f"sanitized params: xyz={invalid_xyz} scale={invalid_scale} rot={invalid_rot} opacity={invalid_opacity} "
                f"clamped_scale={clamped_scale} clamped_pos={clamped_pos}"
            )
        return ""

    def sanitize_and_add_noise(
        self,
        prev: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        scene_extent: float,
        max_anisotropy: float,
        xyz_lr: float,
        noise_lr: float,
    ) -> str:
        prev_xyz, prev_color, prev_opacity, prev_log_scale, prev_rotation = prev
        scene_extent_tensor = torch.as_tensor(scene_extent, device=self.device, dtype=self._xyz.dtype)
        max_anisotropy_tensor = torch.as_tensor(max_anisotropy, device=self.device, dtype=self._xyz.dtype)
        xyz_lr_tensor = torch.as_tensor(xyz_lr, device=self.device, dtype=self._xyz.dtype)
        noise_lr_tensor = torch.as_tensor(noise_lr, device=self.device, dtype=self._xyz.dtype)
        (
            xyz,
            color,
            opacity,
            log_scale,
            rotation,
            invalid_xyz,
            invalid_scale,
            invalid_rot,
            invalid_opacity,
            clamped_scale,
            clamped_pos,
        ) = _fused_sanitize_noise_impl(
            self._xyz.detach(),
            self._color.detach(),
            self._opacity.detach(),
            self._log_scale.detach(),
            self._rotation.detach(),
            prev_xyz,
            prev_color,
            prev_opacity,
            prev_log_scale,
            prev_rotation,
            scene_extent_tensor,
            max_anisotropy_tensor,
            xyz_lr_tensor,
            noise_lr_tensor,
        )

        self._xyz.data.copy_(xyz)
        self._color.data.copy_(color)
        self._opacity.data.copy_(opacity)
        self._log_scale.data.copy_(log_scale)
        self._rotation.data.copy_(rotation)

        invalid_xyz_i = int(invalid_xyz.item())
        invalid_scale_i = int(invalid_scale.item())
        invalid_rot_i = int(invalid_rot.item())
        invalid_opacity_i = int(invalid_opacity.item())
        clamped_scale_i = int(clamped_scale.item())
        clamped_pos_i = int(clamped_pos.item())
        if invalid_xyz_i or invalid_scale_i or invalid_rot_i or invalid_opacity_i or clamped_scale_i or clamped_pos_i:
            return (
                f"sanitized params: xyz={invalid_xyz_i} scale={invalid_scale_i} rot={invalid_rot_i} opacity={invalid_opacity_i} "
                f"clamped_scale={clamped_scale_i} clamped_pos={clamped_pos_i}"
            )
        return ""


class RGBMCMCTrainer:
    def __init__(self, cfg: MCMCConfig, context: SplattingContext | None = None, device: str = "cuda") -> None:
        self.cfg, self.device = cfg, torch.device(device)
        self.context, self.model = context or SplattingContext(), RGBGaussianModel(self.device)
        self.scene: SceneData | None = None
        self.iteration, self._train_stack, self._image_cache = 0, [], {}
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    def initialize(self, scene: SceneData, status: Callable[[str], None] | None = None) -> None:
        self.scene, self.iteration, self._train_stack, self._image_cache = scene, 0, [], {}
        self.model.create_from_scene(scene, self.cfg.init_points, self.cfg.init_opacity, self.cfg.init_scale_spacing_ratio, status=status)
        if status is not None:
            status("Applying initial scale multiplier")
        self.model.scale_init(self.cfg.init_scale_multiplier)
        if status is not None:
            status("Creating optimizer")
        self.model.setup_training(self.cfg, scene.extent_radius)

    def start(self, scene: SceneData, status: Callable[[str], None] | None = None) -> None:
        self.initialize(scene, status=status)

    def has_pending_steps(self) -> bool:
        return self.scene is not None

    def snapshot_splats(self) -> torch.Tensor:
        return self.model.splats().detach().contiguous()

    def _render(self, camera: CameraSample, background: tuple[float, float, float]) -> torch.Tensor:
        return render_gaussian_splats(
            self.model.splats(),
            camera.camera_params,
            camera.image_size,
            background=background,
            render_seed=self.iteration,
            context=self.context,
        )[..., :3]

    def _target_image(self, camera: CameraSample) -> torch.Tensor:
        def to_target(image: torch.Tensor) -> torch.Tensor:
            if not image.is_cuda:
                image = image.to(self.device)
            return _image_to_linear_float(image) if image.dtype == torch.uint8 else image

        if camera.image is not None:
            return to_target(camera.image)
        if camera.image_path not in self._image_cache:
            self._image_cache[camera.image_path] = _load_image_tensor(camera.image_path, self.device, preload_cuda=True)
        return to_target(self._image_cache[camera.image_path])

    def _grad_stats(self) -> dict[str, ScalarStats]:
        def stat(grad: torch.Tensor | None) -> ScalarStats:
            return ScalarStats(0.0, 0.0, 0.0) if grad is None or grad.numel() == 0 else ScalarStats(float(grad.abs().mean()), float(grad.abs().max()), float(torch.linalg.vector_norm(grad)))

        return {
    name: stat(grad) for name, grad in (("xyz", self.model._xyz.grad), ("color", self.model._color.grad), ("opacity", self.model._opacity.grad), ("scaling", self.model._log_scale.grad), ("rotation", self.model._rotation.grad))
    }

    def _splat_stats(self) -> SplatStats:
        xyz, scale, opacity = self.model._xyz.detach(), self.model.scaling.detach(), self.model.opacity[:, 0].detach()
        anisotropy = scale.max(dim=1).values / scale.min(dim=1).values
        lo, hi = xyz.min(dim=0).values, xyz.max(dim=0).values
        return SplatStats(float(opacity.min()), float(opacity.mean()), float(opacity.max()), float(scale.min()), float(scale.mean()), float(scale.max()), float(anisotropy.mean()), float(anisotropy.max()), tuple(map(float, lo)), tuple(map(float, hi)))

    def evaluate(self, cameras: list[CameraSample], background: tuple[float, float, float]) -> tuple[float, float]:
        if not cameras:
            return float("nan"), float("nan")
        with torch.no_grad():
            pred_target = [(self._render(cam, background), self._target_image(cam)) for cam in cameras]
        return float(np.mean([float(psnr(pred, target)) for pred, target in pred_target])), float(np.mean([float(ssim(rgb_to_nchw(pred), rgb_to_nchw(target))) for pred, target in pred_target]))

    def step(self, scene: SceneData | None = None) -> TrainingStepStats:
        if scene is not None and self.scene is None:
            self.start(scene)
        if self.scene is None:
            raise StopIteration("Trainer has not been initialized with a scene.")
        start = time.perf_counter()
        self.iteration += 1
        if not self._train_stack:
            self._train_stack = self.scene.train_cameras.copy()
            random.shuffle(self._train_stack)
        camera, xyz_lr = self._train_stack.pop(), self.model.update_learning_rate(self.iteration)
        frame_index = next((idx for idx, item in enumerate(self.scene.train_cameras) if item is camera), -1)
        background = tuple(np.random.rand(3).tolist()) if self.cfg.random_background else self.scene.background
        pred, target = self._render(camera, background), self._target_image(camera)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        opacity = self.model.opacity
        scaling = self.model.scaling
        total, l1, mse, train_psnr_tensor = training_loss(pred, target, opacity, scaling, self.cfg.lambda_dssim, self.cfg.opacity_reg, self.cfg.scale_reg)
        total = total.clone()
        l1 = l1.clone()
        mse = mse.clone()
        train_psnr_tensor = train_psnr_tensor.clone()
        mse_value = float(mse.detach())
        train_psnr = float(train_psnr_tensor.detach())
        total.backward()
        grad_stats = self._grad_stats()
        prev_params = tuple(t.detach().clone() for t in (self.model._xyz, self.model._color, self.model._opacity, self.model._log_scale, self.model._rotation))
        if self.cfg.densify_from_iter < self.iteration < self.cfg.densify_until_iter and self.iteration % self.cfg.densification_interval == 0:
            self.model.densify_gs((self.model.opacity <= 0.005).squeeze(1), self.cfg.cap_max, preserve_grad=True)
            prev_params = tuple(t.detach().clone() for t in (self.model._xyz, self.model._color, self.model._opacity, self.model._log_scale, self.model._rotation))
        self.model.optimizer.step()
        self.model.optimizer.zero_grad(set_to_none=True)
        self.model.sanitize_and_add_noise(prev_params, self.scene.extent_radius, self.context.max_anisotropy, xyz_lr, self.cfg.noise_lr)
        test_psnr = test_ssim = None
        if self.iteration % self.cfg.eval_interval == 0:
            test_psnr, test_ssim = self.evaluate(self.scene.test_cameras, self.scene.background)
        return TrainingStepStats(
            self.iteration,
            float(total.detach()),
            float(l1.detach()),
            test_psnr,
            test_ssim,
            self.model.count,
            float(xyz_lr),
            (time.perf_counter() - start) * 1000.0,
            int(frame_index),
            camera.image_name,
            tuple(map(float, background)),
            float(mse_value),
            float(train_psnr),
            grad_stats,
            self._splat_stats(),
        )

    def train(self, scene: SceneData, output_dir: str | Path | None = None) -> list[TrainingMetrics]:
        if self.scene is not scene or self.iteration != 0 or self.model.optimizer is None:
            self.start(scene)
        output_dir = None if output_dir is None else Path(output_dir)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2))
        metrics = []
        for _ in tqdm(range(1, self.cfg.iterations + 1), desc="MCMC training"):
            step = self.step()
            metrics.append(TrainingMetrics(step.iteration, step.loss, step.l1, step.test_psnr, step.test_ssim, step.point_count))
        if output_dir is not None:
            (output_dir / "metrics.json").write_text(json.dumps([asdict(item) for item in metrics], indent=2))
            np.save(output_dir / "final_splats.npy", self.model.splats().detach().cpu().numpy())
        return metrics
