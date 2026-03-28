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
from .losses import get_expon_lr_func, inverse_sigmoid, l1_loss, psnr, rgb_to_nchw, ssim, training_loss

_PARAM_COUNT = 14
_DEFAULT_INIT_SCALE_SPACING_RATIO = 0.25
_DEFAULT_INIT_OPACITY = 0.5
_MIN_SCALE = 1e-4
_SANITIZE_MIN_OPACITY = 1e-6
_POSITION_LIMIT_MULTIPLIER = 64.0
_SCALE_LIMIT_MULTIPLIER = 0.5
_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


def _image_hwc(image: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    width, height = map(int, image_size)
    if tuple(image.shape[:2]) == (height, width):
        return image
    if tuple(image.shape[:2]) == (width, height):
        return image.permute(1, 0, 2).contiguous()
    raise ValueError(f"Unexpected image shape {tuple(image.shape)} for image_size={image_size}.")


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


@dataclass
class MCMCConfig:
    iterations: int = 30000
    base_lr_init: float = 0.001
    base_lr_final: float = 0.001
    base_lr_delay_mult: float = 1.0
    base_lr_max_steps: int = 30000
    position_lr_mult: float = 0.1
    feature_lr_mult: float = 1.0
    opacity_lr_mult: float = 1.0
    scaling_lr_mult: float = 20.0
    rotation_lr_mult: float = 1.0
    lambda_dssim: float = 0.2
    depth_ratio_weight: float = 0.025
    noise_lr: float = 5e5
    densify_enabled: bool = True
    densify_interval: int = 10
    densify_until_iter: int = 500
    densify_interval_after: int = 50
    densify_target_ratio: float = 0.05
    densify_append_multiplier: float = 2.0
    densify_clone_opacity: float = 0.33
    remove_opacity_threshold: float = 0.001
    max_splats: int = 2_000_000
    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    random_background: bool = False
    init_points: int = 50000
    init_scale_spacing_ratio: float = _DEFAULT_INIT_SCALE_SPACING_RATIO
    init_scale_multiplier: float = 1.0
    init_opacity: float = _DEFAULT_INIT_OPACITY
    eval_interval: int = 250
    dither_strength: float = 1.0
    dither_decay_until_iter: int = 0
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
    tile_count: int
    mse: float
    train_psnr: float
    grad_stats: dict[str, ScalarStats]
    splat_stats: SplatStats


class RGBGaussianModel:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._xyz = nn.Parameter(torch.empty((0, 3), device=device))
        self._color = nn.Parameter(torch.empty((0, 3), device=device))
        self._log_scale = nn.Parameter(torch.empty((0, 3), device=device))
        self._rotation = nn.Parameter(torch.empty((0, 4), device=device))
        self._opacity = nn.Parameter(torch.empty((0, 1), device=device))
        self.optimizer: torch.optim.Adam | None = None
        self.base_lr_scheduler = None
        self.spatial_lr_scale = 1.0
        self.position_lr_mult = 1.0
        self.feature_lr_mult = 1.0
        self.opacity_lr_mult = 1.0
        self.scaling_lr_mult = 1.0
        self.rotation_lr_mult = 1.0

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
        self.spatial_lr_scale = float(spatial_lr_scale)
        self.position_lr_mult = float(cfg.position_lr_mult)
        self.feature_lr_mult = float(cfg.feature_lr_mult)
        self.opacity_lr_mult = float(cfg.opacity_lr_mult)
        self.scaling_lr_mult = float(cfg.scaling_lr_mult)
        self.rotation_lr_mult = float(cfg.rotation_lr_mult)
        self.optimizer = torch.optim.Adam(
            [
                {"params": [self._xyz], "lr": cfg.base_lr_init * self.position_lr_mult * spatial_lr_scale, "name": "xyz"},
                {"params": [self._color], "lr": cfg.base_lr_init * self.feature_lr_mult, "name": "color"},
                {"params": [self._opacity], "lr": cfg.base_lr_init * self.opacity_lr_mult, "name": "opacity"},
                {"params": [self._log_scale], "lr": cfg.base_lr_init * self.scaling_lr_mult, "name": "scaling"},
                {"params": [self._rotation], "lr": cfg.base_lr_init * self.rotation_lr_mult, "name": "rotation"},
            ],
            lr=0.0,
            eps=1e-15,
        )
        self.base_lr_scheduler = get_expon_lr_func(
            cfg.base_lr_init,
            cfg.base_lr_final,
            lr_delay_mult=cfg.base_lr_delay_mult,
            max_steps=cfg.base_lr_max_steps,
        )

    def update_learning_rate(self, iteration: int) -> float:
        base_lr = float(self.base_lr_scheduler(iteration))
        next(group for group in self.optimizer.param_groups if group["name"] == "xyz")["lr"] = base_lr * self.position_lr_mult * self.spatial_lr_scale
        next(group for group in self.optimizer.param_groups if group["name"] == "color")["lr"] = base_lr * self.feature_lr_mult
        next(group for group in self.optimizer.param_groups if group["name"] == "opacity")["lr"] = base_lr * self.opacity_lr_mult
        next(group for group in self.optimizer.param_groups if group["name"] == "scaling")["lr"] = base_lr * self.scaling_lr_mult
        next(group for group in self.optimizer.param_groups if group["name"] == "rotation")["lr"] = base_lr * self.rotation_lr_mult
        return base_lr * self.position_lr_mult * self.spatial_lr_scale

    def splats(self) -> torch.Tensor:
        splats = torch.empty((_PARAM_COUNT, self.count), device=self.device, dtype=torch.float32)
        splats[:3], splats[3:6], splats[6:10], splats[10:13], splats[13] = self._xyz.mT, self._log_scale.mT, self.rotation.mT, self._color.mT, self.opacity[:, 0]
        return splats

    def add_noise(self, xyz_lr: float, noise_lr: float) -> None:
        cov = _build_scaling_rotation(self.scaling, self.rotation)
        sigma = torch.sigmoid(100.0 * (0.005 - self.opacity))
        noise = torch.bmm(cov @ cov.transpose(1, 2), (torch.randn_like(self._xyz) * sigma * noise_lr * xyz_lr).unsqueeze(-1)).squeeze(-1)
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

    def apply_structural_update(
        self,
        xyz: torch.Tensor,
        color: torch.Tensor,
        opacity_logits: torch.Tensor,
        log_scale: torch.Tensor,
        rotation: torch.Tensor,
        old_count: int,
        keep_indices: torch.Tensor,
    ) -> None:
        old_params = {
            "xyz": self._xyz,
            "color": self._color,
            "opacity": self._opacity,
            "scaling": self._log_scale,
            "rotation": self._rotation,
        }
        pre_remove_count = int(xyz.shape[0])
        keep_indices = keep_indices.to(device=self.device, dtype=torch.long)
        final_xyz = xyz.index_select(0, keep_indices).contiguous()
        final_color = color.index_select(0, keep_indices).contiguous()
        final_opacity = opacity_logits.index_select(0, keep_indices).contiguous()
        final_log_scale = log_scale.index_select(0, keep_indices).contiguous()
        final_rotation = rotation.index_select(0, keep_indices).contiguous()
        self._xyz = nn.Parameter(final_xyz)
        self._color = nn.Parameter(final_color)
        self._opacity = nn.Parameter(final_opacity)
        self._log_scale = nn.Parameter(final_log_scale)
        self._rotation = nn.Parameter(final_rotation)
        if self.optimizer is None:
            return
        new_params = {
            "xyz": self._xyz,
            "color": self._color,
            "opacity": self._opacity,
            "scaling": self._log_scale,
            "rotation": self._rotation,
        }
        for group in self.optimizer.param_groups:
            name = group["name"]
            old_param = old_params[name]
            new_param = new_params[name]
            old_state = self.optimizer.state.pop(old_param, {})
            new_state: dict[object, object] = {}
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
            self.optimizer.state[new_param] = new_state

    def apply_clone_and_remove(
        self,
        clone_positions: torch.Tensor,
        clone_ids: torch.Tensor,
        target_colors: torch.Tensor,
        clone_counts: torch.Tensor,
        clone_opacity: float,
        remove_opacity_threshold: float,
    ) -> None:
        old_count = self.count
        if old_count == 0:
            return
        clone_counts = clone_counts.to(device=self.device, dtype=torch.long)
        xyz = self._xyz.detach().clone()
        color = self._color.detach().clone()
        opacity_logits = self._opacity.detach().clone()
        log_scale = self._log_scale.detach().clone()
        rotation = self.rotation.detach().clone()
        parent_mask = clone_counts[:old_count] > 0
        if torch.any(parent_mask):
            divisors = (clone_counts[:old_count][parent_mask].to(dtype=log_scale.dtype) + 1.0).unsqueeze(1)
            parent_scale = torch.exp(log_scale[parent_mask]) / divisors
            log_scale[parent_mask] = torch.log(torch.clamp(parent_scale, min=_MIN_SCALE))
        clone_total = int(clone_positions.shape[0])
        if clone_total > 0:
            clone_ids = clone_ids.to(device=self.device, dtype=torch.long)
            clone_divisors = (clone_counts.index_select(0, clone_ids).to(dtype=log_scale.dtype) + 1.0).unsqueeze(1)
            clone_scale = torch.exp(self._log_scale.detach().index_select(0, clone_ids)) / clone_divisors
            xyz = torch.cat((xyz, clone_positions.to(device=self.device, dtype=xyz.dtype)), dim=0)
            color = torch.cat((color, 0.5 * (self._color.detach().index_select(0, clone_ids) + target_colors.to(device=self.device, dtype=color.dtype))), dim=0)
            opacity_logits = torch.cat(
                (
                    opacity_logits,
                    inverse_sigmoid(torch.full((clone_total, 1), float(clone_opacity), device=self.device, dtype=opacity_logits.dtype)),
                ),
                dim=0,
            )
            log_scale = torch.cat((log_scale, torch.log(torch.clamp(clone_scale, min=_MIN_SCALE))), dim=0)
            rotation = torch.cat((rotation, rotation.index_select(0, clone_ids)), dim=0)
        keep_indices = torch.nonzero(torch.sigmoid(opacity_logits[:, 0]) >= float(remove_opacity_threshold), as_tuple=False).flatten()
        self.apply_structural_update(xyz, color, opacity_logits, log_scale, rotation, old_count, keep_indices)

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

    def apply_runtime_config(self) -> None:
        if self.model.optimizer is None:
            return
        self.model.spatial_lr_scale = 1.0 if self.scene is None else float(self.scene.extent_radius)
        self.model.position_lr_mult = float(self.cfg.position_lr_mult)
        self.model.feature_lr_mult = float(self.cfg.feature_lr_mult)
        self.model.opacity_lr_mult = float(self.cfg.opacity_lr_mult)
        self.model.scaling_lr_mult = float(self.cfg.scaling_lr_mult)
        self.model.rotation_lr_mult = float(self.cfg.rotation_lr_mult)
        self.model.base_lr_scheduler = get_expon_lr_func(
            self.cfg.base_lr_init,
            self.cfg.base_lr_final,
            lr_delay_mult=self.cfg.base_lr_delay_mult,
            max_steps=self.cfg.base_lr_max_steps,
        )
        self.model.update_learning_rate(self.iteration)

    def _densify_current_step(self, target: torch.Tensor) -> None:
        if not bool(self.cfg.densify_enabled) or self.model.count <= 0:
            return
        early_interval = int(self.cfg.densify_interval)
        late_interval = int(self.cfg.densify_interval_after)
        switch_iter = int(self.cfg.densify_until_iter)
        interval = early_interval if switch_iter <= 0 or self.iteration <= switch_iter else late_interval
        if interval <= 0:
            return
        if self.iteration % interval != 0:
            return
        available_slots = max(int(self.cfg.max_splats) - self.model.count, 0)
        if available_slots <= 0:
            return
        pixel_count = int(target.shape[0]) * int(target.shape[1])
        desired_clone_count = min(max(int(math.ceil(float(self.cfg.densify_target_ratio) * self.model.count)), 0), available_slots)
        if pixel_count <= 0 or desired_clone_count <= 0:
            return
        select_probability = min(float(desired_clone_count) / float(pixel_count), 1.0)
        max_clone_candidates = max(int(math.ceil(desired_clone_count * max(float(self.cfg.densify_append_multiplier), 1.0))), 1)
        clones = self.context.clone_candidates_current(
            target,
            select_probability=select_probability,
            max_clone_candidates=max_clone_candidates,
            clone_seed=self.iteration,
        )
        clone_count = int(clones["count"].item()) if torch.is_tensor(clones["count"]) else int(clones["count"])
        clone_count = min(clone_count, available_slots)
        if clone_count == 0:
            self.model.apply_clone_and_remove(
                target.new_empty((0, 3)),
                torch.empty((0,), device=self.device, dtype=torch.long),
                target.new_empty((0, 3)),
                clones["clone_counts"],
                float(self.cfg.densify_clone_opacity),
                float(self.cfg.remove_opacity_threshold),
            )
            return
        self.model.apply_clone_and_remove(
            clones["positions"][:clone_count],
            clones["ids"][:clone_count],
            clones["target_colors"][:clone_count],
            clones["clone_counts"],
            float(self.cfg.densify_clone_opacity),
            float(self.cfg.remove_opacity_threshold),
        )

    def _current_dither_strength(self) -> float:
        base = max(float(self.cfg.dither_strength), 0.0)
        if base <= 0.0:
            return 0.0
        decay_end = int(self.cfg.dither_decay_until_iter)
        if decay_end <= 0:
            decay_end = max(int(self.cfg.iterations), 1)
        if decay_end <= 1:
            return base if self.iteration <= 0 else 0.0
        progress = min(max(self.iteration - 1, 0), decay_end - 1) / float(decay_end - 1)
        return base * max(1.0 - progress, 0.0)

    def has_pending_steps(self) -> bool:
        return self.scene is not None

    def snapshot_splats(self) -> torch.Tensor:
        return self.model.splats().detach().contiguous()

    def _render(self, camera: CameraSample, background: tuple[float, float, float], apply_dither: bool = True) -> torch.Tensor:
        return self._render_image(camera, background, apply_dither=apply_dither)[..., :3]

    def _render_image(self, camera: CameraSample, background: tuple[float, float, float], apply_dither: bool = True) -> torch.Tensor:
        self.context.dither_strength = self._current_dither_strength() if apply_dither else 0.0
        image = render_gaussian_splats(
            self.model.splats(),
            camera.camera_params,
            camera.image_size,
            background=background,
            render_seed=self.iteration,
            context=self.context,
        )
        if self.iteration % 8 == 0:
            self.context.readback_and_reallocate_buffers(refresh_buffers=True)
        return _image_hwc(image, camera.image_size)

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
            pred_target = [(self._render(cam, background, apply_dither=False), self._target_image(cam)) for cam in cameras]
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
        rendered, target = self._render_image(camera, background), self._target_image(camera)
        pred = rendered[..., :3]
        alpha = torch.nan_to_num(rendered[..., 3], nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        depth_ratio = torch.nan_to_num(rendered[..., 4], nan=0.0, posinf=1e6, neginf=0.0)
        tile_count = int(getattr(self.context, "_last_required_total", getattr(self.context, "_last_total", 0)))
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        total, l1, mse, train_psnr_tensor = training_loss(
            pred,
            target,
            alpha,
            depth_ratio,
            self.model.opacity,
            self.model.scaling,
            self.cfg.lambda_dssim,
            self.cfg.depth_ratio_weight,
            self.cfg.opacity_reg,
            self.cfg.scale_reg,
        )
        mse_value = float(mse.detach())
        train_psnr = float(train_psnr_tensor.detach())
        total.backward()
        grad_stats = self._grad_stats()
        prev_params = tuple(t.detach().clone() for t in (self.model._xyz, self.model._color, self.model._opacity, self.model._log_scale, self.model._rotation))
        self.model.optimizer.step()
        self.model.optimizer.zero_grad(set_to_none=True)
        self.model.sanitize_after_step(prev_params, self.scene.extent_radius, self.context.max_anisotropy)
        self._densify_current_step(target)
        self.model.add_noise(xyz_lr, self.cfg.noise_lr)
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
            int(tile_count),
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
        progress = tqdm(range(1, self.cfg.iterations + 1), desc="MCMC training")
        for _ in progress:
            step = self.step()
            progress.set_postfix(loss=f"{step.loss:.4f}", tiles=f"{step.tile_count:,}")
            metrics.append(TrainingMetrics(step.iteration, step.loss, step.l1, step.test_psnr, step.test_ssim, step.point_count))
        if output_dir is not None:
            (output_dir / "metrics.json").write_text(json.dumps([asdict(item) for item in metrics], indent=2))
            np.save(output_dir / "final_splats.npy", self.model.splats().detach().cpu().numpy())
        return metrics
