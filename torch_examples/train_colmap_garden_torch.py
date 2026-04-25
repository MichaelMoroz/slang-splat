from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import math
from pathlib import Path
import time
import sys
from typing import Any

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.metrics import psnr_from_mse
from src.renderer import TorchGaussianRenderSettings, TorchGaussianRendererContext, render_gaussian_splats_torch
from src.scene import (
    ColmapFrame,
    GaussianScene,
    build_training_frames,
    initialize_scene_from_colmap_points,
    load_colmap_reconstruction,
    resolve_supported_sh_coeffs,
    resolve_colmap_init_hparams,
)

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only on non-torch installs
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


_DEFAULT_NEAR = 0.1
_DEFAULT_FAR = 120.0
_MIN_ALPHA = 1e-4
_MAX_ALPHA = 0.9999
_MIN_SCALE = 1e-4
_MAX_SCALE = 3.0
_OUTPUT_RENDER_SUBDIR = "renders"
_DEFAULT_CACHED_RASTER_GRAD_ATOMIC_MODE = "float"
_DEFAULT_CACHED_RASTER_GRAD_FIXED_RO_LOCAL_RANGE = 0.01
_DEFAULT_CACHED_RASTER_GRAD_FIXED_SCALE_RANGE = 0.01
_DEFAULT_CACHED_RASTER_GRAD_FIXED_COLOR_RANGE = 0.2
_DEFAULT_CACHED_RASTER_GRAD_FIXED_OPACITY_RANGE = 0.2
_DEFAULT_THROUGHPUT_WARMUP_STEPS = 1
_DEFAULT_THROUGHPUT_WINDOW = 32


def _require_torch():
    if torch is None:  # pragma: no cover - exercised only on non-torch installs
        raise RuntimeError(
            "PyTorch is required for the torch example trainer. Install a CUDA-enabled PyTorch build first."
        ) from _TORCH_IMPORT_ERROR
    return torch


@dataclass(slots=True)
class TorchGardenTrainConfig:
    colmap_root: Path = Path("dataset/garden")
    images_subdir: str = "images_4"
    iters: int = 30000
    max_gaussians: int = 0
    seed: int = 0
    save_every: int = 1000
    output_dir: Path = Path("outputs/torch_examples/garden_torch")
    near: float = _DEFAULT_NEAR
    far: float = _DEFAULT_FAR
    position_lr: float = 1e-3
    scale_lr: float = 5e-3
    rotation_lr: float = 1e-3
    color_lr: float = 1e-3
    alpha_lr: float = 1e-3
    rgb_mse_weight: float = 0.0
    rgb_l1_weight: float = 1.0
    scale_abs_reg_weight: float = 0.01
    opacity_reg_weight: float = 0.01
    grad_norm_clip: float = 10.0
    lr_final_ratio: float = 0.1
    lr_decay_start_fraction: float = 0.75
    target_psnr: float = 25.0
    cached_raster_grad_atomic_mode: str = _DEFAULT_CACHED_RASTER_GRAD_ATOMIC_MODE
    cached_raster_grad_fixed_ro_local_range: float = _DEFAULT_CACHED_RASTER_GRAD_FIXED_RO_LOCAL_RANGE
    cached_raster_grad_fixed_scale_range: float = _DEFAULT_CACHED_RASTER_GRAD_FIXED_SCALE_RANGE
    cached_raster_grad_fixed_color_range: float = _DEFAULT_CACHED_RASTER_GRAD_FIXED_COLOR_RANGE
    cached_raster_grad_fixed_opacity_range: float = _DEFAULT_CACHED_RASTER_GRAD_FIXED_OPACITY_RANGE
    throughput_warmup_steps: int = _DEFAULT_THROUGHPUT_WARMUP_STEPS
    enable_saves: bool = True
    torch_device: str | None = None
    max_frames: int | None = None

    def __post_init__(self) -> None:
        self.colmap_root = Path(self.colmap_root).resolve()
        self.output_dir = Path(self.output_dir).resolve()
        self.images_subdir = str(self.images_subdir)
        self.iters = max(int(self.iters), 1)
        self.max_gaussians = max(int(self.max_gaussians), 0)
        self.seed = int(self.seed)
        self.save_every = max(int(self.save_every), 0)
        self.near = float(self.near)
        self.far = float(self.far)
        self.position_lr = float(self.position_lr)
        self.scale_lr = float(self.scale_lr)
        self.rotation_lr = float(self.rotation_lr)
        self.color_lr = float(self.color_lr)
        self.alpha_lr = float(self.alpha_lr)
        self.rgb_mse_weight = max(float(self.rgb_mse_weight), 0.0)
        self.rgb_l1_weight = max(float(self.rgb_l1_weight), 0.0)
        self.scale_abs_reg_weight = max(float(self.scale_abs_reg_weight), 0.0)
        self.opacity_reg_weight = max(float(self.opacity_reg_weight), 0.0)
        self.grad_norm_clip = max(float(self.grad_norm_clip), 0.0)
        self.lr_final_ratio = min(max(float(self.lr_final_ratio), 0.0), 1.0)
        self.lr_decay_start_fraction = min(max(float(self.lr_decay_start_fraction), 0.0), 1.0)
        self.target_psnr = float(self.target_psnr)
        self.cached_raster_grad_atomic_mode = str(self.cached_raster_grad_atomic_mode)
        self.cached_raster_grad_fixed_ro_local_range = float(self.cached_raster_grad_fixed_ro_local_range)
        self.cached_raster_grad_fixed_scale_range = float(self.cached_raster_grad_fixed_scale_range)
        self.cached_raster_grad_fixed_color_range = float(self.cached_raster_grad_fixed_color_range)
        self.cached_raster_grad_fixed_opacity_range = float(self.cached_raster_grad_fixed_opacity_range)
        self.throughput_warmup_steps = max(int(self.throughput_warmup_steps), 0)
        self.enable_saves = bool(self.enable_saves)
        if self.max_frames is not None:
            self.max_frames = max(int(self.max_frames), 1)


@dataclass(slots=True)
class FrameOrderState:
    rng: np.random.Generator
    frame_count: int
    frame_order: np.ndarray
    cursor: int

    @classmethod
    def create(cls, frame_count: int, seed: int) -> "FrameOrderState":
        count = max(int(frame_count), 1)
        rng = np.random.default_rng(int(seed))
        return cls(rng=rng, frame_count=count, frame_order=rng.permutation(count).astype(np.int32), cursor=0)


@dataclass(slots=True)
class FrameMetricTracker:
    loss: np.ndarray
    mse: np.ndarray
    psnr: np.ndarray
    visited: np.ndarray

    @classmethod
    def create(cls, frame_count: int) -> "FrameMetricTracker":
        count = max(int(frame_count), 1)
        return cls(
            loss=np.full((count,), np.nan, dtype=np.float64),
            mse=np.full((count,), np.nan, dtype=np.float64),
            psnr=np.full((count,), np.nan, dtype=np.float64),
            visited=np.zeros((count,), dtype=bool),
        )

    def update(self, frame_index: int, loss: float, mse: float, psnr: float) -> None:
        idx = int(frame_index)
        self.loss[idx] = float(loss)
        self.mse[idx] = float(mse)
        self.psnr[idx] = float(psnr)
        self.visited[idx] = True

    def mean(self, name: str) -> float:
        values = getattr(self, name)
        valid = self.visited & np.isfinite(values)
        return float(np.mean(values[valid], dtype=np.float64)) if np.any(valid) else float("nan")


@dataclass(slots=True)
class IterationRateTracker:
    warmup_steps: int = _DEFAULT_THROUGHPUT_WARMUP_STEPS
    window_size: int = _DEFAULT_THROUGHPUT_WINDOW
    active_steps: int = 0
    active_seconds: float = 0.0
    seen_steps: int = 0
    recent_seconds: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.warmup_steps = max(int(self.warmup_steps), 0)
        self.window_size = max(int(self.window_size), 1)
        self.recent_seconds = deque(maxlen=self.window_size)

    def update(self, step_seconds: float) -> None:
        self.seen_steps += 1
        if self.seen_steps <= self.warmup_steps:
            return
        duration = max(float(step_seconds), 1e-9)
        self.active_steps += 1
        self.active_seconds += duration
        self.recent_seconds.append(duration)

    def recent_iters_per_second(self) -> float:
        if not self.recent_seconds:
            return float("nan")
        return float(len(self.recent_seconds) / sum(self.recent_seconds))

    def avg_iters_per_second(self) -> float:
        if self.active_seconds <= 0.0:
            return float("nan")
        return float(self.active_steps / self.active_seconds)


def scene_to_torch_params(scene: GaussianScene, device: Any) -> dict[str, Any]:
    torch_mod = _require_torch()
    sh_coeffs = resolve_supported_sh_coeffs(scene.sh_coeffs, scene.colors)
    return {
        "positions": torch_mod.nn.Parameter(torch_mod.as_tensor(scene.positions, dtype=torch_mod.float32, device=device)),
        "log_scales": torch_mod.nn.Parameter(torch_mod.as_tensor(scene.scales, dtype=torch_mod.float32, device=device)),
        "rotations": torch_mod.nn.Parameter(torch_mod.as_tensor(scene.rotations, dtype=torch_mod.float32, device=device)),
        "sh_coeffs": torch_mod.nn.Parameter(torch_mod.as_tensor(sh_coeffs, dtype=torch_mod.float32, device=device)),
        "alpha": torch_mod.nn.Parameter(torch_mod.as_tensor(scene.opacities[:, None], dtype=torch_mod.float32, device=device)),
    }


def pack_torch_splats(params_dict: dict[str, Any]) -> Any:
    torch_mod = _require_torch()
    return torch_mod.cat(
        (
            params_dict["positions"],
            params_dict["log_scales"],
            params_dict["rotations"],
            params_dict["sh_coeffs"].reshape(params_dict["sh_coeffs"].shape[0], -1),
            params_dict["alpha"],
        ),
        dim=1,
    )


def frame_to_camera_tensor(frame: ColmapFrame, device: Any, near: float = _DEFAULT_NEAR, far: float = _DEFAULT_FAR) -> Any:
    torch_mod = _require_torch()
    return torch_mod.tensor(
        [
            float(frame.q_wxyz[0]),
            float(frame.q_wxyz[1]),
            float(frame.q_wxyz[2]),
            float(frame.q_wxyz[3]),
            float(frame.t_xyz[0]),
            float(frame.t_xyz[1]),
            float(frame.t_xyz[2]),
            float(frame.fx),
            float(frame.fy),
            float(frame.cx),
            float(frame.cy),
            float(near),
            float(far),
            float(frame.k1),
            float(frame.k2),
        ],
        dtype=torch_mod.float32,
        device=device,
    )


def load_target_cache(frames: list[ColmapFrame]) -> list[np.ndarray]:
    cache: list[np.ndarray] = []
    for frame in frames:
        with Image.open(frame.image_path) as pil_image:
            cache.append(np.array(pil_image.convert("RGB"), dtype=np.uint8, copy=True))
    return cache


def rgb8_to_linear_rgb(image: Any) -> Any:
    torch_mod = _require_torch()
    tensor = image
    if tensor.dtype == torch_mod.uint8:
        tensor = tensor.to(dtype=torch_mod.float32) / 255.0
    else:
        tensor = tensor.to(dtype=torch_mod.float32)
    threshold = 0.04045
    low = tensor / 12.92
    high = torch_mod.pow((tensor + 0.055) / 1.055, 2.4)
    return torch_mod.where(tensor <= threshold, low, high)


def linear_rgb_to_srgb8(image: Any) -> np.ndarray:
    torch_mod = _require_torch()
    tensor = torch_mod.clamp(image.detach(), 0.0, 1.0)
    threshold = 0.0031308
    low = tensor * 12.92
    high = 1.055 * torch_mod.pow(torch_mod.clamp(tensor, min=threshold), 1.0 / 2.4) - 0.055
    srgb = torch_mod.where(tensor <= threshold, low, high)
    return torch_mod.clamp(srgb * 255.0 + 0.5, 0.0, 255.0).to(dtype=torch_mod.uint8).detach().cpu().numpy()


def reset_frame_order(state: FrameOrderState) -> None:
    state.frame_order = state.rng.permutation(state.frame_count).astype(np.int32)
    state.cursor = 0


def next_frame_index(state: FrameOrderState) -> int:
    if state.cursor >= state.frame_count:
        reset_frame_order(state)
    frame_index = int(state.frame_order[state.cursor])
    state.cursor += 1
    return frame_index


def project_scene_params_(params_dict: dict[str, Any]) -> None:
    torch_mod = _require_torch()
    with torch_mod.no_grad():
        rotations = params_dict["rotations"]
        norms = torch_mod.linalg.norm(rotations, dim=1, keepdim=True)
        safe_mask = norms > 1e-12
        normalized = torch_mod.where(safe_mask, rotations / torch_mod.clamp(norms, min=1e-12), torch_mod.zeros_like(rotations))
        normalized[~safe_mask.squeeze(1), 0] = 1.0
        params_dict["rotations"].copy_(normalized)
        params_dict["alpha"].clamp_(_MIN_ALPHA, _MAX_ALPHA)
        params_dict["log_scales"].clamp_(math.log(_MIN_SCALE), math.log(_MAX_SCALE))


def compute_rgb_loss_metrics(rendered_rgb: Any, target_rgb: Any) -> tuple[Any, float, float]:
    torch_mod = _require_torch()
    diff = rendered_rgb - target_rgb
    loss = torch_mod.mean(torch_mod.abs(diff))
    mse = float(torch_mod.mean(diff * diff).detach().item())
    return loss, mse, float(psnr_from_mse(mse))


def compute_training_loss_terms(
    rendered_rgb: Any,
    target_rgb: Any,
    params_dict: dict[str, Any],
    config: TorchGardenTrainConfig,
) -> tuple[Any, float, float, float]:
    torch_mod = _require_torch()
    diff = rendered_rgb - target_rgb
    mse = torch_mod.mean(diff * diff)
    l1 = torch_mod.mean(torch_mod.abs(diff))
    scale_abs = torch_mod.mean(torch_mod.exp(params_dict["log_scales"]))
    opacity_mean = torch_mod.mean(params_dict["alpha"])
    total = (
        config.rgb_mse_weight * mse
        + config.rgb_l1_weight * l1
        + config.scale_abs_reg_weight * scale_abs
        + config.opacity_reg_weight * opacity_mean
    )
    mse_value = float(mse.detach().item())
    return total, float(l1.detach().item()), mse_value, float(psnr_from_mse(mse_value))


def build_render_settings(frame: ColmapFrame, config: TorchGardenTrainConfig) -> TorchGaussianRenderSettings:
    return TorchGaussianRenderSettings(
        width=frame.width,
        height=frame.height,
        cached_raster_grad_atomic_mode=config.cached_raster_grad_atomic_mode,
        cached_raster_grad_fixed_ro_local_range=config.cached_raster_grad_fixed_ro_local_range,
        cached_raster_grad_fixed_scale_range=config.cached_raster_grad_fixed_scale_range,
        cached_raster_grad_fixed_color_range=config.cached_raster_grad_fixed_color_range,
        cached_raster_grad_fixed_opacity_range=config.cached_raster_grad_fixed_opacity_range,
    )


def _torch_device_from_config(config: TorchGardenTrainConfig):
    torch_mod = _require_torch()
    if not torch_mod.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required for the torch example trainer.")
    if config.torch_device is None:
        return torch_mod.device(f"cuda:{torch_mod.cuda.current_device()}")
    device = torch_mod.device(config.torch_device)
    if device.type != "cuda":
        raise ValueError(f"torch_device must be CUDA, got {device}.")
    return device


def _select_frames(frames: list[ColmapFrame], max_frames: int | None) -> list[ColmapFrame]:
    if max_frames is None or max_frames >= len(frames):
        return frames
    return list(frames[:max_frames])


def _build_camera_cache(frames: list[ColmapFrame], device: Any, config: TorchGardenTrainConfig) -> list[Any]:
    return [frame_to_camera_tensor(frame, device, near=config.near, far=config.far) for frame in frames]


def _build_render_settings_cache(frames: list[ColmapFrame], config: TorchGardenTrainConfig) -> list[TorchGaussianRenderSettings]:
    return [build_render_settings(frame, config) for frame in frames]


def _target_linear_from_cache(targets_cpu: list[np.ndarray], frame_index: int, device: Any) -> Any:
    torch_mod = _require_torch()
    target = torch_mod.from_numpy(targets_cpu[frame_index]).to(device=device, non_blocking=True)
    return rgb8_to_linear_rgb(target)


def _optimizer_for_params(params_dict: dict[str, Any], config: TorchGardenTrainConfig):
    torch_mod = _require_torch()
    return torch_mod.optim.Adam(
        [
            {"params": [params_dict["positions"]], "lr": config.position_lr},
            {"params": [params_dict["log_scales"]], "lr": config.scale_lr},
            {"params": [params_dict["rotations"]], "lr": config.rotation_lr},
            {"params": [params_dict["sh_coeffs"]], "lr": config.color_lr},
            {"params": [params_dict["alpha"]], "lr": config.alpha_lr},
        ]
    )


def _scheduler_for_optimizer(optimizer: Any, config: TorchGardenTrainConfig):
    torch_mod = _require_torch()
    if config.iters <= 1 or config.lr_final_ratio >= 1.0:
        return None

    def lr_lambda(step: int) -> float:
        progress = min(max(float(step) / float(max(config.iters - 1, 1)), 0.0), 1.0)
        if progress <= config.lr_decay_start_fraction:
            return 1.0
        tail = max(1.0 - config.lr_decay_start_fraction, 1e-6)
        local = min(max((progress - config.lr_decay_start_fraction) / tail, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * local))
        return float(config.lr_final_ratio + (1.0 - config.lr_final_ratio) * cosine)

    return torch_mod.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _optimizer_lr(optimizer: Any) -> float:
    group = optimizer.param_groups[0] if optimizer.param_groups else {"lr": float("nan")}
    return float(group["lr"])


def _ensure_output_dirs(config: TorchGardenTrainConfig) -> tuple[Path, Path]:
    render_dir = config.output_dir / _OUTPUT_RENDER_SUBDIR
    checkpoint_path = config.output_dir / "checkpoint_final.pt"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.enable_saves and config.save_every > 0:
        render_dir.mkdir(parents=True, exist_ok=True)
    return render_dir, checkpoint_path


def _save_render(rendered_rgb: Any, output_path: Path) -> None:
    Image.fromarray(linear_rgb_to_srgb8(rendered_rgb), mode="RGB").save(output_path)


def _save_checkpoint(
    checkpoint_path: Path,
    params_dict: dict[str, Any],
    optimizer: Any,
    step: int,
    history: dict[str, list[float]],
) -> None:
    torch_mod = _require_torch()
    packed = pack_torch_splats(params_dict).detach().cpu()
    torch_mod.save(
        {
            "step": int(step),
            "optimizer_state": optimizer.state_dict(),
            "packed_splats": packed,
            "positions": params_dict["positions"].detach().cpu(),
            "log_scales": params_dict["log_scales"].detach().cpu(),
            "rotations": params_dict["rotations"].detach().cpu(),
            "sh_coeffs": params_dict["sh_coeffs"].detach().cpu(),
            "alpha": params_dict["alpha"].detach().cpu(),
            "history": history,
        },
        checkpoint_path,
    )


def _prewarm_renderer(
    params_dict: dict[str, Any],
    target_linear: Any,
    camera: Any,
    settings: TorchGaussianRenderSettings,
    context: TorchGaussianRendererContext,
    optimizer: Any,
    config: TorchGardenTrainConfig,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    rendered = render_gaussian_splats_torch(pack_torch_splats(params_dict), camera, settings, context)
    loss, _, _, _ = compute_training_loss_terms(rendered[..., :3], target_linear, params_dict, config)
    loss.backward()
    optimizer.zero_grad(set_to_none=True)


def _execute_training_step(
    params_dict: dict[str, Any],
    optimizer: Any,
    target_linear: Any,
    camera: Any,
    settings: TorchGaussianRenderSettings,
    context: TorchGaussianRendererContext,
    config: TorchGardenTrainConfig,
) -> tuple[Any, float, float, float, float]:
    torch_mod = _require_torch()
    optimizer.zero_grad(set_to_none=True)
    rendered = render_gaussian_splats_torch(pack_torch_splats(params_dict), camera, settings, context)
    rendered_rgb = rendered[..., :3]
    loss, l1_value, mse, psnr = compute_training_loss_terms(rendered_rgb, target_linear, params_dict, config)
    loss.backward()
    if config.grad_norm_clip > 0.0:
        torch_mod.nn.utils.clip_grad_norm_(list(params_dict.values()), max_norm=config.grad_norm_clip)
    optimizer.step()
    project_scene_params_(params_dict)
    return rendered_rgb.detach(), float(loss.detach().item()), l1_value, mse, psnr


def run_training(config: TorchGardenTrainConfig) -> dict[str, object]:
    device = _torch_device_from_config(config)
    recon = load_colmap_reconstruction(config.colmap_root)
    frames = _select_frames(build_training_frames(recon, images_subdir=config.images_subdir), config.max_frames)
    resolved_init = resolve_colmap_init_hparams(recon, config.max_gaussians)
    scene = initialize_scene_from_colmap_points(recon, max_gaussians=config.max_gaussians, seed=config.seed, init_hparams=resolved_init)
    params_dict = scene_to_torch_params(scene, device)
    optimizer = _optimizer_for_params(params_dict, config)
    scheduler = _scheduler_for_optimizer(optimizer, config)
    context = TorchGaussianRendererContext(torch_device=device)
    targets_cpu = load_target_cache(frames)
    camera_cache = _build_camera_cache(frames, device, config)
    settings_cache = _build_render_settings_cache(frames, config)
    frame_state = FrameOrderState.create(len(frames), config.seed)
    frame_metrics = FrameMetricTracker.create(len(frames))
    rate_tracker = IterationRateTracker(warmup_steps=config.throughput_warmup_steps)
    render_dir, checkpoint_path = _ensure_output_dirs(config)
    history: dict[str, list[float]] = {"loss": [], "l1": [], "mse": [], "psnr": [], "lr": [], "frame_index": [], "iter_s": [], "avg_iter_s": []}
    last_rendered = last_loss = last_psnr = last_mse = None
    best_psnr = float("-inf")
    best_avg_psnr = float("-inf")

    _prewarm_renderer(
        params_dict,
        _target_linear_from_cache(targets_cpu, 0, device),
        camera_cache[0],
        settings_cache[0],
        context,
        optimizer,
        config,
    )
    progress = tqdm(range(config.iters), total=config.iters, desc="torch-colmap", dynamic_ncols=True)
    for step in progress:
        step_start = time.perf_counter()
        frame_index = next_frame_index(frame_state)
        frame = frames[frame_index]
        target_linear = _target_linear_from_cache(targets_cpu, frame_index, device)
        rendered_rgb, loss_value, l1_value, mse, psnr = _execute_training_step(
            params_dict,
            optimizer,
            target_linear,
            camera_cache[frame_index],
            settings_cache[frame_index],
            context,
            config,
        )
        if scheduler is not None:
            scheduler.step()
        rate_tracker.update(time.perf_counter() - step_start)

        frame_metrics.update(frame_index, loss_value, mse, psnr)
        avg_psnr = frame_metrics.mean("psnr")
        best_psnr = max(best_psnr, psnr)
        best_avg_psnr = max(best_avg_psnr, avg_psnr)
        recent_iter_s = rate_tracker.recent_iters_per_second()
        avg_iter_s = rate_tracker.avg_iters_per_second()
        history["loss"].append(loss_value)
        history["l1"].append(l1_value)
        history["mse"].append(mse)
        history["psnr"].append(psnr)
        history["lr"].append(_optimizer_lr(optimizer))
        history["frame_index"].append(float(frame_index))
        history["iter_s"].append(recent_iter_s)
        history["avg_iter_s"].append(avg_iter_s)
        progress.set_postfix(
            loss=f"{loss_value:.6f}",
            psnr=f"{psnr:.2f}",
            avg_loss=f"{frame_metrics.mean('loss'):.6f}",
            avg_psnr=f"{avg_psnr:.2f}",
            best_psnr=f"{best_psnr:.2f}",
            it_s="warmup" if not math.isfinite(recent_iter_s) else f"{recent_iter_s:.1f}",
            avg_it_s="warmup" if not math.isfinite(avg_iter_s) else f"{avg_iter_s:.1f}",
            lr=f"{_optimizer_lr(optimizer):.2e}",
            frame=frame.image_path.name,
        )

        if config.enable_saves and config.save_every > 0 and ((step + 1) % config.save_every == 0 or step == 0):
            _save_render(rendered_rgb, render_dir / f"step_{step + 1:06d}_{frame.image_path.stem}.png")

        last_rendered = rendered_rgb.detach()
        last_loss, last_mse, last_psnr = loss_value, mse, psnr

    _save_checkpoint(checkpoint_path, params_dict, optimizer, config.iters, history)
    if config.enable_saves and last_rendered is not None:
        _save_render(last_rendered, config.output_dir / "render_final.png")
    return {
        "step": config.iters,
        "checkpoint_path": checkpoint_path,
        "last_loss": float(last_loss if last_loss is not None else float("nan")),
        "last_mse": float(last_mse if last_mse is not None else float("nan")),
        "last_psnr": float(last_psnr if last_psnr is not None else float("nan")),
        "avg_loss": frame_metrics.mean("loss"),
        "avg_psnr": frame_metrics.mean("psnr"),
        "best_psnr": best_psnr,
        "best_avg_psnr": best_avg_psnr,
        "target_psnr": float(config.target_psnr),
        "target_reached": bool(best_avg_psnr >= config.target_psnr or best_psnr >= config.target_psnr),
        "iter_s": rate_tracker.recent_iters_per_second(),
        "avg_iter_s": rate_tracker.avg_iters_per_second(),
        "history": history,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Small torch-side COLMAP trainer example for dataset/garden.")
    parser.add_argument("--colmap-root", type=Path, default=Path("dataset/garden"))
    parser.add_argument("--images-subdir", type=str, default="images_4")
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--max-gaussians", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/torch_examples/garden_torch"))
    parser.add_argument("--near", type=float, default=_DEFAULT_NEAR)
    parser.add_argument("--far", type=float, default=_DEFAULT_FAR)
    parser.add_argument("--position-lr", type=float, default=1e-3)
    parser.add_argument("--scale-lr", type=float, default=5e-3)
    parser.add_argument("--rotation-lr", type=float, default=1e-3)
    parser.add_argument("--color-lr", type=float, default=1e-3)
    parser.add_argument("--alpha-lr", type=float, default=1e-3)
    parser.add_argument("--rgb-mse-weight", type=float, default=0.0)
    parser.add_argument("--rgb-l1-weight", type=float, default=1.0)
    parser.add_argument("--scale-abs-reg-weight", type=float, default=0.01)
    parser.add_argument("--opacity-reg-weight", type=float, default=0.01)
    parser.add_argument("--grad-norm-clip", type=float, default=10.0)
    parser.add_argument("--lr-final-ratio", type=float, default=0.1)
    parser.add_argument("--lr-decay-start-fraction", type=float, default=0.75)
    parser.add_argument("--target-psnr", type=float, default=25.0)
    parser.add_argument("--cached-raster-grad-atomic-mode", type=str, default=_DEFAULT_CACHED_RASTER_GRAD_ATOMIC_MODE)
    parser.add_argument("--cached-raster-grad-fixed-ro-local-range", type=float, default=_DEFAULT_CACHED_RASTER_GRAD_FIXED_RO_LOCAL_RANGE)
    parser.add_argument("--cached-raster-grad-fixed-scale-range", type=float, default=_DEFAULT_CACHED_RASTER_GRAD_FIXED_SCALE_RANGE)
    parser.add_argument("--cached-raster-grad-fixed-color-range", type=float, default=_DEFAULT_CACHED_RASTER_GRAD_FIXED_COLOR_RANGE)
    parser.add_argument("--cached-raster-grad-fixed-opacity-range", type=float, default=_DEFAULT_CACHED_RASTER_GRAD_FIXED_OPACITY_RANGE)
    parser.add_argument("--throughput-warmup-steps", type=int, default=_DEFAULT_THROUGHPUT_WARMUP_STEPS)
    parser.add_argument("--torch-device", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--disable-saves", action="store_true")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    config = TorchGardenTrainConfig(
        colmap_root=args.colmap_root,
        images_subdir=args.images_subdir,
        iters=args.iters,
        max_gaussians=args.max_gaussians,
        seed=args.seed,
        save_every=args.save_every,
        output_dir=args.output_dir,
        near=args.near,
        far=args.far,
        position_lr=args.position_lr,
        scale_lr=args.scale_lr,
        rotation_lr=args.rotation_lr,
        color_lr=args.color_lr,
        alpha_lr=args.alpha_lr,
        rgb_mse_weight=args.rgb_mse_weight,
        rgb_l1_weight=args.rgb_l1_weight,
        scale_abs_reg_weight=args.scale_abs_reg_weight,
        opacity_reg_weight=args.opacity_reg_weight,
        grad_norm_clip=args.grad_norm_clip,
        lr_final_ratio=args.lr_final_ratio,
        lr_decay_start_fraction=args.lr_decay_start_fraction,
        target_psnr=args.target_psnr,
        cached_raster_grad_atomic_mode=args.cached_raster_grad_atomic_mode,
        cached_raster_grad_fixed_ro_local_range=args.cached_raster_grad_fixed_ro_local_range,
        cached_raster_grad_fixed_scale_range=args.cached_raster_grad_fixed_scale_range,
        cached_raster_grad_fixed_color_range=args.cached_raster_grad_fixed_color_range,
        cached_raster_grad_fixed_opacity_range=args.cached_raster_grad_fixed_opacity_range,
        throughput_warmup_steps=args.throughput_warmup_steps,
        enable_saves=not bool(args.disable_saves),
        torch_device=args.torch_device,
        max_frames=args.max_frames,
    )
    summary = run_training(config)
    print(
        f"Finished {summary['step']} steps. "
        f"last_loss={summary['last_loss']:.6f} last_psnr={summary['last_psnr']:.2f} "
        f"avg_loss={summary['avg_loss']:.6f} avg_psnr={summary['avg_psnr']:.2f} "
        f"best_psnr={summary['best_psnr']:.2f} "
        f"iter_s={summary['iter_s']:.1f} avg_iter_s={summary['avg_iter_s']:.1f} "
        f"target={summary['target_psnr']:.2f} "
        f"reached={summary['target_reached']}"
    )
    print(f"Checkpoint: {summary['checkpoint_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
