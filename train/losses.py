from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def _ssim_window(channels: int, device: torch.device, dtype: torch.dtype, size: int = 11) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    kernel = torch.exp(-(coords * coords) / 4.5)
    kernel = (kernel / kernel.sum()).unsqueeze(1)
    return (kernel @ kernel.t()).expand(channels, 1, size, size).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    window = _ssim_window(int(pred.shape[1]), pred.device, pred.dtype, window_size)
    mu0 = F.conv2d(pred, window, padding=window_size // 2, groups=int(pred.shape[1]))
    mu1 = F.conv2d(target, window, padding=window_size // 2, groups=int(target.shape[1]))
    mu00, mu11, mu01 = mu0.square(), mu1.square(), mu0 * mu1
    s00 = F.conv2d(pred.square(), window, padding=window_size // 2, groups=int(pred.shape[1])) - mu00
    s11 = F.conv2d(target.square(), window, padding=window_size // 2, groups=int(target.shape[1])) - mu11
    s01 = F.conv2d(pred * target, window, padding=window_size // 2, groups=int(pred.shape[1])) - mu01
    return (((2.0 * mu01 + 1e-4) * (2.0 * s01 + 9e-4)) / ((mu00 + mu11 + 1e-4) * (s00 + s11 + 9e-4))).mean()


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10((pred - target).square().mean())


def rgb_to_nchw(image: torch.Tensor) -> torch.Tensor:
    return image.permute(2, 0, 1).unsqueeze(0).contiguous()


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(x.dtype).eps
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


def get_expon_lr_func(lr_init: float, lr_final: float, lr_delay_steps: int = 0, lr_delay_mult: float = 1.0, max_steps: int = 1_000_000):
    def helper(step: int) -> float:
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        delay = 1.0 if lr_delay_steps <= 0 else lr_delay_mult + (1.0 - lr_delay_mult) * math.sin(0.5 * math.pi * min(max(step / lr_delay_steps, 0.0), 1.0))
        t = min(max(step / max_steps, 0.0), 1.0)
        return delay * math.exp(math.log(lr_init) * (1.0 - t) + math.log(lr_final) * t)

    return helper
