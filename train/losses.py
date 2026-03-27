from __future__ import annotations

import math

import torch
import torch.nn.functional as F


_ALPHA_OPAQUE_WEIGHT = 0.005


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def _gaussian_kernel_1d(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = (window_size - 1) * 0.5
    x = torch.arange(window_size, device=device, dtype=dtype) - radius
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _ssim_blur(image: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    channels = image.shape[1]
    kernel_1d = _gaussian_kernel_1d(window_size, 1.5, image.device, image.dtype)
    kernel_x = kernel_1d.view(1, 1, 1, window_size).expand(channels, 1, 1, window_size)
    kernel_y = kernel_1d.view(1, 1, window_size, 1).expand(channels, 1, window_size, 1)
    pad = window_size // 2
    blurred = F.conv2d(image, kernel_x, padding=(0, pad), groups=channels)
    return F.conv2d(blurred, kernel_y, padding=(pad, 0), groups=channels)


def _ssim_impl(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    stacked = torch.cat((pred, target, pred.square(), target.square(), pred * target), dim=0)
    mu0, mu1, pred_sq_blur, target_sq_blur, pred_target_blur = _ssim_blur(stacked, window_size).unbind(dim=0)
    mu0 = mu0.unsqueeze(0)
    mu1 = mu1.unsqueeze(0)
    mu00, mu11, mu01 = mu0.square(), mu1.square(), mu0 * mu1
    s00 = pred_sq_blur.unsqueeze(0) - mu00
    s11 = target_sq_blur.unsqueeze(0) - mu11
    s01 = pred_target_blur.unsqueeze(0) - mu01
    return (((2.0 * mu01 + 1e-4) * (2.0 * s01 + 9e-4)) / ((mu00 + mu11 + 1e-4) * (s00 + s11 + 9e-4))).mean()


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    return _ssim_impl(pred, target, window_size)


def _training_loss_impl(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: torch.Tensor,
    depth_ratio: torch.Tensor,
    opacity: torch.Tensor,
    scaling: torch.Tensor,
    lambda_dssim: float,
    depth_ratio_weight: float,
    opacity_reg: float,
    scale_reg: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    l1 = l1_loss(pred, target)
    mse = (pred - target).square().mean()
    total = (
        (1.0 - lambda_dssim) * l1
        + lambda_dssim * (1.0 - ssim(rgb_to_nchw(pred), rgb_to_nchw(target)))
        + _ALPHA_OPAQUE_WEIGHT * (alpha - 1.0).abs().mean()
        + depth_ratio_weight * depth_ratio.mean()
        + opacity_reg * opacity.abs().mean()
        + scale_reg * scaling.abs().mean()
    )
    total = torch.nan_to_num(total, nan=0.0, posinf=1e6, neginf=0.0)
    train_psnr = -10.0 * torch.log10(mse.clamp_min(torch.finfo(mse.dtype).tiny))
    return total.clone(), l1.clone(), mse.clone(), train_psnr.clone()


def training_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: torch.Tensor,
    depth_ratio: torch.Tensor,
    opacity: torch.Tensor,
    scaling: torch.Tensor,
    lambda_dssim: float,
    depth_ratio_weight: float,
    opacity_reg: float,
    scale_reg: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _training_loss_impl(pred, target, alpha, depth_ratio, opacity, scaling, lambda_dssim, depth_ratio_weight, opacity_reg, scale_reg)


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
