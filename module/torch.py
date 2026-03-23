from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import slangpy as spy
import torch

from .splatting import SplattingContext as _CoreSplattingContext, _PARAM_COUNT, _SHADERS

_CAMERA_PARAM_COUNT = 15
_ALPHA_EPS = 1e-6


def _check_cuda_tensor(name: str, value: torch.Tensor, shape0: int | None = None) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use torch.float32.")
    if not value.is_cuda:
        raise ValueError(f"{name} must be CUDA.")
    if value.ndim != 2 if shape0 is not None else value.ndim != 1:
        raise ValueError(f"{name} has invalid rank.")
    if shape0 is not None and value.shape[0] != shape0:
        raise ValueError(f"{name} must have shape [{shape0}, N].")
    if shape0 is None and value.shape[0] != _CAMERA_PARAM_COUNT:
        raise ValueError(f"{name} must have shape [{_CAMERA_PARAM_COUNT}].")


def _camera_dict(camera: torch.Tensor, image_size: tuple[int, int]) -> dict[str, Any]:
    c = camera.detach()
    q = c[0:4] / torch.clamp(torch.linalg.norm(c[0:4]), min=1e-12)
    w, x, y, z = [float(v.item()) for v in q]
    rot = torch.tensor([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], device=c.device, dtype=c.dtype)
    return {"camPos": spy.float3(*((-rot.T @ c[4:7]).cpu().tolist())), "camBasis": spy.float3x3(rot.cpu().numpy()), "focalPixels": spy.float2(float(c[7].item()), float(c[8].item())), "principalPoint": spy.float2(float(c[9].item()), float(c[10].item())), "viewport": spy.float2(*map(float, image_size)), "nearDepth": float(c[11].item()), "farDepth": float(c[12].item()), "k1": float(c[13].item()), "k2": float(c[14].item())}


def _pack_params(splats: torch.Tensor) -> torch.Tensor:
    packed = splats.contiguous().clone()
    packed[13] = torch.logit(torch.clamp(packed[13], _ALPHA_EPS, 1.0 - _ALPHA_EPS))
    return packed.reshape(-1)


@dataclass
class SplattingContext(_CoreSplattingContext):
    def __post_init__(self) -> None:
        self.device = self.device or spy.create_torch_device(type=spy.DeviceType.cuda, include_paths=[_SHADERS])
        self._init_resources()

    def _prepare_splats(
        self,
        splats: torch.Tensor,
        camera: torch.Tensor,
        image_size: tuple[int, int],
        background: tuple[float, float, float],
        render_seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, Any]]:
        splat_count = int(splats.shape[1])
        self.prepare(splat_count, image_size, background)
        self.render_seed = int(render_seed)
        self.scene["g_Params"].copy_from_torch(_pack_params(splats))
        self.device.sync_to_cuda()
        return splats.new_empty((0,), dtype=torch.long), splats, splat_count, _camera_dict(camera, image_size)

    def project(self, splats: torch.Tensor, camera: torch.Tensor, image_size: tuple[int, int]) -> dict[str, torch.Tensor]:
        _check_cuda_tensor("splats", splats, _PARAM_COUNT)
        _check_cuda_tensor("camera_params", camera)
        _, _, splat_count, camera_vars = self._prepare_splats(splats, camera, image_size, (0.0, 0.0, 0.0), 0)
        total_scanlines = super().project(camera_vars, splat_count)
        order = self._sorted_splat_order_tensor.to_torch()[:splat_count].clone().to(dtype=torch.long)
        sorted_splats = splats.index_select(1, order).contiguous()
        if total_scanlines:
            scanline_counts = self.scanlines["g_ScanlineTileCounts"].to_torch()[:total_scanlines].to(dtype=torch.int64)
            scanline_splats = self.scanlines["g_ScanlineEntryData"].to_torch()[:total_scanlines, 0].to(dtype=torch.int64)
            tile_counts = torch.zeros((splat_count,), device=sorted_splats.device, dtype=torch.int64)
            tile_counts.scatter_add_(0, scanline_splats, scanline_counts)
        else:
            tile_counts = torch.zeros((splat_count,), device=sorted_splats.device, dtype=torch.int64)
        projection = self._view(self.scene_views["g_ProjectionState"].tensor, spy.float4, (splat_count, 2)).to_torch()[:splat_count].clone()
        raster = self._view(self.scene_views["g_RasterState"].tensor, spy.float4, (splat_count, 4)).to_torch()[:splat_count].clone()
        return {"order": order, "sorted_splats": sorted_splats, "projection": projection, "raster": raster, "tile_counts": tile_counts}

    def render(
        self,
        splats: torch.Tensor,
        camera: torch.Tensor,
        image_size: tuple[int, int],
        background: tuple[float, float, float],
        render_seed: int = 0,
    ) -> torch.Tensor:
        _, _, splat_count, camera_vars = self._prepare_splats(splats, camera, image_size, background, render_seed)
        image = super().render(camera_vars, splat_count).to_torch().clone()
        order = self._sorted_splat_order_tensor.to_torch()[:splat_count].clone().to(dtype=torch.long)
        sorted_splats = splats.index_select(1, order).contiguous()
        self._last_order = order
        self._last_alpha = torch.clamp(sorted_splats[13], _ALPHA_EPS, 1 - _ALPHA_EPS)
        self._last_camera = camera.detach().clone()
        self._last_render_seed = int(render_seed)
        return image

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        self.frame["g_OutputGrad"].copy_from_torch(grad_output.contiguous())
        self.device.sync_to_cuda()
        self.render_seed = int(getattr(self, "_last_render_seed", 0))
        grads = super().backward(_camera_dict(self._last_camera, self._size), int(self._last_order.shape[0])).to_torch()[: int(self._last_order.shape[0]) * _PARAM_COUNT].clone().reshape(int(self._last_order.shape[0]), _PARAM_COUNT).mT.contiguous()
        grads[13] /= self._last_alpha * (1 - self._last_alpha)
        return grads

class _RenderFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        splats: torch.Tensor,
        camera: torch.Tensor,
        image_size: tuple[int, int],
        background: tuple[float, float, float],
        render_seed: int,
        context: SplattingContext,
    ) -> torch.Tensor:
        _check_cuda_tensor("splats", splats, _PARAM_COUNT)
        _check_cuda_tensor("camera_params", camera)
        ctx.context = context
        return context.render(splats, camera, image_size, background, render_seed=render_seed)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None, None]:
        return ctx.context.backward(grad_output), None, None, None, None, None


def render_gaussian_splats(
    splats: torch.Tensor,
    camera_params: torch.Tensor,
    image_size: tuple[int, int],
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    render_seed: int = 0,
    context: SplattingContext | None = None,
) -> torch.Tensor:
    return _RenderFn.apply(
        splats,
        camera_params,
        tuple(map(int, image_size)),
        tuple(map(float, background)),
        int(render_seed),
        context or SplattingContext(),
    )
