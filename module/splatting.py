# Tile-based Gaussian splatting renderer using SlangPy for GPU compute dispatch.
#
# Splat parameter tensor is component-major [14, N].  The flat GPU buffer
# is param-major: params[paramIdx * N + splatIdx].  Row semantics are
# defined in shaders/utility/splatting/params.slang:
#   0-2   position (x, y, z)
#   3-5   log-scale (x, y, z)
#   6-9   rotation quaternion (x, y, z, w)
#   10-12 colour (r, g, b)
#   13    opacity — stored as probability [0,1] on the Python side,
#         converted to logit space before upload to GPU
#
# Camera parameter vector [15]:
#   [0..3]   quaternion (w, x, y, z)
#   [4..6]   translation t  (world-space camera pos = -R^T @ t)
#   [7..8]   focal length in pixels (fx, fy)
#   [9..10]  principal point (cx, cy)
#   [11..12] near / far depth
#   [13..14] radial distortion (k1, k2)
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import slangpy as spy
import torch

_ROOT = Path(__file__).resolve().parent
_SHADERS = _ROOT / "shaders"
_PARAM_COUNT = 14
_CAMERA_PARAM_COUNT = 15
_ALPHA_EPS = 1e-6
_ALPHA_CUTOFF = 1 / 255
_TRANS_THRESHOLD = 0.005
_RADIUS_SCALE = 1.0


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
    q = c[0:4]
    t = c[4:7]
    q = q / torch.clamp(torch.linalg.norm(q), min=1e-12)
    w, x, y, z = [float(v.item()) for v in q]
    rot = torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        device=c.device,
        dtype=c.dtype,
    )
    cam_pos = (-rot.T @ t).cpu()
    return {
        "camPos": spy.float3(*cam_pos.tolist()),
        "camBasis": spy.float3x3(rot.cpu().numpy()),
        "focalPixels": spy.float2(float(c[7].item()), float(c[8].item())),
        "principalPoint": spy.float2(float(c[9].item()), float(c[10].item())),
        "viewport": spy.float2(float(image_size[0]), float(image_size[1])),
        "nearDepth": float(c[11].item()),
        "farDepth": float(c[12].item()),
        "k1": float(c[13].item()),
        "k2": float(c[14].item()),
    }


def _pack_params(splats: torch.Tensor) -> torch.Tensor:
    packed = splats.contiguous().clone()
    packed[13] = torch.logit(torch.clamp(packed[13], _ALPHA_EPS, 1.0 - _ALPHA_EPS))
    return packed.reshape(-1)


@dataclass
class SplattingContext:
    device: spy.Device | None = None

    def __post_init__(self) -> None:
        self.device = self.device or spy.create_torch_device(type=spy.DeviceType.cuda, include_paths=[_SHADERS])
        self.mod = spy.Module.load_from_file(self.device, str(_SHADERS / "module.slang"))
        self.k_project_scanline_count = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csProjectScanlineCount"]))
        self.k_emit_scanlines = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csEmitScanlines"]))
        self.k_emit_scanline_tile_counts = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csEmitScanlineTileCounts"]))
        self.k_emit_tile_entries = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csEmitTileEntries"]))
        self.k_build_tile_ranges = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csBuildTileRanges"]))
        self.k_raster_fwd = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csRasterForward"]))
        self.k_raster_bwd = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csRasterBackward"]))
        self._size = (0, 0)

    def _view(self, tensor: spy.Tensor, dtype: Any, shape: tuple[int, ...]) -> spy.Tensor:
        if isinstance(dtype, type):
            dtype = spy.Tensor.empty(self.device, shape=(1,), dtype=dtype).dtype
        return spy.Tensor(tensor.storage, getattr(dtype, "struct", dtype), shape)

    def _alloc_frame(self, shape: tuple[int, int]) -> None:
        if self._size == shape:
            return
        width, height = shape
        self.frame = {
            "g_Output": spy.Tensor.empty(self.device, shape=(height, width), dtype=spy.float4),
            "g_OutputGrad": spy.Tensor.empty(self.device, shape=(height, width), dtype=spy.float4),
            "g_ForwardState": spy.Tensor.empty(self.device, shape=(height, width), dtype=spy.float4),
            "g_ForwardEnd": spy.Tensor.empty(self.device, shape=(height, width), dtype="uint"),
        }
        tw, th = (width + 7) // 8, (height + 7) // 8
        self.tiles = {"g_TileRanges": spy.Tensor.empty(self.device, shape=(tw * th * 2,), dtype="uint")}
        self._size = shape

    def _alloc_scene(self, splat_count: int) -> None:
        splat_count = max(splat_count, 1)
        self.scene = {
            "g_Params": spy.Tensor.empty(self.device, shape=(_PARAM_COUNT * splat_count,), dtype=float),
            "g_ScanlineCounts": spy.Tensor.empty(self.device, shape=(splat_count,), dtype="uint"),
            "g_ScanlineOffsets": spy.Tensor.empty(self.device, shape=(splat_count,), dtype="uint"),
            "g_TileCounts": spy.Tensor.empty(self.device, shape=(splat_count,), dtype="uint"),
            "g_ParamGrads": spy.Tensor.empty(self.device, shape=(splat_count * _PARAM_COUNT,), dtype=float),
        }
        self.scene_views = {
            "g_ProjectionState": spy.InstanceTensor(self.mod.ProjectionState, (splat_count,)),
            "g_RasterState": spy.InstanceTensor(self.mod.Gaussian3D, (splat_count,)),
        }

    def _alloc_scanlines(self, scanline_count: int) -> None:
        scanline_count = max(scanline_count, 1)
        self.scanlines = {
            "g_ScanlineEntryData": spy.Tensor.empty(self.device, shape=(scanline_count, 3), dtype="uint"),
            "g_ScanlineTileCounts": spy.Tensor.empty(self.device, shape=(scanline_count,), dtype="uint"),
            "g_ScanlineTileOffsets": spy.Tensor.empty(self.device, shape=(scanline_count,), dtype="uint"),
        }
        self.scanline_views = {
            "g_ScanlineEntries": spy.InstanceTensor(self.mod.ScanlineEntry, (scanline_count,), self._view(self.scanlines["g_ScanlineEntryData"], self.mod.ScanlineEntry, (scanline_count,))),
        }

    def _alloc_entries(self, entry_count: int) -> None:
        entry_count = max(entry_count, 1)
        self.raw = {
            "g_TileEntryData": spy.Tensor.empty(self.device, shape=(entry_count, 2), dtype="uint"),
            "g_SortedEntryData": spy.Tensor.empty(self.device, shape=(entry_count, 2), dtype="uint"),
        }
        self.entry_views = {
            "g_TileEntries": spy.InstanceTensor(self.mod.TileEntry, (entry_count,), self._view(self.raw["g_TileEntryData"], self.mod.TileEntry, (entry_count,))),
            "g_SortedEntries": spy.InstanceTensor(self.mod.TileEntry, (entry_count,), self._view(self.raw["g_SortedEntryData"], self.mod.TileEntry, (entry_count,))),
        }
        self.sorted_tile_ids = self.raw["g_SortedEntryData"].view((entry_count,), (2,))

    def _vars(self, camera: dict[str, Any], splats: int, entries: int) -> dict[str, Any]:
        w, h = self._size
        tw = (w + 7) // 8
        th = (h + 7) // 8
        bg = self.background
        return {
            **self.scene,
            "g_Splats": self.scene["g_Params"],
            "g_ProjectionState": self.scene_views["g_ProjectionState"].tensor,
            "g_RasterState": self.scene_views["g_RasterState"].tensor,
            "g_ScanlineEntries": self.scanline_views["g_ScanlineEntries"].tensor,
            "g_ScanlineTileCounts": self.scanlines["g_ScanlineTileCounts"],
            "g_ScanlineTileOffsets": self.scanlines["g_ScanlineTileOffsets"],
            "g_TileEntries": self.entry_views["g_TileEntries"].tensor,
            "g_SortedEntries": self.entry_views["g_SortedEntries"].tensor,
            **self.tiles,
            **self.frame,
            "g_Camera": camera,
            "g_SplatCount": int(splats),
            "g_SortedEntryCount": int(entries),
            "g_TileGrid": spy.uint2(int(tw), int(th)),
            "g_Background": spy.float3(float(bg[0]), float(bg[1]), float(bg[2])),
            "g_RadiusScale": _RADIUS_SCALE,
            "g_AlphaCutoff": _ALPHA_CUTOFF,
            "g_TransmittanceThreshold": _TRANS_THRESHOLD,
        }

    def _prepare_splats(self, splats: torch.Tensor, camera: torch.Tensor, image_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, Any]]:
        order = torch.argsort(torch.linalg.norm(splats[0:3, :].mT - self._cam_pos(camera), dim=1), stable=True)
        sorted_splats = splats.index_select(1, order).contiguous()
        self._alloc_frame(image_size)
        self._alloc_scene(int(sorted_splats.shape[1]))
        self._alloc_scanlines(1)
        self._alloc_entries(1)
        self.scene["g_Params"].copy_from_torch(_pack_params(sorted_splats))
        self.device.sync_to_cuda()
        return order, sorted_splats, int(sorted_splats.shape[1]), _camera_dict(camera, image_size)

    def _dispatch_scanline_count(self, camera_vars: dict[str, Any], splat_count: int) -> int:
        enc = self.device.create_command_encoder()
        self.k_project_scanline_count.dispatch(thread_count=spy.uint3(splat_count, 1, 1), vars=self._vars(camera_vars, splat_count, 0), command_encoder=enc)
        self.device.submit_command_buffer(enc.finish())
        self.device.sync_to_device()
        scanline_counts = self.scene["g_ScanlineCounts"].to_torch()[:splat_count].to(dtype=torch.int64)
        scanline_offsets = torch.zeros_like(scanline_counts, dtype=torch.int64)
        if scanline_counts.numel() > 1:
            scanline_offsets[1:] = torch.cumsum(scanline_counts[:-1], 0)
        total_scanlines = int(scanline_counts.sum().item())
        self.scene["g_ScanlineOffsets"].copy_from_torch(scanline_offsets.to(dtype=torch.uint32))
        self._alloc_scanlines(max(total_scanlines, 1))
        self.device.sync_to_cuda()
        return total_scanlines

    def _dispatch_scanline_tile_counts(self, camera_vars: dict[str, Any], splat_count: int, total_scanlines: int) -> tuple[torch.Tensor, torch.Tensor]:
        if total_scanlines:
            enc = self.device.create_command_encoder()
            self.k_emit_scanlines.dispatch(thread_count=spy.uint3(splat_count, 1, 1), vars=self._vars(camera_vars, splat_count, total_scanlines), command_encoder=enc)
            self.k_emit_scanline_tile_counts.dispatch(thread_count=spy.uint3(total_scanlines, 1, 1), vars=self._vars(camera_vars, splat_count, total_scanlines), command_encoder=enc)
            self.device.submit_command_buffer(enc.finish())
            self.device.sync_to_device()
            per_scanline_tile_counts = self.scanlines["g_ScanlineTileCounts"].to_torch()[:total_scanlines].to(dtype=torch.int64)
            scanline_splats = self.scanlines["g_ScanlineEntryData"].to_torch()[:total_scanlines, 0].to(dtype=torch.int64)
        else:
            per_scanline_tile_counts = torch.zeros((0,), device=self.scene["g_Params"].to_torch().device, dtype=torch.int64)
            scanline_splats = per_scanline_tile_counts
        tile_counts = torch.zeros((splat_count,), device=self.scene["g_Params"].to_torch().device, dtype=torch.int64)
        if total_scanlines:
            tile_counts.scatter_add_(0, scanline_splats, per_scanline_tile_counts)
        self.scene["g_TileCounts"].copy_from_torch(tile_counts.to(dtype=torch.uint32))
        self.device.sync_to_cuda()
        return per_scanline_tile_counts, scanline_splats

    def project(self, splats: torch.Tensor, camera: torch.Tensor, image_size: tuple[int, int]) -> dict[str, torch.Tensor]:
        _check_cuda_tensor("splats", splats, _PARAM_COUNT)
        _check_cuda_tensor("camera_params", camera)
        self.background = (0.0, 0.0, 0.0)
        order, sorted_splats, splat_count, camera_vars = self._prepare_splats(splats, camera, image_size)
        total_scanlines = self._dispatch_scanline_count(camera_vars, splat_count)
        self._dispatch_scanline_tile_counts(camera_vars, splat_count, total_scanlines)

        projection = self._view(self.scene_views["g_ProjectionState"].tensor, spy.float4, (splat_count, 2)).to_torch()[:splat_count].clone()
        raster = self._view(self.scene_views["g_RasterState"].tensor, spy.float4, (splat_count, 4)).to_torch()[:splat_count].clone()
        tile_counts = self.scene["g_TileCounts"].to_torch()[:splat_count].clone()
        return {"order": order, "sorted_splats": sorted_splats, "projection": projection, "raster": raster, "tile_counts": tile_counts}

    def render(self, splats: torch.Tensor, camera: torch.Tensor, image_size: tuple[int, int], background: tuple[float, float, float]) -> torch.Tensor:
        self.background = background

        # Pass 1: project splats and count emitted scanlines.
        order, sorted_splats, splat_count, camera_vars = self._prepare_splats(splats, camera, image_size)
        total_scanlines = self._dispatch_scanline_count(camera_vars, splat_count)

        # Pass 2: emit scanlines and count the tiles covered by each scanline.
        per_scanline_tile_counts, _ = self._dispatch_scanline_tile_counts(camera_vars, splat_count, total_scanlines)
        scanline_tile_offsets = torch.zeros_like(per_scanline_tile_counts, dtype=torch.int64)
        if per_scanline_tile_counts.numel() > 1:
            scanline_tile_offsets[1:] = torch.cumsum(per_scanline_tile_counts[:-1], 0)
        total = int(per_scanline_tile_counts.sum().item())
        if total_scanlines:
            self.scanlines["g_ScanlineTileOffsets"].copy_from_torch(scanline_tile_offsets.to(dtype=torch.uint32))
        self._alloc_entries(max(total, 1))
        self.device.sync_to_cuda()

        if total_scanlines and total:
            enc = self.device.create_command_encoder()
            self.k_emit_tile_entries.dispatch(thread_count=spy.uint3(total_scanlines, 1, 1), vars=self._vars(camera_vars, splat_count, total_scanlines), command_encoder=enc)
            self.device.submit_command_buffer(enc.finish())
            self.device.sync_to_device()

        # Pass 3: sort emitted tile entries by tile ID and rebuild tile ranges.
        tile_data = self.raw["g_TileEntryData"].to_torch()[:total, :].to(dtype=torch.int64)
        tile_ids = tile_data[:, 0]
        tile_splats = tile_data[:, 1]
        perm = torch.argsort(tile_ids, stable=True)
        self.raw["g_SortedEntryData"].copy_from_torch(torch.stack((tile_ids.index_select(0, perm), tile_splats.index_select(0, perm)), dim=1).to(dtype=torch.uint32))

        tw = (image_size[0] + 7) // 8
        th = (image_size[1] + 7) // 8
        tile_count = tw * th
        ranges = torch.zeros((tile_count * 2,), device=tile_ids.device, dtype=torch.uint32)
        ranges[0::2] = 0xFFFFFFFF
        self.tiles["g_TileRanges"].copy_from_torch(ranges)
        self.device.sync_to_cuda()
        if total:
            enc = self.device.create_command_encoder()
            self.k_build_tile_ranges.dispatch(thread_count=spy.uint3(total, 1, 1), vars=self._vars(camera_vars, splat_count, total), command_encoder=enc)
            self.device.submit_command_buffer(enc.finish())
            self.device.sync_to_device()

        # Pass 4: rasterize sorted tile-local splat lists.
        enc = self.device.create_command_encoder()
        self.k_raster_fwd.dispatch(thread_count=spy.uint3(image_size[0], image_size[1], 1), vars=self._vars(camera_vars, splat_count, total), command_encoder=enc)
        self.device.submit_command_buffer(enc.finish())
        self.device.sync_to_device()

        self._last_order = order
        self._last_total = total
        self._last_alpha = torch.clamp(sorted_splats[13], _ALPHA_EPS, 1 - _ALPHA_EPS)
        return self.frame["g_Output"].to_torch().clone()

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        self.frame["g_OutputGrad"].copy_from_torch(grad_output.contiguous())
        self.device.sync_to_cuda()
        enc = self.device.create_command_encoder()
        self.scene["g_ParamGrads"].clear(command_encoder=enc)
        self.k_raster_bwd.dispatch(thread_count=spy.uint3(self._size[0], self._size[1], 1), vars=self._vars(_camera_dict(self._last_camera, self._size), int(self._last_order.shape[0]), self._last_total), command_encoder=enc)
        self.device.submit_command_buffer(enc.finish())
        self.device.sync_to_device()
        grads = self.scene["g_ParamGrads"].to_torch()[: int(self._last_order.shape[0]) * _PARAM_COUNT].clone().reshape(int(self._last_order.shape[0]), _PARAM_COUNT).mT.contiguous()
        grads[13] = grads[13] / (self._last_alpha * (1 - self._last_alpha))
        out = torch.empty_like(grads)
        out[:, self._last_order] = grads
        return out

    @staticmethod
    def _cam_pos(camera: torch.Tensor) -> torch.Tensor:
        q = camera[0:4]
        q = q / torch.clamp(torch.linalg.norm(q), min=1e-12)
        w, x, y, z = q
        rot = torch.tensor(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ],
            device=camera.device,
            dtype=camera.dtype,
        )
        return -rot.T @ camera[4:7]


class _RenderFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, splats: torch.Tensor, camera: torch.Tensor, image_size: tuple[int, int], background: tuple[float, float, float], context: SplattingContext) -> torch.Tensor:
        _check_cuda_tensor("splats", splats, _PARAM_COUNT)
        _check_cuda_tensor("camera_params", camera)
        ctx.context = context
        ctx.image_size = image_size
        ctx.background = background
        ctx.context._last_camera = camera.detach().clone()
        return context.render(splats, camera, image_size, background)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        return ctx.context.backward(grad_output), None, None, None, None


def render_gaussian_splats(
    splats: torch.Tensor,
    camera_params: torch.Tensor,
    image_size: tuple[int, int],
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    context: SplattingContext | None = None,
) -> torch.Tensor:
    return _RenderFn.apply(splats, camera_params, tuple(map(int, image_size)), tuple(map(float, background)), context or SplattingContext())
