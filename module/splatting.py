from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import slangpy as spy
import torch

_ROOT = Path(__file__).resolve().parent
_SHADERS = _ROOT / "shaders"
_PARAM_COUNT = 14
_PACKED_PARAM_COUNT = 4
_CAMERA_PARAM_COUNT = 15
_ALPHA_EPS = 1e-6
_ALPHA_CUTOFF = 1 / 255
_TRANS_THRESHOLD = 1e-4
_RADIUS_SCALE = 1.0
_MVEE_ITERS = 6
_MVEE_SAFETY = 1.0
_MVEE_PAD = 1.0
_MVEE_EPS = 1e-6


def _check_cuda_tensor(name: str, value: torch.Tensor, shape1: int | None = None) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must use torch.float32.")
    if not value.is_cuda:
        raise ValueError(f"{name} must be CUDA.")
    if value.ndim != 2 if shape1 is not None else value.ndim != 1:
        raise ValueError(f"{name} has invalid rank.")
    if shape1 is not None and value.shape[1] != shape1:
        raise ValueError(f"{name} must have shape [N, {shape1}].")
    if shape1 is None and value.shape[0] != _CAMERA_PARAM_COUNT:
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
    zeros = torch.zeros((splats.shape[0], 2), device=splats.device, dtype=splats.dtype)
    return torch.stack((splats[:, 0:4], splats[:, 4:8], splats[:, 8:12], torch.cat((splats[:, 12:14], zeros), dim=1)), dim=1)


@dataclass
class SplattingContext:
    device: spy.Device | None = None

    def __post_init__(self) -> None:
        self.device = self.device or spy.create_torch_device(type=spy.DeviceType.cuda, include_paths=[_SHADERS])
        self.mod = spy.Module.load_from_file(self.device, str(_SHADERS / "module.slang"))
        self.k_project_count = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csProjectCount"]))
        self.k_project_fill = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csProjectFill"]))
        self.k_raster_fwd = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csRasterForward"]))
        self.k_raster_bwd = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csRasterBackward"]))
        self._size = (0, 0)

    def _view(self, tensor: spy.Tensor, dtype: Any, shape: tuple[int, ...]) -> spy.Tensor:
        if isinstance(dtype, type):
            dtype = spy.Tensor.empty(self.device, shape=(1,), dtype=dtype).dtype
        return spy.Tensor(tensor.storage, getattr(dtype, "struct", dtype), shape)

    def _alloc(self, shape: tuple[int, int], splats: int, entries: int) -> None:
        width, height = shape
        splat_count = max(splats, 1)
        entry_count = max(entries, 1)
        if self._size != shape:
            self.frame = {
                "g_Output": spy.Tensor.empty(self.device, shape=(height, width), dtype=spy.float4),
                "g_OutputGrad": spy.Tensor.empty(self.device, shape=(height, width), dtype=spy.float4),
                "g_ForwardState": spy.Tensor.empty(self.device, shape=(height, width), dtype=spy.float4),
                "g_ForwardEnd": spy.Tensor.empty(self.device, shape=(height, width), dtype="uint"),
            }
            tw, th = (width + 7) // 8, (height + 7) // 8
            self.tiles = {"g_TileRanges": spy.Tensor.empty(self.device, shape=(tw * th,), dtype=spy.uint2)}
            self._size = shape
        self.scene = {
            "g_Params": spy.Tensor.empty(self.device, shape=(splat_count, _PACKED_PARAM_COUNT), dtype=spy.float4),
            "g_TileCounts": spy.Tensor.empty(self.device, shape=(splat_count,), dtype="uint"),
            "g_TileOffsets": spy.Tensor.empty(self.device, shape=(splat_count,), dtype="uint"),
            "g_ParamGrads": spy.Tensor.empty(self.device, shape=(splat_count, _PARAM_COUNT), dtype=float),
        }
        self.raw = {
            "g_TileEntryData": spy.Tensor.empty(self.device, shape=(entry_count, 2), dtype="uint"),
            "g_SortedEntryData": spy.Tensor.empty(self.device, shape=(entry_count, 2), dtype="uint"),
        }
        self.views = {
            "g_Splats": spy.InstanceTensor(self.mod.PackedSplat, (splat_count,), self._view(self.scene["g_Params"], self.mod.PackedSplat, (splat_count,))),
            "g_ProjectionState": spy.InstanceTensor(self.mod.ProjectionState, (splat_count,)),
            "g_RasterState": spy.InstanceTensor(self.mod.RasterState, (splat_count,)),
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
            "g_Splats": self.views["g_Splats"].tensor,
            "g_ProjectionState": self.views["g_ProjectionState"].tensor,
            "g_RasterState": self.views["g_RasterState"].tensor,
            "g_TileEntries": self.views["g_TileEntries"].tensor,
            "g_SortedEntries": self.views["g_SortedEntries"].tensor,
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
            "g_Sampled5Iters": int(_MVEE_ITERS),
            "g_Sampled5SafetyScale": _MVEE_SAFETY,
            "g_Sampled5RadiusPadPx": _MVEE_PAD,
            "g_Sampled5Eps": _MVEE_EPS,
        }

    def render(self, splats: torch.Tensor, camera: torch.Tensor, image_size: tuple[int, int], background: tuple[float, float, float]) -> torch.Tensor:
        self.background = background
        # Sort once by camera distance so the later stable tile sort preserves front-to-back splat order inside each tile.
        order = torch.argsort(torch.linalg.norm(splats[:, 0:3] - self._cam_pos(camera), dim=1), stable=True)
        sorted_splats = splats.index_select(0, order).contiguous()
        packed_splats = _pack_params(sorted_splats)
        self._alloc(image_size, int(sorted_splats.shape[0]), 1)
        self.scene["g_Params"].copy_from_torch(packed_splats)
        self.device.sync_to_cuda()
        camera_vars = _camera_dict(camera, image_size)
        enc = self.device.create_command_encoder()
        self.k_project_count.dispatch(thread_count=spy.uint3(int(sorted_splats.shape[0]), 1, 1), vars=self._vars(camera_vars, int(sorted_splats.shape[0]), 0), command_encoder=enc)
        self.device.submit_command_buffer(enc.finish())
        self.device.sync_to_device()
        counts = self.scene["g_TileCounts"].to_torch()[: sorted_splats.shape[0]].to(dtype=torch.int64)
        total = int(counts.sum().item())
        self._alloc(image_size, int(sorted_splats.shape[0]), max(total, 1))
        self.scene["g_Params"].copy_from_torch(packed_splats)
        self.device.sync_to_cuda()
        enc = self.device.create_command_encoder()
        self.k_project_count.dispatch(thread_count=spy.uint3(int(sorted_splats.shape[0]), 1, 1), vars=self._vars(camera_vars, int(sorted_splats.shape[0]), 0), command_encoder=enc)
        self.device.submit_command_buffer(enc.finish())
        self.device.sync_to_device()
        offsets = torch.zeros_like(counts, dtype=torch.int64)
        if counts.numel() > 1:
            offsets[1:] = torch.cumsum(counts[:-1], 0)
        self.scene["g_TileOffsets"].copy_from_torch(offsets.to(dtype=torch.uint32))
        self.device.sync_to_cuda()
        # Expand each projected splat into its covered tiles, then stable-sort only by tile id.
        enc = self.device.create_command_encoder()
        self.k_project_fill.dispatch(thread_count=spy.uint3(int(sorted_splats.shape[0]), 1, 1), vars=self._vars(camera_vars, int(sorted_splats.shape[0]), total), command_encoder=enc)
        self.device.submit_command_buffer(enc.finish())
        self.device.sync_to_device()
        tile_data = self.raw["g_TileEntryData"].to_torch()[:total, :].to(dtype=torch.int64)
        tile_ids = tile_data[:, 0]
        tile_splats = tile_data[:, 1]
        perm = torch.argsort(tile_ids, stable=True)
        self.raw["g_SortedEntryData"].copy_from_torch(torch.stack((tile_ids.index_select(0, perm), tile_splats.index_select(0, perm)), dim=1).to(dtype=torch.uint32))
        # Tile spans are built with vectorized CUDA torch ops so this stage stays off the CPU.
        tw = (image_size[0] + 7) // 8
        th = (image_size[1] + 7) // 8
        ranges = torch.zeros((tw * th, 2), device=tile_ids.device, dtype=torch.int64)
        ranges[:, 0] = 0xFFFFFFFF
        if total:
            sorted_ids = tile_ids.index_select(0, perm)
            starts = torch.nonzero(torch.cat((torch.ones(1, device=sorted_ids.device, dtype=torch.bool), sorted_ids[1:] != sorted_ids[:-1])), as_tuple=False).reshape(-1)
            ends = torch.cat((starts[1:], torch.tensor([total], device=sorted_ids.device, dtype=starts.dtype)))
            tiles = sorted_ids.index_select(0, starts).to(dtype=torch.long)
            ranges[tiles, 0] = starts
            ranges[tiles, 1] = ends
        self.tiles["g_TileRanges"].copy_from_torch(ranges.to(dtype=torch.uint32))
        # Raster uses the sorted tile spans plus the cached projected splat state from the projection pass.
        enc = self.device.create_command_encoder()
        self.k_raster_fwd.dispatch(thread_count=spy.uint3(image_size[0], image_size[1], 1), vars=self._vars(camera_vars, int(sorted_splats.shape[0]), total), command_encoder=enc)
        self.device.submit_command_buffer(enc.finish())
        self.device.sync_to_device()
        self._last_order = order
        self._last_total = total
        self._last_alpha = torch.clamp(sorted_splats[:, 13], _ALPHA_EPS, 1 - _ALPHA_EPS)
        return self.frame["g_Output"].to_torch().clone()

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        # Backward replays the cached forward state and writes gradients in sorted-splat order, then unsorts once.
        self.scene["g_ParamGrads"].to_torch().zero_()
        self.frame["g_OutputGrad"].copy_from_torch(grad_output.contiguous())
        self.device.sync_to_cuda()
        enc = self.device.create_command_encoder()
        self.k_raster_bwd.dispatch(thread_count=spy.uint3(self._size[0], self._size[1], 1), vars=self._vars(_camera_dict(self._last_camera, self._size), int(self._last_order.shape[0]), self._last_total), command_encoder=enc)
        self.device.submit_command_buffer(enc.finish())
        self.device.sync_to_device()
        grads = self.scene["g_ParamGrads"].to_torch()[: self._last_order.shape[0], :].clone()
        grads[:, 13] = grads[:, 13] / (self._last_alpha * (1 - self._last_alpha))
        out = torch.empty_like(grads)
        out[self._last_order] = grads
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
