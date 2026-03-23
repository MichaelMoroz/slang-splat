from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import slangpy as spy

from utility.debug import debug_group
from utility.utility import GpuUtility

_ROOT = Path(__file__).resolve().parent
_SHADERS = _ROOT / "shaders"
_PARAM_COUNT = 14
_ALPHA_CUTOFF = 0.02
_TRANS_THRESHOLD = 0.005
_RADIUS_SCALE = 1.0
_DEBUG_COLOR = spy.float3(0.91, 0.53, 0.18)
_DEBUG_MODE_NORMAL = 0
_DEBUG_MODE_PROCESSED_COUNT = 1


@dataclass
class SplattingContext:
    device: spy.Device | None = None
    radius_scale: float = 1.0
    max_anisotropy: float = 12.0
    alpha_cutoff: float = 0.02
    trans_threshold: float = 0.005
    debug_mode: int = _DEBUG_MODE_NORMAL

    def __post_init__(self) -> None:
        self.device = self.device or spy.create_device(type=spy.DeviceType.cuda, include_paths=[_SHADERS], enable_cuda_interop=False)
        self._init_resources()

    def _init_resources(self) -> None:
        self.mod = spy.Module.load_from_file(self.device, str(_SHADERS / "module.slang"))
        self.k_generate_distance_sort_keys = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csGenerateDistanceSortKeys"]))
        self.k_project_scanline_count = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csProjectScanlineCount"]))
        self.k_emit_scanlines = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csEmitScanlines"]))
        self.k_emit_scanline_tile_counts = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csEmitScanlineTileCounts"]))
        self.k_emit_tile_entries = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csEmitTileEntries"]))
        self.k_clear_tile_ranges = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csClearTileRanges"]))
        self.k_build_tile_ranges = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csBuildTileRanges"]))
        self.k_raster_fwd = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csRasterForward"]))
        self.k_raster_bwd = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csRasterBackward"]))
        self.util = GpuUtility(self.device)
        self._float4_dtype = spy.Tensor.empty(self.device, shape=(1,), dtype=spy.float4).dtype
        self._projection_dtype = self.mod.ProjectionState
        self._gaussian_dtype = self.mod.Gaussian3D
        self._scanline_dtype = self.mod.ScanlineEntry
        self._size = (0, 0)

    def _shared_usage(self) -> spy.BufferUsage:
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
        if getattr(self.device, "supports_cuda_interop", False):
            usage |= spy.BufferUsage.shared
        return usage

    def _view(self, tensor: spy.Tensor, dtype: Any, shape: tuple[int, ...]) -> spy.Tensor:
        if dtype is spy.float4:
            dtype = self._float4_dtype
        return spy.Tensor(tensor.storage, getattr(dtype, "struct", dtype), shape)

    def _read_uint(self, tensor: spy.Tensor) -> int:
        return int(np.asarray(tensor.to_numpy()).reshape(-1)[0])

    def _alloc_frame(self, shape: tuple[int, int]) -> None:
        if self._size == shape:
            return
        width, height = shape
        self.frame = {"g_Output": spy.Tensor.empty(self.device, shape=(height, width), dtype=spy.float4), "g_OutputGrad": spy.Tensor.empty(self.device, shape=(height, width), dtype=spy.float4), "g_ForwardState": spy.Tensor.empty(self.device, shape=(height, width), dtype=spy.float4), "g_ForwardEnd": spy.Tensor.empty(self.device, shape=(height, width), dtype="uint")}
        tw, th = (width + 7) // 8, (height + 7) // 8
        self.tiles = {"g_TileRanges": spy.Tensor.empty(self.device, shape=(tw * th * 2,), dtype="uint")}
        self._size = shape

    def _alloc_scene(self, splat_count: int) -> None:
        splat_count = max(splat_count, 1)
        if getattr(self, "_scene_capacity", 0) >= splat_count:
            return
        self.scene = {
            "g_Params": spy.Tensor.empty(self.device, shape=(_PARAM_COUNT * splat_count,), dtype=float, usage=self._shared_usage()),
            "g_ScanlineCounts": spy.Tensor.empty(self.device, shape=(splat_count,), dtype="uint"),
            "g_ScanlineOffsets": spy.Tensor.empty(self.device, shape=(splat_count,), dtype="uint"),
            "g_TileCounts": spy.Tensor.empty(self.device, shape=(splat_count,), dtype="uint"),
            "g_ParamGrads": spy.Tensor.empty(self.device, shape=(splat_count * _PARAM_COUNT,), dtype=float),
            "g_ProjectionStateData": spy.Tensor.empty(self.device, shape=(splat_count, 2), dtype=spy.float4),
            "g_RasterStateData": spy.Tensor.empty(self.device, shape=(splat_count, 4), dtype=spy.float4),
        }
        self.scene_views = {
            "g_ProjectionState": spy.InstanceTensor(
                self._projection_dtype,
                (splat_count,),
                self._view(self.scene["g_ProjectionStateData"], self._projection_dtype, (splat_count,)),
            ),
            "g_RasterState": spy.InstanceTensor(
                self._gaussian_dtype,
                (splat_count,),
                self._view(self.scene["g_RasterStateData"], self._gaussian_dtype, (splat_count,)),
            ),
        }
        self._scene_capacity = splat_count

    def _alloc_splat_sort(self, splat_count: int) -> None:
        splat_count = max(splat_count, 1)
        if getattr(self, "_sort_size", 0) >= splat_count:
            return
        self.sort = {
            "g_DistanceKeys": spy.Tensor.empty(self.device, shape=(splat_count,), dtype="uint"),
            "g_SortedDistanceKeys": spy.Tensor.empty(self.device, shape=(splat_count,), dtype="uint"),
            "g_DistanceSortOrder": spy.Tensor.empty(self.device, shape=(splat_count,), dtype="uint"),
            "g_SortedDistanceSortOrder": spy.Tensor.empty(self.device, shape=(splat_count,), dtype="uint"),
        }
        self._sort_size = splat_count
        self._sorted_splat_order_tensor = self.sort["g_DistanceSortOrder"]

    def _alloc_prefix(self, count: int) -> None:
        size = max(self.util.prefix_scratch_elements(count), 1)
        if getattr(self, "_prefix_size", 0) >= size:
            return
        self.prefix = {"g_BlockSums": spy.Tensor.empty(self.device, shape=(size,), dtype="uint"), "g_BlockOffsets": spy.Tensor.empty(self.device, shape=(size,), dtype="uint"), "g_Total": spy.Tensor.empty(self.device, shape=(1,), dtype="uint")}
        self._prefix_size = size

    def _alloc_scanlines(self, scanline_count: int) -> None:
        scanline_count = max(scanline_count, 1)
        if getattr(self, "_scanline_capacity", 0) >= scanline_count:
            return
        self.scanlines = {"g_ScanlineEntryData": spy.Tensor.empty(self.device, shape=(scanline_count, 3), dtype="uint"), "g_ScanlineTileCounts": spy.Tensor.empty(self.device, shape=(scanline_count,), dtype="uint"), "g_ScanlineTileOffsets": spy.Tensor.empty(self.device, shape=(scanline_count,), dtype="uint")}
        self.scanline_views = {"g_ScanlineEntries": spy.InstanceTensor(self._scanline_dtype, (scanline_count,), self._view(self.scanlines["g_ScanlineEntryData"], self._scanline_dtype, (scanline_count,)))}
        self._scanline_capacity = scanline_count

    def _alloc_entries(self, entry_count: int) -> None:
        entry_count = max(entry_count, 1)
        if getattr(self, "_entry_capacity", 0) >= entry_count:
            return
        self.raw = {"g_TileEntryData": spy.Tensor.empty(self.device, shape=(entry_count * 2,), dtype="uint"), "g_SortedEntryData": spy.Tensor.empty(self.device, shape=(entry_count * 2,), dtype="uint")}
        self.tile_keys, self.tile_values = self.raw["g_TileEntryData"].view((entry_count,), (2,), 0), self.raw["g_TileEntryData"].view((entry_count,), (2,), 1)
        self.sorted_tile_keys, self.sorted_tile_values = self.raw["g_SortedEntryData"].view((entry_count,), (2,), 0), self.raw["g_SortedEntryData"].view((entry_count,), (2,), 1)
        self._sorted_entries_tensor = self.raw["g_SortedEntryData"]
        self._entry_capacity = entry_count

    def _alloc_radix(self, count: int) -> None:
        count = max(count, 1)
        hist = self.util.radix_histogram_elements(count)
        prefix = self.util.radix_prefix_elements(count)
        if getattr(self, "_radix_size", 0) >= count and getattr(self, "_hist_size", 0) >= hist and getattr(self, "_hist_prefix_size", 0) >= prefix:
            return
        self.radix = {"g_Histogram": spy.Tensor.empty(self.device, shape=(hist,), dtype="uint"), "g_HistogramPrefix": spy.Tensor.empty(self.device, shape=(prefix,), dtype="uint")}
        self._radix_size = count
        self._hist_size = hist
        self._hist_prefix_size = prefix

    def prepare(self, splat_count: int, image_size: tuple[int, int], background: tuple[float, float, float]) -> None:
        self.background = background
        self._alloc_frame(image_size)
        self._alloc_scene(splat_count)
        self._alloc_splat_sort(splat_count)
        self._alloc_scanlines(1)
        self._alloc_entries(1)

    def _vars(self, camera: dict[str, Any], splats: int, entries: int) -> dict[str, Any]:
        w, h = self._size
        return {
            **self.scene,
            "g_SplatOrder": self._sorted_splat_order_tensor,
            **self.sort,
            "g_Splats": self.scene["g_Params"],
            "g_ProjectionState": self.scene_views["g_ProjectionState"].tensor,
            "g_RasterState": self.scene_views["g_RasterState"].tensor,
            "g_ScanlineEntries": self.scanline_views["g_ScanlineEntries"].tensor,
            "g_ScanlineTileCounts": self.scanlines["g_ScanlineTileCounts"],
            "g_ScanlineTileOffsets": self.scanlines["g_ScanlineTileOffsets"],
            "g_TileEntries": self.raw["g_TileEntryData"],
            "g_SortedEntries": self._sorted_entries_tensor,
            **self.tiles,
            **self.frame,
            "g_Camera": camera,
            "g_SplatCount": int(splats),
            "g_SortedEntryCount": int(entries),
            "g_TileGrid": spy.uint2((w + 7) // 8, (h + 7) // 8),
            "g_Background": spy.float3(*map(float, self.background)),
            "g_RadiusScale": float(self.radius_scale),
            "g_MaxAnisotropy": float(self.max_anisotropy),
            "g_AlphaCutoff": float(self.alpha_cutoff),
            "g_TransmittanceThreshold": float(self.trans_threshold),
            "g_DebugMode": int(self.debug_mode),
            "g_DebugMaxSplatSteps": int(entries),
        }

    def _dispatch_scanline_count(self, camera: dict[str, Any], splat_count: int) -> int:
        self._alloc_prefix(splat_count)
        self._alloc_splat_sort(splat_count)
        self._alloc_radix(splat_count)
        self._alloc_prefix(self.util.radix_histogram_elements(splat_count))
        if splat_count == 0:
            self._sorted_splat_order_tensor = self.sort["g_DistanceSortOrder"]
            return 0
        enc = self.device.create_command_encoder()
        with debug_group(enc, "renderer.project_scanlines", _DEBUG_COLOR):
            sort_vars = self._vars(camera, splat_count, 0)
            with debug_group(enc, "renderer.distance_sort", _DEBUG_COLOR):
                self.k_generate_distance_sort_keys.dispatch(thread_count=spy.uint3(splat_count, 1, 1), vars=sort_vars, command_encoder=enc)
                out_buffer = self.util.radix_sort_uint32(
                    enc,
                    self.sort["g_DistanceKeys"],
                    self.sort["g_DistanceSortOrder"],
                    self.sort["g_SortedDistanceKeys"],
                    self.sort["g_SortedDistanceSortOrder"],
                    self.radix["g_Histogram"],
                    self.radix["g_HistogramPrefix"],
                    self.prefix["g_BlockSums"],
                    self.prefix["g_BlockOffsets"],
                    self.prefix["g_Total"],
                    splat_count,
                    0,
                    32,
                )
            self._sorted_splat_order_tensor = self.sort["g_SortedDistanceSortOrder"] if out_buffer else self.sort["g_DistanceSortOrder"]
            with debug_group(enc, "renderer.project", _DEBUG_COLOR):
                self.k_project_scanline_count.dispatch(thread_count=spy.uint3(splat_count, 1, 1), vars=self._vars(camera, splat_count, 0), command_encoder=enc)
            with debug_group(enc, "renderer.scanline_prefix", _DEBUG_COLOR):
                self.util.prefix_sum_uint32(enc, self.scene["g_ScanlineCounts"], self.scene["g_ScanlineOffsets"], self.prefix["g_BlockSums"], self.prefix["g_BlockOffsets"], self.prefix["g_Total"], splat_count, True)
        self.device.submit_command_buffer(enc.finish())
        self.device.sync_to_device()
        total = self._read_uint(self.prefix["g_Total"])
        self._alloc_scanlines(total)
        self.device.sync_to_cuda()
        return total

    def _emit_scanlines_and_tile_counts(self, camera: dict[str, Any], splat_count: int, total_scanlines: int) -> None:
        if total_scanlines == 0:
            return
        enc = self.device.create_command_encoder()
        with debug_group(enc, "renderer.emit_scanlines", _DEBUG_COLOR):
            vars = self._vars(camera, splat_count, total_scanlines)
            self.k_emit_scanlines.dispatch(thread_count=spy.uint3(splat_count, 1, 1), vars=vars, command_encoder=enc)
            self.k_emit_scanline_tile_counts.dispatch(thread_count=spy.uint3(total_scanlines, 1, 1), vars=vars, command_encoder=enc)
        self.device.submit_command_buffer(enc.finish())
        self.device.sync_to_device()

    def project(self, camera: dict[str, Any], splat_count: int) -> int:
        if splat_count == 0:
            self._dispatch_scanline_count(camera, splat_count)
            return 0
        total = self._dispatch_scanline_count(camera, splat_count)
        self._emit_scanlines_and_tile_counts(camera, splat_count, total)
        return total

    def render(self, camera: dict[str, Any], splat_count: int, command_encoder: spy.CommandEncoder | None = None) -> spy.Tensor:
        total_scanlines = SplattingContext.project(self, camera, splat_count)
        total = 0
        if total_scanlines:
            self._alloc_prefix(total_scanlines)
            prefix_encoder = self.device.create_command_encoder()
            with debug_group(prefix_encoder, "renderer.tile_prefix", _DEBUG_COLOR):
                self.util.prefix_sum_uint32(prefix_encoder, self.scanlines["g_ScanlineTileCounts"], self.scanlines["g_ScanlineTileOffsets"], self.prefix["g_BlockSums"], self.prefix["g_BlockOffsets"], self.prefix["g_Total"], total_scanlines, True)
            self.device.submit_command_buffer(prefix_encoder.finish())
            self.device.sync_to_device()
            total = self._read_uint(self.prefix["g_Total"])
        self._alloc_entries(total)
        self.device.sync_to_cuda()
        tw, th = (self._size[0] + 7) // 8, (self._size[1] + 7) // 8
        tile_count = tw * th
        enc = command_encoder or self.device.create_command_encoder()
        with debug_group(enc, "renderer.render", _DEBUG_COLOR):
            self.k_clear_tile_ranges.dispatch(thread_count=spy.uint3(tile_count, 1, 1), vars=self._vars(camera, splat_count, total), command_encoder=enc)
            if total_scanlines and total:
                with debug_group(enc, "renderer.emit_tiles", _DEBUG_COLOR):
                    self.k_emit_tile_entries.dispatch(thread_count=spy.uint3(total_scanlines, 1, 1), vars=self._vars(camera, splat_count, total_scanlines), command_encoder=enc)
            self._sorted_entries_tensor = self.raw["g_SortedEntryData"]
            if total:
                self._alloc_radix(total)
                self._alloc_prefix(self.util.radix_histogram_elements(total))
                with debug_group(enc, "renderer.sort_tiles", _DEBUG_COLOR):
                    out_buffer = self.util.radix_sort_uint32(enc, self.tile_keys, self.tile_values, self.sorted_tile_keys, self.sorted_tile_values, self.radix["g_Histogram"], self.radix["g_HistogramPrefix"], self.prefix["g_BlockSums"], self.prefix["g_BlockOffsets"], self.prefix["g_Total"], total, 0, max(1, (tw * th - 1).bit_length()))
                self._sorted_entries_tensor = self.raw["g_SortedEntryData"] if out_buffer else self.raw["g_TileEntryData"]
                with debug_group(enc, "renderer.build_tile_ranges", _DEBUG_COLOR):
                    self.k_build_tile_ranges.dispatch(thread_count=spy.uint3(total, 1, 1), vars=self._vars(camera, splat_count, total), command_encoder=enc)
            with debug_group(enc, "renderer.raster_forward", _DEBUG_COLOR):
                self.k_raster_fwd.dispatch(thread_count=spy.uint3(*self._size, 1), vars=self._vars(camera, splat_count, total), command_encoder=enc)
        if command_encoder is None:
            self.device.submit_command_buffer(enc.finish())
            self.device.sync_to_device()
        self._last_total = total
        return self.frame["g_Output"]

    def backward(self, camera: dict[str, Any], splat_count: int, command_encoder: spy.CommandEncoder | None = None) -> spy.Tensor:
        enc = command_encoder or self.device.create_command_encoder()
        with debug_group(enc, "renderer.backward", _DEBUG_COLOR):
            self.scene["g_ParamGrads"].clear(command_encoder=enc)
            self.k_raster_bwd.dispatch(thread_count=spy.uint3(*self._size, 1), vars=self._vars(camera, splat_count, self._last_total), command_encoder=enc)
        if command_encoder is None:
            self.device.submit_command_buffer(enc.finish())
            self.device.sync_to_device()
        return self.scene["g_ParamGrads"]
