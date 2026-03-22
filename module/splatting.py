from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import slangpy as spy

from utility.utility import GpuUtility

_ROOT = Path(__file__).resolve().parent
_SHADERS = _ROOT / "shaders"
_PARAM_COUNT = 14
_ALPHA_CUTOFF = 1 / 255
_TRANS_THRESHOLD = 0.005
_RADIUS_SCALE = 1.0


@dataclass
class SplattingContext:
    device: spy.Device | None = None

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
        self.k_build_tile_ranges = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csBuildTileRanges"]))
        self.k_raster_fwd = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csRasterForward"]))
        self.k_raster_bwd = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csRasterBackward"]))
        self.util = GpuUtility(self.device)
        self._size = (0, 0)

    def _view(self, tensor: spy.Tensor, dtype: Any, shape: tuple[int, ...]) -> spy.Tensor:
        if isinstance(dtype, type):
            dtype = spy.Tensor.empty(self.device, shape=(1,), dtype=dtype).dtype
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
        self.scene = {"g_Params": spy.Tensor.empty(self.device, shape=(_PARAM_COUNT * splat_count,), dtype=float), "g_ScanlineCounts": spy.Tensor.empty(self.device, shape=(splat_count,), dtype="uint"), "g_ScanlineOffsets": spy.Tensor.empty(self.device, shape=(splat_count,), dtype="uint"), "g_TileCounts": spy.Tensor.empty(self.device, shape=(splat_count,), dtype="uint"), "g_ParamGrads": spy.Tensor.empty(self.device, shape=(splat_count * _PARAM_COUNT,), dtype=float)}
        self.scene_views = {"g_ProjectionState": spy.InstanceTensor(self.mod.ProjectionState, (splat_count,)), "g_RasterState": spy.InstanceTensor(self.mod.Gaussian3D, (splat_count,))}

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
        self.scanlines = {"g_ScanlineEntryData": spy.Tensor.empty(self.device, shape=(scanline_count, 3), dtype="uint"), "g_ScanlineTileCounts": spy.Tensor.empty(self.device, shape=(scanline_count,), dtype="uint"), "g_ScanlineTileOffsets": spy.Tensor.empty(self.device, shape=(scanline_count,), dtype="uint")}
        self.scanline_views = {"g_ScanlineEntries": spy.InstanceTensor(self.mod.ScanlineEntry, (scanline_count,), self._view(self.scanlines["g_ScanlineEntryData"], self.mod.ScanlineEntry, (scanline_count,)))}

    def _alloc_entries(self, entry_count: int) -> None:
        entry_count = max(entry_count, 1)
        self.raw = {"g_TileEntryData": spy.Tensor.empty(self.device, shape=(entry_count, 2), dtype="uint"), "g_SortedEntryData": spy.Tensor.empty(self.device, shape=(entry_count, 2), dtype="uint")}
        self.entry_views = {"g_TileEntries": spy.InstanceTensor(self.mod.TileEntry, (entry_count,), self._view(self.raw["g_TileEntryData"], self.mod.TileEntry, (entry_count,))), "g_SortedEntries": spy.InstanceTensor(self.mod.TileEntry, (entry_count,), self._view(self.raw["g_SortedEntryData"], self.mod.TileEntry, (entry_count,)))}
        self.tile_keys, self.tile_values = self.raw["g_TileEntryData"].view((entry_count,), (2,), 0), self.raw["g_TileEntryData"].view((entry_count,), (2,), 1)
        self.sorted_tile_keys, self.sorted_tile_values = self.raw["g_SortedEntryData"].view((entry_count,), (2,), 0), self.raw["g_SortedEntryData"].view((entry_count,), (2,), 1)
        self._sorted_entries_tensor = self.entry_views["g_SortedEntries"].tensor

    def _alloc_radix(self, count: int) -> None:
        count = max(count, 1)
        hist = self.util.radix_histogram_elements(count)
        if getattr(self, "_radix_size", 0) >= count and getattr(self, "_hist_size", 0) >= hist:
            return
        self.radix = {"g_Histogram": spy.Tensor.empty(self.device, shape=(hist,), dtype="uint"), "g_HistogramPrefix": spy.Tensor.empty(self.device, shape=(hist,), dtype="uint")}
        self._radix_size = count
        self._hist_size = hist

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
            "g_TileEntries": self.entry_views["g_TileEntries"].tensor,
            "g_SortedEntries": self._sorted_entries_tensor,
            **self.tiles,
            **self.frame,
            "g_Camera": camera,
            "g_SplatCount": int(splats),
            "g_SortedEntryCount": int(entries),
            "g_TileGrid": spy.uint2((w + 7) // 8, (h + 7) // 8),
            "g_Background": spy.float3(*map(float, self.background)),
            "g_RadiusScale": _RADIUS_SCALE,
            "g_AlphaCutoff": _ALPHA_CUTOFF,
            "g_TransmittanceThreshold": _TRANS_THRESHOLD,
        }

    def _dispatch_scanline_count(self, camera: dict[str, Any], splat_count: int) -> int:
        self._alloc_prefix(splat_count)
        self._alloc_splat_sort(splat_count)
        self._alloc_radix(splat_count)
        self._alloc_prefix(self.util.radix_histogram_elements(splat_count))
        enc = self.device.create_command_encoder()
        sort_vars = self._vars(camera, splat_count, 0)
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
        self.k_project_scanline_count.dispatch(thread_count=spy.uint3(splat_count, 1, 1), vars=self._vars(camera, splat_count, 0), command_encoder=enc)
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
        vars = self._vars(camera, splat_count, total_scanlines)
        self.k_emit_scanlines.dispatch(thread_count=spy.uint3(splat_count, 1, 1), vars=vars, command_encoder=enc)
        self.k_emit_scanline_tile_counts.dispatch(thread_count=spy.uint3(total_scanlines, 1, 1), vars=vars, command_encoder=enc)
        self.device.submit_command_buffer(enc.finish())
        self.device.sync_to_device()

    def project(self, camera: dict[str, Any], splat_count: int) -> int:
        total = self._dispatch_scanline_count(camera, splat_count)
        self._emit_scanlines_and_tile_counts(camera, splat_count, total)
        return total

    def render(self, camera: dict[str, Any], splat_count: int) -> spy.Tensor:
        total_scanlines = SplattingContext.project(self, camera, splat_count)
        total = 0
        if total_scanlines:
            self._alloc_prefix(total_scanlines)
            enc = self.device.create_command_encoder()
            self.util.prefix_sum_uint32(enc, self.scanlines["g_ScanlineTileCounts"], self.scanlines["g_ScanlineTileOffsets"], self.prefix["g_BlockSums"], self.prefix["g_BlockOffsets"], self.prefix["g_Total"], total_scanlines, True)
            self.device.submit_command_buffer(enc.finish())
            self.device.sync_to_device()
            total = self._read_uint(self.prefix["g_Total"])
        self._alloc_entries(total)
        self.device.sync_to_cuda()
        if total_scanlines and total:
            enc = self.device.create_command_encoder()
            self.k_emit_tile_entries.dispatch(thread_count=spy.uint3(total_scanlines, 1, 1), vars=self._vars(camera, splat_count, total_scanlines), command_encoder=enc)
            self.device.submit_command_buffer(enc.finish())
            self.device.sync_to_device()
        tw, th = (self._size[0] + 7) // 8, (self._size[1] + 7) // 8
        ranges = np.zeros((tw * th * 2,), dtype=np.uint32)
        ranges[0::2] = 0xFFFFFFFF
        self.tiles["g_TileRanges"].copy_from_numpy(ranges)
        self.device.sync_to_cuda()
        self._sorted_entries_tensor = self.entry_views["g_SortedEntries"].tensor
        if total:
            self._alloc_radix(total)
            self._alloc_prefix(self.util.radix_histogram_elements(total))
            enc = self.device.create_command_encoder()
            out_buffer = self.util.radix_sort_uint32(enc, self.tile_keys, self.tile_values, self.sorted_tile_keys, self.sorted_tile_values, self.radix["g_Histogram"], self.radix["g_HistogramPrefix"], self.prefix["g_BlockSums"], self.prefix["g_BlockOffsets"], self.prefix["g_Total"], total, 0, max(1, (tw * th - 1).bit_length()))
            self._sorted_entries_tensor = self.entry_views["g_SortedEntries"].tensor if out_buffer else self.entry_views["g_TileEntries"].tensor
            self.k_build_tile_ranges.dispatch(thread_count=spy.uint3(total, 1, 1), vars=self._vars(camera, splat_count, total), command_encoder=enc)
            self.device.submit_command_buffer(enc.finish())
            self.device.sync_to_device()
        enc = self.device.create_command_encoder()
        self.k_raster_fwd.dispatch(thread_count=spy.uint3(*self._size, 1), vars=self._vars(camera, splat_count, total), command_encoder=enc)
        self.device.submit_command_buffer(enc.finish())
        self.device.sync_to_device()
        self._last_total = total
        return self.frame["g_Output"]

    def backward(self, camera: dict[str, Any], splat_count: int) -> spy.Tensor:
        enc = self.device.create_command_encoder()
        self.scene["g_ParamGrads"].clear(command_encoder=enc)
        self.k_raster_bwd.dispatch(thread_count=spy.uint3(*self._size, 1), vars=self._vars(camera, splat_count, self._last_total), command_encoder=enc)
        self.device.submit_command_buffer(enc.finish())
        self.device.sync_to_device()
        return self.scene["g_ParamGrads"]
