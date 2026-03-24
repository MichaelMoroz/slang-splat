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
_DEBUG_MODE_DEPTH_MEAN = 2
_DEBUG_MODE_DEPTH_STD = 3
_DEBUG_MODE_ELLIPSE_OUTLINES = 4
_READBACK_RING_SIZE = 2
_NEAR_CAPACITY_NUM = 9
_NEAR_CAPACITY_DEN = 10
_INITIAL_PREPASS_CAPACITY_MULTIPLIER = 64
_INITIAL_ENTRY_TILE_MULTIPLIER_CAP = 4
_MAX_IMAGE_DIM = 8192
_MAX_IMAGE_PIXELS = 33_554_432
_MAX_HOT_PREPASS_CAPACITY = 33_554_432


@dataclass
class SplattingContext:
    device: spy.Device | None = None
    radius_scale: float = 1.0
    dither_strength: float = 1.0
    max_anisotropy: float = 12.0
    alpha_cutoff: float = 0.02
    trans_threshold: float = 0.005
    render_seed: int = 0
    tile_count_readback_period: int = 8
    debug_mode: int = _DEBUG_MODE_NORMAL
    debug_depth_mean_range: tuple[float, float] = (0.0, 20.0)
    debug_depth_std_range: tuple[float, float] = (0.0, 2.0)

    def __post_init__(self) -> None:
        self.device = self.device or spy.create_device(type=spy.DeviceType.cuda, include_paths=[_SHADERS], enable_cuda_interop=False, enable_hot_reload=False)
        self._init_resources()

    def _init_resources(self) -> None:
        self.mod = spy.Module.load_from_file(self.device, str(_SHADERS / "module.slang"))
        load = lambda entry, pipeline=False: (
            self.device.create_compute_pipeline if pipeline else self.device.create_compute_kernel
        )(self.device.load_program(str(_SHADERS / "kernels.slang"), [entry]))
        for attr, entry, pipeline in (
            ("k_project_visible_splats", "csProjectVisibleSplats", False),
            ("k_count_visible_scanlines", "csCountVisibleScanlines", False),
            ("p_count_visible_scanlines", "csCountVisibleScanlines", True),
            ("k_emit_scanlines", "csEmitScanlines", False),
            ("p_emit_scanlines", "csEmitScanlines", True),
            ("k_emit_scanline_tile_counts", "csEmitScanlineTileCounts", False),
            ("p_emit_scanline_tile_counts", "csEmitScanlineTileCounts", True),
            ("k_emit_tile_entries", "csEmitTileEntries", False),
            ("p_emit_tile_entries", "csEmitTileEntries", True),
            ("k_clear_tile_ranges", "csClearTileRanges", False),
            ("k_build_tile_ranges", "csBuildTileRanges", False),
            ("p_build_tile_ranges", "csBuildTileRanges", True),
            ("k_finalize_total", "csFinalizeTotalWithOverflow", False),
            ("k_raster_fwd", "csRasterForward", False),
            ("k_raster_bwd", "csRasterBackward", False),
        ):
            setattr(self, attr, load(entry, pipeline))
        self.util = GpuUtility(self.device)
        self._float4_dtype = spy.Tensor.empty(self.device, shape=(1,), dtype=spy.float4).dtype
        self._projection_dtype = self.mod.ProjectionState
        self._gaussian_dtype = self.mod.Gaussian3D
        self._scanline_dtype = self.mod.ScanlineEntry
        count_names = ("g_ScanlineTotal", "g_TileTotal", "g_ScanlineOverflow", "g_TileOverflow")
        self.counts = {name: spy.Tensor.empty(self.device, shape=(1,), dtype="uint") for name in count_names}
        self.counts["g_CountParamsDummy"] = spy.Tensor.empty(self.device, shape=(4,), dtype="uint")
        self._tile_count_readback_ring = [
            self.device.create_buffer(size=4, usage=spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination)
            for _ in range(_READBACK_RING_SIZE)
        ]
        self._tile_count_readback_capacity = [0] * _READBACK_RING_SIZE
        self._tile_count_readback_enqueued = 0
        self._render_frame_id = 0
        self._last_total = 0
        self._delayed_tile_total = 0
        self._delayed_tile_total_valid = False
        self._size = (0, 0)

    def _view(self, tensor: spy.Tensor, dtype: Any, shape: tuple[int, ...]) -> spy.Tensor:
        dtype = self._float4_dtype if dtype is spy.float4 else dtype
        return spy.Tensor(tensor.storage, getattr(dtype, "struct", dtype), shape)

    def _read_uint(self, tensor: spy.Tensor) -> int:
        return int(np.asarray(tensor.to_numpy()).reshape(-1)[0])

    def _read_buffer_uint(self, buffer: spy.Buffer) -> int:
        return int(np.frombuffer(buffer.to_numpy().tobytes(), dtype=np.uint32, count=1)[0])

    def _ensure_capacity(self, attr: str, required: int, build: Any) -> int:
        required = max(int(required), 1)
        if getattr(self, attr, 0) < required:
            build(required)
            setattr(self, attr, required)
        return required

    def _tile_grid(self) -> tuple[int, int]:
        return (self._size[0] + 7) // 8, (self._size[1] + 7) // 8

    def _tile_grid_uint2(self) -> spy.uint2: return spy.uint2(*self._tile_grid())

    def _raster_thread_count(self) -> spy.uint3:
        tile_width, tile_height = self._tile_grid()
        return spy.uint3(tile_width * tile_height * 64, 1, 1)

    def _validate_image_size(self, image_size: tuple[int, int]) -> tuple[int, int]:
        width = int(image_size[0])
        height = int(image_size[1])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid render size {width}x{height}")
        if width > _MAX_IMAGE_DIM or height > _MAX_IMAGE_DIM or width * height > _MAX_IMAGE_PIXELS:
            raise ValueError(f"Render size {width}x{height} exceeds safe Vulkan limits")
        return width, height

    def _clamp_hot_prepass_capacity(self, required: int) -> int:
        return max(1, min(int(required), _MAX_HOT_PREPASS_CAPACITY))

    def reserve_hot_prepass_capacity(self, capacity: int | None = None) -> int:
        target = self._clamp_hot_prepass_capacity(_MAX_HOT_PREPASS_CAPACITY if capacity is None else capacity)
        self._alloc_scanlines(target)
        self._alloc_entries(target)
        self._alloc_radix(target)
        return target

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
        def build(count: int) -> None:
            usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
            if getattr(self.device, "supports_cuda_interop", False):
                usage |= spy.BufferUsage.shared
            self.scene = {
                "g_Params": spy.Tensor.empty(self.device, shape=(_PARAM_COUNT * count,), dtype=float, usage=usage),
                "g_ScanlineCounts": spy.Tensor.empty(self.device, shape=(count,), dtype="uint"),
                "g_ScanlineOffsets": spy.Tensor.empty(self.device, shape=(count,), dtype="uint"),
                "g_TileCounts": spy.Tensor.empty(self.device, shape=(count,), dtype="uint"),
                "g_ParamGrads": spy.Tensor.empty(self.device, shape=(count * _PARAM_COUNT,), dtype=float),
                "g_ProjectionStateData": spy.Tensor.empty(self.device, shape=(count, 2), dtype=spy.float4),
                "g_RasterStateData": spy.Tensor.empty(self.device, shape=(count, 4), dtype=spy.float4),
            }
            self.scene_views = {
                name: spy.InstanceTensor(dtype, (count,), self._view(self.scene[key], dtype, (count,)))
                for name, dtype, key in (
                    ("g_ProjectionState", self._projection_dtype, "g_ProjectionStateData"),
                    ("g_RasterState", self._gaussian_dtype, "g_RasterStateData"),
                )
            }
        self._ensure_capacity("_scene_capacity", splat_count, build)

    def _alloc_splat_sort(self, splat_count: int) -> None:
        def build(count: int) -> None:
            names = (
                "g_DistanceKeys",
                "g_SortedDistanceKeys",
                "g_DistanceSortOrder",
                "g_SortedDistanceSortOrder",
            )
            self.sort = {name: spy.Tensor.empty(self.device, shape=(count,), dtype="uint") for name in names}
            self._sorted_splat_order_tensor = self.sort["g_DistanceSortOrder"]
        self._ensure_capacity("_sort_size", splat_count, build)

    def _alloc_prefix(self, count: int) -> None:
        def build(size: int) -> None:
            self.prefix = {
                name: spy.Tensor.empty(self.device, shape=((size,) if name != "g_Total" else (1,)), dtype="uint")
                for name in ("g_BlockSums", "g_BlockOffsets", "g_Total")
            }
        self._ensure_capacity("_prefix_size", self.util.prefix_scratch_elements(count), build)

    def _alloc_scanlines(self, scanline_count: int) -> None:
        def build(count: int) -> None:
            self.scanlines = {
                "g_ScanlineEntryData": spy.Tensor.empty(self.device, shape=(count, 3), dtype="uint"),
                "g_ScanlineTileCounts": spy.Tensor.empty(self.device, shape=(count,), dtype="uint"),
                "g_ScanlineTileOffsets": spy.Tensor.empty(self.device, shape=(count,), dtype="uint"),
            }
            tensor = self._view(self.scanlines["g_ScanlineEntryData"], self._scanline_dtype, (count,))
            self.scanline_views = {"g_ScanlineEntries": spy.InstanceTensor(self._scanline_dtype, (count,), tensor)}
        self._ensure_capacity("_scanline_capacity", scanline_count, build)

    def _alloc_entries(self, entry_count: int) -> None:
        def build(count: int) -> None:
            self.raw = {
                name: spy.Tensor.empty(self.device, shape=(count * 2,), dtype="uint")
                for name in ("g_TileEntryData", "g_SortedEntryData")
            }
            self.tile_keys, self.tile_values = (self.raw["g_TileEntryData"].view((count,), (2,), i) for i in (0, 1))
            self.sorted_tile_keys, self.sorted_tile_values = (self.raw["g_SortedEntryData"].view((count,), (2,), i) for i in (0, 1))
            self._sorted_entries_tensor = self.raw["g_SortedEntryData"]
        self._ensure_capacity("_entry_capacity", entry_count, build)

    def _alloc_radix(self, count: int) -> None:
        count = max(count, 1)
        hist, prefix = self.util.radix_histogram_elements(count), self.util.radix_prefix_elements(count)
        if min(getattr(self, attr, 0) for attr in ("_radix_size", "_hist_size", "_hist_prefix_size")) >= 1 and (
            self._radix_size >= count and self._hist_size >= hist and self._hist_prefix_size >= prefix
        ):
            return
        self.radix = {
            "g_Histogram": spy.Tensor.empty(self.device, shape=(hist,), dtype="uint"),
            "g_HistogramPrefix": spy.Tensor.empty(self.device, shape=(prefix,), dtype="uint"),
        }
        self._radix_size, self._hist_size, self._hist_prefix_size = count, hist, prefix

    def prepare(self, splat_count: int, image_size: tuple[int, int], background: tuple[float, float, float]) -> None:
        self.background = background
        width, height = self._validate_image_size(image_size)
        self._alloc_frame((width, height))
        self._alloc_scene(splat_count)
        self._alloc_splat_sort(splat_count)
        tw, th = (width + 7) // 8, (height + 7) // 8
        base = max(int(splat_count) * _INITIAL_PREPASS_CAPACITY_MULTIPLIER, 1)
        scanline_capacity = self._clamp_hot_prepass_capacity(max(base, int(splat_count) * max(tw, th)))
        entry_capacity = self._clamp_hot_prepass_capacity(max(
            base,
            tw * th * min(max(int(splat_count), 1), _INITIAL_ENTRY_TILE_MULTIPLIER_CAP),
        ))
        self._alloc_scanlines(scanline_capacity)
        self._alloc_entries(entry_capacity)
        self._alloc_radix(entry_capacity)

    def _maybe_apply_delayed_tile_readback(self) -> None:
        period = int(self.tile_count_readback_period)
        if period <= 0 or self._tile_count_readback_enqueued == 0 or self._render_frame_id % period != 0: return
        slot = (self._tile_count_readback_enqueued - 1) % _READBACK_RING_SIZE
        total = self._read_buffer_uint(self._tile_count_readback_ring[slot])
        capacity = int(self._tile_count_readback_capacity[slot])
        self._delayed_tile_total, self._delayed_tile_total_valid = total, True
        near_capacity = total >= max((capacity * _NEAR_CAPACITY_NUM) // _NEAR_CAPACITY_DEN, 1)
        if total > capacity or near_capacity:
            current = max(capacity, getattr(self, "_scanline_capacity", 1), getattr(self, "_entry_capacity", 1), 1)
            required = max(int(total), capacity, 1)
            target = self._clamp_hot_prepass_capacity(max(required, current + current // 2))
            self._alloc_scanlines(target)
            self._alloc_entries(target)
            self._alloc_radix(target)
            self.device.sync_to_cuda()

    def _enqueue_tile_readback(self, command_encoder: spy.CommandEncoder) -> None:
        period = int(self.tile_count_readback_period)
        if period <= 0 or self._render_frame_id % period != 0: return
        slot = self._tile_count_readback_enqueued % _READBACK_RING_SIZE
        command_encoder.copy_buffer(self.counts["g_TileTotal"].storage, 0, self._tile_count_readback_ring[slot], 0, 4)
        self._tile_count_readback_capacity[slot] = int(getattr(self, "_entry_capacity", 1))
        self._tile_count_readback_enqueued += 1

    def _dispatch_indirect(
        self,
        compute_pass: spy.ComputePassEncoder,
        pipeline: spy.ComputePipeline,
        args_buffer: spy.Buffer,
        vars: dict[str, Any],
    ) -> None:
        cursor = spy.ShaderCursor(compute_pass.bind_pipeline(pipeline))
        for name, value in vars.items():
            setattr(cursor, name, value)
        compute_pass.dispatch_compute_indirect(spy.BufferOffsetPair(args_buffer, 0))

    def _indirect_count_vars(self, args_buffer: spy.Buffer, count: int, **extra: Any) -> dict[str, Any]:
        return {
            "g_TileGrid": self._tile_grid_uint2(),
            "g_SortedEntryCount": int(count),
            "g_CountParamsBuffer": args_buffer,
            "g_UseCountParams": 1,
            **extra,
        }

    def _vars(self, camera: dict[str, Any], splats: int, entries: int) -> dict[str, Any]:
        vars = {
            "g_ScanlineCounts": self.scene["g_ScanlineCounts"],
            "g_ScanlineOffsets": self.scene["g_ScanlineOffsets"],
            "g_TileCounts": self.scene["g_TileCounts"],
            "g_ParamGrads": self.scene["g_ParamGrads"],
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
            "g_CountParamsBuffer": self.counts["g_CountParamsDummy"].storage,
            "g_TotalBuffer": self.counts["g_TileTotal"],
            "g_TotalCounterBuffer": self.counts["g_TileTotal"].storage,
            "g_OverflowA": self.counts["g_ScanlineOverflow"],
            "g_OverflowB": self.counts["g_TileOverflow"],
            "g_ScanlineOverflow": self.counts["g_ScanlineOverflow"],
            "g_TileOverflow": self.counts["g_TileOverflow"],
            **self.tiles,
            **self.frame,
            "g_Camera": camera,
            "g_SplatCount": int(splats),
            "g_SortedEntryCount": int(entries),
            "g_UseCountParams": 0,
            "g_ScanlineCapacity": int(getattr(self, "_scanline_capacity", 1)),
            "g_TileEntryCapacity": int(getattr(self, "_entry_capacity", 1)),
        }
        vars.update(
            g_TileGrid=self._tile_grid_uint2(),
            g_Background=spy.float3(*map(float, self.background)),
            g_RadiusScale=float(self.radius_scale),
            g_DitherStrength=float(self.dither_strength),
            g_MaxAnisotropy=float(self.max_anisotropy),
            g_AlphaCutoff=float(self.alpha_cutoff),
            g_TransmittanceThreshold=float(self.trans_threshold),
            g_RenderSeed=int(self.render_seed),
            g_DebugMode=int(self.debug_mode),
            g_DebugMaxSplatSteps=int(entries),
            g_DebugDepthMeanRange=spy.float2(*map(float, self.debug_depth_mean_range)),
            g_DebugDepthStdRange=spy.float2(*map(float, self.debug_depth_std_range)),
            g_TotalCapacity=int(getattr(self, "_entry_capacity", 1)),
        )
        return vars

    def _record_sort_and_project(
        self,
        command_encoder: spy.CommandEncoder,
        camera: dict[str, Any],
        splat_count: int,
        visible_out: spy.Tensor,
        total_out: spy.Tensor,
    ) -> None:
        self._alloc_prefix(splat_count)
        self._alloc_splat_sort(splat_count)
        self._alloc_radix(splat_count)
        self._alloc_prefix(self.util.radix_histogram_elements(splat_count))
        if splat_count == 0:
            self._sorted_splat_order_tensor = self.sort["g_DistanceSortOrder"]
            visible_out.clear(command_encoder=command_encoder)
            total_out.clear(command_encoder=command_encoder)
            return
        visible_out.clear(command_encoder=command_encoder)
        project_vars = self._vars(camera, splat_count, 0)
        project_vars.update(g_TotalBuffer=visible_out, g_TotalCapacity=int(splat_count))
        with debug_group(command_encoder, "renderer.project_visible", _DEBUG_COLOR):
            self.k_project_visible_splats.dispatch(
                thread_count=spy.uint3(splat_count, 1, 1),
                vars=project_vars,
                command_encoder=command_encoder,
            )
        with debug_group(command_encoder, "renderer.distance_sort_visible", _DEBUG_COLOR):
            out_buffer, _ = self.util.radix_sort_uint32_from_count_buffer(
                command_encoder,
                self.sort["g_DistanceKeys"],
                self.sort["g_DistanceSortOrder"],
                self.sort["g_SortedDistanceKeys"],
                self.sort["g_SortedDistanceSortOrder"],
                self.radix["g_Histogram"],
                self.radix["g_HistogramPrefix"],
                visible_out,
                0,
                splat_count,
                0,
                32,
            )
        self._sorted_splat_order_tensor = self.sort["g_SortedDistanceSortOrder"] if out_buffer else self.sort["g_DistanceSortOrder"]
        visible_dispatch_args = self.util.dispatch_args_from_count_buffer(
            command_encoder,
            visible_out,
            0,
            splat_count,
            256,
        )
        with debug_group(command_encoder, "renderer.count_visible_scanlines", _DEBUG_COLOR):
            with command_encoder.begin_compute_pass() as compute_pass:
                self._dispatch_indirect(
                    compute_pass,
                    self.p_count_visible_scanlines,
                    visible_dispatch_args,
                    self._indirect_count_vars(
                        visible_dispatch_args,
                        splat_count,
                        g_SplatOrder=self._sorted_splat_order_tensor,
                        g_ScanlineCounts=self.scene["g_ScanlineCounts"],
                        g_ProjectionState=self.scene_views["g_ProjectionState"].tensor,
                    ),
                )
        with debug_group(command_encoder, "renderer.scanline_prefix", _DEBUG_COLOR):
            self.util.prefix_sum_uint32_from_count_buffer(
                command_encoder,
                self.scene["g_ScanlineCounts"],
                self.scene["g_ScanlineOffsets"],
                self.prefix["g_BlockSums"],
                self.prefix["g_BlockOffsets"],
                total_out,
                visible_out,
                0,
                splat_count,
                True,
            )

    def _record_hot_prepass_counts(self, command_encoder: spy.CommandEncoder, camera: dict[str, Any], splat_count: int) -> None:
        self._record_sort_and_project(command_encoder, camera, splat_count, self.counts["g_TileTotal"], self.counts["g_ScanlineTotal"])
        self._alloc_radix(max(getattr(self, "_entry_capacity", 1), splat_count))
        self.counts["g_ScanlineOverflow"].clear(command_encoder=command_encoder)
        self.counts["g_TileOverflow"].clear(command_encoder=command_encoder)
        visible_args = self.util.dispatch_args_from_count_buffer(
            command_encoder,
            self.counts["g_TileTotal"],
            0,
            splat_count,
            256,
        )
        with debug_group(command_encoder, "renderer.emit_scanlines", _DEBUG_COLOR):
            with command_encoder.begin_compute_pass() as compute_pass:
                self._dispatch_indirect(
                    compute_pass,
                    self.p_emit_scanlines,
                    visible_args,
                    self._indirect_count_vars(
                        visible_args,
                        splat_count,
                        g_SplatOrder=self._sorted_splat_order_tensor,
                        g_ScanlineCounts=self.scene["g_ScanlineCounts"],
                        g_ScanlineOffsets=self.scene["g_ScanlineOffsets"],
                        g_ScanlineEntries=self.scanline_views["g_ScanlineEntries"].tensor,
                        g_ProjectionState=self.scene_views["g_ProjectionState"].tensor,
                        g_ScanlineCapacity=int(getattr(self, "_scanline_capacity", 1)),
                        g_ScanlineOverflow=self.counts["g_ScanlineOverflow"],
                    ),
                )
        scanline_args = self.util.dispatch_args_from_count_buffer(
            command_encoder,
            self.counts["g_ScanlineTotal"],
            0,
            getattr(self, "_scanline_capacity", 1),
            256,
        )
        with debug_group(command_encoder, "renderer.scanline_tile_counts", _DEBUG_COLOR):
            with command_encoder.begin_compute_pass() as compute_pass:
                self._dispatch_indirect(
                    compute_pass,
                    self.p_emit_scanline_tile_counts,
                    scanline_args,
                    self._indirect_count_vars(
                        scanline_args,
                        getattr(self, "_scanline_capacity", 1),
                        g_ScanlineEntries=self.scanline_views["g_ScanlineEntries"].tensor,
                        g_ScanlineTileCounts=self.scanlines["g_ScanlineTileCounts"],
                        g_ProjectionState=self.scene_views["g_ProjectionState"].tensor,
                    ),
                )
        self._alloc_prefix(getattr(self, "_scanline_capacity", 1))
        with debug_group(command_encoder, "renderer.tile_prefix", _DEBUG_COLOR):
            self.util.prefix_sum_uint32_from_count_buffer(
                command_encoder,
                self.scanlines["g_ScanlineTileCounts"],
                self.scanlines["g_ScanlineTileOffsets"],
                self.prefix["g_BlockSums"],
                self.prefix["g_BlockOffsets"],
                self.counts["g_TileTotal"],
                self.counts["g_ScanlineTotal"],
                0,
                getattr(self, "_scanline_capacity", 1),
                True,
            )
        with debug_group(command_encoder, "renderer.emit_tiles", _DEBUG_COLOR):
            with command_encoder.begin_compute_pass() as compute_pass:
                self._dispatch_indirect(
                    compute_pass,
                    self.p_emit_tile_entries,
                    scanline_args,
                    self._indirect_count_vars(
                        scanline_args,
                        getattr(self, "_scanline_capacity", 1),
                        g_ScanlineEntries=self.scanline_views["g_ScanlineEntries"].tensor,
                        g_ScanlineTileOffsets=self.scanlines["g_ScanlineTileOffsets"],
                        g_TileEntries=self.raw["g_TileEntryData"],
                        g_ProjectionState=self.scene_views["g_ProjectionState"].tensor,
                        g_TileEntryCapacity=int(getattr(self, "_entry_capacity", 1)),
                        g_TileOverflow=self.counts["g_TileOverflow"],
                    ),
                )
        finalize_vars = self._vars(camera, splat_count, getattr(self, "_entry_capacity", 1))
        finalize_vars.update(
            g_TotalBuffer=self.counts["g_TileTotal"],
            g_OverflowA=self.counts["g_ScanlineOverflow"],
            g_OverflowB=self.counts["g_TileOverflow"],
            g_TotalCapacity=int(getattr(self, "_entry_capacity", 1)),
        )
        self.k_finalize_total.dispatch(
            thread_count=spy.uint3(1, 1, 1),
            vars=finalize_vars,
            command_encoder=command_encoder,
        )

    def _record_hot_prepass_sort_and_ranges(self, command_encoder: spy.CommandEncoder, camera: dict[str, Any], splat_count: int) -> None:
        tile_args = self.util.dispatch_args_from_count_buffer(
            command_encoder,
            self.counts["g_TileTotal"],
            0,
            getattr(self, "_entry_capacity", 1),
            256,
        )
        tw, th = self._tile_grid()
        tile_count = tw * th
        self.k_clear_tile_ranges.dispatch(
            thread_count=spy.uint3(tile_count, 1, 1),
            vars=self._vars(camera, splat_count, getattr(self, "_entry_capacity", 1)),
            command_encoder=command_encoder,
        )
        self._sorted_entries_tensor = self.raw["g_SortedEntryData"]
        out_buffer, _ = self.util.radix_sort_uint32_from_count_buffer(
            command_encoder,
            self.tile_keys,
            self.tile_values,
            self.sorted_tile_keys,
            self.sorted_tile_values,
            self.radix["g_Histogram"],
            self.radix["g_HistogramPrefix"],
            self.counts["g_TileTotal"],
            0,
            getattr(self, "_entry_capacity", 1),
            0,
            max(1, (tw * th - 1).bit_length()),
        )
        self._sorted_entries_tensor = self.raw["g_SortedEntryData"] if out_buffer else self.raw["g_TileEntryData"]
        with debug_group(command_encoder, "renderer.build_tile_ranges", _DEBUG_COLOR):
            with command_encoder.begin_compute_pass() as compute_pass:
                self._dispatch_indirect(
                    compute_pass,
                    self.p_build_tile_ranges,
                    tile_args,
                    self._indirect_count_vars(
                        tile_args,
                        getattr(self, "_entry_capacity", 1),
                        g_SortedEntries=self._sorted_entries_tensor,
                        g_TileRanges=self.tiles["g_TileRanges"],
                    ),
                )

    def project(self, camera: dict[str, Any], splat_count: int) -> int:
        self._alloc_prefix(splat_count)
        enc = self.device.create_command_encoder()
        with debug_group(enc, "renderer.project_scanlines", _DEBUG_COLOR):
            self._record_sort_and_project(enc, camera, splat_count, self.counts["g_TileTotal"], self.prefix["g_Total"])
        self.device.submit_command_buffer(enc.finish())
        self.device.sync_to_device()
        total = min(self._read_uint(self.prefix["g_Total"]), _MAX_HOT_PREPASS_CAPACITY)
        self._alloc_scanlines(total)
        self.device.sync_to_cuda()
        if total == 0:
            return total
        enc = self.device.create_command_encoder()
        visible_args = self.util.dispatch_args_from_count_buffer(
            enc,
            self.counts["g_TileTotal"],
            0,
            splat_count,
            256,
        )
        with debug_group(enc, "renderer.emit_scanlines", _DEBUG_COLOR):
            vars = self._vars(camera, splat_count, total)
            with enc.begin_compute_pass() as compute_pass:
                self._dispatch_indirect(
                    compute_pass,
                    self.p_emit_scanlines,
                    visible_args,
                    self._indirect_count_vars(
                        visible_args,
                        splat_count,
                        g_SplatOrder=self._sorted_splat_order_tensor,
                        g_ScanlineCounts=self.scene["g_ScanlineCounts"],
                        g_ScanlineOffsets=self.scene["g_ScanlineOffsets"],
                        g_ScanlineEntries=self.scanline_views["g_ScanlineEntries"].tensor,
                        g_ProjectionState=self.scene_views["g_ProjectionState"].tensor,
                        g_ScanlineCapacity=int(getattr(self, "_scanline_capacity", 1)),
                        g_ScanlineOverflow=self.counts["g_ScanlineOverflow"],
                    ),
                )
            self.k_emit_scanline_tile_counts.dispatch(
                thread_count=spy.uint3(total, 1, 1),
                vars=vars,
                command_encoder=enc,
            )
        self.device.submit_command_buffer(enc.finish())
        self.device.sync_to_device()
        return total

    def render(self, camera: dict[str, Any], splat_count: int, command_encoder: spy.CommandEncoder | None = None) -> spy.Tensor:
        self._maybe_apply_delayed_tile_readback()
        enc = command_encoder or self.device.create_command_encoder()
        with debug_group(enc, "renderer.prepass", _DEBUG_COLOR):
            self._record_hot_prepass_counts(enc, camera, splat_count)
        with debug_group(enc, "renderer.tile_ranges", _DEBUG_COLOR):
            self._record_hot_prepass_sort_and_ranges(enc, camera, splat_count)
        with debug_group(enc, "renderer.render", _DEBUG_COLOR):
            with debug_group(enc, "renderer.raster_forward", _DEBUG_COLOR):
                self.k_raster_fwd.dispatch(
                    thread_count=self._raster_thread_count(),
                    vars=self._vars(camera, splat_count, getattr(self, "_entry_capacity", 1)),
                    command_encoder=enc,
                )
            self._enqueue_tile_readback(enc)
        self._last_total = int(self._delayed_tile_total) if self._delayed_tile_total_valid else 0
        self._render_frame_id += 1
        if command_encoder is None:
            self.device.submit_command_buffer(enc.finish()); self.device.sync_to_device()
        return self.frame["g_Output"]

    def backward(self, camera: dict[str, Any], splat_count: int, command_encoder: spy.CommandEncoder | None = None) -> spy.Tensor:
        enc = command_encoder or self.device.create_command_encoder()
        with debug_group(enc, "renderer.backward", _DEBUG_COLOR):
            self.scene["g_ParamGrads"].clear(command_encoder=enc)
            self.k_raster_bwd.dispatch(thread_count=self._raster_thread_count(), vars=self._vars(camera, splat_count, self._last_total), command_encoder=enc)
        if command_encoder is None:
            self.device.submit_command_buffer(enc.finish()); self.device.sync_to_device()
        return self.scene["g_ParamGrads"]
