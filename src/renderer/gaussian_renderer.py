from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import slangpy as spy

from ..common import SHADER_ROOT
from ..scene.gaussian_scene import GaussianScene
from ..sort.radix_sort import GPURadixSort
from .camera import Camera


@dataclass(slots=True)
class RenderOutput:
    image: np.ndarray
    stats: dict[str, int | bool | float]


class GaussianRenderer:
    _COUNTER_READBACK_RING_SIZE = 2
    _SCANLINE_WORK_ITEM_UINTS = 8
    _U32_BYTES = 4
    _MEBIBYTE_BYTES = 1024 * 1024
    _PREPASS_ENTRY_BYTES = (_SCANLINE_WORK_ITEM_UINTS + 2) * _U32_BYTES

    def __init__(
        self,
        device: spy.Device,
        width: int,
        height: int,
        tile_size: int = 16,
        radius_scale: float = 2.6,
        alpha_cutoff: float = 1.0 / 255.0,
        max_splat_steps: int = 32768,
        transmittance_threshold: float = 0.005,
        list_capacity_multiplier: int = 64,
        max_prepass_memory_mb: int = 4096,
        sampled5_mvee_iters: int = 6,
        sampled5_safety_scale: float = 1.0,
        sampled5_radius_pad_px: float = 1.0,
        sampled5_eps: float = 1e-6,
        proj_distortion_k1: float = 0.0,
        proj_distortion_k2: float = 0.0,
        debug_show_ellipses: bool = False,
        debug_show_processed_count: bool = False,
        debug_ellipse_thickness_px: float = 1.0,
        debug_ellipse_color: tuple[float, float, float] = (1.0, 0.15, 0.1),
    ) -> None:
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.tile_size = int(tile_size)
        self.radius_scale = float(radius_scale)
        self.alpha_cutoff = float(alpha_cutoff)
        self.max_splat_steps = int(max_splat_steps)
        self.transmittance_threshold = float(transmittance_threshold)
        self.list_capacity_multiplier = int(list_capacity_multiplier)
        self.max_prepass_memory_mb = max(int(max_prepass_memory_mb), 1)
        self._max_prepass_memory_bytes = self.max_prepass_memory_mb * self._MEBIBYTE_BYTES
        self.sampled5_mvee_iters = int(sampled5_mvee_iters)
        self.sampled5_safety_scale = float(sampled5_safety_scale)
        self.sampled5_radius_pad_px = float(sampled5_radius_pad_px)
        self.sampled5_eps = float(sampled5_eps)
        self.proj_distortion_k1 = float(proj_distortion_k1)
        self.proj_distortion_k2 = float(proj_distortion_k2)
        self.debug_show_ellipses = bool(debug_show_ellipses)
        self.debug_show_processed_count = bool(debug_show_processed_count)
        self.debug_ellipse_thickness_px = float(debug_ellipse_thickness_px)
        self.debug_ellipse_color = np.asarray(debug_ellipse_color, dtype=np.float32).reshape(3)

        self.tile_width = (self.width + self.tile_size - 1) // self.tile_size
        self.tile_height = (self.height + self.tile_size - 1) // self.tile_size
        self.tile_count = self.tile_width * self.tile_height

        self.tile_bits = int(np.ceil(np.log2(max(self.tile_count, 2))))
        self.depth_bits = 32 - self.tile_bits
        self.sort_bits = self.tile_bits + self.depth_bits

        self._project_shader_path = Path(SHADER_ROOT / "renderer" / "gaussian_project_stage.slang")
        self._raster_shader_path = Path(SHADER_ROOT / "renderer" / "gaussian_raster_stage.slang")
        self._create_shaders()
        self._create_sorter()

        self._scene_count = 0
        self._scene_capacity = 0
        self._current_scene: GaussianScene | None = None
        self._max_list_entries = 0
        self._work_splat_capacity = 0
        self._scene_buffers: dict[str, spy.Buffer] = {}
        self._work_buffers: dict[str, spy.Buffer] = {}
        self._output_texture: spy.Texture | None = None
        self._output_grad_texture: spy.Texture | None = None
        self._raster_forward_state_texture: spy.Texture | None = None
        self._raster_processed_count_texture: spy.Texture | None = None
        self._last_stats: dict[str, int | bool | float] = {}
        self._max_scanline_entries = 0
        self._counter_readback_ring: list[spy.Buffer] = []
        self._counter_readback_capacity: list[int] = []
        self._counter_readback_frame_id = 0
        self._pending_min_list_entries = 0
        self._delayed_generated_entries = 0
        self._delayed_written_entries = 0
        self._delayed_overflow = False
        self._delayed_stats_valid = False

    def _create_shaders(self) -> None:
        load_program = self.device.load_program
        self._k_project = self.device.create_compute_kernel(
            load_program(str(self._project_shader_path), ["csProjectAndBin"])
        )
        self._p_compose_scanline = self.device.create_compute_pipeline(
            load_program(str(self._project_shader_path), ["csComposeScanlineKeyValues"])
        )
        self._k_clear_ranges = self.device.create_compute_kernel(
            load_program(str(self._project_shader_path), ["csClearTileRanges"])
        )
        self._p_build_ranges = self.device.create_compute_pipeline(
            load_program(str(self._project_shader_path), ["csBuildTileRanges"])
        )
        self._k_raster = self.device.create_compute_kernel(
            load_program(str(self._raster_shader_path), ["csRasterize"])
        )
        self._k_clear_raster_grads = self.device.create_compute_kernel(
            load_program(str(self._raster_shader_path), ["csClearRasterGrads"])
        )
        self._k_raster_backward = self.device.create_compute_kernel(
            load_program(str(self._raster_shader_path), ["csRasterizeBackward"])
        )

    def _create_sorter(self) -> None:
        self._sorter = GPURadixSort(self.device)

    def _buffer_usage_rw(self) -> spy.BufferUsage:
        return (
            spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access
            | spy.BufferUsage.copy_source
            | spy.BufferUsage.copy_destination
        )

    def _buffer_usage_sr(self) -> spy.BufferUsage:
        return (
            spy.BufferUsage.shader_resource
            | spy.BufferUsage.copy_source
            | spy.BufferUsage.copy_destination
        )

    def _max_prepass_entries_by_budget(self) -> int:
        return max(self._max_prepass_memory_bytes // self._PREPASS_ENTRY_BYTES, 1)

    def _ensure_scene_buffers(self, splat_count: int) -> None:
        if self._scene_buffers and splat_count <= self._scene_capacity:
            self._scene_count = splat_count
            return
        old_capacity = max(self._scene_capacity, 1)
        new_capacity = max(splat_count, old_capacity + old_capacity // 2)
        usage = self._buffer_usage_rw()
        self._scene_buffers = {
            "positions": self.device.create_buffer(size=max(new_capacity, 1) * 16, usage=usage),
            "scales": self.device.create_buffer(size=max(new_capacity, 1) * 16, usage=usage),
            "rotations": self.device.create_buffer(size=max(new_capacity, 1) * 16, usage=usage),
            "color_alpha": self.device.create_buffer(size=max(new_capacity, 1) * 16, usage=usage),
        }
        self._scene_capacity = new_capacity
        self._scene_count = splat_count

    def _ensure_work_buffers(self, splat_count: int, min_list_entries: int = 0) -> None:
        max_prepass_entries = self._max_prepass_entries_by_budget()
        required_splat_capacity = max(splat_count, 1)
        required_list_entries = min(
            max(splat_count * self.list_capacity_multiplier, min_list_entries, 1),
            max_prepass_entries,
        )
        required_scanline_entries = required_list_entries
        if (
            self._work_buffers
            and required_splat_capacity <= self._work_splat_capacity
            and required_list_entries <= self._max_list_entries
            and required_scanline_entries <= self._max_scanline_entries
            and self._output_texture is not None
            and self._output_grad_texture is not None
            and self._raster_forward_state_texture is not None
            and self._raster_processed_count_texture is not None
        ):
            return
        old_splat_capacity = max(self._work_splat_capacity, 1)
        old_list_capacity = max(self._max_list_entries, 1)
        old_scanline_capacity = max(self._max_scanline_entries, 1)
        splat_capacity = max(required_splat_capacity, old_splat_capacity + old_splat_capacity // 2)
        max_list_entries = min(max(required_list_entries, old_list_capacity + old_list_capacity // 2), max_prepass_entries)
        max_scanline_entries = min(
            max(required_scanline_entries, old_scanline_capacity + old_scanline_capacity // 2),
            max_prepass_entries,
        )
        usage = self._buffer_usage_rw()
        self._work_buffers = {
            "screen_center_radius_depth": self.device.create_buffer(
                size=max(splat_capacity, 1) * 16, usage=usage
            ),
            "screen_color_alpha": self.device.create_buffer(size=max(splat_capacity, 1) * 16, usage=usage),
            "screen_ellipse_conic": self.device.create_buffer(size=max(splat_capacity, 1) * 16, usage=usage),
            "keys": self.device.create_buffer(size=max_list_entries * 4, usage=usage),
            "values": self.device.create_buffer(size=max_list_entries * 4, usage=usage),
            "counter": self.device.create_buffer(size=4, usage=usage),
            "splat_list_bases": self.device.create_buffer(size=max(splat_capacity, 1) * self._U32_BYTES, usage=usage),
            "scanline_work_items": self.device.create_buffer(
                size=max_scanline_entries * self._SCANLINE_WORK_ITEM_UINTS * self._U32_BYTES,
                usage=usage,
            ),
            "scanline_counter": self.device.create_buffer(size=self._U32_BYTES, usage=usage),
            "tile_ranges": self.device.create_buffer(size=max(self.tile_count, 1) * 8, usage=usage),
            "grad_positions": self.device.create_buffer(
                size=max(splat_capacity, 1) * 4 * self._U32_BYTES,
                usage=usage,
            ),
            "grad_scales": self.device.create_buffer(
                size=max(splat_capacity, 1) * 4 * self._U32_BYTES,
                usage=usage,
            ),
            "grad_rotations": self.device.create_buffer(
                size=max(splat_capacity, 1) * 4 * self._U32_BYTES,
                usage=usage,
            ),
            "grad_color_alpha": self.device.create_buffer(
                size=max(splat_capacity, 1) * 4 * self._U32_BYTES,
                usage=usage,
            ),
        }
        if self._output_texture is None:
            self._output_texture = self.device.create_texture(
                format=spy.Format.rgba32_float,
                width=self.width,
                height=self.height,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            )
        if self._output_grad_texture is None:
            self._output_grad_texture = self.device.create_texture(
                format=spy.Format.rgba32_float,
                width=self.width,
                height=self.height,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            )
        if self._raster_forward_state_texture is None:
            self._raster_forward_state_texture = self.device.create_texture(
                format=spy.Format.rgba32_float,
                width=self.width,
                height=self.height,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            )
        if self._raster_processed_count_texture is None:
            self._raster_processed_count_texture = self.device.create_texture(
                format=spy.Format.r32_uint,
                width=self.width,
                height=self.height,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            )
        self._work_splat_capacity = splat_capacity
        self._max_list_entries = max_list_entries
        self._max_scanline_entries = max_scanline_entries
        if self._pending_min_list_entries > 0 and self._max_list_entries >= self._pending_min_list_entries:
            self._pending_min_list_entries = 0

    def _pack_scene(self, scene: GaussianScene) -> dict[str, np.ndarray]:
        splat_count = scene.count
        positions = np.zeros((splat_count, 4), dtype=np.float32)
        positions[:, :3] = scene.positions
        scales = np.zeros((splat_count, 4), dtype=np.float32)
        scales[:, :3] = scene.scales
        rotations = np.zeros((splat_count, 4), dtype=np.float32)
        rotations[:, :] = scene.rotations
        color_alpha = np.zeros((splat_count, 4), dtype=np.float32)
        color_alpha[:, :3] = scene.colors
        color_alpha[:, 3] = scene.opacities
        return {"positions": positions, "scales": scales, "rotations": rotations, "color_alpha": color_alpha}

    def _upload_scene(self, scene: GaussianScene) -> None:
        packed = self._pack_scene(scene)
        self._scene_buffers["positions"].copy_from_numpy(packed["positions"])
        self._scene_buffers["scales"].copy_from_numpy(packed["scales"])
        self._scene_buffers["rotations"].copy_from_numpy(packed["rotations"])
        self._scene_buffers["color_alpha"].copy_from_numpy(packed["color_alpha"])

    def _reset_prepass_counters(self) -> None:
        zero = np.array([0], dtype=np.uint32)
        self._work_buffers["counter"].copy_from_numpy(zero)
        self._work_buffers["scanline_counter"].copy_from_numpy(zero)

    def _read_u32(self, buffer: spy.Buffer, count: int) -> np.ndarray:
        raw = buffer.to_numpy()
        values = np.frombuffer(raw.tobytes(), dtype=np.uint32)
        return values[:count].copy()

    def _read_f32x4(self, buffer: spy.Buffer, count: int) -> np.ndarray:
        raw = buffer.to_numpy()
        values = np.frombuffer(raw.tobytes(), dtype=np.float32)
        return values[: count * 4].reshape(count, 4).copy()

    def _read_counter(self) -> int:
        return int(self._read_u32(self._work_buffers["counter"], 1)[0])

    def _ensure_counter_readback_ring(self) -> None:
        if self._counter_readback_ring:
            return
        usage = spy.BufferUsage.copy_destination | spy.BufferUsage.copy_source
        self._counter_readback_ring = [self.device.create_buffer(size=4, usage=usage) for _ in range(self._COUNTER_READBACK_RING_SIZE)]
        self._counter_readback_capacity = [0 for _ in range(self._COUNTER_READBACK_RING_SIZE)]

    def _enqueue_counter_readback(self, encoder: spy.CommandEncoder) -> None:
        self._ensure_counter_readback_ring()
        slot = self._counter_readback_frame_id % self._COUNTER_READBACK_RING_SIZE
        encoder.copy_buffer(self._counter_readback_ring[slot], 0, self._work_buffers["counter"], 0, 4)
        self._counter_readback_capacity[slot] = int(self._max_list_entries)

    def _update_delayed_counter_stats(self) -> None:
        if self._counter_readback_frame_id <= 1:
            self._delayed_stats_valid = False
            return
        slot = (self._counter_readback_frame_id - 2) % self._COUNTER_READBACK_RING_SIZE
        generated = int(self._read_u32(self._counter_readback_ring[slot], 1)[0])
        capacity = int(self._counter_readback_capacity[slot])
        self._apply_counter_feedback(generated, capacity)

    def _apply_counter_feedback(self, generated: int, capacity: int) -> None:
        written = min(generated, capacity)
        overflow = generated > capacity
        self._delayed_generated_entries = generated
        self._delayed_written_entries = written
        self._delayed_overflow = overflow
        self._delayed_stats_valid = True
        if overflow:
            target_capacity = min(generated, self._max_prepass_entries_by_budget())
            self._pending_min_list_entries = max(self._pending_min_list_entries, target_capacity)

    def _camera_uniforms(self, camera: Camera) -> dict[str, object]:
        return {
            "g_Camera": {
                **camera.gpu_params(self.width, self.height),
                "projDistortionK1": float(self.proj_distortion_k1),
                "projDistortionK2": float(self.proj_distortion_k2),
            }
        }

    def _prepass_uniforms(self, splat_count: int, sorted_count_offset: int = 0) -> dict[str, object]:
        return {
            "g_Prepass": {
                "splatCount": int(splat_count),
                "tileSize": int(self.tile_size),
                "tileWidth": int(self.tile_width),
                "tileHeight": int(self.tile_height),
                "tileCount": int(self.tile_count),
                "depthBits": int(self.depth_bits),
                "sortedCountOffset": int(sorted_count_offset),
                "maxListEntries": int(self._max_list_entries),
                "maxScanlineEntries": int(self._max_scanline_entries),
                "radiusScale": float(self.radius_scale),
                "sampled5MVEEIters": int(self.sampled5_mvee_iters),
                "sampled5SafetyScale": float(self.sampled5_safety_scale),
                "sampled5RadiusPadPx": float(self.sampled5_radius_pad_px),
                "sampled5Eps": float(self.sampled5_eps),
            }
        }

    def _raster_uniforms(self, background: np.ndarray) -> dict[str, object]:
        return {
            "g_Raster": {
                "width": int(self.width),
                "height": int(self.height),
                "maxSplatSteps": int(self.max_splat_steps),
                "alphaCutoff": float(self.alpha_cutoff),
                "transmittanceThreshold": float(self.transmittance_threshold),
                "background": spy.float3(*background.tolist()),
                "debugShowEllipses": np.uint32(1 if self.debug_show_ellipses else 0),
                "debugShowProcessedCount": np.uint32(1 if self.debug_show_processed_count else 0),
                "debugEllipseThicknessPx": float(self.debug_ellipse_thickness_px),
                "debugEllipseColor": spy.float3(*self.debug_ellipse_color.tolist()),
            }
        }

    def _bind_prepass_cursor(self, cursor: spy.ShaderCursor, splat_count: int, sorted_count_offset: int = 0) -> None:
        prepass = self._prepass_uniforms(splat_count, sorted_count_offset)["g_Prepass"]
        cursor.g_Prepass.splatCount = prepass["splatCount"]
        cursor.g_Prepass.tileSize = prepass["tileSize"]
        cursor.g_Prepass.tileWidth = prepass["tileWidth"]
        cursor.g_Prepass.tileHeight = prepass["tileHeight"]
        cursor.g_Prepass.tileCount = prepass["tileCount"]
        cursor.g_Prepass.depthBits = prepass["depthBits"]
        cursor.g_Prepass.sortedCountOffset = prepass["sortedCountOffset"]
        cursor.g_Prepass.maxListEntries = prepass["maxListEntries"]
        cursor.g_Prepass.maxScanlineEntries = prepass["maxScanlineEntries"]
        cursor.g_Prepass.radiusScale = prepass["radiusScale"]
        cursor.g_Prepass.sampled5MVEEIters = prepass["sampled5MVEEIters"]
        cursor.g_Prepass.sampled5SafetyScale = prepass["sampled5SafetyScale"]
        cursor.g_Prepass.sampled5RadiusPadPx = prepass["sampled5RadiusPadPx"]
        cursor.g_Prepass.sampled5Eps = prepass["sampled5Eps"]

    def _project_and_bin(self, encoder: spy.CommandEncoder, scene: GaussianScene, camera: Camera) -> None:
        vars = {
            "g_Positions": self._scene_buffers["positions"],
            "g_Scales": self._scene_buffers["scales"],
            "g_Rotations": self._scene_buffers["rotations"],
            "g_ColorAlpha": self._scene_buffers["color_alpha"],
            "g_ScreenCenterRadiusDepth": self._work_buffers["screen_center_radius_depth"],
            "g_ScreenColorAlpha": self._work_buffers["screen_color_alpha"],
            "g_ScreenEllipseConic": self._work_buffers["screen_ellipse_conic"],
            "g_Keys": self._work_buffers["keys"],
            "g_Values": self._work_buffers["values"],
            "g_ListCounter": self._work_buffers["counter"],
            "g_SplatListBases": self._work_buffers["splat_list_bases"],
            "g_ScanlineWorkItems": self._work_buffers["scanline_work_items"],
            "g_ScanlineCounter": self._work_buffers["scanline_counter"],
            **self._prepass_uniforms(scene.count),
            **self._camera_uniforms(camera),
        }
        self._k_project.dispatch(
            thread_count=spy.uint3(scene.count, 1, 1),
            vars=vars,
            command_encoder=encoder,
        )

    def _compute_scanline_dispatch_args(self, encoder: spy.CommandEncoder) -> spy.Buffer:
        args_buffer = self._sorter.ensure_indirect_args()
        self._sorter.compute_indirect_args_from_buffer_dispatch(
            encoder=encoder,
            count_buffer=self._work_buffers["scanline_counter"],
            count_offset=0,
            max_element_count=self._max_scanline_entries,
            args_buffer=args_buffer,
        )
        return args_buffer

    def _compose_scanline_key_values_indirect(self, encoder: spy.CommandEncoder, args_buffer: spy.Buffer) -> None:
        with encoder.begin_compute_pass() as compute_pass:
            cursor = spy.ShaderCursor(compute_pass.bind_pipeline(self._p_compose_scanline))
            cursor.g_ScanlineWorkItems = self._work_buffers["scanline_work_items"]
            cursor.g_ScanlineCounter = self._work_buffers["scanline_counter"]
            cursor.g_SplatListBases = self._work_buffers["splat_list_bases"]
            cursor.g_Keys = self._work_buffers["keys"]
            cursor.g_Values = self._work_buffers["values"]
            self._bind_prepass_cursor(cursor, self._scene_count)
            compute_pass.dispatch_compute_indirect(
                spy.BufferOffsetPair(args_buffer, GPURadixSort.BUILD_RANGE_ARGS_OFFSET * self._U32_BYTES)
            )

    def _clear_tile_ranges(self, encoder: spy.CommandEncoder) -> None:
        self._k_clear_ranges.dispatch(
            thread_count=spy.uint3(self.tile_count, 1, 1),
            vars={
                "g_TileRanges": self._work_buffers["tile_ranges"],
                **self._prepass_uniforms(self._scene_count),
            },
            command_encoder=encoder,
        )

    def _build_tile_ranges_indirect(self, encoder: spy.CommandEncoder, args_buffer: spy.Buffer) -> None:
        with encoder.begin_compute_pass() as compute_pass:
            cursor = spy.ShaderCursor(compute_pass.bind_pipeline(self._p_build_ranges))
            cursor.g_SortedKeys = self._work_buffers["keys"]
            cursor.g_TileRanges = self._work_buffers["tile_ranges"]
            cursor.g_PrepassParams = args_buffer
            self._bind_prepass_cursor(cursor, self._scene_count, sorted_count_offset=18)
            compute_pass.dispatch_compute_indirect(
                spy.BufferOffsetPair(args_buffer, GPURadixSort.BUILD_RANGE_ARGS_OFFSET * 4)
            )

    def _rasterize(self, encoder: spy.CommandEncoder, camera: Camera, background: np.ndarray) -> None:
        self._k_raster.dispatch(
            thread_count=spy.uint3(self.width, self.height, 1),
            vars={
                "g_Positions": self._scene_buffers["positions"],
                "g_Scales": self._scene_buffers["scales"],
                "g_Rotations": self._scene_buffers["rotations"],
                "g_ColorAlpha": self._scene_buffers["color_alpha"],
                "g_ScreenCenterRadiusDepth": self._work_buffers["screen_center_radius_depth"],
                "g_ScreenColorAlpha": self._work_buffers["screen_color_alpha"],
                "g_ScreenEllipseConic": self._work_buffers["screen_ellipse_conic"],
                "g_SortedValues": self._work_buffers["values"],
                "g_TileRanges": self._work_buffers["tile_ranges"],
                "g_Output": self._output_texture,
                "g_RasterForwardState": self._raster_forward_state_texture,
                "g_RasterProcessedCount": self._raster_processed_count_texture,
                **self._prepass_uniforms(self._scene_count),
                **self._raster_uniforms(background),
                **self._camera_uniforms(camera),
            },
            command_encoder=encoder,
        )

    def _clear_raster_grads(self, encoder: spy.CommandEncoder, splat_count: int) -> None:
        grad_count = max(int(splat_count) * 4, 1)
        self._k_clear_raster_grads.dispatch(
            thread_count=spy.uint3(grad_count, 1, 1),
            vars={
                "g_GradPositions": self._work_buffers["grad_positions"],
                "g_GradScales": self._work_buffers["grad_scales"],
                "g_GradRotations": self._work_buffers["grad_rotations"],
                "g_GradColorAlpha": self._work_buffers["grad_color_alpha"],
                **self._prepass_uniforms(splat_count),
            },
            command_encoder=encoder,
        )

    def _rasterize_backward(
        self,
        encoder: spy.CommandEncoder,
        camera: Camera,
        background: np.ndarray,
        output_grad: spy.Texture,
    ) -> None:
        self._k_raster_backward.dispatch(
            thread_count=spy.uint3(self.width, self.height, 1),
            vars={
                "g_Positions": self._scene_buffers["positions"],
                "g_Scales": self._scene_buffers["scales"],
                "g_Rotations": self._scene_buffers["rotations"],
                "g_ColorAlpha": self._scene_buffers["color_alpha"],
                "g_ScreenColorAlpha": self._work_buffers["screen_color_alpha"],
                "g_SortedValues": self._work_buffers["values"],
                "g_TileRanges": self._work_buffers["tile_ranges"],
                "g_OutputGrad": output_grad,
                "g_RasterForwardState": self._raster_forward_state_texture,
                "g_RasterProcessedCount": self._raster_processed_count_texture,
                "g_GradPositions": self._work_buffers["grad_positions"],
                "g_GradScales": self._work_buffers["grad_scales"],
                "g_GradRotations": self._work_buffers["grad_rotations"],
                "g_GradColorAlpha": self._work_buffers["grad_color_alpha"],
                **self._prepass_uniforms(self._scene_count),
                **self._raster_uniforms(background),
                **self._camera_uniforms(camera),
            },
            command_encoder=encoder,
        )

    def _execute_prepass(self, scene: GaussianScene, camera: Camera, sync_counts: bool = False) -> tuple[int, int]:
        self._reset_prepass_counters()
        enc_prepass = self.device.create_command_encoder()
        self._project_and_bin(enc_prepass, scene, camera)
        scanline_args_buffer = self._compute_scanline_dispatch_args(enc_prepass)
        self._compose_scanline_key_values_indirect(enc_prepass, scanline_args_buffer)
        self._enqueue_counter_readback(enc_prepass)
        self._clear_tile_ranges(enc_prepass)
        args_buffer = self._sorter.sort_key_values_from_count_buffer(
            encoder=enc_prepass,
            keys_buffer=self._work_buffers["keys"],
            values_buffer=self._work_buffers["values"],
            count_buffer=self._work_buffers["counter"],
            count_offset=0,
            max_count=self._max_list_entries,
            max_bits=self.sort_bits,
        )
        self._build_tile_ranges_indirect(enc_prepass, args_buffer)
        self.device.submit_command_buffer(enc_prepass.finish())

        if sync_counts:
            self.device.wait()
            generated_entries = self._read_counter()
            sorted_count = min(generated_entries, self._max_list_entries)
            self._apply_counter_feedback(generated_entries, self._max_list_entries)
        else:
            generated_entries = self._delayed_generated_entries if self._delayed_stats_valid else 0
            sorted_count = self._delayed_written_entries if self._delayed_stats_valid else 0
        self._counter_readback_frame_id += 1
        return generated_entries, sorted_count

    def _read_image(self) -> np.ndarray:
        if self._output_texture is None:
            raise RuntimeError("Output texture is not initialized.")
        return np.asarray(self._output_texture.to_numpy(), dtype=np.float32).copy()

    def set_scene(self, scene: GaussianScene) -> None:
        self._ensure_scene_buffers(scene.count)
        self._ensure_work_buffers(scene.count)
        self._upload_scene(scene)
        self._current_scene = scene

    def copy_scene_state_to(self, encoder: spy.CommandEncoder, dst: "GaussianRenderer") -> None:
        if self._current_scene is None:
            raise RuntimeError("Source scene is not set.")
        count = int(self._scene_count)
        if count <= 0:
            raise RuntimeError("Source scene is empty.")
        dst._ensure_scene_buffers(count)
        dst._ensure_work_buffers(count)
        copy_bytes = count * 16
        encoder.copy_buffer(dst._scene_buffers["positions"], 0, self._scene_buffers["positions"], 0, copy_bytes)
        encoder.copy_buffer(dst._scene_buffers["scales"], 0, self._scene_buffers["scales"], 0, copy_bytes)
        encoder.copy_buffer(dst._scene_buffers["rotations"], 0, self._scene_buffers["rotations"], 0, copy_bytes)
        encoder.copy_buffer(dst._scene_buffers["color_alpha"], 0, self._scene_buffers["color_alpha"], 0, copy_bytes)
        dst._scene_count = count
        dst._current_scene = self._current_scene

    @property
    def scene_buffers(self) -> dict[str, spy.Buffer]:
        return self._scene_buffers

    @property
    def work_buffers(self) -> dict[str, spy.Buffer]:
        return self._work_buffers

    @property
    def output_texture(self) -> spy.Texture:
        if self._output_texture is None:
            raise RuntimeError("Output texture is not initialized.")
        return self._output_texture

    @property
    def output_grad_texture(self) -> spy.Texture:
        if self._output_grad_texture is None:
            raise RuntimeError("Output grad texture is not initialized.")
        return self._output_grad_texture

    def execute_prepass_for_current_scene(self, camera: Camera, sync_counts: bool = False) -> tuple[int, int]:
        if self._current_scene is None:
            raise RuntimeError("Scene is not set. Call set_scene() before execute_prepass_for_current_scene().")
        self._ensure_work_buffers(self._current_scene.count, self._pending_min_list_entries)
        return self._execute_prepass(self._current_scene, camera, sync_counts=sync_counts)

    def rasterize_current_scene(self, encoder: spy.CommandEncoder, camera: Camera, background: np.ndarray) -> None:
        if self._current_scene is None:
            raise RuntimeError("Scene is not set. Call set_scene() before rasterize_current_scene().")
        self._rasterize(encoder, camera, background)

    def clear_raster_grads_current_scene(self, encoder: spy.CommandEncoder) -> None:
        if self._current_scene is None:
            raise RuntimeError("Scene is not set. Call set_scene() before clear_raster_grads_current_scene().")
        self._clear_raster_grads(encoder, self._current_scene.count)

    def rasterize_backward_current_scene(
        self,
        encoder: spy.CommandEncoder,
        camera: Camera,
        background: np.ndarray,
        output_grad: spy.Texture,
    ) -> None:
        if self._current_scene is None:
            raise RuntimeError("Scene is not set. Call set_scene() before rasterize_backward_current_scene().")
        self._rasterize_backward(encoder, camera, background, output_grad)

    def render_to_texture(
        self,
        camera: Camera,
        background: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
        read_stats: bool = True,
    ) -> tuple[spy.Texture, dict[str, int | bool | float]]:
        if self._current_scene is None:
            raise RuntimeError("Scene is not set. Call set_scene() before render_to_texture().")
        scene = self._current_scene
        if scene.count <= 0:
            raise RuntimeError("Cannot render empty scene.")
        self._ensure_work_buffers(scene.count, self._pending_min_list_entries)
        background_np = np.asarray(background, dtype=np.float32).reshape(3)
        self._execute_prepass(scene, camera, sync_counts=False)
        enc_raster = self.device.create_command_encoder()
        self._rasterize(enc_raster, camera, background_np)
        self.device.submit_command_buffer(enc_raster.finish())
        self.device.wait()
        self._update_delayed_counter_stats()
        self._last_stats = {
            "generated_entries": int(self._delayed_generated_entries) if read_stats and self._delayed_stats_valid else 0,
            "written_entries": int(self._delayed_written_entries) if read_stats and self._delayed_stats_valid else 0,
            "overflow": bool(self._delayed_overflow) if read_stats and self._delayed_stats_valid else False,
            "capacity_limited": bool(self._delayed_overflow) if read_stats and self._delayed_stats_valid else False,
            "depth_bits": int(self.depth_bits),
            "tile_count": int(self.tile_count),
            "splat_count": int(scene.count),
            "max_list_entries": int(self._max_list_entries),
            "max_scanline_entries": int(self._max_scanline_entries),
            "prepass_entry_cap": int(self._max_prepass_entries_by_budget()),
            "prepass_memory_mb": int(self.max_prepass_memory_mb),
            "stats_valid": bool(self._delayed_stats_valid) if read_stats else False,
            "stats_latency_frames": 1,
        }
        if self._output_texture is None:
            raise RuntimeError("Output texture is not initialized.")
        return self._output_texture, self._last_stats

    @property
    def last_stats(self) -> dict[str, int | bool | float]:
        return self._last_stats.copy()

    def render(
        self,
        scene: GaussianScene,
        camera: Camera,
        background: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> RenderOutput:
        if scene.count <= 0:
            return RenderOutput(
                image=np.zeros((self.height, self.width, 4), dtype=np.float32),
                stats={
                    "generated_entries": 0,
                    "written_entries": 0,
                    "overflow": False,
                    "capacity_limited": False,
                    "depth_bits": self.depth_bits,
                    "max_list_entries": int(self._max_list_entries),
                    "max_scanline_entries": int(self._max_scanline_entries),
                    "prepass_entry_cap": int(self._max_prepass_entries_by_budget()),
                    "prepass_memory_mb": int(self.max_prepass_memory_mb),
                    "stats_valid": True,
                    "stats_latency_frames": 1,
                },
            )

        background_np = np.asarray(background, dtype=np.float32).reshape(3)
        self.set_scene(scene)
        _, stats = self.render_to_texture(camera, background_np)
        return RenderOutput(
            image=self._read_image(),
            stats=stats,
        )

    def debug_raster_backward_grads(
        self,
        scene: GaussianScene,
        camera: Camera,
        background: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, np.ndarray]:
        background_np = np.asarray(background, dtype=np.float32).reshape(3)
        self.set_scene(scene)
        self._execute_prepass(scene, camera, sync_counts=False)
        enc = self.device.create_command_encoder()
        self._rasterize(enc, camera, background_np)
        self._clear_raster_grads(enc, scene.count)
        if self._output_texture is None:
            raise RuntimeError("Output texture is not initialized.")
        self._rasterize_backward(enc, camera, background_np, self._output_texture)
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()
        return {
            "grad_positions": self._read_f32x4(self._work_buffers["grad_positions"], scene.count),
            "grad_scales": self._read_f32x4(self._work_buffers["grad_scales"], scene.count),
            "grad_rotations": self._read_f32x4(self._work_buffers["grad_rotations"], scene.count),
            "grad_color_alpha": self._read_f32x4(self._work_buffers["grad_color_alpha"], scene.count),
        }

    def debug_pipeline_data(self, scene: GaussianScene, camera: Camera) -> dict[str, np.ndarray | int]:
        self._ensure_scene_buffers(scene.count)
        self._ensure_work_buffers(scene.count)
        self._upload_scene(scene)
        generated_entries, sorted_count = self._execute_prepass(scene, camera, sync_counts=True)
        return {
            "generated_entries": generated_entries,
            "sorted_count": sorted_count,
            "keys": self._read_u32(self._work_buffers["keys"], sorted_count),
            "values": self._read_u32(self._work_buffers["values"], sorted_count),
            "tile_ranges": self._read_u32(self._work_buffers["tile_ranges"], self.tile_count * 2).reshape(
                self.tile_count, 2
            ),
            "screen_center_radius_depth": self._read_f32x4(
                self._work_buffers["screen_center_radius_depth"], scene.count
            ),
            "screen_color_alpha": self._read_f32x4(self._work_buffers["screen_color_alpha"], scene.count),
            "screen_ellipse_conic": self._read_f32x4(self._work_buffers["screen_ellipse_conic"], scene.count),
        }
