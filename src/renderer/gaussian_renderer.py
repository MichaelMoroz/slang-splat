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

    def __init__(
        self,
        device: spy.Device,
        width: int,
        height: int,
        tile_size: int = 16,
        radius_scale: float = 2.6,
        max_splat_radius_px: float = 512.0,
        alpha_cutoff: float = 1.0 / 255.0,
        max_splat_steps: int = 32768,
        transmittance_threshold: float = 0.005,
        list_capacity_multiplier: int = 64,
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
        self.max_splat_radius_px = float(max_splat_radius_px)
        self.alpha_cutoff = float(alpha_cutoff)
        self.max_splat_steps = int(max_splat_steps)
        self.transmittance_threshold = float(transmittance_threshold)
        self.list_capacity_multiplier = int(list_capacity_multiplier)
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

    def _ensure_scene_buffers(self, splat_count: int) -> None:
        if self._scene_buffers and splat_count <= self._scene_capacity:
            self._scene_count = splat_count
            return
        old_capacity = max(self._scene_capacity, 1)
        new_capacity = max(splat_count, old_capacity + old_capacity // 2)
        usage = self._buffer_usage_sr()
        self._scene_buffers = {
            "positions": self.device.create_buffer(size=max(new_capacity, 1) * 16, usage=usage),
            "scales": self.device.create_buffer(size=max(new_capacity, 1) * 16, usage=usage),
            "rotations": self.device.create_buffer(size=max(new_capacity, 1) * 16, usage=usage),
            "color_alpha": self.device.create_buffer(size=max(new_capacity, 1) * 16, usage=usage),
        }
        self._scene_capacity = new_capacity
        self._scene_count = splat_count

    def _ensure_work_buffers(self, splat_count: int, min_list_entries: int = 0) -> None:
        required_splat_capacity = max(splat_count, 1)
        required_list_entries = max(splat_count * self.list_capacity_multiplier, min_list_entries, 1)
        required_scanline_entries = required_list_entries
        if (
            self._work_buffers
            and required_splat_capacity <= self._work_splat_capacity
            and required_list_entries <= self._max_list_entries
            and required_scanline_entries <= self._max_scanline_entries
            and self._output_texture is not None
        ):
            return
        old_splat_capacity = max(self._work_splat_capacity, 1)
        old_list_capacity = max(self._max_list_entries, 1)
        old_scanline_capacity = max(self._max_scanline_entries, 1)
        splat_capacity = max(required_splat_capacity, old_splat_capacity + old_splat_capacity // 2)
        max_list_entries = max(required_list_entries, old_list_capacity + old_list_capacity // 2)
        max_scanline_entries = max(
            required_scanline_entries,
            old_scanline_capacity + old_scanline_capacity // 2,
        )
        usage = self._buffer_usage_rw()
        self._work_buffers = {
            "screen_center_radius_depth": self.device.create_buffer(
                size=max(splat_capacity, 1) * 16, usage=usage
            ),
            "screen_color_alpha": self.device.create_buffer(size=max(splat_capacity, 1) * 16, usage=usage),
            "screen_ellipse_conic": self.device.create_buffer(size=max(splat_capacity, 1) * 16, usage=usage),
            "splat_pos_local": self.device.create_buffer(size=max(splat_capacity, 1) * 16, usage=usage),
            "splat_inv_scale": self.device.create_buffer(size=max(splat_capacity, 1) * 16, usage=usage),
            "splat_quat": self.device.create_buffer(size=max(splat_capacity, 1) * 16, usage=usage),
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
        }
        if self._output_texture is None:
            self._output_texture = self.device.create_texture(
                format=spy.Format.rgba32_float,
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
        written = min(generated, capacity)
        overflow = generated > capacity
        self._delayed_generated_entries = generated
        self._delayed_written_entries = written
        self._delayed_overflow = overflow
        self._delayed_stats_valid = True
        if overflow:
            self._pending_min_list_entries = max(self._pending_min_list_entries, generated)

    def _project_and_bin(self, encoder: spy.CommandEncoder, scene: GaussianScene, camera: Camera) -> None:
        vars = {
            "g_Positions": self._scene_buffers["positions"],
            "g_Scales": self._scene_buffers["scales"],
            "g_Rotations": self._scene_buffers["rotations"],
            "g_ColorAlpha": self._scene_buffers["color_alpha"],
            "g_ScreenCenterRadiusDepth": self._work_buffers["screen_center_radius_depth"],
            "g_ScreenColorAlpha": self._work_buffers["screen_color_alpha"],
            "g_ScreenEllipseConic": self._work_buffers["screen_ellipse_conic"],
            "g_SplatPosLocal": self._work_buffers["splat_pos_local"],
            "g_SplatInvScale": self._work_buffers["splat_inv_scale"],
            "g_SplatQuat": self._work_buffers["splat_quat"],
            "g_Keys": self._work_buffers["keys"],
            "g_Values": self._work_buffers["values"],
            "g_ListCounter": self._work_buffers["counter"],
            "g_SplatListBases": self._work_buffers["splat_list_bases"],
            "g_ScanlineWorkItems": self._work_buffers["scanline_work_items"],
            "g_ScanlineCounter": self._work_buffers["scanline_counter"],
            "g_splatCount": scene.count,
            "g_tileSize": self.tile_size,
            "g_tileWidth": self.tile_width,
            "g_tileHeight": self.tile_height,
            "g_tileCount": self.tile_count,
            "g_depthBits": self.depth_bits,
            "g_maxListEntries": self._max_list_entries,
            "g_maxScanlineEntries": self._max_scanline_entries,
            "g_radiusScale": self.radius_scale,
            "g_maxSplatRadiusPx": self.max_splat_radius_px,
            "g_sampled5MVEEIters": self.sampled5_mvee_iters,
            "g_sampled5SafetyScale": self.sampled5_safety_scale,
            "g_sampled5RadiusPadPx": self.sampled5_radius_pad_px,
            "g_sampled5Eps": self.sampled5_eps,
            "g_projDistortionK1": self.proj_distortion_k1,
            "g_projDistortionK2": self.proj_distortion_k2,
            **camera.gpu_params(self.width, self.height),
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
            cursor.g_tileWidth = self.tile_width
            cursor.g_depthBits = self.depth_bits
            cursor.g_maxListEntries = self._max_list_entries
            cursor.g_maxScanlineEntries = self._max_scanline_entries
            compute_pass.dispatch_compute_indirect(
                spy.BufferOffsetPair(args_buffer, GPURadixSort.BUILD_RANGE_ARGS_OFFSET * self._U32_BYTES)
            )

    def _clear_tile_ranges(self, encoder: spy.CommandEncoder) -> None:
        self._k_clear_ranges.dispatch(
            thread_count=spy.uint3(self.tile_count, 1, 1),
            vars={
                "g_TileRanges": self._work_buffers["tile_ranges"],
                "g_tileCount": self.tile_count,
            },
            command_encoder=encoder,
        )

    def _build_tile_ranges_indirect(self, encoder: spy.CommandEncoder, args_buffer: spy.Buffer) -> None:
        with encoder.begin_compute_pass() as compute_pass:
            cursor = spy.ShaderCursor(compute_pass.bind_pipeline(self._p_build_ranges))
            cursor.g_SortedKeys = self._work_buffers["keys"]
            cursor.g_TileRanges = self._work_buffers["tile_ranges"]
            cursor.g_PrepassParams = args_buffer
            cursor.g_sortedCountOffset = 18
            cursor.g_depthBits = self.depth_bits
            cursor.g_tileCount = self.tile_count
            compute_pass.dispatch_compute_indirect(
                spy.BufferOffsetPair(args_buffer, GPURadixSort.BUILD_RANGE_ARGS_OFFSET * 4)
            )

    def _rasterize(self, encoder: spy.CommandEncoder, camera: Camera, background: np.ndarray) -> None:
        self._k_raster.dispatch(
            thread_count=spy.uint3(self.width, self.height, 1),
            vars={
                "g_ScreenCenterRadiusDepth": self._work_buffers["screen_center_radius_depth"],
                "g_ScreenColorAlpha": self._work_buffers["screen_color_alpha"],
                "g_ScreenEllipseConic": self._work_buffers["screen_ellipse_conic"],
                "g_SplatPosLocal": self._work_buffers["splat_pos_local"],
                "g_SplatInvScale": self._work_buffers["splat_inv_scale"],
                "g_SplatQuat": self._work_buffers["splat_quat"],
                "g_SortedValues": self._work_buffers["values"],
                "g_TileRanges": self._work_buffers["tile_ranges"],
                "g_Output": self._output_texture,
                "g_splatCount": self._scene_count,
                "g_tileSize": self.tile_size,
                "g_tileWidth": self.tile_width,
                "g_width": self.width,
                "g_height": self.height,
                "g_maxSplatSteps": self.max_splat_steps,
                "g_alphaCutoff": self.alpha_cutoff,
                "g_transmittanceThreshold": self.transmittance_threshold,
                "g_radiusScale": self.radius_scale,
                "g_maxSplatRadiusPx": self.max_splat_radius_px,
                "g_background": spy.float3(*background.tolist()),
                "g_debugShowEllipses": np.uint32(1 if self.debug_show_ellipses else 0),
                "g_debugShowProcessedCount": np.uint32(1 if self.debug_show_processed_count else 0),
                "g_debugEllipseThicknessPx": float(self.debug_ellipse_thickness_px),
                "g_debugEllipseColor": spy.float3(*self.debug_ellipse_color.tolist()),
                **camera.gpu_params(self.width, self.height),
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

    def render_to_texture(
        self,
        camera: Camera,
        background: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
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
            "generated_entries": int(self._delayed_generated_entries) if self._delayed_stats_valid else 0,
            "written_entries": int(self._delayed_written_entries) if self._delayed_stats_valid else 0,
            "overflow": bool(self._delayed_overflow) if self._delayed_stats_valid else False,
            "depth_bits": int(self.depth_bits),
            "tile_count": int(self.tile_count),
            "splat_count": int(scene.count),
            "stats_valid": bool(self._delayed_stats_valid),
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
                    "depth_bits": self.depth_bits,
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
            "splat_pos_local": self._read_f32x4(self._work_buffers["splat_pos_local"], scene.count),
            "splat_inv_scale": self._read_f32x4(self._work_buffers["splat_inv_scale"], scene.count),
            "splat_quat": self._read_f32x4(self._work_buffers["splat_quat"], scene.count),
        }
