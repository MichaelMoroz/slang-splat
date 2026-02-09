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
    def __init__(
        self,
        device: spy.Device,
        width: int,
        height: int,
        tile_size: int = 16,
        radius_scale: float = 2.6,
        max_splat_radius_px: float = 64.0,
        alpha_cutoff: float = 1.0 / 255.0,
        list_capacity_multiplier: int = 64,
    ) -> None:
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.tile_size = int(tile_size)
        self.radius_scale = float(radius_scale)
        self.max_splat_radius_px = float(max_splat_radius_px)
        self.alpha_cutoff = float(alpha_cutoff)
        self.list_capacity_multiplier = int(list_capacity_multiplier)

        self.tile_width = (self.width + self.tile_size - 1) // self.tile_size
        self.tile_height = (self.height + self.tile_size - 1) // self.tile_size
        self.tile_count = self.tile_width * self.tile_height

        self.tile_bits = int(np.ceil(np.log2(max(self.tile_count, 2))))
        self.depth_bits = 32 - self.tile_bits
        self.sort_bits = self.tile_bits + self.depth_bits

        self._shader_path = Path(SHADER_ROOT / "renderer" / "gaussian_pipeline.slang")
        self._create_shaders()
        self._create_sorter()

        self._scene_count = 0
        self._max_list_entries = 0
        self._scene_buffers: dict[str, spy.Buffer] = {}
        self._work_buffers: dict[str, spy.Buffer] = {}
        self._output_texture: spy.Texture | None = None

    def _create_shaders(self) -> None:
        load_program = self.device.load_program
        self._k_project = self.device.create_compute_kernel(
            load_program(str(self._shader_path), ["csProjectAndBin"])
        )
        self._k_clear_ranges = self.device.create_compute_kernel(
            load_program(str(self._shader_path), ["csClearTileRanges"])
        )
        self._k_build_ranges = self.device.create_compute_kernel(
            load_program(str(self._shader_path), ["csBuildTileRanges"])
        )
        self._k_raster = self.device.create_compute_kernel(
            load_program(str(self._shader_path), ["csRasterize"])
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
        if splat_count == self._scene_count:
            return
        usage = self._buffer_usage_sr()
        self._scene_buffers = {
            "positions": self.device.create_buffer(size=max(splat_count, 1) * 16, usage=usage),
            "scales": self.device.create_buffer(size=max(splat_count, 1) * 16, usage=usage),
            "rotations": self.device.create_buffer(size=max(splat_count, 1) * 16, usage=usage),
            "color_alpha": self.device.create_buffer(size=max(splat_count, 1) * 16, usage=usage),
        }
        self._scene_count = splat_count

    def _ensure_work_buffers(self, splat_count: int) -> None:
        max_list_entries = max(splat_count * self.list_capacity_multiplier, 1)
        if max_list_entries == self._max_list_entries and self._output_texture is not None:
            return
        usage = self._buffer_usage_rw()
        self._work_buffers = {
            "screen_center_radius_depth": self.device.create_buffer(
                size=max(splat_count, 1) * 16, usage=usage
            ),
            "screen_color_alpha": self.device.create_buffer(size=max(splat_count, 1) * 16, usage=usage),
            "screen_valid": self.device.create_buffer(size=max(splat_count, 1) * 4, usage=usage),
            "splat_pos_local": self.device.create_buffer(size=max(splat_count, 1) * 16, usage=usage),
            "splat_inv_scale": self.device.create_buffer(size=max(splat_count, 1) * 16, usage=usage),
            "splat_quat": self.device.create_buffer(size=max(splat_count, 1) * 16, usage=usage),
            "keys": self.device.create_buffer(size=max_list_entries * 4, usage=usage),
            "values": self.device.create_buffer(size=max_list_entries * 4, usage=usage),
            "counter": self.device.create_buffer(size=4, usage=usage),
            "tile_ranges": self.device.create_buffer(size=max(self.tile_count, 1) * 8, usage=usage),
        }
        self._output_texture = self.device.create_texture(
            format=spy.Format.rgba32_float,
            width=self.width,
            height=self.height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        )
        self._max_list_entries = max_list_entries

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

    def _reset_counter(self) -> None:
        self._work_buffers["counter"].copy_from_numpy(np.array([0], dtype=np.uint32))

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

    def _project_and_bin(self, encoder: spy.CommandEncoder, scene: GaussianScene, camera: Camera) -> None:
        vars = {
            "g_Positions": self._scene_buffers["positions"],
            "g_Scales": self._scene_buffers["scales"],
            "g_Rotations": self._scene_buffers["rotations"],
            "g_ColorAlpha": self._scene_buffers["color_alpha"],
            "g_ScreenCenterRadiusDepth": self._work_buffers["screen_center_radius_depth"],
            "g_ScreenColorAlpha": self._work_buffers["screen_color_alpha"],
            "g_ScreenValid": self._work_buffers["screen_valid"],
            "g_SplatPosLocal": self._work_buffers["splat_pos_local"],
            "g_SplatInvScale": self._work_buffers["splat_inv_scale"],
            "g_SplatQuat": self._work_buffers["splat_quat"],
            "g_Keys": self._work_buffers["keys"],
            "g_Values": self._work_buffers["values"],
            "g_ListCounter": self._work_buffers["counter"],
            "g_splatCount": scene.count,
            "g_tileSize": self.tile_size,
            "g_tileWidth": self.tile_width,
            "g_tileHeight": self.tile_height,
            "g_depthBits": self.depth_bits,
            "g_maxListEntries": self._max_list_entries,
            "g_radiusScale": self.radius_scale,
            "g_maxSplatRadiusPx": self.max_splat_radius_px,
            **camera.gpu_params(self.width, self.height),
        }
        self._k_project.dispatch(
            thread_count=spy.uint3(scene.count, 1, 1),
            vars=vars,
            command_encoder=encoder,
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

    def _build_tile_ranges(self, encoder: spy.CommandEncoder, sorted_count: int) -> None:
        if sorted_count <= 0:
            return
        self._k_build_ranges.dispatch(
            thread_count=spy.uint3(sorted_count, 1, 1),
            vars={
                "g_SortedKeys": self._work_buffers["keys"],
                "g_TileRanges": self._work_buffers["tile_ranges"],
                "g_sortedCount": sorted_count,
                "g_depthBits": self.depth_bits,
            },
            command_encoder=encoder,
        )

    def _rasterize(self, encoder: spy.CommandEncoder, camera: Camera, background: np.ndarray) -> None:
        self._k_raster.dispatch(
            thread_count=spy.uint3(self.width, self.height, 1),
            vars={
                "g_ScreenCenterRadiusDepth": self._work_buffers["screen_center_radius_depth"],
                "g_ScreenColorAlpha": self._work_buffers["screen_color_alpha"],
                "g_ScreenValid": self._work_buffers["screen_valid"],
                "g_SplatPosLocal": self._work_buffers["splat_pos_local"],
                "g_SplatInvScale": self._work_buffers["splat_inv_scale"],
                "g_SplatQuat": self._work_buffers["splat_quat"],
                "g_SortedValues": self._work_buffers["values"],
                "g_TileRanges": self._work_buffers["tile_ranges"],
                "g_Output": self._output_texture,
                "g_tileSize": self.tile_size,
                "g_tileWidth": self.tile_width,
                "g_width": self.width,
                "g_height": self.height,
                "g_alphaCutoff": self.alpha_cutoff,
                "g_radiusScale": self.radius_scale,
                "g_maxSplatRadiusPx": self.max_splat_radius_px,
                "g_background": spy.float3(*background.tolist()),
                **camera.gpu_params(self.width, self.height),
            },
            command_encoder=encoder,
        )

    def _execute_prepass(self, scene: GaussianScene, camera: Camera) -> tuple[int, int]:
        self._reset_counter()
        enc_project = self.device.create_command_encoder()
        self._project_and_bin(enc_project, scene, camera)
        self.device.submit_command_buffer(enc_project.finish())
        self.device.wait()

        generated_entries = self._read_counter()
        sorted_count = min(generated_entries, self._max_list_entries)

        enc_sort = self.device.create_command_encoder()
        self._clear_tile_ranges(enc_sort)
        if sorted_count > 0:
            self._sorter.sort_key_values(
                enc_sort,
                self._work_buffers["keys"],
                self._work_buffers["values"],
                sorted_count,
                max_bits=self.sort_bits,
            )
            self._build_tile_ranges(enc_sort, sorted_count)
        self.device.submit_command_buffer(enc_sort.finish())
        self.device.wait()
        return generated_entries, sorted_count

    def _read_image(self) -> np.ndarray:
        if self._output_texture is None:
            raise RuntimeError("Output texture is not initialized.")
        return np.asarray(self._output_texture.to_numpy(), dtype=np.float32).copy()

    def render(
        self,
        scene: GaussianScene,
        camera: Camera,
        background: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> RenderOutput:
        if scene.count <= 0:
            return RenderOutput(
                image=np.zeros((self.height, self.width, 4), dtype=np.float32),
                stats={"generated_entries": 0, "written_entries": 0, "overflow": False, "depth_bits": self.depth_bits},
            )

        background_np = np.asarray(background, dtype=np.float32).reshape(3)
        self._ensure_scene_buffers(scene.count)
        self._ensure_work_buffers(scene.count)
        self._upload_scene(scene)

        generated_entries, sorted_count = self._execute_prepass(scene, camera)

        enc_raster = self.device.create_command_encoder()
        self._rasterize(enc_raster, camera, background_np)
        self.device.submit_command_buffer(enc_raster.finish())
        self.device.wait()

        return RenderOutput(
            image=self._read_image(),
            stats={
                "generated_entries": int(generated_entries),
                "written_entries": int(sorted_count),
                "overflow": bool(generated_entries > sorted_count),
                "depth_bits": int(self.depth_bits),
                "tile_count": int(self.tile_count),
            },
        )

    def debug_pipeline_data(self, scene: GaussianScene, camera: Camera) -> dict[str, np.ndarray | int]:
        self._ensure_scene_buffers(scene.count)
        self._ensure_work_buffers(scene.count)
        self._upload_scene(scene)
        generated_entries, sorted_count = self._execute_prepass(scene, camera)
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
            "screen_valid": self._read_u32(self._work_buffers["screen_valid"], scene.count),
            "splat_pos_local": self._read_f32x4(self._work_buffers["splat_pos_local"], scene.count),
            "splat_inv_scale": self._read_f32x4(self._work_buffers["splat_inv_scale"], scene.count),
            "splat_quat": self._read_f32x4(self._work_buffers["splat_quat"], scene.count),
        }
