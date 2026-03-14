from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
import operator
from pathlib import Path
import re

import numpy as np
import slangpy as spy

from ..common import SHADER_ROOT, buffer_to_numpy, debug_region, remap_named_buffers
from ..scene.gaussian_scene import GaussianScene
from ..sort.radix_sort import GPURadixSort
from .camera import Camera


@dataclass(slots=True)
class RenderOutput:
    image: np.ndarray
    stats: dict[str, int | bool | float]


@dataclass(slots=True)
class SceneBinding:
    count: int


@dataclass(frozen=True, slots=True)
class RasterConfig:
    thread_tile_dim: int
    microtile_dim: int
    effective_tile_size: int
    batch: int


class _UIntExprEvaluator(ast.NodeVisitor):
    _BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.floordiv, ast.FloorDiv: operator.floordiv}
    _UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def __init__(self, expr: str, constants: dict[str, int]) -> None:
        self.expr = expr
        self.constants = constants

    def visit_Expression(self, node: ast.Expression) -> int:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> int:
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError(f"Unsupported constant expression value: {self.expr}")
        return int(node.value)

    def visit_Name(self, node: ast.Name) -> int:
        if node.id not in self.constants:
            raise ValueError(f"Unknown constant '{node.id}' in expression: {self.expr}")
        return self.constants[node.id]

    def visit_UnaryOp(self, node: ast.UnaryOp) -> int:
        op = self._UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator in expression: {self.expr}")
        return int(op(self.visit(node.operand)))

    def visit_BinOp(self, node: ast.BinOp) -> int:
        op = self._BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported binary operator in expression: {self.expr}")
        lhs, rhs = self.visit(node.left), self.visit(node.right)
        if rhs == 0 and op is operator.floordiv:
            raise ValueError(f"Division by zero in expression: {self.expr}")
        return int(op(lhs, rhs))

    def generic_visit(self, node: ast.AST) -> int:
        raise ValueError(f"Unsupported AST node in expression: {self.expr}")


class GaussianRenderer:
    _COUNTER_READBACK_RING_SIZE = 2
    _SCANLINE_WORK_ITEM_UINTS = 8
    _U32_BYTES = 4
    _OPACITY_EPS = 1e-6
    _MEBIBYTE_BYTES = 1024 * 1024
    _PREPASS_ENTRY_BYTES = (_SCANLINE_WORK_ITEM_UINTS + 2) * _U32_BYTES
    _RW_BUFFER_USAGE = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
    PARAM_POSITION_IDS = (0, 1, 2)
    PARAM_SCALE_IDS = (3, 4, 5)
    PARAM_ROTATION_IDS = (6, 7, 8, 9)
    PARAM_COLOR_IDS = (10, 11, 12)
    PARAM_RAW_OPACITY_ID = 13
    TRAINABLE_PARAM_COUNT = 14
    _SCENE_SHADER_VARS = {"splat_params": "g_SplatParams"}
    _SCREEN_SHADER_VARS = {"screen_center_radius_depth": "g_ScreenCenterRadiusDepth", "screen_color_alpha": "g_ScreenColorAlpha", "screen_ellipse_conic": "g_ScreenEllipseConic", "splat_visible": "g_SplatVisible"}
    _GRAD_SHADER_VARS = {"param_grads": "g_ParamGrads"}
    _PREPASS_CURSOR_FIELDS = ("splatCount", "tileSize", "tileWidth", "tileHeight", "tileCount", "depthBits", "sortedCountOffset", "maxListEntries", "maxScanlineEntries", "radiusScale", "sampled5MVEEIters", "sampled5SafetyScale", "sampled5RadiusPadPx", "sampled5Eps")
    _SHADERS = (
        ("_k_project", "kernel", "gaussian_project_stage.slang", "csProjectAndBin"),
        ("_p_compose_scanline", "pipeline", "gaussian_project_stage.slang", "csComposeScanlineKeyValues"),
        ("_k_clear_ranges", "kernel", "gaussian_project_stage.slang", "csClearTileRanges"),
        ("_p_build_ranges", "pipeline", "gaussian_project_stage.slang", "csBuildTileRanges"),
        ("_k_raster", "kernel", "gaussian_raster_stage.slang", "csRasterize"),
        ("_k_raster_training_forward", "kernel", "gaussian_raster_stage.slang", "csRasterizeTrainingForward"),
        ("_k_clear_raster_grads", "kernel", "gaussian_raster_stage.slang", "csClearRasterGrads"),
        ("_k_raster_backward", "kernel", "gaussian_raster_stage.slang", "csRasterizeBackward"),
    )
    _buffer_vars = staticmethod(remap_named_buffers)

    def _dispatch(self, kernel: spy.ComputeKernel | spy.ComputePipeline, encoder: spy.CommandEncoder, thread_count: spy.uint3, vars: dict[str, object], label: str, color_index: int) -> None:
        with debug_region(encoder, label, color_index):
            kernel.dispatch(thread_count=thread_count, vars=vars, command_encoder=encoder)

    @staticmethod
    def _grow(required: int, current: int) -> int:
        return max(required, max(current, 1) + max(current, 1) // 2)

    def _max_prepass_entries_by_budget(self) -> int:
        return max(self._max_prepass_memory_bytes // self._PREPASS_ENTRY_BYTES, 1)

    @staticmethod
    def _background_array(background: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        return np.asarray(background, dtype=np.float32).reshape(3)

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
                "debugShowGradNorm": np.uint32(1 if self.debug_show_grad_norm else 0),
                "debugGradNormThreshold": float(max(self.debug_grad_norm_threshold, 0.0)),
                "debugEllipseThicknessPx": float(self.debug_ellipse_thickness_px),
            }
        }

    def _scene_vars(self) -> dict[str, object]:
        return self._buffer_vars(self._SCENE_SHADER_VARS, self._scene_buffers)

    def _screen_vars(self) -> dict[str, object]:
        return self._buffer_vars(self._SCREEN_SHADER_VARS, self._work_buffers)

    def _grad_vars(self) -> dict[str, object]:
        return self._buffer_vars(self._GRAD_SHADER_VARS, self._work_buffers)

    def _debug_grad_norm_var(self) -> dict[str, object]:
        return {"g_DebugGradNorm": self._debug_grad_norm_buffer if self._debug_grad_norm_buffer is not None else self._work_buffers["debug_grad_norm"]}

    def _raster_thread_count(self) -> spy.uint3:
        return spy.uint3(
            (self.width + self._raster_config.microtile_dim - 1) // self._raster_config.microtile_dim,
            (self.height + self._raster_config.microtile_dim - 1) // self._raster_config.microtile_dim,
            1,
        )

    def _read_image(self) -> np.ndarray:
        return np.asarray(self.output_texture.to_numpy(), dtype=np.float32).copy()

    def _create_shaders(self) -> None:
        for attr, kind, shader_name, entry in self._SHADERS:
            program = self.device.load_program(str(Path(SHADER_ROOT / "renderer" / shader_name)), [entry])
            shader = self.device.create_compute_kernel(program) if kind == "kernel" else self.device.create_compute_pipeline(program)
            setattr(self, attr, shader)

    def _upload_scene(self, scene: GaussianScene) -> None:
        self._scene_buffers["splat_params"].copy_from_numpy(self._pack_scene(scene))

    def _reset_prepass_counters(self) -> None:
        for name in ("counter", "scanline_counter"):
            self._work_buffers[name].copy_from_numpy(np.array([0], dtype=np.uint32))

    def set_scene(self, scene: GaussianScene) -> None:
        self._ensure_scene_buffers(scene.count)
        self._ensure_work_buffers(scene.count)
        self._upload_scene(scene)
        self._current_scene = scene

    def bind_scene_count(self, splat_count: int) -> None:
        count = max(int(splat_count), 0)
        self._ensure_scene_buffers(count)
        self._ensure_work_buffers(count)
        self._current_scene = SceneBinding(count=count)

    def _bind_prepass_cursor(self, cursor: spy.ShaderCursor, splat_count: int, sorted_count_offset: int = 0) -> None:
        prepass = self._prepass_uniforms(splat_count, sorted_count_offset)["g_Prepass"]
        for name in self._PREPASS_CURSOR_FIELDS:
            setattr(cursor.g_Prepass, name, prepass[name])

    def _enqueue_counter_readback(self, encoder: spy.CommandEncoder) -> None:
        self._ensure_counter_readback_ring()
        slot = self._counter_readback_frame_id % self._COUNTER_READBACK_RING_SIZE
        encoder.copy_buffer(self._counter_readback_ring[slot], 0, self._work_buffers["counter"], 0, 4)
        self._counter_readback_capacity[slot] = int(self._max_list_entries)

    def __init__(
        self,
        device: spy.Device,
        width: int,
        height: int,
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
        debug_show_grad_norm: bool = False,
        debug_grad_norm_threshold: float = 2e-4,
        debug_ellipse_thickness_px: float = 2.0,
    ) -> None:
        self.device, self.width, self.height = device, int(width), int(height)
        self._types_shader_path = Path(SHADER_ROOT / "renderer" / "gaussian_types.slang")
        self._project_shader_path = Path(SHADER_ROOT / "renderer" / "gaussian_project_stage.slang")
        self._raster_shader_path = Path(SHADER_ROOT / "renderer" / "gaussian_raster_stage.slang")
        self._raster_config = self._load_raster_config(self._types_shader_path)
        self.tile_size = self._raster_config.effective_tile_size
        self.radius_scale, self.alpha_cutoff = float(radius_scale), float(alpha_cutoff)
        self.max_splat_steps, self.transmittance_threshold = int(max_splat_steps), float(transmittance_threshold)
        self.list_capacity_multiplier = int(list_capacity_multiplier)
        self.max_prepass_memory_mb = max(int(max_prepass_memory_mb), 1)
        self._max_prepass_memory_bytes = self.max_prepass_memory_mb * self._MEBIBYTE_BYTES
        self.sampled5_mvee_iters, self.sampled5_safety_scale = int(sampled5_mvee_iters), float(sampled5_safety_scale)
        self.sampled5_radius_pad_px, self.sampled5_eps = float(sampled5_radius_pad_px), float(sampled5_eps)
        self.proj_distortion_k1, self.proj_distortion_k2 = float(proj_distortion_k1), float(proj_distortion_k2)
        self.debug_show_ellipses, self.debug_show_processed_count, self.debug_show_grad_norm = bool(debug_show_ellipses), bool(debug_show_processed_count), bool(debug_show_grad_norm)
        self.debug_grad_norm_threshold = float(debug_grad_norm_threshold)
        self.debug_ellipse_thickness_px = float(debug_ellipse_thickness_px)
        self.tile_width, self.tile_height = (self.width + self.tile_size - 1) // self.tile_size, (self.height + self.tile_size - 1) // self.tile_size
        self.tile_count = self.tile_width * self.tile_height
        self.tile_bits = int(np.ceil(np.log2(max(self.tile_count, 2))))
        self.depth_bits, self.sort_bits = 32 - self.tile_bits, 32
        self._create_shaders()
        self._sorter = GPURadixSort(self.device)
        self._scene_count = self._scene_capacity = self._max_list_entries = self._work_splat_capacity = self._max_scanline_entries = 0
        self._current_scene: GaussianScene | None = None
        self._scene_buffers: dict[str, spy.Buffer] = {}
        self._work_buffers: dict[str, spy.Buffer] = {}
        self._debug_grad_norm_buffer: spy.Buffer | None = None
        self._output_texture: spy.Texture | None = None
        self._output_grad_buffer: spy.Buffer | None = None
        self._last_stats: dict[str, int | bool | float] = {}
        self._counter_readback_ring: list[spy.Buffer] = []
        self._counter_readback_capacity: list[int] = []
        self._counter_readback_frame_id = 0
        self._pending_min_list_entries = self._delayed_generated_entries = self._delayed_written_entries = 0
        self._delayed_overflow = self._delayed_stats_valid = False

    @staticmethod
    def _eval_uint_constant_expr(expr: str, constants: dict[str, int]) -> int:
        value = _UIntExprEvaluator(expr, constants).visit(ast.parse(expr, mode="eval"))
        if value < 0:
            raise ValueError(f"Expected non-negative uint expression: {expr}")
        return value

    @classmethod
    @lru_cache(maxsize=1)
    def _load_raster_config(cls, shader_path: Path) -> RasterConfig:
        source = shader_path.read_text(encoding="utf-8")
        constants: dict[str, int] = {}
        for name, expr in re.compile(r"static\s+const\s+uint\s+(\w+)\s*=\s*([^;]+);").findall(source):
            constants[name] = cls._eval_uint_constant_expr(re.sub(r"(?<=[0-9A-Fa-f])[uU]\b", "", expr).strip(), constants)
        required = ("RASTER_THREAD_TILE_DIM", "RASTER_MICROTILE_DIM", "RASTER_EFFECTIVE_TILE_SIZE", "RASTER_BATCH")
        missing = [name for name in required if name not in constants]
        if missing:
            raise RuntimeError(f"Missing raster constants in {shader_path}: {', '.join(missing)}")
        return RasterConfig(constants["RASTER_THREAD_TILE_DIM"], constants["RASTER_MICROTILE_DIM"], constants["RASTER_EFFECTIVE_TILE_SIZE"], constants["RASTER_BATCH"])

    def _ensure_scene_buffers(self, splat_count: int) -> None:
        if self._scene_buffers and splat_count <= self._scene_capacity:
            self._scene_count = splat_count
            return
        self._scene_capacity, self._scene_count = self._grow(splat_count, self._scene_capacity), splat_count
        param_bytes = max(self._scene_capacity, 1) * self.TRAINABLE_PARAM_COUNT * self._U32_BYTES
        self._scene_buffers = {name: self.device.create_buffer(size=param_bytes, usage=self._RW_BUFFER_USAGE) for name in self._SCENE_SHADER_VARS}

    def _ensure_work_buffers(self, splat_count: int, min_list_entries: int = 0) -> None:
        max_entries = self._max_prepass_entries_by_budget()
        required_splats = max(splat_count, 1)
        required_entries = min(max(splat_count * self.list_capacity_multiplier, min_list_entries, 1), max_entries)
        if self._work_buffers and required_splats <= self._work_splat_capacity and required_entries <= self._max_list_entries and required_entries <= self._max_scanline_entries and self._output_texture is not None and self._output_grad_buffer is not None:
            return
        self._work_splat_capacity = self._grow(required_splats, self._work_splat_capacity)
        self._max_list_entries = min(self._grow(required_entries, self._max_list_entries), max_entries)
        self._max_scanline_entries = min(self._grow(required_entries, self._max_scanline_entries), max_entries)
        sized = {
            "screen_center_radius_depth": max(self._work_splat_capacity, 1) * 16,
            "screen_color_alpha": max(self._work_splat_capacity, 1) * 16,
            "screen_ellipse_conic": max(self._work_splat_capacity, 1) * 16,
            "splat_visible": max(self._work_splat_capacity, 1) * self._U32_BYTES,
            "debug_grad_norm": max(self._work_splat_capacity, 1) * self._U32_BYTES,
            "keys": self._max_list_entries * 4,
            "values": self._max_list_entries * 4,
            "counter": 4,
            "splat_list_bases": max(self._work_splat_capacity, 1) * self._U32_BYTES,
            "scanline_work_items": self._max_scanline_entries * self._SCANLINE_WORK_ITEM_UINTS * self._U32_BYTES,
            "scanline_counter": self._U32_BYTES,
            "tile_ranges": max(self.tile_count, 1) * 8,
            "training_forward_state": max(self.width * self.height, 1) * 16,
            "training_processed_end": max(self.width * self.height, 1) * self._U32_BYTES,
            **{name: max(self._work_splat_capacity, 1) * self.TRAINABLE_PARAM_COUNT * self._U32_BYTES for name in self._GRAD_SHADER_VARS},
        }
        self._work_buffers = {name: self.device.create_buffer(size=size, usage=self._RW_BUFFER_USAGE) for name, size in sized.items()}
        self._work_buffers["debug_grad_norm"].copy_from_numpy(np.zeros((max(self._work_splat_capacity, 1),), dtype=np.float32))
        self._ensure_output_texture()
        self._ensure_output_grad_buffer()
        if self._pending_min_list_entries > 0 and self._max_list_entries >= self._pending_min_list_entries:
            self._pending_min_list_entries = 0

    def _ensure_output_texture(self) -> None:
        if self._output_texture is None:
            self._output_texture = self.device.create_texture(
                format=spy.Format.rgba32_float,
                width=self.width,
                height=self.height,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            )

    def _ensure_output_grad_buffer(self) -> None:
        if self._output_grad_buffer is None:
            self._output_grad_buffer = self.device.create_buffer(
                size=max(self.width * self.height, 1) * 4 * self._U32_BYTES,
                usage=self._RW_BUFFER_USAGE,
            )

    @classmethod
    def _param_slice(cls, splat_count: int, param_id: int) -> slice:
        start = int(param_id) * int(splat_count)
        return slice(start, start + int(splat_count))

    @classmethod
    def _pack_scene(cls, scene: GaussianScene) -> np.ndarray:
        count = max(int(scene.count), 0)
        packed = np.zeros((cls.TRAINABLE_PARAM_COUNT * max(count, 1),), dtype=np.float32)
        if count <= 0:
            return packed
        for axis, param_id in enumerate(cls.PARAM_POSITION_IDS):
            packed[cls._param_slice(count, param_id)] = np.asarray(scene.positions[:, axis], dtype=np.float32)
        for axis, param_id in enumerate(cls.PARAM_SCALE_IDS):
            packed[cls._param_slice(count, param_id)] = np.asarray(scene.scales[:, axis], dtype=np.float32)
        for axis, param_id in enumerate(cls.PARAM_ROTATION_IDS):
            packed[cls._param_slice(count, param_id)] = np.asarray(scene.rotations[:, axis], dtype=np.float32)
        for axis, param_id in enumerate(cls.PARAM_COLOR_IDS):
            packed[cls._param_slice(count, param_id)] = np.asarray(scene.colors[:, axis], dtype=np.float32)
        packed[cls._param_slice(count, cls.PARAM_RAW_OPACITY_ID)] = cls._raw_opacity_from_alpha(scene.opacities)
        return packed

    @classmethod
    def _unpack_param_groups(cls, packed: np.ndarray, splat_count: int) -> dict[str, np.ndarray]:
        count = max(int(splat_count), 0)
        flat = np.asarray(packed, dtype=np.float32).reshape(-1)
        groups = {
            "positions": np.zeros((count, 4), dtype=np.float32),
            "scales": np.zeros((count, 4), dtype=np.float32),
            "rotations": np.zeros((count, 4), dtype=np.float32),
            "color_alpha": np.zeros((count, 4), dtype=np.float32),
        }
        if count <= 0:
            return groups
        for axis, param_id in enumerate(cls.PARAM_POSITION_IDS):
            groups["positions"][:, axis] = flat[cls._param_slice(count, param_id)]
        for axis, param_id in enumerate(cls.PARAM_SCALE_IDS):
            groups["scales"][:, axis] = flat[cls._param_slice(count, param_id)]
        for axis, param_id in enumerate(cls.PARAM_ROTATION_IDS):
            groups["rotations"][:, axis] = flat[cls._param_slice(count, param_id)]
        for axis, param_id in enumerate(cls.PARAM_COLOR_IDS):
            groups["color_alpha"][:, axis] = flat[cls._param_slice(count, param_id)]
        groups["color_alpha"][:, 3] = flat[cls._param_slice(count, cls.PARAM_RAW_OPACITY_ID)]
        return groups

    @classmethod
    def _pack_param_groups(
        cls,
        splat_count: int,
        *,
        positions: np.ndarray | None = None,
        scales: np.ndarray | None = None,
        rotations: np.ndarray | None = None,
        color_alpha: np.ndarray | None = None,
    ) -> np.ndarray:
        count = max(int(splat_count), 0)
        packed = np.zeros((cls.TRAINABLE_PARAM_COUNT * max(count, 1),), dtype=np.float32)
        if count <= 0:
            return packed
        groups = {
            "positions": np.zeros((count, 4), dtype=np.float32) if positions is None else np.asarray(positions, dtype=np.float32).reshape(count, 4),
            "scales": np.zeros((count, 4), dtype=np.float32) if scales is None else np.asarray(scales, dtype=np.float32).reshape(count, 4),
            "rotations": np.zeros((count, 4), dtype=np.float32) if rotations is None else np.asarray(rotations, dtype=np.float32).reshape(count, 4),
            "color_alpha": np.zeros((count, 4), dtype=np.float32) if color_alpha is None else np.asarray(color_alpha, dtype=np.float32).reshape(count, 4),
        }
        for axis, param_id in enumerate(cls.PARAM_POSITION_IDS):
            packed[cls._param_slice(count, param_id)] = groups["positions"][:, axis]
        for axis, param_id in enumerate(cls.PARAM_SCALE_IDS):
            packed[cls._param_slice(count, param_id)] = groups["scales"][:, axis]
        for axis, param_id in enumerate(cls.PARAM_ROTATION_IDS):
            packed[cls._param_slice(count, param_id)] = groups["rotations"][:, axis]
        for axis, param_id in enumerate(cls.PARAM_COLOR_IDS):
            packed[cls._param_slice(count, param_id)] = groups["color_alpha"][:, axis]
        packed[cls._param_slice(count, cls.PARAM_RAW_OPACITY_ID)] = groups["color_alpha"][:, 3]
        return packed

    @classmethod
    def _raw_opacity_from_alpha(cls, opacity: np.ndarray) -> np.ndarray:
        alpha = np.clip(np.asarray(opacity, dtype=np.float32), cls._OPACITY_EPS, 1.0 - cls._OPACITY_EPS)
        return (np.log(alpha) - np.log1p(-alpha)).astype(np.float32, copy=False)

    def _read_array(self, buffer: spy.Buffer, dtype: np.dtype, count: int, width: int = 1) -> np.ndarray:
        values = buffer_to_numpy(buffer, dtype)
        values = values[: count * width].copy()
        return values if width == 1 else values.reshape(count, width)

    def _ensure_counter_readback_ring(self) -> None:
        if self._counter_readback_ring:
            return
        usage = spy.BufferUsage.copy_destination | spy.BufferUsage.copy_source
        self._counter_readback_ring = [self.device.create_buffer(size=4, usage=usage) for _ in range(self._COUNTER_READBACK_RING_SIZE)]
        self._counter_readback_capacity = [0] * self._COUNTER_READBACK_RING_SIZE

    def _update_delayed_counter_stats(self) -> None:
        if self._counter_readback_frame_id <= 1:
            self._delayed_stats_valid = False
            return
        slot = (self._counter_readback_frame_id - 2) % self._COUNTER_READBACK_RING_SIZE
        generated = int(self._read_array(self._counter_readback_ring[slot], np.uint32, 1)[0])
        capacity = int(self._counter_readback_capacity[slot])
        self._delayed_generated_entries, self._delayed_written_entries = generated, min(generated, capacity)
        self._delayed_overflow, self._delayed_stats_valid = generated > capacity, True
        if self._delayed_overflow:
            self._pending_min_list_entries = max(self._pending_min_list_entries, min(generated, self._max_prepass_entries_by_budget()))

    def _stats_payload(self, splat_count: int, read_stats: bool, stats_valid: bool | None = None) -> dict[str, int | bool | float]:
        valid = bool(self._delayed_stats_valid) if stats_valid is None else bool(stats_valid)
        if read_stats and valid:
            generated, written, overflow = int(self._delayed_generated_entries), int(self._delayed_written_entries), bool(self._delayed_overflow)
        else:
            generated = written = 0
            overflow = False
        return {"generated_entries": generated, "written_entries": written, "overflow": overflow, "capacity_limited": overflow, "depth_bits": int(self.depth_bits), "tile_count": int(self.tile_count), "splat_count": int(splat_count), "max_list_entries": int(self._max_list_entries), "max_scanline_entries": int(self._max_scanline_entries), "prepass_entry_cap": int(self._max_prepass_entries_by_budget()), "prepass_memory_mb": int(self.max_prepass_memory_mb), "stats_valid": bool(valid) if read_stats else False, "stats_latency_frames": 1}

    def _project_and_bin(self, encoder: spy.CommandEncoder, scene: GaussianScene, camera: Camera) -> None:
        self._dispatch(self._k_project, encoder, spy.uint3(scene.count, 1, 1), {**self._scene_vars(), **self._screen_vars(), "g_Keys": self._work_buffers["keys"], "g_Values": self._work_buffers["values"], "g_ListCounter": self._work_buffers["counter"], "g_SplatListBases": self._work_buffers["splat_list_bases"], "g_ScanlineWorkItems": self._work_buffers["scanline_work_items"], "g_ScanlineCounter": self._work_buffers["scanline_counter"], **self._prepass_uniforms(scene.count), **self._camera_uniforms(camera)}, "Project And Bin", 20)

    def _compute_scanline_dispatch_args(self, encoder: spy.CommandEncoder) -> spy.Buffer:
        args_buffer = self._sorter.ensure_indirect_args()
        self._sorter.compute_indirect_args_from_buffer_dispatch(encoder=encoder, count_buffer=self._work_buffers["scanline_counter"], count_offset=0, max_element_count=self._max_scanline_entries, args_buffer=args_buffer)
        return args_buffer

    def _compose_scanline_key_values_indirect(self, encoder: spy.CommandEncoder, args_buffer: spy.Buffer) -> None:
        with encoder.begin_compute_pass() as compute_pass:
            with debug_region(compute_pass, "Compose Scanline Key Values", 21):
                cursor = spy.ShaderCursor(compute_pass.bind_pipeline(self._p_compose_scanline))
                for name, buffer in {"g_ScanlineWorkItems": self._work_buffers["scanline_work_items"], "g_ScanlineCounter": self._work_buffers["scanline_counter"], "g_SplatListBases": self._work_buffers["splat_list_bases"], "g_Keys": self._work_buffers["keys"], "g_Values": self._work_buffers["values"]}.items():
                    setattr(cursor, name, buffer)
                self._bind_prepass_cursor(cursor, self._scene_count)
                compute_pass.dispatch_compute_indirect(spy.BufferOffsetPair(args_buffer, GPURadixSort.BUILD_RANGE_ARGS_OFFSET * self._U32_BYTES))

    def _clear_tile_ranges(self, encoder: spy.CommandEncoder) -> None:
        self._dispatch(self._k_clear_ranges, encoder, spy.uint3(self.tile_count, 1, 1), {"g_TileRanges": self._work_buffers["tile_ranges"], **self._prepass_uniforms(self._scene_count)}, "Clear Tile Ranges", 22)

    def _build_tile_ranges_indirect(self, encoder: spy.CommandEncoder, args_buffer: spy.Buffer) -> None:
        with encoder.begin_compute_pass() as compute_pass:
            with debug_region(compute_pass, "Build Tile Ranges", 23):
                cursor = spy.ShaderCursor(compute_pass.bind_pipeline(self._p_build_ranges))
                cursor.g_SortedKeys, cursor.g_TileRanges, cursor.g_PrepassParams = self._work_buffers["keys"], self._work_buffers["tile_ranges"], args_buffer
                self._bind_prepass_cursor(cursor, self._scene_count, sorted_count_offset=18)
                compute_pass.dispatch_compute_indirect(spy.BufferOffsetPair(args_buffer, GPURadixSort.BUILD_RANGE_ARGS_OFFSET * self._U32_BYTES))

    def _debug_render_enabled(self) -> bool:
        return bool(self.debug_show_processed_count or self.debug_show_grad_norm or self.debug_show_ellipses)

    def _rasterize(self, encoder: spy.CommandEncoder, camera: Camera, background: np.ndarray, output: spy.Texture | None = None) -> None:
        target = self.output_texture if output is None else output
        self._dispatch(self._k_raster, encoder, self._raster_thread_count(), {**self._scene_vars(), **self._screen_vars(), **self._debug_grad_norm_var(), "g_SortedValues": self._work_buffers["values"], "g_TileRanges": self._work_buffers["tile_ranges"], "g_Output": target, **self._prepass_uniforms(self._scene_count), **self._raster_uniforms(background), **self._camera_uniforms(camera)}, "Rasterize", 24)

    def _clear_raster_grads(self, encoder: spy.CommandEncoder, splat_count: int) -> None:
        self._dispatch(self._k_clear_raster_grads, encoder, spy.uint3(max(int(splat_count) * self.TRAINABLE_PARAM_COUNT, 1), 1, 1), {**self._grad_vars(), **self._prepass_uniforms(splat_count)}, "Clear Raster Grads", 25)

    def _rasterize_training_forward(self, encoder: spy.CommandEncoder, camera: Camera, background: np.ndarray, output: spy.Texture | None = None) -> None:
        target = self.output_texture if output is None else output
        self._dispatch(self._k_raster_training_forward, encoder, self._raster_thread_count(), {**self._scene_vars(), **self._screen_vars(), "g_SortedValues": self._work_buffers["values"], "g_TileRanges": self._work_buffers["tile_ranges"], "g_Output": target, "g_TrainingForwardState": self._work_buffers["training_forward_state"], "g_TrainingProcessedEnd": self._work_buffers["training_processed_end"], **self._prepass_uniforms(self._scene_count), **self._raster_uniforms(background), **self._camera_uniforms(camera)}, "Rasterize Training Forward", 26)

    def _rasterize_backward(self, encoder: spy.CommandEncoder, camera: Camera, background: np.ndarray, output_grad: spy.Buffer) -> None:
        self._dispatch(self._k_raster_backward, encoder, self._raster_thread_count(), {**self._scene_vars(), "g_ScreenColorAlpha": self._work_buffers["screen_color_alpha"], "g_SortedValues": self._work_buffers["values"], "g_TileRanges": self._work_buffers["tile_ranges"], "g_OutputGrad": output_grad, "g_TrainingForwardState": self._work_buffers["training_forward_state"], "g_TrainingProcessedEnd": self._work_buffers["training_processed_end"], **self._grad_vars(), **self._prepass_uniforms(self._scene_count), **self._raster_uniforms(background), **self._camera_uniforms(camera)}, "Rasterize Backward", 27)

    def _execute_prepass(self, scene: GaussianScene, camera: Camera, sync_counts: bool = False) -> tuple[int, int]:
        self._reset_prepass_counters()
        enc = self.device.create_command_encoder()
        with debug_region(enc, "Renderer Prepass", 19):
            self._project_and_bin(enc, scene, camera)
            self._compose_scanline_key_values_indirect(enc, self._compute_scanline_dispatch_args(enc))
            self._enqueue_counter_readback(enc)
            self._clear_tile_ranges(enc)
            args_buffer = self._sorter.sort_key_values_from_count_buffer(encoder=enc, keys_buffer=self._work_buffers["keys"], values_buffer=self._work_buffers["values"], count_buffer=self._work_buffers["counter"], count_offset=0, max_count=self._max_list_entries, max_bits=self.sort_bits)
            self._build_tile_ranges_indirect(enc, args_buffer)
        self.device.submit_command_buffer(enc.finish())
        if sync_counts:
            self.device.wait()
            generated = int(self._read_array(self._work_buffers["counter"], np.uint32, 1)[0])
            sorted_count = min(generated, self._max_list_entries)
        else:
            generated = self._delayed_generated_entries if self._delayed_stats_valid else 0
            sorted_count = self._delayed_written_entries if self._delayed_stats_valid else 0
        self._counter_readback_frame_id += 1
        return generated, sorted_count

    def _require_texture(self, attr: str, label: str) -> spy.Texture:
        texture = getattr(self, attr)
        if texture is None:
            raise RuntimeError(f"{label} texture is not initialized.")
        return texture

    def _require_buffer(self, attr: str, label: str) -> spy.Buffer:
        buffer = getattr(self, attr)
        if buffer is None:
            raise RuntimeError(f"{label} buffer is not initialized.")
        return buffer

    def _copy_output_texture_to_grad_buffer(self) -> None:
        self.output_grad_buffer.copy_from_numpy(np.ascontiguousarray(self._read_image().reshape(-1, 4), dtype=np.float32))

    def copy_scene_state_to(self, encoder: spy.CommandEncoder, dst: "GaussianRenderer") -> None:
        if self._current_scene is None:
            raise RuntimeError("Source scene is not set.")
        if self._scene_count <= 0:
            raise RuntimeError("Source scene is empty.")
        dst._ensure_scene_buffers(self._scene_count)
        dst._ensure_work_buffers(self._scene_count)
        copy_bytes = self._scene_count * self.TRAINABLE_PARAM_COUNT * self._U32_BYTES
        for name in self._SCENE_SHADER_VARS:
            encoder.copy_buffer(dst._scene_buffers[name], 0, self._scene_buffers[name], 0, copy_bytes)
        dst._scene_count, dst._current_scene = self._scene_count, self._current_scene

    def read_scene_groups(self, splat_count: int | None = None) -> dict[str, np.ndarray]:
        count = self._scene_count if splat_count is None else int(splat_count)
        flat = self._read_array(self._scene_buffers["splat_params"], np.float32, max(count, 1) * self.TRAINABLE_PARAM_COUNT)
        return self._unpack_param_groups(flat, count)

    def write_scene_groups(
        self,
        splat_count: int,
        *,
        positions: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray,
        color_alpha: np.ndarray,
    ) -> None:
        packed = self._pack_param_groups(
            splat_count,
            positions=positions,
            scales=scales,
            rotations=rotations,
            color_alpha=color_alpha,
        )
        self._scene_buffers["splat_params"].copy_from_numpy(packed)

    def read_grad_groups(self, splat_count: int | None = None) -> dict[str, np.ndarray]:
        count = self._scene_count if splat_count is None else int(splat_count)
        flat = self._read_array(self._work_buffers["param_grads"], np.float32, max(count, 1) * self.TRAINABLE_PARAM_COUNT)
        groups = self._unpack_param_groups(flat, count)
        return {
            "grad_positions": groups["positions"],
            "grad_scales": groups["scales"],
            "grad_rotations": groups["rotations"],
            "grad_color_alpha": groups["color_alpha"],
        }

    def write_grad_groups(
        self,
        splat_count: int,
        *,
        grad_positions: np.ndarray | None = None,
        grad_scales: np.ndarray | None = None,
        grad_rotations: np.ndarray | None = None,
        grad_color_alpha: np.ndarray | None = None,
    ) -> None:
        packed = self._pack_param_groups(
            splat_count,
            positions=grad_positions,
            scales=grad_scales,
            rotations=grad_rotations,
            color_alpha=grad_color_alpha,
        )
        self._work_buffers["param_grads"].copy_from_numpy(packed)

    def _require_scene(self) -> GaussianScene | SceneBinding:
        if self._current_scene is None:
            raise RuntimeError("Scene is not set.")
        return self._current_scene
    def set_debug_grad_norm_buffer(self, buffer: spy.Buffer | None) -> None:
        self._debug_grad_norm_buffer = buffer
    def upload_debug_grad_norm(self, values: np.ndarray) -> None:
        grad = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
        self._ensure_work_buffers(max(int(grad.shape[0]), self._scene_count, 1))
        self._work_buffers["debug_grad_norm"].copy_from_numpy(np.pad(grad, (0, max(self._work_splat_capacity - grad.shape[0], 0))))
        self._debug_grad_norm_buffer = None
    @property
    def scene_buffers(self) -> dict[str, spy.Buffer]:
        return self._scene_buffers

    @property
    def work_buffers(self) -> dict[str, spy.Buffer]:
        return self._work_buffers

    @property
    def output_texture(self) -> spy.Texture:
        return self._require_texture("_output_texture", "Output")

    @property
    def output_grad_buffer(self) -> spy.Buffer:
        return self._require_buffer("_output_grad_buffer", "Output grad")

    def execute_prepass_for_current_scene(self, camera: Camera, sync_counts: bool = False) -> tuple[int, int]:
        return self._execute_prepass(self._require_scene(), camera, sync_counts=sync_counts)

    def rasterize_current_scene(self, encoder: spy.CommandEncoder, camera: Camera, background: np.ndarray) -> None:
        self._require_scene()
        self._rasterize(encoder, camera, background)

    def clear_raster_grads_current_scene(self, encoder: spy.CommandEncoder) -> None:
        self._clear_raster_grads(encoder, self._require_scene().count)

    def rasterize_training_forward_current_scene(self, encoder: spy.CommandEncoder, camera: Camera, background: np.ndarray, output: spy.Texture | None = None) -> None:
        self._require_scene()
        self._rasterize_training_forward(encoder, camera, background, output)

    def rasterize_backward_current_scene(self, encoder: spy.CommandEncoder, camera: Camera, background: np.ndarray, output_grad: spy.Buffer) -> None:
        self._require_scene()
        self._rasterize_backward(encoder, camera, background, output_grad)

    def rasterize_forward_backward_current_scene(self, encoder: spy.CommandEncoder, camera: Camera, background: np.ndarray, output_grad: spy.Buffer) -> None:
        self._require_scene()
        self._rasterize_training_forward(encoder, camera, background)
        self._rasterize_backward(encoder, camera, background, output_grad)

    def render_to_texture(
        self,
        camera: Camera,
        background: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
        read_stats: bool = True,
        command_encoder: spy.CommandEncoder | None = None,
    ) -> tuple[spy.Texture, dict[str, int | bool | float]]:
        scene = self._require_scene()
        if scene.count <= 0:
            raise RuntimeError("Cannot render empty scene.")
        self._ensure_work_buffers(scene.count, self._pending_min_list_entries)
        background_np = self._background_array(background)
        self._execute_prepass(scene, camera, sync_counts=False)
        enc = self.device.create_command_encoder() if command_encoder is None else command_encoder
        self._rasterize(enc, camera, background_np)
        if command_encoder is None:
            self.device.submit_command_buffer(enc.finish())
            self.device.wait()
        if read_stats:
            self._update_delayed_counter_stats()
        self._last_stats = self._stats_payload(scene.count, read_stats)
        return self.output_texture, self._last_stats

    @property
    def last_stats(self) -> dict[str, int | bool | float]:
        return self._last_stats.copy()

    def render(self, scene: GaussianScene, camera: Camera, background: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0)) -> RenderOutput:
        if scene.count <= 0:
            return RenderOutput(image=np.zeros((self.height, self.width, 4), dtype=np.float32), stats=self._stats_payload(0, True, stats_valid=True))
        self.set_scene(scene)
        _, stats = self.render_to_texture(camera, self._background_array(background))
        return RenderOutput(image=self._read_image(), stats=stats)

    def debug_raster_backward_grads(self, scene: GaussianScene, camera: Camera, background: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0)) -> dict[str, np.ndarray]:
        self.set_scene(scene)
        if self._debug_render_enabled():
            raise RuntimeError("Disable debug overlay rendering before requesting raster backward gradients.")
        background_np = self._background_array(background)
        self._execute_prepass(scene, camera, sync_counts=False)
        enc = self.device.create_command_encoder()
        with debug_region(enc, "Debug Raster Forward", 27):
            self._rasterize_training_forward(enc, camera, background_np)
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()
        self._copy_output_texture_to_grad_buffer()
        enc_bwd = self.device.create_command_encoder()
        with debug_region(enc_bwd, "Debug Raster Backward", 28):
            self._clear_raster_grads(enc_bwd, scene.count)
            self._rasterize_backward(enc_bwd, camera, background_np, self.output_grad_buffer)
        self.device.submit_command_buffer(enc_bwd.finish())
        self.device.wait()
        return self.read_grad_groups(scene.count)

    def debug_pipeline_data(self, scene: GaussianScene, camera: Camera) -> dict[str, np.ndarray | int]:
        self._ensure_scene_buffers(scene.count)
        self._ensure_work_buffers(scene.count)
        self._upload_scene(scene)
        generated_entries, sorted_count = self._execute_prepass(scene, camera, sync_counts=True)
        return {
            "generated_entries": generated_entries,
            "sorted_count": sorted_count,
            "keys": self._read_array(self._work_buffers["keys"], np.uint32, sorted_count),
            "values": self._read_array(self._work_buffers["values"], np.uint32, sorted_count),
            "tile_ranges": self._read_array(self._work_buffers["tile_ranges"], np.uint32, self.tile_count, 2),
            "screen_center_radius_depth": self._read_array(self._work_buffers["screen_center_radius_depth"], np.float32, scene.count, 4),
            "screen_color_alpha": self._read_array(self._work_buffers["screen_color_alpha"], np.float32, scene.count, 4),
            "screen_ellipse_conic": self._read_array(self._work_buffers["screen_ellipse_conic"], np.float32, scene.count, 4),
            "splat_visible": self._read_array(self._work_buffers["splat_visible"], np.uint32, scene.count),
        }
