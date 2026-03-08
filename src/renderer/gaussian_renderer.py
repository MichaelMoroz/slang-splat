from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
import operator
from pathlib import Path
import re

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
    _UNARY_OPS = {ast.UAdd: lambda value: value, ast.USub: operator.neg}

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
    _SCENE_SHADER_VARS = {"positions": "g_Positions", "scales": "g_Scales", "rotations": "g_Rotations", "color_alpha": "g_ColorAlpha"}
    _SCREEN_SHADER_VARS = {"screen_center_radius_depth": "g_ScreenCenterRadiusDepth", "screen_color_alpha": "g_ScreenColorAlpha", "screen_ellipse_conic": "g_ScreenEllipseConic"}
    _GRAD_SHADER_VARS = {"grad_positions": "g_GradPositions", "grad_scales": "g_GradScales", "grad_rotations": "g_GradRotations", "grad_color_alpha": "g_GradColorAlpha"}
    _PREPASS_CURSOR_FIELDS = ("splatCount", "tileSize", "tileWidth", "tileHeight", "tileCount", "depthBits", "sortedCountOffset", "maxListEntries", "maxScanlineEntries", "radiusScale", "sampled5MVEEIters", "sampled5SafetyScale", "sampled5RadiusPadPx", "sampled5Eps")
    _SHADERS = (
        ("_k_project", "kernel", "gaussian_project_stage.slang", "csProjectAndBin"),
        ("_p_compose_scanline", "pipeline", "gaussian_project_stage.slang", "csComposeScanlineKeyValues"),
        ("_k_clear_ranges", "kernel", "gaussian_project_stage.slang", "csClearTileRanges"),
        ("_p_build_ranges", "pipeline", "gaussian_project_stage.slang", "csBuildTileRanges"),
        ("_k_raster", "kernel", "gaussian_raster_stage.slang", "csRasterize"),
        ("_k_clear_raster_grads", "kernel", "gaussian_raster_stage.slang", "csClearRasterGrads"),
        ("_k_raster_forward_backward", "kernel", "gaussian_raster_stage.slang", "csRasterizeForwardBackward"),
    )
    _buffer_vars = staticmethod(lambda mapping, source: {shader_name: source[name] for name, shader_name in mapping.items()})
    _dispatch = lambda self, kernel, encoder, thread_count, vars: kernel.dispatch(thread_count=thread_count, vars=vars, command_encoder=encoder)
    _grow = lambda self, required, current: max(required, max(current, 1) + max(current, 1) // 2)
    _max_prepass_entries_by_budget = lambda self: max(self._max_prepass_memory_bytes // self._PREPASS_ENTRY_BYTES, 1)
    _camera_uniforms = lambda self, camera: {"g_Camera": {**camera.gpu_params(self.width, self.height), "projDistortionK1": float(self.proj_distortion_k1), "projDistortionK2": float(self.proj_distortion_k2)}}
    _prepass_uniforms = lambda self, splat_count, sorted_count_offset=0: {"g_Prepass": {"splatCount": int(splat_count), "tileSize": int(self.tile_size), "tileWidth": int(self.tile_width), "tileHeight": int(self.tile_height), "tileCount": int(self.tile_count), "depthBits": int(self.depth_bits), "sortedCountOffset": int(sorted_count_offset), "maxListEntries": int(self._max_list_entries), "maxScanlineEntries": int(self._max_scanline_entries), "radiusScale": float(self.radius_scale), "sampled5MVEEIters": int(self.sampled5_mvee_iters), "sampled5SafetyScale": float(self.sampled5_safety_scale), "sampled5RadiusPadPx": float(self.sampled5_radius_pad_px), "sampled5Eps": float(self.sampled5_eps)}}
    _raster_uniforms = lambda self, background: {"g_Raster": {"width": int(self.width), "height": int(self.height), "maxSplatSteps": int(self.max_splat_steps), "alphaCutoff": float(self.alpha_cutoff), "transmittanceThreshold": float(self.transmittance_threshold), "background": spy.float3(*background.tolist()), "debugShowEllipses": np.uint32(1 if self.debug_show_ellipses else 0), "debugShowProcessedCount": np.uint32(1 if self.debug_show_processed_count else 0), "debugShowGradNorm": np.uint32(1 if self.debug_show_grad_norm else 0), "debugEllipseThicknessPx": float(self.debug_ellipse_thickness_px), "debugEllipseColor": spy.float3(*self.debug_ellipse_color.tolist())}}
    _background_array = staticmethod(lambda background: np.asarray(background, dtype=np.float32).reshape(3))
    _scene_vars = lambda self: self._buffer_vars(self._SCENE_SHADER_VARS, self._scene_buffers)
    _screen_vars = lambda self: self._buffer_vars(self._SCREEN_SHADER_VARS, self._work_buffers)
    _grad_vars = lambda self: self._buffer_vars(self._GRAD_SHADER_VARS, self._work_buffers)
    _debug_grad_norm_var = lambda self: {"g_DebugGradNorm": self._debug_grad_norm_buffer if self._debug_grad_norm_buffer is not None else self._work_buffers["debug_grad_norm"]}
    _raster_thread_count = lambda self: spy.uint3((self.width + self._raster_config.microtile_dim - 1) // self._raster_config.microtile_dim, (self.height + self._raster_config.microtile_dim - 1) // self._raster_config.microtile_dim, 1)
    _read_image = lambda self: np.asarray(self.output_texture.to_numpy(), dtype=np.float32).copy()
    _create_shaders = lambda self: [setattr(self, attr, self.device.create_compute_kernel(program) if kind == "kernel" else self.device.create_compute_pipeline(program)) for attr, kind, shader_name, entry in self._SHADERS for program in [self.device.load_program(str(Path(SHADER_ROOT / "renderer" / shader_name)), [entry])]]
    _ensure_textures = lambda self: [setattr(self, attr, getattr(self, attr) or self.device.create_texture(format=spy.Format.rgba32_float, width=self.width, height=self.height, usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access)) for attr in ("_output_texture", "_output_grad_texture")]
    _upload_scene = lambda self, scene: [self._scene_buffers[name].copy_from_numpy(values) for name, values in self._pack_scene(scene).items()]
    _reset_prepass_counters = lambda self: [self._work_buffers[name].copy_from_numpy(np.array([0], dtype=np.uint32)) for name in ("counter", "scanline_counter")]
    set_scene = lambda self, scene: (self._ensure_scene_buffers(scene.count), self._ensure_work_buffers(scene.count), self._upload_scene(scene), setattr(self, "_current_scene", scene))
    bind_scene_count = lambda self, splat_count: (lambda count: (self._ensure_scene_buffers(count), self._ensure_work_buffers(count), setattr(self, "_current_scene", SceneBinding(count=count))))(max(int(splat_count), 0))
    _bind_prepass_cursor = lambda self, cursor, splat_count, sorted_count_offset=0: [setattr(cursor.g_Prepass, name, self._prepass_uniforms(splat_count, sorted_count_offset)["g_Prepass"][name]) for name in self._PREPASS_CURSOR_FIELDS]
    _enqueue_counter_readback = lambda self, encoder: (self._ensure_counter_readback_ring(), encoder.copy_buffer(self._counter_readback_ring[self._counter_readback_frame_id % self._COUNTER_READBACK_RING_SIZE], 0, self._work_buffers["counter"], 0, 4), self._counter_readback_capacity.__setitem__(self._counter_readback_frame_id % self._COUNTER_READBACK_RING_SIZE, int(self._max_list_entries)))

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
        debug_ellipse_thickness_px: float = 1.0,
        debug_ellipse_color: tuple[float, float, float] = (1.0, 0.15, 0.1),
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
        self.debug_ellipse_thickness_px = float(debug_ellipse_thickness_px)
        self.debug_ellipse_color = np.asarray(debug_ellipse_color, dtype=np.float32).reshape(3)
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
        self._output_grad_texture: spy.Texture | None = None
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
        self._scene_buffers = {name: self.device.create_buffer(size=max(self._scene_capacity, 1) * 16, usage=self._RW_BUFFER_USAGE) for name in self._SCENE_SHADER_VARS}

    def _ensure_work_buffers(self, splat_count: int, min_list_entries: int = 0) -> None:
        max_entries = self._max_prepass_entries_by_budget()
        required_splats = max(splat_count, 1)
        required_entries = min(max(splat_count * self.list_capacity_multiplier, min_list_entries, 1), max_entries)
        if self._work_buffers and required_splats <= self._work_splat_capacity and required_entries <= self._max_list_entries and required_entries <= self._max_scanline_entries and self._output_texture is not None and self._output_grad_texture is not None:
            return
        self._work_splat_capacity = self._grow(required_splats, self._work_splat_capacity)
        self._max_list_entries = min(self._grow(required_entries, self._max_list_entries), max_entries)
        self._max_scanline_entries = min(self._grow(required_entries, self._max_scanline_entries), max_entries)
        sized = {
            "screen_center_radius_depth": max(self._work_splat_capacity, 1) * 16,
            "screen_color_alpha": max(self._work_splat_capacity, 1) * 16,
            "screen_ellipse_conic": max(self._work_splat_capacity, 1) * 16,
            "debug_grad_norm": max(self._work_splat_capacity, 1) * self._U32_BYTES,
            "keys": self._max_list_entries * 4,
            "values": self._max_list_entries * 4,
            "counter": 4,
            "splat_list_bases": max(self._work_splat_capacity, 1) * self._U32_BYTES,
            "scanline_work_items": self._max_scanline_entries * self._SCANLINE_WORK_ITEM_UINTS * self._U32_BYTES,
            "scanline_counter": self._U32_BYTES,
            "tile_ranges": max(self.tile_count, 1) * 8,
            **{name: max(self._work_splat_capacity, 1) * 16 for name in self._GRAD_SHADER_VARS},
        }
        self._work_buffers = {name: self.device.create_buffer(size=size, usage=self._RW_BUFFER_USAGE) for name, size in sized.items()}
        self._work_buffers["debug_grad_norm"].copy_from_numpy(np.zeros((max(self._work_splat_capacity, 1),), dtype=np.float32))
        self._ensure_textures()
        if self._pending_min_list_entries > 0 and self._max_list_entries >= self._pending_min_list_entries:
            self._pending_min_list_entries = 0

    def _pack_scene(self, scene: GaussianScene) -> dict[str, np.ndarray]:
        packed = {name: np.zeros((scene.count, 4), dtype=np.float32) for name in self._SCENE_SHADER_VARS}
        packed["positions"][:, :3], packed["scales"][:, :3] = scene.positions, scene.scales
        packed["rotations"][:], packed["color_alpha"][:, :3], packed["color_alpha"][:, 3] = scene.rotations, scene.colors, self._raw_opacity_from_alpha(scene.opacities)
        return packed

    @classmethod
    def _raw_opacity_from_alpha(cls, opacity: np.ndarray) -> np.ndarray:
        alpha = np.clip(np.asarray(opacity, dtype=np.float32), cls._OPACITY_EPS, 1.0 - cls._OPACITY_EPS)
        return (np.log(alpha) - np.log1p(-alpha)).astype(np.float32, copy=False)

    def _read_array(self, buffer: spy.Buffer, dtype: np.dtype, count: int, width: int = 1) -> np.ndarray:
        values = np.frombuffer(buffer.to_numpy().tobytes(), dtype=dtype)
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
        self._dispatch(self._k_project, encoder, spy.uint3(scene.count, 1, 1), {**self._scene_vars(), **self._screen_vars(), "g_Keys": self._work_buffers["keys"], "g_Values": self._work_buffers["values"], "g_ListCounter": self._work_buffers["counter"], "g_SplatListBases": self._work_buffers["splat_list_bases"], "g_ScanlineWorkItems": self._work_buffers["scanline_work_items"], "g_ScanlineCounter": self._work_buffers["scanline_counter"], **self._prepass_uniforms(scene.count), **self._camera_uniforms(camera)})

    def _compute_scanline_dispatch_args(self, encoder: spy.CommandEncoder) -> spy.Buffer:
        args_buffer = self._sorter.ensure_indirect_args()
        self._sorter.compute_indirect_args_from_buffer_dispatch(encoder=encoder, count_buffer=self._work_buffers["scanline_counter"], count_offset=0, max_element_count=self._max_scanline_entries, args_buffer=args_buffer)
        return args_buffer

    def _compose_scanline_key_values_indirect(self, encoder: spy.CommandEncoder, args_buffer: spy.Buffer) -> None:
        with encoder.begin_compute_pass() as compute_pass:
            cursor = spy.ShaderCursor(compute_pass.bind_pipeline(self._p_compose_scanline))
            for name, buffer in {"g_ScanlineWorkItems": self._work_buffers["scanline_work_items"], "g_ScanlineCounter": self._work_buffers["scanline_counter"], "g_SplatListBases": self._work_buffers["splat_list_bases"], "g_Keys": self._work_buffers["keys"], "g_Values": self._work_buffers["values"]}.items():
                setattr(cursor, name, buffer)
            self._bind_prepass_cursor(cursor, self._scene_count)
            compute_pass.dispatch_compute_indirect(spy.BufferOffsetPair(args_buffer, GPURadixSort.BUILD_RANGE_ARGS_OFFSET * self._U32_BYTES))

    def _clear_tile_ranges(self, encoder: spy.CommandEncoder) -> None:
        self._dispatch(self._k_clear_ranges, encoder, spy.uint3(self.tile_count, 1, 1), {"g_TileRanges": self._work_buffers["tile_ranges"], **self._prepass_uniforms(self._scene_count)})

    def _build_tile_ranges_indirect(self, encoder: spy.CommandEncoder, args_buffer: spy.Buffer) -> None:
        with encoder.begin_compute_pass() as compute_pass:
            cursor = spy.ShaderCursor(compute_pass.bind_pipeline(self._p_build_ranges))
            cursor.g_SortedKeys, cursor.g_TileRanges, cursor.g_PrepassParams = self._work_buffers["keys"], self._work_buffers["tile_ranges"], args_buffer
            self._bind_prepass_cursor(cursor, self._scene_count, sorted_count_offset=18)
            compute_pass.dispatch_compute_indirect(spy.BufferOffsetPair(args_buffer, GPURadixSort.BUILD_RANGE_ARGS_OFFSET * self._U32_BYTES))

    def _rasterize(self, encoder: spy.CommandEncoder, camera: Camera, background: np.ndarray) -> None:
        self._dispatch(self._k_raster, encoder, self._raster_thread_count(), {**self._scene_vars(), **self._screen_vars(), **self._debug_grad_norm_var(), "g_SortedValues": self._work_buffers["values"], "g_TileRanges": self._work_buffers["tile_ranges"], "g_Output": self._output_texture, **self._prepass_uniforms(self._scene_count), **self._raster_uniforms(background), **self._camera_uniforms(camera)})

    def _clear_raster_grads(self, encoder: spy.CommandEncoder, splat_count: int) -> None:
        self._dispatch(self._k_clear_raster_grads, encoder, spy.uint3(max(int(splat_count) * 4, 1), 1, 1), {**self._grad_vars(), **self._prepass_uniforms(splat_count)})

    def _rasterize_forward_backward(self, encoder: spy.CommandEncoder, camera: Camera, background: np.ndarray, output_grad: spy.Texture) -> None:
        self._dispatch(self._k_raster_forward_backward, encoder, self._raster_thread_count(), {**self._scene_vars(), "g_ScreenColorAlpha": self._work_buffers["screen_color_alpha"], **self._debug_grad_norm_var(), "g_SortedValues": self._work_buffers["values"], "g_TileRanges": self._work_buffers["tile_ranges"], "g_OutputGrad": output_grad, **self._grad_vars(), **self._prepass_uniforms(self._scene_count), **self._raster_uniforms(background), **self._camera_uniforms(camera)})

    def _execute_prepass(self, scene: GaussianScene, camera: Camera, sync_counts: bool = False) -> tuple[int, int]:
        self._reset_prepass_counters()
        enc = self.device.create_command_encoder()
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

    def copy_scene_state_to(self, encoder: spy.CommandEncoder, dst: "GaussianRenderer") -> None:
        if self._current_scene is None:
            raise RuntimeError("Source scene is not set.")
        if self._scene_count <= 0:
            raise RuntimeError("Source scene is empty.")
        dst._ensure_scene_buffers(self._scene_count)
        dst._ensure_work_buffers(self._scene_count)
        copy_bytes = self._scene_count * 16
        for name in self._SCENE_SHADER_VARS:
            encoder.copy_buffer(dst._scene_buffers[name], 0, self._scene_buffers[name], 0, copy_bytes)
        dst._scene_count, dst._current_scene = self._scene_count, self._current_scene

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
    scene_buffers = property(lambda self: self._scene_buffers)
    work_buffers = property(lambda self: self._work_buffers)
    output_texture = property(lambda self: self._require_texture("_output_texture", "Output"))
    output_grad_texture = property(lambda self: self._require_texture("_output_grad_texture", "Output grad"))
    execute_prepass_for_current_scene = lambda self, camera, sync_counts=False: self._execute_prepass(self._require_scene(), camera, sync_counts=sync_counts)
    rasterize_current_scene = lambda self, encoder, camera, background: self._require_scene() and self._rasterize(encoder, camera, background)
    clear_raster_grads_current_scene = lambda self, encoder: self._clear_raster_grads(encoder, self._require_scene().count)
    rasterize_forward_backward_current_scene = lambda self, encoder, camera, background, output_grad: self._require_scene() and self._rasterize_forward_backward(encoder, camera, background, output_grad)

    def render_to_texture(self, camera: Camera, background: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0), read_stats: bool = True) -> tuple[spy.Texture, dict[str, int | bool | float]]:
        scene = self._require_scene()
        if scene.count <= 0:
            raise RuntimeError("Cannot render empty scene.")
        self._ensure_work_buffers(scene.count, self._pending_min_list_entries)
        background_np = self._background_array(background)
        self._execute_prepass(scene, camera, sync_counts=False)
        enc = self.device.create_command_encoder()
        self._rasterize(enc, camera, background_np)
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()
        if read_stats:
            self._update_delayed_counter_stats()
        self._last_stats = self._stats_payload(scene.count, read_stats)
        return self.output_texture, self._last_stats

    last_stats = property(lambda self: self._last_stats.copy())

    def render(self, scene: GaussianScene, camera: Camera, background: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0)) -> RenderOutput:
        if scene.count <= 0:
            return RenderOutput(image=np.zeros((self.height, self.width, 4), dtype=np.float32), stats=self._stats_payload(0, True, stats_valid=True))
        self.set_scene(scene)
        _, stats = self.render_to_texture(camera, self._background_array(background))
        return RenderOutput(image=self._read_image(), stats=stats)

    def debug_raster_backward_grads(self, scene: GaussianScene, camera: Camera, background: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0)) -> dict[str, np.ndarray]:
        self.set_scene(scene)
        background_np = self._background_array(background)
        self._execute_prepass(scene, camera, sync_counts=False)
        enc = self.device.create_command_encoder()
        self._rasterize(enc, camera, background_np)
        self._clear_raster_grads(enc, scene.count)
        self._rasterize_forward_backward(enc, camera, background_np, self.output_texture)
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()
        return {name: self._read_array(self._work_buffers[name], np.float32, scene.count, 4) for name in self._GRAD_SHADER_VARS}

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
        }
