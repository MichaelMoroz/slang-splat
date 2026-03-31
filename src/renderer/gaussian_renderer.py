from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
import operator
from pathlib import Path
import re

import numpy as np
import slangpy as spy

from ..common import SHADER_ROOT, buffer_to_numpy, debug_region, dispatch, dispatch_indirect, remap_named_buffers, thread_count_1d
from ..metrics import Metrics, ParamLog10Histograms, ParamTensorRanges
from ..scan.prefix_sum import GPUPrefixSum
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
    tile_size: int
    batch: int

@dataclass(frozen=True, slots=True)
class _RasterGradShaderSet:
    training_forward: spy.ComputeKernel
    clear: spy.ComputeKernel
    backward: spy.ComputeKernel
    backprop: spy.ComputeKernel


@dataclass(slots=True)
class _RendererResourceGroups:
    scene: dict[str, spy.Buffer]
    frame: dict[str, object]
    prepass: dict[str, spy.Buffer]
    raster: dict[str, spy.Buffer]
    grad: dict[str, spy.Buffer]
    debug: dict[str, spy.Buffer]

    def merged_work_buffers(self) -> dict[str, spy.Buffer]:
        return {
            **self.prepass,
            **self.raster,
            **self.grad,
            **self.debug,
        }


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
    DEBUG_MODE_NORMAL = "normal"
    DEBUG_MODE_PROCESSED_COUNT = "processed_count"
    DEBUG_MODE_CLONE_COUNT = "clone_count"
    DEBUG_MODE_DEPTH_MEAN = "depth_mean"
    DEBUG_MODE_DEPTH_STD = "depth_std"
    DEBUG_MODE_ELLIPSE_OUTLINES = "ellipse_outlines"
    DEBUG_MODE_SPLAT_DENSITY = "splat_density"
    DEBUG_MODE_SPLAT_SPATIAL_DENSITY = "splat_spatial_density"
    DEBUG_MODE_SPLAT_SCREEN_DENSITY = "splat_screen_density"
    DEBUG_MODE_GRAD_NORM = "grad_norm"
    DEBUG_MODES = (
        DEBUG_MODE_NORMAL,
        DEBUG_MODE_PROCESSED_COUNT,
        DEBUG_MODE_CLONE_COUNT,
        DEBUG_MODE_DEPTH_MEAN,
        DEBUG_MODE_DEPTH_STD,
        DEBUG_MODE_ELLIPSE_OUTLINES,
        DEBUG_MODE_SPLAT_SPATIAL_DENSITY,
        DEBUG_MODE_SPLAT_SCREEN_DENSITY,
        DEBUG_MODE_GRAD_NORM,
        DEBUG_MODE_SPLAT_DENSITY,
    )
    CACHED_RASTER_GRAD_ATOMIC_MODE_FLOAT = "float"
    CACHED_RASTER_GRAD_ATOMIC_MODE_FIXED = "fixed"
    CACHED_RASTER_GRAD_ATOMIC_MODES = (CACHED_RASTER_GRAD_ATOMIC_MODE_FLOAT, CACHED_RASTER_GRAD_ATOMIC_MODE_FIXED)
    _RASTER_GRAD_FIXED_INT_MAX = np.float32(2147483647.0)
    _DEFAULT_RASTER_GRAD_FIXED_RO_LOCAL_RANGE = np.float32(0.01)
    _DEFAULT_RASTER_GRAD_FIXED_SCALE_RANGE = np.float32(0.01)
    _DEFAULT_RASTER_GRAD_FIXED_QUAT_RANGE = np.float32(0.01)
    _DEFAULT_RASTER_GRAD_FIXED_COLOR_RANGE = np.float32(0.2)
    _DEFAULT_RASTER_GRAD_FIXED_OPACITY_RANGE = np.float32(0.2)
    _DEFAULT_DEBUG_CLONE_COUNT_RANGE = (0.0, 16.0)
    _DEFAULT_DEBUG_DENSITY_RANGE = (0.0, 20.0)
    _DEFAULT_DEBUG_DEPTH_MEAN_RANGE = (0.0, 10.0)
    _DEFAULT_DEBUG_DEPTH_STD_RANGE = (0.0, 0.5)
    _COUNTER_READBACK_RING_SIZE = 2
    _SCANLINE_WORK_ITEM_UINTS = 4
    _U32_BYTES = 4
    _F32X4_BYTES = 16
    _OPACITY_EPS = 1e-6
    _MEBIBYTE_BYTES = 1024 * 1024
    _PREPASS_ENTRY_BYTES = (_SCANLINE_WORK_ITEM_UINTS + 4) * _U32_BYTES
    _RASTER_CACHE_PARAM_COUNT = 14
    _RW_BUFFER_USAGE = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
    PARAM_POSITION_IDS = (0, 1, 2)
    PARAM_SCALE_IDS = (3, 4, 5)
    PARAM_ROTATION_IDS = (6, 7, 8, 9)
    PARAM_COLOR_IDS = (10, 11, 12)
    PARAM_RAW_OPACITY_ID = 13
    TRAINABLE_PARAM_COUNT = 14
    CACHED_RASTER_GRAD_COMPONENT_LABELS = (
        "roLocal.x", "roLocal.y", "roLocal.z",
        "scale.x", "scale.y", "scale.z",
        "quat.w", "quat.x", "quat.y", "quat.z",
        "color.r", "color.g", "color.b", "opacity",
    )
    CACHED_RASTER_GRAD_COMPONENT_SLICES = tuple((label, slice(index, index + 1)) for index, label in enumerate(CACHED_RASTER_GRAD_COMPONENT_LABELS))
    _SCENE_SHADER_VARS = {"splat_params": "g_SplatParams"}
    _SCREEN_SHADER_VARS = {"screen_center_radius_depth": "g_ScreenCenterRadiusDepth", "screen_color_alpha": "g_ScreenColorAlpha", "screen_ellipse_conic": "g_ScreenEllipseConic", "splat_visible": "g_SplatVisible"}
    _GRAD_SHADER_VARS = {"param_grads": "g_ParamGrads"}
    _RASTER_CACHE_SHADER_VARS = {"raster_cache": "g_RasterCache"}
    _RASTER_GRAD_SHADER_VARS = {
        "param_grads": "g_ParamGrads",
        "cached_raster_grads_fixed": "g_CachedRasterGradsFixed",
        "cached_raster_grads_float": "g_CachedRasterGradsFloat",
    }
    _PREPASS_CURSOR_FIELDS = ("splatCount", "tileSize", "tileWidth", "tileHeight", "tileCount", "depthBits", "sortedCountOffset", "maxListEntries", "maxScanlineEntries", "radiusScale")
    _SHADERS = (
        ("_k_project_visible", "kernel", "gaussian_project_stage.slang", "csProjectVisibleSplats"),
        ("_p_count_visible_scanlines", "pipeline", "gaussian_project_stage.slang", "csCountVisibleScanlines"),
        ("_p_emit_scanlines", "pipeline", "gaussian_project_stage.slang", "csEmitScanlines"),
        ("_p_count_scanline_tiles", "pipeline", "gaussian_project_stage.slang", "csCountScanlineTiles"),
        ("_p_emit_tile_entries", "pipeline", "gaussian_project_stage.slang", "csEmitTileEntries"),
        ("_k_clear_ranges", "kernel", "gaussian_project_stage.slang", "csClearTileRanges"),
        ("_p_build_ranges", "pipeline", "gaussian_project_stage.slang", "csBuildTileRanges"),
        ("_k_raster", "kernel", "gaussian_raster_stage.slang", "csRasterize"),
        ("_k_raster_debug", "kernel", "gaussian_raster_stage.slang", "csRasterizeDebug"),
    )
    _buffer_vars = staticmethod(remap_named_buffers)

    def _dispatch(self, kernel: spy.ComputeKernel | spy.ComputePipeline, encoder: spy.CommandEncoder, thread_count: spy.uint3, vars: dict[str, object], label: str, color_index: int) -> None:
        dispatch(
            kernel=kernel,
            thread_count=thread_count,
            vars=vars,
            command_encoder=encoder,
            debug_label=label,
            debug_color_index=color_index,
        )

    @staticmethod
    def _grow(required: int, current: int) -> int:
        return max(required, max(current, 1) + max(current, 1) // 2)

    def _max_prepass_entries_by_budget(self) -> int:
        return max(self._max_prepass_memory_bytes // self._PREPASS_ENTRY_BYTES, 1)

    @staticmethod
    def _background_array(background: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        return np.asarray(background, dtype=np.float32).reshape(3)

    def _camera_uniforms(self, camera: Camera) -> dict[str, object]:
        k1, k2 = camera.distortion_coeffs(self.proj_distortion_k1, self.proj_distortion_k2)
        return {
            "g_Camera": {
                **camera.gpu_params(self.width, self.height),
                "projDistortionK1": float(k1),
                "projDistortionK2": float(k2),
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
                "debugMode": np.uint32(self._debug_mode_u32(self.debug_mode)),
                "debugGradNormThreshold": float(max(self.debug_grad_norm_threshold, 0.0)),
                "debugEllipseThicknessPx": float(self.debug_ellipse_thickness_px),
                "debugCloneCountRange": spy.float2(*self.debug_clone_count_range),
                "debugDensityRange": spy.float2(*self.debug_density_range),
                "debugDepthMeanRange": spy.float2(*self.debug_depth_mean_range),
                "debugDepthStdRange": spy.float2(*self.debug_depth_std_range),
            }
        }

    def _anisotropy_uniforms(self) -> dict[str, object]:
        return {"g_MaxAnisotropy": float(max(self.max_anisotropy, 1.0))}

    def _scene_vars(self) -> dict[str, object]:
        return self._buffer_vars(self._SCENE_SHADER_VARS, self._scene_buffers)

    def _screen_vars(self) -> dict[str, object]:
        return self._buffer_vars(self._SCREEN_SHADER_VARS, self._work_buffers)

    def _grad_vars(self) -> dict[str, object]:
        return self._buffer_vars(self._GRAD_SHADER_VARS, self._work_buffers)

    def _raster_cache_vars(self) -> dict[str, object]:
        return self._buffer_vars(self._RASTER_CACHE_SHADER_VARS, self._work_buffers)

    def _raster_grad_vars(self) -> dict[str, object]:
        return self._buffer_vars(self._RASTER_GRAD_SHADER_VARS, self._work_buffers)

    @staticmethod
    def _raster_grad_decode_scale_var(grad_scale: float) -> dict[str, object]:
        return {"g_RasterGradDecodeScale": float(grad_scale)}

    def _raster_grad_fixed_range_vars(self) -> dict[str, object]:
        return {
            "g_CachedRasterGradFixedROLocalRange": float(self.cached_raster_grad_fixed_ro_local_range),
            "g_CachedRasterGradFixedScaleRange": float(self.cached_raster_grad_fixed_scale_range),
            "g_CachedRasterGradFixedQuatRange": float(self.cached_raster_grad_fixed_quat_range),
            "g_CachedRasterGradFixedColorRange": float(self.cached_raster_grad_fixed_color_range),
            "g_CachedRasterGradFixedOpacityRange": float(self.cached_raster_grad_fixed_opacity_range),
        }

    def _debug_grad_norm_var(self) -> dict[str, object]:
        return {"g_DebugGradNorm": self._debug_grad_norm_buffer if self._debug_grad_norm_buffer is not None else self._work_buffers["debug_grad_norm"]}

    def _debug_clone_count_var(self) -> dict[str, object]:
        return {"g_CloneCounts": self._debug_clone_count_buffer if self._debug_clone_count_buffer is not None else self._work_buffers["debug_clone_count"]}

    @classmethod
    def _validate_debug_mode(cls, mode: str) -> str:
        resolved = str(mode).strip().lower()
        if resolved not in cls.DEBUG_MODES:
            supported = ", ".join(cls.DEBUG_MODES)
            raise ValueError(f"Unsupported debug mode '{mode}'. Expected one of: {supported}.")
        return resolved

    @classmethod
    def _debug_mode_u32(cls, mode: str) -> int:
        return cls.DEBUG_MODES.index(cls._validate_debug_mode(mode))

    @classmethod
    def _resolve_debug_mode(cls, debug_mode: str | None, debug_show_ellipses: bool, debug_show_processed_count: bool, debug_show_grad_norm: bool) -> str:
        if debug_mode is not None:
            return cls._validate_debug_mode(debug_mode)
        if debug_show_grad_norm:
            return cls.DEBUG_MODE_GRAD_NORM
        if debug_show_processed_count:
            return cls.DEBUG_MODE_PROCESSED_COUNT
        if debug_show_ellipses:
            return cls.DEBUG_MODE_ELLIPSE_OUTLINES
        return cls.DEBUG_MODE_NORMAL

    def _raster_thread_count(self) -> spy.uint3:
        return thread_count_1d(self.tile_count * self._raster_config.thread_tile_dim * self._raster_config.thread_tile_dim)

    def _read_image(self) -> np.ndarray:
        return np.asarray(self.output_texture.to_numpy(), dtype=np.float32).copy()

    def _create_shaders(self) -> None:
        for attr, kind, shader_name, entry in self._SHADERS:
            program = self.device.load_program(str(Path(SHADER_ROOT / "renderer" / shader_name)), [entry])
            shader = self.device.create_compute_kernel(program) if kind == "kernel" else self.device.create_compute_pipeline(program)
            setattr(self, attr, shader)
        self._fixed_raster_grad_shaders = self._load_raster_grad_shaders("Fixed")
        metrics_shader = str(Path(SHADER_ROOT / "utility" / "metrics" / "metrics.slang"))
        self._k_clear_float_buffer = self.device.create_compute_kernel(self.device.load_program(metrics_shader, ["csClearFloatBuffer"]))
        self._k_decode_fixed_buffer_to_float = self.device.create_compute_kernel(self.device.load_program(metrics_shader, ["csDecodeFixedIntBufferToFloat"]))

    def _load_raster_grad_shaders(self, entry_suffix: str) -> _RasterGradShaderSet:
        shader_path = str(Path(SHADER_ROOT / "renderer" / "gaussian_raster_stage.slang"))
        kernels = {
            "training_forward": self.device.create_compute_kernel(self.device.load_program(shader_path, [f"csRasterizeTrainingForward{entry_suffix}"])),
            "clear": self.device.create_compute_kernel(self.device.load_program(shader_path, [f"csClearRasterGrads{entry_suffix}"])),
            "backward": self.device.create_compute_kernel(self.device.load_program(shader_path, [f"csRasterizeBackward{entry_suffix}"])),
            "backprop": self.device.create_compute_kernel(self.device.load_program(shader_path, [f"csBackpropCachedRasterGrads{entry_suffix}"])),
        }
        return _RasterGradShaderSet(
            training_forward=kernels["training_forward"],
            clear=kernels["clear"],
            backward=kernels["backward"],
            backprop=kernels["backprop"],
        )

    @classmethod
    def _validate_cached_raster_grad_atomic_mode(cls, mode: str) -> str:
        resolved = str(mode).strip().lower()
        if resolved not in cls.CACHED_RASTER_GRAD_ATOMIC_MODES:
            supported = ", ".join(cls.CACHED_RASTER_GRAD_ATOMIC_MODES)
            raise ValueError(f"Unsupported cached raster grad atomic mode '{mode}'. Expected one of: {supported}.")
        return resolved

    def _ensure_float_raster_grad_shaders(self) -> _RasterGradShaderSet:
        if self._float_raster_grad_shaders is not None:
            return self._float_raster_grad_shaders
        try:
            self._float_raster_grad_shaders = self._load_raster_grad_shaders("Float")
        except Exception as exc:
            raise RuntimeError(
                "Cached raster grad atomic mode 'float' requires shader support for float atomics on the active device/backend."
            ) from exc
        return self._float_raster_grad_shaders

    def _raster_grad_shader_set(self) -> _RasterGradShaderSet:
        return self._fixed_raster_grad_shaders if self._cached_raster_grad_atomic_mode == self.CACHED_RASTER_GRAD_ATOMIC_MODE_FIXED else self._ensure_float_raster_grad_shaders()

    def _upload_scene(self, scene: GaussianScene) -> None:
        self._scene_buffers["splat_params"].copy_from_numpy(self._pack_scene(scene))

    def _reset_prepass_counters(self, encoder: spy.CommandEncoder | None = None) -> None:
        if encoder is None:
            zero = np.array([0], dtype=np.uint32)
            for name in ("visible_counter", "counter", "scanline_counter"):
                self._work_buffers[name].copy_from_numpy(zero)
            return
        for name in ("visible_counter", "counter", "scanline_counter"):
            encoder.copy_buffer(self._work_buffers[name], 0, self._zero_u32_buffer, 0, self._U32_BYTES)

    def _clear_float_buffer(self, encoder: spy.CommandEncoder, buffer: spy.Buffer, count: int) -> None:
        self._k_clear_float_buffer.dispatch(thread_count=thread_count_1d(count), vars={"g_ClearFloatBuffer": buffer, "g_ClearCount": int(count)}, command_encoder=encoder)

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
        radius_scale: float = 1.0,
        alpha_cutoff: float = 1.0 / 255.0,
        max_splat_steps: int = 32768,
        max_anisotropy: float = 32.0,
        transmittance_threshold: float = 0.005,
        list_capacity_multiplier: int = 64,
        max_prepass_memory_mb: int = 4096,
        proj_distortion_k1: float = 0.0,
        proj_distortion_k2: float = 0.0,
        debug_mode: str | None = None,
        debug_show_ellipses: bool = False,
        debug_show_processed_count: bool = False,
        debug_show_grad_norm: bool = False,
        debug_grad_norm_threshold: float = 2e-4,
        debug_ellipse_thickness_px: float = 2.0,
        debug_clone_count_range: tuple[float, float] = _DEFAULT_DEBUG_CLONE_COUNT_RANGE,
        debug_density_range: tuple[float, float] = _DEFAULT_DEBUG_DENSITY_RANGE,
        debug_depth_mean_range: tuple[float, float] = _DEFAULT_DEBUG_DEPTH_MEAN_RANGE,
        debug_depth_std_range: tuple[float, float] = _DEFAULT_DEBUG_DEPTH_STD_RANGE,
        cached_raster_grad_atomic_mode: str = CACHED_RASTER_GRAD_ATOMIC_MODE_FIXED,
        cached_raster_grad_fixed_ro_local_range: float = 0.01,
        cached_raster_grad_fixed_scale_range: float = 0.01,
        cached_raster_grad_fixed_quat_range: float = 0.01,
        cached_raster_grad_fixed_color_range: float = 0.2,
        cached_raster_grad_fixed_opacity_range: float = 0.2,
    ) -> None:
        self.device, self.width, self.height = device, int(width), int(height)
        self._types_shader_path = Path(SHADER_ROOT / "renderer" / "gaussian_types.slang")
        self._project_shader_path = Path(SHADER_ROOT / "renderer" / "gaussian_project_stage.slang")
        self._raster_shader_path = Path(SHADER_ROOT / "renderer" / "gaussian_raster_stage.slang")
        self._raster_config = self._load_raster_config(self._types_shader_path)
        self._project_group_size = self._load_uint_shader_constant(self._types_shader_path, "PROJECT_GROUP_SIZE")
        self.tile_size = self._raster_config.tile_size
        self.radius_scale, self.alpha_cutoff = float(radius_scale), float(alpha_cutoff)
        self.max_splat_steps, self.transmittance_threshold = int(max_splat_steps), float(transmittance_threshold)
        self.max_anisotropy = float(max(max_anisotropy, 1.0))
        self.list_capacity_multiplier = int(list_capacity_multiplier)
        self.max_prepass_memory_mb = max(int(max_prepass_memory_mb), 1)
        self._max_prepass_memory_bytes = self.max_prepass_memory_mb * self._MEBIBYTE_BYTES
        self.proj_distortion_k1, self.proj_distortion_k2 = float(proj_distortion_k1), float(proj_distortion_k2)
        self.debug_mode = self._resolve_debug_mode(debug_mode, debug_show_ellipses, debug_show_processed_count, debug_show_grad_norm)
        self.debug_show_ellipses = self.debug_mode == self.DEBUG_MODE_ELLIPSE_OUTLINES
        self.debug_show_processed_count = self.debug_mode == self.DEBUG_MODE_PROCESSED_COUNT
        self.debug_show_grad_norm = self.debug_mode == self.DEBUG_MODE_GRAD_NORM
        self.debug_grad_norm_threshold = float(debug_grad_norm_threshold)
        self.debug_ellipse_thickness_px = float(debug_ellipse_thickness_px)
        self.debug_clone_count_range = tuple(float(x) for x in debug_clone_count_range)
        self.debug_density_range = tuple(float(x) for x in debug_density_range)
        self.debug_depth_mean_range = tuple(float(x) for x in debug_depth_mean_range)
        self.debug_depth_std_range = tuple(float(x) for x in debug_depth_std_range)
        self.tile_width, self.tile_height = (self.width + self.tile_size - 1) // self.tile_size, (self.height + self.tile_size - 1) // self.tile_size
        self.tile_count = self.tile_width * self.tile_height
        self.tile_bits = int(np.ceil(np.log2(max(self.tile_count, 2))))
        self.depth_bits, self.sort_bits = 32 - self.tile_bits, 32
        self._create_shaders()
        self._sorter = GPURadixSort(self.device)
        self._prefix_sum = GPUPrefixSum(self.device)
        self._scene_count = self._scene_capacity = self._max_list_entries = self._work_splat_capacity = self._max_scanline_entries = 0
        self._current_scene: GaussianScene | None = None
        self._scene_buffers: dict[str, spy.Buffer] = {}
        self._work_buffers: dict[str, spy.Buffer] = {}
        self._resource_groups = _RendererResourceGroups(scene={}, frame={}, prepass={}, raster={}, grad={}, debug={})
        self._debug_grad_norm_buffer: spy.Buffer | None = None
        self._debug_clone_count_buffer: spy.Buffer | None = None
        self._output_texture: spy.Texture | None = None
        self._output_grad_buffer: spy.Buffer | None = None
        self._last_stats: dict[str, int | bool | float] = {}
        self._counter_readback_ring: list[spy.Buffer] = []
        self._counter_readback_capacity: list[int] = []
        self._counter_readback_frame_id = 0
        self._pending_min_list_entries = self._delayed_generated_entries = self._delayed_written_entries = 0
        self._delayed_overflow = self._delayed_stats_valid = False
        self._fixed_raster_grad_shaders: _RasterGradShaderSet
        self._float_raster_grad_shaders: _RasterGradShaderSet | None = None
        self._sorted_keys_buffer: spy.Buffer | None = None
        self._sorted_values_buffer: spy.Buffer | None = None
        self._cached_raster_grad_atomic_mode = self.CACHED_RASTER_GRAD_ATOMIC_MODE_FIXED
        self._cached_raster_grad_fixed_ro_local_range = self._DEFAULT_RASTER_GRAD_FIXED_RO_LOCAL_RANGE
        self._cached_raster_grad_fixed_scale_range = self._DEFAULT_RASTER_GRAD_FIXED_SCALE_RANGE
        self._cached_raster_grad_fixed_quat_range = self._DEFAULT_RASTER_GRAD_FIXED_QUAT_RANGE
        self._cached_raster_grad_fixed_color_range = self._DEFAULT_RASTER_GRAD_FIXED_COLOR_RANGE
        self._cached_raster_grad_fixed_opacity_range = self._DEFAULT_RASTER_GRAD_FIXED_OPACITY_RANGE
        self._zero_u32_buffer = self.device.create_buffer(size=self._U32_BYTES, usage=spy.BufferUsage.shader_resource | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination)
        self._zero_u32_buffer.copy_from_numpy(np.array([0], dtype=np.uint32))
        self.cached_raster_grad_atomic_mode = cached_raster_grad_atomic_mode
        self.cached_raster_grad_fixed_ro_local_range = cached_raster_grad_fixed_ro_local_range
        self.cached_raster_grad_fixed_scale_range = cached_raster_grad_fixed_scale_range
        self.cached_raster_grad_fixed_quat_range = cached_raster_grad_fixed_quat_range
        self.cached_raster_grad_fixed_color_range = cached_raster_grad_fixed_color_range
        self.cached_raster_grad_fixed_opacity_range = cached_raster_grad_fixed_opacity_range

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
        required = ("RASTER_THREAD_TILE_DIM", "RASTER_TILE_SIZE", "RASTER_BATCH")
        missing = [name for name in required if name not in constants]
        if missing:
            raise RuntimeError(f"Missing raster constants in {shader_path}: {', '.join(missing)}")
        return RasterConfig(constants["RASTER_THREAD_TILE_DIM"], constants["RASTER_TILE_SIZE"], constants["RASTER_BATCH"])

    @classmethod
    @lru_cache(maxsize=1)
    def _load_uint_shader_constants(cls, shader_path: Path) -> dict[str, int]:
        source = shader_path.read_text(encoding="utf-8")
        uint_constants: dict[str, int] = {}
        for name, expr in re.compile(r"static\s+const\s+uint\s+(\w+)\s*=\s*([^;]+);").findall(source):
            uint_constants[name] = cls._eval_uint_constant_expr(re.sub(r"(?<=[0-9A-Fa-f])[uU]\b", "", expr).strip(), uint_constants)
        return uint_constants

    @classmethod
    def _load_uint_shader_constant(cls, shader_path: Path, name: str) -> int:
        constants = cls._load_uint_shader_constants(shader_path)
        if name not in constants:
            raise RuntimeError(f"Missing shader uint constant '{name}' in {shader_path}.")
        return constants[name]

    @staticmethod
    @lru_cache(maxsize=None)
    def _load_float_shader_constant(shader_path: Path, name: str) -> float:
        source = shader_path.read_text(encoding="utf-8")
        match = re.search(rf"static\s+const\s+float\s+{name}\s*=\s*([^;]+);", source)
        if match is None:
            raise RuntimeError(f"Missing shader float constant '{name}' in {shader_path}.")
        return float(match.group(1).strip().removesuffix("f").removesuffix("F"))

    def _ensure_scene_buffers(self, splat_count: int) -> None:
        if self._scene_buffers and splat_count <= self._scene_capacity:
            self._scene_count = splat_count
            return
        self._scene_capacity, self._scene_count = self._grow(splat_count, self._scene_capacity), splat_count
        param_bytes = max(self._scene_capacity, 1) * self.TRAINABLE_PARAM_COUNT * self._U32_BYTES
        self._resource_groups.scene = {
            name: self.device.create_buffer(size=param_bytes, usage=self._RW_BUFFER_USAGE)
            for name in self._SCENE_SHADER_VARS
        }
        self._scene_buffers = self._resource_groups.scene

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
            "screen_center_radius_depth": max(self._work_splat_capacity, 1) * self._F32X4_BYTES,
            "screen_color_alpha": max(self._work_splat_capacity, 1) * self._F32X4_BYTES,
            "screen_ellipse_conic": max(self._work_splat_capacity, 1) * self._F32X4_BYTES,
            "splat_visible": max(self._work_splat_capacity, 1) * self._U32_BYTES,
            "debug_clone_count": max(self._work_splat_capacity, 1) * self._U32_BYTES,
            "debug_grad_norm": max(self._work_splat_capacity, 1) * self._U32_BYTES,
            "visible_keys": max(self._work_splat_capacity, 1) * self._U32_BYTES,
            "visible_values": max(self._work_splat_capacity, 1) * self._U32_BYTES,
            "visible_counter": self._U32_BYTES,
            "scanline_counts": max(self._work_splat_capacity, 1) * self._U32_BYTES,
            "scanline_offsets": max(self._work_splat_capacity, 1) * self._U32_BYTES,
            "keys": self._max_list_entries * 4,
            "values": self._max_list_entries * 4,
            "counter": 4,
            "scanline_work_items": self._max_scanline_entries * self._SCANLINE_WORK_ITEM_UINTS * self._U32_BYTES,
            "scanline_counter": self._U32_BYTES,
            "scanline_tile_counts": self._max_scanline_entries * self._U32_BYTES,
            "scanline_tile_offsets": self._max_scanline_entries * self._U32_BYTES,
            "tile_ranges": max(self.tile_count, 1) * 8,
            "training_forward_state": max(self.width * self.height, 1) * self._F32X4_BYTES,
            "training_depth_state": max(self.width * self.height, 1) * self._F32X4_BYTES,
            "training_depth_ratio": max(self.width * self.height, 1) * self._U32_BYTES,
            "training_depth_ratio_grad": max(self.width * self.height, 1) * self._U32_BYTES,
            "training_processed_end": max(self.width * self.height, 1) * self._U32_BYTES,
            "training_batch_end": max(self.tile_count, 1) * self._U32_BYTES,
            "raster_cache": max(self._work_splat_capacity, 1) * self._RASTER_CACHE_PARAM_COUNT * self._U32_BYTES,
            **{name: max(self._work_splat_capacity, 1) * self.TRAINABLE_PARAM_COUNT * self._U32_BYTES for name in self._GRAD_SHADER_VARS},
            "cached_raster_grads_fixed": max(self._work_splat_capacity, 1) * self._RASTER_CACHE_PARAM_COUNT * self._U32_BYTES,
            "cached_raster_grads_float": max(self._work_splat_capacity, 1) * self._RASTER_CACHE_PARAM_COUNT * self._U32_BYTES,
            "cached_raster_grads_metrics_float": max(self._work_splat_capacity, 1) * self._RASTER_CACHE_PARAM_COUNT * self._U32_BYTES,
        }
        allocated = {name: self.device.create_buffer(size=size, usage=self._RW_BUFFER_USAGE) for name, size in sized.items()}
        self._resource_groups.prepass = {
            name: allocated[name]
            for name in (
                "visible_keys",
                "visible_values",
                "visible_counter",
                "scanline_counts",
                "scanline_offsets",
                "keys",
                "values",
                "counter",
                "scanline_work_items",
                "scanline_counter",
                "scanline_tile_counts",
                "scanline_tile_offsets",
                "tile_ranges",
            )
        }
        self._resource_groups.raster = {
            name: allocated[name]
            for name in (
                "screen_center_radius_depth",
                "screen_color_alpha",
                "screen_ellipse_conic",
                "splat_visible",
                "training_forward_state",
                "training_depth_state",
                "training_depth_ratio",
                "training_depth_ratio_grad",
                "training_processed_end",
                "training_batch_end",
                "raster_cache",
            )
        }
        self._resource_groups.grad = {
            name: allocated[name]
            for name in ("param_grads", "cached_raster_grads_fixed", "cached_raster_grads_float", "cached_raster_grads_metrics_float")
        }
        self._resource_groups.debug = {
            "debug_clone_count": allocated["debug_clone_count"],
            "debug_grad_norm": allocated["debug_grad_norm"],
        }
        self._work_buffers = self._resource_groups.merged_work_buffers()
        self._sorted_keys_buffer = self._work_buffers["keys"]
        self._sorted_values_buffer = self._work_buffers["values"]
        self._work_buffers["debug_clone_count"].copy_from_numpy(np.zeros((max(self._work_splat_capacity, 1),), dtype=np.uint32))
        self._work_buffers["debug_grad_norm"].copy_from_numpy(np.zeros((max(self._work_splat_capacity, 1),), dtype=np.float32))
        self._ensure_output_texture()
        self._ensure_output_grad_buffer()
        self._resource_groups.frame = {
            "output_texture": self._output_texture,
            "output_grad_buffer": self._output_grad_buffer,
        }
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

    def _project_visible_splats(self, encoder: spy.CommandEncoder, scene: GaussianScene, camera: Camera) -> None:
        self._dispatch(
            self._k_project_visible,
            encoder,
            spy.uint3(scene.count, 1, 1),
            {
                **self._scene_vars(),
                **self._screen_vars(),
                **self._raster_cache_vars(),
                "g_VisibleKeys": self._work_buffers["visible_keys"],
                "g_VisibleValues": self._work_buffers["visible_values"],
                "g_VisibleCounter": self._work_buffers["visible_counter"],
                **self._prepass_uniforms(scene.count),
                **self._raster_uniforms(np.zeros((3,), dtype=np.float32)),
                **self._anisotropy_uniforms(),
                **self._camera_uniforms(camera),
            },
            "Project Visible Splats",
            20,
        )

    def _sort_visible_splats(self, encoder: spy.CommandEncoder) -> None:
        self._sorter.sort_key_values_from_count_buffer(
            encoder=encoder,
            keys_buffer=self._work_buffers["visible_keys"],
            values_buffer=self._work_buffers["visible_values"],
            count_buffer=self._work_buffers["visible_counter"],
            count_offset=0,
            max_count=self._scene_count,
            max_bits=32,
        )

    def _sorted_keys(self) -> spy.Buffer:
        return self._work_buffers["keys"] if self._sorted_keys_buffer is None else self._sorted_keys_buffer

    def _sorted_values(self) -> spy.Buffer:
        return self._work_buffers["values"] if self._sorted_values_buffer is None else self._sorted_values_buffer

    def _visible_dispatch_args(self, encoder: spy.CommandEncoder) -> spy.Buffer:
        return self._prefix_sum.dispatch_args_from_count_buffer(
            encoder,
            self._work_buffers["visible_counter"],
            0,
            self._scene_count,
            self._project_group_size,
        )

    def _scanline_dispatch_args(self, encoder: spy.CommandEncoder) -> spy.Buffer:
        return self._prefix_sum.dispatch_args_from_count_buffer(
            encoder,
            self._work_buffers["scanline_counter"],
            0,
            self._max_scanline_entries,
            self._project_group_size,
        )

    def _count_visible_scanlines(self, encoder: spy.CommandEncoder, args_buffer: spy.Buffer) -> None:
        dispatch_indirect(
            pipeline=self._p_count_visible_scanlines,
            args_buffer=args_buffer,
            vars={
                **self._screen_vars(),
                "g_VisibleValues": self._work_buffers["visible_values"],
                "g_VisibleCounter": self._work_buffers["visible_counter"],
                "g_ScanlineCounts": self._work_buffers["scanline_counts"],
                **self._prepass_uniforms(self._scene_count),
            },
            command_encoder=encoder,
            debug_label="Count Visible Scanlines",
            debug_color_index=21,
        )

    def _prefix_scanline_counts(self, encoder: spy.CommandEncoder) -> None:
        self._prefix_sum.scan_uint_from_count_buffer(
            encoder,
            self._work_buffers["scanline_counts"],
            self._work_buffers["scanline_offsets"],
            self._work_buffers["visible_counter"],
            0,
            self._scene_count,
            self._work_buffers["scanline_counter"],
            exclusive=True,
        )

    def _emit_scanlines(self, encoder: spy.CommandEncoder, args_buffer: spy.Buffer) -> None:
        dispatch_indirect(
            pipeline=self._p_emit_scanlines,
            args_buffer=args_buffer,
            vars={
                **self._screen_vars(),
                "g_VisibleValues": self._work_buffers["visible_values"],
                "g_VisibleCounter": self._work_buffers["visible_counter"],
                "g_ScanlineCounts": self._work_buffers["scanline_counts"],
                "g_ScanlineOffsets": self._work_buffers["scanline_offsets"],
                "g_ScanlineWorkItems": self._work_buffers["scanline_work_items"],
                **self._prepass_uniforms(self._scene_count),
            },
            command_encoder=encoder,
            debug_label="Emit Scanlines",
            debug_color_index=22,
        )

    def _count_scanline_tiles(self, encoder: spy.CommandEncoder, args_buffer: spy.Buffer) -> None:
        dispatch_indirect(
            pipeline=self._p_count_scanline_tiles,
            args_buffer=args_buffer,
            vars={
                "g_ScanlineWorkItems": self._work_buffers["scanline_work_items"],
                "g_ScanlineCounter": self._work_buffers["scanline_counter"],
                "g_ScanlineTileCounts": self._work_buffers["scanline_tile_counts"],
                **self._prepass_uniforms(self._scene_count),
            },
            command_encoder=encoder,
            debug_label="Count Scanline Tiles",
            debug_color_index=23,
        )

    def _prefix_tile_counts(self, encoder: spy.CommandEncoder) -> None:
        self._prefix_sum.scan_uint_from_count_buffer(
            encoder,
            self._work_buffers["scanline_tile_counts"],
            self._work_buffers["scanline_tile_offsets"],
            self._work_buffers["scanline_counter"],
            0,
            self._max_scanline_entries,
            self._work_buffers["counter"],
            exclusive=True,
        )

    def _emit_tile_entries(self, encoder: spy.CommandEncoder, args_buffer: spy.Buffer) -> None:
        dispatch_indirect(
            pipeline=self._p_emit_tile_entries,
            args_buffer=args_buffer,
            vars={
                "g_ScanlineWorkItems": self._work_buffers["scanline_work_items"],
                "g_ScanlineCounter": self._work_buffers["scanline_counter"],
                "g_ScanlineTileOffsets": self._work_buffers["scanline_tile_offsets"],
                "g_Keys": self._work_buffers["keys"],
                "g_Values": self._work_buffers["values"],
                **self._prepass_uniforms(self._scene_count),
            },
            command_encoder=encoder,
            debug_label="Emit Tile Entries",
            debug_color_index=24,
        )

    def _clear_tile_ranges(self, encoder: spy.CommandEncoder) -> None:
        self._dispatch(self._k_clear_ranges, encoder, spy.uint3(self.tile_count, 1, 1), {"g_TileRanges": self._work_buffers["tile_ranges"], **self._prepass_uniforms(self._scene_count)}, "Clear Tile Ranges", 25)

    def _build_tile_ranges_indirect(self, encoder: spy.CommandEncoder, args_buffer: spy.Buffer) -> None:
        dispatch_indirect(
            pipeline=self._p_build_ranges,
            args_buffer=args_buffer,
            vars={
                "g_SortedKeys": self._sorted_keys(),
                "g_TileRanges": self._work_buffers["tile_ranges"],
                "g_PrepassParams": args_buffer,
                **self._prepass_uniforms(self._scene_count, sorted_count_offset=GPURadixSort.PARAM_ELEMENT_COUNT),
            },
            command_encoder=encoder,
            arg_offset=GPURadixSort.BUILD_RANGE_ARGS_OFFSET,
            debug_label="Build Tile Ranges",
            debug_color_index=26,
        )

    def _record_sort_stage(self, encoder: spy.CommandEncoder) -> spy.Buffer:
        sort_result = self._sorter.sort_key_values_from_count_buffer(
            encoder=encoder,
            keys_buffer=self._work_buffers["keys"],
            values_buffer=self._work_buffers["values"],
            count_buffer=self._work_buffers["counter"],
            count_offset=0,
            max_count=self._max_list_entries,
            max_bits=max(self.tile_bits, 1),
            copy_result_back=False,
        )
        self._sorted_keys_buffer = sort_result.keys_buffer
        self._sorted_values_buffer = sort_result.values_buffer
        return sort_result.args_buffer

    def _record_tile_range_stage(self, encoder: spy.CommandEncoder, args_buffer: spy.Buffer) -> None:
        self._clear_tile_ranges(encoder)
        self._build_tile_ranges_indirect(encoder, args_buffer)

    def _debug_render_enabled(self) -> bool:
        return self.debug_mode != self.DEBUG_MODE_NORMAL

    def _rasterize(self, encoder: spy.CommandEncoder, camera: Camera, background: np.ndarray, output: spy.Texture | None = None) -> None:
        target = self.output_texture if output is None else output
        shader = self._k_raster_debug if self._debug_render_enabled() else self._k_raster
        vars = {**self._scene_vars(), **self._screen_vars(), **self._raster_cache_vars(), "g_SortedValues": self._sorted_values(), "g_TileRanges": self._work_buffers["tile_ranges"], "g_Output": target, **self._raster_grad_decode_scale_var(1.0), **self._raster_grad_fixed_range_vars(), **self._prepass_uniforms(self._scene_count), **self._raster_uniforms(background), **self._anisotropy_uniforms(), **self._camera_uniforms(camera)}
        if self.debug_mode == self.DEBUG_MODE_CLONE_COUNT:
            vars.update(self._debug_clone_count_var())
        if self.debug_mode == self.DEBUG_MODE_GRAD_NORM:
            vars.update(self._debug_grad_norm_var())
        self._dispatch(shader, encoder, self._raster_thread_count(), vars, "Rasterize", 24)

    def _clear_raster_grads(self, encoder: spy.CommandEncoder, splat_count: int) -> None:
        clear_count = int(splat_count) * max(self.TRAINABLE_PARAM_COUNT, self._RASTER_CACHE_PARAM_COUNT)
        self._dispatch(self._raster_grad_shader_set().clear, encoder, spy.uint3(max(clear_count, 1), 1, 1), {**self._raster_grad_vars(), **self._raster_grad_decode_scale_var(1.0), **self._raster_grad_fixed_range_vars(), **self._prepass_uniforms(splat_count)}, "Clear Raster Grads", 25)

    def _rasterize_training_forward(
        self,
        encoder: spy.CommandEncoder,
        camera: Camera,
        background: np.ndarray,
        output: spy.Texture | None = None,
        clone_counts_buffer: spy.Buffer | None = None,
        clone_select_probability: float = 0.0,
        clone_seed: int = 0,
    ) -> None:
        target = self.output_texture if output is None else output
        vars = {
            **self._scene_vars(),
            **self._screen_vars(),
            **self._raster_cache_vars(),
            "g_SortedValues": self._sorted_values(),
            "g_TileRanges": self._work_buffers["tile_ranges"],
            "g_Output": target,
            "g_TrainingForwardState": self._work_buffers["training_forward_state"],
            "g_TrainingDepthState": self._work_buffers["training_depth_state"],
            "g_TrainingDepthRatio": self._work_buffers["training_depth_ratio"],
            "g_TrainingProcessedEnd": self._work_buffers["training_processed_end"],
            "g_TrainingBatchEnd": self._work_buffers["training_batch_end"],
            **self._raster_grad_decode_scale_var(1.0),
            **self._raster_grad_fixed_range_vars(),
            **self._prepass_uniforms(self._scene_count),
            **self._raster_uniforms(background),
            **self._anisotropy_uniforms(),
            **self._camera_uniforms(camera),
        }
        if clone_counts_buffer is not None:
            vars.update(
                {
                    "g_CloneCounts": clone_counts_buffer,
                    "g_CloneSelectProbability": float(max(clone_select_probability, 0.0)),
                    "g_CloneSeed": np.uint32(int(clone_seed)),
                }
            )
        self._dispatch(self._raster_grad_shader_set().training_forward, encoder, self._raster_thread_count(), vars, "Rasterize Training Forward", 26)

    def _rasterize_backward(self, encoder: spy.CommandEncoder, camera: Camera, background: np.ndarray, output_grad: spy.Buffer, depth_ratio_grad: spy.Buffer | None = None) -> None:
        resolved_depth_ratio_grad = self._work_buffers["training_depth_ratio_grad"] if depth_ratio_grad is None else depth_ratio_grad
        if depth_ratio_grad is None:
            self._clear_float_buffer(encoder, resolved_depth_ratio_grad, max(self.width * self.height, 1))
        self._dispatch(self._raster_grad_shader_set().backward, encoder, self._raster_thread_count(), {**self._scene_vars(), **self._raster_cache_vars(), "g_SortedValues": self._sorted_values(), "g_TileRanges": self._work_buffers["tile_ranges"], "g_OutputGrad": output_grad, "g_TrainingForwardState": self._work_buffers["training_forward_state"], "g_TrainingDepthState": self._work_buffers["training_depth_state"], "g_TrainingDepthRatioGrad": resolved_depth_ratio_grad, "g_TrainingProcessedEnd": self._work_buffers["training_processed_end"], "g_TrainingBatchEnd": self._work_buffers["training_batch_end"], **self._raster_grad_vars(), **self._raster_grad_decode_scale_var(1.0), **self._raster_grad_fixed_range_vars(), **self._prepass_uniforms(self._scene_count), **self._raster_uniforms(background), **self._anisotropy_uniforms(), **self._camera_uniforms(camera)}, "Rasterize Backward", 27)

    def _backprop_cached_raster_grads(self, encoder: spy.CommandEncoder, splat_count: int, camera: Camera, grad_scale: float = 1.0) -> None:
        self._dispatch(self._raster_grad_shader_set().backprop, encoder, spy.uint3(max(int(splat_count), 1), 1, 1), {**self._scene_vars(), **self._raster_cache_vars(), **self._raster_grad_vars(), **self._raster_grad_decode_scale_var(grad_scale), **self._raster_grad_fixed_range_vars(), **self._prepass_uniforms(splat_count), **self._anisotropy_uniforms(), **self._camera_uniforms(camera)}, "Backprop Cached Raster Grads", 28)

    def _execute_prepass(self, scene: GaussianScene, camera: Camera, sync_counts: bool = False) -> tuple[int, int]:
        enc = self.device.create_command_encoder()
        with debug_region(enc, "Renderer Prepass", 19):
            self._record_prepass(enc, scene, camera, enqueue_counter_readback=True)
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

    def _record_prepass(self, encoder: spy.CommandEncoder, scene: GaussianScene, camera: Camera, enqueue_counter_readback: bool) -> None:
        self._reset_prepass_counters(encoder)
        self._project_visible_splats(encoder, scene, camera)
        self._sort_visible_splats(encoder)
        visible_args = self._visible_dispatch_args(encoder)
        self._count_visible_scanlines(encoder, visible_args)
        self._prefix_scanline_counts(encoder)
        self._emit_scanlines(encoder, visible_args)
        scanline_args = self._scanline_dispatch_args(encoder)
        self._count_scanline_tiles(encoder, scanline_args)
        self._prefix_tile_counts(encoder)
        if enqueue_counter_readback:
            self._enqueue_counter_readback(encoder)
        self._emit_tile_entries(encoder, scanline_args)
        self._record_tile_range_stage(encoder, self._record_sort_stage(encoder))

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

    def read_raster_cache(self, splat_count: int | None = None) -> np.ndarray:
        count = self._scene_count if splat_count is None else int(splat_count)
        cache = self._read_array(self._work_buffers["raster_cache"], np.float32, max(count, 1), self._RASTER_CACHE_PARAM_COUNT)[:count].copy()
        if count <= 0:
            return cache
        visible = self._read_array(self._work_buffers["splat_visible"], np.uint32, count)
        cache[visible == 0] = 0.0
        return cache

    def read_cached_raster_grads_fixed(self, splat_count: int | None = None) -> np.ndarray:
        count = self._scene_count if splat_count is None else int(splat_count)
        return self._read_array(self._work_buffers["cached_raster_grads_fixed"], np.int32, self._RASTER_CACHE_PARAM_COUNT, max(count, 1)).T[:count].copy()

    def read_cached_raster_grads_fixed_decoded(self, splat_count: int | None = None) -> np.ndarray:
        values = np.asarray(self.read_cached_raster_grads_fixed(splat_count), dtype=np.float64)
        decode_scales = self.cached_raster_grad_fixed_decode_scale_table(values.shape[0])
        return values * decode_scales

    def read_cached_raster_grads_float(self, splat_count: int | None = None) -> np.ndarray:
        count = self._scene_count if splat_count is None else int(splat_count)
        return self._read_array(self._work_buffers["cached_raster_grads_float"], np.float32, self._RASTER_CACHE_PARAM_COUNT, max(count, 1)).T[:count].copy()

    def _prepare_active_cached_raster_grads_float_tensor(self, splat_count: int, command_encoder: spy.CommandEncoder) -> spy.Buffer:
        if self.cached_raster_grad_atomic_mode == self.CACHED_RASTER_GRAD_ATOMIC_MODE_FLOAT:
            return self._work_buffers["cached_raster_grads_float"]
        decoded = self.read_cached_raster_grads_fixed_decoded(splat_count)
        self._work_buffers["cached_raster_grads_metrics_float"].copy_from_numpy(np.ascontiguousarray(decoded.T.reshape(-1), dtype=np.float32))
        return self._work_buffers["cached_raster_grads_metrics_float"]

    def prepare_active_cached_raster_grads_float_tensor(self, splat_count: int | None = None, command_encoder: spy.CommandEncoder | None = None) -> spy.Buffer:
        count = self._scene_count if splat_count is None else int(splat_count)
        if count < 0:
            raise ValueError("splat_count must be non-negative.")
        self._ensure_work_buffers(count)
        if command_encoder is not None:
            return self._prepare_active_cached_raster_grads_float_tensor(count, command_encoder)
        encoder = self.device.create_command_encoder()
        buffer = self._prepare_active_cached_raster_grads_float_tensor(count, encoder)
        self.device.submit_command_buffer(encoder.finish())
        self.device.wait()
        return buffer

    def read_active_cached_raster_grads_float_tensor(self, splat_count: int | None = None) -> np.ndarray:
        count = self._scene_count if splat_count is None else int(splat_count)
        buffer = self.prepare_active_cached_raster_grads_float_tensor(count)
        return self._read_array(buffer, np.float32, self._RASTER_CACHE_PARAM_COUNT, max(count, 1)).T[:count].copy()

    def compute_cached_raster_grad_component_histograms(
        self,
        metrics: Metrics,
        splat_count: int | None = None,
        *,
        bin_count: int = 64,
        min_log10: float = -8.0,
        max_log10: float = 2.0,
    ) -> ParamLog10Histograms:
        count = self._scene_count if splat_count is None else int(splat_count)
        tensor = self.prepare_active_cached_raster_grads_float_tensor(count)
        return metrics.compute_param_tensor_log10_histograms(
            tensor,
            self._RASTER_CACHE_PARAM_COUNT,
            count,
            bin_count=bin_count,
            min_log10=min_log10,
            max_log10=max_log10,
            param_labels=self.CACHED_RASTER_GRAD_COMPONENT_LABELS,
        )

    def compute_cached_raster_grad_component_ranges(self, metrics: Metrics, splat_count: int | None = None) -> ParamTensorRanges:
        count = self._scene_count if splat_count is None else int(splat_count)
        tensor = self.prepare_active_cached_raster_grads_float_tensor(count)
        return metrics.compute_param_tensor_ranges(tensor, self._RASTER_CACHE_PARAM_COUNT, count, param_labels=self.CACHED_RASTER_GRAD_COMPONENT_LABELS)

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

    def _zero_param_grads(self, splat_count: int) -> None:
        self.write_grad_groups(
            splat_count,
            grad_positions=np.zeros((splat_count, 4), dtype=np.float32),
            grad_scales=np.zeros((splat_count, 4), dtype=np.float32),
            grad_rotations=np.zeros((splat_count, 4), dtype=np.float32),
            grad_color_alpha=np.zeros((splat_count, 4), dtype=np.float32),
        )

    def write_cached_raster_grads_fixed(self, splat_count: int, values: np.ndarray) -> None:
        count = int(splat_count)
        data = np.asarray(values, dtype=np.int32).reshape(count, self._RASTER_CACHE_PARAM_COUNT)
        self._work_buffers["cached_raster_grads_fixed"].copy_from_numpy(np.ascontiguousarray(data.T.reshape(-1), dtype=np.int32))

    def write_cached_raster_grads_float(self, splat_count: int, values: np.ndarray) -> None:
        count = int(splat_count)
        data = np.asarray(values, dtype=np.float32).reshape(count, self._RASTER_CACHE_PARAM_COUNT)
        self._work_buffers["cached_raster_grads_float"].copy_from_numpy(np.ascontiguousarray(data.T.reshape(-1), dtype=np.float32))

    def _require_scene(self) -> GaussianScene | SceneBinding:
        if self._current_scene is None:
            raise RuntimeError("Scene is not set.")
        return self._current_scene

    @property
    def cached_raster_grad_atomic_mode(self) -> str:
        return self._cached_raster_grad_atomic_mode

    @cached_raster_grad_atomic_mode.setter
    def cached_raster_grad_atomic_mode(self, mode: str) -> None:
        resolved = self._validate_cached_raster_grad_atomic_mode(mode)
        if resolved == self.CACHED_RASTER_GRAD_ATOMIC_MODE_FLOAT:
            self._ensure_float_raster_grad_shaders()
        self._cached_raster_grad_atomic_mode = resolved

    @staticmethod
    def _validate_positive_finite(name: str, value: float) -> float:
        resolved = float(value)
        if not np.isfinite(resolved) or resolved <= 0.0:
            raise ValueError(f"{name} must be finite and > 0, got {value}.")
        return resolved

    @staticmethod
    def _validate_finite(name: str, value: float) -> float:
        resolved = float(value)
        if not np.isfinite(resolved):
            raise ValueError(f"{name} must be finite, got {value}.")
        return resolved

    @property
    def cached_raster_grad_fixed_ro_local_range(self) -> float:
        return self._cached_raster_grad_fixed_ro_local_range

    @cached_raster_grad_fixed_ro_local_range.setter
    def cached_raster_grad_fixed_ro_local_range(self, value: float) -> None:
        self._cached_raster_grad_fixed_ro_local_range = self._validate_positive_finite("cached_raster_grad_fixed_ro_local_range", value)

    @property
    def cached_raster_grad_fixed_scale_range(self) -> float:
        return self._cached_raster_grad_fixed_scale_range

    @cached_raster_grad_fixed_scale_range.setter
    def cached_raster_grad_fixed_scale_range(self, value: float) -> None:
        self._cached_raster_grad_fixed_scale_range = self._validate_positive_finite("cached_raster_grad_fixed_scale_range", value)

    @property
    def cached_raster_grad_fixed_quat_range(self) -> float:
        return self._cached_raster_grad_fixed_quat_range

    @cached_raster_grad_fixed_quat_range.setter
    def cached_raster_grad_fixed_quat_range(self, value: float) -> None:
        self._cached_raster_grad_fixed_quat_range = self._validate_positive_finite("cached_raster_grad_fixed_quat_range", value)

    @property
    def cached_raster_grad_fixed_color_range(self) -> float:
        return self._cached_raster_grad_fixed_color_range

    @cached_raster_grad_fixed_color_range.setter
    def cached_raster_grad_fixed_color_range(self, value: float) -> None:
        self._cached_raster_grad_fixed_color_range = self._validate_positive_finite("cached_raster_grad_fixed_color_range", value)

    @property
    def cached_raster_grad_fixed_opacity_range(self) -> float:
        return self._cached_raster_grad_fixed_opacity_range

    @cached_raster_grad_fixed_opacity_range.setter
    def cached_raster_grad_fixed_opacity_range(self, value: float) -> None:
        self._cached_raster_grad_fixed_opacity_range = self._validate_positive_finite("cached_raster_grad_fixed_opacity_range", value)

    @property
    def cached_raster_grad_fixed_decode_scales(self) -> np.ndarray:
        return np.array(
            [
                self.cached_raster_grad_fixed_ro_local_range / self._RASTER_GRAD_FIXED_INT_MAX,
                self.cached_raster_grad_fixed_ro_local_range / self._RASTER_GRAD_FIXED_INT_MAX,
                self.cached_raster_grad_fixed_ro_local_range / self._RASTER_GRAD_FIXED_INT_MAX,
                self.cached_raster_grad_fixed_scale_range / self._RASTER_GRAD_FIXED_INT_MAX,
                self.cached_raster_grad_fixed_scale_range / self._RASTER_GRAD_FIXED_INT_MAX,
                self.cached_raster_grad_fixed_scale_range / self._RASTER_GRAD_FIXED_INT_MAX,
                self.cached_raster_grad_fixed_quat_range / self._RASTER_GRAD_FIXED_INT_MAX,
                self.cached_raster_grad_fixed_quat_range / self._RASTER_GRAD_FIXED_INT_MAX,
                self.cached_raster_grad_fixed_quat_range / self._RASTER_GRAD_FIXED_INT_MAX,
                self.cached_raster_grad_fixed_quat_range / self._RASTER_GRAD_FIXED_INT_MAX,
                self.cached_raster_grad_fixed_color_range / self._RASTER_GRAD_FIXED_INT_MAX,
                self.cached_raster_grad_fixed_color_range / self._RASTER_GRAD_FIXED_INT_MAX,
                self.cached_raster_grad_fixed_color_range / self._RASTER_GRAD_FIXED_INT_MAX,
                self.cached_raster_grad_fixed_opacity_range / self._RASTER_GRAD_FIXED_INT_MAX,
            ],
            dtype=np.float32,
        )

    def cached_raster_grad_fixed_decode_scale_table(self, splat_count: int | None = None, raster_cache: np.ndarray | None = None) -> np.ndarray:
        count = self._scene_count if splat_count is None else int(splat_count)
        decode_scales = np.broadcast_to(self.cached_raster_grad_fixed_decode_scales.reshape(1, self._RASTER_CACHE_PARAM_COUNT), (max(count, 0), self._RASTER_CACHE_PARAM_COUNT)).astype(np.float64, copy=True)
        if count <= 0:
            return decode_scales
        cache = self.read_raster_cache(count) if raster_cache is None else np.asarray(raster_cache, dtype=np.float64).reshape(count, self._RASTER_CACHE_PARAM_COUNT)
        scale = np.asarray(cache[:, 3:6], dtype=np.float64)
        avg_scale = np.cbrt(np.maximum(scale[:, 0] * scale[:, 1] * scale[:, 2], 1e-12))
        avg_inv_scale = np.maximum(1.0 / np.maximum(avg_scale, 1e-12), 0.25)
        decode_scales[:, 0:3] *= avg_inv_scale[:, None]
        decode_scales[:, 3:6] *= avg_inv_scale[:, None]
        decode_scales[:, 6:10] *= avg_inv_scale[:, None]
        return decode_scales

    def set_debug_grad_norm_buffer(self, buffer: spy.Buffer | None) -> None:
        self._debug_grad_norm_buffer = buffer

    def set_debug_clone_count_buffer(self, buffer: spy.Buffer | None) -> None:
        self._debug_clone_count_buffer = buffer

    def upload_debug_clone_counts(self, values: np.ndarray) -> None:
        clone_counts = np.ascontiguousarray(values, dtype=np.uint32).reshape(-1)
        self._ensure_work_buffers(max(int(clone_counts.shape[0]), self._scene_count, 1))
        self._work_buffers["debug_clone_count"].copy_from_numpy(np.pad(clone_counts, (0, max(self._work_splat_capacity - clone_counts.shape[0], 0))))
        self._debug_clone_count_buffer = None

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

    def sync_prepass_capacity_for_current_scene(self, camera: Camera) -> bool:
        scene = self._require_scene()
        self._ensure_work_buffers(scene.count, self._pending_min_list_entries)
        generated_entries, _ = self._execute_prepass(scene, camera, sync_counts=True)
        required_entries = min(max(int(generated_entries), 1), self._max_prepass_entries_by_budget())
        if required_entries <= self._max_list_entries:
            return False
        self._pending_min_list_entries = max(self._pending_min_list_entries, required_entries)
        self._ensure_work_buffers(scene.count, self._pending_min_list_entries)
        return True

    def record_prepass_for_current_scene(self, encoder: spy.CommandEncoder, camera: Camera) -> None:
        scene = self._require_scene()
        self._record_prepass(encoder, scene, camera, enqueue_counter_readback=False)

    def rasterize_current_scene(self, encoder: spy.CommandEncoder, camera: Camera, background: np.ndarray) -> None:
        self._require_scene()
        self._rasterize(encoder, camera, background)

    def clear_raster_grads_current_scene(self, encoder: spy.CommandEncoder) -> None:
        self._clear_raster_grads(encoder, self._require_scene().count)

    def rasterize_training_forward_current_scene(
        self,
        encoder: spy.CommandEncoder,
        camera: Camera,
        background: np.ndarray,
        output: spy.Texture | None = None,
        clone_counts_buffer: spy.Buffer | None = None,
        clone_select_probability: float = 0.0,
        clone_seed: int = 0,
    ) -> None:
        self._require_scene()
        self._rasterize_training_forward(encoder, camera, background, output, clone_counts_buffer, clone_select_probability, clone_seed)

    def rasterize_backward_current_scene(self, encoder: spy.CommandEncoder, camera: Camera, background: np.ndarray, output_grad: spy.Buffer, grad_scale: float = 1.0, depth_ratio_grad: spy.Buffer | None = None) -> None:
        self._require_scene()
        self._rasterize_backward(encoder, camera, background, output_grad, depth_ratio_grad)
        self._backprop_cached_raster_grads(encoder, self._scene_count, camera, grad_scale)

    def rasterize_forward_backward_current_scene(self, encoder: spy.CommandEncoder, camera: Camera, background: np.ndarray, output_grad: spy.Buffer, grad_scale: float = 1.0) -> None:
        self._require_scene()
        self._rasterize_training_forward(encoder, camera, background)
        self._rasterize_backward(encoder, camera, background, output_grad)
        self._backprop_cached_raster_grads(encoder, self._scene_count, camera, grad_scale)

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
        if command_encoder is None:
            enc = self.device.create_command_encoder()
            with debug_region(enc, "Renderer Prepass", 19):
                self._record_prepass(enc, scene, camera, enqueue_counter_readback=True)
            self._rasterize(enc, camera, background_np)
            self.device.submit_command_buffer(enc.finish())
            self._counter_readback_frame_id += 1
            self.device.wait()
        else:
            with debug_region(command_encoder, "Renderer Prepass", 19):
                self._record_prepass(command_encoder, scene, camera, enqueue_counter_readback=True)
            self._counter_readback_frame_id += 1
            self._rasterize(command_encoder, camera, background_np)
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
        with debug_region(enc_bwd, "Debug Raster Backward", 29):
            self._clear_raster_grads(enc_bwd, scene.count)
            self.rasterize_backward_current_scene(enc_bwd, camera, background_np, self.output_grad_buffer)
        self.device.submit_command_buffer(enc_bwd.finish())
        self.device.wait()
        grads = self.read_grad_groups(scene.count)
        grads["cached_raster_grads_mode"] = self.cached_raster_grad_atomic_mode
        grads["cached_raster_grads_fixed"] = self.read_cached_raster_grads_fixed(scene.count)
        grads["cached_raster_grads_float"] = self.read_cached_raster_grads_float(scene.count)
        grads["cached_raster_grads_active"] = grads["cached_raster_grads_float"] if self.cached_raster_grad_atomic_mode == self.CACHED_RASTER_GRAD_ATOMIC_MODE_FLOAT else grads["cached_raster_grads_fixed"]
        return grads

    def debug_pipeline_data(self, scene: GaussianScene, camera: Camera) -> dict[str, np.ndarray | int]:
        self._ensure_scene_buffers(scene.count)
        self._ensure_work_buffers(scene.count)
        self._upload_scene(scene)
        generated_entries, sorted_count = self._execute_prepass(scene, camera, sync_counts=True)
        return {
            "generated_entries": generated_entries,
            "sorted_count": sorted_count,
            "keys": self._read_array(self._sorted_keys(), np.uint32, sorted_count),
            "values": self._read_array(self._sorted_values(), np.uint32, sorted_count),
            "tile_ranges": self._read_array(self._work_buffers["tile_ranges"], np.uint32, self.tile_count, 2),
            "screen_center_radius_depth": self._read_array(self._work_buffers["screen_center_radius_depth"], np.float32, scene.count, 4),
            "screen_color_alpha": self._read_array(self._work_buffers["screen_color_alpha"], np.float32, scene.count, 4),
            "screen_ellipse_conic": self._read_array(self._work_buffers["screen_ellipse_conic"], np.float32, scene.count, 4),
            "splat_visible": self._read_array(self._work_buffers["splat_visible"], np.uint32, scene.count),
            "raster_cache": self.read_raster_cache(scene.count),
        }
