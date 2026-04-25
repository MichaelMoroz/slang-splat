from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
import operator
from pathlib import Path
import re

import numpy as np
import slangpy as spy

from ..utility import RW_BUFFER_USAGE, SHADER_ROOT, alloc_buffer, alloc_texture_2d, buffer_to_numpy, debug_region, defer_resource_releases, dispatch, dispatch_indirect, grow_capacity, load_compute_items, load_compute_kernel, load_compute_kernels, remap_named_buffers, thread_count_1d
from ..metrics import Metrics, ParamLog10Histograms, ParamTensorRanges
from ..scan.prefix_sum import GPUPrefixSum
from ..scene.gaussian_scene import GaussianScene
from ..scene.sh_utils import SH_C0, SUPPORTED_SH_COEFF_COUNT, pad_sh_coeffs, resolve_supported_sh_coeffs, rgb_to_sh0, sh_coeffs_to_display_colors
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
    DEBUG_MODE_SPLAT_AGE = "splat_age"
    DEBUG_MODE_DEPTH_MEAN = "depth_mean"
    DEBUG_MODE_DEPTH_STD = "depth_std"
    DEBUG_MODE_DEPTH_LOCAL_MISMATCH = "depth_local_mismatch"
    DEBUG_MODE_ELLIPSE_OUTLINES = "ellipse_outlines"
    DEBUG_MODE_SPLAT_DENSITY = "splat_density"
    DEBUG_MODE_SPLAT_SPATIAL_DENSITY = "splat_spatial_density"
    DEBUG_MODE_SPLAT_SCREEN_DENSITY = "splat_screen_density"
    DEBUG_MODE_GRAD_NORM = "grad_norm"
    DEBUG_MODE_CONTRIBUTION_AMOUNT = "contribution_amount"
    DEBUG_MODE_ADAM_MOMENTUM = "adam_momentum"
    DEBUG_MODE_ADAM_SECOND_MOMENT = "adam_second_moment"
    DEBUG_MODE_SH_VIEW_DEPENDENT = "sh_view_dependent"
    DEBUG_MODE_SH_COEFFICIENT = "sh_coefficient"
    DEBUG_MODE_BLACK_NEGATIVE = "black_negative"
    DEBUG_MODES = (
        DEBUG_MODE_NORMAL,
        DEBUG_MODE_PROCESSED_COUNT,
        DEBUG_MODE_SPLAT_AGE,
        DEBUG_MODE_DEPTH_MEAN,
        DEBUG_MODE_DEPTH_STD,
        DEBUG_MODE_DEPTH_LOCAL_MISMATCH,
        DEBUG_MODE_ELLIPSE_OUTLINES,
        DEBUG_MODE_SPLAT_SPATIAL_DENSITY,
        DEBUG_MODE_SPLAT_SCREEN_DENSITY,
        DEBUG_MODE_GRAD_NORM,
        DEBUG_MODE_SPLAT_DENSITY,
        DEBUG_MODE_CONTRIBUTION_AMOUNT,
        DEBUG_MODE_ADAM_MOMENTUM,
        DEBUG_MODE_ADAM_SECOND_MOMENT,
        DEBUG_MODE_SH_VIEW_DEPENDENT,
        DEBUG_MODE_SH_COEFFICIENT,
        DEBUG_MODE_BLACK_NEGATIVE,
    )
    CACHED_RASTER_GRAD_ATOMIC_MODE_FLOAT = "float"
    CACHED_RASTER_GRAD_ATOMIC_MODE_FIXED = "fixed"
    CACHED_RASTER_GRAD_ATOMIC_MODES = (CACHED_RASTER_GRAD_ATOMIC_MODE_FLOAT, CACHED_RASTER_GRAD_ATOMIC_MODE_FIXED)
    _RASTER_GRAD_FIXED_INT_MAX = np.float32(2147483647.0)
    _DEFAULT_RASTER_GRAD_FIXED_RO_LOCAL_RANGE = np.float32(2.0)
    _DEFAULT_RASTER_GRAD_FIXED_SCALE_RANGE = np.float32(256.0)
    _DEFAULT_RASTER_GRAD_FIXED_COLOR_RANGE = np.float32(8.0)
    _DEFAULT_RASTER_GRAD_FIXED_OPACITY_RANGE = np.float32(8.0)
    _DEFAULT_DEBUG_SPLAT_AGE_RANGE = (0.0, 1.0)
    _DEFAULT_DEBUG_DENSITY_RANGE = (0.0, 20.0)
    _DEFAULT_DEBUG_CONTRIBUTION_RANGE = (0.001, 1.0)
    _DEFAULT_DEBUG_ADAM_MOMENTUM_RANGE = (0.0, 0.1)
    _DEFAULT_DEBUG_DEPTH_MEAN_RANGE = (0.0, 10.0)
    _DEFAULT_DEBUG_DEPTH_STD_RANGE = (0.0, 0.5)
    _DEFAULT_DEBUG_DEPTH_LOCAL_MISMATCH_RANGE = (0.0, 0.5)
    _DEFAULT_DEBUG_SH_COEFF_INDEX = 0
    _SPLAT_CONTRIBUTION_FIXED_SCALE = 256.0
    _COUNTER_READBACK_RING_SIZE = 2
    _SCANLINE_WORK_ITEM_UINTS = 4
    _U32_BYTES = 4
    _F32X4_BYTES = 16
    _OPACITY_EPS = 1e-6
    _MEBIBYTE_BYTES = 1024 * 1024
    _PREPASS_ENTRY_BYTES = (_SCANLINE_WORK_ITEM_UINTS + 4) * _U32_BYTES
    _RASTER_GAUSSIAN_PARAM_COUNT = 11
    _RASTER_GRAD_PARAM_COUNT = 10
    _RW_BUFFER_USAGE = RW_BUFFER_USAGE
    PARAM_POSITION_IDS = (0, 1, 2)
    PARAM_SCALE_IDS = (3, 4, 5)
    PARAM_ROTATION_IDS = (6, 7, 8, 9)
    PARAM_SH_FIRST_ID = 10
    PARAM_SH_COEFF_IDS = tuple(tuple(range(10 + coeff_index * 3, 10 + coeff_index * 3 + 3)) for coeff_index in range(SUPPORTED_SH_COEFF_COUNT))
    PARAM_SH0_IDS = PARAM_SH_COEFF_IDS[0]
    PARAM_SH_IDS = tuple(param_id for group in PARAM_SH_COEFF_IDS for param_id in group)
    PARAM_COLOR_IDS = PARAM_SH_IDS
    PARAM_RAW_OPACITY_ID = PARAM_SH_FIRST_ID + SUPPORTED_SH_COEFF_COUNT * 3
    TRAINABLE_PARAM_COUNT = PARAM_RAW_OPACITY_ID + 1
    SH_COEFF_LABELS = ("SH0 DC", "SH1 X", "SH1 Y", "SH1 Z", "SH2 0", "SH2 1", "SH2 2", "SH2 3", "SH2 4", "SH3 0", "SH3 1", "SH3 2", "SH3 3", "SH3 4", "SH3 5", "SH3 6")
    SH_COMPONENT_RANGE_LABELS = tuple(f"{coeff}.{channel}" for coeff in SH_COEFF_LABELS for channel in ("r", "g", "b"))
    SCENE_PARAM_HISTOGRAM_LABELS = (
        "position.x", "position.y", "position.z",
        "scale.x", "scale.y", "scale.z",
        "quat.w", "quat.x", "quat.y", "quat.z",
        "baseColor.r", "baseColor.g", "baseColor.b",
        *(f"{coeff}.{channel}" for coeff in SH_COEFF_LABELS[1:4] for channel in ("r", "g", "b")),
        *(f"{coeff}.{channel}" for coeff in SH_COEFF_LABELS[4:9] for channel in ("r", "g", "b")),
        *(f"{coeff}.{channel}" for coeff in SH_COEFF_LABELS[9:] for channel in ("r", "g", "b")),
        "opacity",
    )
    SCENE_PARAM_HISTOGRAM_GROUPS = (
        ("position", (0, 1, 2)),
        ("scale", (3, 4, 5)),
        ("quat", (6, 7, 8, 9)),
        ("baseColor (SH0/DC)", (10, 11, 12)),
        ("SH1", tuple(range(13, 22))),
        ("SH2", tuple(range(22, 37))),
        ("SH3", tuple(range(37, 58))),
        ("opacity", (58,)),
    )
    CACHED_RASTER_GRAD_COMPONENT_LABELS = (
        "centerDir.x", "centerDir.y", "centerDir.z",
        "sigmaOrtho.xx", "sigmaOrtho.xy", "sigmaOrtho.yy",
        "color.r", "color.g", "color.b", "opacity",
    )
    _SCENE_SHADER_VARS = {"splat_params": "g_SplatParams"}
    _SCREEN_SHADER_VARS = {"screen_center_radius_depth": "g_ScreenCenterRadiusDepth", "screen_color_alpha": "g_ScreenColorAlpha", "screen_ellipse_conic": "g_ScreenEllipseConic", "splat_visible": "g_SplatVisible", "splat_visible_area_px": "g_SplatVisibleAreaPx"}
    _RASTER_CACHE_SHADER_VARS = {"raster_cache": "g_RasterCache"}
    _RASTER_GRAD_SHADER_VARS = {
        "param_grads": "g_ParamGrads",
        "cached_raster_grads_fixed": "g_CachedRasterGradsFixed",
        "cached_raster_grads_float": "g_CachedRasterGradsFloat",
    }
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

    def _max_prepass_entries_by_budget(self) -> int:
        return max(self._max_prepass_memory_bytes // self._PREPASS_ENTRY_BYTES, 1)

    @staticmethod
    def _background_array(background: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        return np.asarray(background, dtype=np.float32).reshape(3)

    def _camera_uniforms(self, camera: Camera, uniform_name: str = "g_Camera") -> dict[str, object]:
        k1, k2 = camera.distortion_coeffs(self.proj_distortion_k1, self.proj_distortion_k2)
        return {
            str(uniform_name): {
                **camera.gpu_params(self.width, self.height),
                "projDistortionK1": float(k1),
                "projDistortionK2": float(k2),
            }
        }

    def _sort_camera_position_var(self, camera: Camera, sort_camera_position: np.ndarray | None = None, sort_camera_dither_sigma: float = 0.0, sort_camera_dither_seed: int = 0) -> dict[str, object]:
        position = camera.position if sort_camera_position is None else np.asarray(sort_camera_position, dtype=np.float32).reshape(3)
        return {
            "g_SortCameraPosition": spy.float3(float(position[0]), float(position[1]), float(position[2])),
            "g_SortCameraDitherSigma": float(max(sort_camera_dither_sigma, 0.0)),
            "g_SortCameraDitherSeed": np.uint32(int(sort_camera_dither_seed) & 0xFFFFFFFF),
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

    def _raster_uniforms(self, background: np.ndarray, training_background_mode: int = 0, training_background_seed: int = 0) -> dict[str, object]:
        return {
            "g_Raster": {
                "width": int(self.width),
                "height": int(self.height),
                "maxSplatSteps": int(self.max_splat_steps),
                "alphaCutoff": float(self.alpha_cutoff),
                "transmittanceThreshold": float(self.transmittance_threshold),
                "background": spy.float3(*background.tolist()),
                "trainingBackgroundMode": np.uint32(max(int(training_background_mode), 0)),
                "trainingBackgroundSeed": np.uint32(int(training_background_seed)),
                "debugMode": np.uint32(self._debug_mode_u32(self.debug_mode)),
                "shBand": np.uint32(self.sh_band),
                "debugGradNormThreshold": float(max(self.debug_grad_norm_threshold, 0.0)),
                "debugEllipseThicknessPx": float(self.debug_ellipse_thickness_px),
                "debugSplatAgeRange": spy.float2(*self.debug_splat_age_range),
                "debugDensityRange": spy.float2(*self.debug_density_range),
                "debugContributionRange": spy.float2(*self.debug_contribution_range),
                "debugContributionScale": float(self._debug_contribution_scale),
                "debugAdamMomentumRange": spy.float2(*self.debug_adam_momentum_range),
                "debugDepthMeanRange": spy.float2(*self.debug_depth_mean_range),
                "debugDepthStdRange": spy.float2(*self.debug_depth_std_range),
                "debugDepthLocalMismatchRange": spy.float2(*self.debug_depth_local_mismatch_range),
                "debugDepthLocalMismatchSmoothRadius": float(self.debug_depth_local_mismatch_smooth_radius),
                "debugDepthLocalMismatchRejectRadius": float(self.debug_depth_local_mismatch_reject_radius),
                "debugSHCoeffIndex": np.uint32(self.debug_sh_coeff_index),
                "debugGaussianScaleMultiplier": float(self.debug_gaussian_scale_multiplier),
                "debugMinOpacity": float(self.debug_min_opacity),
                "debugOpacityMultiplier": float(self.debug_opacity_multiplier),
                "debugEllipseScaleMultiplier": float(self.debug_ellipse_scale_multiplier),
            }
        }

    def _anisotropy_uniforms(self) -> dict[str, object]:
        return {"g_MaxAnisotropy": float(max(self.max_anisotropy, 1.0))}

    def _disabled_training_sample_vars(self) -> dict[str, object]:
        return {
            "g_TrainingSubsample": {
                "enabled": np.uint32(0),
                "factor": np.uint32(1),
                "nativeWidth": np.uint32(max(int(self.width), 1)),
                "nativeHeight": np.uint32(max(int(self.height), 1)),
                "frameIndex": np.uint32(0),
                "stepIndex": np.uint32(0),
            }
        }

    def _scene_vars(self) -> dict[str, object]:
        return self._buffer_vars(self._SCENE_SHADER_VARS, self._scene_buffers)

    def _screen_vars(self) -> dict[str, object]:
        return self._buffer_vars(self._SCREEN_SHADER_VARS, self._work_buffers)

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
            "g_CachedRasterGradFixedColorRange": float(self.cached_raster_grad_fixed_color_range),
            "g_CachedRasterGradFixedOpacityRange": float(self.cached_raster_grad_fixed_opacity_range),
        }

    def _debug_grad_norm_var(self) -> dict[str, object]:
        return {"g_DebugGradNorm": self._debug_grad_norm_buffer if self._debug_grad_norm_buffer is not None else self._work_buffers["debug_grad_norm"]}

    def _debug_splat_age_var(self) -> dict[str, object]:
        return {"g_SplatAges": self._debug_splat_age_buffer if self._debug_splat_age_buffer is not None else self._work_buffers["debug_splat_age"]}

    def _debug_splat_contribution_var(self) -> dict[str, object]:
        return {"g_SplatContribution": self._debug_splat_contribution_buffer if self._debug_splat_contribution_buffer is not None else self._work_buffers["training_splat_contribution"]}

    def _debug_adam_moments_var(self) -> dict[str, object]:
        if self._debug_adam_moments_buffer is None: raise RuntimeError("Adam moment debug mode requires an Adam moments buffer.")
        return {"g_DebugAdamMoments": self._debug_adam_moments_buffer}

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
        return np.asarray(self.output_texture.to_numpy(), dtype=np.float32)[: self.height, : self.width].copy()

    def _create_shaders(self) -> None:
        for attr, shader in load_compute_items(
            self.device,
            {
                attr: (kind, SHADER_ROOT / "renderer" / shader_name, entry)
                for attr, kind, shader_name, entry in self._SHADERS
            },
        ).items():
            setattr(self, attr, shader)
        self._fixed_raster_grad_shaders = self._load_raster_grad_shaders("Fixed")
        self._k_clear_float_buffer = load_compute_kernel(self.device, SHADER_ROOT / "utility" / "metrics" / "metrics.slang", "csClearFloatBuffer")

    def _load_raster_grad_shaders(self, entry_suffix: str) -> _RasterGradShaderSet:
        kernels = load_compute_kernels(
            self.device,
            SHADER_ROOT / "renderer" / "gaussian_raster_stage.slang",
            {
                "training_forward": f"csRasterizeTrainingForward{entry_suffix}",
                "clear": f"csClearRasterGrads{entry_suffix}",
                "backward": f"csRasterizeBackward{entry_suffix}",
                "backprop": f"csBackpropCachedRasterGrads{entry_suffix}",
            },
        )
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

    def clear_scene_resources(self) -> None:
        defer_resource_releases((
            *self._scene_buffers.values(),
            *self._work_buffers.values(),
            self._output_texture,
            self._output_grad_buffer,
            *self._counter_readback_ring,
        ))
        self._current_scene = None
        self._scene_count = self._scene_capacity = self._max_list_entries = self._work_splat_capacity = self._max_scanline_entries = 0
        self._scene_buffers = {}
        self._work_buffers = {}
        self._resource_groups = _RendererResourceGroups(scene={}, frame={}, prepass={}, raster={}, grad={}, debug={})
        self._output_texture = None
        self._output_grad_buffer = None
        self._sorted_keys_buffer = None
        self._sorted_values_buffer = None
        self._last_stats = {}
        self._counter_readback_ring = []
        self._counter_readback_capacity = []
        self._counter_readback_frame_id = 0
        self._pending_min_list_entries = self._delayed_generated_entries = self._delayed_written_entries = 0
        self._delayed_overflow = self._delayed_stats_valid = False

    def _set_render_resolution_geometry(self, width: int, height: int) -> None:
        self.width, self.height = max(int(width), 1), max(int(height), 1)
        self.tile_width, self.tile_height = (self.width + self.tile_size - 1) // self.tile_size, (self.height + self.tile_size - 1) // self.tile_size
        self.tile_count = self.tile_width * self.tile_height
        self.tile_bits = int(np.ceil(np.log2(max(self.tile_count, 2))))
        self.depth_bits = 32 - self.tile_bits

    def _set_render_capacity_geometry(self, width: int, height: int) -> None:
        self._render_capacity_width, self._render_capacity_height = max(int(width), 1), max(int(height), 1)
        self._render_capacity_tile_width = (self._render_capacity_width + self.tile_size - 1) // self.tile_size
        self._render_capacity_tile_height = (self._render_capacity_height + self.tile_size - 1) // self.tile_size
        self._render_capacity_tile_count = self._render_capacity_tile_width * self._render_capacity_tile_height

    def _render_pixel_capacity(self) -> int:
        return max(int(self._render_capacity_width) * int(self._render_capacity_height), 1)

    def _release_frame_sized_resources(self) -> None:
        defer_resource_releases((
            *self._work_buffers.values(),
            self._output_texture,
            self._output_grad_buffer,
            *self._counter_readback_ring,
        ))
        self._work_buffers = {}
        self._resource_groups.prepass = {}
        self._resource_groups.raster = {}
        self._resource_groups.grad = {}
        self._resource_groups.debug = {}
        self._resource_groups.frame = {}
        self._output_texture = None
        self._output_grad_buffer = None
        self._sorted_keys_buffer = None
        self._sorted_values_buffer = None
        self._max_list_entries = self._work_splat_capacity = self._max_scanline_entries = 0
        self._counter_readback_ring = []
        self._counter_readback_capacity = []
        self._counter_readback_frame_id = 0
        self._pending_min_list_entries = self._delayed_generated_entries = self._delayed_written_entries = 0
        self._delayed_overflow = self._delayed_stats_valid = False

    def ensure_render_capacity(self, width: int, height: int) -> bool:
        target_width = max(max(int(width), 1), int(self._render_capacity_width))
        target_height = max(max(int(height), 1), int(self._render_capacity_height))
        if (target_width, target_height) == (self._render_capacity_width, self._render_capacity_height):
            return False
        self._release_frame_sized_resources()
        self._set_render_capacity_geometry(target_width, target_height)
        if self._scene_count > 0:
            self._ensure_work_buffers(self._scene_count)
        return True

    def set_render_resolution(self, width: int, height: int) -> bool:
        target_size = (max(int(width), 1), max(int(height), 1))
        capacity_changed = self.ensure_render_capacity(*target_size)
        if (self.width, self.height) == target_size:
            return capacity_changed
        self._set_render_resolution_geometry(*target_size)
        if self._scene_count > 0:
            self._ensure_work_buffers(self._scene_count)
        return True

    def resize(self, width: int, height: int) -> bool:
        target_size = (max(int(width), 1), max(int(height), 1))
        if (self.width, self.height) == target_size and (self._render_capacity_width, self._render_capacity_height) == target_size:
            return False
        self._release_frame_sized_resources()
        self._set_render_resolution_geometry(*target_size)
        self._set_render_capacity_geometry(*target_size)
        if self._scene_count > 0:
            self._ensure_work_buffers(self._scene_count)
        return True

    def bind_scene_count(self, splat_count: int) -> None:
        count = max(int(splat_count), 0)
        self._ensure_scene_buffers(count)
        self._ensure_work_buffers(count)
        self._current_scene = SceneBinding(count=count)

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
        debug_ellipse_thickness_px: float = 4.0,
        debug_gaussian_scale_multiplier: float = 1.0,
        debug_min_opacity: float = 0.0,
        debug_opacity_multiplier: float = 1.0,
        debug_ellipse_scale_multiplier: float = 1.0,
        debug_splat_age_range: tuple[float, float] = _DEFAULT_DEBUG_SPLAT_AGE_RANGE,
        debug_density_range: tuple[float, float] = _DEFAULT_DEBUG_DENSITY_RANGE,
        debug_contribution_range: tuple[float, float] = _DEFAULT_DEBUG_CONTRIBUTION_RANGE,
        debug_adam_momentum_range: tuple[float, float] = _DEFAULT_DEBUG_ADAM_MOMENTUM_RANGE,
        debug_depth_mean_range: tuple[float, float] = _DEFAULT_DEBUG_DEPTH_MEAN_RANGE,
        debug_depth_std_range: tuple[float, float] = _DEFAULT_DEBUG_DEPTH_STD_RANGE,
        debug_depth_local_mismatch_range: tuple[float, float] = _DEFAULT_DEBUG_DEPTH_LOCAL_MISMATCH_RANGE,
        debug_depth_local_mismatch_smooth_radius: float = 2.0,
        debug_depth_local_mismatch_reject_radius: float = 4.0,
        debug_sh_coeff_index: int = _DEFAULT_DEBUG_SH_COEFF_INDEX,
        cached_raster_grad_atomic_mode: str = CACHED_RASTER_GRAD_ATOMIC_MODE_FIXED,
        cached_raster_grad_fixed_ro_local_range: float = 2.0,
        cached_raster_grad_fixed_scale_range: float = 256.0,
        cached_raster_grad_fixed_color_range: float = 8.0,
        cached_raster_grad_fixed_opacity_range: float = 8.0,
        use_sh: bool = True,
        sh_band: int | None = None,
    ) -> None:
        self.device, self.width, self.height = device, int(width), int(height)
        self._types_shader_path = Path(SHADER_ROOT / "renderer" / "gaussian_types.slang")
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
        self.sh_band = (3 if bool(use_sh) else 0) if sh_band is None else int(sh_band)
        self.debug_mode = self._resolve_debug_mode(debug_mode, debug_show_ellipses, debug_show_processed_count, debug_show_grad_norm)
        self.debug_show_ellipses = self.debug_mode == self.DEBUG_MODE_ELLIPSE_OUTLINES
        self.debug_show_processed_count = self.debug_mode == self.DEBUG_MODE_PROCESSED_COUNT
        self.debug_show_grad_norm = self.debug_mode == self.DEBUG_MODE_GRAD_NORM
        self.debug_grad_norm_threshold = float(debug_grad_norm_threshold)
        self.debug_ellipse_thickness_px = float(debug_ellipse_thickness_px)
        self.debug_gaussian_scale_multiplier = float(max(debug_gaussian_scale_multiplier, 1e-8))
        self.debug_min_opacity = float(max(debug_min_opacity, 0.0))
        self.debug_opacity_multiplier = float(max(debug_opacity_multiplier, 0.0))
        self.debug_ellipse_scale_multiplier = float(max(debug_ellipse_scale_multiplier, 1e-8))
        self.debug_splat_age_range = tuple(float(x) for x in debug_splat_age_range)
        self.debug_density_range = tuple(float(x) for x in debug_density_range)
        self.debug_contribution_range = tuple(float(x) for x in debug_contribution_range)
        self.debug_adam_momentum_range = tuple(float(x) for x in debug_adam_momentum_range)
        self.debug_depth_mean_range = tuple(float(x) for x in debug_depth_mean_range)
        self.debug_depth_std_range = tuple(float(x) for x in debug_depth_std_range)
        self.debug_depth_local_mismatch_range = tuple(float(x) for x in debug_depth_local_mismatch_range)
        self.debug_depth_local_mismatch_smooth_radius = float(debug_depth_local_mismatch_smooth_radius)
        self.debug_depth_local_mismatch_reject_radius = float(debug_depth_local_mismatch_reject_radius)
        self.debug_sh_coeff_index = min(max(int(debug_sh_coeff_index), 0), SUPPORTED_SH_COEFF_COUNT - 1)
        self._set_render_resolution_geometry(self.width, self.height)
        self._create_shaders()
        self._sorter = GPURadixSort(self.device)
        self._prefix_sum = GPUPrefixSum(self.device)
        self._set_render_capacity_geometry(self.width, self.height)
        self._scene_count = self._scene_capacity = self._max_list_entries = self._work_splat_capacity = self._max_scanline_entries = 0
        self._current_scene: GaussianScene | None = None
        self._scene_buffers: dict[str, spy.Buffer] = {}
        self._work_buffers: dict[str, spy.Buffer] = {}
        self._resource_groups = _RendererResourceGroups(scene={}, frame={}, prepass={}, raster={}, grad={}, debug={})
        self._debug_grad_norm_buffer: spy.Buffer | None = None
        self._debug_splat_age_buffer: spy.Buffer | None = None
        self._debug_splat_contribution_buffer: spy.Buffer | None = None
        self._debug_adam_moments_buffer: spy.Buffer | None = None
        self._debug_contribution_scale = 1.0 / self._SPLAT_CONTRIBUTION_FIXED_SCALE
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
        self._cached_raster_grad_fixed_color_range = self._DEFAULT_RASTER_GRAD_FIXED_COLOR_RANGE
        self._cached_raster_grad_fixed_opacity_range = self._DEFAULT_RASTER_GRAD_FIXED_OPACITY_RANGE
        self._zero_u32_buffer = alloc_buffer(
            self.device,
            name="renderer.zero_u32",
            size=self._U32_BYTES,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination,
        )
        self._zero_u32_buffer.copy_from_numpy(np.array([0], dtype=np.uint32))
        self.cached_raster_grad_atomic_mode = cached_raster_grad_atomic_mode
        self.cached_raster_grad_fixed_ro_local_range = cached_raster_grad_fixed_ro_local_range
        self.cached_raster_grad_fixed_scale_range = cached_raster_grad_fixed_scale_range
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

    def _ensure_scene_buffers(self, splat_count: int) -> None:
        if self._scene_buffers and splat_count <= self._scene_capacity:
            self._scene_count = splat_count
            return
        defer_resource_releases(self._scene_buffers.values())
        self._scene_capacity, self._scene_count = grow_capacity(splat_count, self._scene_capacity), splat_count
        param_bytes = max(self._scene_capacity, 1) * self.TRAINABLE_PARAM_COUNT * self._U32_BYTES
        self._resource_groups.scene = {
            name: alloc_buffer(self.device, name=f"renderer.scene.{name}", size=param_bytes, usage=self._RW_BUFFER_USAGE)
            for name in self._SCENE_SHADER_VARS
        }
        self._scene_buffers = self._resource_groups.scene

    def _ensure_work_buffers(self, splat_count: int, min_list_entries: int = 0) -> None:
        max_entries = self._max_prepass_entries_by_budget()
        required_splats = max(splat_count, 1)
        required_entries = min(max(splat_count * self.list_capacity_multiplier, min_list_entries, 1), max_entries)
        if self._work_buffers and required_splats <= self._work_splat_capacity and required_entries <= self._max_list_entries and required_entries <= self._max_scanline_entries and self._output_texture is not None and self._output_grad_buffer is not None:
            return
        defer_resource_releases(self._work_buffers.values())
        self._work_splat_capacity = grow_capacity(required_splats, self._work_splat_capacity)
        self._max_list_entries = min(grow_capacity(required_entries, self._max_list_entries), max_entries)
        self._max_scanline_entries = min(grow_capacity(required_entries, self._max_scanline_entries), max_entries)
        sized = {
            "screen_center_radius_depth": max(self._work_splat_capacity, 1) * self._F32X4_BYTES,
            "screen_color_alpha": max(self._work_splat_capacity, 1) * self._F32X4_BYTES,
            "screen_ellipse_conic": max(self._work_splat_capacity, 1) * self._F32X4_BYTES,
            "splat_visible": max(self._work_splat_capacity, 1) * self._U32_BYTES,
            "splat_visible_area_px": max(self._work_splat_capacity, 1) * self._U32_BYTES,
            "fallback_clone_counts": max(self._work_splat_capacity, 1) * self._U32_BYTES,
            "debug_splat_age": max(self._work_splat_capacity, 1) * self._U32_BYTES,
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
            "tile_ranges": max(self._render_capacity_tile_count, 1) * 8,
            "training_forward_state": self._render_pixel_capacity() * self._F32X4_BYTES,
            "training_density": self._render_pixel_capacity() * self._U32_BYTES,
            "training_rgb_loss": self._render_pixel_capacity() * self._U32_BYTES,
            "training_rgb_loss_total": self._U32_BYTES,
            "training_target_edge": self._render_pixel_capacity() * self._U32_BYTES,
            "training_target_edge_total": self._U32_BYTES,
            "training_regularizer_grad": self._render_pixel_capacity() * self._U32_BYTES,
            "training_processed_end": self._render_pixel_capacity() * self._U32_BYTES,
            "training_batch_end": max(self._render_capacity_tile_count, 1) * self._U32_BYTES,
            "training_splat_contribution": max(self._work_splat_capacity, 1) * self._U32_BYTES,
            "raster_cache": max(self._work_splat_capacity, 1) * self._RASTER_GAUSSIAN_PARAM_COUNT * self._U32_BYTES,
            "param_grads": max(self._work_splat_capacity, 1) * self.TRAINABLE_PARAM_COUNT * self._U32_BYTES,
            "cached_raster_grads_fixed": max(self._work_splat_capacity, 1) * self._RASTER_GRAD_PARAM_COUNT * self._U32_BYTES,
            "cached_raster_grads_float": max(self._work_splat_capacity, 1) * self._RASTER_GRAD_PARAM_COUNT * self._U32_BYTES,
            "cached_raster_grads_metrics_float": max(self._work_splat_capacity, 1) * self._RASTER_GRAD_PARAM_COUNT * self._U32_BYTES,
        }
        allocated = {name: alloc_buffer(self.device, name=f"renderer.work.{name}", size=size, usage=self._RW_BUFFER_USAGE) for name, size in sized.items()}
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
                "splat_visible_area_px",
                "training_forward_state",
                "training_density",
                "training_rgb_loss",
                "training_rgb_loss_total",
                "training_target_edge",
                "training_target_edge_total",
                "training_regularizer_grad",
                "training_processed_end",
                "training_batch_end",
                "training_splat_contribution",
                "raster_cache",
            )
        }
        self._resource_groups.grad = {
            name: allocated[name]
            for name in ("param_grads", "cached_raster_grads_fixed", "cached_raster_grads_float", "cached_raster_grads_metrics_float")
        }
        self._resource_groups.debug = {
            "fallback_clone_counts": allocated["fallback_clone_counts"],
            "debug_splat_age": allocated["debug_splat_age"],
            "debug_grad_norm": allocated["debug_grad_norm"],
        }
        self._work_buffers = self._resource_groups.merged_work_buffers()
        self._sorted_keys_buffer = self._work_buffers["keys"]
        self._sorted_values_buffer = self._work_buffers["values"]
        self._work_buffers["fallback_clone_counts"].copy_from_numpy(np.zeros((max(self._work_splat_capacity, 1),), dtype=np.uint32))
        self._work_buffers["debug_splat_age"].copy_from_numpy(np.ones((max(self._work_splat_capacity, 1),), dtype=np.float32))
        self._work_buffers["debug_grad_norm"].copy_from_numpy(np.zeros((max(self._work_splat_capacity, 1),), dtype=np.float32))
        self._work_buffers["training_splat_contribution"].copy_from_numpy(np.zeros((max(self._work_splat_capacity, 1),), dtype=np.uint32))
        self._work_buffers["training_rgb_loss"].copy_from_numpy(np.zeros((self._render_pixel_capacity(),), dtype=np.float32))
        self._work_buffers["training_rgb_loss_total"].copy_from_numpy(np.zeros((1,), dtype=np.float32))
        self._work_buffers["training_target_edge"].copy_from_numpy(np.zeros((self._render_pixel_capacity(),), dtype=np.float32))
        self._work_buffers["training_target_edge_total"].copy_from_numpy(np.zeros((1,), dtype=np.float32))
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
            self._output_texture = alloc_texture_2d(
                self.device,
                name="renderer.output_texture",
                format=spy.Format.rgba32_float,
                width=self._render_capacity_width,
                height=self._render_capacity_height,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            )

    def _ensure_output_grad_buffer(self) -> None:
        if self._output_grad_buffer is None:
            self._output_grad_buffer = alloc_buffer(
                self.device,
                name="renderer.output_grad",
                size=self._render_pixel_capacity() * 4 * self._U32_BYTES,
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
        sh_coeffs = resolve_supported_sh_coeffs(scene.sh_coeffs, scene.colors)
        for axis, param_id in enumerate(cls.PARAM_POSITION_IDS):
            packed[cls._param_slice(count, param_id)] = np.asarray(scene.positions[:, axis], dtype=np.float32)
        for axis, param_id in enumerate(cls.PARAM_SCALE_IDS):
            packed[cls._param_slice(count, param_id)] = np.asarray(scene.scales[:, axis], dtype=np.float32)
        for axis, param_id in enumerate(cls.PARAM_ROTATION_IDS):
            packed[cls._param_slice(count, param_id)] = np.asarray(scene.rotations[:, axis], dtype=np.float32)
        for coeff_index, param_ids in enumerate(cls.PARAM_SH_COEFF_IDS):
            for axis, param_id in enumerate(param_ids):
                packed[cls._param_slice(count, param_id)] = np.asarray(sh_coeffs[:, coeff_index, axis], dtype=np.float32)
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
            "sh_coeffs": np.zeros((count, SUPPORTED_SH_COEFF_COUNT, 3), dtype=np.float32),
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
        for coeff_index, param_ids in enumerate(cls.PARAM_SH_COEFF_IDS):
            for axis, param_id in enumerate(param_ids):
                groups["sh_coeffs"][:, coeff_index, axis] = flat[cls._param_slice(count, param_id)]
        groups["color_alpha"][:, :3] = sh_coeffs_to_display_colors(groups["sh_coeffs"])
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
        sh_coeffs: np.ndarray | None = None,
        color_alpha: np.ndarray | None = None,
        color_is_grad: bool = False,
    ) -> np.ndarray:
        count = max(int(splat_count), 0)
        packed = np.zeros((cls.TRAINABLE_PARAM_COUNT * max(count, 1),), dtype=np.float32)
        if count <= 0:
            return packed
        sh_groups = np.zeros((count, SUPPORTED_SH_COEFF_COUNT, 3), dtype=np.float32)
        if sh_coeffs is not None:
            sh_groups = pad_sh_coeffs(np.asarray(sh_coeffs, dtype=np.float32).reshape(count, -1, 3), SUPPORTED_SH_COEFF_COUNT)
        elif color_alpha is not None:
            color_values = np.asarray(color_alpha, dtype=np.float32).reshape(count, 4)
            sh_groups[:, 0, :] = color_values[:, :3] * SH_C0 if color_is_grad else rgb_to_sh0(color_values[:, :3])
        groups = {
            "positions": np.zeros((count, 4), dtype=np.float32) if positions is None else np.asarray(positions, dtype=np.float32).reshape(count, 4),
            "scales": np.zeros((count, 4), dtype=np.float32) if scales is None else np.asarray(scales, dtype=np.float32).reshape(count, 4),
            "rotations": np.zeros((count, 4), dtype=np.float32) if rotations is None else np.asarray(rotations, dtype=np.float32).reshape(count, 4),
            "sh_coeffs": sh_groups,
            "color_alpha": np.zeros((count, 4), dtype=np.float32) if color_alpha is None else np.asarray(color_alpha, dtype=np.float32).reshape(count, 4),
        }
        for axis, param_id in enumerate(cls.PARAM_POSITION_IDS):
            packed[cls._param_slice(count, param_id)] = groups["positions"][:, axis]
        for axis, param_id in enumerate(cls.PARAM_SCALE_IDS):
            packed[cls._param_slice(count, param_id)] = groups["scales"][:, axis]
        for axis, param_id in enumerate(cls.PARAM_ROTATION_IDS):
            packed[cls._param_slice(count, param_id)] = groups["rotations"][:, axis]
        for coeff_index, param_ids in enumerate(cls.PARAM_SH_COEFF_IDS):
            for axis, param_id in enumerate(param_ids):
                packed[cls._param_slice(count, param_id)] = groups["sh_coeffs"][:, coeff_index, axis]
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
        self._counter_readback_ring = [alloc_buffer(self.device, name=f"renderer.counter_readback[{idx}]", size=4, usage=usage) for idx in range(self._COUNTER_READBACK_RING_SIZE)]
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

    def _project_visible_splats(self, encoder: spy.CommandEncoder, scene: GaussianScene, camera: Camera, sort_camera_position: np.ndarray | None = None, sort_camera_dither_sigma: float = 0.0, sort_camera_dither_seed: int = 0) -> None:
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
                **self._sort_camera_position_var(camera, sort_camera_position, sort_camera_dither_sigma, sort_camera_dither_seed),
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
        vars = {**self._scene_vars(), **self._screen_vars(), **self._raster_cache_vars(), "g_SortedValues": self._sorted_values(), "g_TileRanges": self._work_buffers["tile_ranges"], "g_Output": target, **self._raster_grad_decode_scale_var(1.0), **self._raster_grad_fixed_range_vars(), **self._prepass_uniforms(self._scene_count), **self._raster_uniforms(background), **self._anisotropy_uniforms(), **self._camera_uniforms(camera), **self._camera_uniforms(camera, "g_TrainingNativeCamera"), **self._disabled_training_sample_vars()}
        if self.debug_mode == self.DEBUG_MODE_SPLAT_AGE:
            vars.update(self._debug_splat_age_var())
        if self.debug_mode == self.DEBUG_MODE_CONTRIBUTION_AMOUNT:
            vars.update(self._debug_splat_contribution_var())
        if self.debug_mode in (self.DEBUG_MODE_ADAM_MOMENTUM, self.DEBUG_MODE_ADAM_SECOND_MOMENT):
            vars.update(self._debug_adam_moments_var())
        if self.debug_mode == self.DEBUG_MODE_GRAD_NORM:
            vars.update(self._debug_grad_norm_var())
        self._dispatch(shader, encoder, self._raster_thread_count(), vars, "Rasterize", 24)

    def _clear_raster_grads(self, encoder: spy.CommandEncoder, splat_count: int) -> None:
        clear_count = int(splat_count) * max(self.TRAINABLE_PARAM_COUNT, self._RASTER_GAUSSIAN_PARAM_COUNT, self._RASTER_GRAD_PARAM_COUNT)
        self._dispatch(self._raster_grad_shader_set().clear, encoder, spy.uint3(max(clear_count, 1), 1, 1), {**self._raster_grad_vars(), **self._raster_grad_decode_scale_var(1.0), **self._raster_grad_fixed_range_vars(), **self._prepass_uniforms(splat_count)}, "Clear Raster Grads", 25)

    def _rasterize_training_forward(
        self,
        encoder: spy.CommandEncoder,
        camera: Camera,
        background: np.ndarray,
        output: spy.Texture | None = None,
        clone_counts_buffer: spy.Buffer | None = None,
        splat_contribution_buffer: spy.Buffer | None = None,
        training_background_mode: int = 0,
        training_background_seed: int = 0,
        training_native_camera: Camera | None = None,
        training_sample_vars: dict[str, object] | None = None,
    ) -> None:
        target = self.output_texture if output is None else output
        resolved_native_camera = camera if training_native_camera is None else training_native_camera
        resolved_sample_vars = self._disabled_training_sample_vars() if training_sample_vars is None else training_sample_vars
        vars = {
            **self._scene_vars(),
            **self._screen_vars(),
            **self._raster_cache_vars(),
            "g_SortedValues": self._sorted_values(),
            "g_TileRanges": self._work_buffers["tile_ranges"],
            "g_Output": target,
            "g_TrainingForwardState": self._work_buffers["training_forward_state"],
            "g_TrainingDensity": self._work_buffers["training_density"],
            "g_TrainingProcessedEnd": self._work_buffers["training_processed_end"],
            "g_TrainingBatchEnd": self._work_buffers["training_batch_end"],
            "g_CloneCounts": self._work_buffers["fallback_clone_counts"] if clone_counts_buffer is None else clone_counts_buffer,
            "g_SplatContribution": self._work_buffers["training_splat_contribution"] if splat_contribution_buffer is None else splat_contribution_buffer,
            **self._raster_grad_decode_scale_var(1.0),
            **self._raster_grad_fixed_range_vars(),
            **self._prepass_uniforms(self._scene_count),
            **self._raster_uniforms(background, training_background_mode, training_background_seed),
            **self._anisotropy_uniforms(),
            **self._camera_uniforms(camera),
            **self._camera_uniforms(resolved_native_camera, "g_TrainingNativeCamera"),
            **resolved_sample_vars,
        }
        self._dispatch(self._raster_grad_shader_set().training_forward, encoder, self._raster_thread_count(), vars, "Rasterize Training Forward", 26)

    def _rasterize_backward(
        self,
        encoder: spy.CommandEncoder,
        camera: Camera,
        background: np.ndarray,
        output_grad: spy.Buffer,
        regularizer_grad: spy.Buffer | None = None,
        clone_counts_buffer: spy.Buffer | None = None,
        training_background_mode: int = 0,
        training_background_seed: int = 0,
        training_native_camera: Camera | None = None,
        training_sample_vars: dict[str, object] | None = None,
    ) -> None:
        resolved_regularizer_grad = self._work_buffers["training_regularizer_grad"] if regularizer_grad is None else regularizer_grad
        resolved_native_camera = camera if training_native_camera is None else training_native_camera
        resolved_sample_vars = self._disabled_training_sample_vars() if training_sample_vars is None else training_sample_vars
        if regularizer_grad is None:
            self._clear_float_buffer(encoder, resolved_regularizer_grad, max(self.width * self.height, 1))
        vars = {**self._scene_vars(), **self._raster_cache_vars(), "g_SortedValues": self._sorted_values(), "g_TileRanges": self._work_buffers["tile_ranges"], "g_OutputGrad": output_grad, "g_TrainingForwardState": self._work_buffers["training_forward_state"], "g_TrainingRegularizerGrad": resolved_regularizer_grad, "g_TrainingProcessedEnd": self._work_buffers["training_processed_end"], "g_TrainingBatchEnd": self._work_buffers["training_batch_end"], "g_CloneCounts": self._work_buffers["fallback_clone_counts"] if clone_counts_buffer is None else clone_counts_buffer, **self._raster_grad_vars(), **self._raster_grad_decode_scale_var(1.0), **self._raster_grad_fixed_range_vars(), **self._prepass_uniforms(self._scene_count), **self._raster_uniforms(background, training_background_mode, training_background_seed), **self._anisotropy_uniforms(), **self._camera_uniforms(camera), **self._camera_uniforms(resolved_native_camera, "g_TrainingNativeCamera"), **resolved_sample_vars}
        self._dispatch(self._raster_grad_shader_set().backward, encoder, self._raster_thread_count(), vars, "Rasterize Backward", 27)

    def _backprop_cached_raster_grads(self, encoder: spy.CommandEncoder, splat_count: int, camera: Camera, grad_scale: float = 1.0) -> None:
        self._dispatch(
            self._raster_grad_shader_set().backprop,
            encoder,
            spy.uint3(max(int(splat_count), 1), 1, 1),
            {
                **self._scene_vars(),
                **self._raster_cache_vars(),
                **self._raster_grad_vars(),
                **self._raster_grad_decode_scale_var(grad_scale),
                **self._raster_grad_fixed_range_vars(),
                **self._prepass_uniforms(splat_count),
                **self._raster_uniforms(np.zeros((3,), dtype=np.float32)),
                **self._anisotropy_uniforms(),
                **self._camera_uniforms(camera),
            },
            "Backprop Cached Raster Grads",
            28,
        )

    def _execute_prepass(self, scene: GaussianScene, camera: Camera, sync_counts: bool = False, sort_camera_position: np.ndarray | None = None, sort_camera_dither_sigma: float = 0.0, sort_camera_dither_seed: int = 0) -> tuple[int, int]:
        enc = self.device.create_command_encoder()
        with debug_region(enc, "Renderer Prepass", 19):
            self._record_prepass(enc, scene, camera, enqueue_counter_readback=True, sort_camera_position=sort_camera_position, sort_camera_dither_sigma=sort_camera_dither_sigma, sort_camera_dither_seed=sort_camera_dither_seed)
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

    def _record_prepass(self, encoder: spy.CommandEncoder, scene: GaussianScene, camera: Camera, enqueue_counter_readback: bool, sort_camera_position: np.ndarray | None = None, sort_camera_dither_sigma: float = 0.0, sort_camera_dither_seed: int = 0) -> None:
        self._reset_prepass_counters(encoder)
        self._project_visible_splats(encoder, scene, camera, sort_camera_position, sort_camera_dither_sigma, sort_camera_dither_seed)
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
        sh_coeffs: np.ndarray | None = None,
        color_alpha: np.ndarray | None = None,
    ) -> None:
        packed = self._pack_param_groups(
            splat_count,
            positions=positions,
            scales=scales,
            rotations=rotations,
            sh_coeffs=sh_coeffs,
            color_alpha=color_alpha,
        )
        self._scene_buffers["splat_params"].copy_from_numpy(packed)

    def read_grad_groups(self, splat_count: int | None = None) -> dict[str, np.ndarray]:
        count = self._scene_count if splat_count is None else int(splat_count)
        flat = self._read_array(self._work_buffers["param_grads"], np.float32, max(count, 1) * self.TRAINABLE_PARAM_COUNT)
        groups = self._unpack_param_groups(flat, count)
        grad_color_alpha = np.zeros((count, 4), dtype=np.float32)
        grad_color_alpha[:, :3] = groups["sh_coeffs"][:, 0, :] / SH_C0
        grad_color_alpha[:, 3] = groups["color_alpha"][:, 3]
        return {
            "grad_positions": groups["positions"],
            "grad_scales": groups["scales"],
            "grad_rotations": groups["rotations"],
            "grad_sh_coeffs": groups["sh_coeffs"],
            "grad_color_alpha": grad_color_alpha,
        }

    def read_raster_cache(self, splat_count: int | None = None) -> np.ndarray:
        count = self._scene_count if splat_count is None else int(splat_count)
        cache = self._read_array(self._work_buffers["raster_cache"], np.float32, max(count, 1), self._RASTER_GAUSSIAN_PARAM_COUNT)[:count].copy()
        if count <= 0:
            return cache
        visible = self._read_array(self._work_buffers["splat_visible"], np.uint32, count)
        cache[visible == 0] = 0.0
        return cache

    def read_cached_raster_grads_fixed(self, splat_count: int | None = None) -> np.ndarray:
        count = self._scene_count if splat_count is None else int(splat_count)
        return self._read_array(self._work_buffers["cached_raster_grads_fixed"], np.int32, self._RASTER_GRAD_PARAM_COUNT, max(count, 1)).T[:count].copy()

    def read_cached_raster_grads_fixed_decoded(self, splat_count: int | None = None) -> np.ndarray:
        values = np.asarray(self.read_cached_raster_grads_fixed(splat_count), dtype=np.float64)
        decode_scales = self.cached_raster_grad_fixed_decode_scale_table(values.shape[0])
        return values * decode_scales

    def read_cached_raster_grads_float(self, splat_count: int | None = None) -> np.ndarray:
        count = self._scene_count if splat_count is None else int(splat_count)
        return self._read_array(self._work_buffers["cached_raster_grads_float"], np.float32, self._RASTER_GRAD_PARAM_COUNT, max(count, 1)).T[:count].copy()

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
        return self._read_array(buffer, np.float32, self._RASTER_GRAD_PARAM_COUNT, max(count, 1)).T[:count].copy()

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
            self._RASTER_GRAD_PARAM_COUNT,
            count,
            bin_count=bin_count,
            min_log10=min_log10,
            max_log10=max_log10,
            param_labels=self.CACHED_RASTER_GRAD_COMPONENT_LABELS,
        )

    def compute_cached_raster_grad_component_ranges(self, metrics: Metrics, splat_count: int | None = None) -> ParamTensorRanges:
        count = self._scene_count if splat_count is None else int(splat_count)
        tensor = self.prepare_active_cached_raster_grads_float_tensor(count)
        return metrics.compute_param_tensor_ranges(tensor, self._RASTER_GRAD_PARAM_COUNT, count, param_labels=self.CACHED_RASTER_GRAD_COMPONENT_LABELS)

    def compute_sh_component_ranges(self, splat_count: int | None = None) -> ParamTensorRanges:
        count = self._scene_count if splat_count is None else int(splat_count)
        if count <= 0:
            return ParamTensorRanges(
                min_values=np.zeros((0,), dtype=np.float32),
                max_values=np.zeros((0,), dtype=np.float32),
                param_labels=(),
            )
        sh_coeffs = np.asarray(self.read_scene_groups(count)["sh_coeffs"], dtype=np.float32)
        mins = np.min(sh_coeffs, axis=0).reshape(-1).astype(np.float32, copy=False)
        maxs = np.max(sh_coeffs, axis=0).reshape(-1).astype(np.float32, copy=False)
        return ParamTensorRanges(min_values=mins.copy(), max_values=maxs.copy(), param_labels=self.SH_COMPONENT_RANGE_LABELS)

    def _scene_histogram_tensor(self, splat_count: int) -> np.ndarray:
        groups = self.read_scene_groups(splat_count)
        sh_coeffs = np.asarray(groups["sh_coeffs"], dtype=np.float32)
        base_color = sh_coeffs_to_display_colors(sh_coeffs)
        opacity = np.reciprocal(1.0 + np.exp(-np.asarray(groups["color_alpha"][:, 3], dtype=np.float32)))
        tensor = np.concatenate(
            (
                np.asarray(groups["positions"][:, :3], dtype=np.float32),
                np.exp(np.asarray(groups["scales"][:, :3], dtype=np.float32)),
                np.asarray(groups["rotations"][:, [3, 0, 1, 2]], dtype=np.float32),
                np.asarray(base_color, dtype=np.float32),
                np.asarray(sh_coeffs[:, 1:, :], dtype=np.float32).reshape(splat_count, -1),
                np.asarray(opacity[:, None], dtype=np.float32),
            ),
            axis=1,
        )
        return np.ascontiguousarray(tensor.T, dtype=np.float32)

    @staticmethod
    def _param_tensor_histograms(
        tensor: np.ndarray,
        *,
        bin_count: int,
        min_value: float,
        max_value: float,
        param_labels: tuple[str, ...],
        param_groups: tuple[tuple[str, tuple[int, ...]], ...],
    ) -> ParamLog10Histograms:
        bins = max(int(bin_count), 1)
        lo = float(min_value)
        hi = float(max_value) if float(max_value) > lo else lo + 1e-6
        counts = np.zeros((tensor.shape[0], bins), dtype=np.int64)
        inv_bin_size = float(bins) / (hi - lo)
        for param_index in range(tensor.shape[0]):
            values = np.asarray(tensor[param_index], dtype=np.float64)
            valid = np.isfinite(values)
            if not np.any(valid):
                continue
            bin_indices = np.clip(np.floor((values[valid] - lo) * inv_bin_size).astype(np.int64), 0, bins - 1)
            np.add.at(counts[param_index], bin_indices, 1)
        return ParamLog10Histograms(
            counts=counts,
            bin_edges_log10=np.linspace(lo, hi, bins + 1, dtype=np.float64),
            param_labels=param_labels,
            param_groups=param_groups,
        )

    @staticmethod
    def _param_tensor_ranges(
        tensor: np.ndarray,
        *,
        param_labels: tuple[str, ...],
        param_groups: tuple[tuple[str, tuple[int, ...]], ...],
    ) -> ParamTensorRanges:
        min_values = np.full((tensor.shape[0],), np.nan, dtype=np.float32)
        max_values = np.full((tensor.shape[0],), np.nan, dtype=np.float32)
        for param_index in range(tensor.shape[0]):
            values = np.asarray(tensor[param_index], dtype=np.float32)
            valid = values[np.isfinite(values)]
            if valid.size == 0:
                continue
            min_values[param_index] = np.min(valid)
            max_values[param_index] = np.max(valid)
        return ParamTensorRanges(
            min_values=min_values,
            max_values=max_values,
            param_labels=param_labels,
            param_groups=param_groups,
        )

    def compute_scene_param_histograms(
        self,
        splat_count: int | None = None,
        *,
        bin_count: int = 64,
        min_value: float = -1.0,
        max_value: float = 1.0,
    ) -> ParamLog10Histograms:
        count = self._scene_count if splat_count is None else int(splat_count)
        if count <= 0:
            return ParamLog10Histograms(
                counts=np.zeros((0, max(int(bin_count), 1)), dtype=np.int64),
                bin_edges_log10=np.linspace(float(min_value), float(max(max_value, min_value + 1e-6)), max(int(bin_count), 1) + 1, dtype=np.float64),
                param_labels=(),
                param_groups=(),
            )
        return self._param_tensor_histograms(
            self._scene_histogram_tensor(count),
            bin_count=bin_count,
            min_value=min_value,
            max_value=max_value,
            param_labels=self.SCENE_PARAM_HISTOGRAM_LABELS,
            param_groups=self.SCENE_PARAM_HISTOGRAM_GROUPS,
        )

    def compute_scene_param_ranges(self, splat_count: int | None = None) -> ParamTensorRanges:
        count = self._scene_count if splat_count is None else int(splat_count)
        if count <= 0:
            return ParamTensorRanges(
                min_values=np.zeros((0,), dtype=np.float32),
                max_values=np.zeros((0,), dtype=np.float32),
                param_labels=(),
                param_groups=(),
            )
        return self._param_tensor_ranges(
            self._scene_histogram_tensor(count),
            param_labels=self.SCENE_PARAM_HISTOGRAM_LABELS,
            param_groups=self.SCENE_PARAM_HISTOGRAM_GROUPS,
        )

    def write_grad_groups(
        self,
        splat_count: int,
        *,
        grad_positions: np.ndarray | None = None,
        grad_scales: np.ndarray | None = None,
        grad_rotations: np.ndarray | None = None,
        grad_sh_coeffs: np.ndarray | None = None,
        grad_color_alpha: np.ndarray | None = None,
    ) -> None:
        packed = self._pack_param_groups(
            splat_count,
            positions=grad_positions,
            scales=grad_scales,
            rotations=grad_rotations,
            sh_coeffs=grad_sh_coeffs,
            color_alpha=grad_color_alpha,
            color_is_grad=True,
        )
        self._work_buffers["param_grads"].copy_from_numpy(packed)

    def _require_scene(self) -> GaussianScene | SceneBinding:
        if self._current_scene is None:
            raise RuntimeError("Scene is not set.")
        return self._current_scene

    @property
    def sh_band(self) -> int:
        return self._sh_band

    @sh_band.setter
    def sh_band(self, value: int) -> None:
        self._sh_band = min(max(int(value), 0), 3)

    @property
    def use_sh(self) -> bool:
        return self.sh_band > 0

    @use_sh.setter
    def use_sh(self, value: bool) -> None:
        self.sh_band = 3 if bool(value) else 0

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
                self.cached_raster_grad_fixed_color_range / self._RASTER_GRAD_FIXED_INT_MAX,
                self.cached_raster_grad_fixed_color_range / self._RASTER_GRAD_FIXED_INT_MAX,
                self.cached_raster_grad_fixed_color_range / self._RASTER_GRAD_FIXED_INT_MAX,
                self.cached_raster_grad_fixed_opacity_range / self._RASTER_GRAD_FIXED_INT_MAX,
            ],
            dtype=np.float32,
        )

    def cached_raster_grad_fixed_decode_scale_table(self, splat_count: int | None = None, raster_cache: np.ndarray | None = None) -> np.ndarray:
        count = self._scene_count if splat_count is None else int(splat_count)
        decode_scales = np.broadcast_to(self.cached_raster_grad_fixed_decode_scales.reshape(1, self._RASTER_GRAD_PARAM_COUNT), (max(count, 0), self._RASTER_GRAD_PARAM_COUNT)).astype(np.float64, copy=True)
        if count <= 0:
            return decode_scales
        cache = self.read_raster_cache(count) if raster_cache is None else np.asarray(raster_cache, dtype=np.float64).reshape(count, self._RASTER_GAUSSIAN_PARAM_COUNT)
        sigma_ortho = np.asarray(cache[:, 3:6], dtype=np.float64)
        sigma_det = np.maximum(sigma_ortho[:, 0] * sigma_ortho[:, 2] - sigma_ortho[:, 1] * sigma_ortho[:, 1], 1e-24)
        avg_scale = np.sqrt(np.sqrt(sigma_det))
        avg_inv_scale = np.maximum(1.0 / np.maximum(avg_scale, 1e-12), 0.25)
        sigma_inv_scale = avg_inv_scale * avg_inv_scale
        decode_scales[:, 0:3] *= avg_inv_scale[:, None]
        decode_scales[:, 3:6] *= sigma_inv_scale[:, None]
        return decode_scales

    def set_debug_grad_norm_buffer(self, buffer: spy.Buffer | None) -> None:
        self._debug_grad_norm_buffer = buffer

    def set_debug_splat_age_buffer(self, buffer: spy.Buffer | None) -> None:
        self._debug_splat_age_buffer = buffer

    def set_debug_splat_contribution_buffer(self, buffer: spy.Buffer | None) -> None:
        self._debug_splat_contribution_buffer = buffer

    def set_debug_adam_moments_buffer(self, buffer: spy.Buffer | None) -> None:
        self._debug_adam_moments_buffer = buffer

    def set_debug_contribution_observed_pixel_count(self, observed_pixel_count: float) -> None:
        pixels = max(float(observed_pixel_count), 1.0)
        self._debug_contribution_scale = 1.0 / (self._SPLAT_CONTRIBUTION_FIXED_SCALE * pixels)

    def upload_debug_splat_age(self, values: np.ndarray) -> None:
        splat_age = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
        self._ensure_work_buffers(max(int(splat_age.shape[0]), self._scene_count, 1))
        self._work_buffers["debug_splat_age"].copy_from_numpy(np.pad(splat_age, (0, max(self._work_splat_capacity - splat_age.shape[0], 0)), constant_values=1.0))
        self._debug_splat_age_buffer = None

    def upload_debug_splat_contribution(self, values: np.ndarray) -> None:
        contribution = np.ascontiguousarray(values, dtype=np.uint32).reshape(-1)
        self._ensure_work_buffers(max(int(contribution.shape[0]), self._scene_count, 1))
        self._work_buffers["training_splat_contribution"].copy_from_numpy(np.pad(contribution, (0, max(self._work_splat_capacity - contribution.shape[0], 0))))
        self._debug_splat_contribution_buffer = None

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

    def execute_prepass_for_current_scene(self, camera: Camera, sync_counts: bool = False, sort_camera_position: np.ndarray | None = None, sort_camera_dither_sigma: float = 0.0, sort_camera_dither_seed: int = 0) -> tuple[int, int]:
        return self._execute_prepass(self._require_scene(), camera, sync_counts=sync_counts, sort_camera_position=sort_camera_position, sort_camera_dither_sigma=sort_camera_dither_sigma, sort_camera_dither_seed=sort_camera_dither_seed)

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

    def record_prepass_for_current_scene(self, encoder: spy.CommandEncoder, camera: Camera, sort_camera_position: np.ndarray | None = None, sort_camera_dither_sigma: float = 0.0, sort_camera_dither_seed: int = 0) -> None:
        scene = self._require_scene()
        self._record_prepass(encoder, scene, camera, enqueue_counter_readback=False, sort_camera_position=sort_camera_position, sort_camera_dither_sigma=sort_camera_dither_sigma, sort_camera_dither_seed=sort_camera_dither_seed)

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
        splat_contribution_buffer: spy.Buffer | None = None,
        training_background_mode: int = 0,
        training_background_seed: int = 0,
        training_native_camera: Camera | None = None,
        training_sample_vars: dict[str, object] | None = None,
    ) -> None:
        self._require_scene()
        self._rasterize_training_forward(encoder, camera, background, output, clone_counts_buffer, splat_contribution_buffer, training_background_mode, training_background_seed, training_native_camera, training_sample_vars)

    def rasterize_backward_current_scene(
        self,
        encoder: spy.CommandEncoder,
        camera: Camera,
        background: np.ndarray,
        output_grad: spy.Buffer,
        grad_scale: float = 1.0,
        regularizer_grad: spy.Buffer | None = None,
        clone_counts_buffer: spy.Buffer | None = None,
        training_background_mode: int = 0,
        training_background_seed: int = 0,
        training_native_camera: Camera | None = None,
        training_sample_vars: dict[str, object] | None = None,
    ) -> None:
        self._require_scene()
        self._rasterize_backward(encoder, camera, background, output_grad, regularizer_grad, clone_counts_buffer, training_background_mode, training_background_seed, training_native_camera, training_sample_vars)
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

    def render_training_forward_to_texture(
        self,
        camera: Camera,
        background: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
        read_stats: bool = True,
        command_encoder: spy.CommandEncoder | None = None,
        sort_camera_position: np.ndarray | None = None,
        sort_camera_dither_sigma: float = 0.0,
        sort_camera_dither_seed: int = 0,
        training_background_mode: int = 0,
        training_background_seed: int = 0,
        training_native_camera: Camera | None = None,
        training_sample_vars: dict[str, object] | None = None,
    ) -> tuple[spy.Texture, dict[str, int | bool | float]]:
        scene = self._require_scene()
        if scene.count <= 0:
            raise RuntimeError("Cannot render empty scene.")
        self._ensure_work_buffers(scene.count, self._pending_min_list_entries)
        background_np = self._background_array(background)
        if command_encoder is None:
            enc = self.device.create_command_encoder()
            with debug_region(enc, "Renderer Training Preview Prepass", 19):
                self._record_prepass(enc, scene, camera, enqueue_counter_readback=True, sort_camera_position=sort_camera_position, sort_camera_dither_sigma=sort_camera_dither_sigma, sort_camera_dither_seed=sort_camera_dither_seed)
            self._rasterize_training_forward(enc, camera, background_np, training_background_mode=training_background_mode, training_background_seed=training_background_seed, training_native_camera=training_native_camera, training_sample_vars=training_sample_vars)
            self.device.submit_command_buffer(enc.finish())
            self._counter_readback_frame_id += 1
            self.device.wait()
        else:
            with debug_region(command_encoder, "Renderer Training Preview Prepass", 19):
                self._record_prepass(command_encoder, scene, camera, enqueue_counter_readback=True, sort_camera_position=sort_camera_position, sort_camera_dither_sigma=sort_camera_dither_sigma, sort_camera_dither_seed=sort_camera_dither_seed)
            self._counter_readback_frame_id += 1
            self._rasterize_training_forward(command_encoder, camera, background_np, training_background_mode=training_background_mode, training_background_seed=training_background_seed, training_native_camera=training_native_camera, training_sample_vars=training_sample_vars)
        if read_stats:
            self._update_delayed_counter_stats()
        self._last_stats = self._stats_payload(scene.count, read_stats)
        return self.output_texture, self._last_stats

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

    def debug_pipeline_data(self, scene: GaussianScene, camera: Camera, sort_camera_position: np.ndarray | None = None, sort_camera_dither_sigma: float = 0.0, sort_camera_dither_seed: int = 0) -> dict[str, np.ndarray | int]:
        self._ensure_scene_buffers(scene.count)
        self._ensure_work_buffers(scene.count)
        self._upload_scene(scene)
        generated_entries, sorted_count = self._execute_prepass(scene, camera, sync_counts=True, sort_camera_position=sort_camera_position, sort_camera_dither_sigma=sort_camera_dither_sigma, sort_camera_dither_seed=sort_camera_dither_seed)
        return {
            "generated_entries": generated_entries,
            "sorted_count": sorted_count,
            "keys": self._read_array(self._sorted_keys(), np.uint32, sorted_count),
            "values": self._read_array(self._sorted_values(), np.uint32, sorted_count),
            "tile_ranges": self._read_array(self._work_buffers["tile_ranges"], np.uint32, self.tile_count, 2),
            "screen_center_radius_depth": self._read_array(self._work_buffers["screen_center_radius_depth"], np.float32, scene.count, 4),
            "screen_color_alpha": self._read_array(self._work_buffers["screen_color_alpha"], np.float32, scene.count, 4),
            "screen_ellipse_conic": self._read_array(self._work_buffers["screen_ellipse_conic"], np.float32, scene.count, 4),
            "splat_visible_area_px": self._read_array(self._work_buffers["splat_visible_area_px"], np.float32, scene.count),
            "splat_visible": self._read_array(self._work_buffers["splat_visible"], np.uint32, scene.count),
            "raster_cache": self.read_raster_cache(scene.count),
        }
