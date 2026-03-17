from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import slangpy as spy

from ..common import SHADER_INCLUDE_PATHS, SHADER_ROOT, debug_region, thread_count_2d
from .camera import Camera
from .gaussian_renderer import GaussianRenderer

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only on non-torch installs
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


_TORCH_CAMERA_PARAM_COUNT = 15
_TORCH_RENDER_PARAM_COUNT = GaussianRenderer.TRAINABLE_PARAM_COUNT
_TORCH_OUTPUT_CHANNELS = 4
_TORCH_SCENE_INPUT_BUFFER_NAME = "scene_input"
_TORCH_IMAGE_OUTPUT_BUFFER_NAME = "image_output"
_TORCH_OUTPUT_GRAD_BUFFER_NAME = "output_grad"
_TORCH_PARAM_GRAD_BUFFER_NAME = "param_grad"
_TORCH_BRIDGE_USAGE = (
    spy.BufferUsage.shader_resource
    | spy.BufferUsage.unordered_access
    | spy.BufferUsage.copy_source
    | spy.BufferUsage.copy_destination
    | spy.BufferUsage.shared
)


def _require_torch():
    if torch is None:  # pragma: no cover - exercised only on non-torch installs
        raise RuntimeError(
            "PyTorch is required for the torch renderer interface. Install a CUDA-enabled PyTorch build first."
        ) from _TORCH_IMPORT_ERROR
    return torch


@dataclass(frozen=True, slots=True)
class TorchGaussianRenderSettings:
    width: int
    height: int
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius_scale: float = 1.0
    alpha_cutoff: float = 1.0 / 255.0
    max_splat_steps: int = 32768
    transmittance_threshold: float = 0.005
    list_capacity_multiplier: int = 64
    max_prepass_memory_mb: int = 4096
    sampled5_mvee_iters: int = 6
    sampled5_safety_scale: float = 1.0
    sampled5_radius_pad_px: float = 1.0
    sampled5_eps: float = 1e-6
    cached_raster_grad_atomic_mode: str = GaussianRenderer.CACHED_RASTER_GRAD_ATOMIC_MODE_FIXED
    cached_raster_grad_fixed_scale: float = 0.125
    cached_raster_grad_fixed_loffdiag_ref_scale: float = 1.0
    cached_raster_grad_fixed_ro_local_ref_scale: float = 100.0
    cached_raster_grad_fixed_l_ref_scale: float = 100.0
    cached_raster_grad_fixed_color_range: float = 200.0
    cached_raster_grad_fixed_opacity_range: float = 200.0
    cached_raster_grad_fixed_l_distance_norm_power: float = 0.0

    def __post_init__(self) -> None:
        width = max(int(self.width), 1)
        height = max(int(self.height), 1)
        background = tuple(float(x) for x in self.background)
        if len(background) != 3:
            raise ValueError("background must contain exactly 3 floats.")
        GaussianRenderer._validate_cached_raster_grad_atomic_mode(self.cached_raster_grad_atomic_mode)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "background", background)

    def background_array(self) -> np.ndarray:
        return np.asarray(self.background, dtype=np.float32)

    def renderer_kwargs(self) -> dict[str, object]:
        return {
            "radius_scale": float(self.radius_scale),
            "alpha_cutoff": float(self.alpha_cutoff),
            "max_splat_steps": int(self.max_splat_steps),
            "transmittance_threshold": float(self.transmittance_threshold),
            "list_capacity_multiplier": int(self.list_capacity_multiplier),
            "max_prepass_memory_mb": int(self.max_prepass_memory_mb),
            "sampled5_mvee_iters": int(self.sampled5_mvee_iters),
            "sampled5_safety_scale": float(self.sampled5_safety_scale),
            "sampled5_radius_pad_px": float(self.sampled5_radius_pad_px),
            "sampled5_eps": float(self.sampled5_eps),
            "cached_raster_grad_atomic_mode": str(self.cached_raster_grad_atomic_mode),
            "cached_raster_grad_fixed_scale": float(self.cached_raster_grad_fixed_scale),
            "cached_raster_grad_fixed_loffdiag_ref_scale": float(self.cached_raster_grad_fixed_loffdiag_ref_scale),
            "cached_raster_grad_fixed_ro_local_ref_scale": float(self.cached_raster_grad_fixed_ro_local_ref_scale),
            "cached_raster_grad_fixed_l_ref_scale": float(self.cached_raster_grad_fixed_l_ref_scale),
            "cached_raster_grad_fixed_color_range": float(self.cached_raster_grad_fixed_color_range),
            "cached_raster_grad_fixed_opacity_range": float(self.cached_raster_grad_fixed_opacity_range),
            "cached_raster_grad_fixed_l_distance_norm_power": float(self.cached_raster_grad_fixed_l_distance_norm_power),
        }

    def renderer_key(self) -> tuple[object, ...]:
        kwargs = self.renderer_kwargs()
        return (int(self.width), int(self.height), *(kwargs[name] for name in sorted(kwargs)))


class TorchGaussianRendererContext:
    _output_pack_shader_path = Path(SHADER_ROOT / "renderer" / "gaussian_torch_stage.slang")

    def __init__(
        self,
        *,
        device: spy.Device | None = None,
        torch_device: Any | None = None,
        enable_debug_layers: bool = False,
        enable_print: bool = False,
        enable_hot_reload: bool = False,
        enable_compilation_reports: bool = False,
    ) -> None:
        torch_mod = _require_torch()
        self._torch_device = self._resolve_torch_device(torch_device, torch_mod)
        self._device = (
            self._create_device(
                self._torch_device,
                enable_debug_layers=enable_debug_layers,
                enable_print=enable_print,
                enable_hot_reload=enable_hot_reload,
                enable_compilation_reports=enable_compilation_reports,
            )
            if device is None
            else device
        )
        self._renderer: GaussianRenderer | None = None
        self._renderer_key: tuple[object, ...] | None = None
        self._bridge_buffers: dict[str, spy.Buffer] = {}
        self._splat_capacity = 0
        self._pixel_capacity = 0
        self._pack_output_kernel = self._create_pack_output_kernel()

    @staticmethod
    def _resolve_torch_device(torch_device: Any | None, torch_mod) -> Any:
        if not torch_mod.cuda.is_available():
            raise RuntimeError("CUDA PyTorch is required for render_gaussian_splats_torch.")
        if torch_device is None:
            index = torch_mod.cuda.current_device()
            return torch_mod.device(f"cuda:{index}")
        resolved = torch_mod.device(torch_device)
        if resolved.type != "cuda":
            raise ValueError(f"torch_device must be CUDA, got {resolved}.")
        index = torch_mod.cuda.current_device() if resolved.index is None else int(resolved.index)
        return torch_mod.device(f"cuda:{index}")

    @staticmethod
    def _create_device(
        torch_device: Any,
        *,
        enable_debug_layers: bool,
        enable_print: bool,
        enable_hot_reload: bool,
        enable_compilation_reports: bool,
    ) -> spy.Device:
        torch_mod = _require_torch()
        torch_mod.cuda.init()
        torch_mod.cuda.current_device()
        torch_mod.cuda.current_stream()
        with torch_mod.device(torch_device):
            handles = spy.get_cuda_current_context_native_handles()
        return spy.Device(
            type=spy.DeviceType.cuda,
            compiler_options={"include_paths": [str(path) for path in SHADER_INCLUDE_PATHS]},
            enable_debug_layers=bool(enable_debug_layers),
            enable_rhi_validation=False,
            enable_cuda_interop=False,
            enable_print=bool(enable_print),
            enable_hot_reload=bool(enable_hot_reload),
            enable_compilation_reports=bool(enable_compilation_reports),
            existing_device_handles=handles,
        )

    def _create_pack_output_kernel(self) -> spy.ComputeKernel:
        program = self._device.load_program(str(self._output_pack_shader_path), ["csPackOutputTexture"])
        return self._device.create_compute_kernel(program)

    @staticmethod
    def _grow(required: int, current: int) -> int:
        base = max(int(current), 1)
        return max(int(required), base + base // 2)

    def _create_bridge_buffer(self, size: int) -> spy.Buffer:
        return self._device.create_buffer(size=max(int(size), 1), usage=_TORCH_BRIDGE_USAGE)

    def _ensure_bridge_buffers(self, splat_count: int, pixel_count: int) -> None:
        required_splats = max(int(splat_count), 1)
        required_pixels = max(int(pixel_count), 1)
        if required_splats > self._splat_capacity:
            self._splat_capacity = self._grow(required_splats, self._splat_capacity)
            byte_count = self._splat_capacity * _TORCH_RENDER_PARAM_COUNT * np.dtype(np.float32).itemsize
            self._bridge_buffers[_TORCH_SCENE_INPUT_BUFFER_NAME] = self._create_bridge_buffer(byte_count)
            self._bridge_buffers[_TORCH_PARAM_GRAD_BUFFER_NAME] = self._create_bridge_buffer(byte_count)
        if required_pixels > self._pixel_capacity:
            self._pixel_capacity = self._grow(required_pixels, self._pixel_capacity)
            byte_count = self._pixel_capacity * _TORCH_OUTPUT_CHANNELS * np.dtype(np.float32).itemsize
            self._bridge_buffers[_TORCH_IMAGE_OUTPUT_BUFFER_NAME] = self._create_bridge_buffer(byte_count)
            self._bridge_buffers[_TORCH_OUTPUT_GRAD_BUFFER_NAME] = self._create_bridge_buffer(byte_count)

    def _ensure_renderer(self, settings: TorchGaussianRenderSettings) -> GaussianRenderer:
        key = settings.renderer_key()
        if self._renderer is not None and self._renderer_key == key:
            return self._renderer
        self._renderer = GaussianRenderer(self._device, width=settings.width, height=settings.height, **settings.renderer_kwargs())
        self._renderer_key = key
        return self._renderer

    @property
    def device(self) -> spy.Device:
        return self._device

    @property
    def renderer(self) -> GaussianRenderer:
        if self._renderer is None:
            raise RuntimeError("Renderer is not initialized yet.")
        return self._renderer

    @property
    def torch_device(self) -> Any:
        return self._torch_device

    def _copy_scene_input(self, packed_scene: Any, renderer: GaussianRenderer, splat_count: int, encoder: spy.CommandEncoder) -> None:
        scene_input = self._bridge_buffers[_TORCH_SCENE_INPUT_BUFFER_NAME]
        spy.copy_torch_tensor_to_buffer(packed_scene, scene_input)
        self._device.sync_to_cuda()
        byte_count = max(int(splat_count), 1) * _TORCH_RENDER_PARAM_COUNT * np.dtype(np.float32).itemsize
        encoder.copy_buffer(renderer.scene_buffers["splat_params"], 0, scene_input, 0, byte_count)

    def _dispatch_pack_output(self, renderer: GaussianRenderer, encoder: spy.CommandEncoder) -> None:
        with debug_region(encoder, "Torch Pack Output", 60):
            self._pack_output_kernel.dispatch(
                thread_count=thread_count_2d(renderer.width, renderer.height),
                vars={
                    "g_TorchInput": renderer.output_texture,
                    "g_TorchOutput": self._bridge_buffers[_TORCH_IMAGE_OUTPUT_BUFFER_NAME],
                    "g_TorchWidth": int(renderer.width),
                    "g_TorchHeight": int(renderer.height),
                },
                command_encoder=encoder,
            )

    def _read_output_tensor(self, settings: TorchGaussianRenderSettings):
        torch_mod = _require_torch()
        output = torch_mod.empty(
            (settings.height, settings.width, _TORCH_OUTPUT_CHANNELS),
            device=self._torch_device,
            dtype=torch_mod.float32,
        )
        spy.copy_buffer_to_torch_tensor(self._bridge_buffers[_TORCH_IMAGE_OUTPUT_BUFFER_NAME], output)
        return output

    def _copy_output_grad(self, grad_output: Any) -> None:
        spy.copy_torch_tensor_to_buffer(grad_output.contiguous(), self._bridge_buffers[_TORCH_OUTPUT_GRAD_BUFFER_NAME])
        self._device.sync_to_cuda()

    def _read_param_grads(self, splat_count: int):
        torch_mod = _require_torch()
        grads = torch_mod.empty(
            (_TORCH_RENDER_PARAM_COUNT, max(int(splat_count), 1)),
            device=self._torch_device,
            dtype=torch_mod.float32,
        )
        spy.copy_buffer_to_torch_tensor(self._bridge_buffers[_TORCH_PARAM_GRAD_BUFFER_NAME], grads)
        return grads[:, : max(int(splat_count), 0)]

    def render_forward(self, packed_scene: Any, camera: Camera, settings: TorchGaussianRenderSettings):
        splat_count = int(packed_scene.numel() // _TORCH_RENDER_PARAM_COUNT)
        pixel_count = int(settings.width * settings.height)
        renderer = self._ensure_renderer(settings)
        renderer.bind_scene_count(splat_count)
        self._ensure_bridge_buffers(splat_count, pixel_count)
        enc = self._device.create_command_encoder()
        self._copy_scene_input(packed_scene, renderer, splat_count, enc)
        renderer.record_prepass_for_current_scene(enc, camera)
        renderer.rasterize_training_forward_current_scene(enc, camera, settings.background_array())
        self._dispatch_pack_output(renderer, enc)
        self._device.submit_command_buffer(enc.finish())
        self._device.sync_to_device()
        return self._read_output_tensor(settings)

    def render_backward(
        self,
        grad_output: Any,
        camera: Camera,
        settings: TorchGaussianRenderSettings,
        splat_count: int,
    ):
        renderer = self.renderer
        self._ensure_bridge_buffers(splat_count, int(settings.width * settings.height))
        self._copy_output_grad(grad_output)
        enc = self._device.create_command_encoder()
        renderer.clear_raster_grads_current_scene(enc)
        renderer.rasterize_backward_current_scene(
            enc,
            camera,
            settings.background_array(),
            self._bridge_buffers[_TORCH_OUTPUT_GRAD_BUFFER_NAME],
            grad_scale=1.0,
        )
        byte_count = max(int(splat_count), 1) * _TORCH_RENDER_PARAM_COUNT * np.dtype(np.float32).itemsize
        enc.copy_buffer(
            self._bridge_buffers[_TORCH_PARAM_GRAD_BUFFER_NAME],
            0,
            renderer.work_buffers["param_grads"],
            0,
            byte_count,
        )
        self._device.submit_command_buffer(enc.finish())
        self._device.sync_to_device()
        return self._read_param_grads(splat_count)


def _validate_tensor(name: str, value: Any, *, shape: tuple[int, ...] | None = None) -> None:
    torch_mod = _require_torch()
    if not isinstance(value, torch_mod.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.dtype != torch_mod.float32:
        raise TypeError(f"{name} must use torch.float32, got {value.dtype}.")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor.")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}.")


def _validate_tensor_device(name: str, value: Any, torch_device: Any) -> None:
    if int(value.device.index) != int(torch_device.index):
        raise ValueError(f"{name} must be on {torch_device}, got {value.device}.")


def _pack_public_splats(splats: Any):
    torch_mod = _require_torch()
    alpha = torch_mod.clamp(splats[:, 13], GaussianRenderer._OPACITY_EPS, 1.0 - GaussianRenderer._OPACITY_EPS)
    groups = (
        splats[:, 0:3].transpose(0, 1),
        splats[:, 3:6].transpose(0, 1),
        splats[:, 6:10].transpose(0, 1),
        splats[:, 10:13].transpose(0, 1),
        torch_mod.logit(alpha).unsqueeze(0),
    )
    return torch_mod.cat(groups, dim=0).reshape(-1).contiguous(), alpha


def _unpack_public_grads(packed_grads: Any, alpha: Any):
    torch_mod = _require_torch()
    param_major = packed_grads.reshape(_TORCH_RENDER_PARAM_COUNT, -1)
    public = param_major.transpose(0, 1).contiguous()
    public[:, 13] = public[:, 13] * torch_mod.reciprocal(alpha * (1.0 - alpha))
    return public


def _camera_from_tensor(camera_params: Any) -> Camera:
    values = np.asarray(camera_params.detach().cpu(), dtype=np.float32).reshape(_TORCH_CAMERA_PARAM_COUNT)
    return Camera.from_colmap(
        q_wxyz=values[0:4],
        t_xyz=values[4:7],
        fx=float(values[7]),
        fy=float(values[8]),
        cx=float(values[9]),
        cy=float(values[10]),
        near=float(values[11]),
        far=float(values[12]),
    )


if torch is not None:

    class _TorchGaussianRenderFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, splats: Any, camera_params: Any, settings: TorchGaussianRenderSettings, context: TorchGaussianRendererContext):
            _validate_tensor("splats", splats)
            _validate_tensor("camera_params", camera_params, shape=(_TORCH_CAMERA_PARAM_COUNT,))
            _validate_tensor_device("splats", splats, context.torch_device)
            _validate_tensor_device("camera_params", camera_params, context.torch_device)
            if camera_params.requires_grad:
                raise ValueError("camera_params gradients are not supported in v1.")
            if splats.ndim != 2 or splats.shape[1] != _TORCH_RENDER_PARAM_COUNT:
                raise ValueError(f"splats must have shape [N, {_TORCH_RENDER_PARAM_COUNT}], got {tuple(splats.shape)}.")
            if int(splats.shape[0]) <= 0:
                raise ValueError("splats must contain at least one gaussian.")
            packed_scene, alpha = _pack_public_splats(splats.contiguous())
            camera = _camera_from_tensor(camera_params)
            renderer = context._ensure_renderer(settings)
            renderer.proj_distortion_k1 = float(camera_params[13].detach().cpu())
            renderer.proj_distortion_k2 = float(camera_params[14].detach().cpu())
            output = context.render_forward(packed_scene, camera, settings).clone()
            ctx.context = context
            ctx.camera = camera
            ctx.settings = settings
            ctx.splat_count = int(splats.shape[0])
            ctx.save_for_backward(alpha)
            return output

        @staticmethod
        def backward(ctx: Any, grad_output: Any):
            (alpha,) = ctx.saved_tensors
            packed_grads = ctx.context.render_backward(
                grad_output.contiguous(),
                ctx.camera,
                ctx.settings,
                ctx.splat_count,
            )
            return _unpack_public_grads(packed_grads, alpha), None, None, None

else:

    class _TorchGaussianRenderFunction:  # pragma: no cover - exercised only on non-torch installs
        @staticmethod
        def apply(*args: Any, **kwargs: Any) -> Any:
            _require_torch()
            raise AssertionError("unreachable")


def render_gaussian_splats_torch(
    splats: Any,
    camera_params: Any,
    settings: TorchGaussianRenderSettings,
    context: TorchGaussianRendererContext | None = None,
):
    torch_mod = _require_torch()
    if context is None:
        if not isinstance(splats, torch_mod.Tensor) or not splats.is_cuda:
            raise ValueError("splats must be a CUDA torch.Tensor when no context is provided.")
        context = TorchGaussianRendererContext(torch_device=splats.device)
    return _TorchGaussianRenderFunction.apply(splats, camera_params, settings, context)


__all__ = [
    "TorchGaussianRenderSettings",
    "TorchGaussianRendererContext",
    "render_gaussian_splats_torch",
]
