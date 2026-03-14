from __future__ import annotations

from pathlib import Path

import numpy as np
import slangpy as spy

from ..common import SHADER_ROOT, debug_region, thread_count_2d


class SeparableGaussianBlur:
    _BUFFER_USAGE = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
    _TEXTURE_USAGE = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
    _MAX_TEXTURE_RADIUS = 24
    _MAX_TEXTURE_TAP_COUNT = _MAX_TEXTURE_RADIUS * 2 + 1
    _KERNEL_ENTRIES = {
        "horizontal": "csGaussianBlurHorizontal",
        "vertical": "csGaussianBlurVertical",
        "horizontal_texture": "csGaussianBlurTextureHorizontal",
        "vertical_texture": "csGaussianBlurTextureVertical",
    }

    def _dispatch(self, kernel: str, encoder: spy.CommandEncoder, channel_count: int, vars: dict[str, object]) -> None:
        with debug_region(encoder, f"Blur::{kernel}", 80 + len(kernel)):
            self._kernels[kernel].dispatch(thread_count=thread_count_2d(self.width, self.height, channel_count), vars=vars, command_encoder=encoder)

    def __init__(self, device: spy.Device, width: int, height: int) -> None:
        self.device, self.width, self.height = device, int(width), int(height)
        self._shader_path = Path(SHADER_ROOT / "utility" / "blur" / "separable_gaussian_blur.slang")
        self._kernels = {name: self.device.create_compute_kernel(self.device.load_program(str(self._shader_path), [entry])) for name, entry in self._KERNEL_ENTRIES.items()}
        self._scratch_buffers: dict[int, spy.Buffer] = {}
        self._scratch_textures: dict[spy.Format, spy.Texture] = {}
        self._weight_buffers: dict[tuple[int, int], spy.Buffer] = {}

    def make_buffer(self, channel_count: int) -> spy.Buffer:
        return self.device.create_buffer(size=self._buffer_size(channel_count), usage=self._BUFFER_USAGE)

    def make_texture(self, fmt: spy.Format = spy.Format.rgba32_float) -> spy.Texture:
        return self.device.create_texture(format=fmt, width=self.width, height=self.height, usage=self._TEXTURE_USAGE)

    def _buffer_size(self, channel_count: int) -> int:
        channels = int(channel_count)
        if channels <= 0:
            raise ValueError("Blur channel_count must be positive.")
        return self.width * self.height * channels * np.dtype(np.float32).itemsize

    def _ensure_scratch_buffer(self, channel_count: int) -> spy.Buffer:
        channels = int(channel_count)
        if channels not in self._scratch_buffers:
            self._scratch_buffers[channels] = self.make_buffer(channels)
        return self._scratch_buffers[channels]

    def _ensure_scratch_texture(self, fmt: spy.Format) -> spy.Texture:
        if fmt not in self._scratch_textures:
            self._scratch_textures[fmt] = self.make_texture(fmt)
        return self._scratch_textures[fmt]

    def _gaussian_weights(self, radius: int, sigma: float) -> np.ndarray:
        resolved_radius = max(0, min(int(radius), self._MAX_TEXTURE_RADIUS))
        resolved_sigma = max(float(sigma), 1e-6)
        weights = np.zeros((self._MAX_TEXTURE_TAP_COUNT,), dtype=np.float32)
        if resolved_radius <= 0:
            weights[self._MAX_TEXTURE_RADIUS] = 1.0
            return weights
        offsets = np.arange(-resolved_radius, resolved_radius + 1, dtype=np.float32)
        taps = np.exp(-0.5 * np.square(offsets / resolved_sigma)).astype(np.float32)
        taps /= np.sum(taps, dtype=np.float32)
        start = self._MAX_TEXTURE_RADIUS - resolved_radius
        weights[start : start + taps.shape[0]] = taps
        return weights

    def _weight_buffer(self, radius: int, sigma: float) -> tuple[spy.Buffer, int]:
        resolved_radius = max(0, min(int(radius), self._MAX_TEXTURE_RADIUS))
        sigma_key = int(round(max(float(sigma), 1e-6) * 1024.0))
        key = (resolved_radius, sigma_key)
        if key not in self._weight_buffers:
            weights = self._gaussian_weights(resolved_radius, float(sigma_key) / 1024.0)
            buffer = self.device.create_buffer(size=weights.size * np.dtype(np.float32).itemsize, usage=self._BUFFER_USAGE)
            buffer.copy_from_numpy(weights)
            self._weight_buffers[key] = buffer
        return self._weight_buffers[key], resolved_radius

    def blur(self, encoder: spy.CommandEncoder, input_buffer: spy.Buffer, output_buffer: spy.Buffer, channel_count: int) -> spy.Buffer:
        shared = {"g_BlurWidth": self.width, "g_BlurHeight": self.height, "g_BlurChannelCount": int(channel_count)}
        scratch = self._ensure_scratch_buffer(channel_count)
        self._dispatch("horizontal", encoder, channel_count, {"g_BlurInput": input_buffer, "g_BlurOutput": scratch, **shared})
        self._dispatch("vertical", encoder, channel_count, {"g_BlurInput": scratch, "g_BlurOutput": output_buffer, **shared})
        return output_buffer

    def blur_texture(self, encoder: spy.CommandEncoder, input_texture: spy.Texture, output_texture: spy.Texture, radius: int, sigma: float) -> spy.Texture:
        weights, resolved_radius = self._weight_buffer(radius, sigma)
        shared = {
            "g_BlurWidth": self.width,
            "g_BlurHeight": self.height,
            "g_BlurWeightsDynamic": weights,
            "g_BlurDynamicRadius": resolved_radius,
        }
        scratch = self._ensure_scratch_texture(spy.Format.rgba32_float)
        self._dispatch("horizontal_texture", encoder, 1, {"g_BlurTextureInput": input_texture, "g_BlurTextureOutput": scratch, **shared})
        self._dispatch("vertical_texture", encoder, 1, {"g_BlurTextureInput": scratch, "g_BlurTextureOutput": output_texture, **shared})
        return output_texture
