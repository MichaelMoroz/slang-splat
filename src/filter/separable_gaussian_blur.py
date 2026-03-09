from __future__ import annotations

from pathlib import Path

import numpy as np
import slangpy as spy

from ..common import SHADER_ROOT, thread_count_2d


class SeparableGaussianBlur:
    _BUFFER_USAGE = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
    _KERNEL_ENTRIES = {"horizontal": "csGaussianBlurHorizontal", "vertical": "csGaussianBlurVertical"}

    def _dispatch(self, kernel: str, encoder: spy.CommandEncoder, channel_count: int, vars: dict[str, object]) -> None:
        self._kernels[kernel].dispatch(thread_count=thread_count_2d(self.width, self.height, channel_count), vars=vars, command_encoder=encoder)

    def __init__(self, device: spy.Device, width: int, height: int) -> None:
        self.device, self.width, self.height = device, int(width), int(height)
        self._shader_path = Path(SHADER_ROOT / "utility" / "blur" / "separable_gaussian_blur.slang")
        self._kernels = {name: self.device.create_compute_kernel(self.device.load_program(str(self._shader_path), [entry])) for name, entry in self._KERNEL_ENTRIES.items()}
        self._scratch_buffers: dict[int, spy.Buffer] = {}

    def make_buffer(self, channel_count: int) -> spy.Buffer:
        return self.device.create_buffer(size=self._buffer_size(channel_count), usage=self._BUFFER_USAGE)

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

    def blur(self, encoder: spy.CommandEncoder, input_buffer: spy.Buffer, output_buffer: spy.Buffer, channel_count: int) -> spy.Buffer:
        shared = {"g_BlurWidth": self.width, "g_BlurHeight": self.height, "g_BlurChannelCount": int(channel_count)}
        scratch = self._ensure_scratch_buffer(channel_count)
        self._dispatch("horizontal", encoder, channel_count, {"g_BlurInput": input_buffer, "g_BlurOutput": scratch, **shared})
        self._dispatch("vertical", encoder, channel_count, {"g_BlurInput": scratch, "g_BlurOutput": output_buffer, **shared})
        return output_buffer
