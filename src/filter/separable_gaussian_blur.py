from __future__ import annotations

from pathlib import Path

import numpy as np
import slangpy as spy

from ..utility import RW_BUFFER_USAGE, SHADER_ROOT, alloc_buffer, dispatch, load_compute_kernels


class SeparableGaussianBlur:
    _KERNEL_ENTRIES = {
        "horizontal": "csGaussianBlurHorizontal",
        "vertical": "csGaussianBlurVertical",
        "horizontal_adjoint": "csGaussianBlurHorizontalAdjoint",
        "vertical_adjoint": "csGaussianBlurVerticalAdjoint",
    }

    def _channel_block_count(self, channel_count: int) -> int:
        return (int(channel_count) + self.channel_pack - 1) // self.channel_pack

    def _thread_count(self, kernel: str, channel_count: int) -> spy.uint3:
        channels = self._channel_block_count(channel_count) * self.channel_pack
        if kernel in {"horizontal", "horizontal_adjoint"}:
            return spy.uint3(channels, self.width, self.height)
        return spy.uint3(channels, self.height, self.width)

    def _dispatch(self, kernel: str, encoder: spy.CommandEncoder, channel_count: int, vars: dict[str, object]) -> None:
        dispatch(
            kernel=self._kernels[kernel],
            thread_count=self._thread_count(kernel, channel_count),
            vars=vars,
            command_encoder=encoder,
            debug_label=f"Blur::{kernel}",
            debug_color_index=80 + len(kernel),
        )

    def __init__(self, device: spy.Device, width: int, height: int, shader_path: str | Path | None = None, channel_pack: int = 16) -> None:
        self.device, self.width, self.height = device, int(width), int(height)
        self.channel_pack = max(int(channel_pack), 1)
        path = Path(shader_path) if shader_path is not None else SHADER_ROOT / "utility" / "blur" / "separable_gaussian_blur.slang"
        self._kernels = load_compute_kernels(device, path, self._KERNEL_ENTRIES)
        self._scratch_buffers: dict[int, spy.Buffer] = {}

    def make_buffer(self, channel_count: int, name: str | None = None) -> spy.Buffer:
        buffer_name = str(name) if name is not None else f"blur.{self.width}x{self.height}.{int(channel_count)}ch"
        return alloc_buffer(self.device, name=buffer_name, size=self._buffer_size(channel_count), usage=RW_BUFFER_USAGE)

    def _buffer_size(self, channel_count: int) -> int:
        channels = int(channel_count)
        if channels <= 0:
            raise ValueError("Blur channel_count must be positive.")
        return self.width * self.height * channels * np.dtype(np.float32).itemsize

    def _ensure_scratch_buffer(self, channel_count: int) -> spy.Buffer:
        channels = int(channel_count)
        if channels not in self._scratch_buffers:
            self._scratch_buffers[channels] = self.make_buffer(channels, name=f"blur.scratch.{self.width}x{self.height}.{channels}ch")
        return self._scratch_buffers[channels]

    def blur(self, encoder: spy.CommandEncoder, input_buffer: spy.Buffer, output_buffer: spy.Buffer, channel_count: int) -> spy.Buffer:
        shared = {"g_BlurWidth": self.width, "g_BlurHeight": self.height, "g_BlurChannelCount": int(channel_count)}
        scratch = self._ensure_scratch_buffer(channel_count)
        self._dispatch("horizontal", encoder, channel_count, {"g_BlurInput": input_buffer, "g_BlurOutput": scratch, **shared})
        self._dispatch("vertical", encoder, channel_count, {"g_BlurInput": scratch, "g_BlurOutput": output_buffer, **shared})
        return output_buffer

    def blur_adjoint(self, encoder: spy.CommandEncoder, input_buffer: spy.Buffer, output_buffer: spy.Buffer, channel_count: int) -> spy.Buffer:
        shared = {"g_BlurWidth": self.width, "g_BlurHeight": self.height, "g_BlurChannelCount": int(channel_count)}
        scratch = self._ensure_scratch_buffer(channel_count)
        self._dispatch("vertical_adjoint", encoder, channel_count, {"g_BlurInput": input_buffer, "g_BlurOutput": scratch, **shared})
        self._dispatch("horizontal_adjoint", encoder, channel_count, {"g_BlurInput": scratch, "g_BlurOutput": output_buffer, **shared})
        return output_buffer
