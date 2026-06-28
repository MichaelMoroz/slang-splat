from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import slangpy as spy

from ..renderer import GaussianRenderer
from ..scene import ColmapFrame
from ..utility import RW_BUFFER_USAGE, SHADER_ROOT, alloc_buffer, debug_region, defer_resource_release, dispatch, grow_capacity, load_compute_kernels, thread_count_1d


class TrainingImageColorInitializer:
    SHADER_PATH = SHADER_ROOT / "utility" / "splatting" / "training_image_color_init.slang"
    KERNEL_ENTRIES = {
        "clear": "csClearTrainingImageColorInit",
        "sample": "csSampleTrainingImageColorInit",
    }

    def __init__(self, device: spy.Device, shader_path: str | Path | None = None) -> None:
        self.device = device
        self.shader_path = Path(shader_path) if shader_path is not None else self.SHADER_PATH
        self._kernels = self._create_shaders()
        self._best_distance: spy.Buffer | None = None
        self._best_distance_capacity = 0

    def _create_shaders(self) -> dict[str, spy.ComputeKernel]:
        return load_compute_kernels(self.device, self.shader_path, self.KERNEL_ENTRIES)

    def _ensure_buffers(self, splat_count: int) -> None:
        count = max(int(splat_count), 1)
        if self._best_distance is not None and count <= self._best_distance_capacity:
            return
        if self._best_distance is not None:
            defer_resource_release(self._best_distance)
        self._best_distance_capacity = grow_capacity(count, self._best_distance_capacity)
        self._best_distance = alloc_buffer(
            self.device,
            name="training.image_color_init.best_distance",
            size=int(self._best_distance_capacity) * 4,
            usage=RW_BUFFER_USAGE,
        )

    def _dispatch_clear(self, encoder: spy.CommandEncoder, splat_count: int) -> None:
        dispatch(
            kernel=self._kernels["clear"],
            thread_count=thread_count_1d(splat_count),
            vars={
                "g_BestDistance": self._best_distance,
                "g_SplatCount": int(splat_count),
            },
            command_encoder=encoder,
            debug_label="Training Image Color Init Clear",
            debug_color_index=120,
        )

    def _dispatch_sample_frame(
        self,
        encoder: spy.CommandEncoder,
        renderer: GaussianRenderer,
        frame: ColmapFrame,
        texture: spy.Texture,
        splat_count: int,
    ) -> None:
        width, height = max(int(texture.width), 1), max(int(texture.height), 1)
        camera = frame.make_camera()
        dispatch(
            kernel=self._kernels["sample"],
            thread_count=thread_count_1d(splat_count),
            vars={
                "g_SplatParams": renderer.scene_buffers["splat_params"],
                "g_BestDistance": self._best_distance,
                "g_TrainingImage": texture,
                "g_Camera": camera.gpu_params(width, height),
                "g_SplatCount": int(splat_count),
                "g_PackedParamCount": int(renderer.packed_trainable_param_count),
                "g_ImageWidth": int(width),
                "g_ImageHeight": int(height),
            },
            command_encoder=encoder,
            debug_label="Training Image Color Init Sample",
            debug_color_index=121,
        )

    def apply(
        self,
        encoder: spy.CommandEncoder,
        renderer: GaussianRenderer,
        frames: list[ColmapFrame],
        frame_textures: list[spy.Texture],
        splat_count: int,
    ) -> None:
        count = max(int(splat_count), 0)
        if count <= 0 or len(frames) == 0 or len(frames) != len(frame_textures):
            return
        self._ensure_buffers(count)
        with debug_region(encoder, "Training Image Color Init", 119):
            self._dispatch_clear(encoder, count)
            for frame, texture in zip(frames, frame_textures, strict=False):
                self._dispatch_sample_frame(encoder, renderer, frame, texture, count)

    def apply_streaming(
        self,
        renderer: GaussianRenderer,
        frames: list[ColmapFrame],
        texture_provider: Callable[[int, spy.CommandEncoder], spy.Texture],
        mark_submitted: Callable[[int, int], None],
        splat_count: int,
    ) -> None:
        count = max(int(splat_count), 0)
        if count <= 0 or len(frames) == 0:
            return
        self._ensure_buffers(count)
        encoder = self.device.create_command_encoder()
        with debug_region(encoder, "Training Image Color Init Clear", 119):
            self._dispatch_clear(encoder, count)
        self.device.submit_command_buffer(encoder.finish())
        self.device.wait()
        for frame_index, frame in enumerate(frames):
            encoder = self.device.create_command_encoder()
            texture = texture_provider(frame_index, encoder)
            with debug_region(encoder, "Training Image Color Init Sample", 119):
                self._dispatch_sample_frame(encoder, renderer, frame, texture, count)
            submission = self.device.submit_command_buffer(encoder.finish())
            mark_submitted(frame_index, submission)
            self.device.wait()
