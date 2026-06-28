from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import slangpy as spy

from ..utility import RO_BUFFER_USAGE, alloc_buffer, alloc_texture_2d, defer_resource_release

PREFETCH_THREADS = 2
BC7_BLOCK_BYTES = 16
BC_BLOCK_SIZE = 4


def align4(value: int) -> int:
    return ((max(int(value), 1) + BC_BLOCK_SIZE - 1) // BC_BLOCK_SIZE) * BC_BLOCK_SIZE


def _bc7_block_count(value: int) -> int:
    return align4(value) // BC_BLOCK_SIZE


def _is_bc7_format(texture_format: spy.Format) -> bool:
    return texture_format in (spy.Format.bc7_unorm, spy.Format.bc7_unorm_srgb)


@dataclass(slots=True)
class _PoolSlot:
    texture: spy.Texture
    staging: spy.Buffer | None
    frame_index: int = -1
    pending: bool = False
    submission: int | None = None
    last_used: int = 0


class PreloadedDatasetTextures:
    def __init__(self, textures: list[spy.Texture]) -> None:
        self.textures = list(textures)
        self.pool_size = len(self.textures)
        self.is_streaming = False

    def acquire(self, frame_index: int, encoder: spy.CommandEncoder | None = None) -> spy.Texture:
        return self.textures[int(frame_index)]

    def prefetch(self, frame_indices: object) -> None:
        return

    def mark_submitted(self, frame_indices: object, submission: int | None) -> None:
        return

    def release(self, *, preserve_frame_targets: bool = False) -> None:
        if not preserve_frame_targets:
            for texture in self.textures:
                defer_resource_release(texture)
            self.textures = []

    @property
    def frame_targets_native(self) -> list[spy.Texture]:
        return list(self.textures)


class DatasetTexturePool:
    def __init__(
        self,
        device: spy.Device,
        frame_sizes: list[tuple[int, int]],
        payload_provider: Callable[[int], Any],
        *,
        texture_format: spy.Format,
        pool_size: int,
        max_width: int,
        max_height: int,
        prefetch_depth: int,
        prefetch_threads: int = PREFETCH_THREADS,
    ) -> None:
        self.device = device
        self.frame_sizes = [(max(int(width), 1), max(int(height), 1)) for width, height in frame_sizes]
        self.payload_provider = payload_provider
        self.texture_format = texture_format
        self.pool_size = max(int(pool_size), 1)
        self.prefetch_depth = max(int(prefetch_depth), 1)
        self.is_bc7 = _is_bc7_format(texture_format)
        self.is_streaming = True
        texture_width = align4(max_width) if self.is_bc7 else max(int(max_width), 1)
        texture_height = align4(max_height) if self.is_bc7 else max(int(max_height), 1)
        self._slots = [
            _PoolSlot(
                texture=alloc_texture_2d(
                    device,
                    name="trainer.dataset_pool.target",
                    format=texture_format,
                    width=texture_width,
                    height=texture_height,
                    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
                ),
                staging=alloc_buffer(
                    device,
                    name="trainer.dataset_pool.bc7_staging",
                    size=_bc7_payload_size(texture_width, texture_height),
                    usage=RO_BUFFER_USAGE,
                )
                if self.is_bc7
                else None,
            )
            for _ in range(self.pool_size)
        ]
        self._frame_to_slot: dict[int, int] = {}
        self._futures: OrderedDict[int, Future[Any]] = OrderedDict()
        self._executor = ThreadPoolExecutor(max_workers=max(int(prefetch_threads), 1), thread_name_prefix="dataset-pool")
        self._clock = 0

    @property
    def frame_targets_native(self) -> list[spy.Texture]:
        return [slot.texture for slot in self._slots]

    def acquire(self, frame_index: int, encoder: spy.CommandEncoder) -> spy.Texture:
        frame_index = int(frame_index)
        existing = self._frame_to_slot.get(frame_index)
        if existing is not None:
            slot = self._slots[existing]
            slot.pending = True
            slot.last_used = self._tick()
            return slot.texture
        payload = self._consume_payload(frame_index)
        slot_index = self._select_slot()
        slot = self._slots[slot_index]
        if slot.frame_index >= 0:
            self._frame_to_slot.pop(slot.frame_index, None)
        self._upload_payload(encoder, slot, frame_index, payload)
        slot.frame_index = frame_index
        slot.pending = True
        slot.submission = None
        slot.last_used = self._tick()
        self._frame_to_slot[frame_index] = slot_index
        return slot.texture

    def prefetch(self, frame_indices: object) -> None:
        for raw_index in frame_indices:
            frame_index = int(raw_index)
            if frame_index in self._frame_to_slot or frame_index in self._futures:
                continue
            self._futures[frame_index] = self._executor.submit(self.payload_provider, frame_index)
            self._trim_futures()

    def mark_submitted(self, frame_indices: object, submission: int | None) -> None:
        for raw_index in frame_indices:
            slot_index = self._frame_to_slot.get(int(raw_index))
            if slot_index is None:
                continue
            slot = self._slots[slot_index]
            slot.pending = False
            slot.submission = None if submission is None else int(submission)

    def release(self, *, preserve_frame_targets: bool = False) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._futures.clear()
        if preserve_frame_targets:
            return
        for slot in self._slots:
            defer_resource_release(slot.texture)
            defer_resource_release(slot.staging)
        self._slots = []
        self._frame_to_slot.clear()

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _slot_finished(self, slot: _PoolSlot) -> bool:
        if slot.pending:
            return False
        return slot.submission is None or bool(self.device.is_submit_finished(int(slot.submission)))

    def _select_slot(self) -> int:
        best_index = -1
        best_used = 0
        for index, slot in enumerate(self._slots):
            if slot.frame_index < 0:
                return index
            if self._slot_finished(slot) and (best_index < 0 or slot.last_used < best_used):
                best_index, best_used = index, slot.last_used
        if best_index >= 0:
            return best_index
        wait = getattr(self.device, "wait", None)
        if callable(wait):
            wait()
        for index, slot in enumerate(self._slots):
            if self._slot_finished(slot):
                return index
        raise RuntimeError("Dataset texture pool has no reusable slot.")

    def _consume_payload(self, frame_index: int) -> Any:
        future = self._futures.pop(frame_index, None)
        if future is not None:
            return future.result()
        return self.payload_provider(frame_index)

    def _trim_futures(self) -> None:
        while len(self._futures) > self.prefetch_depth:
            _, future = self._futures.popitem(last=False)
            future.cancel()

    def _upload_payload(self, encoder: spy.CommandEncoder, slot: _PoolSlot, frame_index: int, payload: Any) -> None:
        width, height = self.frame_sizes[frame_index]
        if self.is_bc7:
            if slot.staging is None:
                raise RuntimeError("BC7 dataset pool slot is missing its staging buffer.")
            raw = np.ascontiguousarray(getattr(payload, "payload"), dtype=np.uint8)
            encoder.upload_buffer_data(slot.staging, 0, raw)
            blocks_x = _bc7_block_count(width)
            encoder.copy_buffer_to_texture(
                slot.texture,
                0,
                0,
                spy.uint3(0, 0, 0),
                slot.staging,
                0,
                int(raw.nbytes),
                int(blocks_x * BC7_BLOCK_BYTES),
                spy.uint3(align4(width), align4(height), 1),
            )
            return
        encoder.upload_texture_data(
            slot.texture,
            spy.uint3(0, 0, 0),
            spy.uint3(width, height, 1),
            spy.SubresourceRange({"layer": 0, "layer_count": 1, "mip": 0, "mip_count": 1}),
            [np.ascontiguousarray(payload, dtype=np.uint8)],
        )


def _bc7_payload_size(width: int, height: int) -> int:
    return _bc7_block_count(width) * _bc7_block_count(height) * BC7_BLOCK_BYTES
