from __future__ import annotations

from threading import Event
from types import SimpleNamespace

import numpy as np
import slangpy as spy

from src.training.dataset_texture_pool import DatasetTexturePool
from src.training.gaussian_trainer import GaussianTrainer


class _FakeTexture:
    def __init__(self, *, format: spy.Format, width: int, height: int, usage: object, label: str) -> None:
        self.format = format
        self.width = int(width)
        self.height = int(height)
        self.usage = usage
        self.label = label


class _FakeBuffer:
    def __init__(self, *, size: int, usage: object, label: str) -> None:
        self.size = int(size)
        self.usage = usage
        self.label = label


class _FakeDevice:
    def __init__(self) -> None:
        self.finished: set[int] = set()
        self.wait_count = 0

    def create_texture(self, **kwargs):
        return _FakeTexture(**kwargs)

    def create_buffer(self, **kwargs):
        return _FakeBuffer(**kwargs)

    def is_submit_finished(self, submission: int) -> bool:
        return int(submission) in self.finished

    def wait(self) -> None:
        self.wait_count += 1
        self.finished.add(1)


class _FakeEncoder:
    def __init__(self) -> None:
        self.uploads: list[tuple[object, int, np.ndarray]] = []
        self.copies: list[tuple[object, object, int, int, object]] = []

    def upload_buffer_data(self, buffer, offset: int, data: np.ndarray) -> None:
        self.uploads.append((buffer, int(offset), np.asarray(data).copy()))

    def copy_buffer_to_texture(self, dst, dst_layer: int, dst_mip: int, dst_offset, src, src_offset: int, src_size: int, src_row_pitch: int, extent) -> None:
        self.copies.append((dst, src, int(src_size), int(src_row_pitch), extent))


def _uint3_tuple(value) -> tuple[int, int, int]:
    return int(value.x), int(value.y), int(value.z)


def test_bc7_pool_uses_aligned_texture_and_upload_extent() -> None:
    payload = np.arange(2 * 2 * 16, dtype=np.uint8)
    provider = lambda _frame_index: SimpleNamespace(width=5, height=7, format=spy.Format.bc7_unorm_srgb, payload=payload)
    device = _FakeDevice()
    pool = DatasetTexturePool(device, [(5, 7)], provider, texture_format=spy.Format.bc7_unorm_srgb, pool_size=1, max_width=5, max_height=7, prefetch_depth=1)
    encoder = _FakeEncoder()

    texture = pool.acquire(0, encoder)

    assert texture.width == 8
    assert texture.height == 8
    assert len(encoder.uploads) == 1
    assert len(encoder.copies) == 1
    _, _, src_size, row_pitch, extent = encoder.copies[0]
    assert src_size == payload.nbytes
    assert row_pitch == 2 * 16
    assert _uint3_tuple(extent) == (8, 8, 1)


def test_pool_reuses_slot_only_after_submission_finishes() -> None:
    provider = lambda frame_index: np.full((2, 2, 4), int(frame_index), dtype=np.uint8)
    device = _FakeDevice()
    pool = DatasetTexturePool(device, [(2, 2), (2, 2)], provider, texture_format=spy.Format.rgba8_unorm_srgb, pool_size=1, max_width=2, max_height=2, prefetch_depth=1)
    first_encoder = SimpleNamespace(upload_texture_data=lambda *args: None)
    first_texture = pool.acquire(0, first_encoder)
    pool.mark_submitted([0], 1)

    second_encoder = SimpleNamespace(upload_texture_data=lambda *args: None)
    second_texture = pool.acquire(1, second_encoder)

    assert second_texture is first_texture
    assert device.wait_count == 1


def test_prefetch_trims_oldest_futures_to_depth() -> None:
    release_provider = Event()

    def provider(frame_index: int) -> np.ndarray:
        if frame_index == 0:
            release_provider.wait(timeout=1.0)
        return np.full((2, 2, 4), int(frame_index), dtype=np.uint8)

    pool = DatasetTexturePool(
        _FakeDevice(),
        [(2, 2)] * 4,
        provider,
        texture_format=spy.Format.rgba8_unorm_srgb,
        pool_size=2,
        max_width=2,
        max_height=2,
        prefetch_depth=2,
        prefetch_threads=1,
    )
    try:
        pool.prefetch([0, 1, 2, 3])
        assert list(pool._futures.keys()) == [2, 3]
    finally:
        release_provider.set()
        pool.release()


def test_effective_batch_request_clamps_only_streaming_dataset_pool() -> None:
    trainer = object.__new__(GaussianTrainer)

    trainer._dataset_targets = SimpleNamespace(is_streaming=True, pool_size=2)
    assert trainer._effective_batch_request(5) == 2

    trainer._dataset_targets = SimpleNamespace(is_streaming=False, pool_size=2)
    assert trainer._effective_batch_request(5) == 5
