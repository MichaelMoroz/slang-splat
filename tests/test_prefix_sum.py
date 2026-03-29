from __future__ import annotations

import os

import numpy as np
import pytest
import slangpy as spy

from src.scan import GPUPrefixSum


_USAGE = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
_RUN_LARGE = os.environ.get("RUN_SLOW_GPU_UTILITY_TESTS") == "1"
_LARGE_TEST_COUNT = 33_554_432


def _buffer_u32(device: spy.Device, values: np.ndarray) -> spy.Buffer:
    data = np.ascontiguousarray(values, dtype=np.uint32).reshape(-1)
    buffer = device.create_buffer(size=max(data.size, 1) * 4, usage=_USAGE)
    if data.size > 0:
        buffer.copy_from_numpy(data)
    return buffer


def _read_u32(buffer: spy.Buffer, count: int) -> np.ndarray:
    return np.frombuffer(buffer.to_numpy().tobytes(), dtype=np.uint32)[:count].copy()


def _run_prefix(device: spy.Device, values: np.ndarray, exclusive: bool) -> tuple[np.ndarray, int]:
    scan = GPUPrefixSum(device)
    src = _buffer_u32(device, values)
    dst = _buffer_u32(device, np.zeros((values.size,), dtype=np.uint32))
    total = _buffer_u32(device, np.zeros((1,), dtype=np.uint32))
    enc = device.create_command_encoder()
    scan.scan_uint(enc, src, dst, values.size, total, exclusive=exclusive)
    device.submit_command_buffer(enc.finish())
    device.wait()
    return _read_u32(dst, values.size), int(_read_u32(total, 1)[0])


def _run_prefix_from_count_buffer(device: spy.Device, values: np.ndarray, exclusive: bool) -> tuple[np.ndarray, int]:
    scan = GPUPrefixSum(device)
    src = _buffer_u32(device, values)
    dst = _buffer_u32(device, np.zeros((values.size,), dtype=np.uint32))
    total = _buffer_u32(device, np.zeros((1,), dtype=np.uint32))
    count = _buffer_u32(device, np.array([values.size], dtype=np.uint32))
    enc = device.create_command_encoder()
    scan.scan_uint_from_count_buffer(enc, src, dst, count, 0, values.size, total, exclusive=exclusive)
    device.submit_command_buffer(enc.finish())
    device.wait()
    return _read_u32(dst, values.size), int(_read_u32(total, 1)[0])


@pytest.mark.parametrize("count", [0, 7, 513, 8197])
@pytest.mark.parametrize("exclusive", [False, True])
def test_prefix_sum_matches_numpy(device: spy.Device, count: int, exclusive: bool) -> None:
    values = np.random.default_rng(1234 + count + int(exclusive)).integers(0, 17, size=count, dtype=np.uint32)
    out, total = _run_prefix(device, values, exclusive)
    expected = np.cumsum(values.astype(np.uint64), dtype=np.uint64).astype(np.uint32, copy=False)
    if exclusive and count > 0:
        expected = np.concatenate((np.zeros((1,), dtype=np.uint32), expected[:-1]))
    elif exclusive:
        expected = np.zeros((0,), dtype=np.uint32)
    np.testing.assert_array_equal(out, expected)
    assert total == int(values.astype(np.uint64).sum(dtype=np.uint64))


@pytest.mark.parametrize("count", [0, 513, 131072])
@pytest.mark.parametrize("exclusive", [False, True])
def test_prefix_sum_from_count_buffer_matches_direct(device: spy.Device, count: int, exclusive: bool) -> None:
    values = np.random.default_rng(4321 + count + int(exclusive)).integers(0, 17, size=count, dtype=np.uint32)
    direct_out, direct_total = _run_prefix(device, values, exclusive)
    indirect_out, indirect_total = _run_prefix_from_count_buffer(device, values, exclusive)
    np.testing.assert_array_equal(indirect_out, direct_out)
    assert indirect_total == direct_total


@pytest.mark.skipif(not _RUN_LARGE, reason="set RUN_SLOW_GPU_UTILITY_TESTS=1 to run large prefix-sum regressions")
@pytest.mark.parametrize("exclusive", [False, True])
def test_prefix_sum_from_count_buffer_matches_direct_large(device: spy.Device, exclusive: bool) -> None:
    values = np.random.default_rng(987654 + int(exclusive)).integers(0, 17, size=_LARGE_TEST_COUNT, dtype=np.uint32)
    direct_out, direct_total = _run_prefix(device, values, exclusive)
    indirect_out, indirect_total = _run_prefix_from_count_buffer(device, values, exclusive)
    np.testing.assert_array_equal(indirect_out, direct_out)
    assert indirect_total == direct_total
