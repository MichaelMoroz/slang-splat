from __future__ import annotations

import os

import numpy as np
import pytest
import slangpy as spy

from src.sort import GPURadixSort, sort_numpy


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


def _cpu_sort(keys: np.ndarray, values: np.ndarray, start_bit: int, bit_count: int) -> tuple[np.ndarray, np.ndarray]:
    mask = (1 << bit_count) - 1
    masked = (keys.astype(np.uint64) >> start_bit) & mask
    perm = np.argsort(masked, kind="stable")
    return keys[perm], values[perm]


def _run_sort(device: spy.Device, keys: np.ndarray, values: np.ndarray, start_bit: int, bit_count: int) -> tuple[np.ndarray, np.ndarray]:
    sorter = GPURadixSort(device)
    keys_buffer = _buffer_u32(device, keys)
    values_buffer = _buffer_u32(device, values)
    enc = device.create_command_encoder()
    sorter.sort_key_values(enc, keys_buffer, values_buffer, keys.size, max_bits=bit_count + start_bit)
    device.submit_command_buffer(enc.finish())
    device.wait()
    sorted_keys = _read_u32(keys_buffer, keys.size)
    sorted_values = _read_u32(values_buffer, values.size)
    if start_bit == 0 and bit_count + start_bit == 32:
        return sorted_keys, sorted_values
    ref_keys, ref_values = _cpu_sort(keys, values, start_bit, bit_count)
    return sorted_keys, sorted_values if start_bit == 0 else (sorted_keys, sorted_values)


def _run_sort_from_count_buffer(device: spy.Device, keys: np.ndarray, values: np.ndarray, start_bit: int, bit_count: int) -> tuple[np.ndarray, np.ndarray]:
    sorter = GPURadixSort(device)
    keys_buffer = _buffer_u32(device, keys)
    values_buffer = _buffer_u32(device, values)
    count_buffer = _buffer_u32(device, np.array([keys.size], dtype=np.uint32))
    enc = device.create_command_encoder()
    sorter.sort_key_values_from_count_buffer(enc, keys_buffer, values_buffer, count_buffer, 0, keys.size, max_bits=bit_count + start_bit)
    device.submit_command_buffer(enc.finish())
    device.wait()
    return _read_u32(keys_buffer, keys.size), _read_u32(values_buffer, values.size)


@pytest.mark.parametrize(("count", "start_bit", "bit_count"), [(0, 0, 1), (37, 0, 5), (513, 0, 17), (131072, 0, 17), (1025, 8, 24)])
def test_gpu_radix_sort_matches_cpu_reference(device: spy.Device, count: int, start_bit: int, bit_count: int) -> None:
    rng = np.random.default_rng(2468 + count + start_bit * 17 + bit_count)
    keys = rng.integers(0, np.iinfo(np.uint32).max, size=count, dtype=np.uint32)
    if count > 0:
        keys = keys.copy()
        keys[::7] = keys[0]
    values = np.arange(count, dtype=np.uint32)
    sorted_keys, sorted_values = sort_numpy(device, keys, values, max_bits=bit_count + start_bit)
    expected_keys, expected_values = _cpu_sort(keys, values, 0, bit_count + start_bit)
    np.testing.assert_array_equal(sorted_keys, expected_keys)
    np.testing.assert_array_equal(sorted_values, expected_values)


def test_gpu_radix_sort_preserves_duplicate_key_stability(device: spy.Device) -> None:
    keys = np.array([5, 2, 5, 5, 1, 2, 5, 1], dtype=np.uint32)
    values = np.arange(keys.size, dtype=np.uint32)
    sorted_keys, sorted_values = sort_numpy(device, keys, values, max_bits=32)
    expected_keys, expected_values = _cpu_sort(keys, values, 0, 32)
    np.testing.assert_array_equal(sorted_keys, expected_keys)
    np.testing.assert_array_equal(sorted_values, expected_values)


@pytest.mark.parametrize(("count", "bit_count"), [(0, 1), (128, 13), (1025, 24), (131072, 32)])
def test_radix_sort_from_count_buffer_matches_direct(device: spy.Device, count: int, bit_count: int) -> None:
    rng = np.random.default_rng(8642 + count + bit_count)
    keys = rng.integers(0, np.iinfo(np.uint32).max, size=count, dtype=np.uint32)
    if count > 0:
        keys = keys.copy()
        keys[::7] = keys[0]
    values = np.arange(count, dtype=np.uint32)
    direct_keys, direct_values = sort_numpy(device, keys, values, max_bits=bit_count)
    indirect_keys, indirect_values = _run_sort_from_count_buffer(device, keys, values, 0, bit_count)
    np.testing.assert_array_equal(indirect_keys, direct_keys)
    np.testing.assert_array_equal(indirect_values, direct_values)


@pytest.mark.skipif(not _RUN_LARGE, reason="set RUN_SLOW_GPU_UTILITY_TESTS=1 to run large radix-sort regressions")
@pytest.mark.parametrize("bit_count", [17, 32])
def test_radix_sort_from_count_buffer_matches_direct_large(device: spy.Device, bit_count: int) -> None:
    rng = np.random.default_rng(7654321 + bit_count)
    keys = rng.integers(0, np.iinfo(np.uint32).max, size=_LARGE_TEST_COUNT, dtype=np.uint32)
    if _LARGE_TEST_COUNT > 0:
        keys = keys.copy()
        keys[::7] = keys[0]
    values = np.arange(_LARGE_TEST_COUNT, dtype=np.uint32)
    direct_keys, direct_values = sort_numpy(device, keys, values, max_bits=bit_count)
    indirect_keys, indirect_values = _run_sort_from_count_buffer(device, keys, values, 0, bit_count)
    np.testing.assert_array_equal(indirect_keys, direct_keys)
    np.testing.assert_array_equal(indirect_values, direct_values)
