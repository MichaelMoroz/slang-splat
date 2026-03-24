from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import slangpy as spy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utility.utility import GpuUtility

_SHADERS = Path(__file__).resolve().parents[1] / "utility" / "shaders"


@pytest.fixture
def utility_context(backend_name: str) -> tuple[spy.Device, GpuUtility]:
    try:
        device = spy.create_device(type=getattr(spy.DeviceType, backend_name), include_paths=[_SHADERS], enable_cuda_interop=False, enable_hot_reload=False)
    except RuntimeError as exc:
        pytest.skip(f"{backend_name} unavailable: {exc}")
    return device, GpuUtility(device)


def _tensor(device: spy.Device, count: int) -> spy.Tensor:
    return spy.Tensor.empty(device, shape=(max(count, 1),), dtype="uint")


def _to_int64(tensor: spy.Tensor, count: int) -> np.ndarray:
    return np.asarray(tensor.to_numpy()).reshape(-1)[:count].astype(np.int64, copy=False)


def _run_prefix(device: spy.Device, util: GpuUtility, values: np.ndarray, exclusive: bool) -> tuple[np.ndarray, int]:
    count = int(values.size)
    inp, out = _tensor(device, count), _tensor(device, count)
    scratch = util.prefix_scratch_elements(count)
    sums, offsets, total = _tensor(device, scratch), _tensor(device, scratch), _tensor(device, 1)
    if count:
        inp.copy_from_numpy(np.ascontiguousarray(values.astype(np.uint32, copy=False)))
    enc = device.create_command_encoder()
    util.prefix_sum_uint32(enc, inp, out, sums, offsets, total, count, exclusive)
    device.submit_command_buffer(enc.finish())
    return _to_int64(out, count), int(_to_int64(total, 1)[0])


def _run_prefix_from_count_buffer(device: spy.Device, util: GpuUtility, values: np.ndarray, exclusive: bool) -> tuple[np.ndarray, int]:
    count = int(values.size)
    inp, out = _tensor(device, count), _tensor(device, count)
    scratch = util.prefix_scratch_elements(count)
    sums, offsets, total = _tensor(device, scratch), _tensor(device, scratch), _tensor(device, 1)
    count_buffer = _tensor(device, 1)
    count_buffer.copy_from_numpy(np.array([count], dtype=np.uint32))
    if count:
        inp.copy_from_numpy(np.ascontiguousarray(values.astype(np.uint32, copy=False)))
    enc = device.create_command_encoder()
    util.prefix_sum_uint32_from_count_buffer(enc, inp, out, sums, offsets, total, count_buffer, 0, count, exclusive)
    device.submit_command_buffer(enc.finish())
    return _to_int64(out, count), int(_to_int64(total, 1)[0])


def _run_sort(device: spy.Device, util: GpuUtility, keys: np.ndarray, values: np.ndarray, start_bit: int, bit_count: int) -> tuple[np.ndarray, np.ndarray]:
    count = int(keys.size)
    keys_in, values_in = _tensor(device, count), _tensor(device, count)
    keys_out, values_out = _tensor(device, count), _tensor(device, count)
    hist_count = util.radix_histogram_elements(count)
    histogram, hist_prefix = _tensor(device, hist_count), _tensor(device, util.radix_prefix_elements(count))
    scratch = util.prefix_scratch_elements(hist_count)
    sums, offsets, total = _tensor(device, scratch), _tensor(device, scratch), _tensor(device, 1)
    if count:
        keys_in.copy_from_numpy(np.ascontiguousarray(keys.astype(np.uint32, copy=False)))
        values_in.copy_from_numpy(np.ascontiguousarray(values.astype(np.uint32, copy=False)))
    enc = device.create_command_encoder()
    out_buffer = util.radix_sort_uint32(enc, keys_in, values_in, keys_out, values_out, histogram, hist_prefix, sums, offsets, total, count, start_bit, bit_count)
    device.submit_command_buffer(enc.finish())
    final_keys, final_values = (keys_out, values_out) if out_buffer else (keys_in, values_in)
    return _to_int64(final_keys, count), _to_int64(final_values, count)


def _run_sort_from_count_buffer(
    device: spy.Device, util: GpuUtility, keys: np.ndarray, values: np.ndarray, start_bit: int, bit_count: int
) -> tuple[np.ndarray, np.ndarray]:
    count = int(keys.size)
    keys_in, values_in = _tensor(device, count), _tensor(device, count)
    keys_out, values_out = _tensor(device, count), _tensor(device, count)
    histogram, hist_prefix = _tensor(device, util.radix_histogram_elements(count)), _tensor(device, util.radix_prefix_elements(count))
    count_buffer = _tensor(device, 1)
    count_buffer.copy_from_numpy(np.array([count], dtype=np.uint32))
    if count:
        keys_in.copy_from_numpy(np.ascontiguousarray(keys.astype(np.uint32, copy=False)))
        values_in.copy_from_numpy(np.ascontiguousarray(values.astype(np.uint32, copy=False)))
    enc = device.create_command_encoder()
    out_buffer, _ = util.radix_sort_uint32_from_count_buffer(
        enc, keys_in, values_in, keys_out, values_out, histogram, hist_prefix, count_buffer, 0, count, start_bit, bit_count
    )
    device.submit_command_buffer(enc.finish())
    final_keys, final_values = (keys_out, values_out) if out_buffer else (keys_in, values_in)
    return _to_int64(final_keys, count), _to_int64(final_values, count)


@pytest.mark.parametrize("count", [0, 7, 513])
@pytest.mark.parametrize("exclusive", [False, True])
def test_prefix_sum_matches_torch(utility_context: tuple[spy.Device, GpuUtility], count: int, exclusive: bool) -> None:
    device, util = utility_context
    values = np.random.default_rng(1234 + count + int(exclusive)).integers(0, 17, size=count, dtype=np.int64)
    out, total = _run_prefix(device, util, values, exclusive)
    ref = np.cumsum(values, dtype=np.int64)
    if exclusive and count:
        ref = np.concatenate((np.zeros((1,), dtype=np.int64), ref[:-1]))
    elif exclusive:
        ref = values
    np.testing.assert_array_equal(out, ref)
    assert total == int(values.sum(dtype=np.int64))


@pytest.mark.parametrize("count", [0, 513])
@pytest.mark.parametrize("exclusive", [False, True])
def test_prefix_sum_from_count_buffer_matches_direct(utility_context: tuple[spy.Device, GpuUtility], count: int, exclusive: bool) -> None:
    device, util = utility_context
    values = np.random.default_rng(4321 + count + int(exclusive)).integers(0, 17, size=count, dtype=np.int64)
    direct_out, direct_total = _run_prefix(device, util, values, exclusive)
    indirect_out, indirect_total = _run_prefix_from_count_buffer(device, util, values, exclusive)
    np.testing.assert_array_equal(indirect_out, direct_out)
    assert indirect_total == direct_total


@pytest.mark.parametrize(
    ("count", "start_bit", "bit_count"),
    [(0, 0, 1), (37, 0, 5), (513, 0, 17), (131072, 0, 17)],
)
def test_radix_sort_matches_torch(utility_context: tuple[spy.Device, GpuUtility], count: int, start_bit: int, bit_count: int) -> None:
    device, util = utility_context
    rng = np.random.default_rng(2468 + count + start_bit * 17 + bit_count)
    keys = rng.integers(0, 2**20, size=count, dtype=np.int64)
    if count:
        keys = keys.copy()
        keys[::7] = int(keys[0])
    values = np.arange(count, dtype=np.int64)
    out_keys, out_values = _run_sort(device, util, keys, values, start_bit, bit_count)
    mask = (1 << bit_count) - 1
    masked = (keys >> start_bit) & mask
    perm = np.argsort(masked, kind="stable")
    np.testing.assert_array_equal(out_keys, keys[perm])
    np.testing.assert_array_equal(out_values, values[perm])


@pytest.mark.parametrize(
    ("count", "start_bit", "bit_count"),
    [(0, 0, 1), (128, 4, 9), (1025, 8, 24)],
)
def test_radix_sort_from_count_buffer_matches_direct(
    utility_context: tuple[spy.Device, GpuUtility], count: int, start_bit: int, bit_count: int
) -> None:
    device, util = utility_context
    rng = np.random.default_rng(8642 + count + start_bit * 17 + bit_count)
    keys = rng.integers(0, 2**20, size=count, dtype=np.int64)
    if count:
        keys = keys.copy()
        keys[::7] = int(keys[0])
    values = np.arange(count, dtype=np.int64)
    direct_keys, direct_values = _run_sort(device, util, keys, values, start_bit, bit_count)
    indirect_keys, indirect_values = _run_sort_from_count_buffer(device, util, keys, values, start_bit, bit_count)
    np.testing.assert_array_equal(indirect_keys, direct_keys)
    np.testing.assert_array_equal(indirect_values, direct_values)
