from __future__ import annotations

from pathlib import Path
import sys

import pytest
import slangpy as spy
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utility.utility import GpuUtility

_SHADERS = Path(__file__).resolve().parents[1] / "utility" / "shaders"


@pytest.fixture(scope="module")
def utility_context() -> tuple[spy.Device, GpuUtility]:
    device = spy.create_torch_device(type=spy.DeviceType.cuda, include_paths=[_SHADERS])
    return device, GpuUtility(device)


def _tensor(device: spy.Device, count: int) -> spy.Tensor:
    return spy.Tensor.empty(device, shape=(max(count, 1),), dtype="uint")


def _run_prefix(device: spy.Device, util: GpuUtility, values: torch.Tensor, exclusive: bool) -> tuple[torch.Tensor, int]:
    count = int(values.numel())
    inp, out = _tensor(device, count), _tensor(device, count)
    scratch = util.prefix_scratch_elements(count)
    sums, offsets, total = _tensor(device, scratch), _tensor(device, scratch), _tensor(device, 1)
    if count:
        inp.copy_from_torch(values.to(dtype=torch.uint32))
    enc = device.create_command_encoder()
    util.prefix_sum_uint32(enc, inp, out, sums, offsets, total, count, exclusive)
    device.submit_command_buffer(enc.finish())
    device.sync_to_device()
    return out.to_torch()[:count].to(dtype=torch.int64), int(total.to_torch()[0].item())


def _run_sort(device: spy.Device, util: GpuUtility, keys: torch.Tensor, values: torch.Tensor, start_bit: int, bit_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    count = int(keys.numel())
    keys_in, values_in = _tensor(device, count), _tensor(device, count)
    keys_out, values_out = _tensor(device, count), _tensor(device, count)
    hist_count = util.radix_histogram_elements(count)
    histogram, hist_prefix = _tensor(device, hist_count), _tensor(device, hist_count)
    scratch = util.prefix_scratch_elements(hist_count)
    sums, offsets, total = _tensor(device, scratch), _tensor(device, scratch), _tensor(device, 1)
    if count:
        keys_in.copy_from_torch(keys.to(dtype=torch.uint32))
        values_in.copy_from_torch(values.to(dtype=torch.uint32))
    enc = device.create_command_encoder()
    out_buffer = util.radix_sort_uint32(enc, keys_in, values_in, keys_out, values_out, histogram, hist_prefix, sums, offsets, total, count, start_bit, bit_count)
    device.submit_command_buffer(enc.finish())
    device.sync_to_device()
    final_keys, final_values = (keys_out, values_out) if out_buffer else (keys_in, values_in)
    return final_keys.to_torch()[:count].to(dtype=torch.int64), final_values.to_torch()[:count].to(dtype=torch.int64)


@pytest.mark.parametrize("count", [0, 1, 7, 513, 4097])
@pytest.mark.parametrize("exclusive", [False, True])
def test_prefix_sum_matches_torch(utility_context: tuple[spy.Device, GpuUtility], count: int, exclusive: bool) -> None:
    device, util = utility_context
    values = torch.randint(0, 17, (count,), device="cuda", dtype=torch.int64)
    out, total = _run_prefix(device, util, values, exclusive)
    ref = torch.cumsum(values, 0)
    if exclusive and count:
        ref = torch.cat((torch.zeros((1,), device=values.device, dtype=values.dtype), ref[:-1]))
    elif exclusive:
        ref = values
    torch.testing.assert_close(out, ref)
    assert total == int(values.sum().item())


@pytest.mark.parametrize(
    ("count", "start_bit", "bit_count"),
    [(0, 0, 1), (1, 0, 32), (37, 0, 5), (128, 4, 9), (513, 0, 17), (1025, 8, 24)],
)
def test_radix_sort_matches_torch(utility_context: tuple[spy.Device, GpuUtility], count: int, start_bit: int, bit_count: int) -> None:
    device, util = utility_context
    keys = torch.randint(0, 2**20, (count,), device="cuda", dtype=torch.int64)
    if count:
        keys = keys.clone()
        keys[::7] = int(keys[0].item())
    values = torch.arange(count, device="cuda", dtype=torch.int64)
    out_keys, out_values = _run_sort(device, util, keys, values, start_bit, bit_count)
    mask = (1 << bit_count) - 1
    masked = (keys >> start_bit) & mask
    perm = torch.argsort(masked, stable=True)
    torch.testing.assert_close(out_keys, keys.index_select(0, perm))
    torch.testing.assert_close(out_values, values.index_select(0, perm))
