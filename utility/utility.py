from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import slangpy as spy
from .debug import debug_group

_ROOT = Path(__file__).resolve().parent
_SHADERS = _ROOT / "shaders"
_PREFIX_BLOCK_SIZE = 512
_RADIX_GROUP_SIZE = 128
_RADIX_BIN_COUNT = 256
_DEBUG_COLOR = spy.float3(0.18, 0.58, 0.92)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass
class GpuUtility:
    device: spy.Device

    def __post_init__(self) -> None:
        self.k_prefix_scan = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csPrefixScanBlocks"]))
        self.k_prefix_add = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csPrefixAddOffsets"]))
        self.k_prefix_total = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csPrefixWriteTotal"]))
        self.k_histogram = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csRadixHistogram"]))
        self.k_scatter = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csRadixScatter"]))

    @staticmethod
    def prefix_scratch_elements(count: int) -> int:
        total, size = 1, max(int(count), 1)
        while size > _PREFIX_BLOCK_SIZE:
            size = _ceil_div(size, _PREFIX_BLOCK_SIZE)
            total += size
        return total

    @staticmethod
    def radix_histogram_elements(count: int) -> int:
        return max(_ceil_div(max(int(count), 1), _RADIX_GROUP_SIZE) * _RADIX_BIN_COUNT, _RADIX_BIN_COUNT)

    def _slice(self, tensor: spy.Tensor, count: int, offset: int = 0) -> spy.Tensor:
        return tensor.view((count,), offset=offset)

    def _prefix_recursive(
        self,
        command_encoder: spy.CommandEncoder,
        input_tensor: spy.Tensor,
        output_tensor: spy.Tensor,
        block_sums: spy.Tensor,
        block_offsets: spy.Tensor,
        count: int,
        scratch_offset: int,
        exclusive: bool,
    ) -> None:
        block_count = max(_ceil_div(count, _PREFIX_BLOCK_SIZE), 1)
        sums_view = self._slice(block_sums, block_count, scratch_offset)
        with debug_group(command_encoder, f"prefix.scan[{count}]", _DEBUG_COLOR):
            self.k_prefix_scan.dispatch(
                thread_count=spy.uint3(block_count * (_PREFIX_BLOCK_SIZE // 2), 1, 1),
                command_encoder=command_encoder,
                vars={
                    "g_PrefixInput": input_tensor,
                    "g_PrefixOutput": output_tensor,
                    "g_PrefixBlockSums": sums_view,
                    "g_Count": int(count),
                    "g_Exclusive": 1 if exclusive else 0,
                },
            )
        if block_count <= 1:
            return
        offsets_view = self._slice(block_offsets, block_count, scratch_offset)
        self._prefix_recursive(command_encoder, sums_view, offsets_view, block_sums, block_offsets, block_count, scratch_offset + block_count, True)
        with debug_group(command_encoder, f"prefix.add[{count}]", _DEBUG_COLOR):
            self.k_prefix_add.dispatch(
                thread_count=spy.uint3(count, 1, 1),
                command_encoder=command_encoder,
                vars={
                    "g_PrefixOutput": output_tensor,
                    "g_PrefixBlockOffsets": offsets_view,
                    "g_Count": int(count),
                },
            )

    def prefix_sum_uint32(
        self,
        command_encoder: spy.CommandEncoder,
        input_tensor: spy.Tensor,
        output_tensor: spy.Tensor,
        block_sums: spy.Tensor,
        block_offsets: spy.Tensor,
        total_out: spy.Tensor,
        count: int,
        exclusive: bool = True,
    ) -> None:
        if count <= 0:
            total_out.clear(command_encoder=command_encoder)
            return
        with debug_group(command_encoder, f"prefix_sum_uint32[{count}]", _DEBUG_COLOR):
            self._prefix_recursive(command_encoder, input_tensor, output_tensor, block_sums, block_offsets, count, 0, exclusive)
            self.k_prefix_total.dispatch(
                thread_count=spy.uint3(1, 1, 1),
                command_encoder=command_encoder,
                vars={
                    "g_PrefixInput": input_tensor,
                    "g_PrefixOutput": output_tensor,
                    "g_TotalOut": total_out,
                    "g_Count": int(count),
                    "g_Exclusive": 1 if exclusive else 0,
                },
            )

    def radix_sort_uint32(
        self,
        command_encoder: spy.CommandEncoder,
        keys_in: spy.Tensor,
        values_in: spy.Tensor,
        keys_out: spy.Tensor,
        values_out: spy.Tensor,
        histogram: spy.Tensor,
        histogram_prefix: spy.Tensor,
        prefix_block_sums: spy.Tensor,
        prefix_block_offsets: spy.Tensor,
        total_out: spy.Tensor,
        count: int,
        start_bit: int = 0,
        bit_count: int = 32,
    ) -> bool:
        if not 0 <= start_bit <= 31 or not 1 <= bit_count <= 32 or start_bit + bit_count > 32:
            raise ValueError("Invalid radix bit range.")
        if count <= 0:
            total_out.clear(command_encoder=command_encoder)
            return True

        num_groups = max(_ceil_div(count, _RADIX_GROUP_SIZE), 1)
        hist_count = num_groups * _RADIX_BIN_COUNT
        passes = _ceil_div(bit_count, 8)
        src_keys, src_values = keys_in, values_in
        with debug_group(command_encoder, f"radix_sort_uint32[{count}]", _DEBUG_COLOR):
            for pass_idx in range(passes):
                shift = start_bit + pass_idx * 8
                digit_bits = min(8, bit_count - pass_idx * 8)
                digit_mask = (1 << digit_bits) - 1
                use_output = (pass_idx & 1) == 0
                dst_keys = keys_out if use_output else keys_in
                dst_values = values_out if use_output else values_in
                with debug_group(command_encoder, f"radix.pass[{shift}:{digit_bits}]", _DEBUG_COLOR):
                    self.k_histogram.dispatch(
                        thread_count=spy.uint3(num_groups * _RADIX_GROUP_SIZE, 1, 1),
                        command_encoder=command_encoder,
                        vars={
                            "g_KeysIn": src_keys,
                            "g_Histogram": histogram,
                            "g_ElementCount": int(count),
                            "g_NumGroups": int(num_groups),
                            "g_Shift": int(shift),
                            "g_DigitMask": int(digit_mask),
                        },
                    )
                    self.prefix_sum_uint32(command_encoder, histogram, histogram_prefix, prefix_block_sums, prefix_block_offsets, total_out, hist_count, True)
                    self.k_scatter.dispatch(
                        thread_count=spy.uint3(num_groups * _RADIX_GROUP_SIZE, 1, 1),
                        command_encoder=command_encoder,
                        vars={
                            "g_KeysIn": src_keys,
                            "g_ValuesIn": src_values,
                            "g_KeysOut": dst_keys,
                            "g_ValuesOut": dst_values,
                            "g_HistogramOffsets": histogram_prefix,
                            "g_ElementCount": int(count),
                            "g_NumGroups": int(num_groups),
                            "g_Shift": int(shift),
                            "g_DigitMask": int(digit_mask),
                        },
                    )
                src_keys, src_values = dst_keys, dst_values
        return (passes & 1) == 1
