from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import slangpy as spy

from ..utility import INDIRECT_BUFFER_USAGE, ROOT, alloc_buffer, debug_color, debug_group, dispatch_indirect, grow_capacity, load_compute_items

SHADER_DIR = ROOT / "shaders" / "utility" / "radix_sort"
PREFIX_SHADER_PATH = str(ROOT / "shaders" / "utility" / "prefix_sum" / "prefix_sum.slang")
GROUP_SIZE = 512
HISTOGRAM_SIZE = 256
PACKED_HIST_SIZE = HISTOGRAM_SIZE // 2
PREFIX_THREADS = 256
PREFIX_BLOCK_SIZE = PREFIX_THREADS * 2
BITS_PER_PASS = 8
MAX_PREFIX_LEVELS = 4
INDIRECT_DISPATCH_ARG_STRIDE = 3
HISTOGRAM_ARGS_OFFSET = 0
PREFIX_L0_ARGS_OFFSET = HISTOGRAM_ARGS_OFFSET + INDIRECT_DISPATCH_ARG_STRIDE
SCATTER_ARGS_OFFSET = PREFIX_L0_ARGS_OFFSET + MAX_PREFIX_LEVELS * INDIRECT_DISPATCH_ARG_STRIDE
BUILD_RANGE_ARGS_OFFSET = SCATTER_ARGS_OFFSET + INDIRECT_DISPATCH_ARG_STRIDE
PARAM_ELEMENT_COUNT = BUILD_RANGE_ARGS_OFFSET + INDIRECT_DISPATCH_ARG_STRIDE
PARAM_NUM_GROUPS = PARAM_ELEMENT_COUNT + 1
PARAM_TOTAL_N = PARAM_NUM_GROUPS + 1
PARAM_NUM_LEVELS = PARAM_TOTAL_N + 1
INDIRECT_ARGS_UINT_COUNT = PARAM_NUM_LEVELS + 1


@dataclass(frozen=True, slots=True)
class RadixSortResult:
    args_buffer: spy.Buffer
    keys_buffer: spy.Buffer
    values_buffer: spy.Buffer


class GPURadixSort:
    BUILD_RANGE_ARGS_OFFSET = BUILD_RANGE_ARGS_OFFSET
    PARAM_ELEMENT_COUNT = PARAM_ELEMENT_COUNT

    def __init__(self, device: spy.Device):
        self.device = device
        for name, item in load_compute_items(
            device,
            {
                "compute_args": ("kernel", SHADER_DIR / "compute_indirect_args.slang", "csComputeIndirectArgs"),
                "compute_args_from_buffer": ("kernel", SHADER_DIR / "compute_indirect_args_from_buffer.slang", "csComputeIndirectArgsFromBuffer"),
                "histogram": ("pipeline", SHADER_DIR / "histogram.slang", "csRadixHistogram"),
                "prefix_level": ("pipeline", SHADER_DIR / "prefix_block.slang", "csRadixPrefixLevel"),
                "prefix_add": ("pipeline", PREFIX_SHADER_PATH, "csPrefixAddOffsets"),
                "scatter": ("pipeline", SHADER_DIR / "scatter.slang", "csRadixScatter"),
            },
        ).items():
            setattr(self, name, item)
        self._capacity_n = 0
        self._buffers: dict[str, object] | None = None
        self.indirect_args: spy.Buffer | None = None
        self._rw_usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
        self._copy_usage = self._rw_usage | spy.BufferUsage.copy_source

    @staticmethod
    def _ceil_div(value: int, divisor: int) -> int:
        return (int(value) + int(divisor) - 1) // int(divisor)

    def _num_groups(self, count: int) -> int:
        return max(self._ceil_div(max(int(count), 1), GROUP_SIZE), 1)

    def _total_n(self, count: int) -> int:
        return self._num_groups(count) * HISTOGRAM_SIZE

    def _num_levels(self, count: int) -> int:
        levels = 0
        size = self._total_n(count)
        while levels < MAX_PREFIX_LEVELS and size > 1:
            levels += 1
            size = self._ceil_div(size, PREFIX_THREADS)
        return max(levels, 1)

    def _prefix_elements(self, count: int) -> int:
        size = self._total_n(count)
        used = 0
        while True:
            used += size
            if size <= PREFIX_THREADS:
                return max(used, 1)
            size = self._ceil_div(size, PREFIX_THREADS)

    def _layout(self, count: int) -> dict[str, int]:
        num_groups = self._num_groups(count)
        return {
            "num_groups": num_groups,
            "packed_hist_n": max(num_groups * PACKED_HIST_SIZE, PACKED_HIST_SIZE),
            "total_n": max(num_groups * HISTOGRAM_SIZE, HISTOGRAM_SIZE),
            "num_levels": self._num_levels(count),
            "prefix_n": self._prefix_elements(count),
        }

    def ensure_buffers(self, n: int) -> dict[str, object]:
        if self._buffers is not None and n <= self._capacity_n:
            return self._buffers
        grow_n = grow_capacity(n, self._capacity_n)
        layout = self._layout(grow_n)
        self._buffers = {
            "keys": [
                alloc_buffer(self.device, size=max(grow_n, 1) * 4, usage=self._copy_usage),
                alloc_buffer(self.device, size=max(grow_n, 1) * 4, usage=self._copy_usage),
            ],
            "values": [
                alloc_buffer(self.device, size=max(grow_n, 1) * 4, usage=self._copy_usage),
                alloc_buffer(self.device, size=max(grow_n, 1) * 4, usage=self._copy_usage),
            ],
            "histogram": alloc_buffer(self.device, size=max(layout["packed_hist_n"], 1) * 4, usage=self._rw_usage),
            "prefix": alloc_buffer(self.device, size=max(layout["prefix_n"], 1) * 4, usage=self._rw_usage),
        }
        self._capacity_n = grow_n
        return self._buffers

    def ensure_indirect_args(self) -> spy.Buffer:
        if self.indirect_args is None:
            self.indirect_args = alloc_buffer(self.device, size=INDIRECT_ARGS_UINT_COUNT * 4, usage=INDIRECT_BUFFER_USAGE)
        return self.indirect_args

    def compute_indirect_args_from_buffer_dispatch(
        self,
        encoder: spy.CommandEncoder,
        count_buffer: spy.Buffer,
        count_offset: int,
        max_element_count: int,
        args_buffer: spy.Buffer,
    ) -> None:
        with debug_group(encoder, "Compute Indirect Args", debug_color(0)):
            self.compute_args_from_buffer.dispatch(
                thread_count=spy.uint3(1, 1, 1),
                vars={
                    "g_CountBuffer": count_buffer,
                    "g_CountOffset": int(count_offset),
                    "g_MaxElementCount": int(max_element_count),
                    "g_IndirectArgs": args_buffer,
                },
                command_encoder=encoder,
            )

    def sort_key_values(
        self,
        encoder: spy.CommandEncoder,
        keys_buffer: spy.Buffer,
        values_buffer: spy.Buffer,
        n: int,
        max_bits: int = 32,
        copy_result_back: bool = True,
    ) -> RadixSortResult:
        if n <= 0:
            args_buffer = self.ensure_indirect_args()
            return RadixSortResult(args_buffer=args_buffer, keys_buffer=keys_buffer, values_buffer=values_buffer)
        with debug_group(encoder, "Radix Sort", debug_color(1)):
            working_buffers = self.ensure_buffers(n)
            args_buffer = self.ensure_indirect_args()
            self.compute_args.dispatch(
                thread_count=spy.uint3(1, 1, 1),
                vars={
                    "g_ElementCount": int(n),
                    "g_MaxElementCount": int(n),
                    "g_IndirectArgs": args_buffer,
                },
                command_encoder=encoder,
            )
            final_keys, final_values = self.sort_key_values_indirect(
                encoder=encoder,
                keys_buffer=keys_buffer,
                values_buffer=values_buffer,
                n=n,
                args_buffer=args_buffer,
                working_buffers=working_buffers,
                max_bits=max_bits,
                copy_result_back=copy_result_back,
            )
            return RadixSortResult(args_buffer=args_buffer, keys_buffer=final_keys, values_buffer=final_values)

    def sort_key_values_from_count_buffer(
        self,
        encoder: spy.CommandEncoder,
        keys_buffer: spy.Buffer,
        values_buffer: spy.Buffer,
        count_buffer: spy.Buffer,
        count_offset: int,
        max_count: int,
        max_bits: int = 32,
        copy_result_back: bool = True,
    ) -> RadixSortResult:
        if max_count <= 0:
            args_buffer = self.ensure_indirect_args()
            return RadixSortResult(args_buffer=args_buffer, keys_buffer=keys_buffer, values_buffer=values_buffer)
        with debug_group(encoder, "Radix Sort (Count Buffer)", debug_color(11)):
            working_buffers = self.ensure_buffers(max_count)
            args_buffer = self.ensure_indirect_args()
            self.compute_indirect_args_from_buffer_dispatch(
                encoder=encoder,
                count_buffer=count_buffer,
                count_offset=count_offset,
                max_element_count=max_count,
                args_buffer=args_buffer,
            )
            final_keys, final_values = self.sort_key_values_indirect(
                encoder=encoder,
                keys_buffer=keys_buffer,
                values_buffer=values_buffer,
                n=max_count,
                args_buffer=args_buffer,
                working_buffers=working_buffers,
                max_bits=max_bits,
                use_params_buffer=True,
                copy_result_back=copy_result_back,
            )
            return RadixSortResult(args_buffer=args_buffer, keys_buffer=final_keys, values_buffer=final_values)

    def sort_key_values_indirect(
        self,
        encoder: spy.CommandEncoder,
        keys_buffer: spy.Buffer,
        values_buffer: spy.Buffer,
        n: int,
        args_buffer: spy.Buffer,
        working_buffers: dict[str, object] | None = None,
        max_bits: int = 32,
        use_params_buffer: bool = False,
        copy_result_back: bool = True,
    ) -> tuple[spy.Buffer, spy.Buffer]:
        if working_buffers is None:
            working_buffers = self.ensure_buffers(n)
        layout = self._layout(n)
        num_groups = layout["num_groups"]
        total_n = layout["total_n"]
        num_levels = layout["num_levels"]
        passes = max((int(max_bits) + BITS_PER_PASS - 1) // BITS_PER_PASS, 1)
        keys_in, keys_out = keys_buffer, working_buffers["keys"][0]
        values_in, values_out = values_buffer, working_buffers["values"][0]
        histogram = working_buffers["histogram"]
        prefix = working_buffers["prefix"]

        for pass_index in range(passes):
            shift = pass_index * BITS_PER_PASS
            digit_bits = min(BITS_PER_PASS, int(max_bits) - pass_index * BITS_PER_PASS)
            digit_mask = (1 << digit_bits) - 1
            with debug_group(encoder, f"Pass {pass_index} (bits {shift}-{shift + digit_bits - 1})", debug_color(10)):
                dispatch_indirect(
                    pipeline=self.histogram,
                    args_buffer=args_buffer,
                    vars={
                        "g_KeysIn": keys_in,
                        "g_Histogram": histogram,
                        "g_Shift": int(shift),
                        "g_DigitMask": int(digit_mask),
                        "g_RadixParamsBuffer": args_buffer,
                        "g_UseRadixParams": 1 if use_params_buffer else 0,
                        "g_ElementCount": 0 if use_params_buffer else int(n),
                        "g_NumGroups": 0 if use_params_buffer else int(num_groups),
                    },
                    command_encoder=encoder,
                    arg_offset=HISTOGRAM_ARGS_OFFSET,
                    debug_label="Histogram",
                    debug_color_index=2,
                )
                for level in range(num_levels):
                    dispatch_indirect(
                        pipeline=self.prefix_level,
                        args_buffer=args_buffer,
                        vars={
                            "g_Histogram": histogram,
                            "g_HistogramOffsets": prefix,
                            "g_Shift": int(level),
                            "g_RadixParamsBuffer": args_buffer,
                            "g_UseRadixParams": 1 if use_params_buffer else 0,
                            "g_NumGroups": 0 if use_params_buffer else int(num_groups),
                        },
                        command_encoder=encoder,
                        arg_offset=PREFIX_L0_ARGS_OFFSET + level * INDIRECT_DISPATCH_ARG_STRIDE,
                        debug_label=f"Prefix Level {level}",
                        debug_color_index=3 + level,
                    )
                for level in range(num_levels - 1, 0, -1):
                    dispatch_indirect(
                        pipeline=self.prefix_add,
                        args_buffer=args_buffer,
                        vars={
                            "g_HistogramOffsets": prefix,
                            "g_RadixParamsBuffer": args_buffer,
                            "g_UsePrefixParams": 0,
                            "g_UseRadixPrefixAdd": 1,
                            "g_Level": int(level),
                        },
                        command_encoder=encoder,
                        arg_offset=PREFIX_L0_ARGS_OFFSET + level * INDIRECT_DISPATCH_ARG_STRIDE,
                        debug_label=f"Prefix Add Level {level}",
                        debug_color_index=7 + level,
                    )
                dispatch_indirect(
                    pipeline=self.scatter,
                    args_buffer=args_buffer,
                    vars={
                        "g_KeysIn": keys_in,
                        "g_ValuesIn": values_in,
                        "g_KeysOut": keys_out,
                        "g_ValuesOut": values_out,
                        "g_HistogramOffsets": prefix,
                        "g_Shift": int(shift),
                        "g_DigitMask": int(digit_mask),
                        "g_RadixParamsBuffer": args_buffer,
                        "g_UseRadixParams": 1 if use_params_buffer else 0,
                        "g_ElementCount": 0 if use_params_buffer else int(n),
                        "g_NumGroups": 0 if use_params_buffer else int(num_groups),
                        "g_TotalN": 0 if use_params_buffer else int(total_n),
                        "g_NumLevels": 0 if use_params_buffer else int(num_levels),
                    },
                    command_encoder=encoder,
                    arg_offset=SCATTER_ARGS_OFFSET,
                    debug_label="Scatter",
                    debug_color_index=8,
                )
            keys_in, keys_out = keys_out, keys_in
            values_in, values_out = values_out, values_in

        if passes % 2 == 1 and copy_result_back:
            with debug_group(encoder, "Final Copy", debug_color(9)):
                encoder.copy_buffer(keys_buffer, 0, keys_in, 0, n * 4)
                encoder.copy_buffer(values_buffer, 0, values_in, 0, n * 4)
            return keys_buffer, values_buffer
        return keys_in, values_in


def sort_numpy(device: spy.Device, keys: np.ndarray, values: np.ndarray, max_bits: int = 32) -> tuple[np.ndarray, np.ndarray]:
    n = int(len(keys))
    if n <= 0:
        return keys.copy(), values.copy()
    sorter = GPURadixSort(device)
    usage = (
        spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access
        | spy.BufferUsage.copy_destination
        | spy.BufferUsage.copy_source
    )
    keys_buffer = alloc_buffer(device, size=n * 4, usage=usage)
    values_buffer = alloc_buffer(device, size=n * 4, usage=usage)
    keys_buffer.copy_from_numpy(np.ascontiguousarray(keys.astype(np.uint32, copy=False)))
    values_buffer.copy_from_numpy(np.ascontiguousarray(values.astype(np.uint32, copy=False)))
    encoder = device.create_command_encoder()
    sorter.sort_key_values(encoder, keys_buffer, values_buffer, n, max_bits=max_bits)
    device.submit_command_buffer(encoder.finish())
    device.wait()
    sorted_keys = np.frombuffer(keys_buffer.to_numpy().tobytes(), dtype=np.uint32)[:n].copy()
    sorted_values = np.frombuffer(values_buffer.to_numpy().tobytes(), dtype=np.uint32)[:n].copy()
    return sorted_keys, sorted_values
