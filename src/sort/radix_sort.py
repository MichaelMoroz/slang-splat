from __future__ import annotations

from functools import reduce
from pathlib import Path

import numpy as np
import slangpy as spy

from ..common import ROOT, debug_color

SHADER_DIR = ROOT / "shaders" / "radix_sort"
GROUP_SIZE = 128
HISTOGRAM_SIZE = 256
PACKED_HIST_SIZE = 64
BLOCK_SIZE = 256
BITS_PER_PASS = 8
MAX_PREFIX_LEVELS = 4
BUILD_RANGE_ARGS_OFFSET = 22
INDIRECT_ARGS_UINT_COUNT = 25


class GPURadixSort:
    BUILD_RANGE_ARGS_OFFSET = BUILD_RANGE_ARGS_OFFSET
    level_size = lambda self, value, level: reduce(lambda current, _: (current + BLOCK_SIZE - 1) // BLOCK_SIZE, range(level), value)
    level_offset = lambda self, total_size, level: reduce(lambda state, lvl: (state[0] + ((state[1] + 1) // 2 if lvl == 0 else state[1]), (state[1] + BLOCK_SIZE - 1) // BLOCK_SIZE), range(level), (0, total_size))[0]
    ensure_indirect_args = lambda self: self.indirect_args if self.indirect_args is not None else (setattr(self, "indirect_args", self.device.create_buffer(size=INDIRECT_ARGS_UINT_COUNT * 4, usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.indirect_argument)) or self.indirect_args)
    compute_indirect_args_from_buffer_dispatch = lambda self, encoder, count_buffer, count_offset, max_element_count, args_buffer: (encoder.push_debug_group("Compute Indirect Args", debug_color(0)), self.compute_args_from_buffer.dispatch(thread_count=spy.uint3(1, 1, 1), vars={"g_ElementCountBuffer": count_buffer, "g_elementCountOffset": count_offset, "g_maxElementCount": int(max_element_count), "g_IndirectArgs": args_buffer}, command_encoder=encoder), encoder.pop_debug_group())
    sort_key_values = lambda self, encoder, keys_buffer, values_buffer, n, max_bits=32: None if n <= 0 else (encoder.push_debug_group("Radix Sort", debug_color(1)), (lambda working_buffers, args_buffer: (self.compute_args.dispatch(thread_count=spy.uint3(1, 1, 1), vars={"g_elementCount": n, "g_maxElementCount": n, "g_IndirectArgs": args_buffer}, command_encoder=encoder), self.sort_key_values_indirect(encoder, keys_buffer, values_buffer, n, args_buffer, working_buffers, max_bits=max_bits)))(self.ensure_buffers(n), self.ensure_indirect_args()), encoder.pop_debug_group())

    def __init__(self, device: spy.Device):
        self.device = device
        load = device.load_program
        mk_kernel = lambda name, entry: device.create_compute_kernel(load(str(SHADER_DIR / name), [entry]))
        mk_pipe = lambda name, entry: device.create_compute_pipeline(load(str(SHADER_DIR / name), [entry]))
        self.compute_args = mk_kernel("compute_indirect_args.slang", "csComputeIndirectArgs")
        self.compute_args_from_buffer = mk_kernel(
            "compute_indirect_args_from_buffer.slang", "csComputeIndirectArgsFromBuffer"
        )
        self.histogram = mk_pipe("histogram.slang", "csHistogram")
        self.prefix_block = mk_pipe("prefix_block.slang", "csPrefixBlock")
        self.scatter = mk_pipe("scatter.slang", "csScatter")
        self._capacity_n = 0
        self._buffers: dict[str, object] | None = None
        self.indirect_args: spy.Buffer | None = None

    def _layout(self, n: int) -> dict[str, int]:
        num_groups = max((n + GROUP_SIZE - 1) // GROUP_SIZE, 1)
        packed_hist_n = num_groups * PACKED_HIST_SIZE
        total_n = num_groups * HISTOGRAM_SIZE
        num_levels = 0
        level_size = total_n
        while level_size > 1:
            num_levels += 1
            level_size = (level_size + BLOCK_SIZE - 1) // BLOCK_SIZE
        num_levels += 1
        last_level_size = self.level_size(total_n, num_levels - 1)
        total_prefix_size = self.level_offset(total_n, num_levels - 1) + last_level_size
        return {
            "num_groups": num_groups,
            "packed_hist_n": packed_hist_n,
            "total_n": total_n,
            "num_levels": num_levels,
            "total_prefix_size": total_prefix_size,
        }

    def ensure_buffers(self, n: int) -> dict[str, object]:
        if self._buffers is not None and n <= self._capacity_n:
            return self._buffers
        old_capacity = max(self._capacity_n, 1)
        grow_n = max(n, old_capacity + old_capacity // 2)
        layout = self._layout(grow_n)
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
        copy_usage = usage | spy.BufferUsage.copy_source
        mk = lambda size: self.device.create_buffer(size=max(size, 1) * 4, usage=copy_usage)
        self._buffers = {
            "keys": [mk(grow_n), mk(grow_n)],
            "values": [mk(grow_n), mk(grow_n)],
            "histogram": self.device.create_buffer(size=max(layout["packed_hist_n"], 1) * 4, usage=usage),
            "prefix": self.device.create_buffer(size=max(layout["total_prefix_size"], 1) * 4, usage=usage),
        }
        self._capacity_n = grow_n
        return self._buffers

    def sort_key_values_from_count_buffer(
        self,
        encoder: spy.CommandEncoder,
        keys_buffer: spy.Buffer,
        values_buffer: spy.Buffer,
        count_buffer: spy.Buffer,
        count_offset: int,
        max_count: int,
        max_bits: int = 32,
    ) -> spy.Buffer:
        if max_count <= 0:
            raise ValueError("max_count must be positive")
        encoder.push_debug_group("Radix Sort (Count Buffer)", debug_color(11))
        working_buffers = self.ensure_buffers(max_count)
        args_buffer = self.ensure_indirect_args()
        self.compute_indirect_args_from_buffer_dispatch(
            encoder=encoder,
            count_buffer=count_buffer,
            count_offset=count_offset,
            max_element_count=max_count,
            args_buffer=args_buffer,
        )
        self.sort_key_values_indirect(
            encoder=encoder,
            keys_buffer=keys_buffer,
            values_buffer=values_buffer,
            n=max_count,
            args_buffer=args_buffer,
            working_buffers=working_buffers,
            max_bits=max_bits,
            use_params_buffer=True,
        )
        encoder.pop_debug_group()
        return args_buffer

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
    ) -> None:
        if working_buffers is None:
            working_buffers = self.ensure_buffers(n)
        layout = self._layout(n)
        num_groups = layout["num_groups"]
        total_n = layout["total_n"]
        num_levels = layout["num_levels"]

        indirect = lambda offset: spy.BufferOffsetPair(args_buffer, offset * 4)
        set_common = lambda cursor, shift: (
            setattr(cursor, "g_shift", shift),
            setattr(cursor, "g_useParamsBuffer", 1 if use_params_buffer else 0),
            setattr(cursor, "g_ParamsBuffer", args_buffer if use_params_buffer else None),
            setattr(cursor, "g_elementCount", 0 if use_params_buffer else n),
            setattr(cursor, "g_numGroups", 0 if use_params_buffer else num_groups),
        )

        passes = max((max_bits + BITS_PER_PASS - 1) // BITS_PER_PASS, 1)
        keys_in, keys_out = keys_buffer, working_buffers["keys"][0]
        values_in, values_out = values_buffer, working_buffers["values"][0]
        for pass_index in range(passes):
            shift = pass_index * BITS_PER_PASS
            encoder.push_debug_group(f"Pass {pass_index} (bits {shift}-{shift + BITS_PER_PASS - 1})", debug_color(10))
            with encoder.begin_compute_pass() as compute_pass:
                compute_pass.push_debug_group("Histogram", debug_color(2))
                cursor = spy.ShaderCursor(compute_pass.bind_pipeline(self.histogram))
                set_common(cursor, shift)
                cursor.g_Keys = keys_in
                cursor.g_Histogram = working_buffers["histogram"]
                compute_pass.dispatch_compute_indirect(indirect(0))
                compute_pass.pop_debug_group()
            for level in range(MAX_PREFIX_LEVELS):
                with encoder.begin_compute_pass() as compute_pass:
                    compute_pass.push_debug_group(f"Prefix Level {level}", debug_color(3 + level))
                    cursor = spy.ShaderCursor(compute_pass.bind_pipeline(self.prefix_block))
                    set_common(cursor, shift)
                    cursor.g_level = level
                    cursor.g_Histogram = working_buffers["histogram"]
                    cursor.g_Prefix = working_buffers["prefix"]
                    compute_pass.dispatch_compute_indirect(indirect(3 + level * 3))
                    compute_pass.pop_debug_group()
            with encoder.begin_compute_pass() as compute_pass:
                compute_pass.push_debug_group("Scatter", debug_color(8))
                cursor = spy.ShaderCursor(compute_pass.bind_pipeline(self.scatter))
                set_common(cursor, shift)
                cursor.g_totalN = 0 if use_params_buffer else total_n
                cursor.g_numLevels = 0 if use_params_buffer else num_levels
                cursor.g_KeysIn = keys_in
                cursor.g_ValuesIn = values_in
                cursor.g_Prefix = working_buffers["prefix"]
                cursor.g_KeysOut = keys_out
                cursor.g_ValuesOut = values_out
                compute_pass.dispatch_compute_indirect(indirect(15))
                compute_pass.pop_debug_group()
            encoder.pop_debug_group()
            keys_in, keys_out = keys_out, keys_in
            values_in, values_out = values_out, values_in

        if passes % 2 == 1:
            encoder.push_debug_group("Final Copy", debug_color(9))
            encoder.copy_buffer(keys_buffer, 0, keys_in, 0, n * 4)
            encoder.copy_buffer(values_buffer, 0, values_in, 0, n * 4)
            encoder.pop_debug_group()


def sort_numpy(
    device: spy.Device, keys: np.ndarray, values: np.ndarray, max_bits: int = 32
) -> tuple[np.ndarray, np.ndarray]:
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
    keys_buffer = device.create_buffer(size=n * 4, usage=usage)
    values_buffer = device.create_buffer(size=n * 4, usage=usage)
    keys_buffer.copy_from_numpy(keys.astype(np.uint32))
    values_buffer.copy_from_numpy(values.astype(np.uint32))
    encoder = device.create_command_encoder()
    sorter.sort_key_values(encoder, keys_buffer, values_buffer, n, max_bits=max_bits)
    device.submit_command_buffer(encoder.finish())
    device.wait()
    sorted_keys = np.frombuffer(keys_buffer.to_numpy().tobytes(), dtype=np.uint32)[:n].copy()
    sorted_values = np.frombuffer(values_buffer.to_numpy().tobytes(), dtype=np.uint32)[:n].copy()
    return sorted_keys, sorted_values
