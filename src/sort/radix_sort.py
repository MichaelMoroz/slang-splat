from __future__ import annotations

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


class GPURadixSort:
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
        self.buffers: dict[tuple[int, int], dict[str, object]] = {}
        self.indirect_args: spy.Buffer | None = None

    def level_size(self, value: int, level: int) -> int:
        for _ in range(level):
            value = (value + BLOCK_SIZE - 1) // BLOCK_SIZE
        return value

    def level_offset(self, total_size: int, level: int) -> int:
        offset = 0
        size = total_size
        for lvl in range(level):
            offset += (size + 1) // 2 if lvl == 0 else size
            size = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
        return offset

    def ensure_buffers(self, n: int) -> dict[str, object]:
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
        key = (n, num_groups)
        if key in self.buffers:
            return self.buffers[key]
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
        copy_usage = usage | spy.BufferUsage.copy_source
        mk = lambda size: self.device.create_buffer(size=max(size, 1) * 4, usage=copy_usage)
        self.buffers[key] = {
            "keys": [mk(n), mk(n)],
            "values": [mk(n), mk(n)],
            "histogram": self.device.create_buffer(size=max(packed_hist_n, 1) * 4, usage=usage),
            "prefix": self.device.create_buffer(size=max(total_prefix_size, 1) * 4, usage=usage),
            "num_groups": num_groups,
            "packed_hist_n": packed_hist_n,
            "total_n": total_n,
            "num_levels": num_levels,
        }
        return self.buffers[key]

    def ensure_indirect_args(self) -> spy.Buffer:
        if self.indirect_args is None:
            usage = (
                spy.BufferUsage.shader_resource
                | spy.BufferUsage.unordered_access
                | spy.BufferUsage.indirect_argument
            )
            self.indirect_args = self.device.create_buffer(size=22 * 4, usage=usage)
        return self.indirect_args

    def compute_indirect_args_from_buffer_dispatch(
        self, encoder: spy.CommandEncoder, count_buffer: spy.Buffer, count_offset: int, args_buffer: spy.Buffer
    ) -> None:
        encoder.push_debug_group("Compute Indirect Args", debug_color(0))
        self.compute_args_from_buffer.dispatch(
            thread_count=spy.uint3(1, 1, 1),
            vars={
                "g_ElementCountBuffer": count_buffer,
                "g_elementCountOffset": count_offset,
                "g_IndirectArgs": args_buffer,
            },
            command_encoder=encoder,
        )
        encoder.pop_debug_group()

    def sort_key_values(
        self,
        encoder: spy.CommandEncoder,
        keys_buffer: spy.Buffer,
        values_buffer: spy.Buffer,
        n: int,
        max_bits: int = 32,
    ) -> None:
        if n <= 0:
            return
        encoder.push_debug_group("Radix Sort", debug_color(1))
        working_buffers = self.ensure_buffers(n)
        args_buffer = self.ensure_indirect_args()
        self.compute_args.dispatch(
            thread_count=spy.uint3(1, 1, 1),
            vars={"g_elementCount": n, "g_IndirectArgs": args_buffer},
            command_encoder=encoder,
        )
        self.sort_key_values_indirect(
            encoder, keys_buffer, values_buffer, n, args_buffer, working_buffers, max_bits=max_bits
        )
        encoder.pop_debug_group()

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
        num_groups = int(working_buffers["num_groups"])
        total_n = int(working_buffers["total_n"])
        num_levels = int(working_buffers["num_levels"])

        def indirect(offset: int) -> spy.BufferOffsetPair:
            return spy.BufferOffsetPair(args_buffer, offset * 4)

        def set_common(cursor: spy.ShaderCursor, shift: int) -> None:
            cursor.g_shift = shift
            if use_params_buffer:
                cursor.g_useParamsBuffer = 1
                cursor.g_ParamsBuffer = args_buffer
                cursor.g_elementCount = n
                cursor.g_numGroups = num_groups
            else:
                cursor.g_useParamsBuffer = 0
                cursor.g_elementCount = n
                cursor.g_numGroups = num_groups

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
            for level in range(num_levels):
                current_size = self.level_size(total_n, level)
                prev_offset = self.level_offset(total_n, level - 1) if level > 0 else 0
                current_offset = self.level_offset(total_n, level) if level > 0 else 0
                prev_packed = 1 if level == 1 else 0
                prev_level_size = self.level_size(total_n, level - 1) if level > 1 else total_n
                with encoder.begin_compute_pass() as compute_pass:
                    compute_pass.push_debug_group(f"Prefix Level {level}", debug_color(3 + level))
                    cursor = spy.ShaderCursor(compute_pass.bind_pipeline(self.prefix_block))
                    set_common(cursor, shift)
                    cursor.g_totalN = total_n
                    cursor.g_levelSize = current_size if level > 0 else total_n
                    cursor.g_levelOffset = current_offset
                    cursor.g_prevLevelOffset = prev_offset
                    cursor.g_prevLevelSize = prev_level_size
                    cursor.g_isFirstLevel = 1 if level == 0 else 0
                    cursor.g_packedHistN = int(working_buffers["packed_hist_n"])
                    cursor.g_prevLevelPacked = prev_packed
                    cursor.g_Histogram = working_buffers["histogram"]
                    cursor.g_Prefix = working_buffers["prefix"]
                    compute_pass.dispatch_compute_indirect(indirect(3 + level * 3))
                    compute_pass.pop_debug_group()
            with encoder.begin_compute_pass() as compute_pass:
                compute_pass.push_debug_group("Scatter", debug_color(8))
                cursor = spy.ShaderCursor(compute_pass.bind_pipeline(self.scatter))
                set_common(cursor, shift)
                cursor.g_totalN = total_n
                cursor.g_numLevels = num_levels
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
