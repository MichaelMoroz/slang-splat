from __future__ import annotations

import slangpy as spy

from ..common import ROOT, debug_region

SHADER_DIR = ROOT / "shaders" / "utility" / "prefix_sum"
PREFIX_THREADS = 256
PREFIX_BLOCK_SIZE = PREFIX_THREADS * 2
PREFIX_MAX_LEVELS = 8
INDIRECT_DISPATCH_ARG_STRIDE = 3
PREFIX_SCAN_ARGS_OFFSET = 0
PREFIX_ADD_ARGS_OFFSET = PREFIX_SCAN_ARGS_OFFSET + PREFIX_MAX_LEVELS * INDIRECT_DISPATCH_ARG_STRIDE
PREFIX_PARAM_COUNT_OFFSET = PREFIX_ADD_ARGS_OFFSET + PREFIX_MAX_LEVELS * INDIRECT_DISPATCH_ARG_STRIDE
PREFIX_INDIRECT_ARGS_UINT_COUNT = PREFIX_PARAM_COUNT_OFFSET + 1


class GPUPrefixSum:
    def __init__(self, device: spy.Device):
        self.device = device
        shader_path = str(SHADER_DIR / "prefix_sum.slang")
        load = device.load_program
        self.clear_uint = device.create_compute_kernel(load(shader_path, ["csClearUIntBuffer"]))
        self.scan_blocks = device.create_compute_kernel(load(shader_path, ["csPrefixScanBlocks"]))
        self.add_offsets = device.create_compute_kernel(load(shader_path, ["csPrefixAddOffsets"]))
        self.write_total_kernel = device.create_compute_kernel(load(shader_path, ["csPrefixWriteTotal"]))
        self.compute_dispatch_args_from_buffer_kernel = device.create_compute_kernel(load(shader_path, ["csComputeDispatchArgsFromBuffer"]))
        self.compute_prefix_args_from_buffer_kernel = device.create_compute_kernel(load(shader_path, ["csComputePrefixIndirectArgsFromBuffer"]))
        self.scan_blocks_pipeline = device.create_compute_pipeline(load(shader_path, ["csPrefixScanBlocks"]))
        self.add_offsets_pipeline = device.create_compute_pipeline(load(shader_path, ["csPrefixAddOffsets"]))
        self._indirect_usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.indirect_argument
        self._rw_usage = (
            spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access
            | spy.BufferUsage.copy_source
            | spy.BufferUsage.copy_destination
        )
        self._scratch_capacity = 0
        self._block_sums: spy.Buffer | None = None
        self._block_offsets: spy.Buffer | None = None
        self._dispatch_args: spy.Buffer | None = None
        self._prefix_args: spy.Buffer | None = None

    @staticmethod
    def prefix_scratch_elements(count: int) -> int:
        total = 1
        size = max(int(count), 1)
        while size > PREFIX_BLOCK_SIZE:
            size = (size + PREFIX_BLOCK_SIZE - 1) // PREFIX_BLOCK_SIZE
            total += size
        return total

    def _level_layout(self, count: int) -> list[tuple[int, int, int]]:
        level_count = max(int(count), 1)
        scratch_offset = 0
        layout: list[tuple[int, int, int]] = []
        while True:
            block_count = max((level_count + PREFIX_BLOCK_SIZE - 1) // PREFIX_BLOCK_SIZE, 1)
            layout.append((level_count, block_count, scratch_offset))
            if block_count <= 1:
                return layout
            scratch_offset += block_count
            level_count = block_count

    def _ensure_scratch_buffers(self, count: int) -> tuple[spy.Buffer, spy.Buffer]:
        required = self.prefix_scratch_elements(count)
        if self._block_sums is not None and self._block_offsets is not None and required <= self._scratch_capacity:
            return self._block_sums, self._block_offsets
        old_capacity = max(self._scratch_capacity, 1)
        capacity = max(required, old_capacity + old_capacity // 2)
        self._block_sums = self.device.create_buffer(size=max(capacity, 1) * 4, usage=self._rw_usage)
        self._block_offsets = self.device.create_buffer(size=max(capacity, 1) * 4, usage=self._rw_usage)
        self._scratch_capacity = capacity
        return self._block_sums, self._block_offsets

    def _ensure_dispatch_args(self) -> spy.Buffer:
        if self._dispatch_args is None:
            self._dispatch_args = self.device.create_buffer(size=4 * 4, usage=self._indirect_usage)
        return self._dispatch_args

    def _ensure_prefix_args(self) -> spy.Buffer:
        if self._prefix_args is None:
            self._prefix_args = self.device.create_buffer(size=PREFIX_INDIRECT_ARGS_UINT_COUNT * 4, usage=self._indirect_usage)
        return self._prefix_args

    def _dispatch_indirect(self, encoder: spy.CommandEncoder, pipeline: spy.ComputePipeline, args_buffer: spy.Buffer, arg_offset: int, vars: dict[str, object]) -> None:
        with encoder.begin_compute_pass() as compute_pass:
            cursor = spy.ShaderCursor(compute_pass.bind_pipeline(pipeline))
            for name, value in vars.items():
                setattr(cursor, name, value)
            compute_pass.dispatch_compute_indirect(spy.BufferOffsetPair(args_buffer, int(arg_offset) * 4))

    def clear_u32(self, encoder: spy.CommandEncoder, buffer: spy.Buffer, element_count: int, clear_value: int = 0) -> None:
        with debug_region(encoder, "Prefix Sum Clear UInt", 100):
            self.clear_uint.dispatch(
                thread_count=spy.uint3(max(int(element_count), 1), 1, 1),
                vars={
                    "g_ClearUIntBuffer": buffer,
                    "g_ClearUIntCount": int(element_count),
                    "g_ClearUIntValue": int(clear_value),
                },
                command_encoder=encoder,
            )

    def dispatch_args_from_count_buffer(
        self,
        encoder: spy.CommandEncoder,
        count_buffer: spy.Buffer,
        count_offset: int,
        max_count: int,
        threads_per_group: int,
    ) -> spy.Buffer:
        args_buffer = self._ensure_dispatch_args()
        self.compute_dispatch_args_from_buffer_kernel.dispatch(
            thread_count=spy.uint3(1, 1, 1),
            vars={
                "g_CountBuffer": count_buffer,
                "g_CountOffset": int(count_offset),
                "g_MaxElementCount": int(max_count),
                "g_DispatchThreadsPerGroup": int(max(threads_per_group, 1)),
                "g_DispatchIndirectArgs": args_buffer,
            },
            command_encoder=encoder,
        )
        return args_buffer

    def _write_total(self, encoder: spy.CommandEncoder, input_buffer: spy.Buffer, output_buffer: spy.Buffer, total_buffer: spy.Buffer, count: int, exclusive: bool) -> None:
        with debug_region(encoder, "Prefix Sum Total", 104):
            self.write_total_kernel.dispatch(
                thread_count=spy.uint3(1, 1, 1),
                vars={
                    "g_PrefixInput": input_buffer,
                    "g_PrefixOutput": output_buffer,
                    "g_TotalOut": total_buffer,
                    "g_Count": int(count),
                    "g_Exclusive": 1 if exclusive else 0,
                    "g_UsePrefixParams": 0,
                    "g_Level": 0,
                    "g_PrefixInputOffset": 0,
                    "g_PrefixOutputOffset": 0,
                },
                command_encoder=encoder,
            )

    def scan_uint(
        self,
        encoder: spy.CommandEncoder,
        input_buffer: spy.Buffer,
        output_buffer: spy.Buffer,
        element_count: int,
        total_buffer: spy.Buffer | None = None,
        exclusive: bool = False,
    ) -> None:
        count = int(element_count)
        block_sums, block_offsets = self._ensure_scratch_buffers(count)
        layout = self._level_layout(count)
        for level, (_, block_count, offset) in enumerate(layout):
            prev_offset = layout[level - 1][2] if level > 0 else 0
            scan_input = input_buffer if level == 0 else block_sums
            scan_output = output_buffer if level == 0 else block_offsets
            with debug_region(encoder, f"Prefix Sum Scan Level {level}", 101 + level):
                self.scan_blocks.dispatch(
                    thread_count=spy.uint3(max(block_count * PREFIX_THREADS, 1), 1, 1),
                    vars={
                        "g_PrefixInput": scan_input,
                        "g_PrefixOutput": scan_output,
                        "g_PrefixBlockSums": block_sums,
                        "g_Count": count,
                        "g_Exclusive": 1 if (exclusive or level > 0) else 0,
                        "g_UsePrefixParams": 0,
                        "g_Level": int(level),
                        "g_PrefixInputOffset": int(prev_offset),
                        "g_PrefixOutputOffset": 0 if level == 0 else int(prev_offset),
                        "g_PrefixBlockSumsOffset": int(offset),
                    },
                    command_encoder=encoder,
                )
        for level in range(len(layout) - 2, -1, -1):
            block_count = layout[level][1]
            offset = layout[level][2]
            prev_offset = layout[level - 1][2] if level > 0 else 0
            add_output = output_buffer if level == 0 else block_offsets
            with debug_region(encoder, f"Prefix Sum Add Level {level}", 120 + level):
                self.add_offsets.dispatch(
                    thread_count=spy.uint3(max((layout[level][0] + PREFIX_THREADS - 1) // PREFIX_THREADS * PREFIX_THREADS, 1), 1, 1),
                    vars={
                        "g_PrefixOutput": add_output,
                        "g_PrefixBlockOffsets": block_offsets,
                        "g_Count": count,
                        "g_Exclusive": 1 if exclusive else 0,
                        "g_UsePrefixParams": 0,
                        "g_UseRadixPrefixAdd": 0,
                        "g_Level": int(level),
                        "g_PrefixOutputOffset": 0 if level == 0 else int(prev_offset),
                        "g_PrefixBlockOffsetsOffset": int(offset),
                    },
                    command_encoder=encoder,
                )
        if total_buffer is not None:
            self._write_total(encoder, input_buffer, output_buffer, total_buffer, count, exclusive)

    def scan_uint_from_count_buffer(
        self,
        encoder: spy.CommandEncoder,
        input_buffer: spy.Buffer,
        output_buffer: spy.Buffer,
        count_buffer: spy.Buffer,
        count_offset: int,
        max_count: int,
        total_buffer: spy.Buffer | None = None,
        exclusive: bool = False,
    ) -> spy.Buffer:
        block_sums, block_offsets = self._ensure_scratch_buffers(max_count)
        args_buffer = self._ensure_prefix_args()
        layout = self._level_layout(max_count)
        self.compute_prefix_args_from_buffer_kernel.dispatch(
            thread_count=spy.uint3(1, 1, 1),
            vars={
                "g_CountBuffer": count_buffer,
                "g_CountOffset": int(count_offset),
                "g_MaxElementCount": int(max_count),
                "g_IndirectArgs": args_buffer,
            },
            command_encoder=encoder,
        )
        for level, (_, block_count, offset) in enumerate(layout):
            prev_offset = layout[level - 1][2] if level > 0 else 0
            scan_input = input_buffer if level == 0 else block_sums
            scan_output = output_buffer if level == 0 else block_offsets
            with debug_region(encoder, f"Prefix Sum Indirect Scan Level {level}", 140 + level):
                self._dispatch_indirect(
                    encoder=encoder,
                    pipeline=self.scan_blocks_pipeline,
                    args_buffer=args_buffer,
                    arg_offset=PREFIX_SCAN_ARGS_OFFSET + level * INDIRECT_DISPATCH_ARG_STRIDE,
                    vars={
                        "g_PrefixInput": scan_input,
                        "g_PrefixOutput": scan_output,
                        "g_PrefixBlockSums": block_sums,
                        "g_Exclusive": 1 if (exclusive or level > 0) else 0,
                        "g_PrefixParamsBuffer": args_buffer,
                        "g_UsePrefixParams": 1,
                        "g_Level": int(level),
                        "g_PrefixInputOffset": int(prev_offset),
                        "g_PrefixOutputOffset": 0 if level == 0 else int(prev_offset),
                        "g_PrefixBlockSumsOffset": int(offset),
                    },
                )
        for level in range(len(layout) - 2, -1, -1):
            offset = layout[level][2]
            prev_offset = layout[level - 1][2] if level > 0 else 0
            add_output = output_buffer if level == 0 else block_offsets
            with debug_region(encoder, f"Prefix Sum Indirect Add Level {level}", 160 + level):
                self._dispatch_indirect(
                    encoder=encoder,
                    pipeline=self.add_offsets_pipeline,
                    args_buffer=args_buffer,
                    arg_offset=PREFIX_ADD_ARGS_OFFSET + level * INDIRECT_DISPATCH_ARG_STRIDE,
                    vars={
                        "g_PrefixOutput": add_output,
                        "g_PrefixBlockOffsets": block_offsets,
                        "g_Exclusive": 1 if exclusive else 0,
                        "g_PrefixParamsBuffer": args_buffer,
                        "g_UsePrefixParams": 1,
                        "g_UseRadixPrefixAdd": 0,
                        "g_Level": int(level),
                        "g_PrefixOutputOffset": 0 if level == 0 else int(prev_offset),
                        "g_PrefixBlockOffsetsOffset": int(offset),
                    },
                )
        if total_buffer is not None:
            with debug_region(encoder, "Prefix Sum Indirect Total", 180):
                self.write_total_kernel.dispatch(
                    thread_count=spy.uint3(1, 1, 1),
                    vars={
                        "g_PrefixInput": input_buffer,
                        "g_PrefixOutput": output_buffer,
                        "g_TotalOut": total_buffer,
                        "g_Exclusive": 1 if exclusive else 0,
                        "g_PrefixParamsBuffer": args_buffer,
                        "g_UsePrefixParams": 1,
                        "g_Level": 0,
                        "g_PrefixInputOffset": 0,
                        "g_PrefixOutputOffset": 0,
                    },
                    command_encoder=encoder,
                )
        return args_buffer
