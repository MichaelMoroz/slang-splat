from __future__ import annotations

import slangpy as spy

from ..utility import INDIRECT_BUFFER_USAGE, ROOT, RW_BUFFER_USAGE, alloc_buffer, debug_region, dispatch_indirect, grow_capacity, load_compute_items

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
        for name, item in load_compute_items(
            device,
            {
                "scan_blocks": ("kernel", SHADER_DIR / "prefix_sum.slang", "csPrefixScanBlocks"),
                "add_offsets": ("kernel", SHADER_DIR / "prefix_sum.slang", "csPrefixAddOffsets"),
                "write_total_kernel": ("kernel", SHADER_DIR / "prefix_sum.slang", "csPrefixWriteTotal"),
                "scan_blocks_float": ("kernel", SHADER_DIR / "prefix_sum.slang", "csPrefixScanBlocksFloat"),
                "add_offsets_float": ("kernel", SHADER_DIR / "prefix_sum.slang", "csPrefixAddOffsetsFloat"),
                "write_total_kernel_float": ("kernel", SHADER_DIR / "prefix_sum.slang", "csPrefixWriteTotalFloat"),
                "compute_dispatch_args_from_buffer_kernel": ("kernel", SHADER_DIR / "prefix_sum.slang", "csComputeDispatchArgsFromBuffer"),
                "compute_prefix_args_from_buffer_kernel": ("kernel", SHADER_DIR / "prefix_sum.slang", "csComputePrefixIndirectArgsFromBuffer"),
                "scan_blocks_pipeline": ("pipeline", SHADER_DIR / "prefix_sum.slang", "csPrefixScanBlocks"),
                "add_offsets_pipeline": ("pipeline", SHADER_DIR / "prefix_sum.slang", "csPrefixAddOffsets"),
            },
        ).items():
            setattr(self, name, item)
        self._scratch_capacity = 0
        self._block_sums: spy.Buffer | None = None
        self._block_offsets: spy.Buffer | None = None
        self._float_block_sums: spy.Buffer | None = None
        self._float_block_offsets: spy.Buffer | None = None
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
        capacity = grow_capacity(required, self._scratch_capacity)
        self._block_sums = alloc_buffer(self.device, name="prefix_sum.block_sums", size=max(capacity, 1) * 4, usage=RW_BUFFER_USAGE)
        self._block_offsets = alloc_buffer(self.device, name="prefix_sum.block_offsets", size=max(capacity, 1) * 4, usage=RW_BUFFER_USAGE)
        self._float_block_sums = alloc_buffer(self.device, name="prefix_sum.float_block_sums", size=max(capacity, 1) * 4, usage=RW_BUFFER_USAGE)
        self._float_block_offsets = alloc_buffer(self.device, name="prefix_sum.float_block_offsets", size=max(capacity, 1) * 4, usage=RW_BUFFER_USAGE)
        self._scratch_capacity = capacity
        return self._block_sums, self._block_offsets

    def _ensure_float_scratch_buffers(self, count: int) -> tuple[spy.Buffer, spy.Buffer]:
        self._ensure_scratch_buffers(count)
        if self._float_block_sums is None or self._float_block_offsets is None:
            raise RuntimeError("Float prefix scratch buffers are not initialized.")
        return self._float_block_sums, self._float_block_offsets

    def _ensure_dispatch_args(self) -> spy.Buffer:
        if self._dispatch_args is None:
            self._dispatch_args = alloc_buffer(self.device, name="prefix_sum.dispatch_args", size=4 * 4, usage=INDIRECT_BUFFER_USAGE)
        return self._dispatch_args

    def _ensure_prefix_args(self) -> spy.Buffer:
        if self._prefix_args is None:
            self._prefix_args = alloc_buffer(self.device, name="prefix_sum.prefix_args", size=PREFIX_INDIRECT_ARGS_UINT_COUNT * 4, usage=INDIRECT_BUFFER_USAGE)
        return self._prefix_args

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

    def scan_float(
        self,
        encoder: spy.CommandEncoder,
        input_buffer: spy.Buffer,
        output_buffer: spy.Buffer,
        element_count: int,
        total_buffer: spy.Buffer | None = None,
        exclusive: bool = False,
    ) -> None:
        count = int(element_count)
        block_sums, block_offsets = self._ensure_float_scratch_buffers(count)
        layout = self._level_layout(count)
        for level, (_, block_count, offset) in enumerate(layout):
            prev_offset = layout[level - 1][2] if level > 0 else 0
            scan_input = input_buffer if level == 0 else block_sums
            scan_output = output_buffer if level == 0 else block_offsets
            with debug_region(encoder, f"Prefix Sum Float Scan Level {level}", 201 + level):
                self.scan_blocks_float.dispatch(
                    thread_count=spy.uint3(max(block_count * PREFIX_THREADS, 1), 1, 1),
                    vars={
                        "g_PrefixFloatInput": scan_input,
                        "g_PrefixFloatOutput": scan_output,
                        "g_PrefixFloatBlockSums": block_sums,
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
            offset = layout[level][2]
            prev_offset = layout[level - 1][2] if level > 0 else 0
            add_output = output_buffer if level == 0 else block_offsets
            with debug_region(encoder, f"Prefix Sum Float Add Level {level}", 220 + level):
                self.add_offsets_float.dispatch(
                    thread_count=spy.uint3(max((layout[level][0] + PREFIX_THREADS - 1) // PREFIX_THREADS * PREFIX_THREADS, 1), 1, 1),
                    vars={
                        "g_PrefixFloatOutput": add_output,
                        "g_PrefixFloatBlockOffsets": block_offsets,
                        "g_Count": count,
                        "g_Exclusive": 1 if exclusive else 0,
                        "g_UsePrefixParams": 0,
                        "g_Level": int(level),
                        "g_PrefixOutputOffset": 0 if level == 0 else int(prev_offset),
                        "g_PrefixBlockOffsetsOffset": int(offset),
                    },
                    command_encoder=encoder,
                )
        if total_buffer is not None:
            with debug_region(encoder, "Prefix Sum Float Total", 240):
                self.write_total_kernel_float.dispatch(
                    thread_count=spy.uint3(1, 1, 1),
                    vars={
                        "g_PrefixFloatInput": input_buffer,
                        "g_PrefixFloatOutput": output_buffer,
                        "g_PrefixFloatTotalOut": total_buffer,
                        "g_Count": count,
                        "g_Exclusive": 1 if exclusive else 0,
                        "g_UsePrefixParams": 0,
                        "g_Level": 0,
                        "g_PrefixInputOffset": 0,
                        "g_PrefixOutputOffset": 0,
                    },
                    command_encoder=encoder,
                )

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
            dispatch_indirect(
                pipeline=self.scan_blocks_pipeline,
                args_buffer=args_buffer,
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
                command_encoder=encoder,
                arg_offset=PREFIX_SCAN_ARGS_OFFSET + level * INDIRECT_DISPATCH_ARG_STRIDE,
                debug_label=f"Prefix Sum Indirect Scan Level {level}",
                debug_color_index=140 + level,
            )
        for level in range(len(layout) - 2, -1, -1):
            offset = layout[level][2]
            prev_offset = layout[level - 1][2] if level > 0 else 0
            add_output = output_buffer if level == 0 else block_offsets
            dispatch_indirect(
                pipeline=self.add_offsets_pipeline,
                args_buffer=args_buffer,
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
                command_encoder=encoder,
                arg_offset=PREFIX_ADD_ARGS_OFFSET + level * INDIRECT_DISPATCH_ARG_STRIDE,
                debug_label=f"Prefix Sum Indirect Add Level {level}",
                debug_color_index=160 + level,
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
