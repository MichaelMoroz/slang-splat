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
_RADIX_PACKED_HIST_SIZE = _RADIX_BIN_COUNT // 4
_PREFIX_THREADS = _PREFIX_BLOCK_SIZE // 2
_PREFIX_MAX_LEVELS = 8
_INDIRECT_DISPATCH_ARG_STRIDE = 3
_PREFIX_SCAN_ARGS_OFFSET = 0
_PREFIX_ADD_ARGS_OFFSET = _PREFIX_SCAN_ARGS_OFFSET + _PREFIX_MAX_LEVELS * _INDIRECT_DISPATCH_ARG_STRIDE
_PREFIX_PARAM_COUNT_OFFSET = _PREFIX_ADD_ARGS_OFFSET + _PREFIX_MAX_LEVELS * _INDIRECT_DISPATCH_ARG_STRIDE
_PREFIX_INDIRECT_ARGS_UINT_COUNT = _PREFIX_PARAM_COUNT_OFFSET + 1
_HISTOGRAM_ARGS_OFFSET = 0
_PREFIX_L0_ARGS_OFFSET = _HISTOGRAM_ARGS_OFFSET + _INDIRECT_DISPATCH_ARG_STRIDE
_SCATTER_ARGS_OFFSET = _PREFIX_L0_ARGS_OFFSET + 4 * _INDIRECT_DISPATCH_ARG_STRIDE
_BUILD_RANGE_ARGS_OFFSET = _SCATTER_ARGS_OFFSET + _INDIRECT_DISPATCH_ARG_STRIDE
_PARAM_ELEMENT_COUNT = _BUILD_RANGE_ARGS_OFFSET + _INDIRECT_DISPATCH_ARG_STRIDE
_PARAM_NUM_GROUPS = _PARAM_ELEMENT_COUNT + 1
_PARAM_TOTAL_N = _PARAM_NUM_GROUPS + 1
_PARAM_NUM_LEVELS = _PARAM_TOTAL_N + 1
_RADIX_INDIRECT_ARGS_UINT_COUNT = _PARAM_NUM_LEVELS + 1
_DEBUG_COLOR = spy.float3(0.18, 0.58, 0.92)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass
class GpuUtility:
    device: spy.Device

    def __post_init__(self) -> None:
        self._uint_dtype = spy.Tensor.empty(self.device, shape=(1,), dtype="uint").dtype
        self.k_prefix_scan = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csPrefixScanBlocks"]))
        self.k_prefix_add = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csPrefixAddOffsets"]))
        self.k_prefix_total = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csPrefixWriteTotal"]))
        self.p_prefix_scan = self.device.create_compute_pipeline(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csPrefixScanBlocks"]))
        self.p_prefix_add = self.device.create_compute_pipeline(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csPrefixAddOffsets"]))
        self.p_prefix_total = self.device.create_compute_pipeline(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csPrefixWriteTotal"]))
        self.k_compute_dispatch_args_from_buffer = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csComputeDispatchArgsFromBuffer"]))
        self.k_compute_prefix_args_from_buffer = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csComputePrefixIndirectArgsFromBuffer"]))
        self.k_histogram = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csRadixHistogram"]))
        self.k_radix_prefix_level = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csRadixPrefixLevel"]))
        self.k_scatter = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csRadixScatter"]))
        self.p_histogram = self.device.create_compute_pipeline(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csRadixHistogram"]))
        self.p_radix_prefix_level = self.device.create_compute_pipeline(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csRadixPrefixLevel"]))
        self.p_scatter = self.device.create_compute_pipeline(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csRadixScatter"]))
        self.k_compute_radix_args_from_buffer = self.device.create_compute_kernel(self.device.load_program(str(_SHADERS / "kernels.slang"), ["csComputeRadixIndirectArgsFromBuffer"]))
        self._indirect_usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.indirect_argument
        self._dispatch_args_buffer: spy.Buffer | None = None
        self._prefix_args_buffer: spy.Buffer | None = None
        self._radix_args_buffer: spy.Buffer | None = None

    @staticmethod
    def prefix_scratch_elements(count: int) -> int:
        total, size = 1, max(int(count), 1)
        while size > _PREFIX_BLOCK_SIZE:
            size = _ceil_div(size, _PREFIX_BLOCK_SIZE)
            total += size
        return total

    @staticmethod
    def radix_histogram_elements(count: int) -> int:
        return max(_ceil_div(max(int(count), 1), _RADIX_GROUP_SIZE) * _RADIX_PACKED_HIST_SIZE, _RADIX_PACKED_HIST_SIZE)

    @staticmethod
    def radix_prefix_elements(count: int) -> int:
        total = max(_ceil_div(max(int(count), 1), _RADIX_GROUP_SIZE) * _RADIX_BIN_COUNT, _RADIX_BIN_COUNT)
        size = total
        packed = True
        used = 0
        while True:
            used += (size + 1) // 2 if packed else size
            if size <= _PREFIX_BLOCK_SIZE // 2:
                break
            size = _ceil_div(size, _PREFIX_BLOCK_SIZE // 2)
            packed = False
        return max(used, 1)

    def _slice(self, tensor: spy.Tensor, count: int, offset: int = 0) -> spy.Tensor:
        return tensor.view((count,), offset=offset)

    def _bind_resource(self, value: spy.Tensor | spy.Buffer) -> spy.Tensor:
        if isinstance(value, spy.Tensor):
            return value
        return spy.Tensor(value, self._uint_dtype, (max(int(value.size) // 4, 1),))

    def _kernel_thread_count(self, groups_x: int, threads_per_group: int) -> spy.uint3:
        return spy.uint3(max(int(groups_x), 1) * max(int(threads_per_group), 1), 1, 1)

    def _ensure_dispatch_args_buffer(self) -> spy.Buffer:
        if self._dispatch_args_buffer is None:
            self._dispatch_args_buffer = self.device.create_buffer(size=4 * 4, usage=self._indirect_usage)
        return self._dispatch_args_buffer

    def _ensure_prefix_args_buffer(self) -> spy.Buffer:
        if self._prefix_args_buffer is None:
            self._prefix_args_buffer = self.device.create_buffer(size=_PREFIX_INDIRECT_ARGS_UINT_COUNT * 4, usage=self._indirect_usage)
        return self._prefix_args_buffer

    def _ensure_radix_args_buffer(self) -> spy.Buffer:
        if self._radix_args_buffer is None:
            self._radix_args_buffer = self.device.create_buffer(size=_RADIX_INDIRECT_ARGS_UINT_COUNT * 4, usage=self._indirect_usage)
        return self._radix_args_buffer

    def _prefix_level_layout(self, max_count: int) -> list[tuple[int, int, int]]:
        count = max(int(max_count), 1)
        scratch_offset = 0
        layout: list[tuple[int, int, int]] = []
        while True:
            block_count = max(_ceil_div(count, _PREFIX_BLOCK_SIZE), 1)
            layout.append((count, block_count, scratch_offset))
            if block_count <= 1:
                return layout
            scratch_offset += block_count
            count = block_count

    def dispatch_args_from_count_buffer(
        self,
        command_encoder: spy.CommandEncoder,
        count_buffer: spy.Tensor | spy.Buffer,
        count_offset: int,
        max_count: int,
        threads_per_group: int,
    ) -> spy.Buffer:
        args_buffer = self._ensure_dispatch_args_buffer()
        self.k_compute_dispatch_args_from_buffer.dispatch(
            thread_count=spy.uint3(1, 1, 1),
            command_encoder=command_encoder,
            vars={
                "g_CountBuffer": self._bind_resource(count_buffer),
                "g_CountOffset": int(count_offset),
                "g_MaxElementCount": int(max_count),
                "g_DispatchThreadsPerGroup": int(max(threads_per_group, 1)),
                "g_DispatchIndirectArgs": self._bind_resource(args_buffer),
            },
        )
        command_encoder.global_barrier()
        return args_buffer

    def _dispatch_indirect(self, compute_pass: spy.ComputePassEncoder, pipeline: spy.ComputePipeline, args_buffer: spy.Buffer, arg_offset: int, vars: dict[str, object]) -> None:
        cursor = spy.ShaderCursor(compute_pass.bind_pipeline(pipeline))
        for name, value in vars.items():
            setattr(cursor, name, self._bind_resource(value) if isinstance(value, (spy.Tensor, spy.Buffer)) else value)
        compute_pass.dispatch_compute_indirect(spy.BufferOffsetPair(args_buffer, arg_offset * 4))

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
                thread_count=self._kernel_thread_count(block_count, _PREFIX_THREADS),
                command_encoder=command_encoder,
                vars={
                    "g_PrefixInput": input_tensor,
                    "g_PrefixOutput": output_tensor,
                    "g_PrefixBlockSums": sums_view,
                    "g_Count": int(count),
                    "g_Exclusive": 1 if exclusive else 0,
                    "g_UsePrefixParams": 0,
                    "g_Level": 0,
                },
            )
        if block_count <= 1:
            return
        offsets_view = self._slice(block_offsets, block_count, scratch_offset)
        self._prefix_recursive(command_encoder, sums_view, offsets_view, block_sums, block_offsets, block_count, scratch_offset + block_count, True)
        with debug_group(command_encoder, f"prefix.add[{count}]", _DEBUG_COLOR):
            self.k_prefix_add.dispatch(
                thread_count=self._kernel_thread_count(_ceil_div(count, _PREFIX_THREADS), _PREFIX_THREADS),
                command_encoder=command_encoder,
                vars={
                    "g_PrefixOutput": output_tensor,
                    "g_PrefixBlockOffsets": offsets_view,
                    "g_Count": int(count),
                    "g_UsePrefixParams": 0,
                    "g_Level": 0,
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
                    "g_UsePrefixParams": 0,
                    "g_Level": 0,
                },
            )

    def prefix_sum_uint32_from_count_buffer(
        self,
        command_encoder: spy.CommandEncoder,
        input_tensor: spy.Tensor,
        output_tensor: spy.Tensor,
        block_sums: spy.Tensor,
        block_offsets: spy.Tensor,
        total_out: spy.Tensor,
        count_buffer: spy.Tensor | spy.Buffer,
        count_offset: int,
        max_count: int,
        exclusive: bool = True,
    ) -> spy.Buffer:
        args_buffer = self._ensure_prefix_args_buffer()
        if max_count <= 0:
            total_out.clear(command_encoder=command_encoder)
            return args_buffer
        with debug_group(command_encoder, f"prefix_sum_uint32_from_count_buffer[{max_count}]", _DEBUG_COLOR):
            self.k_compute_prefix_args_from_buffer.dispatch(
                thread_count=spy.uint3(1, 1, 1),
                command_encoder=command_encoder,
                vars={
                    "g_CountBuffer": self._bind_resource(count_buffer),
                    "g_CountOffset": int(count_offset),
                    "g_MaxElementCount": int(max_count),
                    "g_IndirectArgs": self._bind_resource(args_buffer),
                },
            )
            command_encoder.global_barrier()
            layout = self._prefix_level_layout(max_count)
            for level, (_, block_count, offset) in enumerate(layout):
                if level == 0:
                    level_input = input_tensor
                    level_output = output_tensor
                else:
                    prev_block_count = layout[level - 1][1]
                    prev_offset = layout[level - 1][2]
                    level_input = self._slice(block_sums, prev_block_count, prev_offset)
                    level_output = self._slice(block_offsets, prev_block_count, prev_offset)
                level_sums = self._slice(block_sums, block_count, offset)
                with command_encoder.begin_compute_pass() as compute_pass:
                    self._dispatch_indirect(
                        compute_pass,
                        self.p_prefix_scan,
                        args_buffer,
                        _PREFIX_SCAN_ARGS_OFFSET + level * _INDIRECT_DISPATCH_ARG_STRIDE,
                        {
                            "g_PrefixInput": level_input,
                            "g_PrefixOutput": level_output,
                            "g_PrefixBlockSums": level_sums,
                            "g_Exclusive": 1 if (exclusive or level > 0) else 0,
                            "g_PrefixParamsBuffer": self._bind_resource(args_buffer),
                            "g_UsePrefixParams": 1,
                            "g_Level": int(level),
                        },
                    )
            for level in range(len(layout) - 2, -1, -1):
                level_count, block_count, offset = layout[level]
                _ = level_count
                level_output = output_tensor if level == 0 else self._slice(block_offsets, layout[level - 1][1], layout[level - 1][2])
                level_offsets = self._slice(block_offsets, block_count, offset)
                with command_encoder.begin_compute_pass() as compute_pass:
                    self._dispatch_indirect(
                        compute_pass,
                        self.p_prefix_add,
                        args_buffer,
                        _PREFIX_ADD_ARGS_OFFSET + level * _INDIRECT_DISPATCH_ARG_STRIDE,
                        {
                            "g_PrefixOutput": level_output,
                            "g_PrefixBlockOffsets": level_offsets,
                            "g_Exclusive": 1 if exclusive else 0,
                            "g_PrefixParamsBuffer": self._bind_resource(args_buffer),
                            "g_UsePrefixParams": 1,
                            "g_Level": int(level),
                        },
                    )
            self.k_prefix_total.dispatch(
                thread_count=spy.uint3(1, 1, 1),
                command_encoder=command_encoder,
                vars={
                    "g_PrefixInput": input_tensor,
                    "g_PrefixOutput": output_tensor,
                    "g_TotalOut": total_out,
                    "g_Exclusive": 1 if exclusive else 0,
                    "g_PrefixParamsBuffer": self._bind_resource(args_buffer),
                    "g_UsePrefixParams": 1,
                    "g_Level": 0,
                },
            )
        return args_buffer

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
        total_n = num_groups * _RADIX_BIN_COUNT
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
                        thread_count=self._kernel_thread_count(num_groups, _RADIX_GROUP_SIZE),
                        command_encoder=command_encoder,
                        vars={
                            "g_KeysIn": src_keys,
                            "g_Histogram": histogram,
                            "g_ElementCount": int(count),
                            "g_NumGroups": int(num_groups),
                            "g_Shift": int(shift),
                            "g_DigitMask": int(digit_mask),
                            "g_UseRadixParams": 0,
                        },
                    )
                    level_size = total_n
                    level = 0
                    while True:
                        self.k_radix_prefix_level.dispatch(
                            thread_count=self._kernel_thread_count(_ceil_div(level_size, _PREFIX_THREADS), _PREFIX_THREADS),
                            command_encoder=command_encoder,
                            vars={
                                "g_Histogram": histogram,
                                "g_HistogramOffsets": histogram_prefix,
                                "g_NumGroups": int(num_groups),
                                "g_Shift": int(level),
                                "g_UseRadixParams": 0,
                            },
                        )
                        if level_size <= _PREFIX_BLOCK_SIZE // 2:
                            break
                        level_size = _ceil_div(level_size, _PREFIX_BLOCK_SIZE // 2)
                        level += 1
                    self.k_scatter.dispatch(
                        thread_count=self._kernel_thread_count(num_groups, _RADIX_GROUP_SIZE),
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
                            "g_UseRadixParams": 0,
                        },
                    )
                src_keys, src_values = dst_keys, dst_values
        return (passes & 1) == 1

    def radix_sort_uint32_from_count_buffer(
        self,
        command_encoder: spy.CommandEncoder,
        keys_in: spy.Tensor,
        values_in: spy.Tensor,
        keys_out: spy.Tensor,
        values_out: spy.Tensor,
        histogram: spy.Tensor,
        histogram_prefix: spy.Tensor,
        count_buffer: spy.Tensor | spy.Buffer,
        count_offset: int,
        max_count: int,
        start_bit: int = 0,
        bit_count: int = 32,
    ) -> tuple[bool, spy.Buffer]:
        if not 0 <= start_bit <= 31 or not 1 <= bit_count <= 32 or start_bit + bit_count > 32:
            raise ValueError("Invalid radix bit range.")
        args_buffer = self._ensure_radix_args_buffer()
        if max_count <= 0:
            return True, args_buffer
        with debug_group(command_encoder, f"radix_sort_uint32_from_count_buffer[{max_count}]", _DEBUG_COLOR):
            self.k_compute_radix_args_from_buffer.dispatch(
                thread_count=spy.uint3(1, 1, 1),
                command_encoder=command_encoder,
                vars={
                    "g_CountBuffer": self._bind_resource(count_buffer),
                    "g_CountOffset": int(count_offset),
                    "g_MaxElementCount": int(max_count),
                    "g_IndirectArgs": self._bind_resource(args_buffer),
                },
            )
            command_encoder.global_barrier()
            passes = _ceil_div(bit_count, 8)
            src_keys, src_values = keys_in, values_in
            for pass_idx in range(passes):
                shift = start_bit + pass_idx * 8
                digit_bits = min(8, bit_count - pass_idx * 8)
                digit_mask = (1 << digit_bits) - 1
                use_output = (pass_idx & 1) == 0
                dst_keys = keys_out if use_output else keys_in
                dst_values = values_out if use_output else values_in
                with debug_group(command_encoder, f"radix.indirect[{shift}:{digit_bits}]", _DEBUG_COLOR):
                    with command_encoder.begin_compute_pass() as compute_pass:
                        self._dispatch_indirect(
                            compute_pass,
                            self.p_histogram,
                        args_buffer,
                        _HISTOGRAM_ARGS_OFFSET,
                        {
                            "g_KeysIn": src_keys,
                            "g_Histogram": histogram,
                            "g_Shift": int(shift),
                            "g_DigitMask": int(digit_mask),
                            "g_RadixParamsBuffer": self._bind_resource(args_buffer),
                            "g_UseRadixParams": 1,
                        },
                    )
                    for level in range(4):
                        with command_encoder.begin_compute_pass() as compute_pass:
                            self._dispatch_indirect(
                                compute_pass,
                                self.p_radix_prefix_level,
                                args_buffer,
                                _PREFIX_L0_ARGS_OFFSET + level * _INDIRECT_DISPATCH_ARG_STRIDE,
                                {
                                    "g_HistogramOffsets": histogram_prefix,
                                    "g_Histogram": histogram,
                                    "g_Shift": int(level),
                                    "g_RadixParamsBuffer": self._bind_resource(args_buffer),
                                    "g_UseRadixParams": 1,
                                },
                            )
                    with command_encoder.begin_compute_pass() as compute_pass:
                        self._dispatch_indirect(
                            compute_pass,
                            self.p_scatter,
                        args_buffer,
                        _SCATTER_ARGS_OFFSET,
                        {
                            "g_KeysIn": src_keys,
                            "g_ValuesIn": src_values,
                            "g_KeysOut": dst_keys,
                            "g_ValuesOut": dst_values,
                            "g_HistogramOffsets": histogram_prefix,
                            "g_Shift": int(shift),
                            "g_DigitMask": int(digit_mask),
                            "g_RadixParamsBuffer": self._bind_resource(args_buffer),
                            "g_UseRadixParams": 1,
                        },
                    )
                src_keys, src_values = dst_keys, dst_values
        return (passes & 1) == 1, args_buffer
