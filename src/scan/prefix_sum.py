from __future__ import annotations

import slangpy as spy

from ..common import ROOT, debug_region

SHADER_DIR = ROOT / "shaders" / "utility" / "prefix_sum"
BLOCK_SIZE = 256


class GPUPrefixSum:
    def __init__(self, device: spy.Device):
        self.device = device
        load = device.load_program
        shader_path = str(SHADER_DIR / "prefix_sum.slang")
        self.clear_uint = device.create_compute_kernel(load(shader_path, ["csClearUIntBuffer"]))
        self.scan_float_blocks = device.create_compute_kernel(load(shader_path, ["csScanFloatBlocks"]))
        self.scan_uint_blocks = device.create_compute_kernel(load(shader_path, ["csScanUIntBlocks"]))
        self.add_float_block_offsets = device.create_compute_kernel(load(shader_path, ["csAddFloatBlockOffsets"]))
        self.add_uint_block_offsets = device.create_compute_kernel(load(shader_path, ["csAddUIntBlockOffsets"]))
        self.write_float_total_kernel = device.create_compute_kernel(load(shader_path, ["csWriteFloatScanTotal"]))
        self._float_levels: list[tuple[spy.Buffer, spy.Buffer, int]] = []
        self._uint_levels: list[tuple[spy.Buffer, spy.Buffer, int]] = []

    def _ensure_level_buffers(self, storage: list[tuple[spy.Buffer, spy.Buffer, int]], level: int, element_count: int) -> tuple[spy.Buffer, spy.Buffer]:
        usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination
        while len(storage) <= level:
            storage.append((self.device.create_buffer(size=4, usage=usage), self.device.create_buffer(size=4, usage=usage), 1))
        sum_buffer, prefix_buffer, capacity = storage[level]
        if int(element_count) <= int(capacity):
            return sum_buffer, prefix_buffer
        new_capacity = max(int(element_count), max(int(capacity), 1) + max(int(capacity), 1) // 2)
        sum_buffer = self.device.create_buffer(size=new_capacity * 4, usage=usage)
        prefix_buffer = self.device.create_buffer(size=new_capacity * 4, usage=usage)
        storage[level] = (sum_buffer, prefix_buffer, new_capacity)
        return sum_buffer, prefix_buffer

    def clear_u32(self, encoder: spy.CommandEncoder, buffer: spy.Buffer, element_count: int, clear_value: int = 0) -> None:
        with debug_region(encoder, "Prefix Sum Clear UInt", 100):
            self.clear_uint.dispatch(
                thread_count=spy.uint3(max(int(element_count), 1), 1, 1),
                vars={"g_ClearUIntBuffer": buffer, "g_ClearUIntCount": int(element_count), "g_ClearUIntValue": int(clear_value)},
                command_encoder=encoder,
            )

    def scan_float(self, encoder: spy.CommandEncoder, input_buffer: spy.Buffer, output_buffer: spy.Buffer, element_count: int, total_buffer: spy.Buffer | None = None, level: int = 0) -> None:
        if int(element_count) <= 0:
            return
        block_count = (int(element_count) + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_sums, block_prefix = self._ensure_level_buffers(self._float_levels, level, block_count)
        with debug_region(encoder, f"Prefix Sum Float Level {level}", 101 + level):
            self.scan_float_blocks.dispatch(
                thread_count=spy.uint3(max(int(element_count), 1), 1, 1),
                vars={"g_ScanInput": input_buffer, "g_ScanOutput": output_buffer, "g_ScanBlockSums": block_sums, "g_ScanElementCount": int(element_count)},
                command_encoder=encoder,
            )
        if block_count > 1:
            self.scan_float(encoder, block_sums, block_prefix, block_count, None, level + 1)
            with debug_region(encoder, f"Prefix Sum Float Add Offsets {level}", 110 + level):
                self.add_float_block_offsets.dispatch(
                    thread_count=spy.uint3(max(int(element_count), 1), 1, 1),
                    vars={"g_ScanOutput": output_buffer, "g_ScanBlockPrefix": block_prefix, "g_ScanElementCount": int(element_count)},
                    command_encoder=encoder,
                )
        if total_buffer is not None:
            self.write_float_total(encoder, output_buffer, total_buffer, element_count)

    def scan_uint(self, encoder: spy.CommandEncoder, input_buffer: spy.Buffer, output_buffer: spy.Buffer, element_count: int, level: int = 0) -> None:
        if int(element_count) <= 0:
            return
        block_count = (int(element_count) + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_sums, block_prefix = self._ensure_level_buffers(self._uint_levels, level, block_count)
        with debug_region(encoder, f"Prefix Sum UInt Level {level}", 120 + level):
            self.scan_uint_blocks.dispatch(
                thread_count=spy.uint3(max(int(element_count), 1), 1, 1),
                vars={"g_ScanUIntInput": input_buffer, "g_ScanUIntOutput": output_buffer, "g_ScanUIntBlockSums": block_sums, "g_ScanElementCount": int(element_count)},
                command_encoder=encoder,
            )
        if block_count > 1:
            self.scan_uint(encoder, block_sums, block_prefix, block_count, level + 1)
            with debug_region(encoder, f"Prefix Sum UInt Add Offsets {level}", 130 + level):
                self.add_uint_block_offsets.dispatch(
                    thread_count=spy.uint3(max(int(element_count), 1), 1, 1),
                    vars={"g_ScanUIntOutput": output_buffer, "g_ScanUIntBlockPrefix": block_prefix, "g_ScanElementCount": int(element_count)},
                    command_encoder=encoder,
                )

    def write_float_total(self, encoder: spy.CommandEncoder, prefix_buffer: spy.Buffer, total_buffer: spy.Buffer, element_count: int) -> None:
        with debug_region(encoder, "Prefix Sum Float Total", 140):
            self.write_float_total_kernel.dispatch(
                thread_count=spy.uint3(1, 1, 1),
                vars={"g_ScanOutput": prefix_buffer, "g_FloatTotalOut": total_buffer, "g_ScanElementCount": int(element_count)},
                command_encoder=encoder,
            )
