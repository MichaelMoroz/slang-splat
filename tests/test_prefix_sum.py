from __future__ import annotations

import numpy as np
import slangpy as spy

from src.scan import GPUPrefixSum


_USAGE = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.copy_source | spy.BufferUsage.copy_destination


def _buffer_f32(device, values: np.ndarray) -> spy.Buffer:
    data = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    buffer = device.create_buffer(size=max(data.size, 1) * 4, usage=_USAGE)
    if data.size > 0:
        buffer.copy_from_numpy(data)
    return buffer


def _buffer_u32(device, values: np.ndarray) -> spy.Buffer:
    data = np.ascontiguousarray(values, dtype=np.uint32).reshape(-1)
    buffer = device.create_buffer(size=max(data.size, 1) * 4, usage=_USAGE)
    if data.size > 0:
        buffer.copy_from_numpy(data)
    return buffer


def _read_f32(buffer: spy.Buffer, count: int) -> np.ndarray:
    return np.frombuffer(buffer.to_numpy().tobytes(), dtype=np.float32)[:count].copy()


def _read_u32(buffer: spy.Buffer, count: int) -> np.ndarray:
    return np.frombuffer(buffer.to_numpy().tobytes(), dtype=np.uint32)[:count].copy()


def test_prefix_sum_clear_u32(device):
    scan = GPUPrefixSum(device)
    buffer = _buffer_u32(device, np.arange(17, dtype=np.uint32))
    enc = device.create_command_encoder()
    scan.clear_u32(enc, buffer, 17, clear_value=9)
    device.submit_command_buffer(enc.finish())
    device.wait()
    np.testing.assert_array_equal(_read_u32(buffer, 17), np.full((17,), 9, dtype=np.uint32))


def test_prefix_sum_float_scan_matches_numpy_cumsum(device):
    scan = GPUPrefixSum(device)
    values = np.linspace(0.125, 3.0, num=31, dtype=np.float32)
    src = _buffer_f32(device, values)
    dst = _buffer_f32(device, np.zeros_like(values))
    total = _buffer_f32(device, np.zeros((1,), dtype=np.float32))
    enc = device.create_command_encoder()
    scan.scan_float(enc, src, dst, values.size, total)
    device.submit_command_buffer(enc.finish())
    device.wait()
    np.testing.assert_allclose(_read_f32(dst, values.size), np.cumsum(values, dtype=np.float32), rtol=0.0, atol=5e-6)
    np.testing.assert_allclose(_read_f32(total, 1), np.array([np.sum(values, dtype=np.float32)], dtype=np.float32), rtol=0.0, atol=5e-6)


def test_prefix_sum_uint_scan_matches_numpy_cumsum(device):
    scan = GPUPrefixSum(device)
    values = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.uint32)
    src = _buffer_u32(device, values)
    dst = _buffer_u32(device, np.zeros_like(values))
    enc = device.create_command_encoder()
    scan.scan_uint(enc, src, dst, values.size)
    device.submit_command_buffer(enc.finish())
    device.wait()
    np.testing.assert_array_equal(_read_u32(dst, values.size), np.cumsum(values, dtype=np.uint32))


def test_prefix_sum_float_scan_recurses_across_multiple_blocks(device):
    scan = GPUPrefixSum(device)
    values = np.asarray((np.arange(600, dtype=np.float32) % 7.0) * 0.25 + 0.5, dtype=np.float32)
    src = _buffer_f32(device, values)
    dst = _buffer_f32(device, np.zeros_like(values))
    total = _buffer_f32(device, np.zeros((1,), dtype=np.float32))
    enc = device.create_command_encoder()
    scan.scan_float(enc, src, dst, values.size, total)
    device.submit_command_buffer(enc.finish())
    device.wait()
    expected = np.cumsum(values, dtype=np.float32)
    np.testing.assert_allclose(_read_f32(dst, values.size), expected, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(_read_f32(total, 1), np.array([expected[-1]], dtype=np.float32), rtol=0.0, atol=1e-5)


def test_prefix_sum_uint_scan_recurses_across_multiple_blocks(device):
    scan = GPUPrefixSum(device)
    values = np.asarray((np.arange(700, dtype=np.uint32) % 5) + 1, dtype=np.uint32)
    src = _buffer_u32(device, values)
    dst = _buffer_u32(device, np.zeros_like(values))
    enc = device.create_command_encoder()
    scan.scan_uint(enc, src, dst, values.size)
    device.submit_command_buffer(enc.finish())
    device.wait()
    np.testing.assert_array_equal(_read_u32(dst, values.size), np.cumsum(values, dtype=np.uint32))
