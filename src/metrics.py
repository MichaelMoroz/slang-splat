from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import slangpy as spy

from .utility import RW_BUFFER_USAGE, SHADER_ROOT, alloc_buffer, buffer_to_numpy, dispatch, grow_capacity, load_compute_kernels, thread_count_1d, thread_count_2d


_METRICS_SHADER = Path(SHADER_ROOT / "utility" / "metrics" / "metrics.slang")
_HISTOGRAM_RANGE_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class Log10Histogram:
    counts: np.ndarray
    bin_edges_log10: np.ndarray

    @property
    def bin_centers_log10(self) -> np.ndarray:
        return 0.5 * (self.bin_edges_log10[:-1] + self.bin_edges_log10[1:])


@dataclass(frozen=True, slots=True)
class ParamLog10Histograms:
    counts: np.ndarray
    bin_edges_log10: np.ndarray
    param_labels: tuple[str, ...] = ()

    @property
    def bin_centers_log10(self) -> np.ndarray:
        return 0.5 * (self.bin_edges_log10[:-1] + self.bin_edges_log10[1:])


@dataclass(frozen=True, slots=True)
class ParamTensorRanges:
    min_values: np.ndarray
    max_values: np.ndarray
    param_labels: tuple[str, ...] = ()

    @property
    def max_abs_values(self) -> np.ndarray:
        return np.maximum(np.abs(self.min_values), np.abs(self.max_values))


def psnr_from_mse(mse: float, *, peak_value: float = 1.0) -> float:
    peak = float(peak_value)
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("peak_value must be finite and positive.")
    value = float(mse)
    if not math.isfinite(value) or value < 0.0:
        return float("nan")
    if value == 0.0:
        return float("inf")
    return float(10.0 * math.log10((peak * peak) / value))


class Metrics:
    _METRIC_BUFFER_FLOATS = 1

    def __init__(self, device: spy.Device, max_bin_count: int = 256) -> None:
        self.device = device
        self._buffer_usage = RW_BUFFER_USAGE
        self._histogram_capacity = max(int(max_bin_count), 1)
        self._histogram_buffer = self._create_uint_buffer(self._histogram_capacity)
        self._range_capacity = 1
        self._range_buffer = self._create_uint_buffer(self._range_capacity * 2)
        self._image_metric_buffer = self._create_float_buffer(self._METRIC_BUFFER_FLOATS)
        self._create_kernels()

    def _create_uint_buffer(self, count: int) -> spy.Buffer:
        return alloc_buffer(self.device, size=max(int(count), 1) * 4, usage=self._buffer_usage)

    def _create_float_buffer(self, count: int) -> spy.Buffer:
        return alloc_buffer(self.device, size=max(int(count), 1) * 4, usage=self._buffer_usage)

    def _create_kernels(self) -> None:
        for name, kernel in load_compute_kernels(
            self.device,
            _METRICS_SHADER,
            {
                "_k_clear_uint": "csClearUIntBuffer",
                "_k_clear_float": "csClearFloatBuffer",
                "_k_scale_hist": "csHistogramScaleLog10",
                "_k_anisotropy_hist": "csHistogramAnisotropyLog10",
                "_k_param_tensor_hist": "csHistogramParamTensorLog10",
                "_k_init_param_ranges": "csInitParamTensorRanges",
                "_k_param_tensor_range": "csRangeParamTensor",
                "_k_image_mse": "csAccumulateImageMSE",
            },
        ).items():
            setattr(self, name, kernel)

    @staticmethod
    def _validate_histogram_args(bin_count: int, min_log10: float, max_log10: float) -> tuple[int, float, float, float]:
        bins = int(bin_count)
        if bins <= 0:
            raise ValueError("bin_count must be positive.")
        lo = float(min_log10)
        hi = float(max_log10)
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError("Histogram bounds must be finite.")
        if hi <= lo:
            hi = lo + _HISTOGRAM_RANGE_EPS
        return bins, lo, hi, float(bins) / (hi - lo)

    @staticmethod
    def _histogram_edges(bin_count: int, min_log10: float, max_log10: float) -> np.ndarray:
        return np.linspace(float(min_log10), float(max_log10), int(bin_count) + 1, dtype=np.float64)

    def _ensure_histogram_capacity(self, bin_count: int) -> None:
        if int(bin_count) <= self._histogram_capacity:
            return
        self._histogram_capacity = grow_capacity(bin_count, self._histogram_capacity)
        self._histogram_buffer = self._create_uint_buffer(self._histogram_capacity)

    def _ensure_range_capacity(self, param_count: int) -> None:
        if int(param_count) <= self._range_capacity:
            return
        self._range_capacity = grow_capacity(param_count, self._range_capacity)
        self._range_buffer = self._create_uint_buffer(self._range_capacity * 2)

    def _clear_uint_buffer(self, encoder: spy.CommandEncoder, buffer: spy.Buffer, count: int) -> None:
        dispatch(
            kernel=self._k_clear_uint,
            thread_count=thread_count_1d(count),
            vars={"g_ClearUIntBuffer": buffer, "g_ClearCount": int(count)},
            command_encoder=encoder,
            debug_label="Metrics Clear UInt",
            debug_color_index=90,
        )

    def _clear_float_buffer(self, encoder: spy.CommandEncoder, buffer: spy.Buffer, count: int) -> None:
        dispatch(
            kernel=self._k_clear_float,
            thread_count=thread_count_1d(count),
            vars={"g_ClearFloatBuffer": buffer, "g_ClearCount": int(count)},
            command_encoder=encoder,
            debug_label="Metrics Clear Float",
            debug_color_index=91,
        )

    def _dispatch_histogram(self, encoder: spy.CommandEncoder, kernel: spy.ComputeKernel, splat_params: spy.Buffer, splat_count: int, bin_count: int, min_log10: float, inv_bin_size: float) -> None:
        dispatch(
            kernel=kernel,
            thread_count=thread_count_1d(splat_count),
            vars={
                "g_SplatParams": splat_params,
                "g_SplatCount": int(splat_count),
                "g_BinCount": int(bin_count),
                "g_Log10Min": float(min_log10),
                "g_Log10InvBinSize": float(inv_bin_size),
                "g_Histogram": self._histogram_buffer,
            },
            command_encoder=encoder,
            debug_label="Metrics Histogram",
            debug_color_index=92,
        )

    def _dispatch_param_tensor_histogram(self, encoder: spy.CommandEncoder, tensor: spy.Buffer, param_count: int, item_count: int, bin_count: int, min_log10: float, inv_bin_size: float) -> None:
        dispatch(
            kernel=self._k_param_tensor_hist,
            thread_count=thread_count_2d(item_count, param_count),
            vars={
                "g_Tensor": tensor,
                "g_ParamCount": int(param_count),
                "g_ItemCount": int(item_count),
                "g_BinCount": int(bin_count),
                "g_Log10Min": float(min_log10),
                "g_Log10InvBinSize": float(inv_bin_size),
                "g_Histogram": self._histogram_buffer,
            },
            command_encoder=encoder,
            debug_label="Metrics Tensor Histogram",
            debug_color_index=94,
        )

    def _init_param_tensor_ranges(self, encoder: spy.CommandEncoder, param_count: int) -> None:
        dispatch(
            kernel=self._k_init_param_ranges,
            thread_count=thread_count_1d(param_count),
            vars={"g_ParamCount": int(param_count), "g_ParamRanges": self._range_buffer},
            command_encoder=encoder,
            debug_label="Metrics Init Tensor Ranges",
            debug_color_index=95,
        )

    def _dispatch_param_tensor_ranges(self, encoder: spy.CommandEncoder, tensor: spy.Buffer, param_count: int, item_count: int) -> None:
        dispatch(
            kernel=self._k_param_tensor_range,
            thread_count=thread_count_2d(item_count, param_count),
            vars={
                "g_Tensor": tensor,
                "g_ParamCount": int(param_count),
                "g_ItemCount": int(item_count),
                "g_ParamRanges": self._range_buffer,
            },
            command_encoder=encoder,
            debug_label="Metrics Tensor Ranges",
            debug_color_index=96,
        )

    def _read_histogram(self, bin_count: int, min_log10: float, max_log10: float) -> Log10Histogram:
        counts = buffer_to_numpy(self._histogram_buffer, np.uint32)[:bin_count].astype(np.int64, copy=True)
        return Log10Histogram(counts=counts, bin_edges_log10=self._histogram_edges(bin_count, min_log10, max_log10))

    def _read_param_histograms(self, param_count: int, bin_count: int, min_log10: float, max_log10: float, labels: tuple[str, ...]) -> ParamLog10Histograms:
        counts = buffer_to_numpy(self._histogram_buffer, np.uint32)[: param_count * bin_count].astype(np.int64, copy=True).reshape(param_count, bin_count)
        return ParamLog10Histograms(counts=counts, bin_edges_log10=self._histogram_edges(bin_count, min_log10, max_log10), param_labels=labels)

    @staticmethod
    def _ordered_uints_to_floats(values: np.ndarray) -> np.ndarray:
        ordered = np.asarray(values, dtype=np.uint32)
        bits = np.where((ordered & np.uint32(0x80000000)) != 0, ordered & np.uint32(0x7FFFFFFF), np.bitwise_not(ordered))
        return bits.view(np.float32)

    def _read_param_ranges(self, param_count: int, labels: tuple[str, ...]) -> ParamTensorRanges:
        raw = buffer_to_numpy(self._range_buffer, np.uint32)[: param_count * 2].astype(np.uint32, copy=True).reshape(param_count, 2)
        min_values = self._ordered_uints_to_floats(raw[:, 0]).astype(np.float32, copy=False)
        max_values = self._ordered_uints_to_floats(raw[:, 1]).astype(np.float32, copy=False)
        return ParamTensorRanges(min_values=min_values.copy(), max_values=max_values.copy(), param_labels=labels)

    def compute_scale_histogram(self, splat_params: spy.Buffer, splat_count: int, *, bin_count: int = 64, min_log10: float = -6.0, max_log10: float = 1.0) -> Log10Histogram:
        bins, lo, hi, inv_bin_size = self._validate_histogram_args(bin_count, min_log10, max_log10)
        self._ensure_histogram_capacity(bins)
        encoder = self.device.create_command_encoder()
        self._clear_uint_buffer(encoder, self._histogram_buffer, bins)
        self._dispatch_histogram(encoder, self._k_scale_hist, splat_params, int(splat_count), bins, lo, inv_bin_size)
        self.device.submit_command_buffer(encoder.finish())
        self.device.wait()
        return self._read_histogram(bins, lo, hi)

    def compute_anisotropy_histogram(self, splat_params: spy.Buffer, splat_count: int, *, bin_count: int = 64, min_log10: float = 0.0, max_log10: float = 2.0) -> Log10Histogram:
        bins, lo, hi, inv_bin_size = self._validate_histogram_args(bin_count, min_log10, max_log10)
        self._ensure_histogram_capacity(bins)
        encoder = self.device.create_command_encoder()
        self._clear_uint_buffer(encoder, self._histogram_buffer, bins)
        self._dispatch_histogram(encoder, self._k_anisotropy_hist, splat_params, int(splat_count), bins, lo, inv_bin_size)
        self.device.submit_command_buffer(encoder.finish())
        self.device.wait()
        return self._read_histogram(bins, lo, hi)

    def compute_param_tensor_log10_histograms(
        self,
        tensor: spy.Buffer,
        param_count: int,
        item_count: int,
        *,
        bin_count: int = 64,
        min_log10: float = -8.0,
        max_log10: float = 2.0,
        param_labels: tuple[str, ...] | list[str] = (),
    ) -> ParamLog10Histograms:
        bins, lo, hi, inv_bin_size = self._validate_histogram_args(bin_count, min_log10, max_log10)
        params = max(int(param_count), 0)
        items = max(int(item_count), 0)
        labels = tuple(str(label) for label in param_labels)
        if labels and len(labels) != params:
            raise ValueError("param_labels length must match param_count.")
        self._ensure_histogram_capacity(max(params * bins, 1))
        encoder = self.device.create_command_encoder()
        self._clear_uint_buffer(encoder, self._histogram_buffer, max(params * bins, 1))
        if params > 0 and items > 0:
            self._dispatch_param_tensor_histogram(encoder, tensor, params, items, bins, lo, inv_bin_size)
        self.device.submit_command_buffer(encoder.finish())
        self.device.wait()
        return self._read_param_histograms(params, bins, lo, hi, labels)

    def compute_param_tensor_ranges(
        self,
        tensor: spy.Buffer,
        param_count: int,
        item_count: int,
        *,
        param_labels: tuple[str, ...] | list[str] = (),
    ) -> ParamTensorRanges:
        params = max(int(param_count), 0)
        items = max(int(item_count), 0)
        labels = tuple(str(label) for label in param_labels)
        if labels and len(labels) != params:
            raise ValueError("param_labels length must match param_count.")
        self._ensure_range_capacity(max(params, 1))
        encoder = self.device.create_command_encoder()
        if params > 0:
            self._init_param_tensor_ranges(encoder, params)
            if items > 0:
                self._dispatch_param_tensor_ranges(encoder, tensor, params, items)
        self.device.submit_command_buffer(encoder.finish())
        self.device.wait()
        return self._read_param_ranges(params, labels)

    def dispatch_image_mse(self, encoder: spy.CommandEncoder, rendered: spy.Texture, target: spy.Texture, width: int, height: int) -> None:
        self._clear_float_buffer(encoder, self._image_metric_buffer, self._METRIC_BUFFER_FLOATS)
        with debug_region(encoder, "Metrics Image MSE", 93):
            self._k_image_mse.dispatch(
                thread_count=thread_count_2d(width, height),
                vars={
                    "g_Width": int(width),
                    "g_Height": int(height),
                    "g_InvPixelCount": 1.0 / float(max(int(width) * int(height), 1)),
                    "g_Rendered": rendered,
                    "g_Target": target,
                    "g_ImageMetricBuffer": self._image_metric_buffer,
                },
                command_encoder=encoder,
            )

    def read_image_mse(self) -> float:
        values = buffer_to_numpy(self._image_metric_buffer, np.float32)
        return float(values[0]) if values.size > 0 else float("nan")

    def compute_image_mse(self, rendered: spy.Texture, target: spy.Texture, width: int, height: int) -> float:
        encoder = self.device.create_command_encoder()
        self.dispatch_image_mse(encoder, rendered, target, width, height)
        self.device.submit_command_buffer(encoder.finish())
        self.device.wait()
        return self.read_image_mse()

    def compute_psnr(self, rendered: spy.Texture, target: spy.Texture, width: int, height: int, *, peak_value: float = 1.0) -> float:
        return psnr_from_mse(self.compute_image_mse(rendered, target, width, height), peak_value=peak_value)


__all__ = ["Metrics", "Log10Histogram", "ParamLog10Histograms", "ParamTensorRanges", "psnr_from_mse"]
