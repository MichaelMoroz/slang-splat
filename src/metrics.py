from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import slangpy as spy

from .common import SHADER_ROOT, buffer_to_numpy, debug_region, thread_count_1d, thread_count_2d


_METRICS_SHADER = Path(SHADER_ROOT / "utility" / "metrics" / "metrics.slang")
_HISTOGRAM_RANGE_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class Log10Histogram:
    counts: np.ndarray
    bin_edges_log10: np.ndarray

    @property
    def bin_centers_log10(self) -> np.ndarray:
        return 0.5 * (self.bin_edges_log10[:-1] + self.bin_edges_log10[1:])

    @property
    def bin_centers(self) -> np.ndarray:
        return np.power(10.0, self.bin_centers_log10)


@dataclass(frozen=True, slots=True)
class ParamLog10Histograms:
    counts: np.ndarray
    bin_edges_log10: np.ndarray
    param_labels: tuple[str, ...] = ()

    @property
    def bin_centers_log10(self) -> np.ndarray:
        return 0.5 * (self.bin_edges_log10[:-1] + self.bin_edges_log10[1:])

    @property
    def bin_centers(self) -> np.ndarray:
        return np.power(10.0, self.bin_centers_log10)


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
        self._buffer_usage = (
            spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access
            | spy.BufferUsage.copy_source
            | spy.BufferUsage.copy_destination
        )
        self._histogram_capacity = max(int(max_bin_count), 1)
        self._histogram_buffer = self._create_uint_buffer(self._histogram_capacity)
        self._image_metric_buffer = self._create_float_buffer(self._METRIC_BUFFER_FLOATS)
        self._create_kernels()

    def _create_uint_buffer(self, count: int) -> spy.Buffer:
        return self.device.create_buffer(size=max(int(count), 1) * 4, usage=self._buffer_usage)

    def _create_float_buffer(self, count: int) -> spy.Buffer:
        return self.device.create_buffer(size=max(int(count), 1) * 4, usage=self._buffer_usage)

    def _create_kernels(self) -> None:
        self._k_clear_uint = self.device.create_compute_kernel(self.device.load_program(str(_METRICS_SHADER), ["csClearUIntBuffer"]))
        self._k_clear_float = self.device.create_compute_kernel(self.device.load_program(str(_METRICS_SHADER), ["csClearFloatBuffer"]))
        self._k_scale_hist = self.device.create_compute_kernel(self.device.load_program(str(_METRICS_SHADER), ["csHistogramScaleLog10"]))
        self._k_anisotropy_hist = self.device.create_compute_kernel(self.device.load_program(str(_METRICS_SHADER), ["csHistogramAnisotropyLog10"]))
        self._k_param_tensor_hist = self.device.create_compute_kernel(self.device.load_program(str(_METRICS_SHADER), ["csHistogramParamTensorLog10"]))
        self._k_image_mse = self.device.create_compute_kernel(self.device.load_program(str(_METRICS_SHADER), ["csAccumulateImageMSE"]))

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
        self._histogram_capacity = max(int(bin_count), self._histogram_capacity + self._histogram_capacity // 2)
        self._histogram_buffer = self._create_uint_buffer(self._histogram_capacity)

    def _clear_uint_buffer(self, encoder: spy.CommandEncoder, buffer: spy.Buffer, count: int) -> None:
        with debug_region(encoder, "Metrics Clear UInt", 90):
            self._k_clear_uint.dispatch(thread_count=thread_count_1d(count), vars={"g_ClearUIntBuffer": buffer, "g_ClearCount": int(count)}, command_encoder=encoder)

    def _clear_float_buffer(self, encoder: spy.CommandEncoder, buffer: spy.Buffer, count: int) -> None:
        with debug_region(encoder, "Metrics Clear Float", 91):
            self._k_clear_float.dispatch(thread_count=thread_count_1d(count), vars={"g_ClearFloatBuffer": buffer, "g_ClearCount": int(count)}, command_encoder=encoder)

    def _dispatch_histogram(self, encoder: spy.CommandEncoder, kernel: spy.ComputeKernel, splat_params: spy.Buffer, splat_count: int, bin_count: int, min_log10: float, inv_bin_size: float) -> None:
        with debug_region(encoder, "Metrics Histogram", 92):
            kernel.dispatch(
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
            )

    def _dispatch_param_tensor_histogram(self, encoder: spy.CommandEncoder, tensor: spy.Buffer, param_count: int, item_count: int, bin_count: int, min_log10: float, inv_bin_size: float) -> None:
        with debug_region(encoder, "Metrics Tensor Histogram", 94):
            self._k_param_tensor_hist.dispatch(
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
            )

    def _read_histogram(self, bin_count: int, min_log10: float, max_log10: float) -> Log10Histogram:
        counts = buffer_to_numpy(self._histogram_buffer, np.uint32)[:bin_count].astype(np.int64, copy=True)
        return Log10Histogram(counts=counts, bin_edges_log10=self._histogram_edges(bin_count, min_log10, max_log10))

    def _read_param_histograms(self, param_count: int, bin_count: int, min_log10: float, max_log10: float, labels: tuple[str, ...]) -> ParamLog10Histograms:
        counts = buffer_to_numpy(self._histogram_buffer, np.uint32)[: param_count * bin_count].astype(np.int64, copy=True).reshape(param_count, bin_count)
        return ParamLog10Histograms(counts=counts, bin_edges_log10=self._histogram_edges(bin_count, min_log10, max_log10), param_labels=labels)

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


__all__ = ["Metrics", "Log10Histogram", "ParamLog10Histograms", "psnr_from_mse"]
