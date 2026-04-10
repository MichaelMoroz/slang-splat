from __future__ import annotations

import math

import numpy as np
import slangpy as spy

from src.metrics import Metrics, psnr_from_mse
from src.renderer import GaussianRenderer
from src.scene import GaussianScene

_log_scale = lambda scales: np.log(np.asarray(scales, dtype=np.float32))


def _scene(scales: np.ndarray) -> GaussianScene:
    count = int(scales.shape[0])
    rotations = np.zeros((count, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    return GaussianScene(
        positions=np.zeros((count, 3), dtype=np.float32),
        scales=np.asarray(scales, dtype=np.float32),
        rotations=rotations,
        opacities=np.full((count,), 0.5, dtype=np.float32),
        colors=np.full((count, 3), 0.5, dtype=np.float32),
        sh_coeffs=np.zeros((count, 1, 3), dtype=np.float32),
    )


def test_gpu_scale_histogram_buckets_geometric_mean(device) -> None:
    renderer = GaussianRenderer(device, width=8, height=8)
    renderer.set_scene(_scene(_log_scale(np.array([[1e-2, 1e-2, 1e-2], [1e-1, 1e-1, 1e-1], [1.0, 1.0, 1.0]], dtype=np.float32))))
    metrics = Metrics(device)

    hist = metrics.compute_scale_histogram(renderer.scene_buffers["splat_params"], 3, bin_count=3, min_log10=-2.0, max_log10=1.0)

    np.testing.assert_array_equal(hist.counts, np.array([1, 1, 1], dtype=np.int64))
    np.testing.assert_allclose(hist.bin_edges_log10, np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float64))


def test_gpu_anisotropy_histogram_buckets_max_over_min_ratio(device) -> None:
    renderer = GaussianRenderer(device, width=8, height=8)
    renderer.set_scene(_scene(_log_scale(np.array([[1.0, 1.0, 1.0], [10.0, 1.0, 1.0], [100.0, 10.0, 1.0]], dtype=np.float32))))
    metrics = Metrics(device)

    hist = metrics.compute_anisotropy_histogram(renderer.scene_buffers["splat_params"], 3, bin_count=3, min_log10=0.0, max_log10=3.0)

    np.testing.assert_array_equal(hist.counts, np.array([1, 1, 1], dtype=np.int64))


def test_gpu_histograms_ignore_nonfinite_rows(device) -> None:
    renderer = GaussianRenderer(device, width=8, height=8)
    renderer.set_scene(_scene(np.array([[-2.3025851, -2.3025851, -2.3025851], [np.nan, -1.609438, -1.609438], [-1.2039728, np.inf, -1.2039728]], dtype=np.float32)))
    metrics = Metrics(device)

    scale_hist = metrics.compute_scale_histogram(renderer.scene_buffers["splat_params"], 3, bin_count=2, min_log10=-2.0, max_log10=0.0)
    anisotropy_hist = metrics.compute_anisotropy_histogram(renderer.scene_buffers["splat_params"], 3, bin_count=2, min_log10=0.0, max_log10=1.0)

    assert int(scale_hist.counts.sum()) == 1
    assert int(anisotropy_hist.counts.sum()) == 1


def test_gpu_param_tensor_histograms_bucket_per_param(device) -> None:
    metrics = Metrics(device)
    tensor = device.create_buffer(
        size=6 * 4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.copy_destination | spy.BufferUsage.copy_source | spy.BufferUsage.unordered_access,
    )
    tensor.copy_from_numpy(np.array([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0], dtype=np.float32))

    hist = metrics.compute_param_tensor_log10_histograms(
        tensor,
        2,
        3,
        bin_count=3,
        min_log10=-3.0,
        max_log10=3.0,
        param_labels=("first", "second"),
    )

    np.testing.assert_array_equal(hist.counts, np.array([[2, 1, 0], [0, 1, 2]], dtype=np.int64))
    assert hist.param_labels == ("first", "second")


def test_gpu_param_tensor_histograms_ignore_zero_and_nonfinite_values(device) -> None:
    metrics = Metrics(device)
    tensor = device.create_buffer(
        size=8 * 4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.copy_destination | spy.BufferUsage.copy_source | spy.BufferUsage.unordered_access,
    )
    tensor.copy_from_numpy(np.array([0.0, np.nan, -1e-3, np.inf, -1e-2, 0.0, 1e-1, -1.0], dtype=np.float32))

    hist = metrics.compute_param_tensor_log10_histograms(
        tensor,
        2,
        4,
        bin_count=4,
        min_log10=-3.0,
        max_log10=1.0,
    )

    np.testing.assert_array_equal(hist.counts.sum(axis=1), np.array([1, 3], dtype=np.int64))


def test_gpu_param_tensor_ranges_track_signed_extrema(device) -> None:
    metrics = Metrics(device)
    tensor = device.create_buffer(
        size=8 * 4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.copy_destination | spy.BufferUsage.copy_source | spy.BufferUsage.unordered_access,
    )
    tensor.copy_from_numpy(np.array([-4.0, 1.5, 0.0, np.nan, -0.25, 9.0, -2.0, np.inf], dtype=np.float32))

    ranges = metrics.compute_param_tensor_ranges(tensor, 2, 4, param_labels=("first", "second"))

    np.testing.assert_allclose(ranges.min_values, np.array([-4.0, -2.0], dtype=np.float32), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(ranges.max_values, np.array([1.5, 9.0], dtype=np.float32), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(ranges.max_abs_values, np.array([4.0, 9.0], dtype=np.float32), rtol=0.0, atol=0.0)
    assert ranges.param_labels == ("first", "second")


def test_scene_param_histograms_use_linear_value_bins(device) -> None:
    renderer = GaussianRenderer(device, width=8, height=8)
    scene = GaussianScene(
        positions=np.array([[-0.75, 0.0, 0.75], [0.25, -0.25, 0.5]], dtype=np.float32),
        scales=_log_scale(np.ones((2, 3), dtype=np.float32)),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0], [0.5, -0.5, 0.5, -0.5]], dtype=np.float32),
        opacities=np.array([0.0, 0.0], dtype=np.float32),
        colors=np.array([[0.1, 0.2, 0.3], [0.8, 0.6, 0.4]], dtype=np.float32),
        sh_coeffs=np.zeros((2, 1, 3), dtype=np.float32),
    )
    renderer.set_scene(scene)

    hist = renderer.compute_scene_param_histograms(2, bin_count=4, min_value=-1.0, max_value=1.0)

    np.testing.assert_allclose(hist.bin_edges_log10, np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64), rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(hist.counts[0], np.array([1, 0, 1, 0], dtype=np.int64))
    np.testing.assert_array_equal(hist.counts[1], np.array([0, 1, 1, 0], dtype=np.int64))
    np.testing.assert_array_equal(hist.counts[2], np.array([0, 0, 0, 2], dtype=np.int64))


def test_psnr_from_mse_zero_is_infinite() -> None:
    assert math.isinf(psnr_from_mse(0.0))


def test_psnr_from_mse_matches_expected_db_value() -> None:
    assert np.isclose(psnr_from_mse(1e-2), 20.0)


def test_gpu_psnr_matches_direct_image_comparison(device) -> None:
    metrics = Metrics(device)
    rendered = device.create_texture(
        format=spy.Format.rgba32_float,
        width=2,
        height=2,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
    )
    target = device.create_texture(
        format=spy.Format.rgba32_float,
        width=2,
        height=2,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.copy_destination,
    )
    rendered.copy_from_numpy(np.array([[[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]], dtype=np.float32))
    target.copy_from_numpy(np.array([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]], dtype=np.float32))

    assert np.isclose(metrics.compute_psnr(rendered, target, 2, 2), 6.020599913279624)
