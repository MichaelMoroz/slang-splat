# Filtering

## Separable Gaussian Blur

The structured-buffer Gaussian blur in [`shaders/utility/blur/separable_gaussian_blur.slang`](../shaders/utility/blur/separable_gaussian_blur.slang) runs as two 1D compute passes:

1. Horizontal blur over row workgroups.
2. Vertical blur over column workgroups.

The adjoint uses the same separable structure in reverse order with exact clamp-edge accumulation, so the operator still matches the torch reference tests.

### Shared-memory layout

Each workgroup stages one row or one column into groupshared memory with a fixed halo of `GAUSSIAN_BLUR_RADIUS` samples on each side. The structured-buffer path now processes contiguous channel packs per dispatch slice, so each thread loads and stores adjacent channels instead of stepping through memory one channel at a time. Interior groups use direct loads. Boundary groups clamp only the halo and tail elements that cross the image bounds.

The current default uses `BLUR_GROUP_SIZE=16` and `BLUR_CHANNEL_PACK=16`. In the built-in timestamp sweep on a `1024x1024x15` workload, that was the fastest measured point on the benchmarked machine at roughly `0.74 ms` total for forward plus adjoint, which corresponds to about `672-682 GB/s` effective bandwidth from the nominal read/write traffic.

### Benchmarking

Use [`tools/benchmark_gaussian_blur.py`](../tools/benchmark_gaussian_blur.py) to measure forward and adjoint GPU time with Slangpy timestamp queries:

```bash
python tools/benchmark_gaussian_blur.py
```

The default benchmark configuration is `1024x1024` with `15` channels and sweeps several `BLUR_GROUP_SIZE` and `BLUR_CHANNEL_PACK` variants by compiling temporary shader wrappers around the main blur shader.
