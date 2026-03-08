# Training Profiles

Training profiles sit above the raw optimizer and density-control flags and provide scene-specific overrides without changing the legacy defaults used by the viewer and garden regression.

## Profiles
- `legacy`
  - Keeps the existing viewer-oriented defaults.
  - MCMC position noise stays enabled.
  - Density control follows the standard schedule controls.
- `bicycle-images4-psnr`
  - Selected automatically by `cli.py train-colmap` when `colmap-root` resolves to `dataset/bicycle` and `images-subdir` is `images_4`.
  - Uses paper-like per-parameter learning rates (`xyz`, `scale`, `rotation`, `color`, `opacity`) but disables MCMC noise, DSSIM mixing, opacity reset, and density-control pruning because the current RGB-only trainer converges more reliably in that simplified regime.
  - Forces a white background and `init_opacity = 0.1`.
  - Keeps the gaussian cap at `200000` so the benchmark is not blocked by initialization headroom.

## Host Integration
- Profile resolution lives in `src/training/profiles.py`.
- `src/app/shared.py` applies profile overrides onto the `AdamHyperParams`, `StabilityHyperParams`, and `TrainingHyperParams` dataclasses.
- `src/app/cli.py` resolves the active profile before scene initialization so the benchmarked configuration and the optimizer state stay in sync.

## Benchmark
- `tools/benchmark_bicycle_training.py` runs the tuned bicycle `/4` profile for a requested step budget and prints:
  - rolling `avg_psnr`,
  - current `last_psnr`,
  - active splat count,
  - gap to the `23.18 dB` target.
