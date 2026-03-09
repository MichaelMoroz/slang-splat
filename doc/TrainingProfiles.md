# Training Profiles

Training profiles sit above the raw optimizer and density-control flags and provide scene-specific overrides without changing the legacy defaults used by the viewer and garden regression.

## Profiles
- `legacy`
  - Keeps the existing viewer-oriented defaults.
  - MCMC position noise stays enabled.
  - Density control follows the standard schedule controls, now with a lower default split threshold (`1.5e-4`) and scheduled opacity resets enabled again.
- `bicycle-images4-psnr`
  - Selected automatically by `cli.py train-colmap` when `colmap-root` resolves to `dataset/bicycle` and `images-subdir` is `images_4`.
  - Uses paper-like per-parameter learning rates (`xyz`, `scale`, `rotation`, `color`, `opacity`), a mild log-scale regularizer (`1e-4`) to keep splats off the hard min-scale clamp, and the repaired densification path with a moderate schedule tuned around explicit per-splat visibility.
  - Forces `init_opacity = 0.1`.
  - Raises the gaussian cap to `300000`, which slightly reduces the COLMAP initialization scale and improves early densification headroom without changing the active 5k splat count ceiling.

## Host Integration
- Profile resolution lives in `src/training/profiles.py`.
- `src/app/shared.py` applies profile overrides onto the `AdamHyperParams`, `StabilityHyperParams`, and `TrainingHyperParams` dataclasses.
- `src/app/cli.py` resolves the active profile before scene initialization so the benchmarked configuration and the optimizer state stay in sync.

## Benchmark
- `tools/benchmark_bicycle_training.py` runs the tuned bicycle `/4` profile for a requested step budget and prints:
  - rolling train-set `avg_psnr`,
  - current `last_psnr`,
  - active splat count,
  - full-dataset PSNR,
  - eval-split PSNR on the deterministic `every 8th frame` test split,
  - gap to the `23.18 dB` target.
