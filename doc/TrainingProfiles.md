# Training Profiles

Training profiles sit above the raw optimizer and density-control flags and provide scene-specific overrides on top of the shared defaults.

## Profiles
- `legacy`
  - Keeps the shared defaults without scene-specific overrides.
  - That means the current default path is still MCMC-oriented: position noise stays enabled, densification uses `relocate dead -> append`, and the old split/prune/reset controls are not part of the default viewer workflow.
- `bicycle-images4-psnr`
  - Selected automatically by `cli.py train-colmap` when `colmap-root` resolves to `dataset/bicycle` and `images-subdir` is `images_4`.
  - Uses paper-like per-parameter learning rates (`xyz`, `scale`, `rotation`, `color`, `opacity`), a mild log-scale regularizer (`1e-4`) to keep splats off the hard min-scale clamp, and the repaired densification path with a moderate schedule tuned around explicit per-splat visibility.
  - Forces `init_opacity = 0.1`.
  - Raises the gaussian cap to `300000`, which slightly reduces the COLMAP initialization scale and improves early densification headroom without changing the active 5k splat count ceiling.
- `bicycle-images4-mcmc`
  - Keeps the paper learning rates, enables the MCMC position-noise term, uses the paper’s `relocate dead -> append +5%` schedule, and uses the paper-style `L1` scale/opacity regularizers (`0.01`).
  - Extends densification to `25000` iterations and lifts the cap to the paper’s bicycle setting (`5900000`), so 5k-step runs can grow well past the old default cap.
  - Uses the GPU prefix-sum/CDF sampler in `src/scan/prefix_sum.py` instead of the old host-side relocation fallback.

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
