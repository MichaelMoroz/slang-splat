# Training Profiles

Training profiles sit above the raw optimizer and stability parameters and provide named overrides on top of the shared fixed-count defaults.

## Profiles
- `legacy`
  - Keeps the shared defaults without scene-specific overrides.
  - This is the active training path: fixed gaussian count, CPU COLMAP-point initialization, fused L1 image loss, raster backward replay, and fused ADAM updates with scale/opacity regularization.
- `auto`
  - Resolves to `legacy`.
  - It remains as the CLI/viewer default so future profile additions can reuse the same interface without changing user-facing flags.

## Host Integration
- Profile resolution lives in `src/training/profiles.py`.
- `src/app/shared.py` applies profile overrides onto the `AdamHyperParams`, `StabilityHyperParams`, and `TrainingHyperParams` dataclasses.
- `src/app/cli.py` resolves the active profile before scene initialization so the selected hyperparameters and the uploaded scene stay in sync.
