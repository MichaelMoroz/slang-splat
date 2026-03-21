# Torch Examples

This folder contains small torch-side examples built on top of the CUDA torch renderer.

## Garden COLMAP Trainer

Run from an elevated PowerShell session:

```powershell
python torch_examples/train_colmap_garden_torch.py
```

Default behavior:
- loads `dataset/garden`
- uses `images_4`
- initializes from the full COLMAP point cloud (`--max-gaussians 0`)
- runs `30000` Adam steps
- uses native image resolution
- prewarms one forward/backward pass so shader compilation is not mixed into the reported training throughput
- uses fixed cached raster gradients by default, with tuned fixed-point defaults (`scale=16`, `ro/local ref=10`, `L ref=10`) and the full encode controls exposed through CLI flags
- optimizes native-resolution RGB L1 with light scale/opacity regularization and late-stage cosine LR decay
- shows `tqdm` progress with loss, PSNR, recent `it/s`, and warm-run average `it/s`

Fixed-point gradient tuning flags:
- `--cached-raster-grad-atomic-mode {fixed,float}` selects the cached ellipsoid gradient accumulation path
- `--cached-raster-grad-fixed-ro-local-range`, `--cached-raster-grad-fixed-scale-range`, and `--cached-raster-grad-fixed-quat-range` set symmetric `[-X, X]` ranges for avg-inverse-scale-normalized cached geometry gradients
- `--cached-raster-grad-fixed-color-range` and `--cached-raster-grad-fixed-opacity-range` set symmetric `[-X, X]` ranges for non-normalized cached color and opacity gradients
- `--throughput-warmup-steps` controls how many measured optimizer steps are excluded from the `it/s` averages after the explicit compile warmup

Throughput notes:
- The first prewarm pass exists to compile and allocate the fixed renderer path outside the measured loop.
- Reported `it/s` values are still hardware-dependent, but the fixed path should be materially faster than float atomics on the same scene.

Prerequisites:
- CUDA-enabled PyTorch
- `slangpy`
- `dataset/garden`

This is a compact torch example over the existing renderer, not the main project training path.
