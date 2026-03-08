# Slang Splat

Basic Gaussian splat renderer implemented with Slang compute shaders and Slangpy.
Runtime target is Vulkan.

## Features
- PLY Gaussian scene loader for standard 3DGS vertex properties.
- COLMAP loader for `sparse/0` reconstructions and image-set training frames.
- GPU scanline work-item binning pass followed by key/value composition.
- GPU radix sort integration (copied/adapted from prior project code).
- GPU tile range build pass from sorted keys.
- GPU compute rasterizer that blends tile-local sorted splats with `8x8` thread groups and `3x3` microtiles per thread (`24x24` effective tiles).
- Fused raster forward/backward training path for per-splat gradients without per-pixel state buffers.
- Fused one-thread-per-splat ADAM training kernel.
- Reusable separable Gaussian blur utility for SSIM moment filtering and backward aggregation.
- CPU reference implementations in `reference_impls` plus tests for key algorithms.

## Setup
Open an elevated PowerShell session first (Run as Administrator), then:

1. Create and activate virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Install dependencies:
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install slangpy
```
If you have a local Slangpy checkout, replace the last command with `python -m pip install -e <path-to-slangpy>`.

## Render One Frame
```powershell
python render.py --ply C:\path\to\scene.ply --output render.png --width 1280 --height 720
```
Use `--prepass-memory-mb` to cap prepass GPU memory for very large scenes.
Raster tile sizing is internal: the renderer uses fixed `24x24` effective tiles derived from `8x8` thread groups and `3x3` microtiles.

## Realtime Viewer
```powershell
python viewer.py
```
Load scenes from the UI (`Load PLY...` / `Load COLMAP...`).

Viewer controls:
- `LMB + drag`: look around
- `WASDQE`: move camera
- `Mouse wheel`: adjust move speed
- `Load PLY...`: open another scene
- `Load COLMAP...`: open COLMAP dataset root for training setup

## Training CLI
```powershell
python cli.py train-colmap --colmap-root dataset/garden --images-subdir images_4 --iters 100 --max-gaussians 50000
```
Use `--scale-l2` to control autodiff log-scale regularization around the init/reference scale (default `1e-3`).
Use `--max-anisotropy` to hard-limit each gaussian's axis scale ratio (default `10.0`).

Quick smoke configuration:
```powershell
python cli.py train-colmap --colmap-root dataset/garden --images-subdir images_8 --iters 10 --max-gaussians 1024 --width 64 --height 64
```

Tracked regression dataset subset:
- `dataset/garden/images_4`
- `dataset/garden/sparse/0`

## PLY Render CLI
```powershell
python cli.py render-ply --ply D:\Datasets\3DGS\TEST\flowers.ply --output-dir outputs\flowers_views --views 24
```

Training notes:
- Training walks a shuffled permutation of views and reshuffles after every full image epoch.
- Training target images are stored as `rgba8_unorm_srgb` textures (not float32) so shader reads use hardware sRGB decode while keeping GPU memory usage low.
- COLMAP training initialization now uses the COLMAP point cloud directly on CPU: positions come from `points3D`, per-point scale is the nearest-neighbor spacing, rotation is identity, and opacity starts from the configured constant.
- Default COLMAP initialization parameters are still derived from point-cloud nearest-neighbor spacing and requested gaussian count for reference-scale estimation and viewer defaults.
- Default loss is `(1 - lambda_dssim) * L1 + lambda_dssim * DSSIM`, with DSSIM driven by Gaussian-window SSIM moments on the GPU.
- GPU scene buffers store opacity as a raw sigmoid parameter; rasterization, pruning, and opacity reset convert it through the same sigmoid/logit helpers so Slang autodiff differentiates through opacity directly.
- Reported training metrics include total loss, rolling average loss, per-step `last_psnr`, and `avg_psnr` computed from the latest PSNR stored for each training frame slot.
- `last_mse` and PSNR remain plain RGB MSE metrics even though the optimization loss is mixed photometric.
- Target Y-flip is enabled by default.
- Density control follows the original 3DGS structure more closely: gradient/radius stats are accumulated until `densify_until_iter`, then clone/split/prune runs every `densification_interval` after `densify_from_iter`, and opacity reset runs every `opacity_reset_interval`.
- Clone duplicates small high-gradient splats, split replaces large high-gradient splats with two children, and prune removes low-opacity or oversized splats.
- MCMC position noise is now opt-in; leaving it off by default avoids the low-opacity reset interacting with max exploration noise on every splat.
- Numerical reinforcement includes clipping, finite checks, and safe quaternion normalization.
- Scale regularization uses an autodiff log-space penalty around the initialization/reference scale, so equal multiplicative scale deviations are treated more uniformly.
- Scale anisotropy is constrained with a hard per-gaussian clamp on `max(scale) / min(scale)`.
- Shared shader math constants are centralized in `shaders/renderer/math_constants.slang`.

## Run Tests
```powershell
python -m pytest -q
```

The repo intentionally keeps only the `dataset/garden/images_4` and `dataset/garden/sparse/0` subset visible in git for the COLMAP convergence regression. `tests/test_training_garden_regression.py` now runs a fixed-seed `5000`-step training pass and only passes when the final cached `avg_psnr` stays at or above `25 dB`; `last_psnr` remains a single-step diagnostic for the currently trained view.

## Complexity Budget
```powershell
python -m tools.complexity_budget
python -m tools.complexity_budget --check
```
The budget scans only production Python entrypoints plus `src/**`.

## Project Structure
- `src/app`: shared app parameter builders, scene bounds helpers, and CLI command implementations.
- `src/scene`: scene datamodel and PLY loader.
- `src/training`: COLMAP training runtime and hyperparameter dataclasses.
- `src/filter`: reusable image-space Gaussian blur utilities.
- `src/sort`: GPU radix sort wrapper.
- `src/renderer`: camera and renderer orchestration.
- `reference_impls`: CPU and analytical reference implementations used by tests.
- `src/viewer`: viewer state, UI schema, session logic, and frame presentation.
- `shaders/radix_sort`: radix sort shader stages.
- `shaders/renderer`: Gaussian renderer compute stages.
- `tests`: correctness tests against CPU references.
