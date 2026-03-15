# Slang Splat

Basic Gaussian splat renderer implemented with Slang compute shaders and Slangpy.
Runtime target is Vulkan.

## Features
- PLY Gaussian scene loader for standard 3DGS vertex properties.
- COLMAP loader for `sparse/0` reconstructions and image-set training frames.
- GPU scanline work-item binning pass followed by key/value composition.
- GPU radix sort integration (copied/adapted from prior project code).
- GPU tile range build pass from sorted keys.
- GPU compute rasterizer that blends tile-local sorted splats with fixed `8x8` tiles, one pixel per thread, and `256`-splat shared batches.
- Fused raster forward/backward training path for per-splat gradients without per-pixel state buffers.
- Fused one-thread-per-packed-parameter ADAM training kernel for a fixed gaussian count.
- CPU COLMAP point-cloud initialization with nearest-neighbor scales.
- Dynamic training loss downscale in the viewer with auto schedule and manual override modes.
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
Raster tile sizing is internal: the renderer uses fixed `8x8` tiles with one pixel per thread.

## Realtime Viewer
```powershell
python viewer.py
```
Load scenes from the in-window overlay (`Load PLY...` / `Load COLMAP...`).

The viewer now runs as a single Slangpy window. The renderer draws the scene first, then composites the `imgui_bundle` overlay into the same swapchain image through Slangpy's UI draw-data bridge.

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
Use `--scale-l2` to control autodiff log-scale regularization around the init/reference scale (default `0.0`).
Use `--max-anisotropy` to cap each gaussian's axis scale ratio during the ADAM update (default `10.0`).
`train-colmap` also accepts `--training-profile`; `auto` currently resolves to `legacy`.

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
- Viewer training can run with `Auto` downscale scheduling or a fixed manual downscale override; the trainer caches native `rgba8_unorm_srgb` dataset textures and generates a reusable float train target with an exact `NxN` box filter on-GPU.
- COLMAP training initialization now uses the COLMAP point cloud directly on CPU: positions come from `points3D`, per-point scale is the nearest-neighbor spacing after requested-count density adjustment, rotation is identity, and opacity starts from the configured constant.
- The resolved COLMAP init bundle is shared by the CLI and viewer, so `base_scale`, jitter, and opacity overrides flow through the same path in both entrypoints.
- CLI and viewer scene initialization now honor the configured gaussian cap instead of always forcing the full COLMAP point cloud into the initial scene.
- The active trainer keeps a fixed gaussian count after initialization; there is no densification, pruning, opacity reset schedule, MCMC exploration term, or SSIM loss path.
- The training loss is direct RGB L1 with optional scale and opacity regularization accumulated in a separate optimizer pipeline.
- GPU trainable state is packed into one param-major float buffer shared by renderer and trainer: `param[param_id * splat_count + splat_id]`.
- Raster backward accumulates gradients in a separate packed Q16.16 int buffer with int atomics, then decodes once into the float param-major gradient buffer consumed by regularizers and ADAM.
- Optimizer settings live in one structured per-parameter buffer, and ADAM moments are packed into one `float2` buffer (`m`, `v`) per parameter element.
- `src/training/adam.py` is the Python-side generic ADAM module, while `src/training/optimizer.py` keeps Gaussian-specific regularization/projection; `GaussianTrainer` composes both.
- GPU scene buffers store opacity as a raw sigmoid parameter so rasterization and optimization differentiate through effective alpha directly.
- GPU scene buffers store gaussian scale in standard 3DGS log-scale form; rendering decodes `exp(log_scale)`, rasterizes with the true decoded support, and uses `radius_scale=1.0` as the default 3DGS render size.
- Pixel-floor clamping is used only to keep projection/binning conservative for tiny splats; raster alpha still evaluates the true decoded gaussian support.
- Reported training metrics are total loss, rolling average loss, and per-step `last_mse`.
- Target Y-flip is enabled by default.
- ADAM epsilon is compile-time shader state now; it is no longer exposed as a CLI or viewer runtime control.
- The viewer exposes only fixed-count optimization controls: learning rates, regularization weights, and stability clamps.
- The viewer `Train Setup` panel includes `Auto` and manual train-downscale modes. Auto starts from a separate initial factor and walks toward `1x` over training, while manual modes force a fixed factor immediately.
- Numerical reinforcement includes clipping, finite checks, and safe quaternion normalization.
- Scale regularization uses an autodiff log-space penalty around the initialization/reference scale, so equal multiplicative scale deviations are treated more uniformly.
- Min/max scale and the scale regularization reference stay user-facing linear sigma values; conversion to stored log-scale happens inside the optimizer path.
- Scale anisotropy is clamped in the ADAM step with `max(scale) / min(scale) <= max_anisotropy`.
- Shared shader utilities are centralized in `shaders/utility`; see `doc/ShaderUtilities.md`.

## Run Tests
```powershell
python -m pytest -q
```

## Complexity Budget
```powershell
python -m tools.complexity_budget
python -m tools.complexity_budget --check
```
The budget scans only production Python entrypoints plus `src/**`.

## Diagnostics
```powershell
python -m tools.scale_gradient_diagnostic --output-dir outputs\scale_gradient_diagnostic
```
This sweeps a single splat across the pixel-floor transition, records the rendered fitted blob radius, and plots the corresponding training `dL/dscale` curve against scale.

## Project Structure
- `src/app`: shared app parameter builders, scene bounds helpers, and CLI command implementations.
- `src/scene`: scene datamodel and PLY loader.
- `src/training`: COLMAP training runtime and hyperparameter dataclasses.
- `src/filter`: reusable image-space Gaussian blur utilities.
- `src/sort`: GPU radix sort wrapper.
- `src/renderer`: camera and renderer orchestration.
- `reference_impls`: CPU and analytical reference implementations used by tests.
- `src/viewer`: viewer state, UI schema, session logic, and frame presentation.
- `shaders/utility`: reusable shader math, splatting, loss, optimizer, blur, prefix-sum, and radix-sort modules.
- `shaders/renderer`: Gaussian renderer compute stages.
- `tests`: correctness tests against CPU references.
