# Slang Splat

Gaussian splat renderer and trainer built around Slang compute shaders and Slangpy.
The main runtime target is Vulkan.

The repository includes:

- a realtime viewer with training controls and debug tooling,
- a COLMAP-based training pipeline with periodic refinement,
- a PLY scene loader and exporter,
- custom mesh, custom PLY, sparse-point, diffused-point, depth-based, and Fibonacci-shell initialization paths,
- an optional CUDA/PyTorch wrapper over the renderer,
- reference CPU implementations and a fairly broad correctness test suite.

## Highlights

- GPU prepass, sorting, tile-range build, and compute rasterization for Gaussian splats.
- Fused training forward and backward raster paths with packed param-major optimizer state.
- Periodic densification/refinement with contribution culling, alpha culling, and split-family rewrites.
- Stage-controlled training schedule for learning rate, SH band, DSSIM weight, visible angle, sorting dithering, and position-random-step noise.
- Viewer tooling for live training, histogram inspection, GPU buffer inspection, camera overlays, debug views, and defaults export.
- Shared initialization logic between the viewer and CLI so import-time controls map to the same scene-building behavior.

## Setup

Open an elevated PowerShell session first. Slangpy and the test suite are expected to run with elevated permissions in this repo.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install slangpy
```

If you have a local Slangpy checkout, replace the last line with an editable install.

Optional PyTorch support:

```powershell
python -m pip install <cuda-enabled-pytorch-build>
```

Repo-wide CLI, renderer, training, and viewer defaults live in `config/defaults.json`.

## Main Entry Points

### Viewer

```powershell
python viewer.py
```

The viewer is the main interactive workflow. It provides:

- `File -> Load PLY...`
- `File -> Export PLY...`
- `File -> Load COLMAP...`
- `File -> Reload`
- `File -> Reinitialize Gaussians`
- live training start/stop controls,
- renderer/debug controls,
- histogram, buffer, and training-view inspection windows,
- `Update Defaults` to write the current stable control state back into `config/defaults.json`.

Basic camera controls:

- `LMB + drag`: look
- `RMB + drag`: pan
- `WASDQE`: move
- mouse wheel: adjust move speed

### Render One Frame

```powershell
python render.py --ply C:\path\to\scene.ply --output render.png --width 1280 --height 720
```

### CLI Training

```powershell
python cli.py train-colmap --colmap-root dataset/garden --images-subdir images_4 --iters 100 --max-gaussians 50000
```

Quick smoke configuration:

```powershell
python cli.py train-colmap --colmap-root dataset/garden --images-subdir images_8 --iters 10 --max-gaussians 1024 --width 64 --height 64
```

### PLY View Sweep

```powershell
python cli.py render-ply --ply D:\Datasets\3DGS\flowers.ply --output-dir outputs\flowers_views --views 24
```

## COLMAP Import And Initialization

The viewer import window is the main dataset-ingestion surface. It resolves the COLMAP reconstruction, training-image root, optional depth root, camera selection, import-time image downscale, and initialization mode before creating the runtime scene.

Current initialization modes:

- `COLMAP Pointcloud`: sparse COLMAP points with nearest-neighbor scale initialization.
- `Diffused Pointcloud`: resampled sparse points with local jitter before the same NN-scale initialization.
- `Custom PLY`: keep the COLMAP cameras/training frames but seed gaussians from a chosen `.ply` scene.
- `Custom Mesh`: area-weighted triangle sampling with vectorized barycentric sampling and mesh-texture colors.
- `From Depth`: calibrate matched 16-bit PNG depth maps into a point cloud using robust per-pose affine fitting.

Import modifiers:

- image downscale modes: original resolution, max size, or uniform scale factor,
- `Auto Rotate Scene` to enable or disable COLMAP auto-alignment,
- optional target alpha masking for transparent training pixels,
- optional appended Fibonacci shell points around the mean COLMAP camera center.

## Training Overview

The active trainer keeps a fixed packed Gaussian scene and supports periodic refinement instead of the older adaptive training path.

Important current behavior:

- training walks a shuffled full-view epoch and reshuffles between epochs,
- training resolution is controlled by both downscale and subsample logic,
- subsampling is separate from downscaling and can run in `Auto`, `Off`, or fixed-factor modes,
- the schedule is stage-based and currently drives learning rates, SH exposure, DSSIM weight, visible-angle limits, sort dithering, and position-random-step noise,
- training can optionally use target alpha as a per-pixel mask,
- refinement includes contribution culling, alpha culling, growth scheduling, and split-family cloning.

The viewer and CLI share the same training/init parameter model, so import and initialization controls do not diverge between entry points.

## Debugging And Inspection

The viewer exposes several runtime inspection tools:

- `Buffers`: tracked GPU resource table with de-duplication and log export,
- `Histograms`: grouped log-scale parameter histograms for the live training scene,
- `Training Views`: per-frame metrics and optional camera overlays,
- debug render modes including contribution amount, gradient variance, depth mismatch, density views, ellipse outlines, and SH inspection.

Diagnostics and utilities:

```powershell
python -m tools.scale_gradient_diagnostic --output-dir outputs\scale_gradient_diagnostic
python -m tools.complexity_budget
python -m tools.complexity_budget --check
```

## Tests

```powershell
python -m pytest -q
```

The test suite covers renderer kernels, optimizer behavior, COLMAP loading/init logic, viewer/session flows, reference math, and the optional Torch bridge.

## Documentation Map

- `doc/Viewer.md`: viewer lifecycle, import flow, debug windows, and training controls.
- `doc/Training.md`: trainer architecture, optimization loop, schedules, refinement, and CLI notes.
- `doc/Rendering.md`: renderer passes, debug paths, and raster behavior.
- `doc/SceneLoading.md`: scene and COLMAP loading behavior.
- `doc/ProjectionMath.md`: projection and ellipse math derivations.
- `doc/Sorting.md`: radix sort and sort-key flow.
- `doc/GPUPrefixSum.md`: scan/prefix-sum implementation notes.
- `doc/Filtering.md`: image filtering helpers used by training and diagnostics.
- `doc/ShaderUtilities.md`: shared shader-side utility modules.
- `doc/TorchRenderer.md`: CUDA/PyTorch wrapper behavior.
- `doc/TrainingProfiles.md`: training-profile notes and current `auto` resolution.

## Project Layout

- `src/app`: shared app-side parameter builders and bounds helpers.
- `src/filter`: image-space blur and filtering utilities.
- `src/renderer`: renderer orchestration, render params, camera, and Torch bridge.
- `src/scene`: scene data model, loaders, and COLMAP/mesh initialization helpers.
- `src/sort`: GPU radix sort integration.
- `src/training`: trainer, optimizer path, schedule helpers, and defaults.
- `src/utility`: Slangpy runtime helpers and shared resource utilities.
- `src/viewer`: viewer app, UI schema, presenter, session logic, and runtime state.
- `shaders/renderer`: main renderer and training compute stages.
- `shaders/utility`: reusable math, optimizer, splat, loss, blur, and scan utilities.
- `reference_impls`: CPU/reference implementations used by tests.
- `tests`: correctness and regression coverage.

## Notes

- The viewer is the best place to understand the full runtime surface area.
- The README is intentionally high-level; the `doc/` files carry the implementation detail.
- If a default or workflow looks wrong, check `config/defaults.json` and the corresponding doc page before assuming the README is the source of truth.
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
- Pure-CUDA PyTorch renderer wrapper over the existing forward/backward raster path.
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

Optional PyTorch interface:
```powershell
python -m pip install <cuda-enabled-pytorch-build>
```

Torch examples also use `tqdm`, which is included in `requirements.txt`.

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

The viewer now runs as a single Slangpy window. The renderer draws the scene into an offscreen texture sized from the docked viewport panel, then `imgui_bundle` composites that texture and the rest of the UI into the swapchain through Slangpy's draw-data bridge.

Repo-wide CLI and viewer defaults live in `config/defaults.json`. The viewer toolkit footer can write the current control state back into that file through `Update Defaults`.

Viewer controls:
- `LMB + drag`: look around
- `WASDQE`: move camera
- `Mouse wheel`: adjust move speed
- `Load PLY...`: open another scene
- `Load COLMAP...`: open COLMAP dataset root for training setup
- `View -> Interface Scale`: resize the Dear ImGui overlay from `75%` to `200%`

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

## PyTorch CUDA Interface
The optional differentiable CUDA/PyTorch wrapper is documented in `doc/TorchRenderer.md`.
Small torch-side training examples live in `torch_examples/`.

Training notes:
- Training walks a shuffled permutation of views and reshuffles after every full image epoch.
- Training target images are stored as `rgba8_unorm_srgb` textures (not float32) so shader reads use hardware sRGB decode while keeping GPU memory usage low.
- Viewer training can run with `Auto` downscale scheduling or a fixed manual downscale override; the trainer caches native `rgba8_unorm_srgb` dataset textures and generates a reusable float train target with an exact `NxN` box filter on-GPU.
- COLMAP training initialization now uses the COLMAP point cloud directly on CPU: positions come from `points3D`, per-point scale is the nearest-neighbor spacing after requested-count density adjustment, rotation is identity, and opacity starts from the configured constant.
- The resolved COLMAP init bundle is shared by the CLI and viewer, so `base_scale`, jitter, and opacity overrides flow through the same path in both entrypoints.
- CLI and viewer scene initialization now honor the configured gaussian cap instead of always forcing the full COLMAP point cloud into the initial scene.
- The active trainer keeps a fixed gaussian count after initialization with periodic refinement; there is still no pruning, opacity reset schedule, or MCMC exploration term.
- The RGB image term is a blended `(1 - ssim_weight) * L1 + ssim_weight * DSSIM` loss. DSSIM is computed in BT.601 YCbCr from blurred per-pixel first and second moments using the reusable separable Gaussian buffer blur utility.
- GPU trainable state is packed into one param-major float buffer shared by renderer and trainer: `param[param_id * splat_count + splat_id]`.
- Raster backward supports selectable cached ellipsoid gradient atomics: `fixed` atomics by default, with `float` atomics still available when the backend supports them.
- Optimizer settings live in one structured per-parameter buffer, and ADAM moments are packed into one `float2` buffer (`m`, `v`) per parameter element.
- `src/training/adam.py` is the Python-side generic ADAM module, while `src/training/optimizer.py` keeps Gaussian-specific regularization/projection; `GaussianTrainer` composes both.
- GPU scene buffers store opacity as a raw sigmoid parameter so rasterization and optimization differentiate through effective alpha directly.
- GPU scene buffers store gaussian scale in standard 3DGS log-scale form; rendering decodes `exp(log_scale)`, rasterizes with the true decoded support, and uses `radius_scale=1.0` as the default 3DGS render size.
- The renderer and trainer now use the true decoded gaussian support directly for both projection/binning and raster alpha; there is no separate pixel-floor clamp path.
- Reported training metrics are total loss, rolling average loss, and per-step `last_mse`.
- Target Y-flip is enabled by default.
- ADAM epsilon is compile-time shader state now; it is no longer exposed as a CLI or viewer runtime control.
- The viewer exposes only fixed-count optimization controls: learning rates, regularization weights, and stability clamps.
- The viewer `Train Setup` panel includes `Auto` and manual train-downscale modes. Auto starts from a separate initial factor and walks toward `1x` over training, while manual modes force a fixed factor immediately.
- Numerical reinforcement includes clipping, finite checks, and safe quaternion normalization.
- Scale regularization uses an autodiff log-space penalty around the initialization/reference scale, so equal multiplicative scale deviations are treated more uniformly.
- Max scale and the scale regularization reference stay user-facing linear sigma values; conversion to stored log-scale happens inside the optimizer path.
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
This sweeps a single splat across a wide scale range, records the rendered fitted blob radius, and plots the corresponding training `dL/dscale` curve against scale.

## Project Structure
- `src/app`: shared app parameter builders, scene bounds helpers, and CLI command implementations.
- `src/scene`: scene datamodel and PLY loader.
- `src/training`: COLMAP training runtime and hyperparameter dataclasses.
- `src/filter`: reusable image-space Gaussian blur utilities.
- `src/sort`: GPU radix sort wrapper.
- `src/renderer`: camera, renderer orchestration, and optional CUDA/PyTorch bridge.
- `src/utility`: shared Slangpy runtime helpers for device creation, shader loading, dispatch, and resource allocation.
- `reference_impls`: CPU and analytical reference implementations used by tests.
- `src/viewer`: viewer state, UI schema, session logic, and frame presentation.
- `shaders/utility`: reusable shader math, splatting, loss, optimizer, blur, prefix-sum, and radix-sort modules.
- `shaders/renderer`: Gaussian renderer compute stages.
- `tests`: correctness tests against CPU references.
