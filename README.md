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
- Stage-controlled training schedule for learning rates, SH band, DSSIM weight, visible angle, sorting dithering, and position-random-step noise.
- Viewer tooling for live training, histogram inspection, GPU buffer inspection, camera overlays, debug views, and defaults export.
- Shared initialization logic between the viewer and CLI so import-time controls map to the same scene-building behavior.

## Setup

Open an elevated PowerShell session first. Slangpy and the test suite are expected to run with elevated permissions in this repo.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`viewer.py`, `cli.py`, and `render.py` install `slangpy==0.42.0` from pip automatically on first launch when it is missing or the wrong version is installed. SlangPy 0.42.0 currently needs Python 3.9-3.13, so prefer a Python 3.13 repo venv.

`viewer.py`, `cli.py`, and `render.py` also auto-install the Python packages declared in `requirements.txt` when they are missing from the active environment.

If you need `slangpy` before launching those entrypoints, install `slangpy==0.42.0` manually or replace it with an editable install from your local Slangpy checkout.

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
- `doc/PhotometricCompensation.md`: per-frame PPISP photometric trainer, pair-loss kernel, and viewer integration.
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
