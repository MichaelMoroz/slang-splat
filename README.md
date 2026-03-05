# Slang Splat

Basic Gaussian splat renderer implemented with Slang compute shaders and Slangpy.
Runtime target is Vulkan.

## Features
- PLY Gaussian scene loader for standard 3DGS vertex properties.
- COLMAP loader for `sparse/0` reconstructions and image-set training frames.
- GPU scanline work-item binning pass followed by key/value composition.
- GPU radix sort integration (copied/adapted from prior project code).
- GPU tile range build pass from sorted keys.
- GPU compute rasterizer that blends tile-local sorted splats.
- Raster backpropagation path for per-splat gradients.
- Fused one-thread-per-splat ADAM training kernel.
- CPU reference implementations and tests for key algorithms.

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
Use `--scale-l2` to control post-ADAM decoupled L2 decay on gaussian scales (default `1e-4`).

Quick smoke configuration:
```powershell
python cli.py train-colmap --colmap-root dataset/garden --images-subdir images_8 --iters 10 --max-gaussians 1024 --width 64 --height 64
```

## PLY Render CLI
```powershell
python cli.py render-ply --ply D:\Datasets\3DGS\TEST\flowers.ply --output-dir outputs\flowers_views --views 24
```

Training notes:
- One random training image is sampled per step.
- Training target images are stored as `rgba8_unorm` textures (not float32) to reduce GPU memory usage.
- In viewer COLMAP mode, pointcloud XYZ/RGB are uploaded once on dataset load; gaussian reinitialization is done on GPU from those buffers.
- Default loss is RGB MSE.
- Target Y-flip is enabled by default.
- Per-step low-quality reinit is enabled by default: splats with `opacity <= min_opacity` or `max(scale) <= min_scale`
  can be replaced from a random valid donor splat (skip when donor is also low-quality).
- Numerical reinforcement includes clipping, finite checks, and safe quaternion normalization.
- Scale regularization uses decoupled post-ADAM L2 decay (`scale -= scale_lr * scale_l2 * scale`).

## Run Tests
```powershell
python -m pytest -q
```

## Project Structure
- `src/scene`: scene datamodel and PLY loader.
- `src/training`: COLMAP training runtime and hyperparameter dataclasses.
- `src/sort`: GPU radix sort wrapper.
- `src/renderer`: camera, reference CPU algorithms, and renderer orchestration.
- `shaders/radix_sort`: radix sort shader stages.
- `shaders/renderer`: Gaussian renderer compute stages.
- `tests`: correctness tests against CPU references.
