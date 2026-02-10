# Slang Splat

Basic Gaussian splat renderer implemented with Slang compute shaders and Slangpy.
Runtime target is Direct3D 12 (`d3d12`) only.

## Features
- PLY Gaussian scene loader for standard 3DGS vertex properties.
- GPU scanline work-item binning pass followed by key/value composition.
- GPU radix sort integration (copied/adapted from prior project code).
- GPU tile range build pass from sorted keys.
- GPU compute rasterizer that blends tile-local sorted splats.
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

## Realtime Viewer
```powershell
python viewer.py --ply D:\Datasets\3DGS\TEST\flowers.ply
```

Viewer controls:
- `LMB + drag`: look around
- `WASDQE`: move camera
- `Mouse wheel`: adjust move speed
- `Load PLY...`: open another scene

Smoke-test mode:
```powershell
python viewer.py --ply D:\Datasets\3DGS\TEST\flowers.ply --frames 30
```

## Run Tests
```powershell
python -m pytest -q
```

## Project Structure
- `src/scene`: scene datamodel and PLY loader.
- `src/sort`: GPU radix sort wrapper.
- `src/renderer`: camera, reference CPU algorithms, and renderer orchestration.
- `shaders/radix_sort`: radix sort shader stages.
- `shaders/renderer`: Gaussian renderer compute stages.
- `tests`: correctness tests against CPU references.
