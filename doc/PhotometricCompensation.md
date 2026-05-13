# Photometric Compensation

## Overview

Photometric compensation learns one PPISP tonemap parameter set per training image so color differences between tracked observations can be reduced before gaussian training samples target pixels.

The optimization target is a per-pair inverse-consistency objective:

`raw current color -> compare against inverse(current PPISP, other PPISP(other raw color))`

The implementation uses the shared packed ADAM optimizer path, sparse COLMAP track correspondences, and a dedicated Slang compute kernel that accumulates gradients with hierarchical float atomics.

## Code Layout

The feature is split across these files:

- `src/training/ppisp.py`: canonical PPISP parameter schema and provider protocol.
- `src/training/photometric_compensation.py`: pair-pool construction, trainer state, packed parameter storage, image upload, and optimizer stepping.
- `shaders/utility/photometric_compensation.slang`: pair-loss backward kernel.
- `src/training/gaussian_trainer.py`: target-tonemap provider integration for gaussian training.
- `shaders/renderer/gaussian_training_stage.slang`: compensated target sampling for downscaled and native-subsample targets.
- `src/viewer/session.py`, `src/viewer/presenter.py`, `src/viewer/ui.py`: viewer lifecycle, stepping, plots, and the dedicated photometric window.

## Training Signal

The trainer builds a deterministic sparse pair pool from tracked COLMAP observations. Each sampled pair contains:

- the shared 3D point id,
- the track length,
- frame A and frame B indices,
- sensor-space observation coordinates in both images.

For each tracked observation, the trainer precomputes a single `N x N` neighborhood mean color and mean sensor coordinate once on the GPU. Training then samples observation pairs from the sparse tracks, keeps the current-frame observation mean in raw linear space, pushes the other-frame observation mean through the other frame's PPISP, pulls it back through the current frame's PPISP inverse, and measures an `L1` disagreement between those two observation-level means.

A separate regularization term keeps the learned exposure, vignette, chroma, and CRF parameters close to the identity mapping so the optimizer does not drift into scene-wide recoloring.

## Kernel Structure

`shaders/utility/photometric_compensation.slang` keeps the autodiff surface intentionally small.

- Neighborhood averaging is handled once during observation-dataset preparation, and the `L1` chain rule is handled manually in the backward kernel.
- Reverse-mode autodiff only wraps the narrow per-sample PPISP inverse path used by the current frame for that pair orientation.
- Per-sample PPISP differentials are reduced in-group and written back through the shared packed gradient buffer.
- Loss accumulation uses float atomics so the Python trainer can read back a scalar loss for the current step.

This separation avoids the instability that showed up when autodiff covered the full neighborhood sampling path and keeps the step-time kernel focused on the inverse-tonemap residual rather than repeated texture sampling.

## Gaussian Training Integration

The photometric trainer exposes a versioned provider object. When that provider is attached to `GaussianTrainer`, compensated targets are used in both places where gaussian training consumes training images:

- downscaled target generation,
- native-subsample target reads.

The provider version is folded into target cache keys, so gaussian training refreshes target textures only when the learned PPISP parameters change.

## Viewer Workflow

The viewer exposes the trainer through `Debug -> Photometric Compensation`.

The window contains:

- start, stop, and reset controls,
- an `Apply to Gaussian Targets` toggle,
- its own photometric loss plot,
- a frame selector for inspecting the learned PPISP values of a single training image.

The photometric optimizer steps independently from gaussian optimization. If the apply toggle is enabled, gaussian training consumes the latest learned provider on subsequent target refreshes.

The trainer builds a compact precomputed observation dataset once, uploads those observation means into GPU buffers, and then trains directly from those buffers. That avoids multi-gigabyte per-step frame uploads on large COLMAP scenes and removes the step-time neighborhood reduction work from the photometric training loop.

## Validation

The implementation is covered by focused regressions in `tests/test_photometric_compensation.py` and by the existing viewer app/session/presenter/ui test suites.
