# Photometric Compensation

## Overview

Photometric compensation learns one PPISP tonemap parameter set per training image so color differences between tracked observations can be reduced before gaussian training samples target pixels.

The optimization target is:

`Train image colors -> PPISP tonemap -> Target colors`

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

For each sampled observation, the trainer reads an average `N x N` pixel neighborhood around the tracked position, applies PPISP to both source colors, and minimizes an `L1` color disagreement term between the compensated means.

A separate regularization term keeps the learned exposure, vignette, chroma, and CRF parameters close to the identity mapping so the optimizer does not drift into scene-wide recoloring.

## Kernel Structure

`shaders/utility/photometric_compensation.slang` keeps the autodiff surface intentionally small.

- Neighborhood averaging and the `L1` chain rule are handled manually.
- Reverse-mode autodiff only wraps a narrow `photometric_apply_tonemap(...)` call.
- Per-sample PPISP differentials are reduced in-group and written back through the shared packed gradient buffer.
- Loss accumulation uses float atomics so the Python trainer can read back a scalar loss for the current step.

This separation avoids the instability that showed up when autodiff covered the full neighborhood sampling path.

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

The viewer runtime uses a bounded rolling frame window when it samples photometric batches. That keeps the per-step frame upload proportional to the active window instead of the full imported dataset, which avoids multi-gigabyte uploads on large COLMAP scenes.

## Validation

The implementation is covered by focused regressions in `tests/test_photometric_compensation.py` and by the existing viewer app/session/presenter/ui test suites.
