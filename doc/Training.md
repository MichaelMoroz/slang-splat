# COLMAP Training Pipeline

`cli.py` (`train-colmap`) is a thin wrapper over `src/app/cli.py`. `src/training/gaussian_trainer.py` remains the public training facade, while `src/training/optimizer.py` now owns optimizer buffers, optimizer kernel dispatch, and per-parameter optimizer tables.

The active training path is intentionally minimal: initialize a fixed gaussian set from the COLMAP point cloud, render one shuffled training frame, compute a direct L1 image gradient on the GPU, replay raster backward for per-splat gradients, then run a fused ADAM update.

## Data Ingestion
- Loader facade: `src/scene/colmap_loader.py`
- Internal split:
  - `src/scene/_internal/colmap_binary.py`
  - `src/scene/_internal/colmap_ops.py`
  - `src/scene/_internal/colmap_types.py`
- Supported camera models:
  - `SIMPLE_PINHOLE` (model id `0`)
  - `PINHOLE` (model id `1`)
- Files:
  - `sparse/0/cameras.bin`
  - `sparse/0/images.bin`
  - `sparse/0/points3D.bin`
- Training frames are built from an image folder (default `images_4`) and include:
  - resolved image path,
  - COLMAP extrinsics (`q_wxyz`, `t_xyz`),
  - scaled intrinsics (`fx`, `fy`, `cx`, `cy`) for the selected image resolution.
- Frame target textures are stored on GPU as a single train-resolution `rgba8_unorm` set to reduce memory usage.
- Dataset texture creation and point-cloud upload/binding are isolated inside trainer helpers instead of being interleaved with the optimization step.

## Initialization
- Gaussians are initialized directly from COLMAP `points3D` on CPU.
- Position is copied from the COLMAP point cloud.
- Scale starts from nearest-neighbor point spacing repeated across XYZ; the default resolved `base_scale` matches that spacing after the requested-count density adjustment instead of shrinking it again by render-radius heuristics.
- Rotation starts as identity quaternion.
- Opacity starts from the configured constant.
- Initialization parameters:
  - count cap (`max_gaussians`, default `5900000`),
  - constant initial opacity,
  - color from COLMAP RGB.

## Optimization Loop
Each trainer `step()` performs:
1. Pick the next training frame from a shuffled full-view epoch; after all views are consumed once, regenerate a new permutation.
2. Use the cached target texture for that frame at train resolution.
3. Run renderer prepass + raster forward.
4. Run `csClearLossAndGradTex`, then `csComputeL1LossGrad`.
   - The loss kernel computes direct RGB L1 reconstruction loss.
   - It also records RGB MSE as a plain diagnostic metric.
   - The same pass writes `dLoss / dRendered` into `g_OutputGrad`.
5. Run fused raster forward/backward replay to fill per-splat gradient buffers without cached per-pixel forward state.
6. Run the optimizer pipeline:
   - `csAccumulateRegularizationGrads` adds scale and opacity regularizers on the packed param-major state.
   - `csClipPackedParamGrads` clips gradients from per-parameter clip tables owned by the optimizer module.
   - `csAdamStepPacked` applies one-thread-per-packed-parameter ADAM using packed LR and scalar range tables.
   - `csProjectGaussianParams` applies the remaining Gaussian-specific post-step projection (quaternion normalization and anisotropy clamp).
7. Update host-side rolling loss state and the last-frame MSE metric.

There is no densification, pruning, opacity reset schedule, MCMC exploration term, or PSNR/SSIM tracking on the active path.

## Kernels
- `csClearLossAndGradTex`: zero loss + output-grad texture.
- `csComputeL1LossGrad`: computes direct RGB L1 loss, records RGB MSE, and writes `g_OutputGrad`.
- Packed trainable storage remains param-major scalar packing: `param_id * splat_count + splat_id`.
- The stored opacity parameter is the raw sigmoid logit, not direct alpha.
- `optimizer.slang` owns generic optimizer kernels and tables:
  - packed ADAM,
  - packed gradient clipping,
  - per-parameter learning-rate table,
  - per-parameter scalar range table.
- `gaussian_optimizer_stage.slang` owns Gaussian-specific optimizer logic:
  - scale/opacity regularizers,
  - anisotropy clamp,
  - quaternion normalization.
- ADAM epsilon is a compile-time constant in `shaders/utility/optimizer/optimizer.slang`, not a runtime parameter.

## Numerical Reinforcement
- Loss/grad and optimizer math sanitize non-finite values.
- Gradient clipping:
  - per-component clip,
  - norm clip.
- Update clipping (`max_update`).
- Parameter bounds:
  - position absolute clamp,
  - scale min/max clamp,
  - color clamp to `[0, 1]`.
- Quaternion normalization each step with identity fallback.
- Host guard:
  - if loss is non-finite, ADAM step is skipped and moments are reset.
- Host metrics:
  - `last_mse` stores the plain image MSE metric from the current training frame,
  - `avg_loss` remains a rolling mean over one full image cycle for UI/CLI readability.

## CLI
Example:

```powershell
python cli.py train-colmap --colmap-root dataset/garden --images-subdir images_4 --iters 100 --max-gaussians 50000
```

Useful options:
- `--width/--height` for train resolution (defaults to selected image resolution),
- `--lr-*`, `--beta1`, `--beta2`,
- `--scale-l2` for autodiff log-scale regularization around the init/reference scale,
- `--max-anisotropy` for the hard per-gaussian scale-ratio cap,
- `--grad-clip`, `--grad-norm-clip`, `--max-update`,
- `--min-scale`, `--max-scale`, `--min-opacity`, `--max-opacity`.
- `--training-profile auto|legacy`; `auto` currently resolves to `legacy`.

## Validation
- `tests/test_training_kernels.py` covers the fixed-count trainer kernels, ADAM stability clamps, and CPU pointcloud initialization with nearest-neighbor scales.
- `tests/test_training_cli_smoke.py` exercises the simplified CLI training path.
- `tests/test_optimizer_module.py` verifies the packed generic optimizer with Slang autodiff gradients on a standalone quadratic objective.
- The old PSNR regression and bicycle benchmark were removed with the deleted adaptive training code.

## Viewer Integration
- `viewer.py` is a thin launcher over `src/viewer`, which is split into:
  - state (`src/viewer/state.py`)
  - UI schema (`src/viewer/ui.py`)
  - session/runtime operations (`src/viewer/session.py`)
  - frame presentation (`src/viewer/presenter.py`)
- The viewer keeps PLY workflow and COLMAP training controls:
  - load COLMAP folder,
  - set train image directory,
  - initialize training scene,
  - start/stop one-step-per-frame optimization,
  - live loss and MSE display.
