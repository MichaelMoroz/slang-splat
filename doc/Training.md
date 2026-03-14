# COLMAP Training Pipeline

`cli.py` (`train-colmap`) is a thin wrapper over `src/app/cli.py`. `src/training/gaussian_trainer.py` remains the public training facade, `src/training/adam.py` owns generic ADAM buffers and generic optimizer-kernel dispatch, and `src/training/optimizer.py` keeps only Gaussian-specific optimizer logic.

The active training path is intentionally minimal: initialize a fixed gaussian set from the COLMAP point cloud, run an explicit forward pass for rendering and scalar loss evaluation, run a separate backward pass for image gradients and raster replay, then apply the packed ADAM update.

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
- Frame target handling is split into:
  - native dataset textures cached as `rgba8_unorm_srgb`,
  - one reusable train target texture in `rgba32_float`,
  - a GPU box-filter downscale dispatch that writes the current frame into the train target.
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
2. Resolve the active train resolution as `ceil(native / N)` from the current effective train downscale factor.
   - Manual mode uses the selected fixed factor directly.
   - Auto mode starts from `train_auto_start_downscale` and descends toward `1x` with per-phase duration `train_downscale_base_iters + level_index * train_downscale_iter_step`.
3. Use the cached native target texture for that frame and, when needed, refresh the reusable train target texture with an exact `NxN` box filter on the GPU.
4. Run renderer prepass + raster forward.
5. Run the fixed-count forward stage:
   - `csRasterizeTrainingForward` renders the current image and stores per-pixel raster forward cache data for backward,
   - `csClearLossBuffer` resets the scalar loss slots,
   - `csComputeL1LossForward` computes direct RGB L1 reconstruction loss and RGB MSE and reduces those metrics into the loss buffer.
6. Run the fixed-count backward stage:
   - `csComputeL1LossBackward` writes the unnormalized per-pixel RGB L1 sign gradient into flat `RWStructuredBuffer<float4>` `g_OutputGrad`, indexed as `pixel = y * width + x`,
   - `csRasterizeBackward` consumes the cached raster forward state and accumulates packed Q16.16 gradients for the precomputed raster-cache fields,
   - `csBackpropCachedRasterGrads` decodes that cached-field intermediate inline, backprops through `build_cached_ellipsoid`, and writes the final float packed scene-parameter gradient buffer with the final `1 / pixel_count` normalization before the rest of training.
7. Run the optimizer pipeline:
   - `csAccumulateRegularizationGrads` adds scale and opacity regularizers on the packed param-major state.
   - `csClipPackedParamGrads` clips gradients from a structured per-parameter settings buffer owned by the optimizer module.
   - `csComputePackedSplatGradNorms` can optionally reduce the packed gradient vector of each splat into one scalar `L2` norm for debug visualization.
   - `csAdamStepPacked` applies one-thread-per-packed-parameter ADAM using that same settings buffer plus a packed `float2` moments buffer.
   - `csProjectGaussianParams` applies the remaining Gaussian-specific post-step projection (quaternion normalization and anisotropy clamp).
8. Update host-side rolling loss state and the last-frame MSE metric.

There is no densification, pruning, opacity reset schedule, MCMC exploration term, or PSNR/SSIM tracking on the active path.

## Kernels
- `csDownscaleTarget`: exact integer-factor box-filter downscale from the native dataset texture into the reusable train target.
- `csClearLossBuffer`: zero scalar loss slots for the current training step.
- `csComputeL1LossForward`: computes direct RGB L1 loss and RGB MSE only.
- `csComputeL1LossBackward`: computes only the image-space L1 gradient into `g_OutputGrad`.
- Packed trainable storage remains param-major scalar packing: `param_id * splat_count + splat_id`.
- Raster backward uses a separate param-major Q16.16 int accumulation buffer and a decode pass boundary before optimizer consumption.
- The stored opacity parameter is the raw sigmoid logit, not direct alpha.
- `optimizer.slang` owns generic optimizer kernels and tables:
  - packed ADAM,
  - packed gradient clipping,
  - optional packed per-splat gradient-norm reduction,
  - structured per-parameter settings buffer (`lr`, grad clips, scalar clamp range, group metadata),
  - packed `float2` moments buffer (`m`, `v`).
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
  - switch between auto-scheduled and fixed manual train downscale while preserving optimizer state,
  - live loss and MSE display,
  - grad-norm raster debug sourced from the optimizer's per-splat packed-gradient reduction.
