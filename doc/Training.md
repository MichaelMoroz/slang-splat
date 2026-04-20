# COLMAP Training Pipeline

`cli.py` (`train-colmap`) is a thin wrapper over `src/app/cli.py`. `src/training/gaussian_trainer.py` remains the public training facade, `src/training/adam.py` owns generic ADAM buffers and generic optimizer-kernel dispatch, and `src/training/optimizer.py` keeps only Gaussian-specific optimizer logic.

The active training path keeps the fixed packed optimizer flow, but it now also supports periodic refinement: per-step clone-hit counting during raster training forward, alpha culling, contribution culling normalized to observed dataset pixels, and split-family densification on a configurable cadence. Densification growth is enabled by default after step `500`.

Training SH exposure is stage-controlled by a band cap rather than a boolean toggle:

- `SH0`: DC color only
- `SH1`: SH0 + first-order view dependence
- `SH2`: SH0 + SH1 + second-order terms
- `SH3`: full supported SH set

The training schedule resolves that band cap per stage, and the renderer uses the same resolved band for both viewport rendering and raster-backward differentiation.

## Data Ingestion
- Loader facade: `src/scene/colmap_loader.py`
- Internal split:
  - `src/scene/_internal/colmap_binary.py`
  - `src/scene/_internal/colmap_ops.py`
  - `src/scene/_internal/colmap_types.py`
- Supported camera models:
  - `SIMPLE_PINHOLE` (model id `0`)
  - `PINHOLE` (model id `1`)
  - `SIMPLE_RADIAL` (model id `2`)
  - `RADIAL` (model id `3`)
- Radial distortion (`k1`, `k2`) stays attached to each training frame and is applied in both projection and inverse-projection during rendering/training.
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
- COLMAP imports can optionally treat target alpha as a per-pixel training mask; when enabled, transparent pixels contribute no RGB, density, depth-ratio, or refinement-edge loss/gradient.
- Native dataset texture preparation uses a fixed 8-thread CPU loader for image decode and resize work, while GPU texture creation/upload remains serialized on the owning thread and is pipelined against those background CPU tasks.
- Dataset texture creation and point-cloud upload/binding are isolated inside trainer helpers instead of being interleaved with the optimization step.

## Initialization
- Gaussians are initialized directly from COLMAP `points3D` on CPU.
- Position is copied from the COLMAP point cloud.
- Scale starts from nearest-neighbor point spacing repeated across XYZ; the default resolved `base_scale` matches that sigma after the requested-count density adjustment, and the trainable scene stores it as 3DGS log-scale.
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
   - The prepass can dither only the camera position used for sort-distance keys via the scheduled `sorting_order_dithering` parameter (defaults `0.5 -> 0.2 -> 0.05 -> 0.01`). The renderer passes a deterministic frame/step seed and a sigma scaled by the active frame's nearest-neighbor camera distance, then the projection shader generates an independent isotropic Gaussian sort-camera offset per splat. Projection, visibility, SH view direction, raster cache, and debug depth still use the real camera.
5. Run the fixed-count forward stage:
   - `csRasterizeTrainingForward` renders the current image and stores per-pixel raster forward cache data for backward,
  - the same pass also stores softened splat density scalars plus the weighted per-pixel depth accumulation state used by the depth-std-over-mean-depth regularizer,
   - `csClearLossBuffer` resets the scalar loss slots,
   - `csComputeSSIMFeatures` converts rendered and target RGB into BT.601 YCbCr and writes 15 channels of per-pixel first and second moments (`x`, `y`, `x²`, `y²`, `xy`) into a flat buffer,
   - the reusable separable Gaussian buffer blur utility blurs those 15 channels in two dispatches,
  - `csComputeBlendedLossForward` computes RGB MSE, the blended `(1 - ssim_weight) * L1 + ssim_weight * DSSIM` image loss using runtime `ssim_c1` / `ssim_c2` stabilizers, where DSSIM is evaluated from blurred luminance moments so it does not steer hue directly, the density hinge regularizer, and the differentiable windowed-sigmoid depth-std-over-mean-depth ratio regularizer whose strongest gradients lie inside a user-controlled interval, then reduces total and tracked metrics into the loss buffer.
6. Run the fixed-count backward stage:
   - `csComputeSSIMBlurredGradients` differentiates DSSIM with respect to the blurred rendered-side moment channels using Slang autodiff,
   - the same separable Gaussian blur utility is reused as the blur adjoint to propagate those gradients back into the unblurred rendered-side moments,
   - `csComputeBlendedLossBackward` differentiates the rendered-side moment extraction with Slang autodiff, combines the resulting DSSIM image gradient with the weighted RGB L1 sign gradient, and writes the unnormalized per-pixel RGB gradient into flat `RWStructuredBuffer<float4>` `g_OutputGrad`, plus one packed `float2` regularizer gradient buffer for density and depth-ratio replay, indexed as `pixel = y * width + x`,
   - `csRasterizeBackward` consumes the cached raster forward state and accumulates quantized cached raster-field gradients for the precomputed raster-cache fields; the depth-ratio replay uses the same alpha-depth hit evaluation as training forward so the cached depth state and replay stay consistent,
   - `csBackpropCachedRasterGrads` decodes that cached-field intermediate inline, backprops through `build_cached_ellipsoid`, and writes the final float packed scene-parameter gradient buffer with the final `1 / pixel_count` normalization before the rest of training.
7. Run the optimizer pipeline:
  - `csAccumulateRegularizationGrads` adds scale, SH1, and opacity regularizers on the packed param-major state.
   - `csClipPackedParamGrads` clips gradients from a structured per-parameter settings buffer owned by the optimizer module.
   - `csComputePackedSplatGradNorms` can optionally reduce the packed gradient vector of each splat into one scalar `L2` norm for debug visualization.
   - `csAdamStepPacked` applies one-thread-per-packed-parameter ADAM using that same settings buffer plus a packed `float2` moments buffer.
  - `csProjectGaussianParams` applies the remaining Gaussian-specific post-step projection: quaternion normalization, anisotropy clamp, and an angle-only upper screen-size clamp. The clamp keeps `max_screen_fraction` as an area-equivalent viewport fraction, converts it to a circular pixel radius, maps that radius to an angular radius from the active training camera focal length, then converts that angle plus Euclidean camera-to-splat distance into one shared scalar sigma cap applied to all three scale components. It does not use visibility, cropping, projected footprint, or opacity.
8. When the configured refinement boundary is reached, run the refinement pass:
  - `csClampRefinementMinScreenSize` loops over all training cameras on GPU, converts the refinement pixel-angle floor plus Euclidean camera-to-splat distance into a support-radius lower bound for each camera, takes the minimum of those bounds across all cameras, and raises undersized splats before rewrite. Offscreen and otherwise previously invisible splats still participate because this path no longer uses projection or visibility tests,
  - cull splats with alpha below `refinement_alpha_cull_threshold`,
  - multiply the user-facing minimum contribution count by `refinement_min_contribution_decay` after each completed refinement pass (`0.995` by default, i.e. a `0.5%` drop per pass),
  - round that decayed threshold to a non-negative integer count and pass it directly to the shader as `g_RefinementMinContributionThreshold`,
  - compute clone resampling weights as `pow(norm(adam_first_moment), refinement_momentum_weight_exponent)` over eligible splats, then prefix-sum those weights and binary-search random samples against the cumulative distribution,
  - split selected splats into `N + 1` family members from the accumulated clone counts using centered circular Fibonacci samples on the gaussian's largest-area local plane, seeded from a Python-provided hash of the selected training-frame `image_id`,
  - `refinement_sample_radius` controls that local-plane sampling radius at runtime,
  - `refinement_clone_scale_mul` multiplies the split-family sigma after the default `family_size^(-1/3)` shrink and defaults to `1.0`,
  - shrink each child sigma by `family_size^(-1/3)` and offset child means with the analytically matched residual covariance so the expected family covariance stays aligned with the parent,
  - clamp each normalized residual offset sample to a maximum radius of `3 sigma` in splat space before applying it,
  - keep family opacity unchanged because that analytically preserves the expected unnormalized Gaussian kernel amplitude under that covariance split,
  - rewrite the packed scene buffer into a compact destination buffer,
  - migrate packed ADAM `float2` moments into the rewritten topology so unrelated splats do not lose optimizer history.
9. Update host-side rolling loss state and the last-frame MSE metric.

Random training backgrounds now use seeded per-pixel white noise in the training raster path instead of a single random RGB color, while custom mode still uses the configured uniform color.

There is still no opacity reset schedule, MCMC exploration term, or standalone PSNR/SSIM metric tracking on the active path.

## Kernels
- `csDownscaleTarget`: exact integer-factor box-filter downscale from the native dataset texture into the reusable train target.
- `csClearLossBuffer`: zero scalar loss slots for the current training step.
- `csComputeSSIMFeatures`: computes per-pixel BT.601 YCbCr moment features for rendered and target images into a 15-channel flat buffer.
- `csComputeBlendedLossForward`: computes RGB MSE, blended L1+DSSIM image loss with runtime `ssim_c1` / `ssim_c2`, density hinge regularization, and windowed-sigmoid depth-ratio regularization.
- `csComputeSSIMBlurredGradients`: computes the blurred-moment DSSIM adjoint with Slang autodiff.
- `csComputeBlendedLossBackward`: computes the final image-space blended RGB gradient into `g_OutputGrad` plus the per-pixel density/depth-ratio replay gradients; the depth-ratio window is controlled by `depth_ratio_grad_min` and `depth_ratio_grad_max`, with transition softness derived from band width in shader code.
- UI-driven multi-step training batches keep per-substep loss/MSE records on the GPU and defer the single CPU readback until the batch finishes, rather than synchronizing after every substep.
- Packed trainable storage remains param-major scalar packing: `param_id * splat_count + splat_id`.
- Raster backward uses a separate param-major int accumulation buffer for cached raster-field gradients, then backprops that intermediate into final float scene-parameter gradients before optimizer consumption.
- The stored opacity parameter is the raw sigmoid logit, not direct alpha.
- `optimizer.slang` owns generic optimizer kernels and tables:
  - packed ADAM,
  - packed gradient clipping,
  - optional packed per-splat gradient-norm reduction,
  - structured per-parameter settings buffer (`lr`, grad clips, scalar clamp range, group metadata),
  - packed `float2` moments buffer (`m`, `v`).
- The refinement rewrite stage now treats that packed ADAM state as topology-coupled data and rewrites/migrates it alongside the packed scene parameters.
- `gaussian_optimizer_stage.slang` owns Gaussian-specific optimizer logic:
  - scale/SH1/opacity regularizers,
  - anisotropy clamp,
  - quaternion normalization,
  - SH0/DC projection back into valid display color space.
- ADAM epsilon is a compile-time constant in `shaders/utility/optimizer/optimizer.slang`, not a runtime parameter.

## Numerical Reinforcement
- Loss/grad and optimizer math sanitize non-finite values.
- Gradient clipping:
  - per-component clip,
  - norm clip.
- Update clipping (`max_update`).
- Parameter bounds:
  - position absolute clamp,
  - scale max clamp,
  - SH0/DC base color clamp to `[0, 1]`.
- Scale-related runtime controls (`base_scale`, `scale_reg_reference`, `max_scale`) remain user-facing linear sigma values and are converted to stored log-scale at the optimizer boundary.
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
- `--ssim-weight` for the DSSIM blend factor in the RGB image loss,
- the viewer toolkit also exposes `SSIM C1` and `SSIM C2` for the same DSSIM path,
- `--max-anisotropy` for the hard per-gaussian scale-ratio cap,
- `--grad-clip`, `--grad-norm-clip`, `--max-update`,
- `--max-scale`, `--min-opacity`, `--max-opacity`.
- `--training-profile auto|legacy`; `auto` currently resolves to `legacy`.

## Validation
- `tests/test_training_kernels.py` covers the fixed-count trainer kernels, including the YCbCr SSIM moment path, Gaussian blur integration, PyTorch image-gradient validation for the blended RGB loss, ADAM stability clamps, and CPU pointcloud initialization with nearest-neighbor scales.
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
