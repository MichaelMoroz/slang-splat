# COLMAP Training Pipeline

`cli.py train-colmap` and the viewer both feed the same core trainer in `src/training/gaussian_trainer.py`.

The main training modules are:

- `src/training/gaussian_trainer.py`: public trainer facade, runtime scheduling, frame resolution logic, and host-side orchestration.
- `src/training/adam.py`: generic packed ADAM buffers and dispatch helpers.
- `src/training/optimizer.py`: Gaussian-specific optimizer and projection logic.
- `src/training/schedule.py`: schedule-resolution helpers for staged runtime parameters.

The active path is the fixed packed optimizer flow with periodic refinement. It no longer depends on the older adaptive training path.

## Data Ingestion

COLMAP loading flows through:

- `src/scene/colmap_loader.py`
- `src/scene/_internal/colmap_binary.py`
- `src/scene/_internal/colmap_ops.py`
- `src/scene/_internal/colmap_types.py`

Supported camera models:

- `SIMPLE_PINHOLE`
- `PINHOLE`
- `SIMPLE_RADIAL`
- `RADIAL`
- `OPENCV`
- `FULL_OPENCV`

The default sparse lookup accepts `sparse/0`, `sparse`, direct sparse files at the selected root, and one-level named sparse exports. The default image lookup tries `images_4`, `images`, and root-level images, including the named sparse-export parent when one was discovered.

Each training frame stores:

- resolved image path,
- COLMAP extrinsics (`q_wxyz`, `t_xyz`),
- resized intrinsics (`fx`, `fy`, `cx`, `cy`),
- radial distortion (`k1`, `k2`) when available.

Training targets are handled in two layers:

- native dataset textures cached as `rgba8_unorm_srgb`,
- one reusable training target texture in float space for the current effective training resolution.

Import and training can optionally honor the image alpha channel as a per-pixel training mask. When enabled, transparent pixels contribute no RGB, density, or refinement-edge loss/gradient.

## Initialization

The trainer consumes a prepared `GaussianScene`, but the viewer import path and CLI share the same initialization parameter model.

Current scene-seeding paths include:

- direct COLMAP sparse-point initialization,
- diffused sparse-point initialization,
- custom PLY scene seeding,
- custom mesh area-weighted surface sampling,
- depth-based initialization,
- optional appended Fibonacci shell points.

For point-based COLMAP initialization:

- positions come from the filtered sparse point cloud,
- local covariance of nearby points sets the gaussian principal axes and rotation,
- nearest-neighbor spacing still sets the overall scale magnitude before the runtime log-scale conversion,
- opacity starts from the configured constant,
- the stored runtime scene keeps scale as 3DGS log-scale.

Viewer and CLI both use the same resolved initialization hyperparameters, so count caps, scale coefficients, and opacity overrides stay aligned between the two entry points.

## Training Schedule

Training is stage-controlled rather than a single flat hyperparameter set.

The schedule is keyed by:

- `lr_schedule_steps`
- `lr_schedule_stage1_step`
- `lr_schedule_stage2_step`
- `lr_schedule_stage3_step`

Those breakpoints split the run into four stage intervals plus the initial values at step `0`. `lr_schedule_steps` is the Stage 4 endpoint and defaults to `100000`; Stage 4's default base LR target is `0.001`. The schedule helpers resolve piecewise-linear values for multiple runtime parameters.

The staged controls currently include:

- base learning rate,
- position learning-rate multiplier,
- scale learning-rate multiplier,
- rotation learning-rate multiplier,
- color learning-rate multiplier,
- opacity learning-rate multiplier,
- SH learning-rate multiplier,
- SH band cap,
- DSSIM weight,
- max visible angle,
- refinement min screen radius,
- sorting-order dithering,
- colorspace modulation,
- position-random-step noise.

SH exposure is stage-controlled by band cap rather than a single boolean toggle:

- `SH0`: DC only
- `SH1`: first-order view dependence
- `SH2`: second-order terms enabled
- `SH3`: full supported band set

The viewer uses the same resolved SH band for both viewport rendering and training.

## Training Resolution

Training resolution has two separate components.

### Downscale

`train_downscale_mode` can be manual or `Auto`.

- manual modes force a fixed integer factor,
- `Auto` starts from `train_auto_start_downscale` and descends toward `1x`,
- each lower factor lasts `train_downscale_base_iters + level_index * train_downscale_iter_step`,
- `train_downscale_max_iters` bounds the schedule duration.

### Subsample

`train_subsample_factor` is independent from downscale.

- `0` means `Auto`,
- manual values clamp to `1..8`.

`Auto` does not simply clamp the max side. Instead, it picks the factor in `1..8` whose effective resolution is closest to a target area of `1000 x 1000` pixels after the active downscale factor is applied.

The effective training render factor is:

- `downscale_factor * subsample_factor`

That factor drives:

- training renderer resolution,
- box-filter target generation,
- training debug-target preview,
- subsampled native-target sampling.

## Optimization Loop

Each trainer `step()` performs the following high-level sequence.

1. Choose the next frame from a shuffled full-view epoch. After every full pass through the training views, generate a new shuffled permutation.

2. Resolve the current effective training resolution from the active downscale and subsample settings.

3. Reuse the cached native target texture for the frame and, when necessary, refresh the reusable train target texture.
   - Non-subsampled training uses an exact integer-factor GPU box filter.
   - Subsampled training uses seeded native-pixel sampling inside the effective block.

4. Run renderer prepass and raster forward.
   - The schedule can dither only the sort-distance camera used for ordering, without changing the actual projection/debug camera.

5. Run the fixed-count forward loss path.
   - `csRasterizeTrainingForward`
   - `csClearLossBuffer`
   - `csComputeSSIMFeatures`
   - separable Gaussian blur over the moment buffer
   - `csComputeBlendedLossForward`

6. Run the backward loss and raster path.
   - `csComputeSSIMBlurredGradients`
   - blur adjoint
   - `csComputeBlendedLossBackward`
   - `csRasterizeBackward`
   - `csBackpropCachedRasterGrads`

7. Run the optimizer path.
   - optional packed per-splat grad-norm reduction,
   - one fused packed per-parameter update that applies scalar gradient clipping, ADAM, and generic regularization in a single dispatch,
   - one fused per-gaussian post step that applies gaussian-specific regularization and the post-step safety/projection rules (quaternion normalization, anisotropy clamp, screen-size clamp, SH projection).

8. When the current refinement boundary is reached, run the refinement pass.

9. Update rolling host-side loss, MSE, PSNR, timing, and frame-metric state.

The blended image term is:

- `(1 - ssim_weight) * L1 + ssim_weight * DSSIM`

DSSIM is evaluated from blurred BT.601 luminance moments rather than from hue directly.
The tracked SSIM summary is averaged over the pixels that participate in RGB training, so alpha-thresholded transparent regions no longer dilute the reported score.

## Position Random Step Noise

Position-random-step noise is a scheduled training parameter, not just a fixed scalar.

The resolved value comes from:

- `position_random_step_noise_lr`
- `position_random_step_noise_stage1_lr`
- `position_random_step_noise_stage2_lr`
- `position_random_step_noise_stage3_lr`
- `position_random_step_noise_stage4_lr`

The opacity gating behavior is controlled by:

- `position_random_step_opacity_gate_center`
- `position_random_step_opacity_gate_sharpness`

This path acts like a stochastic position regularizer/noise term during optimization and is turned off automatically when the resolved scheduled value reaches zero.

## Refinement

Periodic refinement is part of the active training path.

Important refinement controls:

- `refinement_interval`
- `refinement_growth_ratio`
- `refinement_growth_start_step`
- `refinement_alpha_cull_threshold`
- `refinement_min_contribution`
- `refinement_min_contribution_decay`
- staged `refinement_prune_lowest_contribution_ratio` (defaults: 10%, 5%, 3%, 2%, 1%)
- `refinement_sample_radius`
- `refinement_clone_scale_mul`
- staged `refinement_min_screen_radius_px`

Current behavior:

- contribution thresholds use the raw fixed-unit visible-average contribution,
- contribution values use a bidirectional leave-one-out RGB estimate: prefix transmittance times alpha times the distance between the splat color and the color composited behind it, averaged only across views where the current-frame contribution was nonzero,
- completed refinement passes decay the contribution threshold,
- each refinement pass can additionally prune the exact lowest contribution fraction of otherwise surviving splats by building a GPU candidate mask, radix-sorting contribution/id pairs, and marking the lowest-ranked survivors before the topology rewrite,
- clone-budget growth stays off until `refinement_growth_start_step`, then ramps on by `refinement_growth_ratio`,
- clone resampling weights combine gradient variance and contribution,
- split-family samples are generated from centered Fibonacci samples on the dominant local plane,
- child scales shrink by the family-size rule before `refinement_clone_scale_mul` is applied,
- packed ADAM moments are migrated with the rewritten topology so unrelated splats keep optimizer history.

## Metrics

The trainer tracks:

- total loss,
- rolling average loss,
- plain image MSE,
- PSNR,
- per-frame metrics for the training-view inspector,
- recent and total throughput statistics.

Random training backgrounds use seeded per-pixel white noise in the training raster path. `Custom` background mode still uses the configured uniform RGB color.

There is still no opacity reset schedule, no MCMC exploration term, and no standalone continuously tracked SSIM metric outside the blended loss path.

## Kernels And Packed State

Important kernel groups:

- `csResampleDownscaledTargetNearest`
- `csClearLossBuffer`
- `csComputeSSIMFeatures`
- `csComputeBlendedLossForward`
- `csComputeSSIMBlurredGradients`
- `csComputeBlendedLossBackward`
- `csRasterizeTrainingForward`
- `csRasterizeBackward`
- `csBackpropCachedRasterGrads`
- optimizer kernels in `optimizer.slang`

Packed runtime facts:

- trainable state stays param-major as `param_id * splat_count + splat_id`,
- opacity is stored as raw sigmoid logit,
- scale is stored as 3DGS log-scale,
- ADAM moments are stored as packed `float2` buffers (`m`, `v`),
- raster backward uses cached raster-field intermediates before writing final float parameter gradients.

## CLI And Viewer Integration

The CLI and viewer share the same training/init abstractions.

- `src/app/cli.py` resolves the active training profile and initialization path before trainer construction.
- `src/app/shared.py` applies training-profile overrides to the `AdamHyperParams`, `StabilityHyperParams`, and `TrainingHyperParams` dataclasses.
- the viewer creates `GaussianTrainer` through the same core parameter objects and can reinitialize a training scene without rebuilding the dataset textures.

The default training profile interface remains:

- `legacy`
- `auto`

`auto` currently resolves to `legacy`.

## Validation

The training path is covered primarily by:

- `tests/test_training_kernels.py`
- `tests/test_training_cli_smoke.py`
- `tests/test_optimizer_module.py`
- viewer/session integration tests covering import and training setup

These tests cover the fixed-count trainer kernels, YCbCr SSIM feature path, blur integration, optimizer behavior, and the current CLI/viewer orchestration.
