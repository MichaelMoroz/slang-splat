# COLMAP Training Pipeline

`cli.py` (`train-colmap`) is a thin wrapper over `src/app/cli.py`, and `src/training/gaussian_trainer.py` remains the public training facade over dataset assets, kernel dispatch, and metric/state updates.

The mixed photometric loss uses the reusable separable blur utility in `src/filter/separable_gaussian_blur.py`, backed by `shaders/filter/separable_gaussian_blur.slang`, to build SSIM moments and to aggregate backward partials back into `dLoss / dRendered`.

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
- The repo keeps the `dataset/garden/images_4` images plus `dataset/garden/sparse/0` tracked for the long-running COLMAP regression test; the rest of `dataset/**` stays ignored.
- Training frames are built from an image folder (default `images_4`) and include:
  - resolved image path,
  - COLMAP extrinsics (`q_wxyz`, `t_xyz`),
  - scaled intrinsics (`fx`, `fy`, `cx`, `cy`) for the selected image resolution.
- Frame target textures are stored on GPU as `rgba8_unorm` (both native and train resolution) to reduce memory usage.
- Dataset texture creation and point-cloud upload/binding are isolated inside trainer helpers instead of being interleaved with the optimization step.

## Initialization
- Gaussians are initialized directly from COLMAP `points3D` on CPU.
- Position is copied from the COLMAP point cloud.
- Scale is the nearest-neighbor point spacing repeated across XYZ.
- Rotation starts as identity quaternion.
- Opacity starts from the configured constant.
- When init hyperparameters are omitted, Python derives defaults from COLMAP point-cloud nearest-neighbor spacing and requested gaussian count so initial splats are dense but only lightly overlapping.
- Initialization parameters:
  - count cap (`max_gaussians`, default `50000`),
  - constant initial opacity,
  - color from COLMAP RGB.

## Optimization Loop
Each trainer `step()` performs:
1. Pick the next training frame from a shuffled full-view epoch; after all views are consumed once, regenerate a new permutation.
2. Upload frame image as target texture (Y-flip enabled by default).
3. Run renderer prepass + raster forward.
4. Pack SSIM auxiliary textures (`rendered^2`, `target^2`, `rendered * target`) and run the separable Gaussian blur utility to produce local SSIM moments.
5. Run mixed photometric loss kernels to produce `g_OutputGrad`.
   - Reconstruction loss is `(1 - lambda_dssim) * L1 + lambda_dssim * DSSIM`.
   - DSSIM is computed from Gaussian-window SSIM moments.
   - The local SSIM kernel writes partial derivatives for `G * x`, `G * x^2`, and `G * (x * y)`.
   - Those partial maps are blurred again and composed into the final `dLoss / dRendered` texture before raster backward replay.
6. Run fused raster forward/backward replay to fill per-splat gradient buffers without cached per-pixel forward state.
7. Run fused ADAM kernel (`csAdamStepFused`) with one thread per Gaussian.
   - Scale regularization is computed in-kernel via Slang autodiff on `scale.xyz`.
   - The term is an L2 penalty on `log(scale / referenceScale)`, where the reference scale comes from the initialization base scale when available.
   - The regularization scalar is averaged over the active splat count and added to the reported loss buffer in this ADAM pass.
   - A hard anisotropy clamp enforces `max(scale.xyz) / min(scale.xyz) <= max_anisotropy`.
8. Run densification-stat update (`csUpdateDensificationStats`) to accumulate:
   - EMA of `sqrt(dot(grad_position.xyz, grad_position.xyz))` using `1 / view_count`,
   - maximum observed 2D projected radius for visible splats.
9. On the configured schedule, run `csRegenerateScene`.
   - Clone: duplicate small high-gradient splats while preserving the original optimizer state on the kept copy and zeroing moments on the new copy.
   - Split: replace large high-gradient splats with `N=2` children sampled from the parent Gaussian.
   - Prune: drop splats with low opacity or excessive world/screen footprint.
   - Regeneration resets densification stats for the new active set.
10. On the configured schedule, run `csResetOpacity` to rewrite the stored raw opacity parameter so the effective sigmoid opacity becomes `min(alpha, 0.01)`, then clear alpha optimizer moments.

## Kernels
- `csClearLossAndGradTex`: zero loss + output-grad texture.
- `csPackSSIMAux`: builds `rendered^2`, `target^2`, and `rendered * target` textures for SSIM moments.
- `csComputeMixedLossGrad`: computes mixed `L1 + DSSIM` loss, writes the direct L1 output gradient, and writes SSIM partial maps for backward Gaussian filtering.
- `csComposeMixedLossOutputGrad`: combines the blurred SSIM partial maps into the final `dLoss / dRendered`.
- `csAdamStepFused`: updates all trainable params in one pass:
  - position,
  - scale,
  - quaternion,
  - color,
  - opacity.
  - The stored opacity parameter is the raw sigmoid logit, not direct alpha.
  - Adds autodiff log-scale regularization gradients to scale gradients (`g_ScaleL2Weight`, `g_ScaleRegReference`).
  - Accumulates the averaged scale-regularization scalar contribution into `g_LossBuffer`.
  - Clamps stored scales to `[min_scale, max_scale]` and enforces `max_anisotropy`.
- `csUpdateDensificationStats`: updates per-splat gradient EMA and maximum projected radius from the current step.
- `csRegenerateScene`: inline clone/split/prune classification plus append-buffer regeneration into the next active scene buffers.
- `csResetOpacity`: rewrites the raw opacity parameter to `logit(min(sigmoid(raw_alpha), 0.01))` and clears alpha optimizer moments.
- `csInitializeGaussiansFromPointCloud`: still exists for standalone point-buffer initialization, but the COLMAP training path now builds the initial `GaussianScene` on CPU.

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
- MCMC position noise remains available, but it is disabled by default because global opacity reset drives every splat into the low-opacity regime that the exploration gate treats as maximal-noise.
- Host guard:
  - if loss is non-finite, ADAM step is skipped and moments are reset.
- Host metrics:
  - `last_mse` stores the plain image MSE metric from the mixed-loss pass,
  - `avg_signal_max` is the mean of the latest target signal-max values cached per frame,
  - `last_psnr` is the current training frame's PSNR,
  - `avg_psnr` is the mean of the latest PSNR cached per frame,
  - `avg_loss` remains a rolling mean over one full image cycle for UI/CLI readability.

## CLI
Example:

```powershell
python cli.py train-colmap --colmap-root dataset/garden --images-subdir images_4 --iters 100 --max-gaussians 50000
```

Useful options:
- `--width/--height` for train resolution (defaults to selected image resolution),
- `--lr`, `--beta1`, `--beta2`, `--eps`,
- `--scale-l2` for autodiff log-scale regularization around the init/reference scale,
- `--lambda-dssim` for the SSIM contribution in the mixed photometric loss,
- `--max-anisotropy` for the hard per-gaussian scale-ratio cap,
- `--grad-clip`, `--grad-norm-clip`, `--max-update`,
- `--min-scale`, `--max-scale`, `--min-opacity`, `--max-opacity`.

## Regression Test
- `tests/test_training_garden_regression.py` loads the tracked `dataset/garden` subset, initializes gaussians from the COLMAP point cloud with a fixed seed, forces `opacity_reset_interval = 1000`, and runs exactly `2000` training steps.
- The test asserts on the final cached `avg_psnr >= 25 dB`, so one full post-reset recovery window is part of the regression instead of being hidden by an earlier peak.
- `last_psnr` is still recorded for diagnostics, but the regression gate uses the per-frame cached average to avoid single-view cherry-picking.

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
  - live loss and rolling-average display.
