# COLMAP Training Pipeline

`cli.py` (`train-colmap`) and `src/training/gaussian_trainer.py` implement a basic 3DGS optimization loop over COLMAP reconstructions.

## Data Ingestion
- Loader: `src/scene/colmap_loader.py`
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
- Frame target textures are stored on GPU as `rgba8_unorm` (both native and train resolution) to reduce memory usage.

## Initialization
- Gaussians are initialized by random point-cloud sampling from COLMAP `points3D`.
- Viewer COLMAP reinitialization path uploads COLMAP point tables once, then runs `csInitializeGaussiansFromPointCloud` to rebuild gaussian parameters directly on GPU.
- Initialization parameters:
  - count cap (`max_gaussians`, default `50000`),
  - position jitter,
  - base scale + scale jitter,
  - constant initial opacity,
  - color from COLMAP RGB.

## Optimization Loop
Each trainer `step()` performs:
1. Pick one random training frame.
2. Upload frame image as target texture (Y-flip enabled by default).
3. Run renderer prepass + raster forward.
4. Run loss kernel (`RGB MSE`) to produce `g_OutputGrad`.
5. Run fused raster forward/backward replay to fill per-splat gradient buffers without cached per-pixel forward state.
6. Run fused ADAM kernel (`csAdamStepFused`) with one thread per Gaussian.
   - Scale anisotropy regularization is computed in-kernel via Slang autodiff on `scale.xyz`.
   - The anisotropy term uses smooth pairwise log-scale ratio penalties with threshold ratio `8`.
   - The regularization scalar is added to the reported loss buffer in this ADAM pass.
   - Post-ADAM decoupled scale L2 decay is applied on `scale.xyz`:
     - `scale *= max(1 - scale_lr * scale_l2_weight, 0)`
7. Run low-quality marking kernel (`csMarkLowQualitySplats`) using stability thresholds.
8. Run random low-quality resample kernel (`csResampleLowQualitySplatsRandom`).

## Kernels
- `csClearLossAndGradTex`: zero loss + output-grad texture.
- `csComputeMSELossGrad`: computes RGB MSE via Slang autodiff, writes output gradients, and reduces the target signal max used for PSNR.
- `csAdamStepFused`: updates all trainable params in one pass:
  - position,
  - scale,
  - quaternion,
  - color,
  - opacity.
  - Adds autodiff anisotropy regularization gradients to scale gradients (`g_ScaleAnisoWeight`, threshold ratio `8`).
  - Accumulates anisotropy scalar contribution into `g_LossBuffer`.
  - Then applies post-ADAM decoupled scale L2 decay controlled by `g_ScaleL2Weight`.
- `csMarkLowQualitySplats`: marks splats as low-quality when
  - `opacity <= min_opacity`, or
  - `max(scale.xyz) <= min_scale`.
- `csResampleLowQualitySplatsRandom`: for each marked splat, picks one deterministic random donor; if donor is valid,
  copies donor params, adds optional MCMC-scale position jitter, and resets optimizer moments.
- `csInitializeGaussiansFromPointCloud`: initializes scene buffers and optimizer moments from preuploaded COLMAP point buffers.

## Numerical Reinforcement
- Loss/grad and optimizer math sanitize non-finite values.
- Gradient clipping:
  - per-component clip,
  - norm clip.
- Update clipping (`max_update`).
- Parameter bounds:
  - position absolute clamp,
  - scale min/max clamp,
  - opacity min/max clamp,
  - color clamp to `[0, 1]`.
- Quaternion normalization each step with identity fallback.
- Host guard:
  - if loss is non-finite, ADAM step is skipped and moments are reset.
- Host metrics:
  - `last_mse` stores the image MSE from `csComputeMSELossGrad`,
  - `ema_signal_max` tracks a slow EMA of per-frame target max intensity,
  - `ema_psnr` reports a more heavily smoothed PSNR for UI/CLI display.

## CLI
Example:

```powershell
python cli.py train-colmap --colmap-root dataset/garden --images-subdir images_4 --iters 100 --max-gaussians 50000
```

Useful options:
- `--width/--height` for train resolution (defaults to selected image resolution),
- `--lr`, `--beta1`, `--beta2`, `--eps`,
- `--scale-l2` for post-ADAM decoupled scale weight decay,
- `--scale-aniso` for autodiff scale anisotropy regularization strength,
- `--grad-clip`, `--grad-norm-clip`, `--max-update`,
- `--min-scale`, `--max-scale`, `--min-opacity`, `--max-opacity`.
- `--[no-]low-quality-reinit` enables/disables per-step low-quality resampling.

## Viewer Integration
- `viewer.py` keeps PLY workflow and adds COLMAP training controls:
  - load COLMAP folder,
  - set train image directory,
  - initialize training scene,
  - start/stop one-step-per-frame optimization,
  - live loss and EMA display.
