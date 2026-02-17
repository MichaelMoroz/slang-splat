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

## Initialization
- Gaussians are initialized by random point-cloud sampling from COLMAP `points3D`.
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
5. Run raster backward to fill per-splat gradient buffers.
6. Run fused ADAM kernel (`csAdamStepFused`) with one thread per Gaussian.

## Kernels
- `csClearLossAndGradTex`: zero loss + output-grad texture.
- `csComputeMSELossGrad`: output-space gradient for raster backward.
- `csAdamStepFused`: updates all trainable params in one pass:
  - position,
  - scale,
  - quaternion,
  - color,
  - opacity.

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

## CLI
Example:

```powershell
python cli.py train-colmap --colmap-root dataset/garden --images-subdir images_4 --iters 100 --max-gaussians 50000
```

Useful options:
- `--width/--height` for train resolution (defaults to selected image resolution),
- `--lr`, `--beta1`, `--beta2`, `--eps`,
- `--grad-clip`, `--grad-norm-clip`, `--max-update`,
- `--min-scale`, `--max-scale`, `--min-opacity`, `--max-opacity`.

## Viewer Integration
- `viewer.py` keeps PLY workflow and adds COLMAP training controls:
  - load COLMAP folder,
  - set train image directory,
  - initialize training scene,
  - start/stop one-step-per-frame optimization,
  - live loss and EMA display.
