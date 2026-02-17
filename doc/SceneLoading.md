# Scene Loading

`src/scene` supports:
- `src/scene/ply_loader.py` for Gaussian splat PLY scenes.
- `src/scene/colmap_loader.py` for COLMAP `sparse/0` reconstructions plus image-frame training metadata.

## Supported Attributes
- Position: `x`, `y`, `z`
- Opacity logit: `opacity`
- SH DC: `f_dc_0`, `f_dc_1`, `f_dc_2`
- SH extra coefficients: `f_rest_*` (optional)
- Log scales: `scale_*` (optional, defaults to `1`)
- Rotation quaternion: `rot_*` (optional, defaults to identity)

## Runtime Conversion
- Scales: `exp(raw_scale)`
- Opacity: `sigmoid(raw_opacity)`
- Rotation: normalized quaternion
- Display color: `0.5 + SH_C0 * SH_DC`, clamped to `[0, 1]`

Output is `GaussianScene` with contiguous `float32` arrays.

## COLMAP Loader Notes
- Supported camera models:
  - `SIMPLE_PINHOLE` (id `0`)
  - `PINHOLE` (id `1`)
- Camera intrinsics are scaled from COLMAP camera resolution to selected training image resolution.
- `initialize_scene_from_colmap_points(...)` randomly samples COLMAP points into a trainable `GaussianScene`.
