# Scene Loading

`src/scene/ply_loader.py` loads Gaussian splat scenes from PLY files.

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
