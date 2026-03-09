# Scene Loading

`src/scene` supports:
- `src/scene/ply_loader.py` for Gaussian splat PLY scenes.
- `src/scene/colmap_loader.py` as the stable COLMAP facade, backed by:
  - `src/scene/_internal/colmap_binary.py` for binary parsing,
  - `src/scene/_internal/colmap_ops.py` for frame building and init heuristics,
  - `src/scene/_internal/colmap_types.py` for COLMAP dataclasses and shared point-table extraction.

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
- `initialize_scene_from_colmap_points(...)` converts the COLMAP point cloud directly into a trainable `GaussianScene`, using nearest-neighbor point spacing as the initial scale reference and repeating it across XYZ.
- `resolve_colmap_init_hparams(...)` derives the default COLMAP init bundle from point-cloud spacing and requested gaussian count, and both the CLI and viewer pass that resolved bundle through unchanged.
- Point XYZ/RGB table extraction is centralized so viewer uploads, init heuristics, and scene initialization all consume the same data path.
