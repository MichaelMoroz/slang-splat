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
- SH extra coefficients: `f_rest_*` (optional, loaded up to SH3 / 16 total coefficients)
- Log scales: `scale_*` (optional, defaults to `0`)
- Rotation quaternion: `rot_*` (optional, defaults to identity)

## Runtime Conversion
- Scales: preserved as 3DGS log-scale values
- Opacity: `sigmoid(raw_opacity)`
- Rotation: normalized quaternion
- SH coefficients: padded or truncated to the runtime-supported SH3 layout with `16` coefficients total
- Display color: `0.5 + SH_C0 * SH_DC`, clamped to `[0, 1]`

## Export Notes
- `save_gaussian_ply(...)` always writes the required SH DC triplet so exported files stay loadable as gaussian splat scenes.
- Higher-order SH payload is optional on save: when SH export is disabled, the writer omits every `f_rest_*` property and exports DC-only color.

Output is `GaussianScene` with contiguous `float32` arrays.

## COLMAP Loader Notes
- Supported camera models:
  - `SIMPLE_PINHOLE` (id `0`)
  - `PINHOLE` (id `1`)
  - `SIMPLE_RADIAL` (id `2`)
  - `RADIAL` (id `3`)
  - `OPENCV` (id `4`)
  - `FULL_OPENCV` (id `6`)
- With the default sparse layout setting, COLMAP reconstruction files may live under `sparse/0`, directly under `sparse`, directly under the selected root, or in a one-level named child sparse export such as `sparse-cubic-fixed/sparse`.
- Default training image lookup tries `images_4`, `images`, and then the reconstruction root. If the sparse model was discovered in a named child folder, that folder is searched too.
- Radial distortion is preserved as per-camera `k1` / `k2` coefficients and is consumed by both screen-space projection and raster ray generation.
- Camera intrinsics are scaled from COLMAP camera resolution to selected training image resolution.
- `initialize_scene_from_colmap_points(...)` converts the COLMAP point cloud directly into a trainable `GaussianScene`, using local point-neighborhood covariance eigenframes for gaussian rotation and anisotropy while keeping nearest-neighbor spacing as the overall scale reference before storing 3DGS log-scales.
- Pointcloud-based COLMAP initialization filters sparse points by the importer-selected minimum camera-observation threshold before direct seeding, diffused resampling, and point-spacing heuristics are computed.
- `resolve_colmap_init_hparams(...)` derives the default COLMAP init bundle from point-cloud spacing and requested gaussian count, and both the CLI and viewer pass that resolved bundle through unchanged.
- Point XYZ/RGB table extraction is centralized so viewer uploads, init heuristics, and scene initialization all consume the same data path.
- `sample_colmap_diffused_points(...)` synthesizes viewer-side resampled points by drawing source points with replacement and offsetting each sample with a Gaussian shaped by the local covariance of that point's nearest eight sparse neighbors, scaled by `diffusion_radius`.
- `sample_colmap_fibonacci_sphere_points(...)` builds optional shell points around the arithmetic mean of the aligned COLMAP camera centers, resolves the shell radius from a UI multiplier times the max aligned COLMAP point distance from that center, and applies deterministic radial jitter up to 10% per point to reduce view-dependent ordering aliasing. Those shell points now flow through the same covariance-based point initializer as the other non-PLY point sources.
- The viewer import option `Initialize Colors From Images` runs after the initial GPU scene upload. It projects every initialized splat into each imported training image, keeps the nearest camera with a valid in-image projection, bilinearly samples that image, writes the color into SH0/DC, and clears higher stored SH coefficients. Splats without any valid projection keep their existing initialization color.
