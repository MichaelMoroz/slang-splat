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

`GaussianScene.refinable` is an optional boolean splat mask. When absent, loaders and concatenation treat every splat as refinable. Viewer COLMAP imports can set this independently for pointcloud, diffused, custom PLY, custom mesh, and Fibonacci shell initialization sources.

## COLMAP Loader Notes
- Supported camera models:
  - `SIMPLE_PINHOLE` (id `0`)
  - `PINHOLE` (id `1`)
  - `SIMPLE_RADIAL` (id `2`)
  - `RADIAL` (id `3`)
  - `OPENCV` (id `4`)
  - `OPENCV_FISHEYE` (id `5`)
  - `FULL_OPENCV` (id `6`)
  - `SIMPLE_RADIAL_FISHEYE` (id `8`)
  - `RADIAL_FISHEYE` (id `9`)
  - `SIMPLE_FISHEYE` (id `14`)
  - `FISHEYE` (id `15`)
  - `EQUIRECTANGULAR` (id `17`)
- With the default sparse layout setting, COLMAP reconstruction files may live under `sparse/0`, directly under `sparse`, directly under the selected root, or in a one-level named child sparse export such as `sparse-cubic-fixed/sparse`.
- Default training image lookup tries `images_4`, `images`, and then the reconstruction root. If the sparse model was discovered in a named child folder, that folder is searched too.
- Radial and OPENCV distortion terms are preserved per camera and consumed by both screen-space projection and raster ray generation.
- `EQUIRECTANGULAR` cameras use COLMAP's two metadata parameters (`width`, `height`) and do not carry focal, principal point, or distortion values. Imported training frames keep those fields at zero and select spherical camera projection from the stored model id.
- Fisheye models (`OPENCV_FISHEYE`, `SIMPLE_RADIAL_FISHEYE`, `RADIAL_FISHEYE`, and the coefficient-free upstream `SIMPLE_FISHEYE`/`FISHEYE`, which reduce to the pure equidistant mapping) use Kannala-Brandt theta-polynomial semantics: `thetaD = theta * (1 + k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8)` with `theta = atan2(|xy|, z)`, so lenses beyond 180 degrees remain well defined. The theta coefficients ride in the standard `k1`..`k4` fields (dimensionless — they do not rescale with image size) and select fisheye camera projection from the stored model id.
- The importer's `Fisheye Mask FOV (deg)` setting (0 = off, auto-detected from the dark image border on COLMAP root selection) excludes pixels outside the image circle implied by `thetaD(fov/2)` from the training loss, SSIM, metrics, and refinement statistics. The mask is a training-time geometric test in normalized sensor UV (exact under crop, subsample, and downscale) driven by two shader uniforms; the dataset images and BC7 caches are never modified and no Target Alpha mode is required.
- Camera intrinsics are scaled from COLMAP camera resolution to selected training image resolution.
- `initialize_scene_from_colmap_points(...)` converts the COLMAP point cloud directly into a trainable `GaussianScene`, using local point-neighborhood covariance eigenframes for gaussian rotation and anisotropy while keeping nearest-neighbor spacing as the overall scale reference before storing 3DGS log-scales.
- Pointcloud-based COLMAP initialization filters sparse points by the importer-selected minimum camera-observation threshold before direct seeding, diffused resampling, and point-spacing heuristics are computed.
- `resolve_colmap_init_hparams(...)` derives the default COLMAP init bundle from point-cloud spacing and requested gaussian count, and both interactive and headless viewer imports pass that resolved bundle through unchanged.
- Point XYZ/RGB table extraction is centralized so viewer uploads, init heuristics, and scene initialization all consume the same data path.
- `sample_colmap_diffused_points(...)` synthesizes viewer-side resampled points by drawing source points with replacement and offsetting each sample with a Gaussian shaped by the local covariance of that point's nearest eight sparse neighbors, scaled by `diffusion_radius`.
- `sample_colmap_fibonacci_sphere_points(...)` builds optional shell points around the arithmetic mean of the aligned COLMAP camera centers, resolves the shell radius from a UI multiplier times the max aligned COLMAP point distance from that center, and applies deterministic radial jitter up to 10% per point to reduce view-dependent ordering aliasing. The sampler can optionally restrict those synthesized points to the upper hemisphere above the shell center. Those shell points now flow through the same covariance-based point initializer as the other non-PLY point sources.
- The viewer import option `Initialize Colors From Images` runs after the initial GPU scene upload. It projects every initialized splat into each imported training image, keeps the nearest camera with a valid in-image projection, bilinearly samples that image, writes the color into SH0/DC, and clears higher stored SH coefficients. Splats without any valid projection keep their existing initialization color.
