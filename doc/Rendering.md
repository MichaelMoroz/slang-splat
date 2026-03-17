# Rendering Pipeline

`src/renderer/gaussian_renderer.py` is the public facade over renderer resource allocation, dispatch orchestration, and readback helpers.
Prepass scheduling is GPU-driven via indirect dispatch arguments generated from the GPU list counter.

## Uniform Parameter Layout
- Camera parameters and camera-space math are centralized in `shaders/utility/math/camera.slang`:
  - `CameraParams`
  - `ICamera`
  - `PinholeCamera`
- Shared splat data structures and reusable projection/raster math live under `shaders/utility/splatting`, while renderer bindings remain grouped in `shaders/renderer/gaussian_types.slang`:
  - `g_Camera` (`CameraParams`) for camera basis/position, anisotropic intrinsics (`focalPixels: float2`, `principalPoint: float2`), clip range, and lens distortion.
  - `g_Prepass` (`PrepassParams`) for splat counts, tile/depth packing, prepass capacities, and sampled-5 MVEE controls.
  - `g_Raster` (`RasterParams`) for raster resolution, alpha/transmittance thresholds, background, and debug overlays.
- Shared constants stay in `shaders/utility/math/constants.slang`, organized with commented sections for generic numeric floors, rendering constants, and debug-visualization tuning instead of being split into tiny constants-only files.
- Python-side raster layout defaults are sourced from `shaders/renderer/gaussian_types.slang` by parsing the `static const uint` raster constants instead of duplicating them manually in `GaussianRenderer`.
- Python bindings in `GaussianRenderer` mirror this layout by building reusable grouped binding dictionaries for scene buffers, prepass uniforms, raster uniforms, and readback state.
- Stored gaussian scale follows 3DGS semantics (`log(sigma)` per axis). Rendering decodes `exp(log_scale)` and converts sigma to finite-support ellipsoid radius with `radius_scale * 3.0`.

## 1. Project and Bin
- Shader: `csProjectAndBin`
- For each splat:
  - decode the stored 3DGS log-scale to sigma, convert it to finite-support ellipsoid radius with `radius_scale * 3.0`, and use that same support consistently for projection, binning, and raster evaluation,
  - project to screen space with sampled-5 MVEE fitting,
  - if sampled-5 fitting is unstable, use an analytic depth/scale fallback radius instead of hard max-radius fallback,
  - estimate projected radius,
  - solve scanline spans and reserve final key/value slots once per splat,
  - write per-splat list base offset,
  - append scanline work items for overlapping tile spans with precomputed per-splat local offsets.
- Output buffers:
  - projected splat data for raster stage,
  - scanline work item append buffer,
  - append counter.

## 2. Compose Key/Value Per Scanline
- Shader: `csComposeScanlineKeyValues`
- One thread handles one scanline work item and writes final `(tile_id, depth)` and splat index entries using `splat_base + local_scanline_offset`.
- Atomic model:
  - project/bin uses one atomic per splat to reserve scanline work items,
  - project/bin uses one atomic per splat to reserve final tile-entry slots,
  - compose uses no global reservation atomics.
- Dispatch is indirect from `g_ScanlineCounter`.

## 3. Sort
- Uses `src/sort/radix_sort.py`.
- Sorts key/value pairs by packed key so records are grouped by tile and ordered by depth.
- Dispatch sizes are generated on GPU from the append counter; no same-frame CPU readback is required for sort sizing.

## 4. Tile Ranges
- Shaders: `csClearTileRanges`, `csBuildTileRanges`.
- Builds `[start, end)` index ranges for each tile over sorted key/value buffers.
- Build dispatch is indirect and uses the same GPU-generated clamped count as sorting.

## 5. Rasterize
- Shader: `csRasterize` in `shaders/renderer/gaussian_raster_stage.slang`.
- Raster execution uses fixed `8x8` tiles: one `8x8` thread group covers one raster tile, and each thread owns exactly one pixel.
- Each thread resolves the tile range for its pixel, reuses each staged gaussian loaded from the prepass raster cache for that single forward replay, and writes one output pixel.
- The inner loop performs front-to-back blending with exponential radial falloff while reusing gaussian data already staged in shared memory.
- Shared gaussian staging uses `256`-splat batches.
- Raster evaluation uses the true decoded support cached in prepass with no separate pixel-floor clamp or fallback alpha branch.
- Debug processed-count, grad-norm, and ellipse-outline views are handled in the same forward replay loop as normal rendering rather than by a separate debug pass.
- Writes RGBA output texture.
- Primary ray generation goes through `PinholeCamera.screen_to_world_ray(...)`.

## 6. Debug Raster
- Debug modes are integrated into `csRasterize`.
- Processed-count mode outputs the per-pixel count of splats that successfully blended in the forward replay, normalized in log space against `maxSplatSteps` and visualized with the same `jet` colormap used by grad-norm mode.
- Grad-norm mode keeps the normal alpha/transmittance path and replaces each splat's RGB with a colormapped value derived from `g_DebugGradNorm[splatId]`, which is optionally produced by the optimizer as one packed-gradient `L2` norm per splat.
- Ellipse mode reuses the same tiled forward replay and switches splat alpha evaluation from filled-gaussian coverage to a symmetric antialiased outline band derived from the projected conic's signed edge distance. `debugEllipseThicknessPx` is the full outline thickness in pixels, and the default is `2`, so the visible ring spans both sides of the projected ellipse boundary instead of being clipped to the interior.

## 7. Raster Backward
- Shaders: `csClearRasterGrads`, `csRasterizeTrainingForward`, `csRasterizeBackward`, `csBackpropCachedRasterGrads`.
- Float and fixed cached-gradient variants are compiled from the same `shaders/renderer/gaussian_raster_stage.slang` source. The host loads mode-specific suffixed entrypoints from that file rather than keeping separate shader sources.
- `csClearRasterGrads` zeros the float packed gradient buffer plus both cached-raster intermediate buffers: the float-atomic buffer and the fixed-point fallback/debug buffer.
- `csRasterizeTrainingForward` runs the raster forward path for the fixed-count trainer, writes the rendered output, and caches one per-pixel forward state record plus `processedEnd` for backward replay.
- `csRasterizeBackward` is a pure backward replay kernel: it loads the cached per-pixel forward state, derives `dLoss / dRasterState` from `g_OutputGrad`, walks the staged splats in reverse without replaying forward internally, and accumulates cached raster-field gradients into the selected intermediate buffer.
- The reverse pass reuses one staged gaussian per thread-group lane, accumulates one pixel's contributions in registers, and writes them into either the float-atomic buffer or the fixed-point quantized buffer, depending on the renderer setting.
- `csBackpropCachedRasterGrads` runs one thread per splat, reads the active cached-raster gradient record inline, backprops cached ellipsoid geometry through `build_cached_ellipsoid`, backprops cached opacity through the raw-opacity helper, and writes final float scene-parameter gradients into `g_ParamGrads` with the caller-provided final scale.
- Output gradients are supplied through `g_OutputGrad` (`StructuredBuffer<float4>`) using flat pixel indexing `y * width + x`, and chain-rule terms include gamma output mapping and alpha output (`1 - transmittance`).

## 8. Training Stage
- Shader: `shaders/renderer/gaussian_training_stage.slang`.
- Kernels:
  - `csClearLossBuffer`: clears the scalar loss buffer for the current fixed-count step.
  - `csComputeL1LossForward`: computes RGB L1 loss and RGB MSE only, reducing them into the scalar metrics buffer used by the host.
  - `csComputeL1LossBackward`: computes only the image-space RGB L1 gradient into `g_OutputGrad`.
- The fixed-count trainer runs forward as `rasterize -> loss forward`, then backward as `loss backward -> raster backward -> optimizer`, so the training path keeps distinct forward and backward kernels while still reusing the packed-parameter optimizer path.

## 9. Debug Histograms
- `src/metrics.py` now exposes both single log10 histograms and grouped per-parameter log10 histograms for generic float tensors laid out as `tensor[param_id * item_count + item_id]`.
- The grouped tensor histogram kernel buckets `log10(abs(value))`, ignoring zeros and non-finite values.
- Cached ellipse gradient histogramming uses that generic float-tensor path directly in float atomic mode.
- In fixed atomic mode, the renderer decodes the quantized cached gradient buffer into a float scratch buffer first, then dispatches the same grouped histogram utility.

## Stats Notes
- `generated_entries` / `written_entries` are reported with one-frame latency (`stats_latency_frames = 1`).
- `stats_valid` indicates whether delayed stats are available yet (warm-up frame returns `False`).
- Prepass key/value/scanline capacity is bounded by `max_prepass_memory_mb`; stats expose `prepass_entry_cap`, `max_list_entries`, `max_scanline_entries`, and `capacity_limited`.
- The CPU reference path in `reference_impls/reference_cpu.py` shares a single axis-parameterized scanline span solver for both X-major and Y-major ellipse traversal.
