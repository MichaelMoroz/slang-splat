# Rendering Pipeline

`src/renderer/gaussian_renderer.py` executes a five-stage compute pipeline.
Prepass scheduling is GPU-driven via indirect dispatch arguments generated from the GPU list counter.

## Uniform Parameter Layout
- Camera parameters and camera-space math are centralized in `shaders/renderer/camera.slang`:
  - `CameraParams`
  - `ICamera`
  - `PinholeCamera`
- Shared renderer parameters remain grouped in `shaders/renderer/gaussian_types.slang`:
  - `g_Camera` (`CameraParams`) for camera basis/position, anisotropic intrinsics (`focalPixels: float2`, `principalPoint: float2`), clip range, and lens distortion.
  - `g_Prepass` (`PrepassParams`) for splat counts, tile/depth packing, prepass capacities, and sampled-5 MVEE controls.
  - `g_Raster` (`RasterParams`) for raster resolution, alpha/transmittance thresholds, background, and debug overlays.
- Python-side raster layout defaults are sourced from `shaders/renderer/gaussian_types.slang` by parsing the `static const uint` raster constants instead of duplicating them manually in `GaussianRenderer`.
- Python bindings in `GaussianRenderer` mirror this layout by binding these structs per dispatch so stage code only reads structured fields instead of a large flat uniform list.

## 1. Project and Bin
- Shader: `csProjectAndBin`
- For each splat:
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
- Shader: `csRasterize`.
- Raster execution is microtiled: one `8x8` thread group covers one `24x24` effective raster tile, and each thread owns a fixed `3x3` pixel block in registers.
- Each thread resolves one tile range for its microtile, reuses each staged gaussian across all `9` local pixels, and writes per-pixel output after the forward replay.
- The inner loop performs front-to-back blending with exponential radial falloff while reusing gaussian data already staged in shared memory.
- Writes RGBA output texture.
- Primary ray generation goes through `PinholeCamera.screen_to_world_ray(...)`.

## 6. Raster Backward
- Shaders: `csClearRasterGrads`, `csRasterizeForwardBackward`.
- `csClearRasterGrads` zeros per-splat gradient buffers for:
  - `g_SplatPosLocal`
  - `g_SplatInvScale`
  - `g_SplatQuat`
  - `g_ScreenColorAlpha`
- `csRasterizeForwardBackward` recomputes the forward tile traversal inside the gradient pass, then walks the same batches in reverse, so no per-pixel forward-state textures are needed.
- The reverse pass reuses one staged gaussian per thread-group lane, accumulates the `3x3` microtile's pixel contributions in registers, and emits one shared gradient accumulation per microtile/splat pair before the shared cache is flushed globally.
- Global and groupshared raster loads are implemented via custom derivative functions; the shared gradient cache still flushes into flattened per-splat `float4` gradient buffers (`index = splat_id * 4 + component`).
- Output gradients are supplied through `g_OutputGrad` (`Texture2D<float4>`), and chain-rule terms include gamma output mapping and alpha output (`1 - transmittance`).

## 7. Training Stage
- Shader: `shaders/renderer/gaussian_training_stage.slang`.
- Kernels:
  - `csClearLossAndGradTex`: clears `g_OutputGrad` and scalar loss buffer.
  - `csComputeMSELossGrad`: computes RGB MSE with autodiff, writes output gradients, and reduces the target signal max used for PSNR.
  - `csAdamStepFused`: one-thread-per-splat fused ADAM update over position, scale, quaternion, color, and opacity.
- Stability measures in `csAdamStepFused` include:
  - finite-value sanitization,
  - gradient clipping (component and norm),
  - update clipping,
  - position/scale/opacity/color range clamps,
  - quaternion renormalization with identity fallback.

## Stats Notes
- `generated_entries` / `written_entries` are reported with one-frame latency (`stats_latency_frames = 1`).
- `stats_valid` indicates whether delayed stats are available yet (warm-up frame returns `False`).
- Prepass key/value/scanline capacity is bounded by `max_prepass_memory_mb`; stats expose `prepass_entry_cap`, `max_list_entries`, `max_scanline_entries`, and `capacity_limited`.
