# Rendering Pipeline

`src/renderer/gaussian_renderer.py` executes a five-stage compute pipeline.
Prepass scheduling is GPU-driven via indirect dispatch arguments generated from the GPU list counter.

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
- Each pixel reads its tile range and blends splats front-to-back with exponential radial falloff.
- The inner loop performs cheap screen-space reject checks before expensive local-space Gaussian math to reduce work on heavy tiles.
- Writes RGBA output texture.

## 6. Raster Backward
- Shaders: `csClearRasterGrads`, `csRasterizeBackward`.
- `csClearRasterGrads` zeros per-splat gradient buffers for:
  - `g_SplatPosLocal`
  - `g_SplatInvScale`
  - `g_SplatQuat`
  - `g_ScreenColorAlpha`
- `csRasterizeBackward` replays the forward tile batch loop to recover terminal per-pixel blend state and processed prefix length, then traverses batches in reverse.
- The reverse pass manually differentiates the batch loop and uses `bwd_diff(...)` only inside each per-splat reverse step.
- Global and groupshared raster loads are implemented via custom derivative functions; their backward functions atomically add gradients into flattened per-splat `float4` gradient buffers (`index = splat_id * 4 + component`).
- Output gradients are supplied through `g_OutputGrad` (`Texture2D<float4>`), and chain-rule terms include gamma output mapping and alpha output (`1 - transmittance`).

## Stats Notes
- `generated_entries` / `written_entries` are reported with one-frame latency (`stats_latency_frames = 1`).
- `stats_valid` indicates whether delayed stats are available yet (warm-up frame returns `False`).
- Prepass key/value/scanline capacity is bounded by `max_prepass_memory_mb`; stats expose `prepass_entry_cap`, `max_list_entries`, `max_scanline_entries`, and `capacity_limited`.
