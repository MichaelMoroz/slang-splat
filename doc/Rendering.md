# Rendering Pipeline

`src/renderer/gaussian_renderer.py` executes a five-stage compute pipeline.
Prepass scheduling is GPU-driven via indirect dispatch arguments generated from the GPU list counter.

## 1. Project and Bin
- Shader: `csProjectAndBin`
- For each splat:
  - project to screen space with sampled-5 MVEE fitting,
  - estimate projected radius,
  - append scanline work items for overlapping tile spans.
- Output buffers:
  - projected splat data for raster stage,
  - scanline work item append buffer,
  - append counter.

## 2. Compose Key/Value Per Scanline
- Shader: `csComposeScanlineKeyValues`
- One thread handles one scanline work item and expands it into final `(tile_id, depth)` and splat index entries.
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

## Stats Notes
- `generated_entries` / `written_entries` are reported with one-frame latency (`stats_latency_frames = 1`).
- `stats_valid` indicates whether delayed stats are available yet (warm-up frame returns `False`).
