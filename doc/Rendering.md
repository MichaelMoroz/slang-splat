# Rendering Pipeline

`src/renderer/gaussian_renderer.py` executes a four-stage compute pipeline.
Prepass scheduling is GPU-driven via indirect dispatch arguments generated from the GPU list counter.

## 1. Project and Bin
- Shader: `csProjectAndBin`
- For each splat:
  - project to screen space with sampled-5 MVEE fitting,
  - estimate projected radius,
  - append `(tile_id, depth)` key and splat index for overlapping tiles.
- Output buffers:
  - projected splat data for raster stage,
  - unsorted key/value list,
  - append counter.

## 2. Sort
- Uses `src/sort/radix_sort.py`.
- Sorts key/value pairs by packed key so records are grouped by tile and ordered by depth.
- Dispatch sizes are generated on GPU from the append counter; no same-frame CPU readback is required for sort sizing.

## 3. Tile Ranges
- Shaders: `csClearTileRanges`, `csBuildTileRanges`.
- Builds `[start, end)` index ranges for each tile over sorted key/value buffers.
- Build dispatch is indirect and uses the same GPU-generated clamped count as sorting.

## 4. Rasterize
- Shader: `csRasterize`.
- Each pixel reads its tile range and blends splats front-to-back with exponential radial falloff.
- The inner loop performs cheap screen-space reject checks before expensive local-space Gaussian math to reduce work on heavy tiles.
- Writes RGBA output texture.

## Stats Notes
- `generated_entries` / `written_entries` are reported with one-frame latency (`stats_latency_frames = 1`).
- `stats_valid` indicates whether delayed stats are available yet (warm-up frame returns `False`).
