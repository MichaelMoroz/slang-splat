# Rendering Pipeline

`src/renderer/gaussian_renderer.py` executes a four-stage compute pipeline.

## 1. Project and Bin
- Shader: `csProjectAndBin`
- For each splat:
  - project to screen space,
  - estimate projected radius,
  - append `(tile_id, depth)` key and splat index for overlapping tiles.
- Output buffers:
  - projected splat data for raster stage,
  - unsorted key/value list,
  - append counter.

## 2. Sort
- Uses `src/sort/radix_sort.py`.
- Sorts key/value pairs by packed key so records are grouped by tile and ordered by depth.

## 3. Tile Ranges
- Shaders: `csClearTileRanges`, `csBuildTileRanges`.
- Builds `[start, end)` index ranges for each tile over sorted key/value buffers.

## 4. Rasterize
- Shader: `csRasterize`.
- Each pixel reads its tile range and blends splats front-to-back with exponential radial falloff.
- Writes RGBA output texture.
