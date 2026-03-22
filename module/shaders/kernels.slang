import "slangpy";
import "common.slang";

uniform Tensor<float, 1> g_Splats;
uniform Tensor<uint, 1> g_SplatOrder;
uniform RWTensor<uint, 1> g_DistanceKeys;
uniform RWTensor<uint, 1> g_DistanceSortOrder;
uniform RWTensor<uint, 1> g_TileCounts;
uniform RWTensor<uint, 1> g_ScanlineCounts;
uniform Tensor<uint, 1> g_ScanlineOffsets;
uniform RWTensor<ScanlineEntry, 1> g_ScanlineEntries;
uniform RWTensor<uint, 1> g_ScanlineTileCounts;
uniform Tensor<uint, 1> g_ScanlineTileOffsets;
uniform RWTensor<uint, 1> g_TileEntries;
uniform Tensor<uint, 1> g_SortedEntries;
uniform RWTensor<ProjectionState, 1> g_ProjectionState;
uniform RWTensor<Gaussian3D, 1> g_RasterState;
uniform RWTensor<uint, 1> g_TileRanges;
uniform RWTensor<float4, 2> g_Output;
uniform Tensor<float4, 2> g_OutputGrad;
uniform RWTensor<float4, 2> g_ForwardState;
uniform RWTensor<uint, 2> g_ForwardEnd;
uniform AtomicTensor<float, 1> g_ParamGrads;
uniform Camera g_Camera;
uniform uint g_SplatCount;
uniform uint g_SortedEntryCount;
uniform uint2 g_TileGrid;
uniform float3 g_Background;
uniform float g_RadiusScale;
uniform float g_MaxAnisotropy;
uniform float g_AlphaCutoff;
uniform float g_TransmittanceThreshold;
uniform uint g_DebugMode;
uniform uint g_DebugMaxSplatSteps;

groupshared Gaussian3D gsGaussian[RASTER_BATCH];
groupshared uint gsBatchSplatId[RASTER_BATCH];

uint load_sorted_entry_tile(uint idx) {
    return g_SortedEntries[int(idx * 2u)];
}

uint load_sorted_entry_splat(uint idx) {
    return g_SortedEntries[int(idx * 2u + 1u)];
}

void store_tile_entry(uint idx, uint tileId, uint splatId) {
    g_TileEntries.store(int(idx * 2u), tileId);
    g_TileEntries.store(int(idx * 2u + 1u), splatId);
}

// Generate exact float ordering keys for front-to-back splat sorting.
[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csGenerateDistanceSortKeys(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;
    Gaussian3D gaussian = Gaussian3D.load(g_Splats, g_SplatCount, splatId);
    float distance = length(gaussian.position - g_Camera.camPos);
    g_DistanceKeys.store(int(splatId), float_to_ordered_uint(distance));
    g_DistanceSortOrder.store(int(splatId), splatId);
}

// Raster-state write helpers.
void write_cache(uint splatId, Gaussian3D gaussian, float fusedOpacity) {
    int[1] at = { int(splatId) };
    gaussian.colorAlpha.w = fusedOpacity;
    g_RasterState.store(at, gaussian);
}

// Projection helpers used by the scanline generation stages.
bool load_projected_ellipse(uint splatId, out Ellipse ellipse) {
    ProjectionState projection = g_ProjectionState[int(splatId)];
    ellipse.centerPx = projection.centerRadiusDepth.xy;
    ellipse.radiusPx = projection.centerRadiusDepth.z;
    ellipse.conic = projection.conicOpacity.xyz;
    ellipse.bboxExtent = 0.0;
    return ellipse.prepare_conic();
}

uint count_scanlines(Ellipse ellipse) {
    bool transpose;
    float2 localCenter, localExtent;
    float3 localConic;
    // Normalize the ellipse into the major-axis scanline frame before counting strips.
    ellipse.scanline_frame(transpose, localCenter, localExtent, localConic);
    uint2 localGrid = transpose ? g_TileGrid.yx : g_TileGrid;
    int minStrip = clamp(int(floor((localCenter.y - localExtent.y - TINY_VALUE) / TILE_SIZE)), 0, int(localGrid.y) - 1);
    int maxStrip = clamp(int(ceil((localCenter.y + localExtent.y) / TILE_SIZE)), 0, int(localGrid.y) - 1);
    return maxStrip >= minStrip ? uint(maxStrip - minStrip + 1) : 0u;
}

bool scanline_tile_span(ScanlineEntry scanline, out int spanTileMin, out int spanTileMax) {
    Ellipse ellipse;
    if (!load_projected_ellipse(scanline.splatId, ellipse)) {
        spanTileMin = 0;
        spanTileMax = -1;
        return false;
    }
    bool transpose = scanline.transpose != 0u;
    uint2 localGrid = transpose ? g_TileGrid.yx : g_TileGrid;
    float2 localCenter = transpose ? ellipse.centerPx.yx : ellipse.centerPx;
    float2 localExtent = transpose ? ellipse.bboxExtent.yx : ellipse.bboxExtent;
    float3 localConic = transpose ? float3(ellipse.conic.z, ellipse.conic.y, ellipse.conic.x) : ellipse.conic;
    int minSpan = clamp(int(floor((localCenter.x - localExtent.x - TINY_VALUE) / TILE_SIZE)), 0, int(localGrid.x) - 1);
    int maxSpan = clamp(int(ceil((localCenter.x + localExtent.x) / TILE_SIZE)), 0, int(localGrid.x) - 1);
    // Convert the stored strip index back into local strip coordinates for span evaluation.
    float stripMin = scanline.stripIndex * TILE_SIZE - localCenter.y;
    float stripMax = stripMin + TILE_SIZE;
    float spanMin, spanMax;
    if (!ellipse.span_for_strip(localConic, stripMin, stripMax, spanMin, spanMax)) {
        spanTileMin = 0;
        spanTileMax = -1;
        return false;
    }
    // Clamp the analytic span to the ellipse bbox before emitting tile coordinates.
    spanTileMin = clamp(int(floor((spanMin + localCenter.x - TINY_VALUE) / TILE_SIZE)), minSpan, maxSpan);
    spanTileMax = clamp(int(floor((spanMax + localCenter.x) / TILE_SIZE)), minSpan, maxSpan);
    while (spanTileMin <= spanTileMax) {
        float tileMin = spanTileMin * TILE_SIZE - localCenter.x;
        float tileMax = tileMin + TILE_SIZE;
        float2 boxMin = transpose ? float2(stripMin, tileMin) : float2(tileMin, stripMin);
        float2 boxMax = transpose ? float2(stripMax, tileMax) : float2(tileMax, stripMax);
        if (ellipse.tile_intersects(boxMin, boxMax)) break;
        spanTileMin++;
    }
    while (spanTileMin <= spanTileMax) {
        float tileMin = spanTileMax * TILE_SIZE - localCenter.x;
        float tileMax = tileMin + TILE_SIZE;
        float2 boxMin = transpose ? float2(stripMin, tileMin) : float2(tileMin, stripMin);
        float2 boxMax = transpose ? float2(stripMax, tileMax) : float2(tileMax, stripMax);
        if (ellipse.tile_intersects(boxMin, boxMax)) break;
        spanTileMax--;
    }
    return spanTileMin <= spanTileMax;
}

// Pass 1: project splats, fit screen-space ellipses, and count emitted scanlines.
//TODO: this can be done via a module call, instead of a dispatch
[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csProjectScanlineCount(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint sortedSplatId = dispatchThreadID.x;
    if (sortedSplatId >= g_SplatCount) return;
    int[1] at = { int(sortedSplatId) };
    ProjectionState projection;
    projection.centerRadiusDepth = 0.0;
    projection.conicOpacity = 0.0;
    Gaussian3D rasterGaussian;
    rasterGaussian.position = 0.0;
    rasterGaussian.logScale = 0.0;
    rasterGaussian.rotation = 0.0;
    rasterGaussian.colorAlpha = 0.0;
    uint scanlineCount = 0u;

    uint splatId = g_SplatOrder[int(sortedSplatId)];
    Gaussian3D gaussian = Gaussian3D.load(g_Splats, g_SplatCount, splatId);
    float fusedOpacity = gaussian.build_raster_fused_opacity();
    if (fusedOpacity >= g_AlphaCutoff) {
        // Project only splats whose cutoff isosurface can contribute within the camera frustum.
        float supportSigmaRadius = sqrt(max(-2.0 * log(g_AlphaCutoff / fusedOpacity), 0.0));
        float3 cameraPosition = g_Camera.world_point_to_camera(gaussian.position);
        float depth = cameraPosition.z;
        float cameraDistance = length(gaussian.position - g_Camera.camPos);
        bool hasFit = depth > DEPTH_VISIBILITY_FLOOR
            && cameraDistance > g_Camera.nearDepth
            && cameraDistance < g_Camera.farDepth;
        Ellipse ellipse;
        if (hasFit) hasFit = compute_gaussian_outline_ellipse(gaussian, g_Camera, g_RadiusScale, supportSigmaRadius, g_MaxAnisotropy, ellipse);
        if (hasFit) {
            // Inflate the fitted ellipse before scan conversion so tile coverage stays conservative.
            ellipse.radiusPx = max(ellipse.radiusPx * ELLIPSE_SAFETY_SCALE + ELLIPSE_RADIUS_PAD_PX, MIN_SCREEN_RADIUS_PX);
            ellipse.bboxExtent = float2(ellipse.radiusPx, ellipse.radiusPx);
            hasFit = ellipse.prepare_conic();
        }
        if (hasFit) {
            projection.centerRadiusDepth = float4(ellipse.centerPx, ellipse.radiusPx, cameraDistance);
            projection.conicOpacity = float4(ellipse.conic, 1.0);
            rasterGaussian = gaussian;
            rasterGaussian.colorAlpha.w = fusedOpacity;
            scanlineCount = count_scanlines(ellipse);
        }
    }

    g_ProjectionState.store(at, projection);
    g_RasterState.store(at, rasterGaussian);
    g_ScanlineCounts.store(at, scanlineCount);
}

// Pass 2a: expand each projected ellipse into per-strip scanline work items.
[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csEmitScanlines(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;
    Ellipse ellipse;
    if (!load_projected_ellipse(splatId, ellipse)) return;
    uint scanlineCount = g_ScanlineCounts[int(splatId)];
    if (scanlineCount == 0u) return;
    bool transpose;
    float2 localCenter, localExtent;
    float3 localConic;
    ellipse.scanline_frame(transpose, localCenter, localExtent, localConic);
    uint2 localGrid = transpose ? g_TileGrid.yx : g_TileGrid;
    int minStrip = clamp(int(floor((localCenter.y - localExtent.y - TINY_VALUE) / TILE_SIZE)), 0, int(localGrid.y) - 1);
    uint writeIndex = g_ScanlineOffsets[int(splatId)];
    for (uint localStrip = 0u; localStrip < scanlineCount; localStrip++) {
        // Each work item stores the splat plus one strip index in the chosen scan frame.
        int[1] wi = { int(writeIndex + localStrip) };
        ScanlineEntry scanline;
        scanline.splatId = splatId;
        scanline.stripIndex = uint(minStrip) + localStrip;
        scanline.transpose = transpose ? 1u : 0u;
        g_ScanlineEntries.store(wi, scanline);
    }
}

// Pass 2b: count the number of tiles covered by each emitted scanline.
[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csEmitScanlineTileCounts(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint scanlineId = dispatchThreadID.x;
    if (scanlineId >= g_SortedEntryCount) return;
    ScanlineEntry scanline = g_ScanlineEntries[int(scanlineId)];
    int spanTileMin, spanTileMax;
    uint tileCount = scanline_tile_span(scanline, spanTileMin, spanTileMax) ? uint(spanTileMax - spanTileMin + 1) : 0u;
    g_ScanlineTileCounts.store(int[1](int(scanlineId)), tileCount);
}

// Pass 3: emit final (tileId, splatId) entries for sorting.
[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csEmitTileEntries(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint scanlineId = dispatchThreadID.x;
    if (scanlineId >= g_SortedEntryCount) return;
    ScanlineEntry scanline = g_ScanlineEntries[int(scanlineId)];
    int spanTileMin, spanTileMax;
    if (!scanline_tile_span(scanline, spanTileMin, spanTileMax)) return;
    uint writeIndex = g_ScanlineTileOffsets[int(scanlineId)];
    for (int span = spanTileMin; span <= spanTileMax; span++) {
        uint2 tileXY = scanline.transpose != 0u ? uint2(scanline.stripIndex, uint(span)) : uint2(uint(span), scanline.stripIndex);
        store_tile_entry(writeIndex, tileXY.y * g_TileGrid.x + tileXY.x, scanline.splatId);
        writeIndex++;
    }
}

// Pass 4: rebuild per-tile sorted-entry ranges after the host-side key sort.
[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csClearTileRanges(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint tileId = dispatchThreadID.x;
    uint tileCount = g_TileGrid.x * g_TileGrid.y;
    if (tileId >= tileCount) return;
    g_TileRanges.store(int(tileId * 2u), INVALID_START);
    g_TileRanges.store(int(tileId * 2u + 1u), 0u);
}

[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csBuildTileRanges(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint idx = dispatchThreadID.x;
    if (idx >= g_SortedEntryCount) return;
    uint tile = load_sorted_entry_tile(idx);
    uint prevTile = idx > 0u ? load_sorted_entry_tile(idx - 1u) : 0u;
    uint nextTile = idx + 1u < g_SortedEntryCount ? load_sorted_entry_tile(idx + 1u) : 0u;
    // g_TileRanges is flattened as [start0, end0, start1, end1, ...].
    if (idx == 0u || tile != prevTile) g_TileRanges.store(int(tile * 2u), idx);
    if (idx + 1u == g_SortedEntryCount || tile != nextTile) g_TileRanges.store(int(tile * 2u + 1u), idx + 1u);
}

// Raster batching helpers shared by forward and backward passes.
void prefetch_batch(uint batchStart, uint batchEnd, uint lane) {
    uint batchCount = batchEnd - batchStart;
    for (uint localIdx = lane; localIdx < batchCount; localIdx += RASTER_GROUP_THREADS) {
        uint splatId = load_sorted_entry_splat(batchStart + localIdx);
        gsBatchSplatId[localIdx] = splatId;
        gsGaussian[localIdx] = g_RasterState[int(splatId)];
    }
}

[Differentiable]
float4 shade_step(float3 accum, float trans, Gaussian3D gaussian, float alpha) {
    return float4(accum + trans * alpha * gaussian.colorAlpha.xyz, trans * (1.0 - alpha));
}

// Forward rasterization over sorted tile-local splat lists.
[shader("compute")]
[numthreads(RASTER_THREAD_TILE_DIM, RASTER_THREAD_TILE_DIM, 1)]
void csRasterForward(uint3 dispatchThreadID : SV_DispatchThreadID, uint lane : SV_GroupIndex) {
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= uint(g_Camera.viewport.x) || py >= uint(g_Camera.viewport.y)) return;
    uint tileId = (py / TILE_SIZE) * g_TileGrid.x + (px / TILE_SIZE);
    uint2 range = uint2(g_TileRanges[int(tileId * 2u)], g_TileRanges[int(tileId * 2u + 1u)]);
    float3 accum = 0.0;
    float trans = 1.0;
    uint processedEnd = range.x;
    float3 rayDir = g_Camera.screen_to_world_ray(float2(px, py) + 0.5);
    if (range.x != INVALID_START && range.y > range.x) {
        bool done = false;
        for (uint batchStart = range.x; batchStart < range.y; batchStart += RASTER_BATCH) {
            uint batchEnd = min(batchStart + RASTER_BATCH, range.y);
            uint batchCount = batchEnd - batchStart;
            // Batch splats into groupshared storage so every pixel in the tile reuses the same loads.
            prefetch_batch(batchStart, batchEnd, lane);
            GroupMemoryBarrierWithGroupSync();
            if (!done) {
                for (uint localIdx = 0u; localIdx < batchCount; localIdx++) {
                    processedEnd = batchStart + localIdx + 1u;
                    Gaussian3D gaussian = gsGaussian[localIdx];
                    // The forward loop only needs alpha and the fused color/opacity carried by Gaussian3D.
                    float alpha = raster_eval_alpha(gaussian, g_Camera, g_RadiusScale, g_MaxAnisotropy, rayDir);
                    if (alpha < g_AlphaCutoff) continue;
                    float4 blended = shade_step(accum, trans, gaussian, alpha);
                    accum = blended.xyz;
                    trans = blended.w;
                    if (trans < g_TransmittanceThreshold) { done = true; break; }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    int2 p = int2(int(px), int(py));
    uint processedCount = range.x != INVALID_START && processedEnd >= range.x ? (processedEnd - range.x) : 0u;
    float4 output = g_DebugMode == DEBUG_MODE_PROCESSED_COUNT
        ? float4(raster_processed_count_color(processedCount, g_DebugMaxSplatSteps), 1.0)
        : raster_output(accum, trans, g_Background);
    g_Output.store(p, output);
    g_ForwardState.store(p, float4(accum, trans));
    g_ForwardEnd.store(p, processedEnd);
}

// Atomic accumulation into the flat parameter-gradient buffer.
void add_param_grad(uint splatId, uint paramId, float value) {
    if (value == 0.0) return;
    g_ParamGrads.add(int(splatId * PARAM_COUNT + paramId), value);
}

// Reverse raster replay using the cached forward state and sorted tile ranges.
[shader("compute")]
[numthreads(RASTER_THREAD_TILE_DIM, RASTER_THREAD_TILE_DIM, 1)]
void csRasterBackward(uint3 dispatchThreadID : SV_DispatchThreadID, uint lane : SV_GroupIndex) {
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= uint(g_Camera.viewport.x) || py >= uint(g_Camera.viewport.y)) return;
    uint tileId = (py / TILE_SIZE) * g_TileGrid.x + (px / TILE_SIZE);
    uint2 range = uint2(g_TileRanges[int(tileId * 2u)], g_TileRanges[int(tileId * 2u + 1u)]);
    if (range.x == INVALID_START || range.y <= range.x) return;
    int2 p = int2(int(px), int(py));
    float4 state4 = g_ForwardState[p];
    var dpAccum = diffPair(state4.xyz);
    var dpTrans = diffPair(state4.w);
    bwd_diff(raster_output)(dpAccum, dpTrans, g_Background, g_OutputGrad[p]);
    float3 dAccum = dpAccum.getDifferential();
    float dTrans = dpTrans.getDifferential();
    float3 rayDir = g_Camera.screen_to_world_ray(float2(px, py) + 0.5);
    float3 accum = state4.xyz;
    float trans = state4.w;
    uint processedEnd = g_ForwardEnd[p];
    for (uint batchBase = range.y; batchBase > range.x;) {
        uint batchStart = batchBase > range.x + RASTER_BATCH ? batchBase - RASTER_BATCH : range.x;
        uint batchCount = batchBase - batchStart;
        // Replay the same tile-local batches in reverse order from the saved forward stopping point.
        prefetch_batch(batchStart, batchBase, lane);
        GroupMemoryBarrierWithGroupSync();
        for (uint rev = 0u; rev < batchCount; rev++) {
            uint localIdx = batchCount - 1u - rev;
            uint globalIdx = batchStart + localIdx;
            if (globalIdx >= processedEnd) continue;
            uint splatId = gsBatchSplatId[localIdx];
            Gaussian3D gaussian = gsGaussian[localIdx];
            uint originalSplatId = g_SplatOrder[int(splatId)];
            float alpha = raster_eval_alpha(gaussian, g_Camera, g_RadiusScale, g_MaxAnisotropy, rayDir);
            if (alpha < g_AlphaCutoff) continue;
            // Invert the forward blend step, then differentiate that exact primitive via bwd_diff(shade_step).
            float prevTrans = trans / max(1.0 - alpha, SMALL_VALUE);
            float3 prevAccum = accum - prevTrans * alpha * gaussian.colorAlpha.xyz;
            var dpPrevAccum = diffPair(prevAccum);
            var dpPrevTrans = diffPair(prevTrans);
            var dpGaussian = diffPair(gaussian);
            var dpAlpha = diffPair(alpha);
            bwd_diff(shade_step)(dpPrevAccum, dpPrevTrans, dpGaussian, dpAlpha, float4(dAccum, dTrans));
            Gaussian3D.Differential dGaussian = dpGaussian.getDifferential();
            float dAlpha = dpAlpha.getDifferential();
            dAccum = dpPrevAccum.getDifferential();
            dTrans = dpPrevTrans.getDifferential();
            var dpAlphaGaussian = diffPair(gaussian);
            // Propagate dAlpha back into position/scale/rotation through the ray-splat intersection path.
            bwd_diff(raster_eval_alpha)(dpAlphaGaussian, g_Camera, g_RadiusScale, g_MaxAnisotropy, rayDir, dAlpha);
            Gaussian3D.Differential dGaussianAlpha = dpAlphaGaussian.getDifferential();
            dGaussian.position += dGaussianAlpha.position;
            dGaussian.logScale += dGaussianAlpha.logScale;
            dGaussian.rotation += dGaussianAlpha.rotation;
            dGaussian.colorAlpha += dGaussianAlpha.colorAlpha;
            float fused = max(gaussian.colorAlpha.w, SMALL_VALUE);
            // Public opacity is stored on the Python side; convert the fused-opacity differential back to raw opacity.
            float dRawOpacity = dGaussian.colorAlpha.w * fused * (1.0 - fused);
            add_param_grad(originalSplatId, PARAM_COLOR_R, dGaussian.colorAlpha.x);
            add_param_grad(originalSplatId, PARAM_COLOR_G, dGaussian.colorAlpha.y);
            add_param_grad(originalSplatId, PARAM_COLOR_B, dGaussian.colorAlpha.z);
            add_param_grad(originalSplatId, PARAM_POSITION_X, dGaussian.position.x);
            add_param_grad(originalSplatId, PARAM_POSITION_Y, dGaussian.position.y);
            add_param_grad(originalSplatId, PARAM_POSITION_Z, dGaussian.position.z);
            add_param_grad(originalSplatId, PARAM_SCALE_X, dGaussian.logScale.x);
            add_param_grad(originalSplatId, PARAM_SCALE_Y, dGaussian.logScale.y);
            add_param_grad(originalSplatId, PARAM_SCALE_Z, dGaussian.logScale.z);
            add_param_grad(originalSplatId, PARAM_ROTATION_W, dGaussian.rotation.x);
            add_param_grad(originalSplatId, PARAM_ROTATION_X, dGaussian.rotation.y);
            add_param_grad(originalSplatId, PARAM_ROTATION_Y, dGaussian.rotation.z);
            add_param_grad(originalSplatId, PARAM_ROTATION_Z, dGaussian.rotation.w);
            add_param_grad(originalSplatId, PARAM_RAW_OPACITY, dRawOpacity);
            accum = prevAccum;
            trans = prevTrans;
        }
        GroupMemoryBarrierWithGroupSync();
        batchBase = batchStart;
    }
}
