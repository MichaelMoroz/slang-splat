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
uniform StructuredBuffer<uint> g_CountParamsBuffer;
uniform RWTensor<ProjectionState, 1> g_ProjectionState;
uniform RWTensor<Gaussian3D, 1> g_RasterState;
uniform RWTensor<uint, 1> g_TileRanges;
uniform RWTensor<uint, 1> g_BackwardBatchEnd;
uniform RWTensor<uint, 1> g_TotalBuffer;
uniform RWTensor<uint, 1> g_TotalMaxBuffer;
uniform RWStructuredBuffer<uint> g_TotalCounterBuffer;
uniform Tensor<uint, 1> g_OverflowA;
uniform Tensor<uint, 1> g_OverflowB;
uniform RWTensor<float4, 2> g_Output;
uniform Tensor<float4, 2> g_OutputGrad;
uniform RWTensor<float, 2> g_OutputDepth;
uniform Tensor<float, 2> g_OutputDepthGrad;
uniform RWTensor<float4, 2> g_ForwardState;
uniform RWTensor<float4, 2> g_ForwardDepthState;
uniform RWTensor<float3, 2> g_SplatDensities;
uniform RWTensor<uint, 2> g_ForwardEnd;
uniform RWTensor<float, 1> g_ParamGrads;
uniform AtomicTensor<float, 1> g_RasterGradBuffer;
uniform Tensor<float, 1> g_RasterGradBufferRead;
uniform Camera g_Camera;
uniform uint g_SplatCount;
uniform uint g_SortedEntryCount;
uniform uint g_UseCountParams;
uniform uint g_ScanlineCapacity;
uniform uint g_TileEntryCapacity;
uniform uint2 g_TileGrid;
uniform float3 g_Background;
uniform float g_RadiusScale;
uniform float g_DitherStrength;
uniform float g_MaxAnisotropy;
uniform float g_AlphaCutoff;
uniform float g_TransmittanceThreshold;
uniform uint g_RenderSeed;
uniform uint g_DebugMode;
uniform uint g_DebugMaxSplatSteps;
uniform float2 g_DebugDepthMeanRange;
uniform float2 g_DebugDepthStdRange;
uniform float2 g_DebugDensityRange;
uniform uint g_ComputeSplatDensities;
uniform RWTensor<uint, 1> g_ScanlineOverflow;
uniform RWTensor<uint, 1> g_TileOverflow;
uniform uint g_TotalCapacity;

void append_visible_splat(uint distanceKey, uint splatId) {
    uint writeIndex = 0u;
    InterlockedAdd(g_TotalCounterBuffer[0], 1u, writeIndex);
    if (writeIndex >= g_TotalCapacity) return;
    g_DistanceKeys.store(int(writeIndex), distanceKey);
    g_DistanceSortOrder.store(int(writeIndex), splatId);
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

float raster_eval_outline_alpha(Gaussian3D gaussian, ProjectionState projection, float2 samplePoint) {
    float2 delta = samplePoint - projection.centerRadiusDepth.xy;
    float2 grad = float2(
        projection.conicOpacity.x * delta.x + projection.conicOpacity.y * delta.y,
        projection.conicOpacity.y * delta.x + projection.conicOpacity.z * delta.y);
    float quad = dot(delta, grad);
    float gradLen = max(length(2.0 * grad), SMALL_VALUE);
    float fullWidth = max(DEBUG_ELLIPSE_THICKNESS_PX, 2.0 * MIN_DEBUG_THICKNESS_PX);
    float halfWidth = 0.5 * fullWidth;
    float edgeDistance = abs((quad - 1.0) / gradLen);
    float coverage = saturate(1.0 - edgeDistance / halfWidth);
    coverage = coverage * coverage * (3.0 - 2.0 * coverage);
    float alpha = gaussian.colorAlpha.w * coverage;
    return alpha >= g_AlphaCutoff ? alpha : 0.0;
}

uint count_scanlines(Ellipse ellipse) {
    bool transpose;
    float2 localCenter, localExtent;
    float3 localConic;
    // Normalize the ellipse into the major-axis scanline frame before counting strips.
    ellipse.scanline_frame(transpose, localCenter, localExtent, localConic);
    uint2 localGrid = transpose ? g_TileGrid.yx : g_TileGrid;
    int minStrip = clamp(int(floor((localCenter.y - localExtent.y - TINY_VALUE) / TILE_SIZE)), 0, int(localGrid.y) - 1);
    int maxStrip = clamp(int(floor((localCenter.y + localExtent.y + TINY_VALUE) / TILE_SIZE)), 0, int(localGrid.y) - 1);
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
    int maxSpan = clamp(int(floor((localCenter.x + localExtent.x + TINY_VALUE) / TILE_SIZE)), 0, int(localGrid.x) - 1);
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
    spanTileMax = clamp(int(floor((spanMax + localCenter.x + TINY_VALUE) / TILE_SIZE)), minSpan, maxSpan);
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

// Pass 1: project splats in original order and append only visible splat ids plus distance keys.
[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csProjectVisibleSplats(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;
    int[1] at = { int(splatId) };
    ProjectionState projection;
    projection.centerRadiusDepth = 0.0;
    projection.conicOpacity = 0.0;
    Gaussian3D rasterGaussian;
    rasterGaussian.position = 0.0;
    rasterGaussian.logScale = 0.0;
    rasterGaussian.rotation = 0.0;
    rasterGaussian.colorAlpha = 0.0;

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
        if (hasFit) hasFit = compute_gaussian_outline_ellipse(gaussian, g_Camera, cameraPosition, g_RadiusScale, supportSigmaRadius, g_MaxAnisotropy, ellipse);
        if (hasFit) {
            // Inflate the fitted ellipse before scan conversion so tile coverage stays conservative.
            ellipse.radiusPx = max(ellipse.radiusPx * ELLIPSE_SAFETY_SCALE + ELLIPSE_RADIUS_PAD_PX, MIN_SCREEN_RADIUS_PX);
            ellipse.bboxExtent = float2(ellipse.radiusPx, ellipse.radiusPx);
            hasFit = ellipse.prepare_conic();
        }
        if (hasFit) {
            projection.centerRadiusDepth = float4(ellipse.centerPx, ellipse.radiusPx, cameraDistance);
            projection.conicOpacity = float4(ellipse.conic, 1.0);
            rasterGaussian = Gaussian3D.camera_gaussian(gaussian, g_Camera, g_RadiusScale, GAUSSIAN_SUPPORT_SIGMA_RADIUS, g_MaxAnisotropy);
            rasterGaussian.colorAlpha.xyz = gaussian.colorAlpha.xyz;
            rasterGaussian.colorAlpha.w = fusedOpacity;
            append_visible_splat(float_to_ordered_uint(cameraDistance), splatId);
        }
    }

    g_ProjectionState.store(at, projection);
    g_RasterState.store(at, rasterGaussian);
}

// Pass 2: count scanlines for distance-sorted visible splats only.
[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csCountVisibleScanlines(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint visibleSplatId = dispatchThreadID.x;
    if (visibleSplatId >= (g_UseCountParams != 0u ? g_CountParamsBuffer[3] : g_SortedEntryCount)) return;
    uint splatId = g_SplatOrder[int(visibleSplatId)];
    Ellipse ellipse;
    uint scanlineCount = load_projected_ellipse(splatId, ellipse) ? count_scanlines(ellipse) : 0u;
    g_ScanlineCounts.store(int(visibleSplatId), scanlineCount);
}

// Pass 3a: expand each projected ellipse into per-strip scanline work items.
[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csEmitScanlines(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint visibleSplatId = dispatchThreadID.x;
    if (visibleSplatId >= (g_UseCountParams != 0u ? g_CountParamsBuffer[3] : g_SortedEntryCount)) return;
    uint splatId = g_SplatOrder[int(visibleSplatId)];
    Ellipse ellipse;
    if (!load_projected_ellipse(splatId, ellipse)) return;
    uint scanlineCount = g_ScanlineCounts[int(visibleSplatId)];
    if (scanlineCount == 0u) return;
    bool transpose;
    float2 localCenter, localExtent;
    float3 localConic;
    ellipse.scanline_frame(transpose, localCenter, localExtent, localConic);
    uint2 localGrid = transpose ? g_TileGrid.yx : g_TileGrid;
    int minStrip = clamp(int(floor((localCenter.y - localExtent.y - TINY_VALUE) / TILE_SIZE)), 0, int(localGrid.y) - 1);
    uint writeIndex = g_ScanlineOffsets[int(visibleSplatId)];
    for (uint localStrip = 0u; localStrip < scanlineCount; localStrip++) {
        uint dst = writeIndex + localStrip;
        if (dst >= g_ScanlineCapacity) {
            g_ScanlineOverflow.store(0, 1u);
            break;
        }
        // Each work item stores the splat plus one strip index in the chosen scan frame.
        int[1] wi = { int(dst) };
        ScanlineEntry scanline;
        scanline.splatId = splatId;
        scanline.stripIndex = uint(minStrip) + localStrip;
        scanline.transpose = transpose ? 1u : 0u;
        g_ScanlineEntries.store(wi, scanline);
    }
}

// Pass 3b: count the number of tiles covered by each emitted scanline.
[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csEmitScanlineTileCounts(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint scanlineId = dispatchThreadID.x;
    if (scanlineId >= (g_UseCountParams != 0u ? g_CountParamsBuffer[3] : g_SortedEntryCount)) return;
    ScanlineEntry scanline = g_ScanlineEntries[int(scanlineId)];
    int spanTileMin, spanTileMax;
    uint tileCount = scanline_tile_span(scanline, spanTileMin, spanTileMax) ? uint(spanTileMax - spanTileMin + 1) : 0u;
    g_ScanlineTileCounts.store(int[1](int(scanlineId)), tileCount);
}

// Pass 4: emit final (tileId, splatId) entries for sorting.
[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csEmitTileEntries(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint scanlineId = dispatchThreadID.x;
    if (scanlineId >= (g_UseCountParams != 0u ? g_CountParamsBuffer[3] : g_SortedEntryCount)) return;
    ScanlineEntry scanline = g_ScanlineEntries[int(scanlineId)];
    int spanTileMin, spanTileMax;
    if (!scanline_tile_span(scanline, spanTileMin, spanTileMax)) return;
    uint writeIndex = g_ScanlineTileOffsets[int(scanlineId)];
    for (int span = spanTileMin; span <= spanTileMax; span++) {
        if (writeIndex >= g_TileEntryCapacity) {
            g_TileOverflow.store(0, 1u);
            break;
        }
        uint2 tileXY = scanline.transpose != 0u ? uint2(scanline.stripIndex, uint(span)) : uint2(uint(span), scanline.stripIndex);
        g_TileEntries.store(int(writeIndex * 2u), tileXY.y * g_TileGrid.x + tileXY.x);
        g_TileEntries.store(int(writeIndex * 2u + 1u), scanline.splatId);
        writeIndex++;
    }
}

// Pass 5: rebuild per-tile sorted-entry ranges after the host-side key sort.
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
    uint sortedEntryCount = g_UseCountParams != 0u ? g_CountParamsBuffer[3] : g_SortedEntryCount;
    uint idx = dispatchThreadID.x;
    if (idx >= sortedEntryCount) return;
    uint tileCount = g_TileGrid.x * g_TileGrid.y;
    uint tile = g_SortedEntries[int(idx * 2u)];
    if (tile >= tileCount) return;
    uint prevTile = idx > 0u ? g_SortedEntries[int((idx - 1u) * 2u)] : 0u;
    uint nextTile = idx + 1u < sortedEntryCount ? g_SortedEntries[int((idx + 1u) * 2u)] : 0u;
    // g_TileRanges is flattened as [start0, end0, start1, end1, ...].
    if (idx == 0u || tile != prevTile) g_TileRanges.store(int(tile * 2u), idx);
    if (idx + 1u == sortedEntryCount || tile != nextTile) g_TileRanges.store(int(tile * 2u + 1u), idx + 1u);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void csFinalizeTotalWithOverflow(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint total = g_TotalBuffer[0];
    bool overflow = total > g_TotalCapacity || g_OverflowA[0] != 0u || g_OverflowB[0] != 0u;
    if (overflow) total = max(total, g_TotalCapacity + 1u);
    g_TotalMaxBuffer.store(0, max(g_TotalMaxBuffer[0], total));
}

// Raster batching helpers shared by forward and backward passes.
Gaussian3D make_empty_gaussian() {
    Gaussian3D empty;
    empty.position = 0.0;
    empty.logScale = 0.0;
    empty.rotation = 0.0;
    empty.colorAlpha = 0.0;
    return empty;
}

ProjectionState make_empty_projection() {
    ProjectionState empty;
    empty.centerRadiusDepth = 0.0;
    empty.conicOpacity = 0.0;
    return empty;
}

groupshared Gaussian3D gsGaussian[RASTER_BATCH];
groupshared ProjectionState gsProjection[RASTER_BATCH];
groupshared uint gsBatchSplatId[RASTER_BATCH];
groupshared uint gsDoneCount;

void prefetch_batch(uint batchStart, uint batchEnd, uint lane) {
    uint batchCount = batchEnd - batchStart;
    for (uint localIdx = lane; localIdx < batchCount; localIdx += RASTER_GROUP_THREADS) {
        uint splatId = g_SortedEntries[int((batchStart + localIdx) * 2u + 1u)];
        gsBatchSplatId[localIdx] = splatId;
        if (splatId >= g_SplatCount) {
            gsGaussian[localIdx] = make_empty_gaussian();
            continue;
        }
        gsGaussian[localIdx] = g_RasterState[int(splatId)];
    }
}

void prefetch_batch_debug(uint batchStart, uint batchEnd, uint lane) {
    uint batchCount = batchEnd - batchStart;
    for (uint localIdx = lane; localIdx < batchCount; localIdx += RASTER_GROUP_THREADS) {
        uint splatId = g_SortedEntries[int((batchStart + localIdx) * 2u + 1u)];
        gsBatchSplatId[localIdx] = splatId;
        if (splatId >= g_SplatCount) {
            gsGaussian[localIdx] = make_empty_gaussian();
            gsProjection[localIdx] = make_empty_projection();
            continue;
        }
        gsGaussian[localIdx] = g_RasterState[int(splatId)];
        gsProjection[localIdx] = g_ProjectionState[int(splatId)];
    }
}

int2 raster_thread_to_pixel(uint tileId, uint lane) {
    uint tileX = tileId % g_TileGrid.x;
    uint tileY = tileId / g_TileGrid.x;
    uint localX = lane % RASTER_THREAD_TILE_DIM;
    uint localY = lane / RASTER_THREAD_TILE_DIM;
    return int2(
        int(tileX * TILE_SIZE + localX),
        int(tileY * TILE_SIZE + localY));
}

[Differentiable]
float4 shade_step(float3 accum, float trans, Gaussian3D gaussian, float alpha) {
    return float4(accum + trans * alpha * gaussian.colorAlpha.xyz, trans * (1.0 - alpha));
}

// Forward rasterization over sorted tile-local splat lists.
[shader("compute")]
[numthreads(RASTER_GROUP_THREADS, 1, 1)]
void csRasterForward(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex) {
    uint tileId = groupID.x;
    int2 p = raster_thread_to_pixel(tileId, lane);
    bool active = p.x < int(g_Camera.viewport.x) && p.y < int(g_Camera.viewport.y);
    uint2 range = uint2(g_TileRanges[int(tileId * 2u)], g_TileRanges[int(tileId * 2u + 1u)]);
    float3 accum = 0.0;
    float trans = 1.0;
    uint processedEnd = range.x;
    uint pixelId = uint(p.y) * uint(g_Camera.viewport.x) + uint(p.x);
    float2 raySample = float2(p) + stochastic_subpixel_offset(pixelId, g_RenderSeed, g_DitherStrength);
    float3 rayDir = active ? g_Camera.screen_to_world_ray(raySample) : 0.0;
    float depthWeightSum = 0.0;
    float depthMean = 0.0;
    float depthM2 = 0.0;
    uint replayBatchEnd = range.x;
    if (range.x != INVALID_START && range.y > range.x) {
        bool done = !active;
        if (lane == 0u) gsDoneCount = 0u;
        GroupMemoryBarrierWithGroupSync();
        if (!active) InterlockedAdd(gsDoneCount, 1u);
        GroupMemoryBarrierWithGroupSync();
        for (uint batchStart = range.x; batchStart < range.y; batchStart += RASTER_BATCH) {
            if (gsDoneCount == RASTER_GROUP_THREADS) break;
            uint batchEnd = min(batchStart + RASTER_BATCH, range.y);
            uint batchCount = batchEnd - batchStart;
            replayBatchEnd = batchEnd;
            prefetch_batch(batchStart, batchEnd, lane);
            GroupMemoryBarrierWithGroupSync();
            if (!done) {
                for (uint localIdx = 0u; localIdx < batchCount; localIdx++) {
                    processedEnd = batchStart + localIdx + 1u;
                    Gaussian3D gaussian = gsGaussian[localIdx];
                    float2 eval = raster_eval_alpha(gaussian, rayDir);
                    float alpha = eval.x;
                    float rayDepth = eval.y;
                    if (alpha < g_AlphaCutoff) continue;
                    float contributionWeight = trans * alpha;
                    if (contributionWeight > SMALL_VALUE && rayDepth > 0.0) {
                        float3 depthState = depth_state_step(depthWeightSum, depthMean, depthM2, contributionWeight, rayDepth);
                        depthWeightSum = depthState.x;
                        depthMean = depthState.y;
                        depthM2 = depthState.z;
                    }
                    float4 blended = shade_step(accum, trans, gaussian, alpha);
                    accum = blended.xyz;
                    trans = blended.w;
                    if (trans < g_TransmittanceThreshold) {
                        done = true;
                        InterlockedAdd(gsDoneCount, 1u);
                        break;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    if (lane == 0u) g_BackwardBatchEnd.store(int(tileId), replayBatchEnd);
    if (!active) return;
    float4 output = raster_output(accum, trans, g_Background);
    float depthOutput = raster_depth_output(depthWeightSum, depthMean, depthM2);
    g_Output.store(p, output);
    g_ForwardState.store(p, float4(accum, trans));
    g_OutputDepth.store(p, depthOutput);
    g_ForwardDepthState.store(p, float4(depthWeightSum, depthMean, depthM2, 0.0));
    g_ForwardEnd.store(p, processedEnd);
}

[shader("compute")]
[numthreads(RASTER_GROUP_THREADS, 1, 1)]
void csRasterForwardDebug(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex) {
    uint tileId = groupID.x;
    int2 p = raster_thread_to_pixel(tileId, lane);
    bool active = p.x < int(g_Camera.viewport.x) && p.y < int(g_Camera.viewport.y);
    uint debugMode = g_DebugMode;
    uint2 range = uint2(g_TileRanges[int(tileId * 2u)], g_TileRanges[int(tileId * 2u + 1u)]);
    float3 accum = 0.0;
    float trans = 1.0;
    uint processedEnd = range.x;
    uint pixelId = uint(p.y) * uint(g_Camera.viewport.x) + uint(p.x);
    float2 raySample = float2(p) + stochastic_subpixel_offset(pixelId, g_RenderSeed, g_DitherStrength);
    float3 rayDir = active ? g_Camera.screen_to_world_ray(raySample) : 0.0;
    bool computeDensityOutput = g_ComputeSplatDensities != 0u;
    bool needsOutlineDebug = debugMode == DEBUG_MODE_ELLIPSE_OUTLINES;
    bool needsSpatialDensity = computeDensityOutput || debugMode == DEBUG_MODE_SPLAT_SPATIAL_DENSITY;
    bool needsScreenDensity = computeDensityOutput || debugMode == DEBUG_MODE_SPLAT_SCREEN_DENSITY;
    float depthWeightSum = 0.0;
    float depthMean = 0.0;
    float depthM2 = 0.0;
    float spatialDensitySum = 0.0;
    float screenDensitySum = 0.0;
    uint replayBatchEnd = range.x;
    if (range.x != INVALID_START && range.y > range.x) {
        bool done = !active;
        if (lane == 0u) gsDoneCount = 0u;
        GroupMemoryBarrierWithGroupSync();
        if (!active) InterlockedAdd(gsDoneCount, 1u);
        GroupMemoryBarrierWithGroupSync();
        for (uint batchStart = range.x; batchStart < range.y; batchStart += RASTER_BATCH) {
            if (gsDoneCount == RASTER_GROUP_THREADS) break;
            uint batchEnd = min(batchStart + RASTER_BATCH, range.y);
            uint batchCount = batchEnd - batchStart;
            replayBatchEnd = batchEnd;
            prefetch_batch_debug(batchStart, batchEnd, lane);
            GroupMemoryBarrierWithGroupSync();
            if (!done) {
                for (uint localIdx = 0u; localIdx < batchCount; localIdx++) {
                    processedEnd = batchStart + localIdx + 1u;
                    Gaussian3D gaussian = gsGaussian[localIdx];
                    ProjectionState projection = gsProjection[localIdx];
                    uint splatId = gsBatchSplatId[localIdx];
                    float2 eval = needsOutlineDebug
                        ? float2(raster_eval_outline_alpha(gaussian, projection, raySample), 0.0)
                        : raster_eval_alpha(gaussian, rayDir);
                    float alpha = eval.x;
                    float rayDepth = eval.y;
                    if (alpha < g_AlphaCutoff) continue;
                    float contributionWeight = trans * alpha;
                    if (contributionWeight > SMALL_VALUE && rayDepth > 0.0) {
                        float3 depthState = depth_state_step(depthWeightSum, depthMean, depthM2, contributionWeight, rayDepth);
                        depthWeightSum = depthState.x;
                        depthMean = depthState.y;
                        depthM2 = depthState.z;
                    }
                    if (contributionWeight > SMALL_VALUE) {
                        if (needsSpatialDensity) {
                            Gaussian3D originalGaussian = Gaussian3D.load(g_Splats, g_SplatCount, splatId);
                            spatialDensitySum += contributionWeight / gaussian_scale_volume(originalGaussian.logScale);
                        }
                        if (needsScreenDensity) {
                            screenDensitySum += contributionWeight / projected_ellipse_area(projection.conicOpacity.xyz);
                        }
                    }
                    float4 blended = shade_step(accum, trans, gaussian, alpha);
                    accum = blended.xyz;
                    trans = blended.w;
                    if (trans < g_TransmittanceThreshold) {
                        done = true;
                        InterlockedAdd(gsDoneCount, 1u);
                        break;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    if (lane == 0u) g_BackwardBatchEnd.store(int(tileId), replayBatchEnd);
    if (!active) return;
    uint processedCount = range.x != INVALID_START && processedEnd >= range.x ? (processedEnd - range.x) : 0u;
    float4 output = raster_output(accum, trans, g_Background);
    float depthOutput = raster_depth_output(depthWeightSum, depthMean, depthM2);

    if (debugMode == DEBUG_MODE_PROCESSED_COUNT) {
        output = float4(raster_processed_count_color(processedCount, g_DebugMaxSplatSteps), 1.0);
    } else if (debugMode == DEBUG_MODE_SPLAT_SPATIAL_DENSITY) {
        output = float4(jet_colormap(debug_normalize_range(spatialDensitySum, g_DebugDensityRange)), 1.0);
    } else if (debugMode == DEBUG_MODE_SPLAT_SCREEN_DENSITY) {
        output = float4(jet_colormap(debug_normalize_range(screenDensitySum, g_DebugDensityRange)), 1.0);
    } else if (debugMode == DEBUG_MODE_DEPTH_MEAN) {
        output = float4(jet_colormap(debug_normalize_range(depthMean, g_DebugDepthMeanRange)), 1.0);
    } else if (debugMode == DEBUG_MODE_DEPTH_STD) {
        output = float4(jet_colormap(debug_normalize_range(depthOutput, g_DebugDepthStdRange)), 1.0);
    }

    g_Output.store(p, output);
    g_ForwardState.store(p, float4(accum, trans));
    g_OutputDepth.store(p, depthOutput);
    g_ForwardDepthState.store(p, float4(depthWeightSum, depthMean, depthM2, 0.0));
    if (computeDensityOutput) g_SplatDensities.store(p, float3(spatialDensitySum, screenDensitySum, depthMean));
    g_ForwardEnd.store(p, processedEnd);
}

struct PixelBackwardState {
    bool valid;
    float3 accum;
    float trans;
    float depthWeightSum;
    float depthMean;
    float depthM2;
    float3 dAccum;
    float dTrans;
    float dDepthWeightSum;
    float dDepthMean;
    float dDepthM2;
    float3 rayDir;
    uint processedEnd;
};

struct RasterBackwardGrads {
    float3 position;
    float3 logScale;
    float4 rotation;
    float3 color;
    float fusedOpacity;
};

PixelBackwardState make_invalid_pixel_backward_state() {
    PixelBackwardState state;
    state.valid = false;
    state.accum = 0.0;
    state.trans = 0.0;
    state.depthWeightSum = 0.0;
    state.depthMean = 0.0;
    state.depthM2 = 0.0;
    state.dAccum = 0.0;
    state.dTrans = 0.0;
    state.dDepthWeightSum = 0.0;
    state.dDepthMean = 0.0;
    state.dDepthM2 = 0.0;
    state.rayDir = 0.0;
    state.processedEnd = 0u;
    return state;
}

void add_raster_grad(uint splatId, uint paramId, float value) {
    if (value == 0.0) return;
    g_RasterGradBuffer.add(int(splatId * PARAM_COUNT + paramId), value);
}

RasterBackwardGrads make_zero_raster_backward_grads() {
    RasterBackwardGrads grads;
    grads.position = 0.0;
    grads.logScale = 0.0;
    grads.rotation = 0.0;
    grads.color = 0.0;
    grads.fusedOpacity = 0.0;
    return grads;
}

void rewind_depth_backward(inout PixelBackwardState state, float contributionWeight, float rayDepth, float alpha, float prevTrans, inout float dAlpha, out float dRayDepth) {
    dRayDepth = 0.0;
    if (contributionWeight <= SMALL_VALUE || rayDepth <= 0.0) return;
    float3 prevDepthState = depth_state_rewind(state.depthWeightSum, state.depthMean, state.depthM2, contributionWeight, rayDepth);
    var dpPrevDepthWeightSum = diffPair(prevDepthState.x);
    var dpPrevDepthMean = diffPair(prevDepthState.y);
    var dpPrevDepthM2 = diffPair(prevDepthState.z);
    var dpContributionWeight = diffPair(contributionWeight);
    var dpRayDepth = diffPair(rayDepth);
    bwd_diff(depth_state_step)(
        dpPrevDepthWeightSum,
        dpPrevDepthMean,
        dpPrevDepthM2,
        dpContributionWeight,
        dpRayDepth,
        float3(state.dDepthWeightSum, state.dDepthMean, state.dDepthM2)
    );
    state.depthWeightSum = prevDepthState.x;
    state.depthMean = prevDepthState.y;
    state.depthM2 = prevDepthState.z;
    state.dDepthWeightSum = dpPrevDepthWeightSum.getDifferential();
    state.dDepthMean = dpPrevDepthMean.getDifferential();
    state.dDepthM2 = dpPrevDepthM2.getDifferential();
    float dContributionWeight = dpContributionWeight.getDifferential();
    state.dTrans += dContributionWeight * alpha;
    dAlpha += dContributionWeight * prevTrans;
    dRayDepth = dpRayDepth.getDifferential();
}

RasterBackwardGrads raster_gaussian_backward_grads(
    Gaussian3D gaussian,
    float3 rayDir,
    Gaussian3D.Differential dGaussian,
    float dAlpha,
    float dRayDepth)
{
    var dpCachedGaussianAlpha = diffPair(gaussian);
    bwd_diff(raster_eval_alpha_only)(dpCachedGaussianAlpha, rayDir, dAlpha);
    Gaussian3D.Differential dCachedGaussian = dpCachedGaussianAlpha.getDifferential();
    if (dRayDepth != 0.0) {
        var dpCachedGaussianDepth = diffPair(gaussian);
        bwd_diff(raster_eval_depth_only)(dpCachedGaussianDepth, rayDir, dRayDepth);
        Gaussian3D.Differential dCachedGaussianDepth = dpCachedGaussianDepth.getDifferential();
        dCachedGaussian.position += dCachedGaussianDepth.position;
        dCachedGaussian.logScale += dCachedGaussianDepth.logScale;
        dCachedGaussian.rotation += dCachedGaussianDepth.rotation;
        dCachedGaussian.colorAlpha += dCachedGaussianDepth.colorAlpha;
    }

    RasterBackwardGrads grads;
    grads.position = dGaussian.position + dCachedGaussian.position;
    grads.logScale = dGaussian.logScale + dCachedGaussian.logScale;
    grads.rotation = dGaussian.rotation + dCachedGaussian.rotation;
    grads.color = dGaussian.colorAlpha.xyz + dCachedGaussian.colorAlpha.xyz;
    grads.fusedOpacity = dGaussian.colorAlpha.w + dCachedGaussian.colorAlpha.w;
    return grads;
}

PixelBackwardState load_pixel_backward_state(bool active, int2 p, float2 raySample) {
    PixelBackwardState state = make_invalid_pixel_backward_state();
    if (!active) return state;
    float4 state4 = g_ForwardState[p];
    float4 depthState4 = g_ForwardDepthState[p];
    var dpAccum = diffPair(state4.xyz);
    var dpTrans = diffPair(state4.w);
    bwd_diff(raster_output)(dpAccum, dpTrans, g_Background, g_OutputGrad[p]);
    var dpDepthWeightSum = diffPair(depthState4.x);
    var dpDepthMean = diffPair(depthState4.y);
    var dpDepthM2 = diffPair(depthState4.z);
    bwd_diff(raster_depth_output)(dpDepthWeightSum, dpDepthMean, dpDepthM2, g_OutputDepthGrad[p]);
    state.valid = true;
    state.accum = state4.xyz;
    state.trans = state4.w;
    state.depthWeightSum = depthState4.x;
    state.depthMean = depthState4.y;
    state.depthM2 = depthState4.z;
    state.dAccum = dpAccum.getDifferential();
    state.dTrans = dpTrans.getDifferential();
    state.dDepthWeightSum = dpDepthWeightSum.getDifferential();
    state.dDepthMean = dpDepthMean.getDifferential();
    state.dDepthM2 = dpDepthM2.getDifferential();
    state.rayDir = g_Camera.screen_to_world_ray(raySample);
    state.processedEnd = g_ForwardEnd[p];
    return state;
}

void flush_raster_backward_grads(uint splatId, RasterBackwardGrads grads) {
    float4 colorAlpha = WaveActiveSum(float4(grads.color, grads.fusedOpacity));
    float3 position = WaveActiveSum(grads.position);
    float3 logScale = WaveActiveSum(grads.logScale);
    float4 rotation = WaveActiveSum(grads.rotation);
    if (WaveIsFirstLane()) {
        add_raster_grad(splatId, PARAM_COLOR_R, colorAlpha.x);
        add_raster_grad(splatId, PARAM_COLOR_G, colorAlpha.y);
        add_raster_grad(splatId, PARAM_COLOR_B, colorAlpha.z);
        add_raster_grad(splatId, PARAM_POSITION_X, position.x);
        add_raster_grad(splatId, PARAM_POSITION_Y, position.y);
        add_raster_grad(splatId, PARAM_POSITION_Z, position.z);
        add_raster_grad(splatId, PARAM_SCALE_X, logScale.x);
        add_raster_grad(splatId, PARAM_SCALE_Y, logScale.y);
        add_raster_grad(splatId, PARAM_SCALE_Z, logScale.z);
        add_raster_grad(splatId, PARAM_ROTATION_W, rotation.x);
        add_raster_grad(splatId, PARAM_ROTATION_X, rotation.y);
        add_raster_grad(splatId, PARAM_ROTATION_Y, rotation.z);
        add_raster_grad(splatId, PARAM_ROTATION_Z, rotation.w);
        add_raster_grad(splatId, PARAM_RAW_OPACITY, colorAlpha.w);
    }
}

[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csBackprojectRasterGrads(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;

    Gaussian3D originalGaussian = Gaussian3D.load(g_Splats, g_SplatCount, splatId);
    float fusedOpacity = originalGaussian.build_raster_fused_opacity();

    Gaussian3D.Differential dCachedGaussian;
    dCachedGaussian.position = float3(
        g_RasterGradBufferRead[int(splatId * PARAM_COUNT + PARAM_POSITION_X)],
        g_RasterGradBufferRead[int(splatId * PARAM_COUNT + PARAM_POSITION_Y)],
        g_RasterGradBufferRead[int(splatId * PARAM_COUNT + PARAM_POSITION_Z)]);
    dCachedGaussian.logScale = float3(
        g_RasterGradBufferRead[int(splatId * PARAM_COUNT + PARAM_SCALE_X)],
        g_RasterGradBufferRead[int(splatId * PARAM_COUNT + PARAM_SCALE_Y)],
        g_RasterGradBufferRead[int(splatId * PARAM_COUNT + PARAM_SCALE_Z)]);
    dCachedGaussian.rotation = float4(
        g_RasterGradBufferRead[int(splatId * PARAM_COUNT + PARAM_ROTATION_W)],
        g_RasterGradBufferRead[int(splatId * PARAM_COUNT + PARAM_ROTATION_X)],
        g_RasterGradBufferRead[int(splatId * PARAM_COUNT + PARAM_ROTATION_Y)],
        g_RasterGradBufferRead[int(splatId * PARAM_COUNT + PARAM_ROTATION_Z)]);
    float3 colorGrad = float3(
        g_RasterGradBufferRead[int(splatId * PARAM_COUNT + PARAM_COLOR_R)],
        g_RasterGradBufferRead[int(splatId * PARAM_COUNT + PARAM_COLOR_G)],
        g_RasterGradBufferRead[int(splatId * PARAM_COUNT + PARAM_COLOR_B)]);
    float fusedOpacityGrad = g_RasterGradBufferRead[int(splatId * PARAM_COUNT + PARAM_RAW_OPACITY)];
    dCachedGaussian.colorAlpha = 0.0;

    var dpOriginalGaussian = diffPair(originalGaussian);
    bwd_diff(Gaussian3D.camera_gaussian)(dpOriginalGaussian, g_Camera, g_RadiusScale, GAUSSIAN_SUPPORT_SIGMA_RADIUS, g_MaxAnisotropy, dCachedGaussian);
    Gaussian3D.Differential dOriginalGaussian = dpOriginalGaussian.getDifferential();

    g_ParamGrads.store(int(splatId * PARAM_COUNT + PARAM_POSITION_X), dOriginalGaussian.position.x);
    g_ParamGrads.store(int(splatId * PARAM_COUNT + PARAM_POSITION_Y), dOriginalGaussian.position.y);
    g_ParamGrads.store(int(splatId * PARAM_COUNT + PARAM_POSITION_Z), dOriginalGaussian.position.z);
    g_ParamGrads.store(int(splatId * PARAM_COUNT + PARAM_SCALE_X), dOriginalGaussian.logScale.x);
    g_ParamGrads.store(int(splatId * PARAM_COUNT + PARAM_SCALE_Y), dOriginalGaussian.logScale.y);
    g_ParamGrads.store(int(splatId * PARAM_COUNT + PARAM_SCALE_Z), dOriginalGaussian.logScale.z);
    g_ParamGrads.store(int(splatId * PARAM_COUNT + PARAM_ROTATION_W), dOriginalGaussian.rotation.x);
    g_ParamGrads.store(int(splatId * PARAM_COUNT + PARAM_ROTATION_X), dOriginalGaussian.rotation.y);
    g_ParamGrads.store(int(splatId * PARAM_COUNT + PARAM_ROTATION_Y), dOriginalGaussian.rotation.z);
    g_ParamGrads.store(int(splatId * PARAM_COUNT + PARAM_ROTATION_Z), dOriginalGaussian.rotation.w);
    g_ParamGrads.store(int(splatId * PARAM_COUNT + PARAM_COLOR_R), colorGrad.x);
    g_ParamGrads.store(int(splatId * PARAM_COUNT + PARAM_COLOR_G), colorGrad.y);
    g_ParamGrads.store(int(splatId * PARAM_COUNT + PARAM_COLOR_B), colorGrad.z);
    g_ParamGrads.store(int(splatId * PARAM_COUNT + PARAM_RAW_OPACITY), fusedOpacityGrad * fusedOpacity * (1.0 - fusedOpacity));
}

// Reverse raster replay using the cached forward state and sorted tile ranges.
[shader("compute")]
[numthreads(RASTER_GROUP_THREADS, 1, 1)]
void csRasterBackward(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex) {
    uint tileId = groupID.x;
    int2 p = raster_thread_to_pixel(tileId, lane);
    bool active = p.x < int(g_Camera.viewport.x) && p.y < int(g_Camera.viewport.y);
    uint2 range = uint2(g_TileRanges[int(tileId * 2u)], g_TileRanges[int(tileId * 2u + 1u)]);
    uint pixelId = uint(p.y) * uint(g_Camera.viewport.x) + uint(p.x);
    float2 raySample = float2(p) + stochastic_subpixel_offset(pixelId, g_RenderSeed, g_DitherStrength);
    PixelBackwardState state = load_pixel_backward_state(active, p, raySample);
    uint batchBase = g_BackwardBatchEnd[int(tileId)];
    for (; batchBase > range.x;) {
        uint batchStart = batchBase > range.x + RASTER_BATCH ? batchBase - RASTER_BATCH : range.x;
        uint batchCount = batchBase - batchStart;
        if (range.x != INVALID_START && range.y > range.x) {
            prefetch_batch(batchStart, batchBase, lane);
            GroupMemoryBarrierWithGroupSync();
            if (active) {
                for (uint rev = 0u; rev < batchCount; rev++) {
                    uint localIdx = batchCount - 1u - rev;
                    uint globalIdx = batchStart + localIdx;
                    uint splatId = gsBatchSplatId[localIdx];
                    if (splatId >= g_SplatCount) continue;
                    RasterBackwardGrads grads = make_zero_raster_backward_grads();
                    bool replay = state.valid && globalIdx < state.processedEnd;
                    if (replay) {
                        Gaussian3D gaussian = g_RasterState[int(splatId)];

                        float2 eval = raster_eval_alpha(gaussian, state.rayDir);
                        float alpha = eval.x;
                        float rayDepth = eval.y;
                        if (alpha >= g_AlphaCutoff) {
                            float prevTrans = state.trans / max(1.0 - alpha, SMALL_VALUE);
                            float3 prevAccum = state.accum - prevTrans * alpha * gaussian.colorAlpha.xyz;
                            var dpPrevAccum = diffPair(prevAccum);
                            var dpPrevTrans = diffPair(prevTrans);
                            var dpGaussian = diffPair(gaussian);
                            var dpAlpha = diffPair(alpha);
                            bwd_diff(shade_step)(dpPrevAccum, dpPrevTrans, dpGaussian, dpAlpha, float4(state.dAccum, state.dTrans));
                            Gaussian3D.Differential dGaussian = dpGaussian.getDifferential();
                            float dAlpha = dpAlpha.getDifferential();
                            state.dAccum = dpPrevAccum.getDifferential();
                            state.dTrans = dpPrevTrans.getDifferential();

                            float contributionWeight = prevTrans * alpha;
                            float dRayDepth = 0.0;
                            rewind_depth_backward(state, contributionWeight, rayDepth, alpha, prevTrans, dAlpha, dRayDepth);
                            grads = raster_gaussian_backward_grads(gaussian, state.rayDir, dGaussian, dAlpha, dRayDepth);

                            state.accum = prevAccum;
                            state.trans = prevTrans;
                        }
                    }
                    flush_raster_backward_grads(splatId, grads);
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
        batchBase = batchStart;
    }
}
