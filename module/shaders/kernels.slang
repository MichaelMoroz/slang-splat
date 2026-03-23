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
uniform RWTensor<uint, 1> g_TotalBuffer;
uniform Tensor<uint, 1> g_OverflowA;
uniform Tensor<uint, 1> g_OverflowB;
uniform RWTensor<float4, 2> g_Output;
uniform Tensor<float4, 2> g_OutputGrad;
uniform RWTensor<float4, 2> g_ForwardState;
uniform RWTensor<uint, 2> g_ForwardEnd;
uniform AtomicTensor<float, 1> g_ParamGrads;
uniform Camera g_Camera;
uniform uint g_SplatCount;
uniform uint g_SortedEntryCount;
uniform uint g_UseCountParams;
uniform uint g_ScanlineCapacity;
uniform uint g_TileEntryCapacity;
uniform uint2 g_TileGrid;
uniform float3 g_Background;
uniform float g_RadiusScale;
uniform float g_MaxAnisotropy;
uniform float g_AlphaCutoff;
uniform float g_TransmittanceThreshold;
uniform uint g_RenderSeed;
uniform uint g_DebugMode;
uniform uint g_DebugMaxSplatSteps;
uniform float2 g_DebugDepthMeanRange;
uniform float2 g_DebugDepthStdRange;
uniform RWTensor<uint, 1> g_ScanlineOverflow;
uniform RWTensor<uint, 1> g_TileOverflow;
uniform uint g_TotalCapacity;

groupshared Gaussian3D gsGaussian[RASTER_BATCH];
groupshared uint gsBatchSplatId[RASTER_BATCH];
groupshared uint gsDoneCount;

uint load_sorted_entry_tile(uint idx) {
    return g_SortedEntries[int(idx * 2u)];
}

uint load_sorted_entry_splat(uint idx) {
    return g_SortedEntries[int(idx * 2u + 1u)];
}

uint load_sorted_entry_count() {
    return g_UseCountParams != 0u ? g_CountParamsBuffer[3] : g_SortedEntryCount;
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
            rasterGaussian = Gaussian3D.camera_gaussian(gaussian, g_Camera, g_RadiusScale, GAUSSIAN_SUPPORT_SIGMA_RADIUS, g_MaxAnisotropy);
            rasterGaussian.colorAlpha.xyz = gaussian.colorAlpha.xyz;
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

// Pass 2b: count the number of tiles covered by each emitted scanline.
[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csEmitScanlineTileCounts(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint scanlineId = dispatchThreadID.x;
    if (scanlineId >= load_sorted_entry_count()) return;
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
    if (scanlineId >= load_sorted_entry_count()) return;
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
    uint sortedEntryCount = load_sorted_entry_count();
    uint idx = dispatchThreadID.x;
    if (idx >= sortedEntryCount) return;
    uint tileCount = g_TileGrid.x * g_TileGrid.y;
    uint tile = load_sorted_entry_tile(idx);
    if (tile >= tileCount) return;
    uint prevTile = idx > 0u ? load_sorted_entry_tile(idx - 1u) : 0u;
    uint nextTile = idx + 1u < sortedEntryCount ? load_sorted_entry_tile(idx + 1u) : 0u;
    // g_TileRanges is flattened as [start0, end0, start1, end1, ...].
    if (idx == 0u || tile != prevTile) g_TileRanges.store(int(tile * 2u), idx);
    if (idx + 1u == sortedEntryCount || tile != nextTile) g_TileRanges.store(int(tile * 2u + 1u), idx + 1u);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void csFinalizeTotalWithOverflow(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint total = g_TotalBuffer[0];
    bool overflow = total > g_TotalCapacity || g_OverflowA[0] != 0u || g_OverflowB[0] != 0u;
    if (overflow) {
        g_TotalBuffer.store(0, g_TotalCapacity + 1u);
        g_TileOverflow.store(0, 1u);
    }
}

// Raster batching helpers shared by forward and backward passes.
void prefetch_batch(uint batchStart, uint batchEnd, uint lane) {
    uint batchCount = batchEnd - batchStart;
    for (uint localIdx = lane; localIdx < batchCount; localIdx += RASTER_GROUP_THREADS) {
        uint splatId = load_sorted_entry_splat(batchStart + localIdx);
        gsBatchSplatId[localIdx] = splatId;
        if (splatId >= g_SplatCount) {
            Gaussian3D empty;
            empty.position = 0.0;
            empty.logScale = 0.0;
            empty.rotation = 0.0;
            empty.colorAlpha = 0.0;
            gsGaussian[localIdx] = empty;
            continue;
        }
        gsGaussian[localIdx] = g_RasterState[int(splatId)];
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
    float2 raySample = float2(p) + stochastic_subpixel_offset(pixelId, g_RenderSeed);
    float3 rayDir = active ? g_Camera.screen_to_world_ray(raySample) : 0.0;
    bool depthDebug = g_DebugMode == DEBUG_MODE_DEPTH_MEAN || g_DebugMode == DEBUG_MODE_DEPTH_STD;
    float depthWeightSum = 0.0;
    float depthMean = 0.0;
    float depthM2 = 0.0;
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
            // Batch splats into groupshared storage so every pixel in the tile reuses the same loads.
            prefetch_batch(batchStart, batchEnd, lane);
            GroupMemoryBarrierWithGroupSync();
            if (!done) {
                for (uint localIdx = 0u; localIdx < batchCount; localIdx++) {
                    processedEnd = batchStart + localIdx + 1u;
                    Gaussian3D gaussian = gsGaussian[localIdx];
                    // The forward loop only needs alpha and the fused color/opacity carried by Gaussian3D.
                    float alpha = raster_eval_alpha(gaussian, rayDir);
                    if (alpha < g_AlphaCutoff) continue;
                    if (depthDebug) {
                        float contributionWeight = trans * alpha;
                        if (contributionWeight > SMALL_VALUE) {
                            float3 rayLocal = rotate_by_quat(rayDir, gaussian.rotation) / max(gaussian.logScale, float3(SMALL_VALUE));
                            float denom = dot(rayLocal, rayLocal);
                            if (denom > RAY_DENOMINATOR_FLOOR) {
                                float tClosest = -dot(rayLocal, gaussian.position) / denom;
                                if (tClosest > 0.0) {
                                    contributionWeight /= tClosest + 1e-4;
                                    float totalWeight = depthWeightSum + contributionWeight;
                                    float delta = tClosest - depthMean;
                                    depthMean += (contributionWeight / totalWeight) * delta;
                                    depthM2 += depthWeightSum * contributionWeight * delta * delta / totalWeight;
                                    depthWeightSum = totalWeight;
                                }
                            }
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
    if (!active) return;
    uint processedCount = range.x != INVALID_START && processedEnd >= range.x ? (processedEnd - range.x) : 0u;
    float4 output = raster_output(accum, trans, g_Background);
    if (g_DebugMode == DEBUG_MODE_PROCESSED_COUNT) {
        output = float4(raster_processed_count_color(processedCount, g_DebugMaxSplatSteps), 1.0);
    } else if (g_DebugMode == DEBUG_MODE_DEPTH_MEAN || g_DebugMode == DEBUG_MODE_DEPTH_STD) {
        float debugValue = 0.0;
        if (depthWeightSum > SMALL_VALUE) {
            float depthStd = sqrt(max(depthM2 / depthWeightSum, 0.0));
            debugValue = g_DebugMode == DEBUG_MODE_DEPTH_MEAN
                ? debug_normalize_range(depthMean, g_DebugDepthMeanRange)
                : debug_normalize_range(depthStd, g_DebugDepthStdRange);
        }
        output = float4(jet_colormap(debugValue), 1.0);
    }
    g_Output.store(p, output);
    g_ForwardState.store(p, float4(accum, trans));
    g_ForwardEnd.store(p, processedEnd);
}

// Atomic accumulation into the flat parameter-gradient buffer.
void add_param_grad(uint splatId, uint paramId, float value) {
    if (value == 0.0) return;
    g_ParamGrads.add(int(splatId * PARAM_COUNT + paramId), value);
}

struct PixelBackwardState {
    bool valid;
    float3 accum;
    float trans;
    float3 dAccum;
    float dTrans;
    float3 rayDir;
    uint processedEnd;
};

struct GaussianBackwardGrads {
    float3 position;
    float3 logScale;
    float4 rotation;
    float3 color;
    float rawOpacity;
};

PixelBackwardState make_invalid_pixel_backward_state() {
    PixelBackwardState state;
    state.valid = false;
    state.accum = 0.0;
    state.trans = 0.0;
    state.dAccum = 0.0;
    state.dTrans = 0.0;
    state.rayDir = 0.0;
    state.processedEnd = 0u;
    return state;
}

GaussianBackwardGrads make_zero_gaussian_backward_grads() {
    GaussianBackwardGrads grads;
    grads.position = 0.0;
    grads.logScale = 0.0;
    grads.rotation = 0.0;
    grads.color = 0.0;
    grads.rawOpacity = 0.0;
    return grads;
}

PixelBackwardState load_pixel_backward_state(bool active, int2 p, float2 raySample) {
    PixelBackwardState state = make_invalid_pixel_backward_state();
    if (!active) return state;
    float4 state4 = g_ForwardState[p];
    var dpAccum = diffPair(state4.xyz);
    var dpTrans = diffPair(state4.w);
    bwd_diff(raster_output)(dpAccum, dpTrans, g_Background, g_OutputGrad[p]);
    state.valid = true;
    state.accum = state4.xyz;
    state.trans = state4.w;
    state.dAccum = dpAccum.getDifferential();
    state.dTrans = dpTrans.getDifferential();
    state.rayDir = g_Camera.screen_to_world_ray(raySample);
    state.processedEnd = g_ForwardEnd[p];
    return state;
}

GaussianBackwardGrads apply_gaussian_to_pixel_backward_state(
    Gaussian3D gaussian,
    uint sortedEntryIndex,
    uint originalSplatId,
    inout PixelBackwardState state)
{
    if (!state.valid || sortedEntryIndex >= state.processedEnd) return make_zero_gaussian_backward_grads();

    float alpha = raster_eval_alpha(gaussian, state.rayDir);
    if (alpha < g_AlphaCutoff) return make_zero_gaussian_backward_grads();

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

    var dpCachedGaussian = diffPair(gaussian);
    bwd_diff(raster_eval_alpha)(dpCachedGaussian, state.rayDir, dAlpha);
    Gaussian3D.Differential dCachedGaussian = dpCachedGaussian.getDifferential();
    dGaussian.colorAlpha.xyz += dCachedGaussian.colorAlpha.xyz;
    float fused = max(gaussian.colorAlpha.w, SMALL_VALUE);
    float dRawOpacity = dCachedGaussian.colorAlpha.w * fused * (1.0 - fused);
    dCachedGaussian.colorAlpha = 0.0;

    Gaussian3D originalGaussian = Gaussian3D.load(g_Splats, g_SplatCount, originalSplatId);
    var dpOriginalGaussian = diffPair(originalGaussian);
    bwd_diff(Gaussian3D.camera_gaussian)(dpOriginalGaussian, g_Camera, g_RadiusScale, GAUSSIAN_SUPPORT_SIGMA_RADIUS, g_MaxAnisotropy, dCachedGaussian);
    Gaussian3D.Differential dOriginalGaussian = dpOriginalGaussian.getDifferential();

    GaussianBackwardGrads grads;
    grads.position = dOriginalGaussian.position;
    grads.logScale = dOriginalGaussian.logScale;
    grads.rotation = dOriginalGaussian.rotation;
    grads.color = dGaussian.colorAlpha.xyz + dOriginalGaussian.colorAlpha.xyz;
    grads.rawOpacity = dRawOpacity;

    state.accum = prevAccum;
    state.trans = prevTrans;
    return grads;
}

void flush_gaussian_backward_grads(uint originalSplatId, GaussianBackwardGrads grads) {
    float4 colorAlpha = WaveActiveSum(float4(grads.color, grads.rawOpacity));
    float3 position = WaveActiveSum(grads.position);
    float3 logScale = WaveActiveSum(grads.logScale);
    float4 rotation = WaveActiveSum(grads.rotation);
    if (WaveIsFirstLane()) {
        add_param_grad(originalSplatId, PARAM_COLOR_R, colorAlpha.x);
        add_param_grad(originalSplatId, PARAM_COLOR_G, colorAlpha.y);
        add_param_grad(originalSplatId, PARAM_COLOR_B, colorAlpha.z);
        add_param_grad(originalSplatId, PARAM_POSITION_X, position.x);
        add_param_grad(originalSplatId, PARAM_POSITION_Y, position.y);
        add_param_grad(originalSplatId, PARAM_POSITION_Z, position.z);
        add_param_grad(originalSplatId, PARAM_SCALE_X, logScale.x);
        add_param_grad(originalSplatId, PARAM_SCALE_Y, logScale.y);
        add_param_grad(originalSplatId, PARAM_SCALE_Z, logScale.z);
        add_param_grad(originalSplatId, PARAM_ROTATION_W, rotation.x);
        add_param_grad(originalSplatId, PARAM_ROTATION_X, rotation.y);
        add_param_grad(originalSplatId, PARAM_ROTATION_Y, rotation.z);
        add_param_grad(originalSplatId, PARAM_ROTATION_Z, rotation.w);
        add_param_grad(originalSplatId, PARAM_RAW_OPACITY, colorAlpha.w);
    }
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
    float2 raySample = float2(p) + stochastic_subpixel_offset(pixelId, g_RenderSeed);
    PixelBackwardState state = load_pixel_backward_state(active, p, raySample);
    for (uint batchBase = range.y; batchBase > range.x;) {
        uint batchStart = batchBase > range.x + RASTER_BATCH ? batchBase - RASTER_BATCH : range.x;
        uint batchCount = batchBase - batchStart;
        // Replay the same tile-local batches in reverse order from the saved forward stopping point.
        prefetch_batch(batchStart, batchBase, lane);
        GroupMemoryBarrierWithGroupSync();
        if (active && range.x != INVALID_START && range.y > range.x) {
            for (uint rev = 0u; rev < batchCount; rev++) {
                uint localIdx = batchCount - 1u - rev;
                uint globalIdx = batchStart + localIdx;
                uint splatId = gsBatchSplatId[localIdx];
                if (splatId >= g_SplatCount) continue;
                Gaussian3D gaussian = gsGaussian[localIdx];
                uint originalSplatId = g_SplatOrder[int(splatId)];
                if (originalSplatId >= g_SplatCount) continue;
                GaussianBackwardGrads grads = apply_gaussian_to_pixel_backward_state(gaussian, globalIdx, originalSplatId, state);
                flush_gaussian_backward_grads(originalSplatId, grads);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        batchBase = batchStart;
    }
}
