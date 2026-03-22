import "slangpy";
import "common.slang";

uniform Tensor<float, 1> g_Splats;
uniform RWTensor<uint, 1> g_TileCounts;
uniform Tensor<uint, 1> g_TileOffsets;
uniform RWTensor<TileEntry, 1> g_TileEntries;
uniform Tensor<TileEntry, 1> g_SortedEntries;
uniform RWTensor<ProjectionState, 1> g_ProjectionState;
uniform RWTensor<Gaussian3D, 1> g_RasterState;
uniform Tensor<uint2, 1> g_TileRanges;
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
uniform float g_AlphaCutoff;
uniform float g_TransmittanceThreshold;

groupshared Gaussian3D gsGaussian[RASTER_BATCH];
groupshared uint gsBatchSplatId[RASTER_BATCH];

void write_cache(uint splatId, Gaussian3D gaussian, float fusedOpacity) {
    int[1] at = { int(splatId) };
    gaussian.colorAlpha.w = fusedOpacity;
    g_RasterState.store(at, gaussian);
}

uint count_tiles(Ellipse ellipse) {
    bool transpose = ellipse.bboxExtent.x < ellipse.bboxExtent.y;
    float2 localCenter = transpose ? ellipse.centerPx.yx : ellipse.centerPx;
    float2 localExtent = transpose ? ellipse.bboxExtent.yx : ellipse.bboxExtent;
    float3 localConic = transpose ? float3(ellipse.conic.z, ellipse.conic.y, ellipse.conic.x) : ellipse.conic;
    int minSpan = clamp(int(floor((localCenter.x - localExtent.x) / TILE_SIZE)), 0, int(g_TileGrid.x) - 1);
    int maxSpan = clamp(int(ceil((localCenter.x + localExtent.x) / TILE_SIZE)), 0, int(g_TileGrid.x) - 1);
    int minStrip = clamp(int(floor((localCenter.y - localExtent.y) / TILE_SIZE)), 0, int(g_TileGrid.y) - 1);
    int maxStrip = clamp(int(ceil((localCenter.y + localExtent.y) / TILE_SIZE)), 0, int(g_TileGrid.y) - 1);
    if (minSpan > maxSpan || minStrip > maxStrip) return 0u;
    uint count = 0u;
    for (int strip = minStrip; strip <= maxStrip; strip++) {
        float stripMin = strip * TILE_SIZE - localCenter.y;
        float stripMax = stripMin + TILE_SIZE;
        float spanMin, spanMax;
        if (!ellipse.span_for_strip(localConic, stripMin, stripMax, spanMin, spanMax)) continue;
        int spanTileMin = clamp(int(floor((spanMin + localCenter.x) / TILE_SIZE)), minSpan, maxSpan);
        int spanTileMax = clamp(int(floor((spanMax + localCenter.x) / TILE_SIZE)), minSpan, maxSpan);
        if (spanTileMin <= spanTileMax) count += uint(spanTileMax - spanTileMin + 1);
    }
    return count;
}

//TODO: this can be done via a module call, instead of a dispatch
[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csProjectCount(uint3 dispatchThreadID : SV_DispatchThreadID) {
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
    uint tileCount = 0u;

    Gaussian3D gaussian = Gaussian3D.load(g_Splats, g_SplatCount, splatId);
    float fusedOpacity = gaussian.build_raster_fused_opacity();
    if (fusedOpacity >= g_AlphaCutoff) {
        float supportSigmaRadius = sqrt(max(-2.0 * log(g_AlphaCutoff / fusedOpacity), 0.0));
        Gaussian3D cameraGaussian = Gaussian3D.camera_gaussian(gaussian, g_Camera, g_RadiusScale, supportSigmaRadius);
        float3 cameraPosition = g_Camera.world_point_to_camera(gaussian.position);
        float depth = cameraPosition.z;
        float cameraDistance = length(gaussian.position - g_Camera.camPos);
        // Cheap camera-space visibility tests first, then let the outline helper decide whether a fit is worth doing.
        bool hasFit = depth > DEPTH_VISIBILITY_FLOOR
            && cameraDistance > g_Camera.nearDepth
            && cameraDistance < g_Camera.farDepth;
        Ellipse ellipse;
        if (hasFit) {
            hasFit = compute_gaussian_outline_ellipse(gaussian, g_Camera, g_RadiusScale, supportSigmaRadius, ellipse);
        }
        if (hasFit) {
            ellipse.radiusPx = max(ellipse.radiusPx * ELLIPSE_SAFETY_SCALE + ELLIPSE_RADIUS_PAD_PX, MIN_SCREEN_RADIUS_PX);
            ellipse.bboxExtent = float2(ellipse.radiusPx, ellipse.radiusPx);
            hasFit = ellipse.prepare_conic();
        }
        if (hasFit) {
            projection.centerRadiusDepth = float4(ellipse.centerPx, ellipse.radiusPx, cameraDistance);
            projection.conicOpacity = float4(ellipse.conic, 1.0);
            rasterGaussian = gaussian;
            rasterGaussian.colorAlpha.w = fusedOpacity;
            tileCount = count_tiles(ellipse);
        }
    }

    g_ProjectionState.store(at, projection);
    g_RasterState.store(at, rasterGaussian);
    g_TileCounts.store(at, tileCount);
}

// TODO: this can be done via a module call, instead of a dispatch
[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csProjectFill(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;
    uint tileCount = g_TileCounts[int(splatId)];
    if (tileCount == 0u) return;
    ProjectionState projection = g_ProjectionState[int(splatId)];
    Ellipse ellipse;
    ellipse.centerPx = projection.centerRadiusDepth.xy;
    ellipse.radiusPx = projection.centerRadiusDepth.z;
    ellipse.conic = projection.conicOpacity.xyz;
    if (!ellipse.prepare_conic()) return;
    bool transpose = ellipse.bboxExtent.x < ellipse.bboxExtent.y;
    float2 localCenter = transpose ? ellipse.centerPx.yx : ellipse.centerPx;
    float2 localExtent = transpose ? ellipse.bboxExtent.yx : ellipse.bboxExtent;
    float3 localConic = transpose ? float3(ellipse.conic.z, ellipse.conic.y, ellipse.conic.x) : ellipse.conic;
    int minSpan = clamp(int(floor((localCenter.x - localExtent.x) / TILE_SIZE)), 0, int(g_TileGrid.x) - 1);
    int maxSpan = clamp(int(ceil((localCenter.x + localExtent.x) / TILE_SIZE)), 0, int(g_TileGrid.x) - 1);
    int minStrip = clamp(int(floor((localCenter.y - localExtent.y) / TILE_SIZE)), 0, int(g_TileGrid.y) - 1);
    int maxStrip = clamp(int(ceil((localCenter.y + localExtent.y) / TILE_SIZE)), 0, int(g_TileGrid.y) - 1);
    uint writeIndex = g_TileOffsets[int(splatId)];
    for (int strip = minStrip; strip <= maxStrip; strip++) {
        float stripMin = strip * TILE_SIZE - localCenter.y;
        float stripMax = stripMin + TILE_SIZE;
        float spanMin, spanMax;
        if (!ellipse.span_for_strip(localConic, stripMin, stripMax, spanMin, spanMax)) continue;
        int spanTileMin = clamp(int(floor((spanMin + localCenter.x) / TILE_SIZE)), minSpan, maxSpan);
        int spanTileMax = clamp(int(floor((spanMax + localCenter.x) / TILE_SIZE)), minSpan, maxSpan);
        for (int span = spanTileMin; span <= spanTileMax; span++) {
            int[1] wi = { int(writeIndex) };
            TileEntry entry;
            uint2 tileXY = transpose ? uint2(uint(strip), uint(span)) : uint2(uint(span), uint(strip));
            entry.ids = uint2(tileXY.y * g_TileGrid.x + tileXY.x, splatId);
            g_TileEntries.store(wi, entry);
            writeIndex++;
        }
    }
}

void prefetch_batch(uint batchStart, uint batchEnd, uint lane) {
    uint batchCount = batchEnd - batchStart;
    for (uint localIdx = lane; localIdx < batchCount; localIdx += RASTER_GROUP_THREADS) {
        uint splatId = g_SortedEntries[int(batchStart + localIdx)].ids.y;
        gsBatchSplatId[localIdx] = splatId;
        gsGaussian[localIdx] = g_RasterState[int(splatId)];
    }
}

[Differentiable]
float4 shade_step(float3 accum, float trans, Gaussian3D gaussian, float alpha) {
    return float4(accum + trans * alpha * gaussian.colorAlpha.xyz, trans * (1.0 - alpha));
}

[shader("compute")]
[numthreads(RASTER_THREAD_TILE_DIM, RASTER_THREAD_TILE_DIM, 1)]
void csRasterForward(uint3 dispatchThreadID : SV_DispatchThreadID, uint lane : SV_GroupIndex) {
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= uint(g_Camera.viewport.x) || py >= uint(g_Camera.viewport.y)) return;
    uint tileId = (py / TILE_SIZE) * g_TileGrid.x + (px / TILE_SIZE);
    uint2 range = g_TileRanges[int(tileId)];
    float3 accum = 0.0;
    float trans = 1.0;
    uint processedEnd = range.x;
    float3 rayDir = g_Camera.screen_to_world_ray(float2(px, py) + 0.5);
    if (range.x != INVALID_START && range.y > range.x) {
        bool done = false;
        for (uint batchStart = range.x; batchStart < range.y; batchStart += RASTER_BATCH) {
            uint batchEnd = min(batchStart + RASTER_BATCH, range.y);
            uint batchCount = batchEnd - batchStart;
            prefetch_batch(batchStart, batchEnd, lane);
            GroupMemoryBarrierWithGroupSync();
            if (!done) {
                for (uint localIdx = 0u; localIdx < batchCount; localIdx++) {
                    processedEnd = batchStart + localIdx + 1u;
                    Gaussian3D gaussian = gsGaussian[localIdx];
                    float alpha = raster_eval_alpha(gaussian, g_Camera, g_RadiusScale, rayDir);
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
    int2 p = int2(int(py), int(px));
    g_Output.store(p, raster_output(accum, trans, g_Background));
    g_ForwardState.store(p, float4(accum, trans));
    g_ForwardEnd.store(p, processedEnd);
}

void add_param_grad(uint splatId, uint paramId, float value) {
    if (value == 0.0) return;
    g_ParamGrads.add(int(splatId * PARAM_COUNT + paramId), value);
}

[shader("compute")]
[numthreads(RASTER_THREAD_TILE_DIM, RASTER_THREAD_TILE_DIM, 1)]
void csRasterBackward(uint3 dispatchThreadID : SV_DispatchThreadID, uint lane : SV_GroupIndex) {
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= uint(g_Camera.viewport.x) || py >= uint(g_Camera.viewport.y)) return;
    uint tileId = (py / TILE_SIZE) * g_TileGrid.x + (px / TILE_SIZE);
    uint2 range = g_TileRanges[int(tileId)];
    if (range.x == INVALID_START || range.y <= range.x) return;
    int2 p = int2(int(py), int(px));
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
        prefetch_batch(batchStart, batchBase, lane);
        GroupMemoryBarrierWithGroupSync();
        for (uint rev = 0u; rev < batchCount; rev++) {
            uint localIdx = batchCount - 1u - rev;
            uint globalIdx = batchStart + localIdx;
            if (globalIdx >= processedEnd) continue;
            uint splatId = gsBatchSplatId[localIdx];
            Gaussian3D gaussian = gsGaussian[localIdx];
            float alpha = raster_eval_alpha(gaussian, g_Camera, g_RadiusScale, rayDir);
            if (alpha < g_AlphaCutoff) continue;
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
            bwd_diff(raster_eval_alpha)(dpAlphaGaussian, g_Camera, g_RadiusScale, rayDir, dAlpha);
            Gaussian3D.Differential dGaussianAlpha = dpAlphaGaussian.getDifferential();
            dGaussian.position += dGaussianAlpha.position;
            dGaussian.logScale += dGaussianAlpha.logScale;
            dGaussian.rotation += dGaussianAlpha.rotation;
            dGaussian.colorAlpha += dGaussianAlpha.colorAlpha;
            float fused = max(gaussian.colorAlpha.w, SMALL_VALUE);
            float dRawOpacity = dGaussian.colorAlpha.w * fused * (1.0 - fused);
            add_param_grad(splatId, PARAM_COLOR_R, dGaussian.colorAlpha.x);
            add_param_grad(splatId, PARAM_COLOR_G, dGaussian.colorAlpha.y);
            add_param_grad(splatId, PARAM_COLOR_B, dGaussian.colorAlpha.z);
            add_param_grad(splatId, PARAM_POSITION_X, dGaussian.position.x);
            add_param_grad(splatId, PARAM_POSITION_Y, dGaussian.position.y);
            add_param_grad(splatId, PARAM_POSITION_Z, dGaussian.position.z);
            add_param_grad(splatId, PARAM_SCALE_X, dGaussian.logScale.x);
            add_param_grad(splatId, PARAM_SCALE_Y, dGaussian.logScale.y);
            add_param_grad(splatId, PARAM_SCALE_Z, dGaussian.logScale.z);
            add_param_grad(splatId, PARAM_ROTATION_W, dGaussian.rotation.x);
            add_param_grad(splatId, PARAM_ROTATION_X, dGaussian.rotation.y);
            add_param_grad(splatId, PARAM_ROTATION_Y, dGaussian.rotation.z);
            add_param_grad(splatId, PARAM_ROTATION_Z, dGaussian.rotation.w);
            add_param_grad(splatId, PARAM_RAW_OPACITY, dRawOpacity);
            accum = prevAccum;
            trans = prevTrans;
        }
        GroupMemoryBarrierWithGroupSync();
        batchBase = batchStart;
    }
}
