import "slangpy";
import "common.slang";

uniform Tensor<float, 1> g_Splats;
uniform RWTensor<uint, 1> g_TileCounts;
uniform Tensor<uint, 1> g_TileOffsets;
uniform RWTensor<TileEntry, 1> g_TileEntries;
uniform Tensor<TileEntry, 1> g_SortedEntries;
uniform RWTensor<ProjectionState, 1> g_ProjectionState;
uniform RWTensor<RasterState, 1> g_RasterState;
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
uniform uint g_Sampled5Iters;
uniform float g_Sampled5SafetyScale;
uniform float g_Sampled5RadiusPadPx;
uniform float g_Sampled5Eps;

groupshared float3 gsRoLocal[RASTER_BATCH];
groupshared float3 gsScale[RASTER_BATCH];
groupshared float4 gsRotation[RASTER_BATCH];
groupshared float4 gsColorAlpha[RASTER_BATCH];
groupshared uint gsBatchSplatId[RASTER_BATCH];

void write_cache(uint splatId, CachedEllipsoid cached, float4 colorAlpha) {
    int[1] at = { int(splatId) };
    RasterState state;
    state.roLocal = cached.roLocal;
    state.scale = cached.scale;
    state.rotation = cached.rotation;
    state.colorAlpha = colorAlpha;
    g_RasterState.store(at, state);
}

void clear_projection(uint splatId) {
    int[1] i = { int(splatId) };
    ProjectionState projection;
    projection.centerRadiusDepth = 0.0;
    projection.conicOpacity = 0.0;
    RasterState state;
    state.roLocal = 0.0;
    state.scale = 0.0;
    state.rotation = 0.0;
    state.colorAlpha = 0.0;
    g_ProjectionState.store(i, projection);
    g_RasterState.store(i, state);
    g_TileCounts.store(i, 0u);
}

bool project_splat(uint splatId, out Ellipse ellipse) {
    Gaussian3D gaussian = Gaussian3D.load(g_Splats, g_SplatCount, splatId);
    CachedEllipsoid cached = Gaussian3D.build_cached_ellipsoid(gaussian, g_Camera, g_RadiusScale);
    float3 camPos = g_Camera.world_point_to_camera(gaussian.position);
    float depth = camPos.z;
    float3 cameraOffset = gaussian.position - g_Camera.camPos;
    float camDistance = length(cameraOffset);
    float2 fitCenter = 0.0;
    bool projected = g_Camera.project_camera_to_screen(camPos, fitCenter);
    if (!projected) fitCenter = camPos.xy * (g_Camera.focalPixels / max(camPos.z, SMALL_VALUE)) + g_Camera.principalPoint;
    float maxAxisWorld = max(cached.scale.x, max(cached.scale.y, cached.scale.z));
    float fallbackRadius = max(g_Camera.focalPixels.x, g_Camera.focalPixels.y) * maxAxisWorld / max(depth, SMALL_VALUE);
    float radiusCap = max(max(g_Camera.viewport.x, g_Camera.viewport.y), MIN_SCREEN_RADIUS_PX);
    float fitRadius = clamp(fallbackRadius * g_Sampled5SafetyScale + g_Sampled5RadiusPadPx, MIN_SCREEN_RADIUS_PX, radiusCap);
    float invFitRadius2 = 1.0 / max(fitRadius * fitRadius, SMALL_VALUE);
    float3 fitConic = float3(invFitRadius2, 0.0, invFitRadius2);
    float3 localPoints[MVEE_POINT_COUNT];
    bool ok = generate_silhouette_points_local5(cached.roLocal, g_Sampled5Eps, localPoints);
    if (ok) {
        float2 screenPts[MVEE_POINT_COUNT];
        [unroll] for (uint i = 0u; i < MVEE_POINT_COUNT; i++) {
            float3 worldPt = gaussian.position + rotate_by_quat(localPoints[i] * cached.scale, float4(cached.rotation.x, -cached.rotation.yzw));
            float2 sp = 0.0;
            bool ptOk = g_Camera.project_camera_to_screen(g_Camera.world_point_to_camera(worldPt), sp);
            screenPts[i] = sp;
            ok = ok && ptOk;
        }
        if (ok) {
            MveeFitResult fit;
            bool fitOk = fit_mvee_5pt_fixed_iters(screenPts, min(g_Sampled5Iters, MVEE_MAX_ITERS), g_Sampled5Eps, fit);
            if (fitOk) {
                fitCenter = fit.centerPx;
                fitRadius = clamp(fit.radiusPx * g_Sampled5SafetyScale + g_Sampled5RadiusPadPx, MIN_SCREEN_RADIUS_PX, radiusCap);
                float conicScale = fitRadius / max(fit.radiusPx, g_Sampled5Eps);
                fitConic = fit.conic / max(conicScale * conicScale, g_Sampled5Eps);
            }
        }
    }
    ellipse.centerPx = fitCenter;
    ellipse.radiusPx = fitRadius;
    ellipse.conic = fitConic;
    ellipse.bboxExtent = float2(fitRadius, fitRadius);
    if (!ellipse.prepare_conic()) return false;
    bool visible = depth > DEPTH_VISIBILITY_FLOOR
        && camDistance > g_Camera.nearDepth
        && camDistance < g_Camera.farDepth
        && (fitCenter.x + fitRadius) >= 0.0
        && (fitCenter.x - fitRadius) < g_Camera.viewport.x
        && (fitCenter.y + fitRadius) >= 0.0
        && (fitCenter.y - fitRadius) < g_Camera.viewport.y;
    if (!visible) return false;
    float fusedOpacity = gaussian.build_raster_fused_opacity();
    int[1] at = { int(splatId) };
    ProjectionState projection;
    projection.centerRadiusDepth = float4(fitCenter, fitRadius, camDistance);
    projection.conicOpacity = float4(fitConic, 1.0);
    g_ProjectionState.store(at, projection);
    write_cache(splatId, cached, float4(gaussian.colorAlpha.xyz, fusedOpacity));
    return true;
}

uint count_tiles(Ellipse ellipse) {
    int minX = clamp(int(floor((ellipse.centerPx.x - ellipse.bboxExtent.x) / TILE_SIZE)), 0, int(g_TileGrid.x) - 1);
    int maxX = clamp(int(ceil((ellipse.centerPx.x + ellipse.bboxExtent.x) / TILE_SIZE)), 0, int(g_TileGrid.x) - 1);
    int minY = clamp(int(floor((ellipse.centerPx.y - ellipse.bboxExtent.y) / TILE_SIZE)), 0, int(g_TileGrid.y) - 1);
    int maxY = clamp(int(ceil((ellipse.centerPx.y + ellipse.bboxExtent.y) / TILE_SIZE)), 0, int(g_TileGrid.y) - 1);
    if (minX > maxX || minY > maxY) return 0u;
    uint count = 0u;
    for (int y = minY; y <= maxY; y++) for (int x = minX; x <= maxX; x++) if (ellipse.tile_intersects(uint(x), uint(y))) count++;
    return count;
}

//TODO: this can be done via a module call, instead of a dispatch
[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csProjectCount(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;
    clear_projection(splatId);
    Ellipse ellipse;
    if (!project_splat(splatId, ellipse)) return;
    int[1] at = { int(splatId) };
    g_TileCounts.store(at, count_tiles(ellipse));
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
    int minX = clamp(int(floor((ellipse.centerPx.x - ellipse.bboxExtent.x) / TILE_SIZE)), 0, int(g_TileGrid.x) - 1);
    int maxX = clamp(int(ceil((ellipse.centerPx.x + ellipse.bboxExtent.x) / TILE_SIZE)), 0, int(g_TileGrid.x) - 1);
    int minY = clamp(int(floor((ellipse.centerPx.y - ellipse.bboxExtent.y) / TILE_SIZE)), 0, int(g_TileGrid.y) - 1);
    int maxY = clamp(int(ceil((ellipse.centerPx.y + ellipse.bboxExtent.y) / TILE_SIZE)), 0, int(g_TileGrid.y) - 1);
    uint writeIndex = g_TileOffsets[int(splatId)];
    for (int y = minY; y <= maxY; y++) for (int x = minX; x <= maxX; x++) {
        if (!ellipse.tile_intersects(uint(x), uint(y))) continue;
        int[1] wi = { int(writeIndex) };
        TileEntry entry;
        entry.ids = uint2(uint(y) * g_TileGrid.x + uint(x), splatId);
        g_TileEntries.store(wi, entry);
        writeIndex++;
    }
}

void prefetch_batch(uint batchStart, uint batchEnd, uint lane) {
    uint batchCount = batchEnd - batchStart;
    for (uint localIdx = lane; localIdx < batchCount; localIdx += RASTER_GROUP_THREADS) {
        uint splatId = g_SortedEntries[int(batchStart + localIdx)].ids.y;
        gsBatchSplatId[localIdx] = splatId;
        RasterState state = g_RasterState[int(splatId)];
        gsRoLocal[localIdx] = state.roLocal;
        gsScale[localIdx] = state.scale;
        gsRotation[localIdx] = state.rotation;
        gsColorAlpha[localIdx] = state.colorAlpha;
    }
}

[Differentiable]
float4 shade_step(float3 accum, float trans, float4 colorAlpha, float alpha) {
    return float4(accum + trans * alpha * colorAlpha.xyz, trans * (1.0 - alpha));
}

[shader("compute")]
[numthreads(RASTER_THREAD_TILE_DIM, RASTER_THREAD_TILE_DIM, 1)]
void csRasterForward(uint3 dispatchThreadID : SV_DispatchThreadID, uint lane : SV_GroupIndex) {
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= uint(g_Camera.viewport.x) || py >= uint(g_Camera.viewport.y)) return;
    uint tileX = px / TILE_SIZE;
    uint tileY = py / TILE_SIZE;
    uint tileId = tileY * g_TileGrid.x + tileX;
    uint2 range = g_TileRanges[int(tileId)];
    float3 accum = 0.0;
    float trans = 1.0;
    uint processed = 0u;
    uint processedEnd = range.x;
    float3 rayDir = g_Camera.screen_to_world_ray(float2(px, py) + 0.5);
    if (range.x != INVALID_START && range.y > range.x) {
        uint end = range.y;
        bool done = false;
        for (uint batchStart = range.x; batchStart < end; batchStart += RASTER_BATCH) {
            uint batchEnd = min(batchStart + RASTER_BATCH, end);
            uint batchCount = batchEnd - batchStart;
            prefetch_batch(batchStart, batchEnd, lane);
            GroupMemoryBarrierWithGroupSync();
            if (!done) {
                for (uint localIdx = 0u; localIdx < batchCount; localIdx++) {
                    processedEnd = batchStart + localIdx + 1u;
                    float alpha = raster_eval_alpha(gsRoLocal[localIdx], gsScale[localIdx], gsRotation[localIdx], gsColorAlpha[localIdx].w, rayDir);
                    if (alpha < g_AlphaCutoff) continue;
                    float4 blended = shade_step(accum, trans, gsColorAlpha[localIdx], alpha);
                    accum = blended.xyz;
                    trans = blended.w;
                    processed++;
                    if (trans < g_TransmittanceThreshold) { done = true; break; }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    int2 p = int2(int(py), int(px));
    g_Output.store(p, (range.x == INVALID_START || range.y <= range.x) ? float4(pow(max(g_Background, 0.0), OUTPUT_GAMMA), 1.0) : raster_output(accum, trans, g_Background));
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
    uint tileX = px / TILE_SIZE;
    uint tileY = py / TILE_SIZE;
    uint tileId = tileY * g_TileGrid.x + tileX;
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
            float3 roLocal = gsRoLocal[localIdx];
            float3 scale = gsScale[localIdx];
            float4 colorAlpha = gsColorAlpha[localIdx];
            float4 rotation = gsRotation[localIdx];
            float alpha = raster_eval_alpha(roLocal, scale, rotation, colorAlpha.w, rayDir);
            if (alpha < g_AlphaCutoff) continue;
            float prevTrans = trans / max(1.0 - alpha, SMALL_VALUE);
            float3 prevAccum = accum - prevTrans * alpha * colorAlpha.xyz;
            float3 dColor = dAccum * (prevTrans * alpha);
            float dAlpha = dot(dAccum, prevTrans * colorAlpha.xyz) - dTrans * prevTrans;
            add_param_grad(splatId, PARAM_COLOR_R, dColor.x);
            add_param_grad(splatId, PARAM_COLOR_G, dColor.y);
            add_param_grad(splatId, PARAM_COLOR_B, dColor.z);
            Gaussian3D g = Gaussian3D.load(g_Splats, g_SplatCount, splatId);
            CachedEllipsoid cached = { roLocal, scale, rotation };
            var dpRo = diffPair(cached.roLocal);
            var dpSc = diffPair(cached.scale);
            var dpRot = diffPair(cached.rotation);
            var dpFused = diffPair(colorAlpha.w);
            bwd_diff(raster_eval_alpha)(dpRo, dpSc, dpRot, dpFused, rayDir, dAlpha);
            CachedEllipsoid.Differential dCached;
            dCached.roLocal = dpRo.getDifferential();
            dCached.scale = dpSc.getDifferential();
            dCached.rotation = dpRot.getDifferential();
            var dpGaussian = diffPair(g);
            bwd_diff(Gaussian3D.build_cached_ellipsoid)(dpGaussian, g_Camera, g_RadiusScale, dCached);
            float fused = max(colorAlpha.w, SMALL_VALUE);
            float dRawOpacity = dpFused.getDifferential() * fused * (1.0 - fused);
            Gaussian3D.Differential dGaussian = dpGaussian.getDifferential();
            float3 dPos = dGaussian.position;
            float3 dScale = dGaussian.logScale;
            add_param_grad(splatId, PARAM_POSITION_X, dPos.x);
            add_param_grad(splatId, PARAM_POSITION_Y, dPos.y);
            add_param_grad(splatId, PARAM_POSITION_Z, dPos.z);
            add_param_grad(splatId, PARAM_SCALE_X, dScale.x);
            add_param_grad(splatId, PARAM_SCALE_Y, dScale.y);
            add_param_grad(splatId, PARAM_SCALE_Z, dScale.z);
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
