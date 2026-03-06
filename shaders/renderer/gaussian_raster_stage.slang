import "gaussian_types.slang";
import "camera.slang";
import "math_constants.slang";

struct RasterState : IDifferentiable
{
    float3 accum;
    float trans;

    [Differentiable]
    RasterState blend(float3 color, float alpha)
    {
        return { accum + trans * alpha * color, trans * (1.0 - alpha) };
    }

    RasterState undo(float3 color, float alpha)
    {
        float denom = max(1.0 - alpha, SMALL_VALUE);
        float prevTrans = trans / denom;
        float3 prevAccum = accum - prevTrans * alpha * color;
        return { prevAccum, prevTrans };
    }
};

void atomic_add_grad<let N : int>(RWStructuredBuffer<float> gradBuffer, uint base, vector<float, N> grad)
{
    [unroll] for (int i = 0; i < N; i++)
    {
        float value = grad[i];
        if (value == 0.0) continue;
        InterlockedAdd(gradBuffer[base + uint(i)], value);
    }
}

uint grad_state_base(uint localIdx, uint fieldOffset)
{
    return localIdx * SPLAT_GRAD_STATE_STRIDE + fieldOffset;
}

void atomic_add_grad_shared<let N : int>(uint base, vector<float, N> grad)
{
    [unroll] for (int i = 0; i < N; i++)
    {
        float value = grad[i];
        if (value == 0.0) continue;
        InterlockedAdd(gsGradState[base + uint(i)], value);
    }
}

vector<float, N> load_grad_shared<let N : int>(uint base)
{
    vector<float, N> grad = 0.0;
    [unroll] for (int i = 0; i < N; i++) grad[i] = gsGradState[base + uint(i)];
    return grad;
}

void clear_batch_gradient_cache(uint batchCount, uint lane)
{
    if (lane >= batchCount) return;
    uint base = lane * SPLAT_GRAD_STATE_STRIDE;
    [unroll] for (uint i = 0u; i < SPLAT_GRAD_STATE_STRIDE; i++) gsGradState[base + i] = 0.0;
}

struct CachedEllipsoid : IDifferentiable
{
    float3 roLocal;
    float3 invScale;
    float4 quat;
};

[Differentiable]
CachedEllipsoid build_cached_ellipsoid(float3 position, float3 scale, float4 rotation)
{
    Gaussian3D gaussian;
    gaussian.position = position;
    gaussian.scale = scale;
    gaussian.rotation = rotation;
    gaussian.colorAlpha = 0.0;
    Ellipsoid ellipsoid = build_ellipsoid(gaussian);
    return { ellipsoid.roLocal, ellipsoid.invScale, ellipsoid.rotation };
}

struct RasterGaussian : IDifferentiable
{
    float3 roLocal;
    float3 invScale;
    float4 quat;
    float4 colorAlpha;

    static RasterGaussian load_global(uint splatId)
    {
        float3 position = g_Positions[splatId].xyz;
        float3 scale = g_Scales[splatId].xyz;
        float4 rotation = g_Rotations[splatId];
        float4 colorAlpha = g_ColorAlpha[splatId];
        CachedEllipsoid cached = build_cached_ellipsoid(position, scale, rotation);
        return { cached.roLocal, cached.invScale, cached.quat, colorAlpha };
    }

    [BackwardDerivative(raster_gaussian_load_shared_bwd)]
    static RasterGaussian load_shared(uint localIdx)
    {
        RasterGaussian gaussian;
        gaussian.roLocal = gsRoLocal[localIdx].xyz;
        gaussian.invScale = float3(gsInvScale[localIdx].xyz);
        gaussian.quat = float4(gsQuat[localIdx]);
        gaussian.colorAlpha = float4(gsColorAlpha[localIdx]);
        return gaussian;
    }

    [mutating]
    void store_shared(uint localIdx)
    {
        gsRoLocal[localIdx] = float4(roLocal, 0.0);
        gsInvScale[localIdx] = half4(float4(invScale, 0.0));
        gsQuat[localIdx] = half4(quat);
        gsColorAlpha[localIdx] = half4(colorAlpha);
    }

    static void accumulate_gradients_shared(uint localIdx, Differential dOut)
    {
        atomic_add_grad_shared<4>(grad_state_base(localIdx, SPLAT_GRAD_STATE_POS_LOCAL_OFFSET), float4(dOut.roLocal, 0.0));
        atomic_add_grad_shared<4>(grad_state_base(localIdx, SPLAT_GRAD_STATE_INV_SCALE_OFFSET), float4(dOut.invScale, 0.0));
        atomic_add_grad_shared<4>(grad_state_base(localIdx, SPLAT_GRAD_STATE_QUAT_OFFSET), dOut.quat);
        atomic_add_grad_shared<4>(grad_state_base(localIdx, SPLAT_GRAD_STATE_COLOR_ALPHA_OFFSET), dOut.colorAlpha);
    }

    static void raster_gaussian_load_shared_bwd(uint localIdx, Differential dOut)
    {
        accumulate_gradients_shared(localIdx, dOut);
    }
};

void flush_batch_gradient_cache(uint batchCount, uint lane)
{
    if (lane >= batchCount) return;
    uint splatId = gsBatchSplatId[lane];
    uint splatBase = splatId * SPLAT_GRAD_COMPONENTS;

    CachedEllipsoid.Differential dCached;
    dCached.roLocal = load_grad_shared<4>(grad_state_base(lane, SPLAT_GRAD_STATE_POS_LOCAL_OFFSET)).xyz;
    dCached.invScale = load_grad_shared<4>(grad_state_base(lane, SPLAT_GRAD_STATE_INV_SCALE_OFFSET)).xyz;
    dCached.quat = load_grad_shared<4>(grad_state_base(lane, SPLAT_GRAD_STATE_QUAT_OFFSET));
    float4 dColorAlpha = load_grad_shared<4>(grad_state_base(lane, SPLAT_GRAD_STATE_COLOR_ALPHA_OFFSET));

    float3 position = g_Positions[splatId].xyz;
    float3 scale = g_Scales[splatId].xyz;
    float4 rotation = g_Rotations[splatId];
    float4 colorAlpha = g_ColorAlpha[splatId];
    var dpPosition = diffPair(position);
    var dpScale = diffPair(scale);
    var dpRotation = diffPair(rotation);
    bwd_diff(build_cached_ellipsoid)(dpPosition, dpScale, dpRotation, dCached);

    atomic_add_grad<4>(g_GradPositions, splatBase, float4(dpPosition.getDifferential(), 0.0));
    atomic_add_grad<4>(g_GradScales, splatBase, float4(dpScale.getDifferential(), 0.0));
    atomic_add_grad<4>(g_GradRotations, splatBase, dpRotation.getDifferential());
    atomic_add_grad<4>(g_GradColorAlpha, splatBase, dColorAlpha);
}

[Differentiable]
float raster_eval_alpha(uint localIdx, no_diff float3 rayDir)
{
    RasterGaussian gaussian = RasterGaussian.load_shared(localIdx);

    float3 rdLocal = rotate_by_quat(rayDir, gaussian.quat) * gaussian.invScale;
    float denom = dot(rdLocal, rdLocal);
    if (denom <= RAY_DENOMINATOR_FLOOR) return 0.0;

    float tClosest = dot(rdLocal, -gaussian.roLocal) / denom;
    if (tClosest <= 0.0) return 0.0;

    float3 closest = gaussian.roLocal + rdLocal * tClosest;
    float rho = length(closest);
    if (rho >= 1.0) return 0.0;

    float coverage = saturate(1.0 - rho);
    coverage = coverage * coverage * (3.0 - 2.0 * coverage);
    float alpha = saturate(gaussian.colorAlpha.w * coverage);
    if (alpha < g_Raster.alphaCutoff) return 0.0;
    return alpha;
}

[Differentiable]
RasterState raster_blend_step(RasterState state, uint localIdx, float alpha)
{
    RasterGaussian gaussian = RasterGaussian.load_shared(localIdx);
    return state.blend(gaussian.colorAlpha.xyz, alpha);
}

RasterState raster_undo_blend_step(RasterState nextState, float4 colorAlpha, float alpha)
{
    return nextState.undo(colorAlpha.xyz, alpha);
}

[Differentiable]
float4 raster_output_from_state(RasterState state, no_diff float3 background)
{
    float3 finalColor = state.accum + state.trans * background;
    return float4(pow(max(finalColor, 0.0), OUTPUT_GAMMA), 1.0 - state.trans);
}

struct RasterPixelInfo
{
    bool pixelActive;
    bool hasTileRange;
    uint2 pixelCoord;
    uint samplePx;
    uint samplePy;
    uint start;
    uint end;
    float3 rayDir;
};

void prefetch_batch_from_global(uint batchStart, uint batchEnd, uint lane)
{
    uint idx = batchStart + lane;
    if (idx >= batchEnd) return;

    uint splatId = g_SortedValues[idx];
    gsBatchSplatId[lane] = splatId;
    RasterGaussian gaussian = RasterGaussian.load_global(splatId);
    gaussian.store_shared(lane);
}

RasterState raster_forward_replay(
    uint start,
    uint end,
    bool pixelActive,
    float3 rayDir,
    uint lane,
    out uint processedCountOut
)
{
    RasterState state = { 0.0, 1.0 };
    processedCountOut = 0u;
    bool pixelDone = !pixelActive;

    if (lane == 0u) gsStoppedPixels = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pixelDone) InterlockedAdd(gsStoppedPixels, 1u);

    for (uint batchStart = start; batchStart < end; batchStart += RASTER_BATCH)
    {
        prefetch_batch_from_global(batchStart, end, lane);
        GroupMemoryBarrierWithGroupSync();

        uint batchCount = min(RASTER_BATCH, end - batchStart);
        for (uint localIdx = 0u; localIdx < batchCount; localIdx++)
        {
            if (pixelDone) continue;
            processedCountOut += 1u;

            float alpha = raster_eval_alpha(localIdx, rayDir);
            if (alpha <= 0.0) continue;

            state = raster_blend_step(state, localIdx, alpha);
            if (state.trans < g_Raster.transmittanceThreshold)
            {
                pixelDone = true;
                InterlockedAdd(gsStoppedPixels, 1u);
            }
        }

        GroupMemoryBarrierWithGroupSync();
        if (gsStoppedPixels == RASTER_BATCH) break;
    }

    return state;
}

RasterPixelInfo resolve_raster_pixel(uint3 dispatchThreadID)
{
    RasterPixelInfo info;
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    info.pixelCoord = uint2(px, py);
    info.pixelActive = px < g_Raster.width && py < g_Raster.height;
    info.samplePx = min(px, g_Raster.width - 1u);
    info.samplePy = min(py, g_Raster.height - 1u);
    uint tileX = info.samplePx / g_Prepass.tileSize;
    uint tileY = info.samplePy / g_Prepass.tileSize;
    uint tileId = tileY * g_Prepass.tileWidth + tileX;
    uint2 range = g_TileRanges[tileId];
    info.hasTileRange = range.x != INVALID_START && range.y > range.x;
    info.start = info.hasTileRange ? range.x : 0u;
    info.end = info.hasTileRange ? min(range.y, info.start + g_Raster.maxSplatSteps) : info.start;
    PinholeCamera camera = PinholeCamera.from_params(g_Camera);
    info.rayDir = camera.screen_to_world_ray(float2(info.samplePx, info.samplePy) + 0.5);
    return info;
}

float4 raster_forward_output(
    RasterPixelInfo info,
    RasterState state,
    uint processedCount
)
{
    if (!info.pixelActive)
    {
        return float4(0.0, 0.0, 0.0, 1.0);
    }

    if (!info.hasTileRange)
    {
        float3 bgColor = (g_Raster.debugShowProcessedCount != 0u)
            ? float3(0.0)
            : pow(max(g_Raster.background, 0.0), OUTPUT_GAMMA);
        return float4(bgColor, 1.0);
    }

    if (g_Raster.debugShowProcessedCount != 0u)
    {
        float processed = float(processedCount);
        float logMax = log2(float(g_Raster.maxSplatSteps) + 1.0);
        float normalized = (logMax > 0.0) ? saturate(log2(processed + 1.0) / logMax) : 0.0;
        return float4(float3(normalized), 1.0);
    }

    float3 finalColor = state.accum + state.trans * g_Raster.background;
    if (g_Raster.debugShowEllipses != 0u)
    {
        float2 samplePoint = float2(info.pixelCoord) + 0.5;
        float maxEdge = 0.0;
        float3 edgeColor = g_Raster.debugEllipseColor;
        float thicknessPx = max(g_Raster.debugEllipseThicknessPx, MIN_DEBUG_THICKNESS_PX);
        for (uint idx = info.start; idx < info.end; idx++)
        {
            uint splatId = g_SortedValues[idx];
            float4 centerRadiusDepth = g_ScreenCenterRadiusDepth[splatId];
            float2 d = samplePoint - centerRadiusDepth.xy;
            float bound = centerRadiusDepth.z + thicknessPx;
            if (any(abs(d) > bound)) continue;

            float4 conic = g_ScreenEllipseConic[splatId];
            if (conic.w <= 0.0) continue;
            float2x2 conicMatrix = float2x2(conic.x, conic.y, conic.y, conic.z);
            float2 grad = mul(conicMatrix, d);
            float quad = dot(d, grad);
            float gradLen = max(length(2.0 * grad), GRADIENT_LENGTH_FLOOR);
            float edgeDistance = abs(quad - 1.0) / gradLen;
            float edge = saturate(1.0 - edgeDistance / thicknessPx);
            if (edge > maxEdge)
            {
                maxEdge = edge;
                edgeColor = g_ScreenColorAlpha[splatId].xyz;
            }
        }
        finalColor = lerp(finalColor, edgeColor, maxEdge);
    }

    float3 colorOut = pow(max(finalColor, 0.0), OUTPUT_GAMMA);
    return float4(colorOut, 1.0 - state.trans);
}

RasterState.Differential raster_output_gradient(
    RasterPixelInfo info,
    RasterState state
)
{
    RasterState.Differential dState;
    dState.accum = 0.0;
    dState.trans = 0.0;
    if (!info.pixelActive || !info.hasTileRange) return dState;

    var dpState = diffPair(state);
    bwd_diff(raster_output_from_state)(dpState, g_Raster.background, g_OutputGrad[info.pixelCoord]);
    return dpState.getDifferential();
}

void raster_backward_replay(
    RasterPixelInfo info,
    RasterState state,
    uint processedCount,
    RasterState.Differential dState,
    uint lane
)
{
    if (!info.hasTileRange)
    {
        GroupMemoryBarrierWithGroupSync();
        return;
    }

    uint processedEnd = info.start + processedCount;
    for (uint batchBase = info.end; batchBase > info.start;)
    {
        uint batchStart = (batchBase > info.start + RASTER_BATCH) ? (batchBase - RASTER_BATCH) : info.start;
        uint batchCount = batchBase - batchStart;

        prefetch_batch_from_global(batchStart, batchBase, lane);
        clear_batch_gradient_cache(batchCount, lane);
        GroupMemoryBarrierWithGroupSync();

        for (uint rev = 0u; rev < batchCount; rev++)
        {
            uint localIdx = batchCount - 1u - rev;
            uint globalIdx = batchStart + localIdx;
            if (!info.pixelActive || globalIdx >= processedEnd) continue;

            float alpha = raster_eval_alpha(localIdx, info.rayDir);
            if (alpha <= 0.0) continue;

            float4 colorAlpha = float4(gsColorAlpha[localIdx]);
            RasterState prevState = raster_undo_blend_step(state, colorAlpha, alpha);

            var dpPrevState = diffPair(prevState);
            var dpAlpha = diffPair(alpha);
            bwd_diff(raster_blend_step)(dpPrevState, localIdx, dpAlpha, dState);
            bwd_diff(raster_eval_alpha)(localIdx, info.rayDir, dpAlpha.getDifferential());

            state = prevState;
            dState = dpPrevState.getDifferential();
        }

        GroupMemoryBarrierWithGroupSync();
        flush_batch_gradient_cache(batchCount, lane);
        GroupMemoryBarrierWithGroupSync();
        batchBase = batchStart;
    }
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csClearRasterGrads(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint gradCount = g_Prepass.splatCount * SPLAT_GRAD_COMPONENTS;
    uint idx = dispatchThreadID.x;
    if (idx >= gradCount) return;

    g_GradPositions[idx] = 0.0;
    g_GradScales[idx] = 0.0;
    g_GradRotations[idx] = 0.0;
    g_GradColorAlpha[idx] = 0.0;
}

[shader("compute")]
[numthreads(RASTER_TILE_DIM, RASTER_TILE_DIM, 1)]
void csRasterize(uint3 dispatchThreadID : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    RasterPixelInfo info = resolve_raster_pixel(dispatchThreadID);
    RasterState state = { 0.0, 1.0 };
    uint processedCount = 0u;
    if (info.hasTileRange) state = raster_forward_replay(info.start, info.end, info.pixelActive, info.rayDir, lane, processedCount);
    if (info.pixelActive) g_Output[info.pixelCoord] = raster_forward_output(info, state, processedCount);
}

[shader("compute")]
[numthreads(RASTER_TILE_DIM, RASTER_TILE_DIM, 1)]
void csRasterizeForwardBackward(uint3 dispatchThreadID : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    if (g_Raster.debugShowProcessedCount != 0u || g_Raster.debugShowEllipses != 0u) return;
    RasterPixelInfo info = resolve_raster_pixel(dispatchThreadID);
    RasterState state = { 0.0, 1.0 };
    uint processedCount = 0u;
    if (info.hasTileRange) state = raster_forward_replay(info.start, info.end, info.pixelActive, info.rayDir, lane, processedCount);
    RasterState.Differential dState = raster_output_gradient(info, state);
    raster_backward_replay(info, state, processedCount, dState, lane);
}
