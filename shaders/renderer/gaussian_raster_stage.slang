import "gaussian_types.slang";
import "camera.slang";
import "math_constants.slang";
import "opacity.slang";

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

static const float DEBUG_GRAD_NORM_FLOOR = 1e-12;
static const float DEBUG_GRAD_ALPHA_SCALE = 0.1;
static const float LOG2_TO_LOG10 = 0.30102999566;
static const float DEBUG_GRAD_LOG10_MIN = -12.0;
static const float DEBUG_GRAD_LOG10_MAX = 0.0;

void atomic_add_grad<let N : int>(RWStructuredBuffer<float> gradBuffer, uint base, vector<float, N> grad)
{
    [unroll] for (int i = 0; i < N; i++)
    {
        float value = grad[i];
        if (value == 0.0) continue;
        InterlockedAdd(gradBuffer[base + uint(i)], value);
    }
}

void atomic_add_grad_wave<let N : int>(RWStructuredBuffer<float> gradBuffer, uint base, vector<float, N> grad)
{
    [unroll] for (int i = 0; i < N; i++)
    {
        float waveSum = WaveActiveSum(grad[i]);
        if (!WaveIsFirstLane() || waveSum == 0.0) continue;
        InterlockedAdd(gradBuffer[base + uint(i)], waveSum);
    }
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

    static void accumulate_gradients_global(uint localIdx, Differential dOut)
    {
        uint splatId = gsBatchSplatId[localIdx];
        uint splatBase = splatId * SPLAT_GRAD_COMPONENTS;

        CachedEllipsoid.Differential dCached;
        dCached.roLocal = dOut.roLocal;
        dCached.invScale = dOut.invScale;
        dCached.quat = dOut.quat;

        float3 position = g_Positions[splatId].xyz;
        float3 scale = g_Scales[splatId].xyz;
        float4 rotation = g_Rotations[splatId];
        var dpPosition = diffPair(position);
        var dpScale = diffPair(scale);
        var dpRotation = diffPair(rotation);
        bwd_diff(build_cached_ellipsoid)(dpPosition, dpScale, dpRotation, dCached);

        atomic_add_grad_wave<4>(g_GradPositions, splatBase, float4(dpPosition.getDifferential(), 0.0));
        atomic_add_grad_wave<4>(g_GradScales, splatBase, float4(dpScale.getDifferential(), 0.0));
        atomic_add_grad_wave<4>(g_GradRotations, splatBase, dpRotation.getDifferential());
        atomic_add_grad_wave<4>(g_GradColorAlpha, splatBase, dOut.colorAlpha);
    }

    static void raster_gaussian_load_shared_bwd(uint localIdx, Differential dOut)
    {
        accumulate_gradients_global(localIdx, dOut);
    }
};

[Differentiable]
float raster_eval_alpha_gaussian(RasterGaussian gaussian, no_diff float3 rayDir)
{
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
    float alpha = opacity_from_raw(gaussian.colorAlpha.w) * coverage;
    if (alpha < g_Raster.alphaCutoff) return 0.0;
    return alpha;
}

[Differentiable]
RasterState raster_blend_step_gaussian(RasterState state, RasterGaussian gaussian, float alpha)
{
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

struct RasterMicrotileInfo
{
    bool hasAnyPixelActive;
    bool hasTileRange;
    uint start;
    uint end;
    uint2 pixelCoord[RASTER_MICROTILE_PIXELS];
    bool pixelActive[RASTER_MICROTILE_PIXELS];
    float3 rayDir[RASTER_MICROTILE_PIXELS];
};

struct RasterMicrotileForwardState
{
    RasterState pixelState[RASTER_MICROTILE_PIXELS];
    uint processedCount[RASTER_MICROTILE_PIXELS];
    uint processedEnd[RASTER_MICROTILE_PIXELS];
};

struct RasterMicrotileBackwardState
{
    RasterState.Differential pixelState[RASTER_MICROTILE_PIXELS];
};

uint microtile_index(uint x, uint y)
{
    return y * RASTER_MICROTILE_DIM + x;
}

RasterState initial_raster_state()
{
    return { 0.0, 1.0 };
}

float jet_channel(float x)
{
    return saturate(1.5 - abs(4.0 * x - 3.0));
}

float3 jet_colormap(float x)
{
    return float3(jet_channel(x + 0.5), jet_channel(x), jet_channel(x - 0.5));
}

float raster_eval_alpha_gaussian_debug(RasterGaussian gaussian, no_diff float3 rayDir)
{
    return DEBUG_GRAD_ALPHA_SCALE * raster_eval_alpha_gaussian(gaussian, rayDir);
}

float raster_forward_debug_grad_pixel(RasterMicrotileInfo info, uint pixelIdx)
{
    float accum = 0.0;
    float trans = 1.0;
    for (uint idx = info.start; idx < info.end; idx++)
    {
        uint splatId = g_SortedValues[idx];
        RasterGaussian gaussian = RasterGaussian.load_global(splatId);
        float alpha = raster_eval_alpha_gaussian_debug(gaussian, info.rayDir[pixelIdx]);
        if (alpha <= 0.0) continue;
        float gradNorm = g_DebugGradNorm[splatId];
        if (!isfinite(gradNorm)) gradNorm = 0.0;
        accum += trans * alpha * max(gradNorm, 0.0);
        trans *= 1.0 - alpha;
        if (trans < g_Raster.transmittanceThreshold) break;
    }
    return accum;
}

RasterState.Differential zero_raster_state_differential()
{
    RasterState.Differential dState;
    dState.accum = 0.0;
    dState.trans = 0.0;
    return dState;
}

RasterGaussian.Differential zero_raster_gaussian_differential()
{
    RasterGaussian.Differential dGaussian;
    dGaussian.roLocal = 0.0;
    dGaussian.invScale = 0.0;
    dGaussian.quat = 0.0;
    dGaussian.colorAlpha = 0.0;
    return dGaussian;
}

void init_raster_microtile_forward_state(RasterMicrotileInfo info, out RasterMicrotileForwardState forwardState)
{
    [unroll] for (uint y = 0u; y < RASTER_MICROTILE_DIM; y++)
    {
        [unroll] for (uint x = 0u; x < RASTER_MICROTILE_DIM; x++)
        {
            uint pixelIdx = microtile_index(x, y);
            forwardState.pixelState[pixelIdx] = initial_raster_state();
            forwardState.processedCount[pixelIdx] = 0u;
            forwardState.processedEnd[pixelIdx] = info.start;
        }
    }
}

void init_raster_microtile_backward_state(out RasterMicrotileBackwardState backwardState)
{
    [unroll] for (uint y = 0u; y < RASTER_MICROTILE_DIM; y++)
    {
        [unroll] for (uint x = 0u; x < RASTER_MICROTILE_DIM; x++)
        {
            backwardState.pixelState[microtile_index(x, y)] = zero_raster_state_differential();
        }
    }
}

RasterMicrotileInfo resolve_raster_microtile(uint3 dispatchThreadID)
{
    RasterMicrotileInfo info;
    info.hasAnyPixelActive = false;
    info.hasTileRange = false;
    info.start = 0u;
    info.end = 0u;

    PinholeCamera camera = PinholeCamera.from_params(g_Camera);
    uint2 basePixel = uint2(
        dispatchThreadID.x * RASTER_MICROTILE_DIM,
        dispatchThreadID.y * RASTER_MICROTILE_DIM
    );

    [unroll] for (uint y = 0u; y < RASTER_MICROTILE_DIM; y++)
    {
        [unroll] for (uint x = 0u; x < RASTER_MICROTILE_DIM; x++)
        {
            uint pixelIdx = microtile_index(x, y);
            uint px = basePixel.x + x;
            uint py = basePixel.y + y;
            info.pixelCoord[pixelIdx] = uint2(px, py);
            info.pixelActive[pixelIdx] = px < g_Raster.width && py < g_Raster.height;
            info.hasAnyPixelActive = info.hasAnyPixelActive || info.pixelActive[pixelIdx];

            uint samplePx = min(px, g_Raster.width - 1u);
            uint samplePy = min(py, g_Raster.height - 1u);
            info.rayDir[pixelIdx] = camera.screen_to_world_ray(float2(samplePx, samplePy) + 0.5);
        }
    }

    uint sampleBaseX = min(basePixel.x, g_Raster.width - 1u);
    uint sampleBaseY = min(basePixel.y, g_Raster.height - 1u);
    uint tileX = sampleBaseX / g_Prepass.tileSize;
    uint tileY = sampleBaseY / g_Prepass.tileSize;
    uint tileId = tileY * g_Prepass.tileWidth + tileX;
    uint2 range = g_TileRanges[tileId];
    info.hasTileRange = range.x != INVALID_START && range.y > range.x;
    info.start = info.hasTileRange ? range.x : 0u;
    info.end = info.hasTileRange ? min(range.y, info.start + g_Raster.maxSplatSteps) : info.start;
    return info;
}

void prefetch_batch_from_global(uint batchStart, uint batchEnd, uint lane)
{
    uint batchCount = batchEnd - batchStart;
    for (uint localIdx = lane; localIdx < batchCount; localIdx += RASTER_GROUP_THREADS)
    {
        uint splatId = g_SortedValues[batchStart + localIdx];
        gsBatchSplatId[localIdx] = splatId;
        RasterGaussian gaussian = RasterGaussian.load_global(splatId);
        gaussian.store_shared(localIdx);
    }
}

void raster_forward_replay(
    RasterMicrotileInfo info,
    uint lane,
    inout RasterMicrotileForwardState forwardState
)
{
    bool pixelDone[RASTER_MICROTILE_PIXELS];
    uint activePixelCount = 0u;
    uint finishedPixelCount = 0u;

    [unroll] for (uint y = 0u; y < RASTER_MICROTILE_DIM; y++)
    {
        [unroll] for (uint x = 0u; x < RASTER_MICROTILE_DIM; x++)
        {
            uint pixelIdx = microtile_index(x, y);
            pixelDone[pixelIdx] = !info.pixelActive[pixelIdx];
            if (info.pixelActive[pixelIdx]) activePixelCount += 1u;
        }
    }

    bool microtileDone = activePixelCount == 0u;
    if (lane == 0u) gsStoppedMicrotiles = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (microtileDone) InterlockedAdd(gsStoppedMicrotiles, 1u);

    for (uint batchStart = info.start; batchStart < info.end; batchStart += RASTER_BATCH)
    {
        uint batchEnd = min(batchStart + RASTER_BATCH, info.end);
        uint batchCount = batchEnd - batchStart;
        prefetch_batch_from_global(batchStart, batchEnd, lane);
        GroupMemoryBarrierWithGroupSync();

        if (!microtileDone)
        {
            for (uint localIdx = 0u; localIdx < batchCount; localIdx++)
            {
                RasterGaussian gaussian = RasterGaussian.load_shared(localIdx);

                [unroll] for (uint y = 0u; y < RASTER_MICROTILE_DIM; y++)
                {
                    [unroll] for (uint x = 0u; x < RASTER_MICROTILE_DIM; x++)
                    {
                        uint pixelIdx = microtile_index(x, y);
                        if (pixelDone[pixelIdx]) continue;

                        forwardState.processedCount[pixelIdx] += 1u;
                        float alpha = raster_eval_alpha_gaussian(gaussian, info.rayDir[pixelIdx]);
                        if (alpha <= 0.0) continue;

                        RasterState state = forwardState.pixelState[pixelIdx];
                        state = raster_blend_step_gaussian(state, gaussian, alpha);
                        forwardState.pixelState[pixelIdx] = state;
                        if (state.trans >= g_Raster.transmittanceThreshold) continue;

                        pixelDone[pixelIdx] = true;
                        finishedPixelCount += 1u;
                    }
                }

                if (finishedPixelCount == activePixelCount)
                {
                    microtileDone = true;
                    InterlockedAdd(gsStoppedMicrotiles, 1u);
                    break;
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
        if (gsStoppedMicrotiles == RASTER_GROUP_THREADS) break;
    }

    [unroll] for (uint y = 0u; y < RASTER_MICROTILE_DIM; y++)
    {
        [unroll] for (uint x = 0u; x < RASTER_MICROTILE_DIM; x++)
        {
            uint pixelIdx = microtile_index(x, y);
            forwardState.processedEnd[pixelIdx] = info.start + forwardState.processedCount[pixelIdx];
        }
    }
}

float4 raster_forward_output_pixel(
    RasterMicrotileInfo info,
    uint pixelIdx,
    RasterState state,
    uint processedCount
)
{
    if (!info.pixelActive[pixelIdx]) return float4(0.0, 0.0, 0.0, 1.0);

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

    if (g_Raster.debugShowGradNorm != 0u)
    {
        float gradValue = max(raster_forward_debug_grad_pixel(info, pixelIdx), DEBUG_GRAD_NORM_FLOOR);
        float gradLog10 = log2(gradValue) * LOG2_TO_LOG10;
        float normalized = saturate((gradLog10 - DEBUG_GRAD_LOG10_MIN) / (DEBUG_GRAD_LOG10_MAX - DEBUG_GRAD_LOG10_MIN));
        return float4(jet_colormap(normalized), 1.0);
    }

    float3 finalColor = state.accum + state.trans * g_Raster.background;
    if (g_Raster.debugShowEllipses != 0u)
    {
        float2 samplePoint = float2(info.pixelCoord[pixelIdx]) + 0.5;
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
            if (edge <= maxEdge) continue;

            maxEdge = edge;
            edgeColor = g_ScreenColorAlpha[splatId].xyz;
        }
        finalColor = lerp(finalColor, edgeColor, maxEdge);
    }

    float3 colorOut = pow(max(finalColor, 0.0), OUTPUT_GAMMA);
    return float4(colorOut, 1.0 - state.trans);
}

RasterState.Differential raster_output_gradient_pixel(
    RasterMicrotileInfo info,
    uint pixelIdx,
    RasterState state
)
{
    if (!info.pixelActive[pixelIdx] || !info.hasTileRange) return zero_raster_state_differential();

    var dpState = diffPair(state);
    bwd_diff(raster_output_from_state)(dpState, g_Raster.background, g_OutputGrad[info.pixelCoord[pixelIdx]]);
    return dpState.getDifferential();
}

void raster_backward_replay(
    RasterMicrotileInfo info,
    inout RasterMicrotileForwardState forwardState,
    inout RasterMicrotileBackwardState backwardState,
    uint lane
)
{
    if (!info.hasTileRange)
    {
        GroupMemoryBarrierWithGroupSync();
        return;
    }

    for (uint batchBase = info.end; batchBase > info.start;)
    {
        uint batchStart = (batchBase > info.start + RASTER_BATCH) ? (batchBase - RASTER_BATCH) : info.start;
        uint batchCount = batchBase - batchStart;

        prefetch_batch_from_global(batchStart, batchBase, lane);
        GroupMemoryBarrierWithGroupSync();

        for (uint rev = 0u; rev < batchCount; rev++)
        {
            uint localIdx = batchCount - 1u - rev;
            uint globalIdx = batchStart + localIdx;
            RasterGaussian gaussian = RasterGaussian.load_shared(localIdx);
            RasterGaussian.Differential dGaussian = zero_raster_gaussian_differential();

            [unroll] for (uint y = 0u; y < RASTER_MICROTILE_DIM; y++)
            {
                [unroll] for (uint x = 0u; x < RASTER_MICROTILE_DIM; x++)
                {
                    uint pixelIdx = microtile_index(x, y);
                    if (!info.pixelActive[pixelIdx] || globalIdx >= forwardState.processedEnd[pixelIdx]) continue;

                    float alpha = raster_eval_alpha_gaussian(gaussian, info.rayDir[pixelIdx]);
                    if (alpha <= 0.0) continue;

                    RasterState state = forwardState.pixelState[pixelIdx];
                    RasterState prevState = raster_undo_blend_step(state, gaussian.colorAlpha, alpha);

                    var dpPrevState = diffPair(prevState);
                    var dpGaussian = diffPair(gaussian);
                    var dpAlpha = diffPair(alpha);
                    bwd_diff(raster_blend_step_gaussian)(dpPrevState, dpGaussian, dpAlpha, backwardState.pixelState[pixelIdx]);
                    bwd_diff(raster_eval_alpha_gaussian)(dpGaussian, info.rayDir[pixelIdx], dpAlpha.getDifferential());

                    RasterGaussian.Differential dPixelGaussian = dpGaussian.getDifferential();
                    dGaussian.roLocal += dPixelGaussian.roLocal;
                    dGaussian.invScale += dPixelGaussian.invScale;
                    dGaussian.quat += dPixelGaussian.quat;
                    dGaussian.colorAlpha += dPixelGaussian.colorAlpha;
                    forwardState.pixelState[pixelIdx] = prevState;
                    backwardState.pixelState[pixelIdx] = dpPrevState.getDifferential();
                }
            }

            RasterGaussian.accumulate_gradients_global(localIdx, dGaussian);
        }

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
[numthreads(RASTER_THREAD_TILE_DIM, RASTER_THREAD_TILE_DIM, 1)]
void csRasterize(uint3 dispatchThreadID : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    RasterMicrotileInfo info = resolve_raster_microtile(dispatchThreadID);
    RasterMicrotileForwardState forwardState;
    init_raster_microtile_forward_state(info, forwardState);
    if (info.hasTileRange) raster_forward_replay(info, lane, forwardState);

    [unroll] for (uint y = 0u; y < RASTER_MICROTILE_DIM; y++)
    {
        [unroll] for (uint x = 0u; x < RASTER_MICROTILE_DIM; x++)
        {
            uint pixelIdx = microtile_index(x, y);
            if (!info.pixelActive[pixelIdx]) continue;
            g_Output[info.pixelCoord[pixelIdx]] = raster_forward_output_pixel(
                info,
                pixelIdx,
                forwardState.pixelState[pixelIdx],
                forwardState.processedCount[pixelIdx]
            );
        }
    }
}

[shader("compute")]
[numthreads(RASTER_THREAD_TILE_DIM, RASTER_THREAD_TILE_DIM, 1)]
void csRasterizeForwardBackward(uint3 dispatchThreadID : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    if (g_Raster.debugShowProcessedCount != 0u || g_Raster.debugShowEllipses != 0u || g_Raster.debugShowGradNorm != 0u) return;

    RasterMicrotileInfo info = resolve_raster_microtile(dispatchThreadID);
    RasterMicrotileForwardState forwardState;
    RasterMicrotileBackwardState backwardState;
    init_raster_microtile_forward_state(info, forwardState);
    init_raster_microtile_backward_state(backwardState);
    if (info.hasTileRange) raster_forward_replay(info, lane, forwardState);

    [unroll] for (uint y = 0u; y < RASTER_MICROTILE_DIM; y++)
    {
        [unroll] for (uint x = 0u; x < RASTER_MICROTILE_DIM; x++)
        {
            uint pixelIdx = microtile_index(x, y);
            backwardState.pixelState[pixelIdx] = raster_output_gradient_pixel(
                info,
                pixelIdx,
                forwardState.pixelState[pixelIdx]
            );
        }
    }

    raster_backward_replay(info, forwardState, backwardState, lane);
}
