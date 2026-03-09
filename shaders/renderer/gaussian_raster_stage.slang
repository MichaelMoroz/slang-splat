import "gaussian_types.slang";
import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/splatting/common.slang";
import "utility/splatting/opacity.slang";
import "utility/splatting/params.slang";
import "utility/splatting/raster.slang";

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

void atomic_add_param_grad_wave<let N : int>(uint splatCount, uint firstParamId, uint splatId, vector<float, N> grad)
{
    [unroll] for (int i = 0; i < N; i++)
    {
        float waveSum = WaveActiveSum(grad[i]);
        if (!WaveIsFirstLane() || waveSum == 0.0) continue;
        InterlockedAdd(g_ParamGrads[splat_param_index(splatCount, firstParamId + uint(i), splatId)], waveSum);
    }
}

struct RasterGaussian : IDifferentiable
{
    float3 roLocal;
    float3 invScale;
    float4 quat;
    float opacityScale;
    float4 colorAlpha;

    static RasterGaussian load_global(uint splatId)
    {
        Gaussian3D gaussian = load_gaussian3d(splatId);
        CachedEllipsoid cached = build_cached_ellipsoid(g_Camera, g_Prepass.radiusScale, gaussian.position, gaussian.scale, gaussian.rotation);
        return { cached.roLocal, cached.invScale, cached.quat, cached.opacityScale, gaussian.colorAlpha };
    }

    [BackwardDerivative(raster_gaussian_load_shared_bwd)]
    static RasterGaussian load_shared(uint localIdx)
    {
        RasterGaussian gaussian;
        gaussian.roLocal = gsRoLocal[localIdx].xyz;
        gaussian.invScale = float3(gsInvScale[localIdx].xyz);
        gaussian.quat = float4(gsQuat[localIdx]);
        gaussian.opacityScale = float(gsInvScale[localIdx].w);
        gaussian.colorAlpha = float4(gsColorAlpha[localIdx]);
        return gaussian;
    }

    [mutating]
    void store_shared(uint localIdx)
    {
        gsRoLocal[localIdx] = float4(roLocal, 0.0);
        gsInvScale[localIdx] = half4(float4(invScale, opacityScale));
        gsQuat[localIdx] = half4(quat);
        gsColorAlpha[localIdx] = half4(colorAlpha);
    }

    static void accumulate_gradients_global(uint localIdx, Differential dOut)
    {
        uint splatId = gsBatchSplatId[localIdx];
        uint splatCount = g_Prepass.splatCount;

        CachedEllipsoid.Differential dCached;
        dCached.roLocal = dOut.roLocal;
        dCached.invScale = dOut.invScale;
        dCached.quat = dOut.quat;
        dCached.opacityScale = dOut.opacityScale;

        Gaussian3D gaussian = load_gaussian3d(splatId);
        var dpPosition = diffPair(gaussian.position);
        var dpScale = diffPair(gaussian.scale);
        var dpRotation = diffPair(gaussian.rotation);
        bwd_diff(build_cached_ellipsoid)(g_Camera, g_Prepass.radiusScale, dpPosition, dpScale, dpRotation, dCached);

        atomic_add_param_grad_wave<3>(splatCount, PARAM_POSITION_X, splatId, dpPosition.getDifferential());
        atomic_add_param_grad_wave<3>(splatCount, PARAM_SCALE_X, splatId, dpScale.getDifferential());
        atomic_add_param_grad_wave<4>(splatCount, PARAM_ROTATION_X, splatId, dpRotation.getDifferential());
        atomic_add_param_grad_wave<4>(splatCount, PARAM_COLOR_R, splatId, dOut.colorAlpha);
    }

    static void raster_gaussian_load_shared_bwd(uint localIdx, Differential dOut)
    {
        accumulate_gradients_global(localIdx, dOut);
    }
};

[Differentiable]
float raster_eval_alpha_gaussian(RasterGaussian gaussian, no_diff uint splatId, no_diff float3 rayDir, no_diff float2 samplePoint)
{
    if (g_Raster.debugShowEllipses != 0u && g_Raster.debugShowProcessedCount == 0u && g_Raster.debugShowGradNorm == 0u)
    {
        float4 centerRadiusDepth = g_ScreenCenterRadiusDepth[splatId];
        float4 conicAndOpacity = g_ScreenEllipseConic[splatId];
        float2 delta = samplePoint - centerRadiusDepth.xy;
        float2 grad = mul(float2x2(conicAndOpacity.x, conicAndOpacity.y, conicAndOpacity.y, conicAndOpacity.z), delta);
        float quad = dot(delta, grad);
        float gradLen = max(length(2.0 * grad), GRADIENT_LENGTH_FLOOR);
        float fullWidth = max(g_Raster.debugEllipseThicknessPx, 2.0 * MIN_DEBUG_THICKNESS_PX);
        float halfWidth = 0.5 * fullWidth;
        float edgeDistance = abs((quad - 1.0) / gradLen);
        float coverage = saturate(1.0 - edgeDistance / halfWidth);
        coverage = coverage * coverage * (3.0 - 2.0 * coverage);
        float baseAlpha = opacity_from_raw(gaussian.colorAlpha.w) * coverage;
        return baseAlpha >= g_Raster.alphaCutoff ? baseAlpha * gaussian.opacityScale : 0.0;
    }
    return raster_eval_alpha(gaussian.roLocal, gaussian.quat, gaussian.invScale, gaussian.colorAlpha.w, gaussian.opacityScale, rayDir, g_Raster.alphaCutoff);
}

[Differentiable]
RasterState raster_blend_step_gaussian(RasterState state, RasterGaussian gaussian, float alpha)
{
    return state.blend(gaussian.colorAlpha.xyz, alpha);
}

float3 jet_colormap(float x)
{
    float t = saturate(x);
    return float3(
        saturate(1.5 - abs(4.0 * t - 3.0)),
        saturate(1.5 - abs(4.0 * t - 2.0)),
        saturate(1.5 - abs(4.0 * t - 1.0))
    );
}

float raster_normalize_grad_norm(float gradNorm)
{
    float gradValue = max(gradNorm, DEBUG_GRAD_NORM_FLOOR);
    float gradLog10 = log2(gradValue) * LOG2_TO_LOG10;
    float gradThreshold = max(g_Raster.debugGradNormThreshold, DEBUG_GRAD_NORM_FLOOR);
    float gradLog10Min = log2(max(gradThreshold * DEBUG_GRAD_THRESHOLD_MIN_SCALE, DEBUG_GRAD_NORM_FLOOR)) * LOG2_TO_LOG10;
    float gradLog10Max = log2(max(gradThreshold * DEBUG_GRAD_THRESHOLD_MAX_SCALE, DEBUG_GRAD_NORM_FLOOR)) * LOG2_TO_LOG10;
    return saturate((gradLog10 - gradLog10Min) / max(gradLog10Max - gradLog10Min, TINY_VALUE));
}

float3 raster_forward_color(RasterGaussian gaussian, uint splatId)
{
    if (g_Raster.debugShowProcessedCount != 0u) return 0.0;
    if (g_Raster.debugShowGradNorm == 0u) return gaussian.colorAlpha.xyz;
    float gradNorm = g_DebugGradNorm[splatId];
    if (!isfinite(gradNorm)) gradNorm = 0.0;
    return jet_colormap(raster_normalize_grad_norm(max(gradNorm, 0.0)));
}

float3 raster_processed_count_color(uint processedCount)
{
    float logMax = log2(float(g_Raster.maxSplatSteps) + 1.0);
    float normalized = (logMax > 0.0) ? saturate(log2(float(processedCount) + 1.0) / logMax) : 0.0;
    return jet_colormap(normalized);
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
    float2 samplePoint[RASTER_MICROTILE_PIXELS];
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

void init_raster_microtile_forward_state(RasterMicrotileInfo info, out RasterMicrotileForwardState forwardState)
{
    [unroll] for (uint pix = 0u; pix < RASTER_MICROTILE_PIXELS; pix++)
    {
        forwardState.pixelState[pix] = { 0.0, 1.0 };
        forwardState.processedCount[pix] = 0u;
        forwardState.processedEnd[pix] = info.start;
    }
}

void init_raster_microtile_backward_state(out RasterMicrotileBackwardState backwardState)
{
    [unroll] for (uint pix = 0u; pix < RASTER_MICROTILE_PIXELS; pix++)
    {
        backwardState.pixelState[pix] = { 0.0, 0.0 };
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

    [unroll] for (uint pix = 0u; pix < RASTER_MICROTILE_PIXELS; pix++)
    {
        uint y = pix / RASTER_MICROTILE_DIM;
        uint x = pix - y * RASTER_MICROTILE_DIM;
        uint px = basePixel.x + x;
        uint py = basePixel.y + y;
        info.pixelCoord[pix] = uint2(px, py);
        info.pixelActive[pix] = px < g_Raster.width && py < g_Raster.height;
        info.hasAnyPixelActive = info.hasAnyPixelActive || info.pixelActive[pix];
        info.samplePoint[pix] = float2(px, py) + 0.5;

        uint samplePx = min(px, g_Raster.width - 1u);
        uint samplePy = min(py, g_Raster.height - 1u);
        info.rayDir[pix] = camera.screen_to_world_ray(float2(samplePx, samplePy) + 0.5);
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
    uint replayCount[RASTER_MICROTILE_PIXELS];
    uint activePixelCount = 0u;
    uint finishedPixelCount = 0u;

    [unroll] for (uint pix = 0u; pix < RASTER_MICROTILE_PIXELS; pix++)
    {
        pixelDone[pix] = !info.pixelActive[pix];
        replayCount[pix] = 0u;
        if (info.pixelActive[pix]) activePixelCount += 1u;
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
                uint splatId = gsBatchSplatId[localIdx];
                float3 color = raster_forward_color(gaussian, splatId);

                [unroll] for (uint pix = 0u; pix < RASTER_MICROTILE_PIXELS; pix++)
                {
                    if (pixelDone[pix]) continue;

                    replayCount[pix] += 1u;
                    float alpha = raster_eval_alpha_gaussian(gaussian, splatId, info.rayDir[pix], info.samplePoint[pix]);
                    if (alpha <= 0.0) continue;

                    RasterState state = forwardState.pixelState[pix];
                    state = state.blend(color, alpha);
                    forwardState.pixelState[pix] = state;
                    forwardState.processedCount[pix] += 1u;
                    if (state.trans >= g_Raster.transmittanceThreshold) continue;

                    pixelDone[pix] = true;
                    finishedPixelCount += 1u;
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

    [unroll] for (uint pix = 0u; pix < RASTER_MICROTILE_PIXELS; pix++)
    {
        forwardState.processedEnd[pix] = info.start + replayCount[pix];
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
    if (g_Raster.debugShowProcessedCount != 0u) return float4(raster_processed_count_color(processedCount), 1.0);
    return info.hasTileRange
        ? raster_output_from_state(state, g_Raster.background)
        : float4(pow(max(g_Raster.background, 0.0), OUTPUT_GAMMA), 1.0);
}

RasterState.Differential raster_output_gradient_pixel(
    RasterMicrotileInfo info,
    uint pixelIdx,
    RasterState state
)
{
    if (!info.pixelActive[pixelIdx] || !info.hasTileRange) return { 0.0, 0.0 };

    var dpState = diffPair(state);
    bwd_diff(raster_output_from_state)(dpState, g_Raster.background, g_OutputGrad[info.pixelCoord[pixelIdx].y * g_Raster.width + info.pixelCoord[pixelIdx].x]);
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
            RasterGaussian.Differential dGaussian = { 0.0, 0.0, 0.0, 0.0, 0.0 };

            [unroll] for (uint pix = 0u; pix < RASTER_MICROTILE_PIXELS; pix++)
            {
                if (!info.pixelActive[pix] || globalIdx >= forwardState.processedEnd[pix]) continue;

                float alpha = raster_eval_alpha_gaussian(gaussian, gsBatchSplatId[localIdx], info.rayDir[pix], info.samplePoint[pix]);
                if (alpha <= 0.0) continue;

                RasterState state = forwardState.pixelState[pix];
                RasterState prevState = state.undo(gaussian.colorAlpha.xyz, alpha);

                var dpPrevState = diffPair(prevState);
                var dpGaussian = diffPair(gaussian);
                var dpAlpha = diffPair(alpha);
                bwd_diff(raster_blend_step_gaussian)(dpPrevState, dpGaussian, dpAlpha, backwardState.pixelState[pix]);
                bwd_diff(raster_eval_alpha_gaussian)(dpGaussian, gsBatchSplatId[localIdx], info.rayDir[pix], info.samplePoint[pix], dpAlpha.getDifferential());

                RasterGaussian.Differential dPixelGaussian = dpGaussian.getDifferential();
                dGaussian.roLocal += dPixelGaussian.roLocal;
                dGaussian.invScale += dPixelGaussian.invScale;
                dGaussian.quat += dPixelGaussian.quat;
                dGaussian.opacityScale += dPixelGaussian.opacityScale;
                dGaussian.colorAlpha += dPixelGaussian.colorAlpha;
                forwardState.pixelState[pix] = prevState;
                backwardState.pixelState[pix] = dpPrevState.getDifferential();
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
    uint gradCount = g_Prepass.splatCount * TRAINABLE_PARAM_COUNT;
    uint idx = dispatchThreadID.x;
    if (idx >= gradCount) return;

    g_ParamGrads[idx] = 0.0;
}

[shader("compute")]
[numthreads(RASTER_THREAD_TILE_DIM, RASTER_THREAD_TILE_DIM, 1)]
void csRasterize(uint3 dispatchThreadID : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    RasterMicrotileInfo info = resolve_raster_microtile(dispatchThreadID);
    RasterMicrotileForwardState forwardState;
    init_raster_microtile_forward_state(info, forwardState);
    if (info.hasTileRange) raster_forward_replay(info, lane, forwardState);

    [unroll] for (uint pix = 0u; pix < RASTER_MICROTILE_PIXELS; pix++)
    {
        if (!info.pixelActive[pix]) continue;
        g_Output[info.pixelCoord[pix]] = raster_forward_output_pixel(
            info,
            pix,
            forwardState.pixelState[pix],
            forwardState.processedCount[pix]
        );
    }
}

[shader("compute")]
[numthreads(RASTER_THREAD_TILE_DIM, RASTER_THREAD_TILE_DIM, 1)]
void csRasterizeForwardBackward(uint3 dispatchThreadID : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    RasterMicrotileInfo info = resolve_raster_microtile(dispatchThreadID);
    RasterMicrotileForwardState forwardState;
    RasterMicrotileBackwardState backwardState;
    init_raster_microtile_forward_state(info, forwardState);
    init_raster_microtile_backward_state(backwardState);
    if (info.hasTileRange) raster_forward_replay(info, lane, forwardState);

    [unroll] for (uint pix = 0u; pix < RASTER_MICROTILE_PIXELS; pix++)
    {
        backwardState.pixelState[pix] = raster_output_gradient_pixel(
            info,
            pix,
            forwardState.pixelState[pix]
        );
    }

    raster_backward_replay(info, forwardState, backwardState, lane);
}
