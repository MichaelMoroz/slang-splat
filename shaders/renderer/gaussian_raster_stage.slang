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

static const float RASTER_GRAD_FIXED_SCALE = float(1u << RASTER_GRAD_FIXED_FRAC_BITS);
static const float RASTER_GRAD_FIXED_INV_SCALE = 1.0 / RASTER_GRAD_FIXED_SCALE;
static const float RASTER_GRAD_FIXED_INT_MIN_F = -2147483648.0;
static const float RASTER_GRAD_FIXED_INT_MAX_F = 2147483647.0;

int raster_grad_fixed_encode(float value)
{
    float scaled = round(value * RASTER_GRAD_FIXED_SCALE);
    if (scaled <= RASTER_GRAD_FIXED_INT_MIN_F) return -2147483648;
    if (scaled >= RASTER_GRAD_FIXED_INT_MAX_F) return 2147483647;
    return int(scaled);
}

float raster_grad_fixed_decode(int value)
{
    return float(value) * RASTER_GRAD_FIXED_INV_SCALE;
}

int3 raster_grad_fixed_encode3(float3 value)
{
    return int3(
        raster_grad_fixed_encode(value.x),
        raster_grad_fixed_encode(value.y),
        raster_grad_fixed_encode(value.z)
    );
}

int4 raster_grad_fixed_encode4(float4 value)
{
    return int4(
        raster_grad_fixed_encode(value.x),
        raster_grad_fixed_encode(value.y),
        raster_grad_fixed_encode(value.z),
        raster_grad_fixed_encode(value.w)
    );
}

float3 raster_grad_fixed_decode3(int3 value)
{
    return float3(
        raster_grad_fixed_decode(value.x),
        raster_grad_fixed_decode(value.y),
        raster_grad_fixed_decode(value.z)
    );
}

float4 raster_grad_fixed_decode4(int4 value)
{
    return float4(
        raster_grad_fixed_decode(value.x),
        raster_grad_fixed_decode(value.y),
        raster_grad_fixed_decode(value.z),
        raster_grad_fixed_decode(value.w)
    );
}

void atomic_add_param_grad<let N : int>(uint splatCount, uint firstParamId, uint splatId, vector<float, N> grad)
{
    [unroll] for (int i = 0; i < N; i++)
    {
        int value = raster_grad_fixed_encode(grad[i]);
        if (value == 0) continue;
        InterlockedAdd(g_ParamGradsFixed[splat_param_index(splatCount, firstParamId + uint(i), splatId)], value);
    }
}

uint grad_state_base(uint localIdx, uint fieldOffset)
{
    return localIdx * SPLAT_GRAD_STATE_STRIDE + fieldOffset;
}

void atomic_add_grad_shared(uint base, float3 grad)
{
    int3 fixedWaveSum = raster_grad_fixed_encode3(float3(WaveActiveSum(grad.x), WaveActiveSum(grad.y), WaveActiveSum(grad.z)));
    if (!WaveIsFirstLane()) return;
    if (fixedWaveSum.x != 0) InterlockedAdd(gsGradState[base + 0u], fixedWaveSum.x);
    if (fixedWaveSum.y != 0) InterlockedAdd(gsGradState[base + 1u], fixedWaveSum.y);
    if (fixedWaveSum.z != 0) InterlockedAdd(gsGradState[base + 2u], fixedWaveSum.z);
}

void atomic_add_grad_shared(uint base, float4 grad)
{
    int4 fixedWaveSum = raster_grad_fixed_encode4(float4(WaveActiveSum(grad.x), WaveActiveSum(grad.y), WaveActiveSum(grad.z), WaveActiveSum(grad.w)));
    if (!WaveIsFirstLane()) return;
    if (fixedWaveSum.x != 0) InterlockedAdd(gsGradState[base + 0u], fixedWaveSum.x);
    if (fixedWaveSum.y != 0) InterlockedAdd(gsGradState[base + 1u], fixedWaveSum.y);
    if (fixedWaveSum.z != 0) InterlockedAdd(gsGradState[base + 2u], fixedWaveSum.z);
    if (fixedWaveSum.w != 0) InterlockedAdd(gsGradState[base + 3u], fixedWaveSum.w);
}

float3 load_grad_shared3(uint base)
{
    return raster_grad_fixed_decode3(int3(gsGradState[base + 0u], gsGradState[base + 1u], gsGradState[base + 2u]));
}

float4 load_grad_shared4(uint base)
{
    return raster_grad_fixed_decode4(int4(gsGradState[base + 0u], gsGradState[base + 1u], gsGradState[base + 2u], gsGradState[base + 3u]));
}

void clear_batch_gradient_cache(uint batchCount, uint lane)
{
    for (uint localIdx = lane; localIdx < batchCount; localIdx += RASTER_GROUP_THREADS)
    {
        uint base = grad_state_base(localIdx, 0u);
        [unroll] for (uint i = 0u; i < SPLAT_GRAD_STATE_STRIDE; i++) gsGradState[base + i] = 0;
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
        gaussian.invScale = float3(gsInvScale[localIdx]);
        gaussian.quat = float4(gsQuat[localIdx]);
        gaussian.opacityScale = gsRoLocal[localIdx].w;
        gaussian.colorAlpha = float4(gsColorAlpha[localIdx]);
        return gaussian;
    }

    [mutating]
    void store_shared(uint localIdx)
    {
        gsRoLocal[localIdx] = float4(roLocal, opacityScale);
        gsInvScale[localIdx] = half3(invScale);
        gsQuat[localIdx] = half4(quat);
        gsColorAlpha[localIdx] = half4(colorAlpha);
    }

    static void accumulate_gradients_shared(uint localIdx, Differential dOut)
    {
        atomic_add_grad_shared(grad_state_base(localIdx, SPLAT_GRAD_STATE_RO_LOCAL_OPACITY_OFFSET), float4(dOut.roLocal, dOut.opacityScale));
        atomic_add_grad_shared(grad_state_base(localIdx, SPLAT_GRAD_STATE_INV_SCALE_OFFSET), dOut.invScale);
        atomic_add_grad_shared(grad_state_base(localIdx, SPLAT_GRAD_STATE_QUAT_OFFSET), dOut.quat);
        atomic_add_grad_shared(grad_state_base(localIdx, SPLAT_GRAD_STATE_COLOR_ALPHA_OFFSET), dOut.colorAlpha);
    }

    static void raster_gaussian_load_shared_bwd(uint localIdx, Differential dOut)
    {
        accumulate_gradients_shared(localIdx, dOut);
    }
};

void flush_batch_gradient_cache(uint batchCount, uint lane)
{
    for (uint localIdx = lane; localIdx < batchCount; localIdx += RASTER_GROUP_THREADS)
    {
        uint splatId = gsBatchSplatId[localIdx];
        uint splatCount = g_Prepass.splatCount;

        CachedEllipsoid.Differential dCached;
        float4 dRoLocalOpacity = load_grad_shared4(grad_state_base(localIdx, SPLAT_GRAD_STATE_RO_LOCAL_OPACITY_OFFSET));
        dCached.roLocal = dRoLocalOpacity.xyz;
        dCached.opacityScale = dRoLocalOpacity.w;
        dCached.invScale = load_grad_shared3(grad_state_base(localIdx, SPLAT_GRAD_STATE_INV_SCALE_OFFSET));
        dCached.quat = load_grad_shared4(grad_state_base(localIdx, SPLAT_GRAD_STATE_QUAT_OFFSET));
        float4 dColorAlpha = load_grad_shared4(grad_state_base(localIdx, SPLAT_GRAD_STATE_COLOR_ALPHA_OFFSET));

        Gaussian3D gaussian = load_gaussian3d(splatId);
        var dpPosition = diffPair(gaussian.position);
        var dpScale = diffPair(gaussian.scale);
        var dpRotation = diffPair(gaussian.rotation);
        bwd_diff(build_cached_ellipsoid)(g_Camera, g_Prepass.radiusScale, dpPosition, dpScale, dpRotation, dCached);

        atomic_add_param_grad<3>(splatCount, PARAM_POSITION_X, splatId, dpPosition.getDifferential());
        atomic_add_param_grad<3>(splatCount, PARAM_SCALE_X, splatId, dpScale.getDifferential());
        atomic_add_param_grad<4>(splatCount, PARAM_ROTATION_X, splatId, dpRotation.getDifferential());
        atomic_add_param_grad<4>(splatCount, PARAM_COLOR_R, splatId, dColorAlpha);
    }
}

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

uint raster_pixel_index(uint2 pixelCoord)
{
    return pixelCoord.y * g_Raster.width + pixelCoord.x;
}

void store_training_forward_cache(RasterMicrotileInfo info, RasterMicrotileForwardState forwardState)
{
    [unroll] for (uint pix = 0u; pix < RASTER_MICROTILE_PIXELS; pix++)
    {
        if (!info.pixelActive[pix]) continue;
        uint pixelIndex = raster_pixel_index(info.pixelCoord[pix]);
        g_TrainingForwardState[pixelIndex] = float4(forwardState.pixelState[pix].accum, forwardState.pixelState[pix].trans);
        g_TrainingProcessedEnd[pixelIndex] = forwardState.processedEnd[pix];
    }
}

void load_training_forward_cache(RasterMicrotileInfo info, out RasterMicrotileForwardState forwardState)
{
    init_raster_microtile_forward_state(info, forwardState);
    [unroll] for (uint pix = 0u; pix < RASTER_MICROTILE_PIXELS; pix++)
    {
        if (!info.pixelActive[pix]) continue;
        uint pixelIndex = raster_pixel_index(info.pixelCoord[pix]);
        float4 cachedState = g_TrainingForwardState[pixelIndex];
        forwardState.pixelState[pix] = { cachedState.xyz, cachedState.w };
        forwardState.processedEnd[pix] = g_TrainingProcessedEnd[pixelIndex];
    }
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
        clear_batch_gradient_cache(batchCount, lane);
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
            RasterGaussian.accumulate_gradients_shared(localIdx, dGaussian);
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
    uint gradCount = g_Prepass.splatCount * TRAINABLE_PARAM_COUNT;
    uint idx = dispatchThreadID.x;
    if (idx >= gradCount) return;

    g_ParamGrads[idx] = 0.0;
    g_ParamGradsFixed[idx] = 0;
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csDecodeRasterGradsFixed(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint gradCount = g_Prepass.splatCount * TRAINABLE_PARAM_COUNT;
    uint idx = dispatchThreadID.x;
    if (idx >= gradCount) return;

    g_ParamGrads[idx] = raster_grad_fixed_decode(g_ParamGradsFixed[idx]) * g_RasterGradDecodeScale;
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
void csRasterizeTrainingForward(uint3 dispatchThreadID : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    RasterMicrotileInfo info = resolve_raster_microtile(dispatchThreadID);
    RasterMicrotileForwardState forwardState;
    init_raster_microtile_forward_state(info, forwardState);
    if (info.hasTileRange) raster_forward_replay(info, lane, forwardState);
    store_training_forward_cache(info, forwardState);

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
void csRasterizeBackward(uint3 dispatchThreadID : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    RasterMicrotileInfo info = resolve_raster_microtile(dispatchThreadID);
    RasterMicrotileForwardState forwardState;
    RasterMicrotileBackwardState backwardState;
    load_training_forward_cache(info, forwardState);
    init_raster_microtile_backward_state(backwardState);

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
