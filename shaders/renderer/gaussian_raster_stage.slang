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
static const float RASTER_GRAD_FIXED_COLOR_RANGE = 100.0;
static const float RASTER_GRAD_FIXED_OPACITY_RANGE = 100.0;
static const float RASTER_GRAD_FIXED_RO_LOCAL_REF_SCALE = 100.0;
static const float RASTER_GRAD_FIXED_SCALE_RO_LOCAL = 134217728.0;
static const float RASTER_GRAD_FIXED_INV_SCALE_RO_LOCAL = 1.0 / RASTER_GRAD_FIXED_SCALE_RO_LOCAL;
static const float RASTER_GRAD_FIXED_SCALE_LOG_L_DIAG = 33554432.0;
static const float RASTER_GRAD_FIXED_INV_SCALE_LOG_L_DIAG = 1.0 / RASTER_GRAD_FIXED_SCALE_LOG_L_DIAG;
static const float RASTER_GRAD_FIXED_SCALE_L_OFFDIAG = 16777216.0;
static const float RASTER_GRAD_FIXED_INV_SCALE_L_OFFDIAG = 1.0 / RASTER_GRAD_FIXED_SCALE_L_OFFDIAG;
static const float RASTER_GRAD_FIXED_SCALE_COLOR = RASTER_GRAD_FIXED_INT_MAX_F / RASTER_GRAD_FIXED_COLOR_RANGE;
static const float RASTER_GRAD_FIXED_INV_SCALE_COLOR = RASTER_GRAD_FIXED_COLOR_RANGE / RASTER_GRAD_FIXED_INT_MAX_F;
static const float RASTER_GRAD_FIXED_SCALE_OPACITY = RASTER_GRAD_FIXED_INT_MAX_F / RASTER_GRAD_FIXED_OPACITY_RANGE;
static const float RASTER_GRAD_FIXED_INV_SCALE_OPACITY = RASTER_GRAD_FIXED_OPACITY_RANGE / RASTER_GRAD_FIXED_INT_MAX_F;
static const float RASTER_GRAD_FIXED_LOG_L_DIAG_REF_FLOOR = 0.25;
static const float RASTER_GRAD_FIXED_L_OFFDIAG_REF_FLOOR = 0.25;

float raster_grad_fixed_base_encode_scale(uint fieldId)
{
    return fieldId < RASTER_CACHE_LOG_L_DIAG_X ? RASTER_GRAD_FIXED_SCALE_RO_LOCAL :
        fieldId < RASTER_CACHE_L_OFFDIAG_YX ? RASTER_GRAD_FIXED_SCALE_LOG_L_DIAG :
        fieldId < RASTER_CACHE_COLOR_R ? RASTER_GRAD_FIXED_SCALE_L_OFFDIAG :
        fieldId < RASTER_CACHE_FUSED_OPACITY ? RASTER_GRAD_FIXED_SCALE_COLOR :
        RASTER_GRAD_FIXED_SCALE_OPACITY;
}

float raster_grad_fixed_encode_scale(uint fieldId, float refValue)
{
    return raster_grad_fixed_base_encode_scale(fieldId) * g_CachedRasterGradFixedScale / max(refValue, SMALL_VALUE);
}

float raster_grad_fixed_base_decode_scale(uint fieldId)
{
    return fieldId < RASTER_CACHE_LOG_L_DIAG_X ? RASTER_GRAD_FIXED_INV_SCALE_RO_LOCAL :
        fieldId < RASTER_CACHE_L_OFFDIAG_YX ? RASTER_GRAD_FIXED_INV_SCALE_LOG_L_DIAG :
        fieldId < RASTER_CACHE_COLOR_R ? RASTER_GRAD_FIXED_INV_SCALE_L_OFFDIAG :
        fieldId < RASTER_CACHE_FUSED_OPACITY ? RASTER_GRAD_FIXED_INV_SCALE_COLOR :
        RASTER_GRAD_FIXED_INV_SCALE_OPACITY;
}

float raster_grad_fixed_decode_scale(uint fieldId, float refValue)
{
    return raster_grad_fixed_base_decode_scale(fieldId) * max(refValue, SMALL_VALUE) / g_CachedRasterGradFixedScale;
}

int raster_grad_fixed_encode(float value, uint fieldId, float refValue)
{
    float scaled = value * raster_grad_fixed_encode_scale(fieldId, refValue);
    return int(scaled >= 0.0 ? floor(scaled + 0.5) : ceil(scaled - 0.5));
}

float raster_grad_fixed_decode(int value, uint fieldId, float refValue)
{
    return float(value) * raster_grad_fixed_decode_scale(fieldId, refValue);
}

float3 raster_grad_fixed_log_ldiag_refs(float3 logLDiag)
{
    return max(abs(logLDiag), float3(RASTER_GRAD_FIXED_LOG_L_DIAG_REF_FLOOR));
}

float raster_grad_fixed_shape_alpha(float3 logLDiag)
{
    float3 lDiag = exp(logLDiag);
    return max(pow(max(lDiag.x * lDiag.y * lDiag.z, SMALL_VALUE), 1.0 / 3.0), RASTER_GRAD_FIXED_L_OFFDIAG_REF_FLOOR);
}

float3 raster_grad_fixed_ro_local_refs(float3 logLDiag)
{
    return float3(raster_grad_fixed_shape_alpha(logLDiag) * RASTER_GRAD_FIXED_RO_LOCAL_REF_SCALE);
}

float3 raster_grad_fixed_loffdiag_refs(float3 logLDiag)
{
    return float3(raster_grad_fixed_shape_alpha(logLDiag) * g_CachedRasterGradFixedLOffdiagRefScale);
}

void atomic_add_raster_cache_grad_fixed<let N : int>(uint splatCount, uint firstFieldId, uint splatId, vector<float, N> grad, vector<float, N> refs)
{
    [unroll] for (int i = 0; i < N; i++)
    {
        // float waveGrad = WaveActiveSum(grad[i]);
        // if (!WaveIsFirstLane()) continue;
        float waveGrad = grad[i];
        uint fieldId = firstFieldId + uint(i);
        int value = raster_grad_fixed_encode(waveGrad, fieldId, refs[i]);
        if (value == 0) continue;
        InterlockedAdd(g_CachedRasterGradsFixed[raster_cache_index(splatCount, fieldId, splatId)], value);
    }
}

void store_param_grad<let N : int>(uint splatCount, uint firstParamId, uint splatId, vector<float, N> grad)
{
    [unroll] for (int i = 0; i < N; i++)
    {
        g_ParamGrads[splat_param_index(splatCount, firstParamId + uint(i), splatId)] = grad[i];
    }
}

struct RasterGaussian : IDifferentiable
{
    float3 roLocal;
    float3 logLDiag;
    float3 lOffDiag;
    float4 colorAlpha;

    static RasterGaussian load_global(uint splatId)
    {
        uint splatCount = g_Prepass.splatCount;
        RasterGaussian gaussian;
        gaussian.roLocal = float3(
            g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_RO_LOCAL_X, splatId)],
            g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_RO_LOCAL_Y, splatId)],
            g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_RO_LOCAL_Z, splatId)]
        );
        gaussian.logLDiag = float3(
            g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_LOG_L_DIAG_X, splatId)],
            g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_LOG_L_DIAG_Y, splatId)],
            g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_LOG_L_DIAG_Z, splatId)]
        );
        gaussian.lOffDiag = float3(
            g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_L_OFFDIAG_YX, splatId)],
            g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_L_OFFDIAG_ZX, splatId)],
            g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_L_OFFDIAG_ZY, splatId)]
        );
        gaussian.colorAlpha = float4(
            g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_COLOR_R, splatId)],
            g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_COLOR_G, splatId)],
            g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_COLOR_B, splatId)],
            g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_FUSED_OPACITY, splatId)]
        );
        return gaussian;
    }

    static RasterGaussian load_shared(uint localIdx)
    {
        RasterGaussian gaussian;
        gaussian.roLocal = gsRoLocal[localIdx];
        gaussian.logLDiag = gsLogDiag[localIdx];
        gaussian.lOffDiag = gsNormOffDiag[localIdx];
        gaussian.colorAlpha = float4(gsColorAlpha[localIdx]);
        return gaussian;
    }

    [mutating]
    void store_shared(uint localIdx)
    {
        gsRoLocal[localIdx] = roLocal;
        gsLogDiag[localIdx] = logLDiag;
        gsNormOffDiag[localIdx] = lOffDiag;
        gsColorAlpha[localIdx] = half4(colorAlpha);
    }

    static void capture_gradient_contribution(uint splatId, uint2 pixelCoord, uint globalIdx, Differential grad)
    {
        if (!raster_grad_contribution_capture_enabled()) return;
        uint slotPlusOne = g_CachedRasterGradContributionSelect[splatId];
        if (slotPlusOne == 0u) return;
        uint slot = slotPlusOne - 1u;
        uint captureIdx;
        InterlockedAdd(g_CachedRasterGradContributionCounter[slot], 1u, captureIdx);
        if (captureIdx >= g_CachedRasterGradContributionPerSplatCapacity)
        {
            InterlockedAdd(g_CachedRasterGradContributionCounter[g_CachedRasterGradContributionSelectedCount], 1u);
            return;
        }

        RasterGradContributionRecord record;
        record.meta = float4(float(splatId), float(pixelCoord.x), float(pixelCoord.y), float(globalIdx));
        record.grad0 = float4(grad.roLocal, grad.logLDiag.x);
        record.grad1 = float4(grad.logLDiag.yz, grad.lOffDiag.xy);
        record.grad2 = float4(grad.lOffDiag.z, grad.colorAlpha.xyz);
        record.grad3 = float4(grad.colorAlpha.w, 0.0, 0.0, 0.0);
        g_CachedRasterGradContributionRecords[slot * g_CachedRasterGradContributionPerSplatCapacity + captureIdx] = record;
    }

    static void accumulate_gradients_fixed(uint localIdx, uint2 pixelCoord, uint globalIdx, Differential dOut)
    {
        uint splatId = gsBatchSplatId[localIdx];
        uint splatCount = g_Prepass.splatCount;
        RasterGaussian gaussian = load_shared(localIdx);
        float3 roLocalRefs = raster_grad_fixed_ro_local_refs(gaussian.logLDiag);
        float3 logLDiagRefs = raster_grad_fixed_log_ldiag_refs(gaussian.logLDiag);
        float3 lOffDiagRefs = raster_grad_fixed_loffdiag_refs(gaussian.logLDiag);
        capture_gradient_contribution(splatId, pixelCoord, globalIdx, dOut);
        atomic_add_raster_cache_grad_fixed<3>(splatCount, RASTER_CACHE_RO_LOCAL_X, splatId, dOut.roLocal, roLocalRefs);
        atomic_add_raster_cache_grad_fixed<3>(splatCount, RASTER_CACHE_LOG_L_DIAG_X, splatId, dOut.logLDiag, logLDiagRefs);
        atomic_add_raster_cache_grad_fixed<3>(splatCount, RASTER_CACHE_L_OFFDIAG_YX, splatId, dOut.lOffDiag, lOffDiagRefs);
        atomic_add_raster_cache_grad_fixed<4>(splatCount, RASTER_CACHE_COLOR_R, splatId, dOut.colorAlpha, float4(1.0));
    }
};

RasterGaussian.Differential load_cached_raster_grad(uint splatId)
{
    uint splatCount = g_Prepass.splatCount;
    float3 cachedLogLDiag = float3(
        g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_LOG_L_DIAG_X, splatId)],
        g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_LOG_L_DIAG_Y, splatId)],
        g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_LOG_L_DIAG_Z, splatId)]
    );
    float3 roLocalRefs = raster_grad_fixed_ro_local_refs(cachedLogLDiag);
    float3 logLDiagRefs = raster_grad_fixed_log_ldiag_refs(cachedLogLDiag);
    float3 lOffDiagRefs = raster_grad_fixed_loffdiag_refs(cachedLogLDiag);
    RasterGaussian.Differential grad;
    grad.roLocal = float3(
        raster_grad_fixed_decode(g_CachedRasterGradsFixed[raster_cache_index(splatCount, RASTER_CACHE_RO_LOCAL_X, splatId)], RASTER_CACHE_RO_LOCAL_X, roLocalRefs.x),
        raster_grad_fixed_decode(g_CachedRasterGradsFixed[raster_cache_index(splatCount, RASTER_CACHE_RO_LOCAL_Y, splatId)], RASTER_CACHE_RO_LOCAL_Y, roLocalRefs.y),
        raster_grad_fixed_decode(g_CachedRasterGradsFixed[raster_cache_index(splatCount, RASTER_CACHE_RO_LOCAL_Z, splatId)], RASTER_CACHE_RO_LOCAL_Z, roLocalRefs.z)
    );
    grad.logLDiag = float3(
        raster_grad_fixed_decode(g_CachedRasterGradsFixed[raster_cache_index(splatCount, RASTER_CACHE_LOG_L_DIAG_X, splatId)], RASTER_CACHE_LOG_L_DIAG_X, logLDiagRefs.x),
        raster_grad_fixed_decode(g_CachedRasterGradsFixed[raster_cache_index(splatCount, RASTER_CACHE_LOG_L_DIAG_Y, splatId)], RASTER_CACHE_LOG_L_DIAG_Y, logLDiagRefs.y),
        raster_grad_fixed_decode(g_CachedRasterGradsFixed[raster_cache_index(splatCount, RASTER_CACHE_LOG_L_DIAG_Z, splatId)], RASTER_CACHE_LOG_L_DIAG_Z, logLDiagRefs.z)
    );
    grad.lOffDiag = float3(
        raster_grad_fixed_decode(g_CachedRasterGradsFixed[raster_cache_index(splatCount, RASTER_CACHE_L_OFFDIAG_YX, splatId)], RASTER_CACHE_L_OFFDIAG_YX, lOffDiagRefs.x),
        raster_grad_fixed_decode(g_CachedRasterGradsFixed[raster_cache_index(splatCount, RASTER_CACHE_L_OFFDIAG_ZX, splatId)], RASTER_CACHE_L_OFFDIAG_ZX, lOffDiagRefs.y),
        raster_grad_fixed_decode(g_CachedRasterGradsFixed[raster_cache_index(splatCount, RASTER_CACHE_L_OFFDIAG_ZY, splatId)], RASTER_CACHE_L_OFFDIAG_ZY, lOffDiagRefs.z)
    );
    grad.colorAlpha = float4(
        raster_grad_fixed_decode(g_CachedRasterGradsFixed[raster_cache_index(splatCount, RASTER_CACHE_COLOR_R, splatId)], RASTER_CACHE_COLOR_R, 1.0),
        raster_grad_fixed_decode(g_CachedRasterGradsFixed[raster_cache_index(splatCount, RASTER_CACHE_COLOR_G, splatId)], RASTER_CACHE_COLOR_G, 1.0),
        raster_grad_fixed_decode(g_CachedRasterGradsFixed[raster_cache_index(splatCount, RASTER_CACHE_COLOR_B, splatId)], RASTER_CACHE_COLOR_B, 1.0),
        raster_grad_fixed_decode(g_CachedRasterGradsFixed[raster_cache_index(splatCount, RASTER_CACHE_FUSED_OPACITY, splatId)], RASTER_CACHE_FUSED_OPACITY, 1.0)
    );
    return grad;
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
        float alpha = gaussian.colorAlpha.w * coverage;
        return alpha >= g_Raster.alphaCutoff ? alpha : 0.0;
    }
    float alpha = raster_eval_alpha(gaussian.roLocal, gaussian.logLDiag, gaussian.lOffDiag, gaussian.colorAlpha.w, rayDir, 0.0);
    return alpha >= g_Raster.alphaCutoff ? alpha : 0.0;
}

[Differentiable]
RasterState raster_blend_step_gaussian(RasterState state, RasterGaussian gaussian, float alpha)
{
    return state.blend(gaussian.colorAlpha.xyz, alpha);
}

[Differentiable]
RasterState raster_blend_step_color_alpha(RasterState state, float4 colorAlpha, float alpha)
{
    return state.blend(colorAlpha.xyz, alpha);
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

struct RasterPixelInfo
{
    bool pixelActive;
    bool hasTileRange;
    uint start;
    uint end;
    uint2 pixelCoord;
    float2 samplePoint;
    float3 rayDir;
};

struct RasterForwardState
{
    RasterState pixelState;
    uint processedCount;
    uint processedEnd;
};

struct RasterBackwardState
{
    RasterState.Differential pixelState;
};

void init_raster_forward_state(RasterPixelInfo info, out RasterForwardState forwardState)
{
    forwardState.pixelState = { 0.0, 1.0 };
    forwardState.processedCount = 0u;
    forwardState.processedEnd = info.start;
}

void init_raster_backward_state(out RasterBackwardState backwardState)
{
    backwardState.pixelState = { 0.0, 0.0 };
}

RasterPixelInfo resolve_raster_pixel(uint3 dispatchThreadID)
{
    RasterPixelInfo info;
    info.pixelCoord = dispatchThreadID.xy;
    info.pixelActive = info.pixelCoord.x < g_Raster.width && info.pixelCoord.y < g_Raster.height;
    info.hasTileRange = false;
    info.start = 0u;
    info.end = 0u;
    info.samplePoint = float2(info.pixelCoord) + 0.5;

    PinholeCamera camera = PinholeCamera.from_params(g_Camera);
    uint samplePx = min(info.pixelCoord.x, g_Raster.width - 1u);
    uint samplePy = min(info.pixelCoord.y, g_Raster.height - 1u);
    info.rayDir = camera.screen_to_world_ray(float2(samplePx, samplePy) + 0.5);

    uint tileX = samplePx / g_Prepass.tileSize;
    uint tileY = samplePy / g_Prepass.tileSize;
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
    RasterPixelInfo info,
    uint lane,
    inout RasterForwardState forwardState
)
{
    bool pixelDone = !info.pixelActive;
    uint replayCount = 0u;

    for (uint batchStart = info.start; batchStart < info.end; batchStart += RASTER_BATCH)
    {
        uint batchEnd = min(batchStart + RASTER_BATCH, info.end);
        uint batchCount = batchEnd - batchStart;
        prefetch_batch_from_global(batchStart, batchEnd, lane);
        GroupMemoryBarrierWithGroupSync();

        if (!pixelDone)
        {
            for (uint localIdx = 0u; localIdx < batchCount; localIdx++)
            {
                RasterGaussian gaussian = RasterGaussian.load_shared(localIdx);
                uint splatId = gsBatchSplatId[localIdx];
                float3 color = raster_forward_color(gaussian, splatId);
                replayCount += 1u;
                float alpha = raster_eval_alpha_gaussian(gaussian, splatId, info.rayDir, info.samplePoint);
                if (alpha <= 0.0) continue;

                RasterState state = forwardState.pixelState;
                state = state.blend(color, alpha);
                forwardState.pixelState = state;
                forwardState.processedCount += 1u;
                if (state.trans >= g_Raster.transmittanceThreshold) continue;

                pixelDone = true;
                break;
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }
    forwardState.processedEnd = info.start + replayCount;
}

float4 raster_forward_output_pixel(
    RasterPixelInfo info,
    RasterState state,
    uint processedCount
)
{
    if (!info.pixelActive) return float4(0.0, 0.0, 0.0, 1.0);
    if (g_Raster.debugShowProcessedCount != 0u) return float4(raster_processed_count_color(processedCount), 1.0);
    return info.hasTileRange
        ? raster_output_from_state(state, g_Raster.background)
        : float4(pow(max(g_Raster.background, 0.0), OUTPUT_GAMMA), 1.0);
}

RasterState.Differential raster_output_gradient_pixel(
    RasterPixelInfo info,
    RasterState state
)
{
    if (!info.pixelActive || !info.hasTileRange) return { 0.0, 0.0 };

    var dpState = diffPair(state);
    bwd_diff(raster_output_from_state)(dpState, g_Raster.background, g_OutputGrad[info.pixelCoord.y * g_Raster.width + info.pixelCoord.x]);
    return dpState.getDifferential();
}

uint raster_pixel_index(uint2 pixelCoord)
{
    return pixelCoord.y * g_Raster.width + pixelCoord.x;
}

void store_training_forward_cache(RasterPixelInfo info, RasterForwardState forwardState)
{
    if (!info.pixelActive) return;
    uint pixelIndex = raster_pixel_index(info.pixelCoord);
    g_TrainingForwardState[pixelIndex] = float4(forwardState.pixelState.accum, forwardState.pixelState.trans);
    g_TrainingProcessedEnd[pixelIndex] = forwardState.processedEnd;
}

void load_training_forward_cache(RasterPixelInfo info, out RasterForwardState forwardState)
{
    init_raster_forward_state(info, forwardState);
    if (!info.pixelActive) return;
    uint pixelIndex = raster_pixel_index(info.pixelCoord);
    float4 cachedState = g_TrainingForwardState[pixelIndex];
    forwardState.pixelState = { cachedState.xyz, cachedState.w };
    forwardState.processedEnd = g_TrainingProcessedEnd[pixelIndex];
}

void raster_backward_replay(
    RasterPixelInfo info,
    inout RasterForwardState forwardState,
    inout RasterBackwardState backwardState,
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
            RasterGaussian.Differential dGaussian = { 0.0, 0.0, 0.0, 0.0 };
            if (info.pixelActive && globalIdx < forwardState.processedEnd)
            {
                float alpha = raster_eval_alpha_gaussian(gaussian, gsBatchSplatId[localIdx], info.rayDir, info.samplePoint);
                if (alpha > 0.0)
                {
                    RasterState state = forwardState.pixelState;
                    RasterState prevState = state.undo(gaussian.colorAlpha.xyz, alpha);

                    var dpPrevState = diffPair(prevState);
                    var dpGaussian = diffPair(gaussian);
                    var dpColorAlpha = diffPair(gaussian.colorAlpha);
                    var dpAlpha = diffPair(alpha);
                    bwd_diff(raster_blend_step_color_alpha)(dpPrevState, dpColorAlpha, dpAlpha, backwardState.pixelState);
                    bwd_diff(raster_eval_alpha_gaussian)(dpGaussian, gsBatchSplatId[localIdx], info.rayDir, info.samplePoint, dpAlpha.getDifferential());

                    forwardState.pixelState = prevState;
                    backwardState.pixelState = dpPrevState.getDifferential();
                    RasterGaussian.Differential dGaussian = dpGaussian.getDifferential();
                    dGaussian.colorAlpha += dpColorAlpha.getDifferential();
                    RasterGaussian.accumulate_gradients_fixed(localIdx, info.pixelCoord, globalIdx, dGaussian);
                }
            }
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
    uint cachedGradCount = g_Prepass.splatCount * RASTER_CACHE_PARAM_COUNT;
    uint idx = dispatchThreadID.x;
    if (idx < gradCount) g_ParamGrads[idx] = 0.0;
    if (idx < cachedGradCount) g_CachedRasterGradsFixed[idx] = 0;
    if (idx < cachedGradCount) g_CachedRasterGradsFloat[idx] = 0.0;
    if (idx < g_CachedRasterGradContributionSelectedCount + 1u) g_CachedRasterGradContributionCounter[idx] = 0u;
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csBackpropCachedRasterGrads(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_Prepass.splatCount) return;

    RasterGaussian.Differential dRaster = load_cached_raster_grad(splatId);
    CachedEllipsoid.Differential dCached;
    dCached.roLocal = dRaster.roLocal;
    dCached.logLDiag = dRaster.logLDiag;
    dCached.lOffDiag = dRaster.lOffDiag;

    Gaussian3D gaussian = load_gaussian3d(splatId);
    var dpCachedPosition = diffPair(gaussian.position);
    var dpCachedScale = diffPair(gaussian.logScale);
    var dpRotation = diffPair(gaussian.rotation);
    var dpOpacityPosition = diffPair(gaussian.position);
    var dpOpacityScale = diffPair(gaussian.logScale);
    var dpRawOpacity = diffPair(gaussian.colorAlpha.w);
    bwd_diff(build_cached_ellipsoid)(g_Camera, g_Prepass.radiusScale, dpCachedPosition, dpCachedScale, dpRotation, dCached);
    bwd_diff(build_raster_fused_opacity)(g_Camera, g_Prepass.radiusScale, dpOpacityPosition, dpOpacityScale, dpRawOpacity, dRaster.colorAlpha.w);

    float gradScale = g_RasterGradDecodeScale;
    store_param_grad<3>(g_Prepass.splatCount, PARAM_POSITION_X, splatId, (dpCachedPosition.getDifferential() + dpOpacityPosition.getDifferential()) * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, PARAM_SCALE_X, splatId, (dpCachedScale.getDifferential() + dpOpacityScale.getDifferential()) * gradScale);
    store_param_grad<4>(g_Prepass.splatCount, PARAM_ROTATION_X, splatId, dpRotation.getDifferential() * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, PARAM_COLOR_R, splatId, dRaster.colorAlpha.xyz * gradScale);
    store_param_grad<1>(g_Prepass.splatCount, PARAM_RAW_OPACITY, splatId, dpRawOpacity.getDifferential() * gradScale);
}

[shader("compute")]
[numthreads(RASTER_THREAD_TILE_DIM, RASTER_THREAD_TILE_DIM, 1)]
void csRasterize(uint3 dispatchThreadID : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    RasterPixelInfo info = resolve_raster_pixel(dispatchThreadID);
    RasterForwardState forwardState;
    init_raster_forward_state(info, forwardState);
    if (info.hasTileRange) raster_forward_replay(info, lane, forwardState);
    if (!info.pixelActive) return;
    g_Output[info.pixelCoord] = raster_forward_output_pixel(info, forwardState.pixelState, forwardState.processedCount);
}

[shader("compute")]
[numthreads(RASTER_THREAD_TILE_DIM, RASTER_THREAD_TILE_DIM, 1)]
void csRasterizeTrainingForward(uint3 dispatchThreadID : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    RasterPixelInfo info = resolve_raster_pixel(dispatchThreadID);
    RasterForwardState forwardState;
    init_raster_forward_state(info, forwardState);
    if (info.hasTileRange) raster_forward_replay(info, lane, forwardState);
    store_training_forward_cache(info, forwardState);
    if (!info.pixelActive) return;
    g_Output[info.pixelCoord] = raster_forward_output_pixel(info, forwardState.pixelState, forwardState.processedCount);
}

[shader("compute")]
[numthreads(RASTER_THREAD_TILE_DIM, RASTER_THREAD_TILE_DIM, 1)]
void csRasterizeBackward(uint3 dispatchThreadID : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    RasterPixelInfo info = resolve_raster_pixel(dispatchThreadID);
    RasterForwardState forwardState;
    RasterBackwardState backwardState;
    load_training_forward_cache(info, forwardState);
    init_raster_backward_state(backwardState);
    backwardState.pixelState = raster_output_gradient_pixel(info, forwardState.pixelState);

    raster_backward_replay(info, forwardState, backwardState, lane);
}
