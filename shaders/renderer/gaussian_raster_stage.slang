import "gaussian_types.slang";
import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/random/random.slang";
import "utility/splatting/common.slang";
import "utility/splatting/opacity.slang";
import "utility/splatting/params.slang";
import "utility/splatting/raster.slang";

uniform float g_MaxAnisotropy;
uniform RWStructuredBuffer<uint> g_CloneCounts;
uniform float g_CloneSelectProbability;
uniform uint g_CloneSeed;

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

static const float RASTER_GRAD_FIXED_INT_MIN_F = -2147483648.0;
static const float RASTER_GRAD_FIXED_INT_MAX_F = 2147483647.0;
static const float RASTER_GRAD_FIXED_AVG_INV_SCALE_FLOOR = 0.25;

int raster_grad_fixed_quantize(float value, float range)
{
    float normalized = clamp(value / range, -1.0, 1.0) * RASTER_GRAD_FIXED_INT_MAX_F;
    return int(normalized >= 0.0 ? floor(normalized + 0.5) : ceil(normalized - 0.5));
}

float raster_grad_fixed_dequantize(int value, float range)
{
    return float(value) * range / RASTER_GRAD_FIXED_INT_MAX_F;
}

float raster_grad_fixed_avg_inv_scale(float3 scale)
{
    float avgScale = pow(max(scale.x * scale.y * scale.z, SMALL_VALUE), 1.0 / 3.0);
    return max(1.0 / max(avgScale, SMALL_VALUE), RASTER_GRAD_FIXED_AVG_INV_SCALE_FLOOR);
}

void store_param_grad<let N : int>(uint splatCount, uint firstParamId, uint splatId, vector<float, N> grad)
{
    [unroll] for (int i = 0; i < N; i++)
    {
        g_ParamGrads[splat_param_index(splatCount, firstParamId + uint(i), splatId)] = grad[i];
    }
}

RasterGaussian load_cached_raster_grad_fixed(uint splatId)
{
    uint splatCount = g_Prepass.splatCount;
    RasterGaussian cached = g_RasterCache[splatId];
    float avgInvScale = raster_grad_fixed_avg_inv_scale(cached.scale);
    RasterGaussian grad;
    grad.roLocal = float3(
        raster_grad_fixed_dequantize(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_RO_LOCAL_X, splatId)], g_CachedRasterGradFixedROLocalRange) * avgInvScale,
        raster_grad_fixed_dequantize(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_RO_LOCAL_Y, splatId)], g_CachedRasterGradFixedROLocalRange) * avgInvScale,
        raster_grad_fixed_dequantize(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_RO_LOCAL_Z, splatId)], g_CachedRasterGradFixedROLocalRange) * avgInvScale
    );
    grad.scale = float3(
        raster_grad_fixed_dequantize(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_SCALE_X, splatId)], g_CachedRasterGradFixedScaleRange) * avgInvScale,
        raster_grad_fixed_dequantize(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_SCALE_Y, splatId)], g_CachedRasterGradFixedScaleRange) * avgInvScale,
        raster_grad_fixed_dequantize(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_SCALE_Z, splatId)], g_CachedRasterGradFixedScaleRange) * avgInvScale
    );
    grad.rotation = float4(
        raster_grad_fixed_dequantize(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_QUAT_W, splatId)], g_CachedRasterGradFixedQuatRange) * avgInvScale,
        raster_grad_fixed_dequantize(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_QUAT_X, splatId)], g_CachedRasterGradFixedQuatRange) * avgInvScale,
        raster_grad_fixed_dequantize(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_QUAT_Y, splatId)], g_CachedRasterGradFixedQuatRange) * avgInvScale,
        raster_grad_fixed_dequantize(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_QUAT_Z, splatId)], g_CachedRasterGradFixedQuatRange) * avgInvScale
    );
    grad.colorAlpha = float4(
        raster_grad_fixed_dequantize(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_COLOR_R, splatId)], g_CachedRasterGradFixedColorRange),
        raster_grad_fixed_dequantize(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_COLOR_G, splatId)], g_CachedRasterGradFixedColorRange),
        raster_grad_fixed_dequantize(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_COLOR_B, splatId)], g_CachedRasterGradFixedColorRange),
        raster_grad_fixed_dequantize(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_FUSED_OPACITY, splatId)], g_CachedRasterGradFixedOpacityRange)
    );
    return grad;
}

RasterGaussian load_cached_raster_grad_float(uint splatId)
{
    uint splatCount = g_Prepass.splatCount;
    RasterGaussian grad;
    grad.roLocal = float3(
        g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_RO_LOCAL_X, splatId)],
        g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_RO_LOCAL_Y, splatId)],
        g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_RO_LOCAL_Z, splatId)]
    );
    grad.scale = float3(
        g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_SCALE_X, splatId)],
        g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_SCALE_Y, splatId)],
        g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_SCALE_Z, splatId)]
    );
    grad.rotation = float4(
        g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_QUAT_W, splatId)],
        g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_QUAT_X, splatId)],
        g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_QUAT_Y, splatId)],
        g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_QUAT_Z, splatId)]
    );
    grad.colorAlpha = float4(
        g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_COLOR_R, splatId)],
        g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_COLOR_G, splatId)],
        g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_COLOR_B, splatId)],
        g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_FUSED_OPACITY, splatId)]
    );
    return grad;
}

void flush_cached_raster_grad_fixed(uint splatId, RasterGaussian gaussian, RasterGaussian.Differential dGaussian)
{
    uint splatCount = g_Prepass.splatCount;
    float avgInvScale = 1.0 / raster_grad_fixed_avg_inv_scale(gaussian.scale);
    // float3 roLocal = WaveActiveSum(dGaussian.roLocal * avgInvScale);
    // float3 scale = WaveActiveSum(dGaussian.scale * avgInvScale);
    // float4 rotation = WaveActiveSum(dGaussian.rotation * avgInvScale);
    // float4 colorAlpha = WaveActiveSum(dGaussian.colorAlpha);
    // if (!WaveIsFirstLane()) return;

    // for fixed point wave ops don't help, at least on nvidia
    float3 roLocal = dGaussian.roLocal * avgInvScale;
    float3 scale = dGaussian.scale * avgInvScale;
    float4 rotation = dGaussian.rotation * avgInvScale;
    float4 colorAlpha = dGaussian.colorAlpha;

    int qRoLocalX = raster_grad_fixed_quantize(roLocal.x, g_CachedRasterGradFixedROLocalRange);
    int qRoLocalY = raster_grad_fixed_quantize(roLocal.y, g_CachedRasterGradFixedROLocalRange);
    int qRoLocalZ = raster_grad_fixed_quantize(roLocal.z, g_CachedRasterGradFixedROLocalRange);
    int qScaleX = raster_grad_fixed_quantize(scale.x, g_CachedRasterGradFixedScaleRange);
    int qScaleY = raster_grad_fixed_quantize(scale.y, g_CachedRasterGradFixedScaleRange);
    int qScaleZ = raster_grad_fixed_quantize(scale.z, g_CachedRasterGradFixedScaleRange);
    int qRotationW = raster_grad_fixed_quantize(rotation.x, g_CachedRasterGradFixedQuatRange);
    int qRotationX = raster_grad_fixed_quantize(rotation.y, g_CachedRasterGradFixedQuatRange);
    int qRotationY = raster_grad_fixed_quantize(rotation.z, g_CachedRasterGradFixedQuatRange);
    int qRotationZ = raster_grad_fixed_quantize(rotation.w, g_CachedRasterGradFixedQuatRange);
    int qColorR = raster_grad_fixed_quantize(colorAlpha.x, g_CachedRasterGradFixedColorRange);
    int qColorG = raster_grad_fixed_quantize(colorAlpha.y, g_CachedRasterGradFixedColorRange);
    int qColorB = raster_grad_fixed_quantize(colorAlpha.z, g_CachedRasterGradFixedColorRange);
    int qOpacity = raster_grad_fixed_quantize(colorAlpha.w, g_CachedRasterGradFixedOpacityRange);

    if (qRoLocalX != 0) InterlockedAdd(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_RO_LOCAL_X, splatId)], qRoLocalX);
    if (qRoLocalY != 0) InterlockedAdd(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_RO_LOCAL_Y, splatId)], qRoLocalY);
    if (qRoLocalZ != 0) InterlockedAdd(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_RO_LOCAL_Z, splatId)], qRoLocalZ);
    if (qScaleX != 0) InterlockedAdd(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_SCALE_X, splatId)], qScaleX);
    if (qScaleY != 0) InterlockedAdd(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_SCALE_Y, splatId)], qScaleY);
    if (qScaleZ != 0) InterlockedAdd(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_SCALE_Z, splatId)], qScaleZ);
    if (qRotationW != 0) InterlockedAdd(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_QUAT_W, splatId)], qRotationW);
    if (qRotationX != 0) InterlockedAdd(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_QUAT_X, splatId)], qRotationX);
    if (qRotationY != 0) InterlockedAdd(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_QUAT_Y, splatId)], qRotationY);
    if (qRotationZ != 0) InterlockedAdd(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_QUAT_Z, splatId)], qRotationZ);
    if (qColorR != 0) InterlockedAdd(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_COLOR_R, splatId)], qColorR);
    if (qColorG != 0) InterlockedAdd(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_COLOR_G, splatId)], qColorG);
    if (qColorB != 0) InterlockedAdd(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_COLOR_B, splatId)], qColorB);
    if (qOpacity != 0) InterlockedAdd(g_CachedRasterGradsFixed[cached_raster_grad_index(splatCount, RASTER_CACHE_FUSED_OPACITY, splatId)], qOpacity);
}

void flush_cached_raster_grad_float(uint splatId, RasterGaussian.Differential dGaussian)
{
    uint splatCount = g_Prepass.splatCount;
    float3 roLocal = WaveActiveSum(dGaussian.roLocal);
    float3 scale = WaveActiveSum(dGaussian.scale);
    float4 rotation = WaveActiveSum(dGaussian.rotation);
    float4 colorAlpha = WaveActiveSum(dGaussian.colorAlpha);
    if (!WaveIsFirstLane()) return;

    if (roLocal.x != 0.0) InterlockedAdd(g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_RO_LOCAL_X, splatId)], roLocal.x);
    if (roLocal.y != 0.0) InterlockedAdd(g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_RO_LOCAL_Y, splatId)], roLocal.y);
    if (roLocal.z != 0.0) InterlockedAdd(g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_RO_LOCAL_Z, splatId)], roLocal.z);
    if (scale.x != 0.0) InterlockedAdd(g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_SCALE_X, splatId)], scale.x);
    if (scale.y != 0.0) InterlockedAdd(g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_SCALE_Y, splatId)], scale.y);
    if (scale.z != 0.0) InterlockedAdd(g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_SCALE_Z, splatId)], scale.z);
    if (rotation.x != 0.0) InterlockedAdd(g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_QUAT_W, splatId)], rotation.x);
    if (rotation.y != 0.0) InterlockedAdd(g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_QUAT_X, splatId)], rotation.y);
    if (rotation.z != 0.0) InterlockedAdd(g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_QUAT_Y, splatId)], rotation.z);
    if (rotation.w != 0.0) InterlockedAdd(g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_QUAT_Z, splatId)], rotation.w);
    if (colorAlpha.x != 0.0) InterlockedAdd(g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_COLOR_R, splatId)], colorAlpha.x);
    if (colorAlpha.y != 0.0) InterlockedAdd(g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_COLOR_G, splatId)], colorAlpha.y);
    if (colorAlpha.z != 0.0) InterlockedAdd(g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_COLOR_B, splatId)], colorAlpha.z);
    if (colorAlpha.w != 0.0) InterlockedAdd(g_CachedRasterGradsFloat[cached_raster_grad_index(splatCount, RASTER_CACHE_FUSED_OPACITY, splatId)], colorAlpha.w);
}

float raster_eval_outline_alpha(uint splatId, float fusedOpacity, float2 samplePoint)
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
    float alpha = fusedOpacity * coverage;
    return alpha >= g_Raster.alphaCutoff ? alpha : 0.0;
}

[Differentiable]
float raster_eval_alpha_gaussian(RasterGaussian gaussian, no_diff float3 rayDir)
{
    return raster_eval_alpha(gaussian.roLocal, gaussian.scale, gaussian.rotation, gaussian.colorAlpha.w, rayDir, g_Raster.alphaCutoff);
}

[Differentiable]
float2 raster_eval_alpha_depth_gaussian(RasterGaussian gaussian, no_diff float3 rayDir)
{
    float3 rayLocal = world_to_local_dir(rayDir, gaussian.scale, gaussian.rotation);
    float denom = dot(rayLocal, rayLocal);
    if (denom <= RAY_DENOMINATOR_FLOOR) return 0.0;

    float numer = dot(rayLocal, gaussian.roLocal);
    float tClosest = -numer / denom;
    if (tClosest <= 0.0) return 0.0;

    float3 closest = gaussian.roLocal + rayLocal * tClosest;
    float rho2 = max(dot(closest, closest), 0.0);
    float alpha = gaussian.colorAlpha.w * exp(-0.5 * ENVELOPE_SIGMA_RADIUS * ENVELOPE_SIGMA_RADIUS * rho2);
    if (alpha < g_Raster.alphaCutoff) return 0.0;
    return float2(alpha, tClosest);
}

[Differentiable]
RasterState raster_blend_step_color_alpha(RasterState state, float4 colorAlpha, float alpha)
{
    return state.blend(colorAlpha.xyz, alpha);
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

float3 raster_processed_count_color(uint processedCount)
{
    float normalized = log2(float(processedCount) + 1.0) / max(log2(float(g_Raster.maxSplatSteps) + 1.0), 1.0);
    return jet_colormap(saturate(normalized));
}

float3 raster_clone_count_color(uint cloneCount)
{
    return jet_colormap(debug_normalize_range(float(cloneCount), g_Raster.debugCloneCountRange));
}

float3 raster_forward_color(RasterGaussian gaussian, uint splatId, uint debugMode)
{
    if (debugMode == DEBUG_MODE_CLONE_COUNT) return raster_clone_count_color(g_CloneCounts[splatId]);
    if (debugMode != DEBUG_MODE_GRAD_NORM) return gaussian.colorAlpha.xyz;
    float gradNorm = g_DebugGradNorm[splatId];
    if (!isfinite(gradNorm)) gradNorm = 0.0;
    return jet_colormap(raster_normalize_grad_norm(max(gradNorm, 0.0)));
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
    uint tileId;
    uint2 pixelCoord;
    float2 samplePoint;
    float3 rayDir;
};

struct RasterReplayState
{
    RasterState pixelState;
    uint processedCount;
    uint processedEnd;
    uint replayBatchEnd;
    bool cloneSelected;
    float cloneThreshold;
    uint debugMode;
    float3 depthState;
    float spatialDensitySum;
    float screenDensitySum;
};

struct RasterBackwardState
{
    RasterState.Differential pixelState;
    float3 depthState;
};

void init_raster_forward_state(RasterPixelInfo info, out RasterReplayState replayState)
{
    replayState.pixelState = { 0.0, 1.0 };
    replayState.processedCount = 0u;
    replayState.processedEnd = info.start;
    replayState.replayBatchEnd = info.start;
    uint pixelId = raster_pixel_index(info.pixelCoord);
    uint cloneSeed = hash_u32(pixelId ^ (g_CloneSeed * 0x9e3779b9u + 0x85ebca6bu));
    replayState.cloneSelected = info.pixelActive && g_CloneSelectProbability > 0.0 && ((float(hash_u32(cloneSeed ^ 0x8da6b343u)) + 0.5) * 2.3283064365386963e-10) <= saturate(g_CloneSelectProbability);
    replayState.cloneThreshold = replayState.cloneSelected ? (float(hash_u32(cloneSeed ^ 0x5f2ce0b1u)) + 0.5) * 2.3283064365386963e-10 : -1.0;
    replayState.debugMode = DEBUG_MODE_NORMAL;
    replayState.depthState = 0.0;
    replayState.spatialDensitySum = 0.0;
    replayState.screenDensitySum = 0.0;
}

void init_raster_backward_state(out RasterBackwardState backwardState)
{
    backwardState.pixelState = { 0.0, 0.0 };
    backwardState.depthState = 0.0;
}

void init_raster_debug_state(RasterPixelInfo info, out RasterReplayState replayState)
{
    replayState.pixelState = { 0.0, 1.0 };
    replayState.processedCount = 0u;
    replayState.processedEnd = info.start;
    replayState.replayBatchEnd = info.start;
    replayState.cloneSelected = false;
    replayState.cloneThreshold = -1.0;
    replayState.debugMode = g_Raster.debugMode;
    replayState.depthState = 0.0;
    replayState.spatialDensitySum = 0.0;
    replayState.screenDensitySum = 0.0;
}

RasterPixelInfo resolve_raster_pixel(uint3 groupID, uint lane)
{
    RasterPixelInfo info;
    uint tileId = groupID.x;
    uint tileX = tileId % g_Prepass.tileWidth;
    uint tileY = tileId / g_Prepass.tileWidth;
    uint localX = lane % RASTER_THREAD_TILE_DIM;
    uint localY = lane / RASTER_THREAD_TILE_DIM;
    info.pixelCoord = uint2(tileX * g_Prepass.tileSize + localX, tileY * g_Prepass.tileSize + localY);
    info.pixelActive = info.pixelCoord.x < g_Raster.width && info.pixelCoord.y < g_Raster.height;
    info.hasTileRange = false;
    info.start = 0u;
    info.end = 0u;
    info.samplePoint = float2(info.pixelCoord) + 0.5;

    PinholeCamera camera = PinholeCamera.from_params(g_Camera);
    uint samplePx = min(info.pixelCoord.x, g_Raster.width - 1u);
    uint samplePy = min(info.pixelCoord.y, g_Raster.height - 1u);
    info.rayDir = camera.screen_to_world_ray(float2(samplePx, samplePy) + 0.5);

    info.tileId = tileY * g_Prepass.tileWidth + tileX;
    uint2 range = g_TileRanges[info.tileId];
    info.hasTileRange = range.x != INVALID_START && range.y > range.x;
    info.start = info.hasTileRange ? range.x : 0u;
    info.end = info.hasTileRange ? range.y : info.start;
    return info;
}

interface IRasterReplayVariant
{
    static void init_state(RasterPixelInfo info, out RasterReplayState replayState);
    static bool eval_hit(RasterPixelInfo info, uint splatId, RasterGaussian gaussian, out float alpha, out float rayDepth);
    static void process_hit(RasterPixelInfo info, uint splatId, RasterGaussian gaussian, float alpha, float rayDepth, inout RasterReplayState replayState);
}

struct ForwardRasterReplayVariant : IRasterReplayVariant
{
    static void init_state(RasterPixelInfo info, out RasterReplayState replayState)
    {
        init_raster_forward_state(info, replayState);
    }

    static bool eval_hit(RasterPixelInfo info, uint splatId, RasterGaussian gaussian, out float alpha, out float rayDepth)
    {
        alpha = raster_eval_alpha_gaussian(gaussian, info.rayDir);
        rayDepth = 0.0;
        return alpha > 0.0;
    }

    static void process_hit(RasterPixelInfo info, uint splatId, RasterGaussian gaussian, float alpha, float rayDepth, inout RasterReplayState replayState)
    {
        RasterState state = replayState.pixelState;
        float nextTrans = state.trans * (1.0 - alpha);
        if (replayState.cloneSelected && nextTrans <= replayState.cloneThreshold && state.trans > replayState.cloneThreshold)
        {
            InterlockedAdd(g_CloneCounts[splatId], 1u);
        }
        replayState.pixelState = state.blend(raster_forward_color(gaussian, splatId, replayState.debugMode), alpha);
        replayState.processedCount += 1u;
    }
}

struct TrainingRasterReplayVariant : IRasterReplayVariant
{
    static void init_state(RasterPixelInfo info, out RasterReplayState replayState)
    {
        init_raster_forward_state(info, replayState);
    }

    static bool eval_hit(RasterPixelInfo info, uint splatId, RasterGaussian gaussian, out float alpha, out float rayDepth)
    {
        float2 eval = raster_eval_alpha_depth_gaussian(gaussian, info.rayDir);
        alpha = eval.x;
        rayDepth = eval.y;
        return alpha > 0.0;
    }

    static void process_hit(RasterPixelInfo info, uint splatId, RasterGaussian gaussian, float alpha, float rayDepth, inout RasterReplayState replayState)
    {
        RasterState state = replayState.pixelState;
        float nextTrans = state.trans * (1.0 - alpha);
        if (replayState.cloneSelected && nextTrans <= replayState.cloneThreshold && state.trans > replayState.cloneThreshold)
        {
            InterlockedAdd(g_CloneCounts[splatId], 1u);
        }
        float contributionWeight = state.trans * alpha;
        if (rayDepth > 0.0) replayState.depthState = depth_state_step(replayState.depthState, contributionWeight, rayDepth);
        replayState.pixelState = state.blend(raster_forward_color(gaussian, splatId, replayState.debugMode), alpha);
        replayState.processedCount += 1u;
    }
}

struct DebugRasterReplayVariant : IRasterReplayVariant
{
    static void init_state(RasterPixelInfo info, out RasterReplayState replayState)
    {
        init_raster_debug_state(info, replayState);
    }

    static bool eval_hit(RasterPixelInfo info, uint splatId, RasterGaussian gaussian, out float alpha, out float rayDepth)
    {
        rayDepth = 0.0;
        if (g_Raster.debugMode == DEBUG_MODE_ELLIPSE_OUTLINES)
        {
            alpha = raster_eval_outline_alpha(splatId, gaussian.colorAlpha.w, info.samplePoint);
            return alpha > 0.0;
        }

        float2 eval = raster_eval_alpha_depth_gaussian(gaussian, info.rayDir);
        alpha = eval.x;
        rayDepth = eval.y;
        return alpha > 0.0;
    }

    static void process_hit(RasterPixelInfo info, uint splatId, RasterGaussian gaussian, float alpha, float rayDepth, inout RasterReplayState replayState)
    {
        uint debugMode = replayState.debugMode;
        float contributionWeight = replayState.pixelState.trans * alpha;
        if ((debugMode == DEBUG_MODE_DEPTH_MEAN || debugMode == DEBUG_MODE_DEPTH_STD) && rayDepth > 0.0) replayState.depthState = depth_state_step(replayState.depthState, contributionWeight, rayDepth);
        if (debugMode == DEBUG_MODE_SPLAT_SPATIAL_DENSITY && contributionWeight > SMALL_VALUE)
        {
            Gaussian3D originalGaussian = load_gaussian3d(splatId);
            replayState.spatialDensitySum += contributionWeight / gaussian_scale_volume(originalGaussian.logScale);
        }
        if (debugMode == DEBUG_MODE_SPLAT_SCREEN_DENSITY && contributionWeight > SMALL_VALUE)
        {
            replayState.screenDensitySum += contributionWeight / projected_ellipse_area(g_ScreenEllipseConic[splatId].xyz);
        }

        replayState.pixelState = replayState.pixelState.blend(raster_forward_color(gaussian, splatId, debugMode), alpha);
    }
}

void prefetch_batch_from_global(uint batchStart, uint batchEnd, uint lane)
{
    uint batchCount = batchEnd - batchStart;
    for (uint localIdx = lane; localIdx < batchCount; localIdx += RASTER_GROUP_THREADS)
    {
        uint splatId = g_SortedValues[batchStart + localIdx];
        gsBatchSplatId[localIdx] = splatId;
        gsGaussian[localIdx] = g_RasterCache[splatId];
    }
}

void raster_replay<T : IRasterReplayVariant>(RasterPixelInfo info, uint lane, inout RasterReplayState replayState)
{
    bool pixelDone = !info.pixelActive;
    if (lane == 0u) gsDoneCount = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (!info.pixelActive) InterlockedAdd(gsDoneCount, 1u);
    GroupMemoryBarrierWithGroupSync();

    for (uint batchStart = info.start; batchStart < info.end; batchStart += RASTER_BATCH)
    {
        if (gsDoneCount == RASTER_GROUP_THREADS) break;
        uint batchEnd = min(batchStart + RASTER_BATCH, info.end);
        uint batchCount = batchEnd - batchStart;
        replayState.replayBatchEnd = batchEnd;
        prefetch_batch_from_global(batchStart, batchEnd, lane);
        GroupMemoryBarrierWithGroupSync();

        for (uint localIdx = 0u; (localIdx < batchCount) && !pixelDone; localIdx++)
        {
            replayState.processedEnd = batchStart + localIdx + 1u;
            RasterGaussian gaussian = gsGaussian[localIdx];
            uint splatId = gsBatchSplatId[localIdx];
            float alpha = 0.0;
            float rayDepth = 0.0;
            if (!T.eval_hit(info, splatId, gaussian, alpha, rayDepth)) continue;

            T.process_hit(info, splatId, gaussian, alpha, rayDepth, replayState);
            if (replayState.pixelState.trans < g_Raster.transmittanceThreshold)
            {
                pixelDone = true;
                InterlockedAdd(gsDoneCount, 1u);
                break;
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }
}

float4 raster_forward_output_pixel(RasterPixelInfo info, RasterState state)
{
    if (!info.pixelActive) return float4(0.0, 0.0, 0.0, 1.0);
    return info.hasTileRange
        ? raster_output_from_state(state, g_Raster.background)
        : float4(pow(max(g_Raster.background, 0.0), OUTPUT_GAMMA), 1.0);
}

float4 raster_debug_output_pixel(RasterPixelInfo info, RasterReplayState replayState)
{
    if (!info.pixelActive) return float4(0.0, 0.0, 0.0, 1.0);
    uint debugMode = g_Raster.debugMode;
    uint processedCount = info.hasTileRange && replayState.processedEnd >= info.start ? (replayState.processedEnd - info.start) : 0u;
    if (debugMode == DEBUG_MODE_PROCESSED_COUNT) return float4(raster_processed_count_color(processedCount), 1.0);
    if (debugMode == DEBUG_MODE_SPLAT_SPATIAL_DENSITY) return float4(jet_colormap(debug_normalize_range(replayState.spatialDensitySum, g_Raster.debugDensityRange)), 1.0);
    if (debugMode == DEBUG_MODE_SPLAT_SCREEN_DENSITY) return float4(jet_colormap(debug_normalize_range(replayState.screenDensitySum, g_Raster.debugDensityRange)), 1.0);
    if (debugMode == DEBUG_MODE_DEPTH_MEAN) return float4(jet_colormap(debug_normalize_range(replayState.depthState.y, g_Raster.debugDepthMeanRange)), 1.0);
    if (debugMode == DEBUG_MODE_DEPTH_STD) return float4(jet_colormap(debug_normalize_range(raster_depth_output(replayState.depthState), g_Raster.debugDepthStdRange)), 1.0);
    return raster_forward_output_pixel(info, replayState.pixelState);
}

RasterState.Differential raster_output_gradient_pixel(RasterPixelInfo info, RasterState state)
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

void store_training_forward_cache(RasterPixelInfo info, RasterReplayState replayState, uint lane)
{
    if (lane == 0u) g_TrainingBatchEnd[info.tileId] = replayState.replayBatchEnd;
    if (!info.pixelActive) return;
    uint pixelIndex = raster_pixel_index(info.pixelCoord);
    g_TrainingForwardState[pixelIndex] = float4(replayState.pixelState.accum, replayState.pixelState.trans);
    g_TrainingDepthState[pixelIndex] = float4(replayState.depthState, 0.0);
    g_TrainingDepthRatio[pixelIndex] = raster_depth_output(replayState.depthState);
    g_TrainingProcessedEnd[pixelIndex] = replayState.processedEnd;
}

void load_training_forward_cache(RasterPixelInfo info, out RasterReplayState replayState)
{
    init_raster_forward_state(info, replayState);
    replayState.replayBatchEnd = g_TrainingBatchEnd[info.tileId];
    if (!info.pixelActive) return;
    uint pixelIndex = raster_pixel_index(info.pixelCoord);
    float4 cachedState = g_TrainingForwardState[pixelIndex];
    replayState.pixelState = { cachedState.xyz, cachedState.w };
    replayState.depthState = g_TrainingDepthState[pixelIndex].xyz;
    replayState.processedEnd = g_TrainingProcessedEnd[pixelIndex];
}

interface IRasterGradVariant
{
    static RasterGaussian load_cached_raster_grad(uint splatId);
    static void flush_cached_raster_grad(uint splatId, RasterGaussian gaussian, RasterGaussian.Differential dGaussian);
}

struct FixedRasterGradVariant : IRasterGradVariant
{
    static RasterGaussian load_cached_raster_grad(uint splatId)
    {
        return load_cached_raster_grad_fixed(splatId);
    }

    static void flush_cached_raster_grad(uint splatId, RasterGaussian gaussian, RasterGaussian.Differential dGaussian)
    {
        flush_cached_raster_grad_fixed(splatId, gaussian, dGaussian);
    }
}

struct FloatRasterGradVariant : IRasterGradVariant
{
    static RasterGaussian load_cached_raster_grad(uint splatId)
    {
        return load_cached_raster_grad_float(splatId);
    }

    static void flush_cached_raster_grad(uint splatId, RasterGaussian gaussian, RasterGaussian.Differential dGaussian)
    {
        flush_cached_raster_grad_float(splatId, dGaussian);
    }
}

void raster_backward_replay<T : IRasterGradVariant>(RasterPixelInfo info, inout RasterReplayState replayState, inout RasterBackwardState backwardState, uint lane)
{
    if (!info.hasTileRange)
    {
        GroupMemoryBarrierWithGroupSync();
        return;
    }

    for (uint batchBase = replayState.replayBatchEnd; batchBase > info.start;)
    {
        uint batchStart = batchBase > info.start + RASTER_BATCH ? (batchBase - RASTER_BATCH) : info.start;
        uint batchCount = batchBase - batchStart;
        prefetch_batch_from_global(batchStart, batchBase, lane);
        GroupMemoryBarrierWithGroupSync();

        for (uint rev = 0u; rev < batchCount; rev++)
        {
            uint localIdx = batchCount - 1u - rev;
            uint globalIdx = batchStart + localIdx;
            if (!(info.pixelActive && globalIdx < replayState.processedEnd)) continue;

            uint splatId = gsBatchSplatId[localIdx];
            RasterGaussian gaussian = gsGaussian[localIdx];
            float2 eval = raster_eval_alpha_depth_gaussian(gaussian, info.rayDir);
            float alpha = eval.x;
            float rayDepth = eval.y;
            if (alpha <= 0.0) continue;

            RasterState state = replayState.pixelState;
            RasterState prevState = state.undo(gaussian.colorAlpha.xyz, alpha);
            var dpPrevState = diffPair(prevState);
            var dpGaussian = diffPair(gaussian);
            var dpColorAlpha = diffPair(gaussian.colorAlpha);
            var dpAlpha = diffPair(alpha);
            bwd_diff(raster_blend_step_color_alpha)(dpPrevState, dpColorAlpha, dpAlpha, backwardState.pixelState);
            RasterState.Differential dPrevState = dpPrevState.getDifferential();
            float dAlpha = dpAlpha.getDifferential();
            float contributionWeight = prevState.trans * alpha;
            float3 prevDepthState = depth_state_rewind(replayState.depthState, contributionWeight, rayDepth);
            var dpPrevDepthState = diffPair(prevDepthState);
            var dpContributionWeight = diffPair(contributionWeight);
            var dpRayDepth = diffPair(rayDepth);
            bwd_diff(depth_state_step)(dpPrevDepthState, dpContributionWeight, dpRayDepth, backwardState.depthState);
            replayState.depthState = prevDepthState;
            backwardState.depthState = dpPrevDepthState.getDifferential();
            dPrevState.trans += dpContributionWeight.getDifferential() * alpha;
            dAlpha += dpContributionWeight.getDifferential() * prevState.trans;
            bwd_diff(raster_eval_alpha_depth_gaussian)(dpGaussian, info.rayDir, float2(dAlpha, dpRayDepth.getDifferential()));

            replayState.pixelState = prevState;
            backwardState.pixelState = dPrevState;
            RasterGaussian.Differential dGaussian = dpGaussian.getDifferential();
            dGaussian.colorAlpha += dpColorAlpha.getDifferential();
            T.flush_cached_raster_grad(splatId, gaussian, dGaussian);
        }

        GroupMemoryBarrierWithGroupSync();
        batchBase = batchStart;
    }
}

[shader("compute")]
[numthreads(RASTER_GROUP_THREADS, 1, 1)]
void csRasterize(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex)
{
    RasterPixelInfo info = resolve_raster_pixel(groupID, lane);
    RasterReplayState replayState;
    ForwardRasterReplayVariant.init_state(info, replayState);
    if (info.hasTileRange) raster_replay<ForwardRasterReplayVariant>(info, lane, replayState);
    if (!info.pixelActive) return;
    g_Output[info.pixelCoord] = raster_forward_output_pixel(info, replayState.pixelState);
}

[shader("compute")]
[numthreads(RASTER_GROUP_THREADS, 1, 1)]
void csRasterizeDebug(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex)
{
    RasterPixelInfo info = resolve_raster_pixel(groupID, lane);
    RasterReplayState replayState;
    DebugRasterReplayVariant.init_state(info, replayState);
    if (info.hasTileRange) raster_replay<DebugRasterReplayVariant>(info, lane, replayState);
    if (!info.pixelActive) return;
    g_Output[info.pixelCoord] = raster_debug_output_pixel(info, replayState);
}

void clear_raster_grads(uint3 dispatchThreadID)
{
    uint splatCount = g_Prepass.splatCount;
    uint gradCount = splatCount * TRAINABLE_PARAM_COUNT;
    uint cachedGradCount = splatCount * RASTER_CACHE_PARAM_COUNT;
    uint idx = dispatchThreadID.x;
    if (idx < gradCount) g_ParamGrads[idx] = 0.0;
    if (idx < cachedGradCount) g_CachedRasterGradsFixed[idx] = 0;
    if (idx < cachedGradCount) g_CachedRasterGradsFloat[idx] = 0.0;
}

void backprop_cached_raster_grads<T : IRasterGradVariant>(uint3 dispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_Prepass.splatCount) return;

    RasterGaussian dRaster = T.load_cached_raster_grad(splatId);
    CachedEllipsoid.Differential dCached;
    dCached.roLocal = dRaster.roLocal;
    dCached.scale = dRaster.scale;
    dCached.rotation = dRaster.rotation;

    Gaussian3D gaussian = load_gaussian3d(splatId);
    var dpCachedPosition = diffPair(gaussian.position);
    var dpCachedScale = diffPair(gaussian.logScale);
    var dpRotation = diffPair(gaussian.rotation);
    var dpOpacityPosition = diffPair(gaussian.position);
    var dpOpacityScale = diffPair(gaussian.logScale);
    var dpRawOpacity = diffPair(gaussian.colorAlpha.w);
    bwd_diff(build_cached_ellipsoid)(g_Camera, g_Prepass.radiusScale, g_MaxAnisotropy, dpCachedPosition, dpCachedScale, dpRotation, dCached);
    bwd_diff(build_raster_fused_opacity)(g_Camera, g_Prepass.radiusScale, dpOpacityPosition, dpOpacityScale, dpRawOpacity, dRaster.colorAlpha.w);

    float gradScale = g_RasterGradDecodeScale;
    store_param_grad<3>(g_Prepass.splatCount, PARAM_POSITION_X, splatId, (dpCachedPosition.getDifferential() + dpOpacityPosition.getDifferential()) * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, PARAM_SCALE_X, splatId, (dpCachedScale.getDifferential() + dpOpacityScale.getDifferential()) * gradScale);
    store_param_grad<4>(g_Prepass.splatCount, PARAM_ROTATION_X, splatId, dpRotation.getDifferential() * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, PARAM_COLOR_R, splatId, dRaster.colorAlpha.xyz * gradScale);
    store_param_grad<1>(g_Prepass.splatCount, PARAM_RAW_OPACITY, splatId, float1(dpRawOpacity.getDifferential()) * gradScale);
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csClearRasterGradsFixed(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    clear_raster_grads(dispatchThreadID);
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csClearRasterGradsFloat(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    clear_raster_grads(dispatchThreadID);
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csBackpropCachedRasterGradsFixed(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    backprop_cached_raster_grads<FixedRasterGradVariant>(dispatchThreadID);
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csBackpropCachedRasterGradsFloat(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    backprop_cached_raster_grads<FloatRasterGradVariant>(dispatchThreadID);
}

[shader("compute")]
[numthreads(RASTER_GROUP_THREADS, 1, 1)]
void csRasterizeTrainingForwardFixed(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex)
{
    RasterPixelInfo info = resolve_raster_pixel(groupID, lane);
    RasterReplayState replayState;
    TrainingRasterReplayVariant.init_state(info, replayState);
    if (info.hasTileRange) raster_replay<TrainingRasterReplayVariant>(info, lane, replayState);
    store_training_forward_cache(info, replayState, lane);
    if (!info.pixelActive) return;
    g_Output[info.pixelCoord] = raster_forward_output_pixel(info, replayState.pixelState);
}

[shader("compute")]
[numthreads(RASTER_GROUP_THREADS, 1, 1)]
void csRasterizeTrainingForwardFloat(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex)
{
    RasterPixelInfo info = resolve_raster_pixel(groupID, lane);
    RasterReplayState replayState;
    TrainingRasterReplayVariant.init_state(info, replayState);
    if (info.hasTileRange) raster_replay<TrainingRasterReplayVariant>(info, lane, replayState);
    store_training_forward_cache(info, replayState, lane);
    if (!info.pixelActive) return;
    g_Output[info.pixelCoord] = raster_forward_output_pixel(info, replayState.pixelState);
}

[shader("compute")]
[numthreads(RASTER_GROUP_THREADS, 1, 1)]
void csRasterizeBackwardFixed(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex)
{
    RasterPixelInfo info = resolve_raster_pixel(groupID, lane);
    RasterReplayState replayState;
    RasterBackwardState backwardState;
    load_training_forward_cache(info, replayState);
    init_raster_backward_state(backwardState);
    backwardState.pixelState = raster_output_gradient_pixel(info, replayState.pixelState);
    if (info.pixelActive && info.hasTileRange)
    {
        uint pixelIndex = raster_pixel_index(info.pixelCoord);
        var dpDepthState = diffPair(replayState.depthState);
        bwd_diff(raster_depth_output)(dpDepthState, g_TrainingDepthRatioGrad[pixelIndex]);
        backwardState.depthState = dpDepthState.getDifferential();
    }
    raster_backward_replay<FixedRasterGradVariant>(info, replayState, backwardState, lane);
}

[shader("compute")]
[numthreads(RASTER_GROUP_THREADS, 1, 1)]
void csRasterizeBackwardFloat(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex)
{
    RasterPixelInfo info = resolve_raster_pixel(groupID, lane);
    RasterReplayState replayState;
    RasterBackwardState backwardState;
    load_training_forward_cache(info, replayState);
    init_raster_backward_state(backwardState);
    backwardState.pixelState = raster_output_gradient_pixel(info, replayState.pixelState);
    if (info.pixelActive && info.hasTileRange)
    {
        uint pixelIndex = raster_pixel_index(info.pixelCoord);
        var dpDepthState = diffPair(replayState.depthState);
        bwd_diff(raster_depth_output)(dpDepthState, g_TrainingDepthRatioGrad[pixelIndex]);
        backwardState.depthState = dpDepthState.getDifferential();
    }
    raster_backward_replay<FloatRasterGradVariant>(info, replayState, backwardState, lane);
}
