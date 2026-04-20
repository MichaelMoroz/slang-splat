import "gaussian_types.slang";
import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/random/random.slang";
import "utility/splatting/common.slang";
import "utility/splatting/opacity.slang";
import "utility/splatting/params.slang";
import "utility/splatting/raster.slang";

uniform float g_MaxAnisotropy;
uniform CameraParams g_TrainingNativeCamera;
uniform TrainingSubsampleParams g_TrainingSubsample;
uniform RWStructuredBuffer<uint> g_CloneCounts;
uniform RWStructuredBuffer<uint> g_SplatContribution;

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
static const float SPLAT_CONTRIBUTION_FIXED_SCALE = 256.0;
static const float SPLAT_CONTRIBUTION_FIXED_MAX_F = 16777215.99609375;
static const float DEPTH_LOCAL_MISMATCH_REJECT_ALPHA_SCALE_MAX = 2.0;
static const float DEBUG_ELLIPSE_FILL_ALPHA_SCALE = 0.25;
static const float DEBUG_ELLIPSE_OUTLINE_ALPHA_SCALE = 1.0;
static const float DEBUG_ELLIPSE_OUTLINE_ALPHA_MIN = 0.5;
static const float UINT_TO_UNIT_FLOAT = 2.3283064365386963e-10;

int raster_grad_fixed_quantize(float value, float range)
{
    float normalized = clamp(value / range, -1.0, 1.0) * RASTER_GRAD_FIXED_INT_MAX_F;
    return int(normalized >= 0.0 ? floor(normalized + 0.5) : ceil(normalized - 0.5));
}

float raster_grad_fixed_dequantize(int value, float range)
{
    return float(value) * range / RASTER_GRAD_FIXED_INT_MAX_F;
}

float3 raster_debug_signed_color(float3 value)
{
    return saturate(0.5 + value);
}

float raster_grad_fixed_avg_inv_scale(float3 scale)
{
    float avgScale = pow(max(scale.x * scale.y * scale.z, SMALL_VALUE), 1.0 / 3.0);
    return max(1.0 / max(avgScale, SMALL_VALUE), RASTER_GRAD_FIXED_AVG_INV_SCALE_FLOOR);
}

uint splat_contribution_fixed_quantize(float value)
{
    float clamped = clamp(value, 0.0, SPLAT_CONTRIBUTION_FIXED_MAX_F);
    return uint(floor(clamped * SPLAT_CONTRIBUTION_FIXED_SCALE + 0.5));
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

[Differentiable]
float raster_splat_density_weight(float trans, float alpha, float fusedOpacity)
{
    if (trans <= 0.0 || alpha <= 0.0 || fusedOpacity <= SMALL_VALUE) return 0.0;
    return sqrt(trans) * alpha / fusedOpacity;
}

[Differentiable]
float raster_splat_density_step(float state, float trans, float alpha, float fusedOpacity)
{
    return state + raster_splat_density_weight(trans, alpha, fusedOpacity);
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

float raster_normalize_threshold_band(float value, float threshold)
{
    float normalizedValue = max(value, DEBUG_GRAD_NORM_FLOOR);
    float valueLog10 = log2(normalizedValue) * LOG2_TO_LOG10;
    float normalizedThreshold = max(threshold, DEBUG_GRAD_NORM_FLOOR);
    float bandLog10Min = log2(max(normalizedThreshold * DEBUG_GRAD_THRESHOLD_MIN_SCALE, DEBUG_GRAD_NORM_FLOOR)) * LOG2_TO_LOG10;
    float bandLog10Max = log2(max(normalizedThreshold * DEBUG_GRAD_THRESHOLD_MAX_SCALE, DEBUG_GRAD_NORM_FLOOR)) * LOG2_TO_LOG10;
    return saturate((valueLog10 - bandLog10Min) / max(bandLog10Max - bandLog10Min, TINY_VALUE));
}

float raster_threshold_from_range(float2 valueRange)
{
    float minValue = max(min(valueRange.x, valueRange.y), DEBUG_GRAD_NORM_FLOOR);
    float maxValue = max(max(valueRange.x, valueRange.y), DEBUG_GRAD_NORM_FLOOR);
    float thresholdMin = minValue / DEBUG_GRAD_THRESHOLD_MIN_SCALE;
    float thresholdMax = maxValue / DEBUG_GRAD_THRESHOLD_MAX_SCALE;
    return sqrt(max(thresholdMin * thresholdMax, DEBUG_GRAD_NORM_FLOOR));
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

float3 raster_contribution_amount_color(uint contribution)
{
    float contributionPercent = float(contribution) * max(g_Raster.debugContributionPercentScale, 0.0);
    return jet_colormap(debug_normalize_log_range(contributionPercent, max(g_Raster.debugContributionRange.x, DEBUG_CONTRIBUTION_AMOUNT_FLOOR), max(g_Raster.debugContributionRange.y, DEBUG_CONTRIBUTION_AMOUNT_FLOOR)));
}

uint raster_adam_moment_index(uint splatCount, uint paramId, uint splatId)
{
    return paramId * splatCount + splatId;
}

float raster_adam_momentum_length(uint splatId)
{
    uint splatCount = g_Prepass.splatCount;
    float momentumSq = 0.0;
    [unroll] for (uint paramId = 0u; paramId < TRAINABLE_PARAM_COUNT; paramId++)
    {
        float moment = g_DebugAdamMoments[raster_adam_moment_index(splatCount, paramId, splatId)].x;
        if (isfinite(moment)) momentumSq += moment * moment;
    }
    return sqrt(max(momentumSq, 0.0));
}

float3 raster_forward_color(RasterGaussian gaussian, uint splatId, uint debugMode)
{
    if (debugMode == DEBUG_MODE_SH_VIEW_DEPENDENT) return raster_debug_signed_color(gaussian_sh_view_dependent_component(load_gaussian3d(splatId), g_Camera.camPos));
    if (debugMode == DEBUG_MODE_SH_COEFFICIENT) return raster_debug_signed_color(gaussian_sh_coeff(load_gaussian3d(splatId), min(g_Raster.debugSHCoeffIndex, DEBUG_SH_COEFF_COUNT - 1u)));
    if (debugMode == DEBUG_MODE_CLONE_COUNT) return raster_clone_count_color(g_CloneCounts[splatId]);
    if (debugMode == DEBUG_MODE_CONTRIBUTION_AMOUNT) return raster_contribution_amount_color(g_SplatContribution[splatId]);
    if (debugMode == DEBUG_MODE_ADAM_MOMENTUM) return jet_colormap(raster_normalize_threshold_band(raster_adam_momentum_length(splatId), raster_threshold_from_range(g_Raster.debugAdamMomentumRange)));
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

[Differentiable]
float4 raster_training_output_from_state(RasterState state, no_diff float3 background)
{
    float3 finalColor = state.accum + state.trans * background;
    return float4(finalColor, 1.0 - state.trans);
}

float raster_random_unit_float(uint seed)
{
    return (float(hash_u32(seed)) + 0.5) * UINT_TO_UNIT_FLOAT;
}

float3 raster_training_background(uint2 pixelCoord)
{
    uint baseSeed = hash_u32(g_Raster.trainingBackgroundSeed ^ (pixelCoord.x * 0x9e3779b9u) ^ (pixelCoord.y * 0x85ebca6bu));
    return float3(
        raster_random_unit_float(baseSeed ^ 0x243f6a88u),
        raster_random_unit_float(baseSeed ^ 0xb7e15162u),
        raster_random_unit_float(baseSeed ^ 0x13198a2eu)
    );
}

float3 raster_resolved_background(uint2 pixelCoord)
{
    return g_Raster.trainingBackgroundMode == 0u ? g_Raster.background : raster_training_background(pixelCoord);
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
    uint debugMode;
    float3 depthState;
    float splatDensitySum;
    float spatialDensitySum;
    float screenDensitySum;
    float4 depthLocalMismatchState;
};

struct RasterBackwardState
{
    RasterState.Differential pixelState;
    float3 depthState;
    float splatDensityState;
};

void init_raster_forward_state(RasterPixelInfo info, out RasterReplayState replayState)
{
    replayState.pixelState = { 0.0, 1.0 };
    replayState.processedCount = 0u;
    replayState.processedEnd = info.start;
    replayState.replayBatchEnd = info.start;
    replayState.debugMode = DEBUG_MODE_NORMAL;
    replayState.depthState = 0.0;
    replayState.splatDensitySum = 0.0;
    replayState.spatialDensitySum = 0.0;
    replayState.screenDensitySum = 0.0;
    replayState.depthLocalMismatchState = 0.0;
}

void init_raster_backward_state(out RasterBackwardState backwardState)
{
    backwardState.pixelState = { 0.0, 0.0 };
    backwardState.depthState = 0.0;
    backwardState.splatDensityState = 0.0;
}

void init_raster_debug_state(RasterPixelInfo info, out RasterReplayState replayState)
{
    replayState.pixelState = { 0.0, 1.0 };
    replayState.processedCount = 0u;
    replayState.processedEnd = info.start;
    replayState.replayBatchEnd = info.start;
    replayState.debugMode = g_Raster.debugMode;
    replayState.depthState = 0.0;
    replayState.splatDensitySum = 0.0;
    replayState.spatialDensitySum = 0.0;
    replayState.screenDensitySum = 0.0;
    replayState.depthLocalMismatchState = 0.0;
}

void init_training_raster_state(RasterPixelInfo info, out RasterReplayState replayState)
{
    replayState.pixelState = { 0.0, 1.0 };
    replayState.processedCount = 0u;
    replayState.processedEnd = info.start;
    replayState.replayBatchEnd = info.start;
    replayState.debugMode = DEBUG_MODE_NORMAL;
    replayState.depthState = 0.0;
    replayState.splatDensitySum = 0.0;
    replayState.spatialDensitySum = 0.0;
    replayState.screenDensitySum = 0.0;
    replayState.depthLocalMismatchState = 0.0;
}

float raster_depth_debug_sigma_scale()
{
    return max(g_Prepass.radiusScale * GAUSSIAN_SUPPORT_SIGMA_RADIUS, SMALL_VALUE);
}

float raster_support_ray_scale(RasterGaussian gaussian, float3 rayDir)
{
    float3 rayLocal = world_to_local_dir(rayDir, gaussian.scale, gaussian.rotation);
    return rsqrt(max(dot(rayLocal, rayLocal), SMALL_VALUE));
}

float raster_sigma_ray(RasterGaussian gaussian, float3 rayDir)
{
    return raster_support_ray_scale(gaussian, rayDir) / raster_depth_debug_sigma_scale();
}

float raster_mean_sigma(RasterGaussian gaussian)
{
    return max((gaussian.scale.x + gaussian.scale.y + gaussian.scale.z) / 3.0, SMALL_VALUE) / raster_depth_debug_sigma_scale();
}

void depth_local_mismatch_step(inout float4 state, float contributionWeight, float depth, float sigmaRay, float meanSigma, float alpha)
{
    if (!(contributionWeight > SMALL_VALUE) || !(depth > 0.0)) return;

    float estimateWeight = state.x;
    if (!(estimateWeight > SMALL_VALUE))
    {
        state.x = contributionWeight;
        state.y = depth;
        return;
    }

    float estimateDepth = state.y;
    float delta = abs(depth - estimateDepth);
    float smoothDist = max(g_Raster.debugDepthLocalMismatchSmoothRadius, 0.0) * max(sigmaRay, SMALL_VALUE);
    float alphaT = saturate(alpha);
    float alphaSmooth = alphaT * alphaT * (3.0 - 2.0 * alphaT);
    float rejectAlphaScale = 1.0 + (DEPTH_LOCAL_MISMATCH_REJECT_ALPHA_SCALE_MAX - 1.0) * alphaSmooth;
    float rejectDist = max(g_Raster.debugDepthLocalMismatchRejectRadius, 0.0) * max(meanSigma, SMALL_VALUE) * rejectAlphaScale;
    rejectDist = max(rejectDist, smoothDist + SMALL_VALUE);
    float u = saturate((delta - smoothDist) / max(rejectDist - smoothDist, SMALL_VALUE));
    float gate = 1.0 - u * u * (3.0 - 2.0 * u);
    float prevEstimateWeight = estimateWeight * gate;
    float nextEstimateWeight = prevEstimateWeight + contributionWeight;
    state.y = (prevEstimateWeight * estimateDepth + contributionWeight * depth) / max(nextEstimateWeight, SMALL_VALUE);
    state.x = nextEstimateWeight;
    float mismatchWeight = contributionWeight * gate;
    state.z += mismatchWeight;
    state.w += mismatchWeight * delta;
}

float raster_depth_local_mismatch_output(float4 state)
{
    float meanMismatch = state.w / max(state.z, SMALL_VALUE);
    return meanMismatch / max(abs(state.y), SMALL_VALUE);
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
    info.samplePoint = float2(info.pixelCoord);

    if (g_TrainingSubsample.enabled != 0u)
    {
        PinholeCamera camera = PinholeCamera.from_params(g_TrainingNativeCamera);
        uint2 samplePixel = training_sample_native_pixel(info.pixelCoord, g_TrainingSubsample);
        info.rayDir = camera.screen_to_world_ray(float2(samplePixel));
    } else
    {
        PinholeCamera camera = PinholeCamera.from_params(g_Camera);
        uint samplePx = min(info.pixelCoord.x, g_Raster.width - 1u);
        uint samplePy = min(info.pixelCoord.y, g_Raster.height - 1u);
        info.rayDir = camera.screen_to_world_ray(float2(samplePx, samplePy));
    }

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
        float2 eval = raster_eval_alpha_depth_gaussian(gaussian, info.rayDir);
        alpha = eval.x;
        rayDepth = eval.y;
        return alpha > 0.0;
    }

    static void process_hit(RasterPixelInfo info, uint splatId, RasterGaussian gaussian, float alpha, float rayDepth, inout RasterReplayState replayState)
    {
        RasterState state = replayState.pixelState;
        replayState.pixelState = state.blend(raster_forward_color(gaussian, splatId, replayState.debugMode), alpha);
        replayState.processedCount += 1u;
    }
}

struct TrainingRasterReplayVariant : IRasterReplayVariant
{
    static void init_state(RasterPixelInfo info, out RasterReplayState replayState)
    {
        init_training_raster_state(info, replayState);
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
        float contributionWeight = state.trans * alpha;
        float3 color = raster_forward_color(gaussian, splatId, replayState.debugMode);
        uint contribution = splat_contribution_fixed_quantize(contributionWeight);
        if (contribution > 0u) InterlockedAdd(g_SplatContribution[splatId], contribution);
        if (rayDepth > 0.0) replayState.depthState = depth_state_step(replayState.depthState, contributionWeight, rayDepth);
        replayState.splatDensitySum = raster_splat_density_step(replayState.splatDensitySum, state.trans, alpha, gaussian.colorAlpha.w);
        replayState.pixelState = state.blend(color, alpha);
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
        float2 eval = raster_eval_alpha_depth_gaussian(gaussian, info.rayDir);
        alpha = eval.x;
        rayDepth = eval.y;
        return alpha > 0.0;
    }

    static void process_hit(RasterPixelInfo info, uint splatId, RasterGaussian gaussian, float alpha, float rayDepth, inout RasterReplayState replayState)
    {
        uint debugMode = replayState.debugMode;
        float trans = replayState.pixelState.trans;
        float contributionWeight = smoothstep(0.0, 0.5, trans) * smoothstep(1.0, 0.5, trans) * alpha;
        if ((debugMode == DEBUG_MODE_DEPTH_MEAN || debugMode == DEBUG_MODE_DEPTH_STD) && rayDepth > 0.0) replayState.depthState = depth_state_step(replayState.depthState, contributionWeight, rayDepth);
        if (debugMode == DEBUG_MODE_DEPTH_LOCAL_MISMATCH && rayDepth > 0.0)
        {
            depth_local_mismatch_step(replayState.depthLocalMismatchState, contributionWeight, rayDepth, raster_sigma_ray(gaussian, info.rayDir), raster_mean_sigma(gaussian), alpha);
        }
        if (debugMode == DEBUG_MODE_SPLAT_DENSITY) replayState.splatDensitySum = raster_splat_density_step(replayState.splatDensitySum, trans, alpha, gaussian.colorAlpha.w);
        if (debugMode == DEBUG_MODE_SPLAT_SPATIAL_DENSITY && contributionWeight > SMALL_VALUE)
        {
            Gaussian3D originalGaussian = load_gaussian3d(splatId);
            replayState.spatialDensitySum += contributionWeight / gaussian_scale_volume(originalGaussian.logScale);
        }
        if (debugMode == DEBUG_MODE_SPLAT_SCREEN_DENSITY && contributionWeight > SMALL_VALUE)
        {
            replayState.screenDensitySum += contributionWeight / projected_ellipse_area(g_ScreenEllipseConic[splatId].xyz);
        }

        float3 color = raster_forward_color(gaussian, splatId, debugMode);
        if (debugMode == DEBUG_MODE_ELLIPSE_OUTLINES)
        {
            replayState.pixelState = replayState.pixelState.blend(color, DEBUG_ELLIPSE_FILL_ALPHA_SCALE * alpha);
            float outlineAlpha = raster_eval_outline_alpha(splatId, gaussian.colorAlpha.w, info.samplePoint);
            if (outlineAlpha > 0.0) outlineAlpha = saturate(max(DEBUG_ELLIPSE_OUTLINE_ALPHA_MIN, DEBUG_ELLIPSE_OUTLINE_ALPHA_SCALE * outlineAlpha));
            if (outlineAlpha > 0.0) replayState.pixelState = replayState.pixelState.blend(color, outlineAlpha);
            return;
        }
        replayState.pixelState = replayState.pixelState.blend(color, alpha);
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
    float3 background = raster_resolved_background(info.pixelCoord);
    return info.hasTileRange
        ? raster_output_from_state(state, background)
        : float4(pow(max(background, 0.0), OUTPUT_GAMMA), 1.0);
}

float4 raster_training_output_pixel(RasterPixelInfo info, RasterState state)
{
    if (!info.pixelActive) return float4(0.0, 0.0, 0.0, 1.0);
    float3 background = raster_resolved_background(info.pixelCoord);
    return info.hasTileRange
        ? raster_training_output_from_state(state, background)
        : float4(background, 1.0);
}

float4 raster_debug_output_pixel(RasterPixelInfo info, RasterReplayState replayState)
{
    if (!info.pixelActive) return float4(0.0, 0.0, 0.0, 1.0);
    uint debugMode = g_Raster.debugMode;
    uint processedCount = info.hasTileRange && replayState.processedEnd >= info.start ? (replayState.processedEnd - info.start) : 0u;
    if (debugMode == DEBUG_MODE_PROCESSED_COUNT) return float4(raster_processed_count_color(processedCount), 1.0);
    if (debugMode == DEBUG_MODE_SPLAT_DENSITY) return float4(jet_colormap(debug_normalize_range(replayState.splatDensitySum, g_Raster.debugDensityRange)), 1.0);
    if (debugMode == DEBUG_MODE_SPLAT_SPATIAL_DENSITY) return float4(jet_colormap(debug_normalize_range(replayState.spatialDensitySum, g_Raster.debugDensityRange)), 1.0);
    if (debugMode == DEBUG_MODE_SPLAT_SCREEN_DENSITY) return float4(jet_colormap(debug_normalize_range(replayState.screenDensitySum, g_Raster.debugDensityRange)), 1.0);
    if (debugMode == DEBUG_MODE_DEPTH_MEAN) return float4(jet_colormap(debug_normalize_range(replayState.depthState.y, g_Raster.debugDepthMeanRange)), 1.0);
    if (debugMode == DEBUG_MODE_DEPTH_STD) return float4(jet_colormap(debug_normalize_range(raster_depth_output(replayState.depthState), g_Raster.debugDepthStdRange)), 1.0);
    if (debugMode == DEBUG_MODE_DEPTH_LOCAL_MISMATCH) return float4(jet_colormap(debug_normalize_range(raster_depth_local_mismatch_output(replayState.depthLocalMismatchState), g_Raster.debugDepthLocalMismatchRange)), 1.0);
    return raster_forward_output_pixel(info, replayState.pixelState);
}

RasterState.Differential raster_output_gradient_pixel(RasterPixelInfo info, RasterState state)
{
    if (!info.pixelActive || !info.hasTileRange) return { 0.0, 0.0 };

    var dpState = diffPair(state);
    bwd_diff(raster_training_output_from_state)(dpState, raster_resolved_background(info.pixelCoord), g_OutputGrad[info.pixelCoord.y * g_Raster.width + info.pixelCoord.x]);
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
    g_TrainingDepthStats[info.pixelCoord] = float4(replayState.depthState, raster_depth_output(replayState.depthState));
    g_TrainingDensity[pixelIndex] = replayState.splatDensitySum;
    g_TrainingProcessedEnd[pixelIndex] = replayState.processedEnd;
}

void load_training_forward_cache(RasterPixelInfo info, out RasterReplayState replayState)
{
    init_training_raster_state(info, replayState);
    replayState.replayBatchEnd = g_TrainingBatchEnd[info.tileId];
    if (!info.pixelActive) return;
    uint pixelIndex = raster_pixel_index(info.pixelCoord);
    float4 cachedState = g_TrainingForwardState[pixelIndex];
    float4 cachedDepthStats = g_TrainingDepthStats[info.pixelCoord];
    replayState.pixelState = { cachedState.xyz, cachedState.w };
    replayState.depthState = cachedDepthStats.xyz;
    replayState.splatDensitySum = g_TrainingDensity[pixelIndex];
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
            float densityWeight = raster_splat_density_weight(prevState.trans, alpha, gaussian.colorAlpha.w);
            float prevDensityState = replayState.splatDensitySum - densityWeight;
            var dpPrevDensityState = diffPair(prevDensityState);
            var dpDensityTrans = diffPair(prevState.trans);
            var dpDensityAlpha = diffPair(alpha);
            var dpDensityOpacity = diffPair(gaussian.colorAlpha.w);
            bwd_diff(raster_splat_density_step)(dpPrevDensityState, dpDensityTrans, dpDensityAlpha, dpDensityOpacity, backwardState.splatDensityState);
            float contributionWeight = prevState.trans * alpha;
            float3 prevDepthState = depth_state_rewind(replayState.depthState, contributionWeight, rayDepth);
            var dpPrevDepthState = diffPair(prevDepthState);
            var dpContributionWeight = diffPair(contributionWeight);
            var dpRayDepth = diffPair(rayDepth);
            bwd_diff(depth_state_step)(dpPrevDepthState, dpContributionWeight, dpRayDepth, backwardState.depthState);
            replayState.splatDensitySum = prevDensityState;
            replayState.depthState = prevDepthState;
            backwardState.splatDensityState = dpPrevDensityState.getDifferential();
            backwardState.depthState = dpPrevDepthState.getDifferential();
            dPrevState.trans += dpDensityTrans.getDifferential();
            dAlpha += dpDensityAlpha.getDifferential();
            float dContributionWeight = dpContributionWeight.getDifferential();
            dPrevState.trans += dContributionWeight * alpha;
            dAlpha += dContributionWeight * prevState.trans;
            bwd_diff(raster_eval_alpha_depth_gaussian)(dpGaussian, info.rayDir, float2(dAlpha, dpRayDepth.getDifferential()));

            replayState.pixelState = prevState;
            backwardState.pixelState = dPrevState;
            RasterGaussian.Differential dGaussian = dpGaussian.getDifferential();
            dGaussian.colorAlpha += dpColorAlpha.getDifferential();
            dGaussian.colorAlpha.w += dpDensityOpacity.getDifferential();
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
    var dpRawOpacity = diffPair(gaussian.rawOpacity);
    var dpColorPosition = diffPair(gaussian.position);
    var dpSh0 = diffPair(gaussian.sh0);
    var dpSh1X = diffPair(gaussian.sh1X);
    var dpSh1Y = diffPair(gaussian.sh1Y);
    var dpSh1Z = diffPair(gaussian.sh1Z);
    var dpSh2_0 = diffPair(gaussian.sh2_0);
    var dpSh2_1 = diffPair(gaussian.sh2_1);
    var dpSh2_2 = diffPair(gaussian.sh2_2);
    var dpSh2_3 = diffPair(gaussian.sh2_3);
    var dpSh2_4 = diffPair(gaussian.sh2_4);
    var dpSh3_0 = diffPair(gaussian.sh3_0);
    var dpSh3_1 = diffPair(gaussian.sh3_1);
    var dpSh3_2 = diffPair(gaussian.sh3_2);
    var dpSh3_3 = diffPair(gaussian.sh3_3);
    var dpSh3_4 = diffPair(gaussian.sh3_4);
    var dpSh3_5 = diffPair(gaussian.sh3_5);
    var dpSh3_6 = diffPair(gaussian.sh3_6);
    bwd_diff(build_cached_ellipsoid)(g_Camera, g_Prepass.radiusScale, g_MaxAnisotropy, dpCachedPosition, dpCachedScale, dpRotation, dCached);
    bwd_diff(build_raster_fused_opacity)(g_Camera, g_Prepass.radiusScale, dpOpacityPosition, dpOpacityScale, dpRawOpacity, dRaster.colorAlpha.w);
    bwd_diff(evaluate_gaussian_color)(
        dpSh0,
        dpSh1X,
        dpSh1Y,
        dpSh1Z,
        dpSh2_0,
        dpSh2_1,
        dpSh2_2,
        dpSh2_3,
        dpSh2_4,
        dpSh3_0,
        dpSh3_1,
        dpSh3_2,
        dpSh3_3,
        dpSh3_4,
        dpSh3_5,
        dpSh3_6,
        g_Camera.camPos,
        dpColorPosition,
        g_Raster.shBand,
        dRaster.colorAlpha.xyz
    );
    float gradScale = g_RasterGradDecodeScale;
    store_param_grad<3>(g_Prepass.splatCount, PARAM_POSITION_X, splatId, (dpCachedPosition.getDifferential() + dpOpacityPosition.getDifferential() + dpColorPosition.getDifferential()) * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, PARAM_SCALE_X, splatId, (dpCachedScale.getDifferential() + dpOpacityScale.getDifferential()) * gradScale);
    store_param_grad<4>(g_Prepass.splatCount, PARAM_ROTATION_X, splatId, dpRotation.getDifferential() * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, PARAM_SH0_R, splatId, dpSh0.getDifferential() * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, PARAM_SH1_X_R, splatId, dpSh1X.getDifferential() * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, PARAM_SH1_Y_R, splatId, dpSh1Y.getDifferential() * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, PARAM_SH1_Z_R, splatId, dpSh1Z.getDifferential() * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, sh_param_id(4u, 0u), splatId, dpSh2_0.getDifferential() * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, sh_param_id(5u, 0u), splatId, dpSh2_1.getDifferential() * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, sh_param_id(6u, 0u), splatId, dpSh2_2.getDifferential() * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, sh_param_id(7u, 0u), splatId, dpSh2_3.getDifferential() * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, sh_param_id(8u, 0u), splatId, dpSh2_4.getDifferential() * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, sh_param_id(9u, 0u), splatId, dpSh3_0.getDifferential() * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, sh_param_id(10u, 0u), splatId, dpSh3_1.getDifferential() * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, sh_param_id(11u, 0u), splatId, dpSh3_2.getDifferential() * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, sh_param_id(12u, 0u), splatId, dpSh3_3.getDifferential() * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, sh_param_id(13u, 0u), splatId, dpSh3_4.getDifferential() * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, sh_param_id(14u, 0u), splatId, dpSh3_5.getDifferential() * gradScale);
    store_param_grad<3>(g_Prepass.splatCount, sh_param_id(15u, 0u), splatId, dpSh3_6.getDifferential() * gradScale);
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
    g_Output[info.pixelCoord] = raster_training_output_pixel(info, replayState.pixelState);
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
    g_Output[info.pixelCoord] = raster_training_output_pixel(info, replayState.pixelState);
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
        float2 regularizerGrad = g_TrainingRegularizerGrad[raster_pixel_index(info.pixelCoord)];
        backwardState.splatDensityState = regularizerGrad.x;
        var dpDepthState = diffPair(replayState.depthState);
        bwd_diff(raster_depth_output)(dpDepthState, regularizerGrad.y);
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
        float2 regularizerGrad = g_TrainingRegularizerGrad[raster_pixel_index(info.pixelCoord)];
        backwardState.splatDensityState = regularizerGrad.x;
        var dpDepthState = diffPair(replayState.depthState);
        bwd_diff(raster_depth_output)(dpDepthState, regularizerGrad.y);
        backwardState.depthState = dpDepthState.getDifferential();
    }
    raster_backward_replay<FloatRasterGradVariant>(info, replayState, backwardState, lane);
}
