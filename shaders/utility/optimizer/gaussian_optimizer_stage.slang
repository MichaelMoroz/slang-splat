import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/math/numeric.slang";
import "utility/optimizer/common.slang";
import "utility/math/quaternion.slang";
import "utility/random/random.slang";
import "utility/splatting/common.slang";
import "utility/splatting/opacity.slang";
import "utility/splatting/params.slang";

static const uint OPTIMIZER_GROUP_SIZE = 256u;
static const uint SH_NON_NEGATIVE_HASH_SPLAT = 0x9e3779b9u;
static const uint SH_NON_NEGATIVE_HASH_PHI = 0x85ebca6bu;
static const uint SH_NON_NEGATIVE_HASH_Z = 0xc2b2ae35u;
static const uint SH_CLAMP_SAMPLE_COUNT = 8u;
static const uint SH_CLAMP_HASH_SPLAT = 0x68e31da4u;
static const uint SH_CLAMP_HASH_SAMPLE = 0xb5297a4du;
static const uint SH_CLAMP_HASH_PHI = 0x1b56c4e9u;
static const uint SH_CLAMP_HASH_Z = 0x7f4a7c15u;

struct GaussianStabilityParams
{
    float gradComponentClip;
    float gradNormClip;
    float maxUpdate;
    float maxAnisotropy;
    float minOpacity;
    float maxOpacity;
    float positionAbsMax;
    float hugeValue;
};

uniform GaussianStabilityParams g_Stability;
uniform uint g_SplatCount;
uniform CameraParams g_CurrentCamera;
uniform uint g_EnableCurrentCameraScreenScaleCap;
uniform float g_MaxViewAngleSlope;
uniform float g_RendererRadiusScale;
uniform uint g_SHBand;
uniform uint g_SHNonNegativeStepIndex;
uniform uint g_StoredPackedParamCount;

uniform StructuredBuffer<OptimizerParamSettings> g_OptimizerParamSettings;
uniform float3 g_GaussianOptimizerRegularization;
uniform float3 g_GaussianOptimizerRegularizationCameraPosition;
uniform StructuredBuffer<uint4> g_GaussianOptimizerSplatContributionInfo;
uniform RWStructuredBuffer<float> g_SplatParamsRW;

struct TrainableGaussianState : IDifferentiable
{
    float3 position;
    float3 logScale;
    float4 rotation;
    float3 shCoeffs[SH_COEFF_COUNT];
    float rawOpacity;

    static TrainableGaussianState load(uint splatId)
    {
        TrainableGaussianState state;
        state.position = load_splat_param_vector<3>(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_X, splatId);
        state.logScale = load_splat_param_vector<3>(g_SplatParamsRW, g_SplatCount, PARAM_SCALE_X, splatId);
        state.rotation = load_splat_param_vector<4>(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_X, splatId);
        [unroll] for (uint coeffIndex = 0u; coeffIndex < SH_COEFF_COUNT; coeffIndex++)
        {
            state.shCoeffs[coeffIndex] = load_splat_sh_coeff(g_SplatParamsRW, g_SplatCount, g_StoredPackedParamCount, coeffIndex, splatId);
        }
        state.rawOpacity = load_splat_raw_opacity(g_SplatParamsRW, g_SplatCount, g_StoredPackedParamCount, splatId);
        return state;
    }

    void store(uint splatId)
    {
        store_splat_param_vector<3>(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_X, splatId, position);
        store_splat_param_vector<3>(g_SplatParamsRW, g_SplatCount, PARAM_SCALE_X, splatId, logScale);
        store_splat_param_vector<4>(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_X, splatId, rotation);
        [unroll] for (uint coeffIndex = 0u; coeffIndex < SH_COEFF_COUNT; coeffIndex++)
        {
            store_splat_sh_coeff(g_SplatParamsRW, g_SplatCount, g_StoredPackedParamCount, coeffIndex, splatId, shCoeffs[coeffIndex]);
        }
        store_splat_raw_opacity(g_SplatParamsRW, g_SplatCount, g_StoredPackedParamCount, splatId, rawOpacity);
    }
};

float3 clamp_scale_anisotropy(float3 logScale)
{
    float maxAnisotropy = max(g_Stability.maxAnisotropy, 1.0);
    float maxLogScale = max(logScale.x, max(logScale.y, logScale.z));
    float minAllowedLogScale = maxLogScale - log(maxAnisotropy);
    return max(logScale, float3(minAllowedLogScale));
}

float3 clamp_sh0_base_color(float3 sh0)
{
    return (clamp(0.5 + SH_C0 * sh0, 0.0, 1.0) - 0.5) / SH_C0;
}

float3 clamp_sh1_coeffs(float3 sh1)
{
    return clamp(sh1, -1.0, 1.0);
}

float3 clamp_sh_rest_coeffs(float3 sh)
{
    return clamp(sh, -1.0, 1.0);
}

float sh_clamp_random01(uint seed)
{
    return (float(hash_u32(seed)) + 1.0) * RNG_OPEN01_SCALE;
}

float3 sh_clamp_sample_dir(uint splatId, uint sampleId)
{
    uint splatSeed = hash_u32(((splatId + 1u) * SH_CLAMP_HASH_SPLAT) ^ ((sampleId + 1u) * SH_CLAMP_HASH_SAMPLE));
    float z = 1.0 - 2.0 * sh_clamp_random01(splatSeed ^ SH_CLAMP_HASH_Z);
    float phi = TAU_RADIANS * sh_clamp_random01(splatSeed ^ SH_CLAMP_HASH_PHI);
    float sinPhi, cosPhi;
    sincos(phi, sinPhi, cosPhi);
    float radial = sqrt(max(1.0 - z * z, 0.0));
    return float3(radial * cosPhi, radial * sinPhi, z);
}

float sh_view_dependent_scale_limit(float base, float delta)
{
    if (delta > 0.0) return (1.0 - base) / delta;
    if (delta < 0.0) return base / -delta;
    return 1.0;
}

float sh_view_dependent_scale_limit(float3 baseColor, float3 deltaColor)
{
    return min(
        sh_view_dependent_scale_limit(baseColor.x, deltaColor.x),
        min(
            sh_view_dependent_scale_limit(baseColor.y, deltaColor.y),
            sh_view_dependent_scale_limit(baseColor.z, deltaColor.z)
        )
    );
}

float sh_rescale_to_sampled_color_range(TrainableGaussianState state, no_diff uint shBand, no_diff uint splatId)
{
    if (shBand == 0u) return 1.0;
    float3 baseColor = evaluate_sh0_color_raw(state.shCoeffs[0u]);
    float scale = 1.0;
    [unroll] for (uint sampleId = 0u; sampleId < SH_CLAMP_SAMPLE_COUNT; sampleId++)
    {
        float3 sampleColor = evaluate_sh_band_raw(state.shCoeffs, shBand, sh_clamp_sample_dir(splatId, sampleId));
        scale = min(scale, sh_view_dependent_scale_limit(baseColor, sampleColor - baseColor));
    }
    return saturate(scale);
}

void scale_active_sh_coeffs(inout TrainableGaussianState state, float scale, uint activeCoeffCount)
{
    if (!(scale < 1.0) || activeCoeffCount <= 1u) return;
    [unroll] for (uint coeffIndex = 1u; coeffIndex < SH_COEFF_COUNT; coeffIndex++) if (coeffIndex < activeCoeffCount) state.shCoeffs[coeffIndex] *= scale;
}

float3 clamp_screen_scale_upper_bound(float3 position, float3 logScale)
{
    if (g_EnableCurrentCameraScreenScaleCap == 0u) return logScale;
    PinholeCamera camera = PinholeCamera.from_params(g_CurrentCamera);
    if (length(position - camera.position_world()) < g_CurrentCamera.minCameraDistance) return logScale;
    float maxSigmaSlope = g_MaxViewAngleSlope / max(g_RendererRadiusScale * GAUSSIAN_SUPPORT_SIGMA_RADIUS, SMALL_VALUE);
    if (!(maxSigmaSlope > SMALL_VALUE)) return logScale;
    float maxSigma = length(position - camera.position_world()) * maxSigmaSlope;
    float3 scale = exp(logScale);
    return log(max(min(scale, float3(maxSigma)), float3(TINY_VALUE)));
}

float sh_non_negative_random01(uint seed)
{
    return (float(hash_u32(seed)) + 1.0) * RNG_OPEN01_SCALE;
}

float3 sh_non_negative_sample_dir(uint splatId, uint stepIndex)
{
    uint splatSeed = hash_u32(stepIndex ^ ((splatId + 1u) * SH_NON_NEGATIVE_HASH_SPLAT));
    float z = 1.0 - 2.0 * sh_non_negative_random01(splatSeed ^ SH_NON_NEGATIVE_HASH_Z);
    float phi = TAU_RADIANS * sh_non_negative_random01(splatSeed ^ SH_NON_NEGATIVE_HASH_PHI);
    float sinPhi, cosPhi;
    sincos(phi, sinPhi, cosPhi);
    float radial = sqrt(max(1.0 - z * z, 0.0));
    return float3(radial * cosPhi, radial * sinPhi, z);
}

float sh_non_negative_rescale(TrainableGaussianState state, no_diff uint shBand, no_diff uint splatId, no_diff uint stepIndex)
{
    if (shBand == 0u) return 1.0;
    float3 sampleColor = evaluate_sh_band_raw(state.shCoeffs, shBand, sh_non_negative_sample_dir(splatId, stepIndex));
    if (all(sampleColor >= 0.0)) return 1.0;
    float3 baseColor = evaluate_sh0_color_raw(state.shCoeffs[0u]);
    return saturate(sh_view_dependent_scale_limit(baseColor, sampleColor - baseColor));
}

float gaussian_optimizer_relative_current_contribution(uint splatId)
{
    uint current = g_GaussianOptimizerSplatContributionInfo[splatId].x;
    uint currentMax = g_GaussianOptimizerSplatContributionInfo[0].w;
    return currentMax > 0u ? saturate(float(current) / float(currentMax)) : 0.0;
}

[shader("compute")]
[numthreads(ADAM_GROUP_SIZE, 1, 1)]
void csRegularizeGaussianPacked(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint paramIndex = dispatchThreadID.x;
    uint packedParamCount = g_SplatCount * g_StoredPackedParamCount;
    if (paramIndex >= packedParamCount) return;

    uint splatCount = max(g_SplatCount, 1u);
    uint paramId = paramIndex / splatCount;
    uint splatId = paramIndex % splatCount;
    uint rawOpacityParamId = packed_raw_opacity_param_id(g_StoredPackedParamCount);
    OptimizerParamSettings settings = g_OptimizerParamSettings[paramId];
    float value = sanitize_scalar(g_SplatParamsRW[paramIndex], g_Stability.hugeValue);

    if (paramId >= PARAM_POSITION_X && paramId <= PARAM_POSITION_Z && g_GaussianOptimizerRegularization.z > 0.0)
    {
        float3 position = load_splat_param_vector<3>(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_X, splatId);
        float3 logScale = load_splat_param_vector<3>(g_SplatParamsRW, g_SplatCount, PARAM_SCALE_X, splatId);
        float3 cameraToSplat = position - g_GaussianOptimizerRegularizationCameraPosition;
        float distance2 = dot(cameraToSplat, cameraToSplat);
        if (distance2 > TINY_VALUE * TINY_VALUE)
        {
            float avgLogScale = (sanitize_scalar(logScale.x, g_Stability.hugeValue) + sanitize_scalar(logScale.y, g_Stability.hugeValue) + sanitize_scalar(logScale.z, g_Stability.hugeValue)) * (1.0 / 3.0);
            float geoMeanScale = sanitize_scalar(exp(min(avgLogScale, log(max(g_Stability.hugeValue, 1.0)))), g_Stability.hugeValue);
            float3 pushDir = cameraToSplat * rsqrt(distance2);
            float pushStep = sanitize_scalar(geoMeanScale * gaussian_optimizer_relative_current_contribution(splatId) * g_GaussianOptimizerRegularization.z, g_Stability.hugeValue);
            uint axisIndex = paramId - PARAM_POSITION_X;
            if (axisIndex == 0u) value += pushDir.x * pushStep;
            else if (axisIndex == 1u) value += pushDir.y * pushStep;
            else value += pushDir.z * pushStep;
        }
    }

    if (paramId >= PARAM_SCALE_X && paramId <= PARAM_SCALE_Z)
    {
        if (g_GaussianOptimizerRegularization.x > 0.0) value -= exp(value) * (settings.lr * g_GaussianOptimizerRegularization.x * (1.0 / 3.0));
    } else if (paramId == rawOpacityParamId) {
        if (g_GaussianOptimizerRegularization.y > 0.0)
        {
            float opacity = opacity_from_raw(value);
            value -= settings.lr * g_GaussianOptimizerRegularization.y * opacity * (1.0 - opacity);
        }
    }

    g_SplatParamsRW[paramIndex] = clamp(sanitize_scalar(value, g_Stability.hugeValue), settings.valueMin, settings.valueMax);
}

[shader("compute")]
[numthreads(OPTIMIZER_GROUP_SIZE, 1, 1)]
void csProjectGaussianParams(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;

    TrainableGaussianState state = TrainableGaussianState.load(splatId);
    uint activeCoeffCount = min(g_SHBand == 0u ? 1u : g_SHBand == 1u ? 4u : g_SHBand == 2u ? 9u : SH_COEFF_COUNT, packed_sh_coeff_count(g_StoredPackedParamCount));
    float4 normalizedRotation = normalize_quaternion_safe(sanitize_float4(state.rotation, g_Stability.hugeValue));
    state.logScale = sanitize_float3(state.logScale, g_Stability.hugeValue);
    state.logScale = clamp_scale_anisotropy(state.logScale);
    state.logScale = clamp_screen_scale_upper_bound(state.position, state.logScale);
    state.rotation = normalizedRotation;
    state.shCoeffs[0u] = clamp_sh0_base_color(sanitize_float3(state.shCoeffs[0u], g_Stability.hugeValue));
    [unroll] for (uint coeffIndex = 1u; coeffIndex < SH_COEFF_COUNT; coeffIndex++)
    {
        float3 coeff = sanitize_float3(state.shCoeffs[coeffIndex], g_Stability.hugeValue);
        state.shCoeffs[coeffIndex] = coeffIndex < 4u ? clamp_sh1_coeffs(coeff) : clamp_sh_rest_coeffs(coeff);
    }
    scale_active_sh_coeffs(state, sh_rescale_to_sampled_color_range(state, g_SHBand, splatId), activeCoeffCount);
    scale_active_sh_coeffs(state, sh_non_negative_rescale(state, g_SHBand, splatId, g_SHNonNegativeStepIndex), activeCoeffCount);
    state.store(splatId);
}
