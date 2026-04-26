import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/math/numeric.slang";
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
    float maxScale;
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
            state.shCoeffs[coeffIndex] = load_splat_param_vector<3>(g_SplatParamsRW, g_SplatCount, sh_param_id(coeffIndex, 0u), splatId);
        }
        state.rawOpacity = load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_RAW_OPACITY, splatId);
        return state;
    }

    void store(uint splatId)
    {
        store_splat_param_vector<3>(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_X, splatId, position);
        store_splat_param_vector<3>(g_SplatParamsRW, g_SplatCount, PARAM_SCALE_X, splatId, logScale);
        store_splat_param_vector<4>(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_X, splatId, rotation);
        [unroll] for (uint coeffIndex = 0u; coeffIndex < SH_COEFF_COUNT; coeffIndex++)
        {
            store_splat_param_vector<3>(g_SplatParamsRW, g_SplatCount, sh_param_id(coeffIndex, 0u), splatId, shCoeffs[coeffIndex]);
        }
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_RAW_OPACITY, splatId, rawOpacity);
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

void scale_active_sh_coeffs(inout TrainableGaussianState state, float scale, uint shBand)
{
    if (!(scale < 1.0) || shBand == 0u) return;
    uint activeCoeffCount = shBand == 1u ? 4u : shBand == 2u ? 9u : SH_COEFF_COUNT;
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

[shader("compute")]
[numthreads(OPTIMIZER_GROUP_SIZE, 1, 1)]
void csProjectGaussianParams(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;

    TrainableGaussianState state = TrainableGaussianState.load(splatId);
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
    scale_active_sh_coeffs(state, sh_rescale_to_sampled_color_range(state, g_SHBand, splatId), g_SHBand);
    scale_active_sh_coeffs(state, sh_non_negative_rescale(state, g_SHBand, splatId, g_SHNonNegativeStepIndex), g_SHBand);
    state.store(splatId);
}
