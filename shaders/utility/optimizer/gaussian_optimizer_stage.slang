import "utility/losses/losses.slang";
import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/math/numeric.slang";
import "utility/math/quaternion.slang";
import "utility/random/random.slang";
import "utility/splatting/common.slang";
import "utility/splatting/opacity.slang";
import "utility/splatting/params.slang";

static const uint LOSS_SLOT_TOTAL = 0u;
static const uint OPTIMIZER_GROUP_SIZE = 256u;
static const uint COLOR_NON_NEGATIVE_HASH_SPLAT = 0x9e3779b9u;
static const uint COLOR_NON_NEGATIVE_HASH_PHI = 0x85ebca6bu;
static const uint COLOR_NON_NEGATIVE_HASH_Z = 0xc2b2ae35u;
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
uniform float g_ScaleL2Weight;
uniform float g_ScaleAbsRegWeight;
uniform float g_SH1RegWeight;
uniform float g_OpacityRegWeight;
uniform float g_ColorNonNegativeRegWeight;
uniform float g_ScaleRegReference;
uniform uint g_SHBand;
uniform uint g_ColorNonNegativeRegSeed;

uniform RWStructuredBuffer<float> g_LossBuffer;
uniform RWStructuredBuffer<float> g_ParamGrads;
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

float load_param_grad(uint paramId, uint splatId)
{
    return g_ParamGrads[splat_param_index(g_SplatCount, paramId, splatId)];
}

void store_param_grad(uint paramId, uint splatId, float value)
{
    g_ParamGrads[splat_param_index(g_SplatCount, paramId, splatId)] = value;
}

float3 load_param_grad3(uint firstParamId, uint splatId)
{
    return float3(
        load_param_grad(firstParamId + 0u, splatId),
        load_param_grad(firstParamId + 1u, splatId),
        load_param_grad(firstParamId + 2u, splatId)
    );
}

float4 load_param_grad4(uint firstParamId, uint splatId)
{
    return float4(
        load_param_grad(firstParamId + 0u, splatId),
        load_param_grad(firstParamId + 1u, splatId),
        load_param_grad(firstParamId + 2u, splatId),
        load_param_grad(firstParamId + 3u, splatId)
    );
}

void store_param_grad3(uint firstParamId, uint splatId, float3 value)
{
    store_param_grad(firstParamId + 0u, splatId, value.x);
    store_param_grad(firstParamId + 1u, splatId, value.y);
    store_param_grad(firstParamId + 2u, splatId, value.z);
}

void store_param_grad4(uint firstParamId, uint splatId, float4 value)
{
    store_param_grad(firstParamId + 0u, splatId, value.x);
    store_param_grad(firstParamId + 1u, splatId, value.y);
    store_param_grad(firstParamId + 2u, splatId, value.z);
    store_param_grad(firstParamId + 3u, splatId, value.w);
}

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

[Differentiable]
float sh_rest_abs_l1_loss(TrainableGaussianState state)
{
    float3 sumAbs = 0.0;
    [unroll] for (uint coeffIndex = 1u; coeffIndex < SH_COEFF_COUNT; coeffIndex++) sumAbs += abs(state.shCoeffs[coeffIndex]);
    return dot(sumAbs, float3(1.0, 1.0, 1.0)) * (1.0 / float((SH_COEFF_COUNT - 1u) * SH_PARAM_STRIDE));
}

[Differentiable]
float color_non_negative_loss(TrainableGaussianState state, no_diff uint shBand, no_diff float3 viewDir)
{
    float3 color = evaluate_sh_band_raw(state.shCoeffs, shBand, viewDir);
    float3 negative = max(-color, 0.0);
    return dot(negative, negative);
}

float color_non_negative_random01(uint seed)
{
    return (float(hash_u32(seed)) + 1.0) * RNG_OPEN01_SCALE;
}

float3 color_non_negative_sample_dir(uint seed, uint splatId)
{
    uint splatSeed = hash_u32(seed ^ ((splatId + 1u) * COLOR_NON_NEGATIVE_HASH_SPLAT));
    float z = 1.0 - 2.0 * color_non_negative_random01(splatSeed ^ COLOR_NON_NEGATIVE_HASH_Z);
    float phi = TAU_RADIANS * color_non_negative_random01(splatSeed ^ COLOR_NON_NEGATIVE_HASH_PHI);
    float sinPhi, cosPhi;
    sincos(phi, sinPhi, cosPhi);
    float radial = sqrt(max(1.0 - z * z, 0.0));
    return float3(radial * cosPhi, radial * sinPhi, z);
}

[shader("compute")]
[numthreads(OPTIMIZER_GROUP_SIZE, 1, 1)]
void csAccumulateRegularizationGrads(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;

    TrainableGaussianState state = TrainableGaussianState.load(splatId);
    float3 gradScale = load_param_grad3(PARAM_SCALE_X, splatId);
    float3 gradSh[SH_COEFF_COUNT];
    [unroll] for (uint coeffIndex = 0u; coeffIndex < SH_COEFF_COUNT; coeffIndex++) gradSh[coeffIndex] = load_param_grad3(sh_param_id(coeffIndex, 0u), splatId);
    float gradRawOpacity = load_param_grad(PARAM_RAW_OPACITY, splatId);
    float invSplatCount = 1.0 / max(float(g_SplatCount), 1.0);

    float scaleL2Weight = max(g_ScaleL2Weight, 0.0);
    if (scaleL2Weight > 0.0)
    {
        float scaleRegReference = max(g_ScaleRegReference, TINY_VALUE);
        var dpScaleReg = diffPair(state.logScale);
        float scaleRegLoss = scale_log_reference_loss(state.logScale, scaleRegReference);
        bwd_diff(scale_log_reference_loss)(dpScaleReg, scaleRegReference, scaleL2Weight * invSplatCount);
        gradScale += dpScaleReg.getDifferential();
        float lossAdd = scaleL2Weight * scaleRegLoss * invSplatCount;
        if (isfinite(lossAdd) && lossAdd != 0.0) InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], lossAdd);
    }

    float scaleAbsRegWeight = max(g_ScaleAbsRegWeight, 0.0);
    if (scaleAbsRegWeight > 0.0)
    {
        var dpScaleAbsReg = diffPair(state.logScale);
        float scaleAbsRegLoss = scale_abs_l1_loss(state.logScale);
        bwd_diff(scale_abs_l1_loss)(dpScaleAbsReg, scaleAbsRegWeight * invSplatCount);
        gradScale += dpScaleAbsReg.getDifferential();
        float lossAdd = scaleAbsRegWeight * scaleAbsRegLoss * invSplatCount;
        if (isfinite(lossAdd) && lossAdd != 0.0) InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], lossAdd);
    }

    float opacityRegWeight = max(g_OpacityRegWeight, 0.0);
    if (opacityRegWeight > 0.0)
    {
        var dpOpacityReg = diffPair(state.rawOpacity);
        float opacityRegLoss = opacity_l1_loss(state.rawOpacity);
        bwd_diff(opacity_l1_loss)(dpOpacityReg, opacityRegWeight * invSplatCount);
        gradRawOpacity += dpOpacityReg.getDifferential();
        float lossAdd = opacityRegWeight * opacityRegLoss * invSplatCount;
        if (isfinite(lossAdd) && lossAdd != 0.0) InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], lossAdd);
    }

    float sh1RegWeight = max(g_SH1RegWeight, 0.0);
    if (sh1RegWeight > 0.0)
    {
        var dpShRest = diffPair(state);
        float sh1RegLoss = sh_rest_abs_l1_loss(state);
        bwd_diff(sh_rest_abs_l1_loss)(dpShRest, sh1RegWeight * invSplatCount);
        TrainableGaussianState.Differential dShRest = dpShRest.getDifferential();
        [unroll] for (uint coeffIndex = 1u; coeffIndex < SH_COEFF_COUNT; coeffIndex++) gradSh[coeffIndex] += dShRest.shCoeffs[coeffIndex];
        float lossAdd = sh1RegWeight * sh1RegLoss * invSplatCount;
        if (isfinite(lossAdd) && lossAdd != 0.0) InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], lossAdd);
    }

    float colorNonNegativeRegWeight = max(g_ColorNonNegativeRegWeight, 0.0);
    if (colorNonNegativeRegWeight > 0.0)
    {
        var dpColorNonNegative = diffPair(state);
        float sampleWeight = colorNonNegativeRegWeight * invSplatCount;
        float3 viewDir = color_non_negative_sample_dir(g_ColorNonNegativeRegSeed, splatId);
        float sampleLoss = color_non_negative_loss(state, g_SHBand, viewDir);
        bwd_diff(color_non_negative_loss)(dpColorNonNegative, g_SHBand, viewDir, sampleWeight);
        float lossAdd = sampleWeight * sampleLoss;
        TrainableGaussianState.Differential dColorNonNegative = dpColorNonNegative.getDifferential();
        [unroll] for (uint coeffIndex = 0u; coeffIndex < SH_COEFF_COUNT; coeffIndex++) gradSh[coeffIndex] += dColorNonNegative.shCoeffs[coeffIndex];
        if (isfinite(lossAdd) && lossAdd != 0.0) InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], lossAdd);
    }

    store_param_grad3(PARAM_SCALE_X, splatId, gradScale);
    [unroll] for (uint coeffIndex = 0u; coeffIndex < SH_COEFF_COUNT; coeffIndex++) store_param_grad3(sh_param_id(coeffIndex, 0u), splatId, gradSh[coeffIndex]);
    store_param_grad(PARAM_RAW_OPACITY, splatId, gradRawOpacity);
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
    state.store(splatId);
}
