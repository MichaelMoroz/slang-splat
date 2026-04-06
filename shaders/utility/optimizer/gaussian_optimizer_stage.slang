import "utility/losses/losses.slang";
import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/math/numeric.slang";
import "utility/math/quaternion.slang";
import "utility/splatting/common.slang";
import "utility/splatting/params.slang";

static const uint LOSS_SLOT_TOTAL = 0u;
static const uint OPTIMIZER_GROUP_SIZE = 256u;

struct GaussianStabilityParams
{
    float gradComponentClip;
    float gradNormClip;
    float maxUpdate;
    float minScale;
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
uniform float g_RadiusScale;
uniform float g_ScaleL2Weight;
uniform float g_ScaleAbsRegWeight;
uniform float g_SH1RegWeight;
uniform float g_OpacityRegWeight;
uniform float g_ScaleRegReference;

uniform RWStructuredBuffer<float> g_LossBuffer;
uniform RWStructuredBuffer<float> g_ParamGrads;
uniform RWStructuredBuffer<float> g_SplatParamsRW;

struct TrainableGaussianState
{
    float3 position;
    float3 logScale;
    float4 rotation;
    float3 sh0;
    float3 sh1X;
    float3 sh1Y;
    float3 sh1Z;
    float3 sh2_0;
    float3 sh2_1;
    float3 sh2_2;
    float3 sh2_3;
    float3 sh2_4;
    float3 sh3_0;
    float3 sh3_1;
    float3 sh3_2;
    float3 sh3_3;
    float3 sh3_4;
    float3 sh3_5;
    float3 sh3_6;
    float rawOpacity;

    static TrainableGaussianState load(uint splatId)
    {
        TrainableGaussianState state;
        state.position = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_X, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_Y, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_Z, splatId)
        );
        state.logScale = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SCALE_X, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SCALE_Y, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SCALE_Z, splatId)
        );
        state.rotation = float4(
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_X, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_Y, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_Z, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_W, splatId)
        );
        state.sh0 = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH0_R, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH0_G, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH0_B, splatId)
        );
        state.sh1X = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH1_X_R, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH1_X_G, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH1_X_B, splatId)
        );
        state.sh1Y = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH1_Y_R, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH1_Y_G, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH1_Y_B, splatId)
        );
        state.sh1Z = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH1_Z_R, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH1_Z_G, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH1_Z_B, splatId)
        );
        state.sh2_0 = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(4u, 0u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(4u, 1u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(4u, 2u), splatId)
        );
        state.sh2_1 = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(5u, 0u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(5u, 1u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(5u, 2u), splatId)
        );
        state.sh2_2 = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(6u, 0u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(6u, 1u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(6u, 2u), splatId)
        );
        state.sh2_3 = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(7u, 0u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(7u, 1u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(7u, 2u), splatId)
        );
        state.sh2_4 = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(8u, 0u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(8u, 1u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(8u, 2u), splatId)
        );
        state.sh3_0 = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(9u, 0u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(9u, 1u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(9u, 2u), splatId)
        );
        state.sh3_1 = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(10u, 0u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(10u, 1u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(10u, 2u), splatId)
        );
        state.sh3_2 = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(11u, 0u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(11u, 1u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(11u, 2u), splatId)
        );
        state.sh3_3 = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(12u, 0u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(12u, 1u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(12u, 2u), splatId)
        );
        state.sh3_4 = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(13u, 0u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(13u, 1u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(13u, 2u), splatId)
        );
        state.sh3_5 = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(14u, 0u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(14u, 1u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(14u, 2u), splatId)
        );
        state.sh3_6 = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(15u, 0u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(15u, 1u), splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(15u, 2u), splatId)
        );
        state.rawOpacity = load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_RAW_OPACITY, splatId);
        return state;
    }

    void store(uint splatId)
    {
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_X, splatId, position.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_Y, splatId, position.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_Z, splatId, position.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SCALE_X, splatId, logScale.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SCALE_Y, splatId, logScale.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SCALE_Z, splatId, logScale.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_X, splatId, rotation.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_Y, splatId, rotation.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_Z, splatId, rotation.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_W, splatId, rotation.w);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH0_R, splatId, sh0.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH0_G, splatId, sh0.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH0_B, splatId, sh0.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH1_X_R, splatId, sh1X.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH1_X_G, splatId, sh1X.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH1_X_B, splatId, sh1X.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH1_Y_R, splatId, sh1Y.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH1_Y_G, splatId, sh1Y.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH1_Y_B, splatId, sh1Y.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH1_Z_R, splatId, sh1Z.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH1_Z_G, splatId, sh1Z.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SH1_Z_B, splatId, sh1Z.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(4u, 0u), splatId, sh2_0.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(4u, 1u), splatId, sh2_0.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(4u, 2u), splatId, sh2_0.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(5u, 0u), splatId, sh2_1.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(5u, 1u), splatId, sh2_1.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(5u, 2u), splatId, sh2_1.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(6u, 0u), splatId, sh2_2.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(6u, 1u), splatId, sh2_2.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(6u, 2u), splatId, sh2_2.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(7u, 0u), splatId, sh2_3.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(7u, 1u), splatId, sh2_3.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(7u, 2u), splatId, sh2_3.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(8u, 0u), splatId, sh2_4.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(8u, 1u), splatId, sh2_4.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(8u, 2u), splatId, sh2_4.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(9u, 0u), splatId, sh3_0.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(9u, 1u), splatId, sh3_0.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(9u, 2u), splatId, sh3_0.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(10u, 0u), splatId, sh3_1.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(10u, 1u), splatId, sh3_1.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(10u, 2u), splatId, sh3_1.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(11u, 0u), splatId, sh3_2.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(11u, 1u), splatId, sh3_2.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(11u, 2u), splatId, sh3_2.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(12u, 0u), splatId, sh3_3.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(12u, 1u), splatId, sh3_3.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(12u, 2u), splatId, sh3_3.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(13u, 0u), splatId, sh3_4.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(13u, 1u), splatId, sh3_4.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(13u, 2u), splatId, sh3_4.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(14u, 0u), splatId, sh3_5.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(14u, 1u), splatId, sh3_5.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(14u, 2u), splatId, sh3_5.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(15u, 0u), splatId, sh3_6.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(15u, 1u), splatId, sh3_6.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, sh_param_id(15u, 2u), splatId, sh3_6.z);
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

float3 clamp_screen_area_upper_bound(float3 position, float3 logScale)
{
    if (g_EnableCurrentCameraScreenScaleCap == 0u) return logScale;

    PinholeCamera camera = PinholeCamera.from_params(g_CurrentCamera);
    float currentMaxScale = max(exp(logScale.x), max(exp(logScale.y), exp(logScale.z)));
    if (!(currentMaxScale > SMALL_VALUE)) return logScale;

    float maxScreenRadiusPx = sqrt(max(TRAINING_MAX_SCREEN_FRACTION * g_CurrentCamera.viewport.x * g_CurrentCamera.viewport.y / PI, SMALL_VALUE));
    float maxSupportScale;
    if (!world_support_radius_from_screen_circle(camera, position, maxScreenRadiusPx, maxSupportScale)) return logScale;

    float currentSupportScale = currentMaxScale * max(g_RadiusScale, SMALL_VALUE) * GAUSSIAN_SUPPORT_SIGMA_RADIUS;
    if (!(maxSupportScale > SMALL_VALUE) || currentSupportScale <= maxSupportScale) return logScale;

    return logScale + log(maxSupportScale / currentSupportScale);
}

[shader("compute")]
[numthreads(OPTIMIZER_GROUP_SIZE, 1, 1)]
void csAccumulateRegularizationGrads(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;

    TrainableGaussianState state = TrainableGaussianState.load(splatId);
    float3 gradScale = load_param_grad3(PARAM_SCALE_X, splatId);
    float3 gradSh1X = load_param_grad3(PARAM_SH1_X_R, splatId);
    float3 gradSh1Y = load_param_grad3(PARAM_SH1_Y_R, splatId);
    float3 gradSh1Z = load_param_grad3(PARAM_SH1_Z_R, splatId);
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
        var dpSh1X = diffPair(state.sh1X);
        var dpSh1Y = diffPair(state.sh1Y);
        var dpSh1Z = diffPair(state.sh1Z);
        float sh1RegLoss = sh1_abs_l1_loss(state.sh1X, state.sh1Y, state.sh1Z);
        bwd_diff(sh1_abs_l1_loss)(dpSh1X, dpSh1Y, dpSh1Z, sh1RegWeight * invSplatCount);
        gradSh1X += dpSh1X.getDifferential();
        gradSh1Y += dpSh1Y.getDifferential();
        gradSh1Z += dpSh1Z.getDifferential();
        float lossAdd = sh1RegWeight * sh1RegLoss * invSplatCount;
        if (isfinite(lossAdd) && lossAdd != 0.0) InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], lossAdd);
    }

    store_param_grad3(PARAM_SCALE_X, splatId, gradScale);
    store_param_grad3(PARAM_SH1_X_R, splatId, gradSh1X);
    store_param_grad3(PARAM_SH1_Y_R, splatId, gradSh1Y);
    store_param_grad3(PARAM_SH1_Z_R, splatId, gradSh1Z);
    store_param_grad(PARAM_RAW_OPACITY, splatId, gradRawOpacity);
}

[shader("compute")]
[numthreads(OPTIMIZER_GROUP_SIZE, 1, 1)]
void csProjectGaussianParams(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;

    TrainableGaussianState state = TrainableGaussianState.load(splatId);
    state.logScale = clamp_screen_area_upper_bound(state.position, sanitize_float3(state.logScale, g_Stability.hugeValue));
    state.logScale = clamp_scale_anisotropy(state.logScale);
    state.rotation = normalize_quaternion_safe(sanitize_float4(state.rotation, g_Stability.hugeValue));
    state.sh0 = clamp_sh0_base_color(sanitize_float3(state.sh0, g_Stability.hugeValue));
    state.sh1X = clamp_sh1_coeffs(sanitize_float3(state.sh1X, g_Stability.hugeValue));
    state.sh1Y = clamp_sh1_coeffs(sanitize_float3(state.sh1Y, g_Stability.hugeValue));
    state.sh1Z = clamp_sh1_coeffs(sanitize_float3(state.sh1Z, g_Stability.hugeValue));
    state.sh2_0 = clamp_sh_rest_coeffs(sanitize_float3(state.sh2_0, g_Stability.hugeValue));
    state.sh2_1 = clamp_sh_rest_coeffs(sanitize_float3(state.sh2_1, g_Stability.hugeValue));
    state.sh2_2 = clamp_sh_rest_coeffs(sanitize_float3(state.sh2_2, g_Stability.hugeValue));
    state.sh2_3 = clamp_sh_rest_coeffs(sanitize_float3(state.sh2_3, g_Stability.hugeValue));
    state.sh2_4 = clamp_sh_rest_coeffs(sanitize_float3(state.sh2_4, g_Stability.hugeValue));
    state.sh3_0 = clamp_sh_rest_coeffs(sanitize_float3(state.sh3_0, g_Stability.hugeValue));
    state.sh3_1 = clamp_sh_rest_coeffs(sanitize_float3(state.sh3_1, g_Stability.hugeValue));
    state.sh3_2 = clamp_sh_rest_coeffs(sanitize_float3(state.sh3_2, g_Stability.hugeValue));
    state.sh3_3 = clamp_sh_rest_coeffs(sanitize_float3(state.sh3_3, g_Stability.hugeValue));
    state.sh3_4 = clamp_sh_rest_coeffs(sanitize_float3(state.sh3_4, g_Stability.hugeValue));
    state.sh3_5 = clamp_sh_rest_coeffs(sanitize_float3(state.sh3_5, g_Stability.hugeValue));
    state.sh3_6 = clamp_sh_rest_coeffs(sanitize_float3(state.sh3_6, g_Stability.hugeValue));
    state.store(splatId);
}
