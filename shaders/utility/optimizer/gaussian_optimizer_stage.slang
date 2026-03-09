import "utility/losses/losses.slang";
import "utility/math/constants.slang";
import "utility/math/numeric.slang";
import "utility/math/quaternion.slang";
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
uniform float g_ScaleL2Weight;
uniform float g_ScaleAbsRegWeight;
uniform float g_OpacityRegWeight;
uniform float g_ScaleRegReference;

uniform RWStructuredBuffer<float> g_LossBuffer;
uniform RWStructuredBuffer<float> g_ParamGrads;
uniform RWStructuredBuffer<float> g_SplatParamsRW;

struct TrainableGaussianState
{
    float3 position;
    float3 scale;
    float4 rotation;
    float4 colorAlpha;

    static TrainableGaussianState load(uint splatId)
    {
        TrainableGaussianState state;
        state.position = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_X, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_Y, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_Z, splatId)
        );
        state.scale = float3(
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
        state.colorAlpha = float4(
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_COLOR_R, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_COLOR_G, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_COLOR_B, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_RAW_OPACITY, splatId)
        );
        return state;
    }

    void store(uint splatId)
    {
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_X, splatId, position.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_Y, splatId, position.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_Z, splatId, position.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SCALE_X, splatId, scale.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SCALE_Y, splatId, scale.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SCALE_Z, splatId, scale.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_X, splatId, rotation.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_Y, splatId, rotation.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_Z, splatId, rotation.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_W, splatId, rotation.w);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_COLOR_R, splatId, colorAlpha.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_COLOR_G, splatId, colorAlpha.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_COLOR_B, splatId, colorAlpha.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_RAW_OPACITY, splatId, colorAlpha.w);
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

float3 clamp_scale_anisotropy(float3 scale)
{
    float maxAnisotropy = max(g_Stability.maxAnisotropy, 1.0);
    float maxScale = max(scale.x, max(scale.y, scale.z));
    return max(scale, float3(maxScale / maxAnisotropy));
}

[shader("compute")]
[numthreads(OPTIMIZER_GROUP_SIZE, 1, 1)]
void csAccumulateRegularizationGrads(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;

    TrainableGaussianState state = TrainableGaussianState.load(splatId);
    float3 gradScale = load_param_grad3(PARAM_SCALE_X, splatId);
    float4 gradColorAlpha = load_param_grad4(PARAM_COLOR_R, splatId);
    float invSplatCount = 1.0 / max(float(g_SplatCount), 1.0);

    float scaleL2Weight = max(g_ScaleL2Weight, 0.0);
    if (scaleL2Weight > 0.0)
    {
        float scaleRegReference = max(g_ScaleRegReference, TINY_VALUE);
        var dpScaleReg = diffPair(state.scale);
        float scaleRegLoss = scale_log_reference_loss(state.scale, scaleRegReference);
        bwd_diff(scale_log_reference_loss)(dpScaleReg, scaleRegReference, scaleL2Weight * invSplatCount);
        gradScale += dpScaleReg.getDifferential();
        float lossAdd = scaleL2Weight * scaleRegLoss * invSplatCount;
        if (isfinite(lossAdd) && lossAdd != 0.0) InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], lossAdd);
    }

    float scaleAbsRegWeight = max(g_ScaleAbsRegWeight, 0.0);
    if (scaleAbsRegWeight > 0.0)
    {
        float3 signGrad = float3(state.scale.x > 0.0 ? 1.0 : state.scale.x < 0.0 ? -1.0 : 0.0, state.scale.y > 0.0 ? 1.0 : state.scale.y < 0.0 ? -1.0 : 0.0, state.scale.z > 0.0 ? 1.0 : state.scale.z < 0.0 ? -1.0 : 0.0);
        gradScale += signGrad * (scaleAbsRegWeight * invSplatCount * (1.0 / 3.0));
        float lossAdd = scaleAbsRegWeight * scale_abs_l1_loss(state.scale) * invSplatCount;
        if (isfinite(lossAdd) && lossAdd != 0.0) InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], lossAdd);
    }

    float opacityRegWeight = max(g_OpacityRegWeight, 0.0);
    if (opacityRegWeight > 0.0)
    {
        var dpOpacityReg = diffPair(state.colorAlpha.w);
        float opacityRegLoss = opacity_l1_loss(state.colorAlpha.w);
        bwd_diff(opacity_l1_loss)(dpOpacityReg, opacityRegWeight * invSplatCount);
        gradColorAlpha.w += dpOpacityReg.getDifferential();
        float lossAdd = opacityRegWeight * opacityRegLoss * invSplatCount;
        if (isfinite(lossAdd) && lossAdd != 0.0) InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], lossAdd);
    }

    store_param_grad3(PARAM_SCALE_X, splatId, gradScale);
    store_param_grad4(PARAM_COLOR_R, splatId, gradColorAlpha);
}

[shader("compute")]
[numthreads(OPTIMIZER_GROUP_SIZE, 1, 1)]
void csProjectGaussianParams(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;

    TrainableGaussianState state = TrainableGaussianState.load(splatId);
    state.scale = clamp_scale_anisotropy(sanitize_float3(state.scale, g_Stability.hugeValue));
    state.rotation = normalize_quaternion_safe(sanitize_float4(state.rotation, g_Stability.hugeValue));
    state.store(splatId);
}
