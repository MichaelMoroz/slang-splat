import "utility/math/constants.slang";
import "utility/math/numeric.slang";
import "utility/splatting/params.slang";

struct AdamParams
{
    float beta1;
    float beta2;
    uint stepIndex;
};

static const float ADAM_EPSILON = 1e-8;

struct OptimizerRuntimeParams
{
    float gradComponentClip;
    float gradNormClip;
    float maxUpdate;
    float hugeValue;
};

struct OptimizerParamSettings
{
    float lr;
    float gradClipAbs;
    float gradNormClip;
    float valueMin;
    float valueMax;
    uint groupStart;
    uint groupSize;
    uint reserved;
};

static const uint ADAM_GROUP_SIZE = 256u;

uniform AdamParams g_OptimizerAdam;
uniform OptimizerRuntimeParams g_OptimizerRuntime;
uniform uint g_OptimizerSplatCount;
uniform uint g_OptimizerParamCount;
uniform uint g_OptimizerParamGroupSize;
uniform uint g_OptimizerParamSettingsCount;

uniform StructuredBuffer<OptimizerParamSettings> g_OptimizerParamSettings;
uniform RWStructuredBuffer<float> g_OptimizerParams;
uniform RWStructuredBuffer<float> g_OptimizerGrads;
uniform RWStructuredBuffer<float2> g_OptimizerAdamMoments;

float load_optimizer_grad(uint paramId, uint splatId)
{
    return g_OptimizerGrads[splat_param_index(g_OptimizerSplatCount, paramId, splatId)];
}

void store_optimizer_grad(uint paramId, uint splatId, float value)
{
    g_OptimizerGrads[splat_param_index(g_OptimizerSplatCount, paramId, splatId)] = value;
}

[shader("compute")]
[numthreads(ADAM_GROUP_SIZE, 1, 1)]
void csClipPackedParamGrads(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_OptimizerSplatCount) return;

    [unroll] for (uint paramId = 0u; paramId < g_OptimizerParamSettingsCount; paramId++)
    {
        OptimizerParamSettings settings = g_OptimizerParamSettings[paramId];
        if (settings.groupStart != paramId) continue;
        if (settings.groupSize == 3u)
        {
            float3 grad = float3(
                sanitize_scalar(load_optimizer_grad(paramId + 0u, splatId), g_OptimizerParamSettings[paramId + 0u].gradClipAbs),
                sanitize_scalar(load_optimizer_grad(paramId + 1u, splatId), g_OptimizerParamSettings[paramId + 1u].gradClipAbs),
                sanitize_scalar(load_optimizer_grad(paramId + 2u, splatId), g_OptimizerParamSettings[paramId + 2u].gradClipAbs)
            );
            grad = clip_norm3(grad, settings.gradNormClip);
            store_optimizer_grad(paramId + 0u, splatId, grad.x);
            store_optimizer_grad(paramId + 1u, splatId, grad.y);
            store_optimizer_grad(paramId + 2u, splatId, grad.z);
        } else if (settings.groupSize == 4u) {
            float4 grad = float4(
                sanitize_scalar(load_optimizer_grad(paramId + 0u, splatId), g_OptimizerParamSettings[paramId + 0u].gradClipAbs),
                sanitize_scalar(load_optimizer_grad(paramId + 1u, splatId), g_OptimizerParamSettings[paramId + 1u].gradClipAbs),
                sanitize_scalar(load_optimizer_grad(paramId + 2u, splatId), g_OptimizerParamSettings[paramId + 2u].gradClipAbs),
                sanitize_scalar(load_optimizer_grad(paramId + 3u, splatId), g_OptimizerParamSettings[paramId + 3u].gradClipAbs)
            );
            grad = clip_norm4(grad, settings.gradNormClip);
            store_optimizer_grad(paramId + 0u, splatId, grad.x);
            store_optimizer_grad(paramId + 1u, splatId, grad.y);
            store_optimizer_grad(paramId + 2u, splatId, grad.z);
            store_optimizer_grad(paramId + 3u, splatId, grad.w);
        } else {
            [unroll] for (uint offset = 0u; offset < 4u; offset++)
            {
                if (offset >= settings.groupSize) break;
                uint groupParamId = paramId + offset;
                store_optimizer_grad(groupParamId, splatId, sanitize_scalar(load_optimizer_grad(groupParamId, splatId), g_OptimizerParamSettings[groupParamId].gradClipAbs));
            }
        }
    }
}

[shader("compute")]
[numthreads(ADAM_GROUP_SIZE, 1, 1)]
void csAdamStepPacked(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint paramIndex = dispatchThreadID.x;
    if (paramIndex >= g_OptimizerParamCount) return;

    uint paramGroupSize = max(g_OptimizerParamGroupSize, 1u);
    uint paramId = paramIndex / paramGroupSize;
    OptimizerParamSettings settings = g_OptimizerParamSettings[paramId];
    float value = sanitize_scalar(g_OptimizerParams[paramIndex], g_OptimizerRuntime.hugeValue);
    float grad = sanitize_scalar(g_OptimizerGrads[paramIndex], settings.gradClipAbs);
    float2 moments = float2(
        sanitize_scalar(g_OptimizerAdamMoments[paramIndex].x, g_OptimizerRuntime.hugeValue),
        sanitize_scalar(g_OptimizerAdamMoments[paramIndex].y, g_OptimizerRuntime.hugeValue)
    );
    float beta1Pow = pow(g_OptimizerAdam.beta1, float(g_OptimizerAdam.stepIndex));
    float beta2Pow = pow(g_OptimizerAdam.beta2, float(g_OptimizerAdam.stepIndex));
    float biasCorr1 = 1.0 - beta1Pow;
    float biasCorr2 = 1.0 - beta2Pow;
    float oneMinusBeta1 = 1.0 - g_OptimizerAdam.beta1;
    float oneMinusBeta2 = 1.0 - g_OptimizerAdam.beta2;

    moments.x = sanitize_scalar(g_OptimizerAdam.beta1 * moments.x + oneMinusBeta1 * grad, g_OptimizerRuntime.hugeValue);
    moments.y = sanitize_scalar(g_OptimizerAdam.beta2 * moments.y + oneMinusBeta2 * (grad * grad), g_OptimizerRuntime.hugeValue);
    float mHat = moments.x / max(biasCorr1, TINY_VALUE);
    float vHat = moments.y / max(biasCorr2, TINY_VALUE);
    float update = settings.lr * mHat / (sqrt(max(vHat, 0.0)) + ADAM_EPSILON);
    update = clamp(sanitize_scalar(update, g_OptimizerRuntime.maxUpdate), -g_OptimizerRuntime.maxUpdate, g_OptimizerRuntime.maxUpdate);
    float nextValue = sanitize_scalar(value - update, g_OptimizerRuntime.hugeValue);
    g_OptimizerParams[paramIndex] = clamp(nextValue, settings.valueMin, settings.valueMax);
    g_OptimizerAdamMoments[paramIndex] = moments;
}
