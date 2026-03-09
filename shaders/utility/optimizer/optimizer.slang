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

struct TrainingStabilityParams
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

static const uint ADAM_GROUP_SIZE = 256u;

uniform AdamParams g_OptimizerAdam;
uniform TrainingStabilityParams g_OptimizerStability;
uniform uint g_OptimizerSplatCount;
uniform uint g_OptimizerParamCount;
uniform uint g_OptimizerParamGroupCount;
uniform uint g_OptimizerParamGroupSize;

uniform StructuredBuffer<float> g_OptimizerParamLRs;
uniform StructuredBuffer<float> g_OptimizerParamGradClipAbs;
uniform StructuredBuffer<float> g_OptimizerParamGradNormClip;
uniform StructuredBuffer<float> g_OptimizerParamValueMin;
uniform StructuredBuffer<float> g_OptimizerParamValueMax;
uniform StructuredBuffer<uint> g_OptimizerParamGroupStarts;
uniform StructuredBuffer<uint> g_OptimizerParamGroupSizes;
uniform RWStructuredBuffer<float> g_OptimizerParams;
uniform RWStructuredBuffer<float> g_OptimizerGrads;
uniform RWStructuredBuffer<float> g_OptimizerAdamM;
uniform RWStructuredBuffer<float> g_OptimizerAdamV;

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

    [unroll] for (uint groupIndex = 0u; groupIndex < g_OptimizerParamGroupCount; groupIndex++)
    {
        uint firstParamId = g_OptimizerParamGroupStarts[groupIndex];
        uint groupSize = g_OptimizerParamGroupSizes[groupIndex];
        float normClip = g_OptimizerParamGradNormClip[firstParamId];
        if (groupSize == 3u)
        {
            float3 grad = float3(
                sanitize_scalar(load_optimizer_grad(firstParamId + 0u, splatId), g_OptimizerParamGradClipAbs[firstParamId + 0u]),
                sanitize_scalar(load_optimizer_grad(firstParamId + 1u, splatId), g_OptimizerParamGradClipAbs[firstParamId + 1u]),
                sanitize_scalar(load_optimizer_grad(firstParamId + 2u, splatId), g_OptimizerParamGradClipAbs[firstParamId + 2u])
            );
            grad = clip_norm3(grad, normClip);
            store_optimizer_grad(firstParamId + 0u, splatId, grad.x);
            store_optimizer_grad(firstParamId + 1u, splatId, grad.y);
            store_optimizer_grad(firstParamId + 2u, splatId, grad.z);
        }
        else if (groupSize == 4u)
        {
            float4 grad = float4(
                sanitize_scalar(load_optimizer_grad(firstParamId + 0u, splatId), g_OptimizerParamGradClipAbs[firstParamId + 0u]),
                sanitize_scalar(load_optimizer_grad(firstParamId + 1u, splatId), g_OptimizerParamGradClipAbs[firstParamId + 1u]),
                sanitize_scalar(load_optimizer_grad(firstParamId + 2u, splatId), g_OptimizerParamGradClipAbs[firstParamId + 2u]),
                sanitize_scalar(load_optimizer_grad(firstParamId + 3u, splatId), g_OptimizerParamGradClipAbs[firstParamId + 3u])
            );
            grad = clip_norm4(grad, normClip);
            store_optimizer_grad(firstParamId + 0u, splatId, grad.x);
            store_optimizer_grad(firstParamId + 1u, splatId, grad.y);
            store_optimizer_grad(firstParamId + 2u, splatId, grad.z);
            store_optimizer_grad(firstParamId + 3u, splatId, grad.w);
        }
        else
        {
            [unroll] for (uint offset = 0u; offset < 4u; offset++)
            {
                if (offset >= groupSize) break;
                uint paramId = firstParamId + offset;
                store_optimizer_grad(paramId, splatId, sanitize_scalar(load_optimizer_grad(paramId, splatId), g_OptimizerParamGradClipAbs[paramId]));
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
    float value = sanitize_scalar(g_OptimizerParams[paramIndex], g_OptimizerStability.hugeValue);
    float grad = sanitize_scalar(g_OptimizerGrads[paramIndex], g_OptimizerParamGradClipAbs[paramId]);
    float momentM = sanitize_scalar(g_OptimizerAdamM[paramIndex], g_OptimizerStability.hugeValue);
    float momentV = sanitize_scalar(g_OptimizerAdamV[paramIndex], g_OptimizerStability.hugeValue);
    float beta1Pow = pow(g_OptimizerAdam.beta1, float(g_OptimizerAdam.stepIndex));
    float beta2Pow = pow(g_OptimizerAdam.beta2, float(g_OptimizerAdam.stepIndex));
    float biasCorr1 = 1.0 - beta1Pow;
    float biasCorr2 = 1.0 - beta2Pow;
    float oneMinusBeta1 = 1.0 - g_OptimizerAdam.beta1;
    float oneMinusBeta2 = 1.0 - g_OptimizerAdam.beta2;

    momentM = sanitize_scalar(g_OptimizerAdam.beta1 * momentM + oneMinusBeta1 * grad, g_OptimizerStability.hugeValue);
    momentV = sanitize_scalar(g_OptimizerAdam.beta2 * momentV + oneMinusBeta2 * (grad * grad), g_OptimizerStability.hugeValue);
    float mHat = momentM / max(biasCorr1, TINY_VALUE);
    float vHat = momentV / max(biasCorr2, TINY_VALUE);
    float update = g_OptimizerParamLRs[paramId] * mHat / (sqrt(max(vHat, 0.0)) + ADAM_EPSILON);
    update = clamp(sanitize_scalar(update, g_OptimizerStability.maxUpdate), -g_OptimizerStability.maxUpdate, g_OptimizerStability.maxUpdate);
    float nextValue = sanitize_scalar(value - update, g_OptimizerStability.hugeValue);
    g_OptimizerParams[paramIndex] = clamp(nextValue, g_OptimizerParamValueMin[paramId], g_OptimizerParamValueMax[paramId]);
    g_OptimizerAdamM[paramIndex] = momentM;
    g_OptimizerAdamV[paramIndex] = momentV;
}
