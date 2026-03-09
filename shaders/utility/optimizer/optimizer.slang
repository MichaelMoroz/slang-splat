import "utility/math/constants.slang";
import "utility/math/numeric.slang";

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
uniform uint g_OptimizerParamCount;
uniform uint g_OptimizerParamGroupSize;

uniform StructuredBuffer<float> g_OptimizerParamLRs;
uniform RWStructuredBuffer<float> g_OptimizerParams;
uniform RWStructuredBuffer<float> g_OptimizerGrads;
uniform RWStructuredBuffer<float> g_OptimizerAdamM;
uniform RWStructuredBuffer<float> g_OptimizerAdamV;

float adam_update_param_scalar(
    float param,
    float grad,
    inout float momentM,
    inout float momentV,
    float lr,
    float biasCorr1,
    float biasCorr2,
    AdamParams adam,
    TrainingStabilityParams stability
)
{
    float beta1 = adam.beta1;
    float beta2 = adam.beta2;
    float oneMinusBeta1 = 1.0 - beta1;
    float oneMinusBeta2 = 1.0 - beta2;

    momentM = sanitize_scalar(momentM, stability.hugeValue);
    momentV = sanitize_scalar(momentV, stability.hugeValue);
    momentM = beta1 * momentM + oneMinusBeta1 * grad;
    momentV = beta2 * momentV + oneMinusBeta2 * (grad * grad);
    momentM = sanitize_scalar(momentM, stability.hugeValue);
    momentV = sanitize_scalar(momentV, stability.hugeValue);

    float mHat = momentM / max(biasCorr1, TINY_VALUE);
    float vHat = momentV / max(biasCorr2, TINY_VALUE);
    float update = lr * mHat / (sqrt(max(vHat, 0.0)) + ADAM_EPSILON);
    update = sanitize_scalar(update, stability.maxUpdate);
    update = clamp(update, -stability.maxUpdate, stability.maxUpdate);
    return sanitize_scalar(param - update, stability.hugeValue);
}

[shader("compute")]
[numthreads(ADAM_GROUP_SIZE, 1, 1)]
void csAdamStepPacked(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint paramIndex = dispatchThreadID.x;
    if (paramIndex >= g_OptimizerParamCount) return;

    uint paramGroupSize = max(g_OptimizerParamGroupSize, 1u);
    uint paramId = paramIndex / paramGroupSize;
    float value = g_OptimizerParams[paramIndex];
    float grad = g_OptimizerGrads[paramIndex];
    float momentM = g_OptimizerAdamM[paramIndex];
    float momentV = g_OptimizerAdamV[paramIndex];
    float beta1Pow = pow(g_OptimizerAdam.beta1, float(g_OptimizerAdam.stepIndex));
    float beta2Pow = pow(g_OptimizerAdam.beta2, float(g_OptimizerAdam.stepIndex));
    float biasCorr1 = 1.0 - beta1Pow;
    float biasCorr2 = 1.0 - beta2Pow;

    g_OptimizerParams[paramIndex] = adam_update_param_scalar(value, grad, momentM, momentV, g_OptimizerParamLRs[paramId], biasCorr1, biasCorr2, g_OptimizerAdam, g_OptimizerStability);
    g_OptimizerAdamM[paramIndex] = momentM;
    g_OptimizerAdamV[paramIndex] = momentV;
}
