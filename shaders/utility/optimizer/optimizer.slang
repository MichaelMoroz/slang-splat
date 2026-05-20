import "utility/math/constants.slang";
import "utility/math/numeric.slang";
import "utility/optimizer/common.slang";

uniform AdamParams g_OptimizerAdam;
uniform OptimizerRuntimeParams g_OptimizerRuntime;
uniform uint g_OptimizerElementCount;
uniform uint g_OptimizerParamCount;
uniform uint g_OptimizerParamGroupSize;
uniform uint g_OptimizerParamSettingsCount;

uniform StructuredBuffer<OptimizerParamSettings> g_OptimizerParamSettings;
uniform RWStructuredBuffer<float> g_OptimizerParams;
uniform RWStructuredBuffer<float> g_OptimizerGrads;
uniform RWStructuredBuffer<float2> g_OptimizerAdamMoments;
uniform RWStructuredBuffer<float> g_OptimizerElementGradNorms;

uint optimizer_param_index(uint elementCount, uint paramId, uint elementId)
{
    return paramId * elementCount + elementId;
}

float load_optimizer_grad(uint paramId, uint elementId)
{
    return g_OptimizerGrads[optimizer_param_index(g_OptimizerElementCount, paramId, elementId)];
}

float optimizer_shrink_abs(float value, float delta)
{
    if (!(delta > 0.0)) return value;
    if (value > 0.0) return max(value - delta, 0.0);
    if (value < 0.0) return min(value + delta, 0.0);
    return 0.0;
}

float optimizer_shrink_abs_toward(float value, float target, float delta)
{
    return optimizer_shrink_abs(value - target, delta) + target;
}

[shader("compute")]
[numthreads(ADAM_GROUP_SIZE, 1, 1)]
void csComputePackedElementGradNorms(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint elementId = dispatchThreadID.x;
    if (elementId >= g_OptimizerElementCount) return;

    float gradNorm2 = 0.0;
    [unroll] for (uint paramId = 0u; paramId < g_OptimizerParamSettingsCount; paramId++)
    {
        float grad = sanitize_scalar(load_optimizer_grad(paramId, elementId), g_OptimizerParamSettings[paramId].gradClipAbs);
        gradNorm2 += grad * grad;
    }
    g_OptimizerElementGradNorms[elementId] = sqrt(max(gradNorm2, 0.0));
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
    if (settings.regularizeTowardWeight > 0.0)
    {
        nextValue = lerp(nextValue, settings.regularizeTowardValue, saturate(settings.lr * settings.regularizeTowardWeight));
    }
    if (settings.regularizeL1Weight > 0.0)
    {
        nextValue = optimizer_shrink_abs_toward(nextValue, settings.regularizeTowardValue, settings.lr * settings.regularizeL1Weight);
    }
    g_OptimizerParams[paramIndex] = clamp(sanitize_scalar(nextValue, g_OptimizerRuntime.hugeValue), settings.valueMin, settings.valueMax);
    g_OptimizerAdamMoments[paramIndex] = moments;
}
