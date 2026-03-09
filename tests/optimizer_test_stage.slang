import "utility/math/constants.slang";
import "utility/math/numeric.slang";
import "utility/optimizer/optimizer.slang";

uniform uint g_ParamCount;
uniform StructuredBuffer<float> g_Targets;
uniform StructuredBuffer<float> g_ParamLRs;
uniform RWStructuredBuffer<float> g_Params;
uniform RWStructuredBuffer<float> g_Grads;
uniform RWStructuredBuffer<float> g_AdamM;
uniform RWStructuredBuffer<float> g_AdamV;
uniform AdamParams g_Adam;
uniform TrainingStabilityParams g_Stability;

[Differentiable]
float quadratic_loss(float value, no_diff float target)
{
    float diff = value - target;
    return 0.5 * diff * diff;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csComputeQuadraticGrad(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint paramId = dispatchThreadID.x;
    if (paramId >= g_ParamCount) return;

    float value = g_Params[paramId];
    var dpValue = diffPair(value);
    bwd_diff(quadratic_loss)(dpValue, g_Targets[paramId], 1.0 / max(float(g_ParamCount), 1.0));
    g_Grads[paramId] = sanitize_scalar(dpValue.getDifferential(), g_Stability.gradComponentClip);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csAdamStepGeneric(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint paramId = dispatchThreadID.x;
    if (paramId >= g_ParamCount) return;

    float value = g_Params[paramId];
    float grad = sanitize_scalar(g_Grads[paramId], g_Stability.gradComponentClip);
    float momentM = g_AdamM[paramId];
    float momentV = g_AdamV[paramId];
    float beta1Pow = pow(g_Adam.beta1, float(g_Adam.stepIndex));
    float beta2Pow = pow(g_Adam.beta2, float(g_Adam.stepIndex));
    float biasCorr1 = 1.0 - beta1Pow;
    float biasCorr2 = 1.0 - beta2Pow;

    g_Params[paramId] = adam_update_param_scalar(value, grad, momentM, momentV, g_ParamLRs[paramId], biasCorr1, biasCorr2, g_Adam, g_Stability);
    g_AdamM[paramId] = momentM;
    g_AdamV[paramId] = momentV;
}
