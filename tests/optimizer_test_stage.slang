import "utility/math/constants.slang";
import "utility/math/numeric.slang";
import "utility/optimizer/optimizer.slang";

uniform uint g_ParamCount;
uniform StructuredBuffer<float> g_Targets;
uniform RWStructuredBuffer<float> g_Params;
uniform RWStructuredBuffer<float> g_Grads;
uniform OptimizerRuntimeParams g_Stability;

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
