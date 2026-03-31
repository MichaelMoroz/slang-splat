import "utility/math/constants.slang";
import "utility/splatting/opacity.slang";

[Differentiable]
float rgb_mse_loss(float3 rendered, no_diff float3 target, no_diff float invPixelCount)
{
    float3 diff = rendered - target;
    return dot(diff, diff) * (1.0 / 3.0) * invPixelCount;
}

float rgb_l1_loss(float3 rendered, no_diff float3 target, no_diff float invPixelCount)
{
    float3 diff = abs(rendered - target);
    return dot(diff, float3(1.0, 1.0, 1.0)) * (1.0 / 3.0) * invPixelCount;
}

float3 rgb_l1_grad(float3 rendered, no_diff float3 target)
{
    float3 diff = rendered - target;
    float3 signGrad = float3(diff.x > 0.0 ? 1.0 : diff.x < 0.0 ? -1.0 : 0.0, diff.y > 0.0 ? 1.0 : diff.y < 0.0 ? -1.0 : 0.0, diff.z > 0.0 ? 1.0 : diff.z < 0.0 ? -1.0 : 0.0);
    return signGrad * (1.0 / 3.0);
}

[Differentiable]
float scale_log_reference_loss(float3 logScale, no_diff float referenceScale)
{
    float logReference = log(max(referenceScale, TINY_VALUE));
    float3 delta = logScale - float3(logReference);
    return dot(delta, delta) * (1.0 / 3.0);
}

[Differentiable]
float opacity_l1_loss(float rawOpacity)
{
    return opacity_from_raw(rawOpacity);
}

[Differentiable]
float scale_abs_l1_loss(float3 logScale)
{
    float3 scale = exp(logScale);
    return (scale.x + scale.y + scale.z) * (1.0 / 3.0);
}

[Differentiable]
float sh1_abs_l1_loss(float3 sh1X, float3 sh1Y, float3 sh1Z)
{
    float3 sumAbs = abs(sh1X) + abs(sh1Y) + abs(sh1Z);
    return dot(sumAbs, float3(1.0, 1.0, 1.0)) * (1.0 / 9.0);
}
