import "utility/math/constants.slang";
import "utility/splatting/opacity.slang";

static const float SSIM_C1 = 0.0001;
static const float SSIM_C2 = 0.0009;

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

float3 rgb_l1_grad(float3 rendered, no_diff float3 target, no_diff float invPixelCount)
{
    float3 diff = rendered - target;
    float3 signGrad = float3(diff.x > 0.0 ? 1.0 : diff.x < 0.0 ? -1.0 : 0.0, diff.y > 0.0 ? 1.0 : diff.y < 0.0 ? -1.0 : 0.0, diff.z > 0.0 ? 1.0 : diff.z < 0.0 ? -1.0 : 0.0);
    return signGrad * ((1.0 / 3.0) * invPixelCount);
}

[Differentiable]
float rgb_dssim_loss(
    float3 muRendered,
    no_diff float3 muTarget,
    float3 blurRenderedSq,
    no_diff float3 blurTargetSq,
    float3 blurRenderTarget,
    no_diff float invPixelCount,
    no_diff float lambdaDSSIM
)
{
    float3 sigmaRendered = blurRenderedSq - muRendered * muRendered;
    float3 sigmaTarget = blurTargetSq - muTarget * muTarget;
    float3 sigmaCross = blurRenderTarget - muRendered * muTarget;
    float3 numer = (2.0 * muRendered * muTarget + SSIM_C1) * (2.0 * sigmaCross + SSIM_C2);
    float3 denom = max(muRendered * muRendered + muTarget * muTarget + SSIM_C1, float3(TINY_VALUE)) * max(sigmaRendered + sigmaTarget + SSIM_C2, float3(TINY_VALUE));
    float3 ssim = numer / denom;
    return lambdaDSSIM * (1.0 - dot(ssim, float3(1.0, 1.0, 1.0)) * (1.0 / 3.0)) * invPixelCount;
}

float rgb_ssim_metric(float3 muRendered, float3 muTarget, float3 blurRenderedSq, float3 blurTargetSq, float3 blurRenderTarget)
{
    float3 sigmaRendered = blurRenderedSq - muRendered * muRendered;
    float3 sigmaTarget = blurTargetSq - muTarget * muTarget;
    float3 sigmaCross = blurRenderTarget - muRendered * muTarget;
    float3 numer = (2.0 * muRendered * muTarget + SSIM_C1) * (2.0 * sigmaCross + SSIM_C2);
    float3 denom = max(muRendered * muRendered + muTarget * muTarget + SSIM_C1, float3(TINY_VALUE)) * max(sigmaRendered + sigmaTarget + SSIM_C2, float3(TINY_VALUE));
    float3 ssim = numer / denom;
    return dot(ssim, float3(1.0, 1.0, 1.0)) * (1.0 / 3.0);
}

[Differentiable]
float scale_log_reference_loss(float3 scale, no_diff float referenceScale)
{
    float3 scaleSafe = max(scale, float3(TINY_VALUE));
    float refSafe = max(referenceScale, TINY_VALUE);
    float3 logScale = log(scaleSafe / refSafe);
    return dot(logScale, logScale) * (1.0 / 3.0);
}

[Differentiable]
float opacity_l1_loss(float rawOpacity)
{
    return opacity_from_raw(rawOpacity);
}

float scale_abs_l1_loss(float3 scale)
{
    return (abs(scale.x) + abs(scale.y) + abs(scale.z)) * (1.0 / 3.0);
}
