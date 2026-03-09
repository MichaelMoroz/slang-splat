import "utility/math/constants.slang";
import "utility/math/numeric.slang";

struct AdamParams
{
    float positionLR;
    float scaleLR;
    float rotationLR;
    float colorLR;
    float opacityLR;
    float beta1;
    float beta2;
    float epsilon;
    uint stepIndex;
};

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

float4 adam_update_param(
    float4 param,
    float4 grad,
    inout float4 momentM,
    inout float4 momentV,
    float4 lr,
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

    momentM = sanitize_float4(momentM, stability.hugeValue);
    momentV = sanitize_float4(momentV, stability.hugeValue);
    momentM = beta1 * momentM + oneMinusBeta1 * grad;
    momentV = beta2 * momentV + oneMinusBeta2 * (grad * grad);
    momentM = sanitize_float4(momentM, stability.hugeValue);
    momentV = sanitize_float4(momentV, stability.hugeValue);

    float4 mHat = momentM / max(biasCorr1, TINY_VALUE);
    float4 vHat = momentV / max(biasCorr2, TINY_VALUE);
    float4 update = lr * mHat / (sqrt(max(vHat, 0.0)) + adam.epsilon);
    update = sanitize_float4(update, stability.maxUpdate);
    update = clamp(update, -stability.maxUpdate, stability.maxUpdate);
    return sanitize_float4(param - update, stability.hugeValue);
}
