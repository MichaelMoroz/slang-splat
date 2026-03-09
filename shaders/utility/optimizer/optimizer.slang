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
