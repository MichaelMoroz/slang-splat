import "utility/math/constants.slang";
import "utility/math/numeric.slang";
import "utility/splatting/opacity.slang";
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

struct OptimizerRegularizationParams
{
    float scaleL2Weight;
    float scaleAbsWeight;
    float sh1Weight;
    float opacityWeight;
    float scaleReference;
};

static const uint ADAM_GROUP_SIZE = 256u;

uniform AdamParams g_OptimizerAdam;
uniform OptimizerRuntimeParams g_OptimizerRuntime;
uniform uint g_OptimizerElementCount;
uniform uint g_OptimizerParamCount;
uniform uint g_OptimizerParamGroupSize;
uniform uint g_OptimizerParamSettingsCount;
uniform OptimizerRegularizationParams g_OptimizerRegularization;

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

void store_optimizer_grad(uint paramId, uint elementId, float value)
{
    g_OptimizerGrads[optimizer_param_index(g_OptimizerElementCount, paramId, elementId)] = value;
}

float optimizer_shrink_abs(float value, float delta)
{
    if (!(delta > 0.0)) return value;
    if (value > 0.0) return max(value - delta, 0.0);
    if (value < 0.0) return min(value + delta, 0.0);
    return 0.0;
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
void csClipPackedParamGrads(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint elementId = dispatchThreadID.x;
    if (elementId >= g_OptimizerElementCount) return;

    [unroll] for (uint paramId = 0u; paramId < g_OptimizerParamSettingsCount; paramId++)
    {
        OptimizerParamSettings settings = g_OptimizerParamSettings[paramId];
        if (settings.groupStart != paramId) continue;
        if (settings.groupSize == 3u)
        {
            float3 grad = float3(
                sanitize_scalar(load_optimizer_grad(paramId + 0u, elementId), g_OptimizerParamSettings[paramId + 0u].gradClipAbs),
                sanitize_scalar(load_optimizer_grad(paramId + 1u, elementId), g_OptimizerParamSettings[paramId + 1u].gradClipAbs),
                sanitize_scalar(load_optimizer_grad(paramId + 2u, elementId), g_OptimizerParamSettings[paramId + 2u].gradClipAbs)
            );
            grad = clip_norm3(grad, settings.gradNormClip);
            store_optimizer_grad(paramId + 0u, elementId, grad.x);
            store_optimizer_grad(paramId + 1u, elementId, grad.y);
            store_optimizer_grad(paramId + 2u, elementId, grad.z);
        } else if (settings.groupSize == 4u) {
            float4 grad = float4(
                sanitize_scalar(load_optimizer_grad(paramId + 0u, elementId), g_OptimizerParamSettings[paramId + 0u].gradClipAbs),
                sanitize_scalar(load_optimizer_grad(paramId + 1u, elementId), g_OptimizerParamSettings[paramId + 1u].gradClipAbs),
                sanitize_scalar(load_optimizer_grad(paramId + 2u, elementId), g_OptimizerParamSettings[paramId + 2u].gradClipAbs),
                sanitize_scalar(load_optimizer_grad(paramId + 3u, elementId), g_OptimizerParamSettings[paramId + 3u].gradClipAbs)
            );
            grad = clip_norm4(grad, settings.gradNormClip);
            store_optimizer_grad(paramId + 0u, elementId, grad.x);
            store_optimizer_grad(paramId + 1u, elementId, grad.y);
            store_optimizer_grad(paramId + 2u, elementId, grad.z);
            store_optimizer_grad(paramId + 3u, elementId, grad.w);
        } else {
            [unroll] for (uint offset = 0u; offset < 4u; offset++)
            {
                if (offset >= settings.groupSize) break;
                uint groupParamId = paramId + offset;
                store_optimizer_grad(groupParamId, elementId, sanitize_scalar(load_optimizer_grad(groupParamId, elementId), g_OptimizerParamSettings[groupParamId].gradClipAbs));
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

[shader("compute")]
[numthreads(ADAM_GROUP_SIZE, 1, 1)]
void csRegularizePacked(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint paramIndex = dispatchThreadID.x;
    if (paramIndex >= g_OptimizerParamCount) return;

    uint paramGroupSize = max(g_OptimizerParamGroupSize, 1u);
    uint paramId = paramIndex / paramGroupSize;
    OptimizerParamSettings settings = g_OptimizerParamSettings[paramId];
    float value = sanitize_scalar(g_OptimizerParams[paramIndex], g_OptimizerRuntime.hugeValue);

    if (paramId >= PARAM_SCALE_X && paramId <= PARAM_SCALE_Z)
    {
        if (g_OptimizerRegularization.scaleL2Weight > 0.0)
        {
            float logReference = log(max(g_OptimizerRegularization.scaleReference, TINY_VALUE));
            value = lerp(value, logReference, saturate(settings.lr * g_OptimizerRegularization.scaleL2Weight * (2.0 / 3.0)));
        }
        if (g_OptimizerRegularization.scaleAbsWeight > 0.0) value -= exp(value) * (settings.lr * g_OptimizerRegularization.scaleAbsWeight * (1.0 / 3.0));
    } else if (paramId > PARAM_SH0_B && paramId < PARAM_RAW_OPACITY) {
        if (g_OptimizerRegularization.sh1Weight > 0.0) value = optimizer_shrink_abs(value, settings.lr * g_OptimizerRegularization.sh1Weight * (1.0 / float((SH_COEFF_COUNT - 1u) * SH_PARAM_STRIDE)));
    } else if (paramId == PARAM_RAW_OPACITY) {
        if (g_OptimizerRegularization.opacityWeight > 0.0)
        {
            float opacity = opacity_from_raw(value);
            value -= settings.lr * g_OptimizerRegularization.opacityWeight * opacity * (1.0 - opacity);
        }
    }

    g_OptimizerParams[paramIndex] = clamp(sanitize_scalar(value, g_OptimizerRuntime.hugeValue), settings.valueMin, settings.valueMax);
}
