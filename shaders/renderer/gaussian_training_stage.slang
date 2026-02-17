static const uint TRAIN_TILE_DIM = 16u;
static const float TRAIN_EPS = 1e-8;

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
    float minOpacity;
    float maxOpacity;
    float positionAbsMax;
    float hugeValue;
};

uniform AdamParams g_Adam;
uniform TrainingStabilityParams g_Stability;
uniform uint g_SplatCount;
uniform uint g_Width;
uniform uint g_Height;
uniform float g_InvPixelCount;
uniform float g_LossGradClip;

uniform Texture2D<float4> g_Rendered;
uniform Texture2D<float4> g_Target;
uniform RWTexture2D<float4> g_OutputGrad;
uniform RWStructuredBuffer<float> g_LossBuffer;
uniform Texture2D<float4> g_DebugRendered;
uniform Texture2D<float4> g_DebugTarget;
uniform RWTexture2D<float4> g_DebugOutput;
uniform uint g_DebugWidth;
uniform uint g_DebugHeight;
uniform Texture2D<float4> g_LetterboxSource;
uniform RWTexture2D<float4> g_LetterboxOutput;
uniform uint g_LetterboxSourceWidth;
uniform uint g_LetterboxSourceHeight;
uniform uint g_LetterboxOutputWidth;
uniform uint g_LetterboxOutputHeight;
groupshared float gsLoss[TRAIN_TILE_DIM * TRAIN_TILE_DIM];

uniform StructuredBuffer<float> g_GradPositions;
uniform StructuredBuffer<float> g_GradScales;
uniform StructuredBuffer<float> g_GradRotations;
uniform StructuredBuffer<float> g_GradColorAlpha;

uniform RWStructuredBuffer<float4> g_PositionsRW;
uniform RWStructuredBuffer<float4> g_ScalesRW;
uniform RWStructuredBuffer<float4> g_RotationsRW;
uniform RWStructuredBuffer<float4> g_ColorAlphaRW;

uniform RWStructuredBuffer<float4> g_AdamMPos;
uniform RWStructuredBuffer<float4> g_AdamVPos;
uniform RWStructuredBuffer<float4> g_AdamMScale;
uniform RWStructuredBuffer<float4> g_AdamVScale;
uniform RWStructuredBuffer<float4> g_AdamMQuat;
uniform RWStructuredBuffer<float4> g_AdamVQuat;
uniform RWStructuredBuffer<float4> g_AdamMColorAlpha;
uniform RWStructuredBuffer<float4> g_AdamVColorAlpha;

float sanitize_scalar(float value, float clampAbs)
{
    if (!isfinite(value)) return 0.0;
    return clamp(value, -clampAbs, clampAbs);
}

float3 sanitize_float3(float3 value, float clampAbs)
{
    return float3(
        sanitize_scalar(value.x, clampAbs),
        sanitize_scalar(value.y, clampAbs),
        sanitize_scalar(value.z, clampAbs)
    );
}

float4 sanitize_float4(float4 value, float clampAbs)
{
    return float4(
        sanitize_scalar(value.x, clampAbs),
        sanitize_scalar(value.y, clampAbs),
        sanitize_scalar(value.z, clampAbs),
        sanitize_scalar(value.w, clampAbs)
    );
}

float3 clip_norm3(float3 value, float maxNorm)
{
    if (maxNorm <= 0.0) return value;
    float len = length(value);
    if (!isfinite(len) || len <= maxNorm) return value;
    return value * (maxNorm / max(len, TRAIN_EPS));
}

float4 clip_norm4(float4 value, float maxNorm)
{
    if (maxNorm <= 0.0) return value;
    float len = length(value);
    if (!isfinite(len) || len <= maxNorm) return value;
    return value * (maxNorm / max(len, TRAIN_EPS));
}

float4 normalize_quaternion_safe(float4 q)
{
    float normQ = length(q);
    if (!isfinite(normQ) || normQ < TRAIN_EPS) return float4(1.0, 0.0, 0.0, 0.0);
    return q / normQ;
}

float4 adam_update_param(
    float4 param,
    float4 grad,
    inout float4 momentM,
    inout float4 momentV,
    float4 lr,
    float biasCorr1,
    float biasCorr2
)
{
    float beta1 = g_Adam.beta1;
    float beta2 = g_Adam.beta2;
    float oneMinusBeta1 = 1.0 - beta1;
    float oneMinusBeta2 = 1.0 - beta2;

    momentM = sanitize_float4(momentM, g_Stability.hugeValue);
    momentV = sanitize_float4(momentV, g_Stability.hugeValue);
    momentM = beta1 * momentM + oneMinusBeta1 * grad;
    momentV = beta2 * momentV + oneMinusBeta2 * (grad * grad);
    momentM = sanitize_float4(momentM, g_Stability.hugeValue);
    momentV = sanitize_float4(momentV, g_Stability.hugeValue);

    float4 mHat = momentM / max(biasCorr1, TRAIN_EPS);
    float4 vHat = momentV / max(biasCorr2, TRAIN_EPS);
    float4 update = lr * mHat / (sqrt(max(vHat, 0.0)) + g_Adam.epsilon);
    update = sanitize_float4(update, g_Stability.maxUpdate);
    update = clamp(update, -g_Stability.maxUpdate, g_Stability.maxUpdate);
    float4 nextParam = sanitize_float4(param - update, g_Stability.hugeValue);
    return nextParam;
}

[shader("compute")]
[numthreads(TRAIN_TILE_DIM, TRAIN_TILE_DIM, 1)]
void csClearLossAndGradTex(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px < g_Width && py < g_Height) g_OutputGrad[uint2(px, py)] = 0.0;
    if (px == 0u && py == 0u) g_LossBuffer[0] = 0.0;
}

[shader("compute")]
[numthreads(TRAIN_TILE_DIM, TRAIN_TILE_DIM, 1)]
void csComputeMSELossGrad(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID
)
{
    uint localIndex = groupThreadID.y * TRAIN_TILE_DIM + groupThreadID.x;
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;

    float localLoss = 0.0;
    if (px < g_Width && py < g_Height)
    {
        uint2 pixel = uint2(px, py);
        float3 rendered = g_Rendered[pixel].xyz;
        float3 target = g_Target[pixel].xyz;
        float3 diff = sanitize_float3(rendered - target, g_Stability.hugeValue);

        float loss = dot(diff, diff) * (1.0 / 3.0) * g_InvPixelCount;
        localLoss = isfinite(loss) ? loss : 0.0;

        float3 dOut = 2.0 * diff * (1.0 / 3.0) * g_InvPixelCount;
        dOut = sanitize_float3(dOut, g_LossGradClip);
        dOut = clamp(dOut, -g_LossGradClip, g_LossGradClip);
        g_OutputGrad[pixel] = float4(dOut, 0.0);
    } else {
        localLoss = 0.0;
    }

    gsLoss[localIndex] = localLoss;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = (TRAIN_TILE_DIM * TRAIN_TILE_DIM) >> 1; stride > 0; stride >>= 1)
    {
        if (localIndex < stride) gsLoss[localIndex] += gsLoss[localIndex + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (localIndex == 0u && isfinite(gsLoss[0])) InterlockedAdd(g_LossBuffer[0], gsLoss[0]);
}

[shader("compute")]
[numthreads(TRAIN_TILE_DIM, TRAIN_TILE_DIM, 1)]
void csComposeAbsDiffDebug(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= g_DebugWidth || py >= g_DebugHeight) return;
    uint2 pixel = uint2(px, py);
    float3 rendered = g_DebugRendered[pixel].xyz;
    float3 target = g_DebugTarget[pixel].xyz;
    float3 diff = abs(sanitize_float3(rendered - target, g_Stability.hugeValue));
    g_DebugOutput[pixel] = float4(diff, 1.0);
}

[shader("compute")]
[numthreads(TRAIN_TILE_DIM, TRAIN_TILE_DIM, 1)]
void csComposeLetterboxDebug(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= g_LetterboxOutputWidth || py >= g_LetterboxOutputHeight) return;

    float srcW = max(float(g_LetterboxSourceWidth), 1.0);
    float srcH = max(float(g_LetterboxSourceHeight), 1.0);
    float dstW = max(float(g_LetterboxOutputWidth), 1.0);
    float dstH = max(float(g_LetterboxOutputHeight), 1.0);
    float srcAspect = srcW / srcH;
    float dstAspect = dstW / dstH;

    float fitW = dstW;
    float fitH = dstH;
    if (dstAspect > srcAspect) fitW = dstH * srcAspect;
    else fitH = dstW / srcAspect;

    float offsetX = 0.5 * (dstW - fitW);
    float offsetY = 0.5 * (dstH - fitH);
    float sampleX = float(px) + 0.5;
    float sampleY = float(py) + 0.5;
    bool inside = sampleX >= offsetX && sampleX < (offsetX + fitW) && sampleY >= offsetY && sampleY < (offsetY + fitH);
    if (!inside)
    {
        g_LetterboxOutput[uint2(px, py)] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float u = (sampleX - offsetX) / max(fitW, TRAIN_EPS);
    float v = (sampleY - offsetY) / max(fitH, TRAIN_EPS);
    uint srcX = min(uint(u * srcW), g_LetterboxSourceWidth - 1u);
    uint srcY = min(uint(v * srcH), g_LetterboxSourceHeight - 1u);
    g_LetterboxOutput[uint2(px, py)] = g_LetterboxSource[uint2(srcX, srcY)];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csAdamStepFused(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;
    uint gradBase = splatId * 4u;

    float4 gradPos = float4(
        g_GradPositions[gradBase + 0u],
        g_GradPositions[gradBase + 1u],
        g_GradPositions[gradBase + 2u],
        g_GradPositions[gradBase + 3u]
    );
    float4 gradScale = float4(
        g_GradScales[gradBase + 0u],
        g_GradScales[gradBase + 1u],
        g_GradScales[gradBase + 2u],
        g_GradScales[gradBase + 3u]
    );
    float4 gradQuat = float4(
        g_GradRotations[gradBase + 0u],
        g_GradRotations[gradBase + 1u],
        g_GradRotations[gradBase + 2u],
        g_GradRotations[gradBase + 3u]
    );
    float4 gradColorAlpha = float4(
        g_GradColorAlpha[gradBase + 0u],
        g_GradColorAlpha[gradBase + 1u],
        g_GradColorAlpha[gradBase + 2u],
        g_GradColorAlpha[gradBase + 3u]
    );

    float clipAbs = g_Stability.gradComponentClip;
    gradPos = sanitize_float4(gradPos, clipAbs);
    gradScale = sanitize_float4(gradScale, clipAbs);
    gradQuat = sanitize_float4(gradQuat, clipAbs);
    gradColorAlpha = sanitize_float4(gradColorAlpha, clipAbs);
    gradPos.xyz = clip_norm3(gradPos.xyz, g_Stability.gradNormClip);
    gradScale.xyz = clip_norm3(gradScale.xyz, g_Stability.gradNormClip);
    gradQuat = clip_norm4(gradQuat, g_Stability.gradNormClip);
    gradColorAlpha = clip_norm4(gradColorAlpha, g_Stability.gradNormClip);

    float4 pos = g_PositionsRW[splatId];
    float4 scale = g_ScalesRW[splatId];
    float4 rot = g_RotationsRW[splatId];
    float4 colorAlpha = g_ColorAlphaRW[splatId];

    float beta1Pow = pow(g_Adam.beta1, float(g_Adam.stepIndex));
    float beta2Pow = pow(g_Adam.beta2, float(g_Adam.stepIndex));
    float biasCorr1 = 1.0 - beta1Pow;
    float biasCorr2 = 1.0 - beta2Pow;

    float4 mPos = g_AdamMPos[splatId];
    float4 vPos = g_AdamVPos[splatId];
    float4 mScale = g_AdamMScale[splatId];
    float4 vScale = g_AdamVScale[splatId];
    float4 mQuat = g_AdamMQuat[splatId];
    float4 vQuat = g_AdamVQuat[splatId];
    float4 mColorAlpha = g_AdamMColorAlpha[splatId];
    float4 vColorAlpha = g_AdamVColorAlpha[splatId];

    float4 posLR = float4(g_Adam.positionLR, g_Adam.positionLR, g_Adam.positionLR, 0.0);
    float4 scaleLR = float4(g_Adam.scaleLR, g_Adam.scaleLR, g_Adam.scaleLR, 0.0);
    float4 rotLR = float4(g_Adam.rotationLR, g_Adam.rotationLR, g_Adam.rotationLR, g_Adam.rotationLR);
    float4 colorAlphaLR = float4(g_Adam.colorLR, g_Adam.colorLR, g_Adam.colorLR, g_Adam.opacityLR);
    pos = adam_update_param(pos, gradPos, mPos, vPos, posLR, biasCorr1, biasCorr2);
    scale = adam_update_param(scale, gradScale, mScale, vScale, scaleLR, biasCorr1, biasCorr2);
    rot = adam_update_param(rot, gradQuat, mQuat, vQuat, rotLR, biasCorr1, biasCorr2);
    colorAlpha = adam_update_param(colorAlpha, gradColorAlpha, mColorAlpha, vColorAlpha, colorAlphaLR, biasCorr1, biasCorr2);

    pos.xyz = clamp(pos.xyz, -g_Stability.positionAbsMax, g_Stability.positionAbsMax);
    scale.xyz = clamp(scale.xyz, g_Stability.minScale, g_Stability.maxScale);
    rot = normalize_quaternion_safe(rot);
    colorAlpha.xyz = clamp(colorAlpha.xyz, 0.0, 1.0);
    colorAlpha.w = clamp(colorAlpha.w, g_Stability.minOpacity, g_Stability.maxOpacity);

    g_PositionsRW[splatId] = pos;
    g_ScalesRW[splatId] = scale;
    g_RotationsRW[splatId] = rot;
    g_ColorAlphaRW[splatId] = colorAlpha;
    g_AdamMPos[splatId] = mPos;
    g_AdamVPos[splatId] = vPos;
    g_AdamMScale[splatId] = mScale;
    g_AdamVScale[splatId] = vScale;
    g_AdamMQuat[splatId] = mQuat;
    g_AdamVQuat[splatId] = vQuat;
    g_AdamMColorAlpha[splatId] = mColorAlpha;
    g_AdamVColorAlpha[splatId] = vColorAlpha;
}
