import "math_constants.slang";

static const uint TRAIN_TILE_DIM = 16u;
static const uint LOSS_SLOT_TOTAL = 0u;
static const uint LOSS_SLOT_MSE = 1u;
static const uint RNG_LINEAR_A = 1664525u;
static const uint RNG_LINEAR_C = 1013904223u;

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

struct TrainingMCMCParams
{
    uint enabled;
    float positionNoiseScale;
    float opacityGateSharpness;
    float opacityGateCenter;
};

uniform AdamParams g_Adam;
uniform TrainingStabilityParams g_Stability;
uniform TrainingMCMCParams g_MCMC;
uniform uint g_SplatCount;
uniform uint g_LowQualityReinitEnabled;
uniform uint g_Width;
uniform uint g_Height;
uniform float g_InvPixelCount;
uniform float g_LossGradClip;
uniform float g_ScaleL2Weight;
uniform float g_ScaleAnisoWeight;

uniform Texture2D<float4> g_Rendered;
uniform Texture2D<float4> g_Target;
uniform RWTexture2D<float4> g_OutputGrad;
uniform RWStructuredBuffer<float> g_LossBuffer;
uniform RWStructuredBuffer<uint> g_SignalMaxBuffer;
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
groupshared float gsSignalMax[TRAIN_TILE_DIM * TRAIN_TILE_DIM];

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
uniform RWStructuredBuffer<uint> g_LowQualityFlags;
uniform StructuredBuffer<float4> g_InitPointPositions;
uniform StructuredBuffer<float4> g_InitPointColors;
uniform uint g_InitPointCount;
uniform uint g_InitSeed;
uniform float g_InitPositionJitterStd;
uniform float g_InitBaseScale;
uniform float g_InitScaleJitterRatio;
uniform float g_InitInitialOpacity;
uniform float g_InitColorJitterStd;

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
    return value * (maxNorm / max(len, TINY_VALUE));
}

float4 clip_norm4(float4 value, float maxNorm)
{
    if (maxNorm <= 0.0) return value;
    float len = length(value);
    if (!isfinite(len) || len <= maxNorm) return value;
    return value * (maxNorm / max(len, TINY_VALUE));
}

float4 normalize_quaternion_safe(float4 q)
{
    float normQ = length(q);
    if (!isfinite(normQ) || normQ < TINY_VALUE) return float4(1.0, 0.0, 0.0, 0.0);
    return q / normQ;
}

float3 rotate_by_quat(float3 v, float4 q)
{
    float3 qv = q.yzw;
    return v + 2.0 * cross(cross(v, qv) + q.x * v, qv);
}

uint rng_next(inout uint state)
{
    state = RNG_LINEAR_A * state + RNG_LINEAR_C;
    return state;
}

uint hash_u32(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float rng_uniform_open01(inout uint state)
{
    uint bits = rng_next(state);
    return (float(bits) + 1.0) * RNG_OPEN01_SCALE;
}

float2 rng_normal2_box_muller(inout uint state)
{
    float u1 = max(rng_uniform_open01(state), TINY_VALUE);
    float u2 = rng_uniform_open01(state);
    float radius = sqrt(max(-2.0 * log(u1), 0.0));
    float s, c;
    sincos(TAU_RADIANS * u2, s, c);
    return radius * float2(c, s);
}

float3 rng_normal3_box_muller(inout uint state)
{
    float2 n01 = rng_normal2_box_muller(state);
    float2 n23 = rng_normal2_box_muller(state);
    return float3(n01.x, n01.y, n23.x);
}

float4 rng_normal4_box_muller(inout uint state)
{
    float2 n01 = rng_normal2_box_muller(state);
    float2 n23 = rng_normal2_box_muller(state);
    return float4(n01.x, n01.y, n23.x, n23.y);
}

float opacity_noise_gate(float opacity)
{
    float x = -g_MCMC.opacityGateSharpness * (opacity - g_MCMC.opacityGateCenter);
    x = clamp(x, -SOFTPLUS_INPUT_LIMIT, SOFTPLUS_INPUT_LIMIT);
    return 1.0 / (1.0 + exp(-x));
}

[Differentiable]
float rgb_mse_loss(float3 rendered, no_diff float3 target, no_diff float invPixelCount)
{
    float3 diff = rendered - target;
    return dot(diff, diff) * (1.0 / 3.0) * invPixelCount;
}

[Differentiable]
float scale_aniso_loss(float3 scale)
{
    float3 s = max(scale, float3(TINY_VALUE));
    float meanScale = dot(s, float3(1.0 / 3.0));
    float invMeanScale = 1.0 / max(meanScale, TINY_VALUE);
    float3 relativeScale = s * invMeanScale - 1.0;
    return dot(relativeScale, relativeScale) * (1.0 / 3.0);
}

bool is_low_quality(float3 scale, float opacity)
{
    float maxSize = max(scale.x, max(scale.y, scale.z));
    return (opacity <= g_Stability.minOpacity) || (maxSize <= g_Stability.minScale);
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

    float4 mHat = momentM / max(biasCorr1, TINY_VALUE);
    float4 vHat = momentV / max(biasCorr2, TINY_VALUE);
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
    if (px == 0u && py == 0u)
    {
        g_LossBuffer[LOSS_SLOT_TOTAL] = 0.0;
        g_LossBuffer[LOSS_SLOT_MSE] = 0.0;
        g_SignalMaxBuffer[0] = 0u;
    }
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
    float localSignalMax = 0.0;
    if (px < g_Width && py < g_Height)
    {
        uint2 pixel = uint2(px, py);
        float3 rendered = sanitize_float3(g_Rendered[pixel].xyz, g_Stability.hugeValue);
        float3 target = sanitize_float3(g_Target[pixel].xyz, g_Stability.hugeValue);
        float loss = rgb_mse_loss(rendered, target, g_InvPixelCount);
        localLoss = isfinite(loss) ? loss : 0.0;
        localSignalMax = max(target.x, max(target.y, target.z));

        var dpRendered = diffPair(rendered);
        bwd_diff(rgb_mse_loss)(dpRendered, target, g_InvPixelCount, 1.0);
        float3 dOut = sanitize_float3(dpRendered.getDifferential(), g_LossGradClip);
        dOut = clamp(dOut, -g_LossGradClip, g_LossGradClip);
        g_OutputGrad[pixel] = float4(dOut, 0.0);
    }

    gsLoss[localIndex] = localLoss;
    gsSignalMax[localIndex] = isfinite(localSignalMax) ? max(localSignalMax, 0.0) : 0.0;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = (TRAIN_TILE_DIM * TRAIN_TILE_DIM) >> 1; stride > 0; stride >>= 1)
    {
        if (localIndex < stride) gsLoss[localIndex] += gsLoss[localIndex + stride];
        if (localIndex < stride) gsSignalMax[localIndex] = max(gsSignalMax[localIndex], gsSignalMax[localIndex + stride]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (localIndex == 0u)
    {
        if (isfinite(gsLoss[0]))
        {
            InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], gsLoss[0]);
            InterlockedAdd(g_LossBuffer[LOSS_SLOT_MSE], gsLoss[0]);
        }
        InterlockedMax(g_SignalMaxBuffer[0], asuint(max(gsSignalMax[0], 0.0)));
    }
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

    float u = (sampleX - offsetX) / max(fitW, TINY_VALUE);
    float v = (sampleY - offsetY) / max(fitH, TINY_VALUE);
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
    float4 pos = g_PositionsRW[splatId];
    float4 scale = g_ScalesRW[splatId];
    float4 rot = g_RotationsRW[splatId];
    float4 colorAlpha = g_ColorAlphaRW[splatId];

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

    float scaleAnisoWeight = max(g_ScaleAnisoWeight, 0.0);
    if (scaleAnisoWeight > 0.0)
    {
        var dpScaleAniso = diffPair(scale.xyz);
        float anisoLoss = scale_aniso_loss(scale.xyz);
        bwd_diff(scale_aniso_loss)(dpScaleAniso, scaleAnisoWeight);
        gradScale.xyz += dpScaleAniso.getDifferential();
        float lossAdd = scaleAnisoWeight * anisoLoss / max(float(g_SplatCount), 1.0);
        if (isfinite(lossAdd) && lossAdd != 0.0) InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], lossAdd);
    }

    float clipAbs = g_Stability.gradComponentClip;
    gradPos = sanitize_float4(gradPos, clipAbs);
    gradScale = sanitize_float4(gradScale, clipAbs);
    gradQuat = sanitize_float4(gradQuat, clipAbs);
    gradColorAlpha = sanitize_float4(gradColorAlpha, clipAbs);
    gradPos.xyz = clip_norm3(gradPos.xyz, g_Stability.gradNormClip);
    gradScale.xyz = clip_norm3(gradScale.xyz, g_Stability.gradNormClip);
    gradQuat = clip_norm4(gradQuat, g_Stability.gradNormClip);
    gradColorAlpha = clip_norm4(gradColorAlpha, g_Stability.gradNormClip);

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
    float scaleDecay = g_Adam.scaleLR * max(g_ScaleL2Weight, 0.0);
    scale.xyz = sanitize_float3(scale.xyz * max(1.0 - scaleDecay, 0.0), g_Stability.hugeValue);

    if (g_MCMC.enabled != 0u && g_MCMC.positionNoiseScale > 0.0)
    {
        float4 rotSafe = normalize_quaternion_safe(rot);
        float4 qInv = float4(rotSafe.x, -rotSafe.y, -rotSafe.z, -rotSafe.w);
        float3 scaleSafe = max(scale.xyz, float3(g_Stability.minScale));
        float gate = opacity_noise_gate(colorAlpha.w);
        uint rngSeed = splatId * 0x9e3779b9u ^ g_Adam.stepIndex * 0x85ebca6bu ^ 0xc2b2ae35u;
        uint rngState = hash_u32(rngSeed);
        if (rngState == 0u) rngState = 1u;
        float3 eta = rng_normal3_box_muller(rngState);
        float3 localNoise = scaleSafe * eta;
        float3 worldNoise = rotate_by_quat(localNoise, qInv);
        float noiseScale = g_Adam.positionLR * g_MCMC.positionNoiseScale * gate;
        float3 noise = sanitize_float3(worldNoise * noiseScale, g_Stability.maxUpdate);
        pos.xyz = sanitize_float3(pos.xyz + noise, g_Stability.hugeValue);
    }

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

[shader("compute")]
[numthreads(256, 1, 1)]
void csMarkLowQualitySplats(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;
    float3 scale = g_ScalesRW[splatId].xyz;
    float opacity = g_ColorAlphaRW[splatId].w;
    g_LowQualityFlags[splatId] = is_low_quality(scale, opacity) ? 1u : 0u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csResampleLowQualitySplatsRandom(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount || g_LowQualityReinitEnabled == 0u) return;
    if (g_LowQualityFlags[splatId] == 0u) return;

    uint rngSeed = splatId * 0x9e3779b9u ^ g_Adam.stepIndex * 0x85ebca6bu ^ 0x27d4eb2du;
    uint rngState = hash_u32(rngSeed);
    if (rngState == 0u) rngState = 1u;

    uint donorId = rng_next(rngState) % g_SplatCount;
    if (g_LowQualityFlags[donorId] != 0u) return;

    float4 donorPos = g_PositionsRW[donorId];
    float4 donorScale = g_ScalesRW[donorId];
    float4 donorRot = g_RotationsRW[donorId];
    float4 donorColorAlpha = g_ColorAlphaRW[donorId];

    float4 nextPos = sanitize_float4(donorPos, g_Stability.hugeValue);
    float3 donorScaleSafe = max(sanitize_float3(donorScale.xyz, g_Stability.hugeValue), float3(g_Stability.minScale));
    if (g_MCMC.positionNoiseScale > 0.0)
    {
        float3 eta = rng_normal3_box_muller(rngState);
        float3 jitter = sanitize_float3(donorScaleSafe * eta * g_MCMC.positionNoiseScale, g_Stability.maxUpdate);
        nextPos.xyz = sanitize_float3(nextPos.xyz + jitter, g_Stability.hugeValue);
    }

    float4 nextScale = sanitize_float4(donorScale, g_Stability.hugeValue);
    float4 nextRot = normalize_quaternion_safe(sanitize_float4(donorRot, g_Stability.hugeValue));
    float4 nextColorAlpha = sanitize_float4(donorColorAlpha, g_Stability.hugeValue);

    nextPos.xyz = clamp(nextPos.xyz, -g_Stability.positionAbsMax, g_Stability.positionAbsMax);
    nextScale.xyz = clamp(nextScale.xyz, g_Stability.minScale, g_Stability.maxScale);
    nextColorAlpha.xyz = clamp(nextColorAlpha.xyz, 0.0, 1.0);
    nextColorAlpha.w = clamp(nextColorAlpha.w, g_Stability.minOpacity, g_Stability.maxOpacity);

    g_PositionsRW[splatId] = nextPos;
    g_ScalesRW[splatId] = nextScale;
    g_RotationsRW[splatId] = nextRot;
    g_ColorAlphaRW[splatId] = nextColorAlpha;
    g_AdamMPos[splatId] = 0.0;
    g_AdamVPos[splatId] = 0.0;
    g_AdamMScale[splatId] = 0.0;
    g_AdamVScale[splatId] = 0.0;
    g_AdamMQuat[splatId] = 0.0;
    g_AdamVQuat[splatId] = 0.0;
    g_AdamMColorAlpha[splatId] = 0.0;
    g_AdamVColorAlpha[splatId] = 0.0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csInitializeGaussiansFromPointCloud(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount || g_InitPointCount == 0u) return;

    uint rngSeed = splatId * 0x9e3779b9u ^ g_InitSeed * 0x85ebca6bu ^ 0x165667b1u;
    uint rngState = hash_u32(rngSeed);
    if (rngState == 0u) rngState = 1u;

    uint pointId = rng_next(rngState) % g_InitPointCount;
    float3 pointPos = g_InitPointPositions[pointId].xyz;
    float3 pointColor = g_InitPointColors[pointId].xyz;

    float4 pos = float4(pointPos, 0.0);
    if (g_InitPositionJitterStd > 0.0)
    {
        float3 dPos = rng_normal3_box_muller(rngState) * g_InitPositionJitterStd;
        pos.xyz += dPos;
    }
    pos.xyz = clamp(sanitize_float3(pos.xyz, g_Stability.hugeValue), -g_Stability.positionAbsMax, g_Stability.positionAbsMax);

    float baseScale = max(g_InitBaseScale, 1e-4);
    float ratio = max(g_InitScaleJitterRatio, 0.0);
    float lo = max(1.0 - ratio, 1e-4);
    float hi = 1.0 + ratio;
    float3 scaleMul = float3(
        lerp(lo, hi, rng_uniform_open01(rngState)),
        lerp(lo, hi, rng_uniform_open01(rngState)),
        lerp(lo, hi, rng_uniform_open01(rngState))
    );
    float4 scale = float4(baseScale * scaleMul, 0.0);
    scale.xyz = clamp(sanitize_float3(scale.xyz, g_Stability.hugeValue), g_Stability.minScale, g_Stability.maxScale);

    float4 rot = normalize_quaternion_safe(rng_normal4_box_muller(rngState));
    float3 color = pointColor;
    if (g_InitColorJitterStd > 0.0) color += rng_normal3_box_muller(rngState) * g_InitColorJitterStd;
    color = clamp(sanitize_float3(color, g_Stability.hugeValue), 0.0, 1.0);
    float opacity = clamp(g_InitInitialOpacity, g_Stability.minOpacity, g_Stability.maxOpacity);
    float4 colorAlpha = float4(color, opacity);

    g_PositionsRW[splatId] = pos;
    g_ScalesRW[splatId] = scale;
    g_RotationsRW[splatId] = rot;
    g_ColorAlphaRW[splatId] = colorAlpha;
    g_AdamMPos[splatId] = 0.0;
    g_AdamVPos[splatId] = 0.0;
    g_AdamMScale[splatId] = 0.0;
    g_AdamVScale[splatId] = 0.0;
    g_AdamMQuat[splatId] = 0.0;
    g_AdamVQuat[splatId] = 0.0;
    g_AdamMColorAlpha[splatId] = 0.0;
    g_AdamVColorAlpha[splatId] = 0.0;
}
