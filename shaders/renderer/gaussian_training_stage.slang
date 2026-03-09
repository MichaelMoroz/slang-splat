import "utility/losses/losses.slang";
import "utility/optimizer/optimizer.slang";
import "utility/math/constants.slang";
import "utility/math/numeric.slang";
import "utility/math/quaternion.slang";
import "utility/splatting/opacity.slang";
import "utility/splatting/params.slang";

static const uint TRAIN_TILE_DIM = 16u;
static const uint LOSS_SLOT_TOTAL = 0u;
static const uint LOSS_SLOT_MSE = 1u;

uniform AdamParams g_Adam;
uniform TrainingStabilityParams g_Stability;
uniform uint g_SplatCount;
uniform uint g_Width;
uniform uint g_Height;
uniform float g_InvPixelCount;
uniform float g_LossGradClip;
uniform float g_ScaleL2Weight;
uniform float g_ScaleAbsRegWeight;
uniform float g_OpacityRegWeight;
uniform float g_ScaleRegReference;

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
groupshared float gsMSE[TRAIN_TILE_DIM * TRAIN_TILE_DIM];

uniform StructuredBuffer<float> g_ParamGrads;
uniform RWStructuredBuffer<float> g_SplatParamsRW;
uniform RWStructuredBuffer<float> g_AdamM;
uniform RWStructuredBuffer<float> g_AdamV;
uniform StructuredBuffer<float> g_ParamLRs;

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
    }
}

[shader("compute")]
[numthreads(TRAIN_TILE_DIM, TRAIN_TILE_DIM, 1)]
void csComputeL1LossGrad(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID
)
{
    uint localIndex = groupThreadID.y * TRAIN_TILE_DIM + groupThreadID.x;
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;

    float localLoss = 0.0;
    float localMSE = 0.0;
    if (px < g_Width && py < g_Height)
    {
        uint2 pixel = uint2(px, py);
        float3 rendered = sanitize_float3(g_Rendered[pixel].xyz, g_Stability.hugeValue);
        float3 target = sanitize_float3(g_Target[pixel].xyz, g_Stability.hugeValue);
        float loss = rgb_l1_loss(rendered, target, g_InvPixelCount);
        float mse = rgb_mse_loss(rendered, target, g_InvPixelCount);
        localLoss = isfinite(loss) ? loss : 0.0;
        localMSE = isfinite(mse) ? mse : 0.0;
        float3 dOut = clamp(sanitize_float3(rgb_l1_grad(rendered, target, g_InvPixelCount), g_LossGradClip), -g_LossGradClip, g_LossGradClip);
        g_OutputGrad[pixel] = float4(dOut, 0.0);
    }

    gsLoss[localIndex] = localLoss;
    gsMSE[localIndex] = localMSE;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = (TRAIN_TILE_DIM * TRAIN_TILE_DIM) >> 1; stride > 0; stride >>= 1)
    {
        if (localIndex < stride) gsLoss[localIndex] += gsLoss[localIndex + stride];
        if (localIndex < stride) gsMSE[localIndex] += gsMSE[localIndex + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (localIndex == 0u)
    {
        if (isfinite(gsLoss[0])) InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], gsLoss[0]);
        if (isfinite(gsMSE[0])) InterlockedAdd(g_LossBuffer[LOSS_SLOT_MSE], gsMSE[0]);
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

struct TrainableGaussianState
{
    float3 position;
    float3 scale;
    float4 rotation;
    float4 colorAlpha;

    static TrainableGaussianState load(uint splatId)
    {
        TrainableGaussianState state;
        state.position = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_X, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_Y, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_Z, splatId)
        );
        state.scale = float3(
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SCALE_X, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SCALE_Y, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SCALE_Z, splatId)
        );
        state.rotation = float4(
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_X, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_Y, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_Z, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_W, splatId)
        );
        state.colorAlpha = float4(
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_COLOR_R, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_COLOR_G, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_COLOR_B, splatId),
            load_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_RAW_OPACITY, splatId)
        );
        return state;
    }

    void store(uint splatId)
    {
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_X, splatId, position.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_Y, splatId, position.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_POSITION_Z, splatId, position.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SCALE_X, splatId, scale.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SCALE_Y, splatId, scale.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_SCALE_Z, splatId, scale.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_X, splatId, rotation.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_Y, splatId, rotation.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_Z, splatId, rotation.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_ROTATION_W, splatId, rotation.w);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_COLOR_R, splatId, colorAlpha.x);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_COLOR_G, splatId, colorAlpha.y);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_COLOR_B, splatId, colorAlpha.z);
        store_splat_param(g_SplatParamsRW, g_SplatCount, PARAM_RAW_OPACITY, splatId, colorAlpha.w);
    }
};

float load_param_grad(uint paramId, uint splatId)
{
    return g_ParamGrads[splat_param_index(g_SplatCount, paramId, splatId)];
}

float3 load_param_grad3(uint firstParamId, uint splatId)
{
    return float3(
        load_param_grad(firstParamId + 0u, splatId),
        load_param_grad(firstParamId + 1u, splatId),
        load_param_grad(firstParamId + 2u, splatId)
    );
}

float4 load_param_grad4(uint firstParamId, uint splatId)
{
    return float4(
        load_param_grad(firstParamId + 0u, splatId),
        load_param_grad(firstParamId + 1u, splatId),
        load_param_grad(firstParamId + 2u, splatId),
        load_param_grad(firstParamId + 3u, splatId)
    );
}

void adam_update_param_group<let N : int>(
    uint splatId,
    uint firstParamId,
    inout vector<float, N> values,
    vector<float, N> grads,
    float biasCorr1,
    float biasCorr2
)
{
    [unroll] for (int i = 0; i < N; i++)
    {
        uint paramId = firstParamId + uint(i);
        uint paramIndex = splat_param_index(g_SplatCount, paramId, splatId);
        float momentM = g_AdamM[paramIndex];
        float momentV = g_AdamV[paramIndex];
        values[i] = adam_update_param_scalar(values[i], grads[i], momentM, momentV, g_ParamLRs[paramId], biasCorr1, biasCorr2, g_Adam, g_Stability);
        g_AdamM[paramIndex] = momentM;
        g_AdamV[paramIndex] = momentV;
    }
}

float3 clamp_scale_anisotropy(float3 scale)
{
    float maxAnisotropy = max(g_Stability.maxAnisotropy, 1.0);
    float maxScale = max(scale.x, max(scale.y, scale.z));
    return max(scale, float3(maxScale / maxAnisotropy));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csAdamStepFused(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;
    TrainableGaussianState state = TrainableGaussianState.load(splatId);
    float3 gradPos = load_param_grad3(PARAM_POSITION_X, splatId);
    float3 gradScale = load_param_grad3(PARAM_SCALE_X, splatId);
    float4 gradQuat = load_param_grad4(PARAM_ROTATION_X, splatId);
    float4 gradColorAlpha = load_param_grad4(PARAM_COLOR_R, splatId);

    float scaleL2Weight = max(g_ScaleL2Weight, 0.0);
    if (scaleL2Weight > 0.0)
    {
        float invSplatCount = 1.0 / max(float(g_SplatCount), 1.0);
        float scaleRegReference = max(g_ScaleRegReference, TINY_VALUE);
        var dpScaleReg = diffPair(state.scale);
        float scaleRegLoss = scale_log_reference_loss(state.scale, scaleRegReference);
        bwd_diff(scale_log_reference_loss)(dpScaleReg, scaleRegReference, scaleL2Weight * invSplatCount);
        gradScale += dpScaleReg.getDifferential();
        float lossAdd = scaleL2Weight * scaleRegLoss * invSplatCount;
        if (isfinite(lossAdd) && lossAdd != 0.0) InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], lossAdd);
    }
    float scaleAbsRegWeight = max(g_ScaleAbsRegWeight, 0.0);
    if (scaleAbsRegWeight > 0.0)
    {
        float invSplatCount = 1.0 / max(float(g_SplatCount), 1.0);
        float3 signGrad = float3(state.scale.x > 0.0 ? 1.0 : state.scale.x < 0.0 ? -1.0 : 0.0, state.scale.y > 0.0 ? 1.0 : state.scale.y < 0.0 ? -1.0 : 0.0, state.scale.z > 0.0 ? 1.0 : state.scale.z < 0.0 ? -1.0 : 0.0);
        gradScale += signGrad * (scaleAbsRegWeight * invSplatCount * (1.0 / 3.0));
        float lossAdd = scaleAbsRegWeight * scale_abs_l1_loss(state.scale) * invSplatCount;
        if (isfinite(lossAdd) && lossAdd != 0.0) InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], lossAdd);
    }
    float opacityRegWeight = max(g_OpacityRegWeight, 0.0);
    if (opacityRegWeight > 0.0)
    {
        float invSplatCount = 1.0 / max(float(g_SplatCount), 1.0);
        var dpOpacityReg = diffPair(state.colorAlpha.w);
        float opacityRegLoss = opacity_l1_loss(state.colorAlpha.w);
        bwd_diff(opacity_l1_loss)(dpOpacityReg, opacityRegWeight * invSplatCount);
        gradColorAlpha.w += dpOpacityReg.getDifferential();
        float lossAdd = opacityRegWeight * opacityRegLoss * invSplatCount;
        if (isfinite(lossAdd) && lossAdd != 0.0) InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], lossAdd);
    }

    float clipAbs = g_Stability.gradComponentClip;
    gradPos = sanitize_float3(gradPos, clipAbs);
    gradScale = sanitize_float3(gradScale, clipAbs);
    gradQuat = sanitize_float4(gradQuat, clipAbs);
    gradColorAlpha = sanitize_float4(gradColorAlpha, clipAbs);
    gradPos = clip_norm3(gradPos, g_Stability.gradNormClip);
    gradScale = clip_norm3(gradScale, g_Stability.gradNormClip);
    gradQuat = clip_norm4(gradQuat, g_Stability.gradNormClip);
    gradColorAlpha = clip_norm4(gradColorAlpha, g_Stability.gradNormClip);

    float beta1Pow = pow(g_Adam.beta1, float(g_Adam.stepIndex));
    float beta2Pow = pow(g_Adam.beta2, float(g_Adam.stepIndex));
    float biasCorr1 = 1.0 - beta1Pow;
    float biasCorr2 = 1.0 - beta2Pow;

    adam_update_param_group<3>(splatId, PARAM_POSITION_X, state.position, gradPos, biasCorr1, biasCorr2);
    adam_update_param_group<3>(splatId, PARAM_SCALE_X, state.scale, gradScale, biasCorr1, biasCorr2);
    adam_update_param_group<4>(splatId, PARAM_ROTATION_X, state.rotation, gradQuat, biasCorr1, biasCorr2);
    adam_update_param_group<4>(splatId, PARAM_COLOR_R, state.colorAlpha, gradColorAlpha, biasCorr1, biasCorr2);

    state.position = clamp(state.position, -g_Stability.positionAbsMax, g_Stability.positionAbsMax);
    state.scale = clamp(state.scale, g_Stability.minScale, g_Stability.maxScale);
    state.scale = clamp_scale_anisotropy(state.scale);
    state.rotation = normalize_quaternion_safe(state.rotation);
    state.colorAlpha.xyz = clamp(state.colorAlpha.xyz, 0.0, 1.0);
    state.store(splatId);
}
