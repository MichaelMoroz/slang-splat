import "utility/losses/losses.slang";
import "utility/optimizer/optimizer.slang";
import "utility/random/random.slang";
import "utility/math/constants.slang";
import "utility/math/numeric.slang";
import "utility/math/quaternion.slang";
import "utility/splatting/opacity.slang";

static const uint TRAIN_TILE_DIM = 16u;
static const uint SCALE_HISTOGRAM_GUARD_BIN_COUNT = 2u;
static const uint LOSS_SLOT_TOTAL = 0u;
static const uint LOSS_SLOT_MSE = 1u;
static const uint LOSS_SLOT_SSIM = 2u;
static const float OPACITY_RESET_VALUE = 0.1;
static const uint SPLIT_CHILD_COUNT = 2u;
static const float SPLIT_CHILD_SCALE_RATIO = 0.625;
static const float CLAMPED_SCALE_DENSIFY_RATIO = 1.25;
static const uint INVALID_SPLAT_INDEX = 0xFFFFFFFFu;
static const uint MCMC_RELOCATION_N_MAX = 51u;

struct TrainingMCMCParams
{
    uint enabled;
    float positionNoiseScale;
    float opacityGateSharpness;
    float opacityGateCenter;
};

struct DensifyParams
{
    uint maxGaussianCount;
    float gradThreshold;
    float percentDense;
    float sceneExtent;
    float pruneMinOpacity;
    float screenSizeThreshold;
    float worldSizeRatio;
    uint outputCapacity;
    uint enableScreenSizePrune;
    uint forceSplitAll;
    float forcePruneScaleThreshold;
};

struct HistogramParams
{
    uint binCount;
    float minLog10;
    float maxLog10;
};

uniform AdamParams g_Adam;
uniform TrainingStabilityParams g_Stability;
uniform TrainingMCMCParams g_MCMC;
uniform DensifyParams g_Densify;
uniform HistogramParams g_Hist;
uniform uint g_SplatCount;
uniform uint g_CompactedCount;
uniform uint g_Width;
uniform uint g_Height;
uniform float g_InvPixelCount;
uniform float g_LossGradClip;
uniform float g_ScaleL2Weight;
uniform float g_ScaleAbsRegWeight;
uniform float g_OpacityRegWeight;
uniform float g_ScaleRegReference;
uniform float g_LambdaDSSIM;
uniform uint g_ScanElementCount;
uniform uint g_MCMCAppendBaseIndex;
uniform uint g_MCMCSampleSeed;

uniform Texture2D<float4> g_Rendered;
uniform Texture2D<float4> g_Target;
uniform RWTexture2D<float4> g_OutputGrad;
uniform RWStructuredBuffer<float> g_LossBuffer;
uniform RWStructuredBuffer<uint> g_SignalMaxBuffer;
uniform RWTexture2D<float4> g_RenderedSq;
uniform RWTexture2D<float4> g_TargetSq;
uniform RWTexture2D<float4> g_RenderTarget;
uniform Texture2D<float4> g_MuRendered;
uniform Texture2D<float4> g_MuTarget;
uniform Texture2D<float4> g_BlurRenderedSq;
uniform Texture2D<float4> g_BlurTargetSq;
uniform Texture2D<float4> g_BlurRenderTarget;
uniform RWTexture2D<float4> g_GradMuRendered;
uniform RWTexture2D<float4> g_GradBlurRenderedSq;
uniform RWTexture2D<float4> g_GradBlurRenderTarget;
uniform Texture2D<float4> g_BlurredGradMuRendered;
uniform Texture2D<float4> g_BlurredGradRenderedSq;
uniform Texture2D<float4> g_BlurredGradRenderTarget;
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
groupshared float gsSSIM[TRAIN_TILE_DIM * TRAIN_TILE_DIM];
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
uniform RWStructuredBuffer<float> g_GradNormEMA;
uniform RWStructuredBuffer<uint> g_GradCount;
uniform RWStructuredBuffer<float> g_MaxScreenRadius;
uniform StructuredBuffer<uint> g_SplatVisible;
uniform StructuredBuffer<float4> g_ScreenCenterRadiusDepth;
uniform RWStructuredBuffer<float4> g_OutPositionsRW;
uniform RWStructuredBuffer<float4> g_OutScalesRW;
uniform RWStructuredBuffer<float4> g_OutRotationsRW;
uniform RWStructuredBuffer<float4> g_OutColorAlphaRW;
uniform RWStructuredBuffer<float4> g_OutAdamMPos;
uniform RWStructuredBuffer<float4> g_OutAdamVPos;
uniform RWStructuredBuffer<float4> g_OutAdamMScale;
uniform RWStructuredBuffer<float4> g_OutAdamVScale;
uniform RWStructuredBuffer<float4> g_OutAdamMQuat;
uniform RWStructuredBuffer<float4> g_OutAdamVQuat;
uniform RWStructuredBuffer<float4> g_OutAdamMColorAlpha;
uniform RWStructuredBuffer<float4> g_OutAdamVColorAlpha;
uniform RWStructuredBuffer<float> g_OutGradNormEMA;
uniform RWStructuredBuffer<uint> g_OutGradCount;
uniform RWStructuredBuffer<float> g_OutMaxScreenRadius;
uniform RWStructuredBuffer<uint> g_OutputCount;
uniform RWStructuredBuffer<uint> g_ScaleHistogram;
uniform RWStructuredBuffer<float> g_MCMCWeights;
uniform RWStructuredBuffer<uint> g_MCMCFlags;
uniform StructuredBuffer<float> g_MCMCWeightPrefix;
uniform StructuredBuffer<uint> g_MCMCFlagPrefix;
uniform RWStructuredBuffer<float> g_MCMCWeightTotal;
uniform RWStructuredBuffer<uint> g_MCMCDeadIndices;
uniform RWStructuredBuffer<uint> g_MCMCSampleSources;
uniform RWStructuredBuffer<uint> g_MCMCFamilyCounts;
uniform RWStructuredBuffer<uint> g_MCMCSampleCount;
uniform StructuredBuffer<float4> g_InitPointPositions;
uniform StructuredBuffer<float4> g_InitPointColors;
uniform uint g_InitPointCount;
uniform uint g_InitSeed;
uniform float g_InitPositionJitterStd;
uniform float g_InitBaseScale;
uniform float g_InitScaleJitterRatio;
uniform float g_InitInitialOpacity;
uniform float g_InitColorJitterStd;

float opacity_noise_gate(float opacity)
{
    float x = g_MCMC.opacityGateSharpness * ((1.0 - opacity) - g_MCMC.opacityGateCenter);
    x = clamp(x, -SOFTPLUS_INPUT_LIMIT, SOFTPLUS_INPUT_LIMIT);
    return 1.0 / (1.0 + exp(-x));
}

uint mcmc_relocation_family_size(uint sampleCount)
{
    return min(max(sampleCount + 1u, 1u), MCMC_RELOCATION_N_MAX - 1u);
}

float mcmc_binom_term(uint n, uint k)
{
    if (k > n) return 0.0;
    float result = 1.0;
    [unroll] for (uint i = 1u; i <= 50u; ++i)
    {
        if (i > k) break;
        result *= float(n + 1u - i) / float(i);
    }
    return result;
}

float mcmc_relocation_scale_coeff(float oldAlpha, float newAlpha, uint familySize)
{
    if (familySize <= 1u) return 1.0;
    float denom = 0.0;
    [unroll] for (uint subsetSize = 1u; subsetSize < MCMC_RELOCATION_N_MAX; ++subsetSize)
    {
        if (subsetSize > familySize) break;
        [unroll] for (uint k = 0u; k < MCMC_RELOCATION_N_MAX; ++k)
        {
            if (k >= subsetSize) break;
            float sign = ((k & 1u) == 0u) ? 1.0 : -1.0;
            denom += mcmc_binom_term(subsetSize - 1u, k) * sign * pow(newAlpha, float(k + 1u)) * rsqrt(float(k + 1u));
        }
    }
    return oldAlpha / max(denom, TINY_VALUE);
}

void mcmc_relocate_opacity_scale(float oldOpacity, float3 oldScale, uint sampleCount, out float newOpacity, out float3 newScale)
{
    uint familySize = mcmc_relocation_family_size(sampleCount);
    float alpha = clamp(oldOpacity, TINY_VALUE, 1.0 - TINY_VALUE);
    float relocatedAlpha = 1.0 - pow(1.0 - alpha, 1.0 / float(familySize));
    float coeff = mcmc_relocation_scale_coeff(alpha, relocatedAlpha, familySize);
    newOpacity = clamp(relocatedAlpha, max(g_Densify.pruneMinOpacity, g_Stability.minOpacity), min(g_Stability.maxOpacity, 1.0 - TINY_VALUE));
    newScale = clamp_scale_anisotropy(clamp(oldScale * coeff, g_Stability.minScale, g_Stability.maxScale));
}

float3 clamp_scale_anisotropy(float3 scale)
{
    float maxAnisotropy = max(g_Stability.maxAnisotropy, 1.0);
    float maxScale = max(scale.x, max(scale.y, scale.z));
    return max(scale, float3(maxScale / maxAnisotropy));
}

float scene_small_scale_threshold()
{
    return max(g_Densify.percentDense * g_Densify.sceneExtent, g_Stability.minScale);
}

uint scale_histogram_bin(float3 scale)
{
    uint binCount = max(g_Hist.binCount, SCALE_HISTOGRAM_GUARD_BIN_COUNT);
    uint interiorCount = max(binCount - SCALE_HISTOGRAM_GUARD_BIN_COUNT, 1u);
    float meanLog10 = (log10(max(scale.x, TINY_VALUE)) + log10(max(scale.y, TINY_VALUE)) + log10(max(scale.z, TINY_VALUE))) * (1.0 / 3.0);
    if (meanLog10 < g_Hist.minLog10) return 0u;
    if (meanLog10 >= g_Hist.maxLog10) return binCount - 1u;
    float t = saturate((meanLog10 - g_Hist.minLog10) / max(g_Hist.maxLog10 - g_Hist.minLog10, TINY_VALUE));
    return 1u + min(uint(t * float(interiorCount)), interiorCount - 1u);
}

void write_output_splat(
    uint outputId,
    float4 pos,
    float4 scale,
    float4 rot,
    float4 colorAlpha,
    float4 adamMPos,
    float4 adamVPos,
    float4 adamMScale,
    float4 adamVScale,
    float4 adamMQuat,
    float4 adamVQuat,
    float4 adamMColorAlpha,
    float4 adamVColorAlpha,
    float gradNormEMA,
    uint gradCount,
    float maxScreenRadius
)
{
    if (outputId >= g_Densify.outputCapacity) return;
    g_OutPositionsRW[outputId] = pos;
    g_OutScalesRW[outputId] = scale;
    g_OutRotationsRW[outputId] = rot;
    g_OutColorAlphaRW[outputId] = colorAlpha;
    g_OutAdamMPos[outputId] = adamMPos;
    g_OutAdamVPos[outputId] = adamVPos;
    g_OutAdamMScale[outputId] = adamMScale;
    g_OutAdamVScale[outputId] = adamVScale;
    g_OutAdamMQuat[outputId] = adamMQuat;
    g_OutAdamVQuat[outputId] = adamVQuat;
    g_OutAdamMColorAlpha[outputId] = adamMColorAlpha;
    g_OutAdamVColorAlpha[outputId] = adamVColorAlpha;
    g_OutGradNormEMA[outputId] = gradNormEMA;
    g_OutGradCount[outputId] = gradCount;
    g_OutMaxScreenRadius[outputId] = maxScreenRadius;
}

bool reserve_output_slot(out uint outputId)
{
    outputId = 0u;
    [loop] while (true)
    {
        uint current = g_OutputCount[0];
        if (current >= g_Densify.outputCapacity) return false;
        uint previous = 0u;
        InterlockedCompareExchange(g_OutputCount[0], current, current + 1u, previous);
        if (previous == current)
        {
            outputId = current;
            return true;
        }
    }
    return false;
}

uint append_output_splat(
    float4 pos,
    float4 scale,
    float4 rot,
    float4 colorAlpha,
    float4 adamMPos,
    float4 adamVPos,
    float4 adamMScale,
    float4 adamVScale,
    float4 adamMQuat,
    float4 adamVQuat,
    float4 adamMColorAlpha,
    float4 adamVColorAlpha
)
{
    uint outputId = 0u;
    if (!reserve_output_slot(outputId)) return g_Densify.outputCapacity;
    write_output_splat(outputId, pos, scale, rot, colorAlpha, adamMPos, adamVPos, adamMScale, adamVScale, adamMQuat, adamVQuat, adamMColorAlpha, adamVColorAlpha, 0.0, 0u, 0.0);
    return outputId;
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
        g_LossBuffer[LOSS_SLOT_SSIM] = 0.0;
        g_SignalMaxBuffer[0] = 0u;
    }
}

[shader("compute")]
[numthreads(TRAIN_TILE_DIM, TRAIN_TILE_DIM, 1)]
void csPackSSIMAux(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= g_Width || py >= g_Height) return;
    uint2 pixel = uint2(px, py);
    float3 rendered = sanitize_float3(g_Rendered[pixel].xyz, g_Stability.hugeValue);
    float3 target = sanitize_float3(g_Target[pixel].xyz, g_Stability.hugeValue);
    g_RenderedSq[pixel] = float4(rendered * rendered, 0.0);
    g_TargetSq[pixel] = float4(target * target, 0.0);
    g_RenderTarget[pixel] = float4(rendered * target, 0.0);
}

[shader("compute")]
[numthreads(TRAIN_TILE_DIM, TRAIN_TILE_DIM, 1)]
void csComputeMixedLossGrad(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID
)
{
    uint localIndex = groupThreadID.y * TRAIN_TILE_DIM + groupThreadID.x;
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;

    float localLoss = 0.0;
    float localMSE = 0.0;
    float localSSIM = 0.0;
    float localSignalMax = 0.0;
    if (px < g_Width && py < g_Height)
    {
        uint2 pixel = uint2(px, py);
        float3 rendered = sanitize_float3(g_Rendered[pixel].xyz, g_Stability.hugeValue);
        float3 target = sanitize_float3(g_Target[pixel].xyz, g_Stability.hugeValue);
        float3 muRendered = sanitize_float3(g_MuRendered[pixel].xyz, g_Stability.hugeValue);
        float3 muTarget = sanitize_float3(g_MuTarget[pixel].xyz, g_Stability.hugeValue);
        float3 blurRenderedSq = sanitize_float3(g_BlurRenderedSq[pixel].xyz, g_Stability.hugeValue);
        float3 blurTargetSq = sanitize_float3(g_BlurTargetSq[pixel].xyz, g_Stability.hugeValue);
        float3 blurRenderTarget = sanitize_float3(g_BlurRenderTarget[pixel].xyz, g_Stability.hugeValue);
        float mseLoss = rgb_mse_loss(rendered, target, g_InvPixelCount);
        float l1Loss = (1.0 - g_LambdaDSSIM) * rgb_l1_loss(rendered, target, g_InvPixelCount);
        float ssimLoss = rgb_dssim_loss(muRendered, muTarget, blurRenderedSq, blurTargetSq, blurRenderTarget, g_InvPixelCount, g_LambdaDSSIM);
        float ssimMetric = rgb_ssim_metric(muRendered, muTarget, blurRenderedSq, blurTargetSq, blurRenderTarget);
        float loss = l1Loss + ssimLoss;
        localLoss = isfinite(loss) ? loss : 0.0;
        localMSE = isfinite(mseLoss) ? mseLoss : 0.0;
        localSSIM = isfinite(ssimMetric) ? ssimMetric * g_InvPixelCount : 0.0;
        localSignalMax = max(target.x, max(target.y, target.z));
        float3 l1Grad = (1.0 - g_LambdaDSSIM) * rgb_l1_grad(rendered, target, g_InvPixelCount);

        var dpMuRendered = diffPair(muRendered);
        var dpBlurRenderedSq = diffPair(blurRenderedSq);
        var dpBlurRenderTarget = diffPair(blurRenderTarget);
        bwd_diff(rgb_dssim_loss)(dpMuRendered, muTarget, dpBlurRenderedSq, blurTargetSq, dpBlurRenderTarget, g_InvPixelCount, g_LambdaDSSIM, 1.0);

        float3 dOut = sanitize_float3(l1Grad, g_LossGradClip);
        dOut = clamp(dOut, -g_LossGradClip, g_LossGradClip);
        g_OutputGrad[pixel] = float4(dOut, 0.0);
        g_GradMuRendered[pixel] = float4(sanitize_float3(dpMuRendered.getDifferential(), g_LossGradClip), 0.0);
        g_GradBlurRenderedSq[pixel] = float4(sanitize_float3(dpBlurRenderedSq.getDifferential(), g_LossGradClip), 0.0);
        g_GradBlurRenderTarget[pixel] = float4(sanitize_float3(dpBlurRenderTarget.getDifferential(), g_LossGradClip), 0.0);
    }

    gsLoss[localIndex] = localLoss;
    gsMSE[localIndex] = localMSE;
    gsSSIM[localIndex] = localSSIM;
    gsSignalMax[localIndex] = isfinite(localSignalMax) ? max(localSignalMax, 0.0) : 0.0;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = (TRAIN_TILE_DIM * TRAIN_TILE_DIM) >> 1; stride > 0; stride >>= 1)
    {
        if (localIndex < stride) gsLoss[localIndex] += gsLoss[localIndex + stride];
        if (localIndex < stride) gsMSE[localIndex] += gsMSE[localIndex + stride];
        if (localIndex < stride) gsSSIM[localIndex] += gsSSIM[localIndex + stride];
        if (localIndex < stride) gsSignalMax[localIndex] = max(gsSignalMax[localIndex], gsSignalMax[localIndex + stride]);
        GroupMemoryBarrierWithGroupSync();
    }
    if (localIndex == 0u)
    {
        if (isfinite(gsLoss[0])) InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], gsLoss[0]);
        if (isfinite(gsMSE[0])) InterlockedAdd(g_LossBuffer[LOSS_SLOT_MSE], gsMSE[0]);
        if (isfinite(gsSSIM[0])) InterlockedAdd(g_LossBuffer[LOSS_SLOT_SSIM], gsSSIM[0]);
        InterlockedMax(g_SignalMaxBuffer[0], asuint(max(gsSignalMax[0], 0.0)));
    }
}

[shader("compute")]
[numthreads(TRAIN_TILE_DIM, TRAIN_TILE_DIM, 1)]
void csComposeMixedLossOutputGrad(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= g_Width || py >= g_Height) return;

    uint2 pixel = uint2(px, py);
    float3 rendered = sanitize_float3(g_Rendered[pixel].xyz, g_Stability.hugeValue);
    float3 target = sanitize_float3(g_Target[pixel].xyz, g_Stability.hugeValue);
    float3 gradMu = sanitize_float3(g_BlurredGradMuRendered[pixel].xyz, g_LossGradClip);
    float3 gradRenderedSq = sanitize_float3(g_BlurredGradRenderedSq[pixel].xyz, g_LossGradClip);
    float3 gradRenderTarget = sanitize_float3(g_BlurredGradRenderTarget[pixel].xyz, g_LossGradClip);
    float3 directGrad = sanitize_float3(g_OutputGrad[pixel].xyz, g_LossGradClip);
    float3 totalGrad = directGrad + gradMu + 2.0 * rendered * gradRenderedSq + target * gradRenderTarget;
    totalGrad = sanitize_float3(totalGrad, g_LossGradClip);
    totalGrad = clamp(totalGrad, -g_LossGradClip, g_LossGradClip);
    g_OutputGrad[pixel] = float4(totalGrad, 0.0);
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

    float scaleL2Weight = max(g_ScaleL2Weight, 0.0);
    if (scaleL2Weight > 0.0)
    {
        float invSplatCount = 1.0 / max(float(g_SplatCount), 1.0);
        float scaleRegReference = max(g_ScaleRegReference, TINY_VALUE);
        var dpScaleReg = diffPair(scale.xyz);
        float scaleRegLoss = scale_log_reference_loss(scale.xyz, scaleRegReference);
        bwd_diff(scale_log_reference_loss)(dpScaleReg, scaleRegReference, scaleL2Weight * invSplatCount);
        gradScale.xyz += dpScaleReg.getDifferential();
        float lossAdd = scaleL2Weight * scaleRegLoss * invSplatCount;
        if (isfinite(lossAdd) && lossAdd != 0.0) InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], lossAdd);
    }
    float scaleAbsRegWeight = max(g_ScaleAbsRegWeight, 0.0);
    if (scaleAbsRegWeight > 0.0)
    {
        float invSplatCount = 1.0 / max(float(g_SplatCount), 1.0);
        float3 signGrad = float3(scale.x > 0.0 ? 1.0 : scale.x < 0.0 ? -1.0 : 0.0, scale.y > 0.0 ? 1.0 : scale.y < 0.0 ? -1.0 : 0.0, scale.z > 0.0 ? 1.0 : scale.z < 0.0 ? -1.0 : 0.0);
        gradScale.xyz += signGrad * (scaleAbsRegWeight * invSplatCount * (1.0 / 3.0));
        float lossAdd = scaleAbsRegWeight * scale_abs_l1_loss(scale.xyz) * invSplatCount;
        if (isfinite(lossAdd) && lossAdd != 0.0) InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], lossAdd);
    }
    float opacityRegWeight = max(g_OpacityRegWeight, 0.0);
    if (opacityRegWeight > 0.0)
    {
        float invSplatCount = 1.0 / max(float(g_SplatCount), 1.0);
        var dpOpacityReg = diffPair(colorAlpha.w);
        float opacityRegLoss = opacity_l1_loss(colorAlpha.w);
        bwd_diff(opacity_l1_loss)(dpOpacityReg, opacityRegWeight * invSplatCount);
        gradColorAlpha.w += dpOpacityReg.getDifferential();
        float lossAdd = opacityRegWeight * opacityRegLoss * invSplatCount;
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
    pos = adam_update_param(pos, gradPos, mPos, vPos, posLR, biasCorr1, biasCorr2, g_Adam, g_Stability);
    scale = adam_update_param(scale, gradScale, mScale, vScale, scaleLR, biasCorr1, biasCorr2, g_Adam, g_Stability);
    rot = adam_update_param(rot, gradQuat, mQuat, vQuat, rotLR, biasCorr1, biasCorr2, g_Adam, g_Stability);
    colorAlpha = adam_update_param(colorAlpha, gradColorAlpha, mColorAlpha, vColorAlpha, colorAlphaLR, biasCorr1, biasCorr2, g_Adam, g_Stability);

    if (g_MCMC.enabled != 0u && g_MCMC.positionNoiseScale > 0.0)
    {
        float4 rotSafe = normalize_quaternion_safe(rot);
        uint rngSeed = splatId * 0x9e3779b9u ^ g_Adam.stepIndex * 0x85ebca6bu ^ 0xc2b2ae35u;
        uint rngState = hash_u32(rngSeed);
        if (rngState == 0u) rngState = 1u;
        float3 etaWorld = rng_normal3_box_muller(rngState);
        float4 qInv = float4(rotSafe.x, -rotSafe.y, -rotSafe.z, -rotSafe.w);
        float3 scaleSafe = max(scale.xyz, float3(g_Stability.minScale));
        float gate = opacity_noise_gate(opacity_from_raw(colorAlpha.w));
        float3 etaLocal = rotate_by_quat(etaWorld, qInv);
        float3 worldNoise = rotate_by_quat(scaleSafe * scaleSafe * etaLocal, rotSafe);
        float noiseScale = g_Adam.positionLR * g_MCMC.positionNoiseScale * gate;
        float3 noise = sanitize_float3(worldNoise * noiseScale, g_Stability.hugeValue);
        pos.xyz = sanitize_float3(pos.xyz + noise, g_Stability.hugeValue);
    }

    pos.xyz = clamp(pos.xyz, -g_Stability.positionAbsMax, g_Stability.positionAbsMax);
    scale.xyz = clamp(scale.xyz, g_Stability.minScale, g_Stability.maxScale);
    scale.xyz = clamp_scale_anisotropy(scale.xyz);
    rot = normalize_quaternion_safe(rot);
    colorAlpha.xyz = clamp(colorAlpha.xyz, 0.0, 1.0);

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
void csUpdateDensificationStats(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;
    if (g_SplatVisible[splatId] == 0u) return;
    uint gradBase = splatId * 4u;
    float3 gradPos = float3(
        g_GradPositions[gradBase + 0u],
        g_GradPositions[gradBase + 1u],
        g_GradPositions[gradBase + 2u]
    );
    float4 screen = g_ScreenCenterRadiusDepth[splatId];
    float radius = sanitize_scalar(screen.z, g_Stability.hugeValue);
    if (!isfinite(radius) || radius <= 0.0) return;
    float gradNorm = length(sanitize_float3(gradPos, g_Stability.hugeValue));
    uint nextCount = g_GradCount[splatId] + 1u;
    float prevEMA = sanitize_scalar(g_GradNormEMA[splatId], g_Stability.hugeValue);
    float prevRadius = sanitize_scalar(g_MaxScreenRadius[splatId], g_Stability.hugeValue);
    g_GradNormEMA[splatId] = prevEMA + (gradNorm - prevEMA) / float(nextCount);
    g_GradCount[splatId] = nextCount;
    g_MaxScreenRadius[splatId] = max(prevRadius, radius);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csClearScaleHistogram(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint binId = dispatchThreadID.x;
    if (binId >= g_Hist.binCount) return;
    g_ScaleHistogram[binId] = 0u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csAccumulateScaleHistogram(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;
    uint binId = scale_histogram_bin(max(g_ScalesRW[splatId].xyz, float3(TINY_VALUE)));
    InterlockedAdd(g_ScaleHistogram[binId], 1u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csPrepareMCMCRelocationWeights(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;
    float opacity = opacity_from_raw(g_ColorAlphaRW[splatId].w);
    bool dead = opacity <= g_Densify.pruneMinOpacity;
    g_MCMCWeights[splatId] = dead ? 0.0 : max(opacity, 0.0);
    g_MCMCFlags[splatId] = dead ? 1u : 0u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csPrepareMCMCAppendWeights(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;
    float opacity = opacity_from_raw(g_ColorAlphaRW[splatId].w);
    g_MCMCWeights[splatId] = max(opacity, 0.0);
    g_MCMCFlags[splatId] = 0u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csCompactDeadIndices(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_ScanElementCount) return;
    uint flag = g_MCMCFlags[splatId];
    uint prefix = g_MCMCFlagPrefix[splatId];
    if (flag != 0u && prefix > 0u) g_MCMCDeadIndices[prefix - 1u] = splatId;
    if (splatId + 1u == g_ScanElementCount) g_MCMCSampleCount[0] = prefix;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csSampleMCMCSources(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint sampleId = dispatchThreadID.x;
    uint sampleCount = g_MCMCSampleCount[0];
    if (sampleId >= sampleCount) return;
    float totalWeight = g_MCMCWeightTotal[0];
    if (!isfinite(totalWeight) || totalWeight <= 0.0)
    {
        g_MCMCSampleSources[sampleId] = INVALID_SPLAT_INDEX;
        return;
    }
    uint rngState = hash_u32(g_MCMCSampleSeed ^ sampleId * 0x85ebca6bu);
    if (rngState == 0u) rngState = 1u;
    float target = rng_uniform_open01(rngState) * totalWeight;
    uint lo = 0u;
    uint hi = max(g_SplatCount, 1u);
    while (lo < hi)
    {
        uint mid = (lo + hi) >> 1u;
        float prefix = g_MCMCWeightPrefix[mid];
        if (prefix > target) hi = mid;
        else lo = mid + 1u;
    }
    uint sourceId = min(lo, max(g_SplatCount, 1u) - 1u);
    g_MCMCSampleSources[sampleId] = sourceId;
    InterlockedAdd(g_MCMCFamilyCounts[sourceId], 1u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csUpdateMCMCSources(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;
    uint sampleCount = g_MCMCFamilyCounts[splatId];
    if (sampleCount == 0u) return;
    float4 scale = g_ScalesRW[splatId];
    float4 colorAlpha = g_ColorAlphaRW[splatId];
    float newOpacity = 0.0;
    float3 newScale = 0.0;
    mcmc_relocate_opacity_scale(opacity_from_raw(colorAlpha.w), scale.xyz, sampleCount, newOpacity, newScale);
    g_ScalesRW[splatId] = float4(newScale, 0.0);
    colorAlpha.w = raw_opacity_from_alpha(newOpacity);
    g_ColorAlphaRW[splatId] = colorAlpha;
    g_AdamMPos[splatId] = 0.0;
    g_AdamVPos[splatId] = 0.0;
    g_AdamMScale[splatId] = 0.0;
    g_AdamVScale[splatId] = 0.0;
    g_AdamMQuat[splatId] = 0.0;
    g_AdamVQuat[splatId] = 0.0;
    g_AdamMColorAlpha[splatId] = 0.0;
    g_AdamVColorAlpha[splatId] = 0.0;
    g_GradNormEMA[splatId] = 0.0;
    g_GradCount[splatId] = 0u;
    g_MaxScreenRadius[splatId] = 0.0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csApplyMCMCRelocationSamples(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint sampleId = dispatchThreadID.x;
    uint sampleCount = g_MCMCSampleCount[0];
    if (sampleId >= sampleCount) return;
    uint sourceId = g_MCMCSampleSources[sampleId];
    if (sourceId == INVALID_SPLAT_INDEX || sourceId >= g_SplatCount) return;
    uint targetId = g_MCMCDeadIndices[sampleId];
    if (targetId >= g_SplatCount) return;
    g_PositionsRW[targetId] = g_PositionsRW[sourceId];
    g_ScalesRW[targetId] = g_ScalesRW[sourceId];
    g_RotationsRW[targetId] = g_RotationsRW[sourceId];
    g_ColorAlphaRW[targetId] = g_ColorAlphaRW[sourceId];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csAppendMCMCSamples(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint sampleId = dispatchThreadID.x;
    uint sampleCount = g_MCMCSampleCount[0];
    if (sampleId >= sampleCount) return;
    uint sourceId = g_MCMCSampleSources[sampleId];
    uint targetId = g_MCMCAppendBaseIndex + sampleId;
    if (sourceId == INVALID_SPLAT_INDEX || sourceId >= g_SplatCount || targetId >= g_Densify.outputCapacity) return;
    g_PositionsRW[targetId] = g_PositionsRW[sourceId];
    g_ScalesRW[targetId] = g_ScalesRW[sourceId];
    g_RotationsRW[targetId] = g_RotationsRW[sourceId];
    g_ColorAlphaRW[targetId] = g_ColorAlphaRW[sourceId];
    g_AdamMPos[targetId] = 0.0;
    g_AdamVPos[targetId] = 0.0;
    g_AdamMScale[targetId] = 0.0;
    g_AdamVScale[targetId] = 0.0;
    g_AdamMQuat[targetId] = 0.0;
    g_AdamVQuat[targetId] = 0.0;
    g_AdamMColorAlpha[targetId] = 0.0;
    g_AdamVColorAlpha[targetId] = 0.0;
    g_GradNormEMA[targetId] = 0.0;
    g_GradCount[targetId] = 0u;
    g_MaxScreenRadius[targetId] = 0.0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csCompactScene(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;

    float4 pos = g_PositionsRW[splatId];
    float4 scale = g_ScalesRW[splatId];
    float4 rot = normalize_quaternion_safe(g_RotationsRW[splatId]);
    float4 colorAlpha = g_ColorAlphaRW[splatId];
    float4 adamMPos = g_AdamMPos[splatId];
    float4 adamVPos = g_AdamVPos[splatId];
    float4 adamMScale = g_AdamMScale[splatId];
    float4 adamVScale = g_AdamVScale[splatId];
    float4 adamMQuat = g_AdamMQuat[splatId];
    float4 adamVQuat = g_AdamVQuat[splatId];
    float4 adamMColorAlpha = g_AdamMColorAlpha[splatId];
    float4 adamVColorAlpha = g_AdamVColorAlpha[splatId];
    float gradEMA = sanitize_scalar(g_GradNormEMA[splatId], g_Stability.hugeValue);
    uint gradCount = g_GradCount[splatId];
    float maxScreenRadius = sanitize_scalar(g_MaxScreenRadius[splatId], g_Stability.hugeValue);
    float opacity = opacity_from_raw(colorAlpha.w);
    float maxAxisScale = max(scale.x, max(scale.y, scale.z));
    bool prune = opacity < g_Densify.pruneMinOpacity || maxAxisScale < g_Densify.forcePruneScaleThreshold;
    if (g_Densify.enableScreenSizePrune != 0u && g_Densify.screenSizeThreshold > 0.0) prune = prune || maxScreenRadius > g_Densify.screenSizeThreshold;
    if (g_Densify.enableScreenSizePrune != 0u && g_Densify.worldSizeRatio > 0.0) prune = prune || maxAxisScale > g_Densify.worldSizeRatio * g_Densify.sceneExtent;
    if (prune) return;

    uint outputId = 0u;
    if (!reserve_output_slot(outputId)) return;
    write_output_splat(outputId, pos, scale, rot, colorAlpha, adamMPos, adamVPos, adamMScale, adamVScale, adamMQuat, adamVQuat, adamMColorAlpha, adamVColorAlpha, gradEMA, gradCount, maxScreenRadius);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csAppendRegeneratedSplats(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_CompactedCount) return;

    float4 pos = g_OutPositionsRW[splatId];
    float4 scale = g_OutScalesRW[splatId];
    float4 rot = normalize_quaternion_safe(g_OutRotationsRW[splatId]);
    float4 colorAlpha = g_OutColorAlphaRW[splatId];
    float gradEMA = sanitize_scalar(g_OutGradNormEMA[splatId], g_Stability.hugeValue);
    float maxScale = max(scale.x, max(scale.y, scale.z));
    bool nearMinScale = maxScale <= g_Stability.minScale * CLAMPED_SCALE_DENSIFY_RATIO;
    bool allowGrowth = g_Densify.maxGaussianCount > g_CompactedCount;
    bool highGrad = gradEMA >= g_Densify.gradThreshold;
    bool small = maxScale <= scene_small_scale_threshold();
    bool shouldClone = allowGrowth && g_Densify.forceSplitAll == 0u && highGrad && small && !nearMinScale;
    bool shouldSplit = allowGrowth && (g_Densify.forceSplitAll != 0u || (highGrad && !small));

    if (shouldSplit)
    {
        float4 qInv = float4(rot.x, -rot.y, -rot.z, -rot.w);
        float3 splitStd = max(scale.xyz, float3(g_Stability.minScale));
        float3 childScale = clamp_scale_anisotropy(clamp(splitStd * SPLIT_CHILD_SCALE_RATIO, g_Stability.minScale, g_Stability.maxScale));
        uint firstSeed = hash_u32(splatId * 0x9e3779b9u ^ g_Adam.stepIndex * 0xc2b2ae35u ^ 0x27d4eb2fu);
        if (firstSeed == 0u) firstSeed = 1u;
        float3 firstOffset = rotate_by_quat(splitStd * rng_normal3_box_muller(firstSeed), qInv);
        pos.xyz = clamp(sanitize_float3(pos.xyz + firstOffset, g_Stability.hugeValue), -g_Stability.positionAbsMax, g_Stability.positionAbsMax);
        scale.xyz = childScale;
        g_OutPositionsRW[splatId] = pos;
        g_OutScalesRW[splatId] = float4(childScale, 0.0);
        g_OutRotationsRW[splatId] = rot;
        g_OutColorAlphaRW[splatId] = colorAlpha;
        g_OutAdamMPos[splatId] = 0.0;
        g_OutAdamVPos[splatId] = 0.0;
        g_OutAdamMScale[splatId] = 0.0;
        g_OutAdamVScale[splatId] = 0.0;
        g_OutAdamMQuat[splatId] = 0.0;
        g_OutAdamVQuat[splatId] = 0.0;
        g_OutAdamMColorAlpha[splatId] = 0.0;
        g_OutAdamVColorAlpha[splatId] = 0.0;
        g_OutGradNormEMA[splatId] = 0.0;
        g_OutGradCount[splatId] = 0u;
        g_OutMaxScreenRadius[splatId] = 0.0;

        uint secondSeed = hash_u32(splatId * 0x9e3779b9u ^ 1u * 0x85ebca6bu ^ g_Adam.stepIndex * 0xc2b2ae35u ^ 0x27d4eb2fu);
        if (secondSeed == 0u) secondSeed = 1u;
        float3 secondOffset = rotate_by_quat(splitStd * rng_normal3_box_muller(secondSeed), qInv);
        float4 childPos = pos;
        childPos.xyz = clamp(sanitize_float3(g_OutPositionsRW[splatId].xyz + (secondOffset - firstOffset), g_Stability.hugeValue), -g_Stability.positionAbsMax, g_Stability.positionAbsMax);
        append_output_splat(childPos, float4(childScale, 0.0), rot, colorAlpha, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        return;
    }

    if (shouldClone) append_output_splat(pos, scale, rot, colorAlpha, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csRegenerateScene(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;

    float4 pos = g_PositionsRW[splatId];
    float4 scale = g_ScalesRW[splatId];
    float4 rot = normalize_quaternion_safe(g_RotationsRW[splatId]);
    float4 colorAlpha = g_ColorAlphaRW[splatId];
    float4 adamMPos = g_AdamMPos[splatId];
    float4 adamVPos = g_AdamVPos[splatId];
    float4 adamMScale = g_AdamMScale[splatId];
    float4 adamVScale = g_AdamVScale[splatId];
    float4 adamMQuat = g_AdamMQuat[splatId];
    float4 adamVQuat = g_AdamVQuat[splatId];
    float4 adamMColorAlpha = g_AdamMColorAlpha[splatId];
    float4 adamVColorAlpha = g_AdamVColorAlpha[splatId];
    float gradEMA = sanitize_scalar(g_GradNormEMA[splatId], g_Stability.hugeValue);
    bool allowGrowth = g_Densify.maxGaussianCount > g_SplatCount;
    float opacity = opacity_from_raw(colorAlpha.w);
    float maxScale = max(scale.x, max(scale.y, scale.z));
    bool nearMinScale = maxScale <= g_Stability.minScale * CLAMPED_SCALE_DENSIFY_RATIO;

    bool prune = opacity < g_Densify.pruneMinOpacity || maxScale < g_Densify.forcePruneScaleThreshold;
    if (prune) return;

    bool highGrad = gradEMA >= g_Densify.gradThreshold;
    bool small = maxScale <= scene_small_scale_threshold();
    bool shouldClone = allowGrowth && g_Densify.forceSplitAll == 0u && highGrad && small && !nearMinScale;
    bool shouldSplit = allowGrowth && (g_Densify.forceSplitAll != 0u || (highGrad && !small));

    if (shouldClone)
    {
        append_output_splat(pos, scale, rot, colorAlpha, adamMPos, adamVPos, adamMScale, adamVScale, adamMQuat, adamVQuat, adamMColorAlpha, adamVColorAlpha);
        append_output_splat(pos, scale, rot, colorAlpha, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        return;
    }

    if (shouldSplit)
    {
        float4 qInv = float4(rot.x, -rot.y, -rot.z, -rot.w);
        float3 splitStd = max(scale.xyz, float3(g_Stability.minScale));
        float3 childScale = clamp_scale_anisotropy(clamp(splitStd * SPLIT_CHILD_SCALE_RATIO, g_Stability.minScale, g_Stability.maxScale));
        [unroll] for (uint childIdx = 0u; childIdx < SPLIT_CHILD_COUNT; ++childIdx)
        {
            uint rngState = hash_u32(splatId * 0x9e3779b9u ^ childIdx * 0x85ebca6bu ^ g_Adam.stepIndex * 0xc2b2ae35u ^ 0x27d4eb2fu);
            if (rngState == 0u) rngState = 1u;
            float3 localOffset = splitStd * rng_normal3_box_muller(rngState);
            float3 worldOffset = rotate_by_quat(localOffset, qInv);
            float4 childPos = pos;
            childPos.xyz = clamp(sanitize_float3(pos.xyz + worldOffset, g_Stability.hugeValue), -g_Stability.positionAbsMax, g_Stability.positionAbsMax);
            append_output_splat(childPos, float4(childScale, 0.0), rot, colorAlpha, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
        return;
    }

    append_output_splat(pos, scale, rot, colorAlpha, adamMPos, adamVPos, adamMScale, adamVScale, adamMQuat, adamVQuat, adamMColorAlpha, adamVColorAlpha);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csResetOpacity(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;
    float4 colorAlpha = g_ColorAlphaRW[splatId];
    colorAlpha.w = raw_opacity_from_alpha(min(opacity_from_raw(colorAlpha.w), OPACITY_RESET_VALUE));
    g_ColorAlphaRW[splatId] = colorAlpha;
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
    float4 colorAlpha = float4(color, raw_opacity_from_alpha(g_InitInitialOpacity));

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
