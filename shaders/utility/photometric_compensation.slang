import "utility/math/constants.slang";
import "utility/math/numeric.slang";
import "utility/ppisp_tonemap.slang";

static const uint PHOTOMETRIC_THREADS = 64u;
static const uint PPISP_PARAM_COUNT = 36u;
static const float SANITIZE_HUGE_VALUE = 1e8;

uniform StructuredBuffer<float> g_Params;
uniform RWStructuredBuffer<float> g_Grads;
uniform RWStructuredBuffer<float> g_LossSum;

uniform StructuredBuffer<float4> g_FramePixels;
uniform StructuredBuffer<uint4> g_FrameInfo;

uniform StructuredBuffer<uint> g_PairFrameIndicesA;
uniform StructuredBuffer<uint> g_PairFrameIndicesB;
uniform StructuredBuffer<float2> g_PairXYA;
uniform StructuredBuffer<float2> g_PairXYB;
uniform StructuredBuffer<uint> g_FramePairRanges;
uniform StructuredBuffer<uint2> g_FramePairEntries;

uniform uint g_FrameCount;
uniform uint g_PairCount;
uniform uint g_NeighborhoodRadius;

groupshared float gsThreadGrads[PHOTOMETRIC_THREADS * PPISP_PARAM_COUNT];
groupshared float gsThreadLoss[PHOTOMETRIC_THREADS];

struct FrameImageInfo
{
    uint offset;
    uint width;
    uint height;
};

FrameImageInfo load_frame_image_info(uint frameIndex)
{
    uint4 raw = g_FrameInfo[frameIndex];
    FrameImageInfo info;
    info.offset = raw.x;
    info.width = raw.y;
    info.height = raw.z;
    return info;
}

float load_param(uint frameIndex, uint paramId)
{
    return g_Params[paramId * g_FrameCount + frameIndex];
}

PPISPTonemap load_ppisp(uint frameIndex)
{
    PPISPTonemap params;
    params.exposureEv = load_param(frameIndex, 0u);
    params.vignetteCenterX = float3(load_param(frameIndex, 1u), load_param(frameIndex, 2u), load_param(frameIndex, 3u));
    params.vignetteCenterY = float3(load_param(frameIndex, 4u), load_param(frameIndex, 5u), load_param(frameIndex, 6u));
    params.vignetteCoeffR2 = float3(load_param(frameIndex, 7u), load_param(frameIndex, 8u), load_param(frameIndex, 9u));
    params.vignetteCoeffR4 = float3(load_param(frameIndex, 10u), load_param(frameIndex, 11u), load_param(frameIndex, 12u));
    params.vignetteCoeffR6 = float3(load_param(frameIndex, 13u), load_param(frameIndex, 14u), load_param(frameIndex, 15u));
    params.chromaOffsetR = float2(load_param(frameIndex, 16u), load_param(frameIndex, 17u));
    params.chromaOffsetG = float2(load_param(frameIndex, 18u), load_param(frameIndex, 19u));
    params.chromaOffsetB = float2(load_param(frameIndex, 20u), load_param(frameIndex, 21u));
    params.chromaOffsetW = float2(load_param(frameIndex, 22u), load_param(frameIndex, 23u));
    params.crfTau = float3(load_param(frameIndex, 24u), load_param(frameIndex, 25u), load_param(frameIndex, 26u));
    params.crfEta = float3(load_param(frameIndex, 27u), load_param(frameIndex, 28u), load_param(frameIndex, 29u));
    params.crfXi = float3(load_param(frameIndex, 30u), load_param(frameIndex, 31u), load_param(frameIndex, 32u));
    params.crfGamma = float3(load_param(frameIndex, 33u), load_param(frameIndex, 34u), load_param(frameIndex, 35u));
    return params;
}

PPISPTonemap.Differential zero_ppisp_differential()
{
    PPISPTonemap.Differential diff;
    diff.exposureEv = 0.0;
    diff.vignetteCenterX = 0.0;
    diff.vignetteCenterY = 0.0;
    diff.vignetteCoeffR2 = 0.0;
    diff.vignetteCoeffR4 = 0.0;
    diff.vignetteCoeffR6 = 0.0;
    diff.chromaOffsetR = 0.0;
    diff.chromaOffsetG = 0.0;
    diff.chromaOffsetB = 0.0;
    diff.chromaOffsetW = 0.0;
    diff.crfTau = 0.0;
    diff.crfEta = 0.0;
    diff.crfXi = 0.0;
    diff.crfGamma = 0.0;
    return diff;
}

void clear_local_grads(inout float grads[PPISP_PARAM_COUNT])
{
    [unroll] for (uint paramId = 0u; paramId < PPISP_PARAM_COUNT; paramId++) grads[paramId] = 0.0;
}

void add_local_ppisp_grad(inout float grads[PPISP_PARAM_COUNT], PPISPTonemap.Differential diff)
{
    grads[0u] += diff.exposureEv;
    grads[1u] += diff.vignetteCenterX.x;
    grads[2u] += diff.vignetteCenterX.y;
    grads[3u] += diff.vignetteCenterX.z;
    grads[4u] += diff.vignetteCenterY.x;
    grads[5u] += diff.vignetteCenterY.y;
    grads[6u] += diff.vignetteCenterY.z;
    grads[7u] += diff.vignetteCoeffR2.x;
    grads[8u] += diff.vignetteCoeffR2.y;
    grads[9u] += diff.vignetteCoeffR2.z;
    grads[10u] += diff.vignetteCoeffR4.x;
    grads[11u] += diff.vignetteCoeffR4.y;
    grads[12u] += diff.vignetteCoeffR4.z;
    grads[13u] += diff.vignetteCoeffR6.x;
    grads[14u] += diff.vignetteCoeffR6.y;
    grads[15u] += diff.vignetteCoeffR6.z;
    grads[16u] += diff.chromaOffsetR.x;
    grads[17u] += diff.chromaOffsetR.y;
    grads[18u] += diff.chromaOffsetG.x;
    grads[19u] += diff.chromaOffsetG.y;
    grads[20u] += diff.chromaOffsetB.x;
    grads[21u] += diff.chromaOffsetB.y;
    grads[22u] += diff.chromaOffsetW.x;
    grads[23u] += diff.chromaOffsetW.y;
    grads[24u] += diff.crfTau.x;
    grads[25u] += diff.crfTau.y;
    grads[26u] += diff.crfTau.z;
    grads[27u] += diff.crfEta.x;
    grads[28u] += diff.crfEta.y;
    grads[29u] += diff.crfEta.z;
    grads[30u] += diff.crfXi.x;
    grads[31u] += diff.crfXi.y;
    grads[32u] += diff.crfXi.z;
    grads[33u] += diff.crfGamma.x;
    grads[34u] += diff.crfGamma.y;
    grads[35u] += diff.crfGamma.z;
}

uint2 clamp_frame_pixel(FrameImageInfo info, int2 pixel)
{
    return uint2(
        uint(clamp(pixel.x, 0, int(info.width) - 1)),
        uint(clamp(pixel.y, 0, int(info.height) - 1))
    );
}

int2 neighborhood_center(float2 xy)
{
    return int2(int(floor(xy.x)), int(floor(xy.y)));
}

float4 load_frame_pixel(FrameImageInfo info, uint2 pixel)
{
    return sanitize_float4(g_FramePixels[info.offset + pixel.y * info.width + pixel.x], SANITIZE_HUGE_VALUE);
}

[Differentiable]
float3 photometric_apply_tonemap(no_diff float3 radiance, no_diff float2 sensorCoord, PPISPTonemap params)
{
    return ppisp_apply_tonemap(radiance, sensorCoord, params);
}

[shader("compute")]
[numthreads(PHOTOMETRIC_THREADS, 1, 1)]
void csPhotometricPairLossBackward(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID)
{
    uint frameIndex = groupID.x;
    uint localId = groupThreadID.x;
    if (frameIndex >= g_FrameCount) return;

    uint start = g_FramePairRanges[frameIndex];
    uint stop = g_FramePairRanges[frameIndex + 1u];
    float lossScale = 1.0 / max(float(g_PairCount), 1.0);
    PPISPTonemap currentParams = load_ppisp(frameIndex);
    FrameImageInfo currentInfo = load_frame_image_info(frameIndex);
    float localGrads[PPISP_PARAM_COUNT];
    clear_local_grads(localGrads);
    float localLoss = 0.0;

    for (uint entryIndex = start + localId; entryIndex < stop; entryIndex += PHOTOMETRIC_THREADS)
    {
        uint2 entry = g_FramePairEntries[entryIndex];
        uint pairIndex = entry.x;
        bool currentIsB = entry.y != 0u;
        uint otherFrameIndex = currentIsB ? g_PairFrameIndicesA[pairIndex] : g_PairFrameIndicesB[pairIndex];
        PPISPTonemap otherParams = load_ppisp(otherFrameIndex);
        FrameImageInfo otherInfo = load_frame_image_info(otherFrameIndex);
        float2 currentXY = currentIsB ? g_PairXYB[pairIndex] : g_PairXYA[pairIndex];
        float2 otherXY = currentIsB ? g_PairXYA[pairIndex] : g_PairXYB[pairIndex];
        int2 currentCenter = neighborhood_center(currentXY);
        int2 otherCenter = neighborhood_center(otherXY);
        float3 currentMean = 0.0;
        float3 otherMean = 0.0;
        float sampleCount = 0.0;

        for (int y = -int(g_NeighborhoodRadius); y <= int(g_NeighborhoodRadius); y++)
        {
            for (int x = -int(g_NeighborhoodRadius); x <= int(g_NeighborhoodRadius); x++)
            {
                uint2 currentPixel = clamp_frame_pixel(currentInfo, currentCenter + int2(x, y));
                uint2 otherPixel = clamp_frame_pixel(otherInfo, otherCenter + int2(x, y));
                float4 currentSample = load_frame_pixel(currentInfo, currentPixel);
                float4 otherSample = load_frame_pixel(otherInfo, otherPixel);
                float2 currentSensorCoord = float2((float(currentPixel.x) + 0.5) / max(float(currentInfo.width), 1.0), (float(currentPixel.y) + 0.5) / max(float(currentInfo.height), 1.0));
                float2 otherSensorCoord = float2((float(otherPixel.x) + 0.5) / max(float(otherInfo.width), 1.0), (float(otherPixel.y) + 0.5) / max(float(otherInfo.height), 1.0));
                currentMean += photometric_apply_tonemap(max(currentSample.xyz, 0.0), currentSensorCoord, currentParams);
                otherMean += photometric_apply_tonemap(max(otherSample.xyz, 0.0), otherSensorCoord, otherParams);
                sampleCount += 1.0;
            }
        }

        currentMean /= max(sampleCount, 1.0);
        otherMean /= max(sampleCount, 1.0);
        float3 meanDiff = currentMean - otherMean;
        float pairLoss = dot(abs(meanDiff), float3(1.0 / 3.0));
        float3 dCurrentMean = sign(meanDiff) * (lossScale / 3.0);
        float3 dCurrentSample = dCurrentMean / max(sampleCount, 1.0);

        for (int y = -int(g_NeighborhoodRadius); y <= int(g_NeighborhoodRadius); y++)
        {
            for (int x = -int(g_NeighborhoodRadius); x <= int(g_NeighborhoodRadius); x++)
            {
                uint2 currentPixel = clamp_frame_pixel(currentInfo, currentCenter + int2(x, y));
                float4 currentSample = load_frame_pixel(currentInfo, currentPixel);
                float2 currentSensorCoord = float2((float(currentPixel.x) + 0.5) / max(float(currentInfo.width), 1.0), (float(currentPixel.y) + 0.5) / max(float(currentInfo.height), 1.0));
                DifferentialPair<PPISPTonemap> dpCurrent = diffPair(currentParams, zero_ppisp_differential());
                bwd_diff(photometric_apply_tonemap)(max(currentSample.xyz, 0.0), currentSensorCoord, dpCurrent, dCurrentSample);
                add_local_ppisp_grad(localGrads, dpCurrent.d);
            }
        }

        if (!currentIsB) localLoss += pairLoss * lossScale;
    }

    [unroll] for (uint paramId = 0u; paramId < PPISP_PARAM_COUNT; paramId++) gsThreadGrads[localId * PPISP_PARAM_COUNT + paramId] = localGrads[paramId];
    gsThreadLoss[localId] = localLoss;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = PHOTOMETRIC_THREADS / 2u; stride > 0u; stride >>= 1u)
    {
        if (localId < stride)
        {
            uint base = localId * PPISP_PARAM_COUNT;
            uint otherBase = (localId + stride) * PPISP_PARAM_COUNT;
            [unroll] for (uint paramId = 0u; paramId < PPISP_PARAM_COUNT; paramId++) gsThreadGrads[base + paramId] += gsThreadGrads[otherBase + paramId];
            gsThreadLoss[localId] += gsThreadLoss[localId + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (localId < PPISP_PARAM_COUNT) g_Grads[localId * g_FrameCount + frameIndex] = gsThreadGrads[localId];
    if (localId == 0u && gsThreadLoss[0] != 0.0) InterlockedAdd(g_LossSum[0], gsThreadLoss[0]);
}