import "utility/math/constants.slang";
import "utility/math/numeric.slang";
import "utility/ppisp_tonemap.slang";

static const uint PHOTOMETRIC_THREADS = 64u;
static const uint PHOTOMETRIC_TARGET_TILE_DIM = 16u;
static const uint PPISP_PARAM_COUNT = 36u;
static const float SANITIZE_HUGE_VALUE = 1e8;

uniform StructuredBuffer<float> g_Params;
uniform RWStructuredBuffer<float> g_Grads;
uniform RWStructuredBuffer<float> g_LossSum;

uniform StructuredBuffer<uint> g_PairIndices;
uniform StructuredBuffer<uint> g_PairFrameIndicesA;
uniform StructuredBuffer<uint> g_PairFrameIndicesB;
uniform RWStructuredBuffer<float4> g_PairSamplesA;
uniform RWStructuredBuffer<float4> g_PairSamplesB;
uniform RWStructuredBuffer<float2> g_PairSensorCoordsA;
uniform RWStructuredBuffer<float2> g_PairSensorCoordsB;
uniform StructuredBuffer<uint> g_FramePairRanges;
uniform StructuredBuffer<uint2> g_FramePairEntries;

uniform uint g_FrameCount;
uniform uint g_PairCount;
uniform uint g_NeighborhoodSize;
uniform uint g_NeighborhoodSampleCount;

uniform Texture2D<float4> g_PairDatasetSourceFrame;
uniform StructuredBuffer<float2> g_PairXYA;
uniform StructuredBuffer<float2> g_PairXYB;
uniform StructuredBuffer<uint> g_DatasetFramePairRanges;
uniform StructuredBuffer<uint2> g_DatasetFramePairEntries;
uniform uint g_DatasetFrameIndex;
uniform uint g_DatasetFrameWidth;
uniform uint g_DatasetFrameHeight;

uniform Texture2D<float4> g_PhotometricSourceTarget;
uniform RWTexture2D<float4> g_PhotometricTarget;
uniform uint g_PhotometricSourceWidth;
uniform uint g_PhotometricSourceHeight;
uniform uint g_PhotometricTargetWidth;
uniform uint g_PhotometricTargetHeight;
uniform PPISPTonemap g_PhotometricTargetPPISPParams;

groupshared float gsThreadGrads[PHOTOMETRIC_THREADS * PPISP_PARAM_COUNT];
groupshared float gsThreadLoss[PHOTOMETRIC_THREADS];

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

float4 load_pair_sample(uint pairIndex, uint sampleIndex, bool sampleFromB)
{
    uint offset = pairIndex * g_NeighborhoodSampleCount + sampleIndex;
    return sanitize_float4(sampleFromB ? g_PairSamplesB[offset] : g_PairSamplesA[offset], SANITIZE_HUGE_VALUE);
}

float2 load_pair_sensor_coord(uint pairIndex, uint sampleIndex, bool sampleFromB)
{
    uint offset = pairIndex * g_NeighborhoodSampleCount + sampleIndex;
    return sampleFromB ? g_PairSensorCoordsB[offset] : g_PairSensorCoordsA[offset];
}

[shader("compute")]
[numthreads(PHOTOMETRIC_THREADS, 1, 1)]
void csBuildPhotometricPairDataset(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint start = g_DatasetFramePairRanges[g_DatasetFrameIndex];
    uint stop = g_DatasetFramePairRanges[g_DatasetFrameIndex + 1u];
    uint entryCount = stop - start;
    uint threadIndex = dispatchThreadID.x;
    uint totalSamples = entryCount * g_NeighborhoodSampleCount;
    if (threadIndex >= totalSamples) return;

    uint localEntryIndex = threadIndex / g_NeighborhoodSampleCount;
    uint sampleIndex = threadIndex % g_NeighborhoodSampleCount;
    uint2 entry = g_DatasetFramePairEntries[start + localEntryIndex];
    uint pairIndex = entry.x;
    bool sampleFromB = entry.y != 0u;
    float2 center = sampleFromB ? g_PairXYB[pairIndex] : g_PairXYA[pairIndex];
    uint neighborhoodSize = max(g_NeighborhoodSize, 1u);
    uint radius = neighborhoodSize / 2u;
    int2 basePixel = int2(floor(center));
    int dx = int(sampleIndex % neighborhoodSize) - int(radius);
    int dy = int(sampleIndex / neighborhoodSize) - int(radius);
    uint width = max(g_DatasetFrameWidth, 1u);
    uint height = max(g_DatasetFrameHeight, 1u);
    uint px = uint(clamp(basePixel.x + dx, 0, int(width) - 1));
    uint py = uint(clamp(basePixel.y + dy, 0, int(height) - 1));
    uint2 pixel = uint2(px, py);
    uint pairSampleOffset = pairIndex * g_NeighborhoodSampleCount + sampleIndex;
    float4 sample = sanitize_float4(g_PairDatasetSourceFrame[pixel], SANITIZE_HUGE_VALUE);
    float2 sensorCoord = float2(
        (float(px) + 0.5) / max(float(width), 1.0),
        (float(py) + 0.5) / max(float(height), 1.0)
    );

    if (sampleFromB) {
        g_PairSamplesB[pairSampleOffset] = sample;
        g_PairSensorCoordsB[pairSampleOffset] = sensorCoord;
    } else {
        g_PairSamplesA[pairSampleOffset] = sample;
        g_PairSensorCoordsA[pairSampleOffset] = sensorCoord;
    }
}

[Differentiable]
float3 photometric_apply_tonemap(no_diff float3 radiance, no_diff float2 sensorCoord, PPISPTonemap params)
{
    return ppisp_apply_tonemap(radiance, sensorCoord, params);
}

float3 photometric_target_to_linear_rgb(float3 target)
{
    return pow(max(target, 0.0), 1.0 / OUTPUT_GAMMA);
}

[shader("compute")]
[numthreads(PHOTOMETRIC_TARGET_TILE_DIM, PHOTOMETRIC_TARGET_TILE_DIM, 1)]
void csApplyPhotometricTargetTonemap(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= g_PhotometricTargetWidth || py >= g_PhotometricTargetHeight) return;

    float4 sample = sanitize_float4(g_PhotometricSourceTarget[uint2(px, py)], SANITIZE_HUGE_VALUE);
    float2 sensorCoord = float2(
        (float(px) + 0.5) / max(float(g_PhotometricSourceWidth), 1.0),
        (float(py) + 0.5) / max(float(g_PhotometricSourceHeight), 1.0)
    );
    sample.xyz = photometric_target_to_linear_rgb(ppisp_apply_tonemap(max(sample.xyz, 0.0), sensorCoord, g_PhotometricTargetPPISPParams));
    g_PhotometricTarget[uint2(px, py)] = sample;
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
    float localGrads[PPISP_PARAM_COUNT];
    clear_local_grads(localGrads);
    float localLoss = 0.0;

    for (uint entryIndex = start + localId; entryIndex < stop; entryIndex += PHOTOMETRIC_THREADS)
    {
        uint2 entry = g_FramePairEntries[entryIndex];
        uint batchPairIndex = entry.x;
        uint pairIndex = g_PairIndices[batchPairIndex];
        bool currentIsB = entry.y != 0u;
        uint otherFrameIndex = currentIsB ? g_PairFrameIndicesA[batchPairIndex] : g_PairFrameIndicesB[batchPairIndex];
        PPISPTonemap otherParams = load_ppisp(otherFrameIndex);
        float3 currentMean = 0.0;
        float3 otherMean = 0.0;
        float sampleCount = max(float(g_NeighborhoodSampleCount), 1.0);

        for (uint sampleIndex = 0u; sampleIndex < g_NeighborhoodSampleCount; sampleIndex++)
        {
            float4 currentSample = load_pair_sample(pairIndex, sampleIndex, currentIsB);
            float4 otherSample = load_pair_sample(pairIndex, sampleIndex, !currentIsB);
            float2 currentSensorCoord = load_pair_sensor_coord(pairIndex, sampleIndex, currentIsB);
            float2 otherSensorCoord = load_pair_sensor_coord(pairIndex, sampleIndex, !currentIsB);
            currentMean += photometric_apply_tonemap(max(currentSample.xyz, 0.0), currentSensorCoord, currentParams);
            otherMean += photometric_apply_tonemap(max(otherSample.xyz, 0.0), otherSensorCoord, otherParams);
        }

        currentMean /= sampleCount;
        otherMean /= sampleCount;
        float3 meanDiff = currentMean - otherMean;
        float pairLoss = dot(abs(meanDiff), float3(1.0 / 3.0));
        float3 dCurrentMean = sign(meanDiff) * (lossScale / 3.0);
        float3 dCurrentSample = dCurrentMean / sampleCount;

        for (uint sampleIndex = 0u; sampleIndex < g_NeighborhoodSampleCount; sampleIndex++)
        {
            float4 currentSample = load_pair_sample(pairIndex, sampleIndex, currentIsB);
            float2 currentSensorCoord = load_pair_sensor_coord(pairIndex, sampleIndex, currentIsB);
            DifferentialPair<PPISPTonemap> dpCurrent = diffPair(currentParams, zero_ppisp_differential());
            bwd_diff(photometric_apply_tonemap)(max(currentSample.xyz, 0.0), currentSensorCoord, dpCurrent, dCurrentSample);
            add_local_ppisp_grad(localGrads, dpCurrent.d);
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