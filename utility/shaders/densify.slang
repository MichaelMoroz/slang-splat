import "slangpy";
import "common.slang";

static const uint DENSIFY_THREADS = 256u;

uniform Tensor<float, 1> g_DensifyBaseXYZ;
uniform Tensor<float, 1> g_DensifyBaseScale;
uniform Tensor<float, 1> g_DensifyBaseRotation;
uniform Tensor<uint, 1> g_DensifySplitCounts;
uniform Tensor<uint, 1> g_DensifySourceIds;
uniform Tensor<uint, 1> g_DensifyFamilySlots;
uniform RWTensor<float, 1> g_DensifyXYZOut;
uniform RWTensor<float, 1> g_DensifyLogScaleOut;
uniform uint g_DensifyOutputCount;
uniform uint g_DensifyIteration;
uniform float g_DensifyMinScale;

float3 densify_read3(Tensor<float, 1> buffer, uint index)
{
    uint base = index * 3u;
    return float3(buffer[base], buffer[base + 1u], buffer[base + 2u]);
}

float4 densify_read4(Tensor<float, 1> buffer, uint index)
{
    uint base = index * 4u;
    return float4(buffer[base], buffer[base + 1u], buffer[base + 2u], buffer[base + 3u]);
}

void densify_write3(RWTensor<float, 1> buffer, uint index, float3 value)
{
    uint base = index * 3u;
    buffer[base] = value.x;
    buffer[base + 1u] = value.y;
    buffer[base + 2u] = value.z;
}

float densify_hash_to_unit(float value)
{
    return frac(sin(value) * 43758.5453123);
}

float3 densify_hash_raw(uint parentId, uint familySlot, uint iteration)
{
    float seed = float(familySlot) + float(parentId + 1u) * 0.7548776662466927 + float(iteration + 1u) * 0.5698402909980532;
    return float3(
        densify_hash_to_unit(seed * 12.9898 + 78.233),
        densify_hash_to_unit(seed * 39.3467 + 11.135),
        densify_hash_to_unit(seed * 73.1569 + 53.726)
    ) * 2.0 - 1.0;
}

float3 densify_fallback_raw(uint familySlot)
{
    switch (familySlot & 7u)
    {
    case 0u: return float3(1.0, 0.0, 0.0);
    case 1u: return float3(-1.0, 0.0, 0.0);
    case 2u: return float3(0.0, 1.0, 0.0);
    case 3u: return float3(0.0, -1.0, 0.0);
    case 4u: return float3(0.0, 0.0, 1.0);
    case 5u: return float3(0.0, 0.0, -1.0);
    case 6u: return float3(1.0, 1.0, 0.0);
    default: return float3(-1.0, 1.0, 0.0);
    }
}

float3 densify_centered_direction(uint parentId, uint familySlot, uint familySize, uint iteration, bool useFallback)
{
    float3 mean = 0.0;
    [loop]
    for (uint slot = 0u; slot < familySize; ++slot)
    {
        mean += useFallback ? densify_fallback_raw(slot) : densify_hash_raw(parentId, slot, iteration);
    }
    mean /= max(float(familySize), 1.0);

    float meanNorm = 0.0;
    [loop]
    for (uint slot = 0u; slot < familySize; ++slot)
    {
        float3 centered = (useFallback ? densify_fallback_raw(slot) : densify_hash_raw(parentId, slot, iteration)) - mean;
        meanNorm += length(centered);
    }
    meanNorm = max(meanNorm / max(float(familySize), 1.0), 1e-8);

    float3 raw = useFallback ? densify_fallback_raw(familySlot) : densify_hash_raw(parentId, familySlot, iteration);
    return (raw - mean) / meanNorm;
}

float3 densify_family_direction(uint parentId, uint familySlot, uint familySize, uint iteration)
{
    float3 hashed = densify_centered_direction(parentId, familySlot, familySize, iteration, false);
    return length(hashed) > 1e-6 ? hashed : densify_centered_direction(parentId, familySlot, familySize, iteration, true);
}

float3 quat_rotate(float4 quat, float3 value)
{
    float3 qv = quat.yzw;
    return value + 2.0 * cross(qv, cross(qv, value) + quat.x * value);
}

[shader("compute")]
[numthreads(DENSIFY_THREADS, 1, 1)]
void csBuildDensifyFamilySplats(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint outputIndex = dispatchThreadID.x;
    if (outputIndex >= g_DensifyOutputCount) return;

    uint sourceId = g_DensifySourceIds[outputIndex];
    uint familySlot = g_DensifyFamilySlots[outputIndex];
    uint familySize = g_DensifySplitCounts[sourceId] + 1u;
    float shrink = pow(float(familySize), -1.0 / 3.0);
    float3 baseXYZ = densify_read3(g_DensifyBaseXYZ, sourceId);
    float3 baseScale = densify_read3(g_DensifyBaseScale, sourceId);
    float3 shrunkScale = max(baseScale * shrink, g_DensifyMinScale);
    float3 offsetSigma = baseScale * sqrt(max(1.0 - shrink * shrink, 0.0));
    float3 localOffset = densify_family_direction(sourceId, familySlot, familySize, g_DensifyIteration) * offsetSigma;
    float3 worldOffset = quat_rotate(densify_read4(g_DensifyBaseRotation, sourceId), localOffset);

    densify_write3(g_DensifyXYZOut, outputIndex, baseXYZ + worldOffset);
    densify_write3(g_DensifyLogScaleOut, outputIndex, log(shrunkScale));
}
