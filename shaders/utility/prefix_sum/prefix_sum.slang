groupshared float gsScanFloat[256];
groupshared uint gsScanUint[256];

uniform uint g_ScanElementCount;
uniform uint g_ClearUIntCount;
uniform uint g_ClearUIntValue;
uniform RWStructuredBuffer<uint> g_ClearUIntBuffer;
uniform StructuredBuffer<float> g_ScanInput;
uniform RWStructuredBuffer<float> g_ScanOutput;
uniform RWStructuredBuffer<float> g_ScanBlockSums;
uniform StructuredBuffer<float> g_ScanBlockPrefix;
uniform StructuredBuffer<uint> g_ScanUIntInput;
uniform RWStructuredBuffer<uint> g_ScanUIntOutput;
uniform RWStructuredBuffer<uint> g_ScanUIntBlockSums;
uniform StructuredBuffer<uint> g_ScanUIntBlockPrefix;
uniform RWStructuredBuffer<float> g_FloatTotalOut;

[shader("compute")]
[numthreads(256, 1, 1)]
void csClearUIntBuffer(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint index = dispatchThreadID.x;
    if (index >= g_ClearUIntCount) return;
    g_ClearUIntBuffer[index] = g_ClearUIntValue;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csScanFloatBlocks(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID)
{
    uint globalIndex = dispatchThreadID.x;
    uint lane = groupThreadID.x;
    gsScanFloat[lane] = (globalIndex < g_ScanElementCount) ? g_ScanInput[globalIndex] : 0.0;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = 1u; stride < 256u; stride <<= 1u)
    {
        float addend = (lane >= stride) ? gsScanFloat[lane - stride] : 0.0;
        GroupMemoryBarrierWithGroupSync();
        gsScanFloat[lane] += addend;
        GroupMemoryBarrierWithGroupSync();
    }
    if (globalIndex < g_ScanElementCount) g_ScanOutput[globalIndex] = gsScanFloat[lane];
    if (lane == 255u) g_ScanBlockSums[groupID.x] = gsScanFloat[255u];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csScanUIntBlocks(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID)
{
    uint globalIndex = dispatchThreadID.x;
    uint lane = groupThreadID.x;
    gsScanUint[lane] = (globalIndex < g_ScanElementCount) ? g_ScanUIntInput[globalIndex] : 0u;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = 1u; stride < 256u; stride <<= 1u)
    {
        uint addend = (lane >= stride) ? gsScanUint[lane - stride] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gsScanUint[lane] += addend;
        GroupMemoryBarrierWithGroupSync();
    }
    if (globalIndex < g_ScanElementCount) g_ScanUIntOutput[globalIndex] = gsScanUint[lane];
    if (lane == 255u) g_ScanUIntBlockSums[groupID.x] = gsScanUint[255u];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csAddFloatBlockOffsets(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint index = dispatchThreadID.x;
    if (index >= g_ScanElementCount) return;
    uint blockId = index >> 8;
    if (blockId == 0u) return;
    g_ScanOutput[index] += g_ScanBlockPrefix[blockId - 1u];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csAddUIntBlockOffsets(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint index = dispatchThreadID.x;
    if (index >= g_ScanElementCount) return;
    uint blockId = index >> 8;
    if (blockId == 0u) return;
    g_ScanUIntOutput[index] += g_ScanUIntBlockPrefix[blockId - 1u];
}

[shader("compute")]
[numthreads(1, 1, 1)]
void csWriteFloatScanTotal(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    g_FloatTotalOut[0] = (g_ScanElementCount == 0u) ? 0.0 : g_ScanOutput[g_ScanElementCount - 1u];
}
