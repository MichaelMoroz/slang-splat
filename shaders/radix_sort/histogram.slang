import "common.slang";

uniform StructuredBuffer<uint> g_Keys;
uniform RWStructuredBuffer<uint> g_Histogram;
groupshared uint localHist[PACKED_HIST_SIZE];

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void csHistogram(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID)
{
    RadixDispatchConfig cfg;
    cfg.load();
    uint group = groupID.x;
    uint lane = localID.x;
    if (lane < PACKED_HIST_SIZE) localHist[lane] = 0;
    GroupMemoryBarrierWithGroupSync();
    uint idx = group * GROUP_SIZE + lane;
    if (idx < cfg.elementCount)
    {
        uint digit = (g_Keys[idx] >> g_shift) & RADIX_BYTE_MASK;
        InterlockedAdd(localHist[digit >> RADIX_PACK_LANE_BITS], packCount(digit, 1u));
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane < PACKED_HIST_SIZE) g_Histogram[lane * cfg.numGroups + group] = localHist[lane];
}
