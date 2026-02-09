// Histogram: count occurrences of each digit per group (packed 4x8-bit)
import "common.slang";

uniform StructuredBuffer<uint> g_Keys;
uniform RWStructuredBuffer<uint> g_Histogram;  // Layout: [digit/4 * numGroups + group]
groupshared uint localHist[PACKED_HIST_SIZE];

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void csHistogram(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID) {
    uint group = groupID.x, lane = localID.x, numGroups = getNumGroups(), elementCount = getElementCount();
    if (lane < PACKED_HIST_SIZE) localHist[lane] = 0;
    GroupMemoryBarrierWithGroupSync();
    uint idx = group * GROUP_SIZE + lane;
    if (idx < elementCount) {
        uint digit = (g_Keys[idx] >> g_shift) & 0xFF;
        InterlockedAdd(localHist[digit >> 2], packCount(digit, 1));
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane < PACKED_HIST_SIZE) g_Histogram[lane * numGroups + group] = localHist[lane];
}
