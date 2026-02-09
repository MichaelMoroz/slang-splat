// Scatter: write keys/values to sorted positions using hierarchical prefix lookup
import "common.slang";

static const uint QUARTER_SIZE = GROUP_SIZE / 4;

uniform uint g_totalN;      // numGroups * HISTOGRAM_SIZE
uniform uint g_numLevels;   // Number of hierarchy levels

uniform StructuredBuffer<uint> g_KeysIn;
uniform StructuredBuffer<uint> g_ValuesIn;
uniform StructuredBuffer<uint> g_Prefix;  // Hybrid: L0 packed 2x16, L1+ unpacked 32
uniform RWStructuredBuffer<uint> g_KeysOut;
uniform RWStructuredBuffer<uint> g_ValuesOut;

groupshared uint tempBits[GROUP_SIZE];
groupshared uint halfCount[HISTOGRAM_SIZE];

// Get global offset for (digit, group) using hierarchical lookup
// Level 0: packed 2x16-bit, higher levels: unpacked 32-bit
uint getGlobalOffset(uint digit, uint group, uint numGroups, uint totalN, uint numLevels) {
    uint linearIdx = digit * numGroups + group;
    uint offset = 0, lvlOff = 0, lvlSz = totalN, idx = linearIdx;
    for (uint lvl = 0; lvl < numLevels; lvl++) {
        if (idx % BLOCK_SIZE != 0) {
            uint prev = idx - 1;
            if (lvl == 0) offset += unpackPrefix16(g_Prefix[lvlOff + prev / 2], prev & 1);
            else offset += g_Prefix[lvlOff + prev];
        }
        lvlOff += (lvl == 0) ? ((lvlSz + 1) / 2) : lvlSz;  // L0 packed, others unpacked
        lvlSz = (lvlSz + BLOCK_SIZE - 1) / BLOCK_SIZE;
        idx /= BLOCK_SIZE;
    }
    return offset;
}

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void csScatter(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID) {
    uint group = groupID.x, lane = localID.x;
    uint elementCount = getElementCount(), numGroups = getNumGroups(), totalN = getTotalN(), numLevels = getNumLevels();
    if (g_numLevels != 0) numLevels = g_numLevels;  // Prefer uniform if set
    if (g_totalN != 0) totalN = g_totalN;
    [unroll] for (uint idx = lane; idx < HISTOGRAM_SIZE; idx += GROUP_SIZE) halfCount[idx] = 0;
    
    uint globalIndex = group * GROUP_SIZE + lane;
    bool active = globalIndex < elementCount;
    uint bit = 0, key = 0, value = 0;
    if (active) {
        key = g_KeysIn[globalIndex];
        bit = (key >> g_shift) & (HISTOGRAM_SIZE - 1);
        tempBits[lane] = bit;
        value = g_ValuesIn[globalIndex];
    } else { tempBits[lane] = 0xFFFFFFFF; }
    GroupMemoryBarrierWithGroupSync();
    
    // Count elements in first 3 quarters using packed counters
    uint quarterIndex = lane / QUARTER_SIZE;
    if (active && quarterIndex < 3) {
        uint inc = 0;
        if (quarterIndex < 1) inc |= 1u;
        if (quarterIndex < 2) inc |= (1u << 8);
        if (quarterIndex < 3) inc |= (1u << 16);
        InterlockedAdd(halfCount[bit], inc);
    }
    GroupMemoryBarrierWithGroupSync();
    
    if (active) {
        uint globalOff = getGlobalOffset(bit, group, numGroups, totalN, numLevels);
        uint quarterOffset = 0;
        if (quarterIndex > 0) {
            uint packed = halfCount[bit];
            uint shiftBytes = (quarterIndex - 1u) * 8u;
            quarterOffset = (packed >> shiftBytes) & 0xFFu;
        }
        uint begin = quarterIndex * QUARTER_SIZE, localCount = 0;
        [unroll] for (uint t = begin; t < lane; ++t) localCount += (tempBits[t] == bit) ? 1u : 0u;
        uint dest = globalOff + quarterOffset + localCount;
        g_KeysOut[dest] = key;
        g_ValuesOut[dest] = value;
    }
}
