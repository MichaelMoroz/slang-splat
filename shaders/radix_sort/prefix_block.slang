// Hierarchical prefix sum - Level 0 uses packed 2x16-bit, higher levels use 32-bit
import "common.slang";

uniform uint g_totalN, g_levelSize, g_levelOffset, g_prevLevelOffset, g_prevLevelSize;
uniform uint g_isFirstLevel, g_packedHistN, g_prevLevelPacked;
uniform StructuredBuffer<uint> g_Histogram;
uniform RWStructuredBuffer<uint> g_Prefix;
groupshared uint gs_prefix[BLOCK_SIZE];

uint readInput(uint i, uint numGroups, uint totalN, uint prevLevelSize) {
    if (g_isFirstLevel != 0) {
        uint digit = i / numGroups, group = i % numGroups;
        return unpackCount(g_Histogram[(digit >> 2) * numGroups + group], digit);
    }
    uint blockEnd = min((i + 1) * BLOCK_SIZE, prevLevelSize) - 1;
    if (g_prevLevelPacked != 0) return unpackPrefix16(g_Prefix[g_prevLevelOffset + (blockEnd >> 1)], blockEnd & 1);
    return g_Prefix[g_prevLevelOffset + blockEnd];
}

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void csPrefixBlock(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID) {
    uint block = groupID.x, lane = localID.x, numGroups = getNumGroups(), totalN = getTotalN();
    uint prevLevelSize = (g_isFirstLevel != 0) ? totalN : g_prevLevelSize;
    uint globalIdx = block * BLOCK_SIZE + lane;
    gs_prefix[lane] = (globalIdx < g_levelSize) ? readInput(globalIdx, numGroups, totalN, prevLevelSize) : 0;
    GroupMemoryBarrierWithGroupSync();

    [unroll] for (uint stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
        uint val = (lane >= stride) ? gs_prefix[lane - stride] : 0;
        GroupMemoryBarrierWithGroupSync();
        gs_prefix[lane] += val;
        GroupMemoryBarrierWithGroupSync();
    }

    if (globalIdx >= g_levelSize) return;
    if (g_isFirstLevel != 0) {
        // Level 0: packed 2x16-bit output
        if ((lane & 1) == 0) {
            uint hi = (lane + 1 < BLOCK_SIZE && globalIdx + 1 < g_levelSize) ? gs_prefix[lane + 1] : 0;
            g_Prefix[g_levelOffset + (globalIdx >> 1)] = packPrefix16(gs_prefix[lane], hi);
        }
    } else {
        g_Prefix[g_levelOffset + globalIdx] = gs_prefix[lane];
    }
}
