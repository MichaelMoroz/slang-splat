// Hierarchical prefix sum - Level 0 uses packed 2x16-bit, higher levels use 32-bit
import "common.slang";

uniform uint g_level;
uniform StructuredBuffer<uint> g_Histogram;
uniform RWStructuredBuffer<uint> g_Prefix;
groupshared uint gs_prefix[BLOCK_SIZE];

uint level_size(uint totalN, uint level)
{
    uint size = totalN;
    for (uint lvl = 0; lvl < level; lvl++)
    {
        size = ceil_div_block(size);
    }
    return size;
}

uint level_offset(uint totalN, uint level)
{
    uint offset = 0;
    uint size = totalN;
    for (uint lvl = 0; lvl < level; lvl++)
    {
        offset += (lvl == 0u) ? ceil_div2(size) : size;
        size = ceil_div_block(size);
    }
    return offset;
}

uint readInput(uint i, uint numGroups, uint totalN, uint level) {
    if (level == 0u) {
        uint digit = i / numGroups, group = i % numGroups;
        return unpackCount(g_Histogram[(digit >> 2) * numGroups + group], digit);
    }
    uint prevLevel = level - 1u;
    uint prevLevelSize = level_size(totalN, prevLevel);
    uint prevLevelOffset = level_offset(totalN, prevLevel);
    uint prevLevelPacked = (prevLevel == 0u) ? 1u : 0u;
    uint blockEnd = min((i + 1) * BLOCK_SIZE, prevLevelSize) - 1;
    if (prevLevelPacked != 0u) return unpackPrefix16(g_Prefix[prevLevelOffset + (blockEnd >> 1)], blockEnd & 1u);
    return g_Prefix[prevLevelOffset + blockEnd];
}

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void csPrefixBlock(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID) {
    uint block = groupID.x, lane = localID.x, numGroups = getNumGroups(), totalN = getTotalN();
    uint level = g_level;
    uint levelSize = level_size(totalN, level);
    uint levelOffset = level_offset(totalN, level);
    uint isFirstLevel = (level == 0u) ? 1u : 0u;
    uint globalIdx = block * BLOCK_SIZE + lane;
    gs_prefix[lane] = (globalIdx < levelSize) ? readInput(globalIdx, numGroups, totalN, level) : 0u;
    GroupMemoryBarrierWithGroupSync();

    [unroll] for (uint stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
        uint val = (lane >= stride) ? gs_prefix[lane - stride] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gs_prefix[lane] += val;
        GroupMemoryBarrierWithGroupSync();
    }

    if (globalIdx >= levelSize) return;
    if (isFirstLevel != 0u) {
        // Level 0: packed 2x16-bit output
        if ((lane & 1) == 0) {
            uint hi = (lane + 1u < BLOCK_SIZE && globalIdx + 1u < levelSize) ? gs_prefix[lane + 1u] : 0u;
            g_Prefix[levelOffset + (globalIdx >> 1)] = packPrefix16(gs_prefix[lane], hi);
        }
    } else {
        g_Prefix[levelOffset + globalIdx] = gs_prefix[lane];
    }
}
