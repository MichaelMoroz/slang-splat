import "common.slang";

uniform uint g_level;
uniform StructuredBuffer<uint> g_Histogram;
uniform RWStructuredBuffer<uint> g_Prefix;
groupshared uint gs_prefix[BLOCK_SIZE];

struct PrefixLevelState
{
    uint level;
    uint numGroups;
    uint totalN;

    uint level_size(uint levelIn)
    {
        uint size = totalN;
        for (uint lvl = 0u; lvl < levelIn; lvl++) size = ceil_div_block(size);
        return size;
    }

    uint level_offset(uint levelIn)
    {
        uint offset = 0u;
        uint size = totalN;
        for (uint lvl = 0u; lvl < levelIn; lvl++)
        {
            offset += (lvl == 0u) ? ceil_div2(size) : size;
            size = ceil_div_block(size);
        }
        return offset;
    }

    uint read_input(uint i)
    {
        if (level == 0u)
        {
            uint digit = i / numGroups;
            uint group = i % numGroups;
            return unpackCount(g_Histogram[(digit >> 2) * numGroups + group], digit);
        }
        uint prevLevel = level - 1u;
        uint prevLevelSize = level_size(prevLevel);
        uint prevLevelOffset = level_offset(prevLevel);
        uint blockEnd = min((i + 1u) * BLOCK_SIZE, prevLevelSize) - 1u;
        if (prevLevel == 0u) return unpackPrefix16(g_Prefix[prevLevelOffset + (blockEnd / RADIX_PAIR_SIZE)], blockEnd & 1u);
        return g_Prefix[prevLevelOffset + blockEnd];
    }
};

[shader("compute")]
[numthreads(BLOCK_SIZE, 1, 1)]
void csPrefixBlock(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID)
{
    RadixDispatchConfig cfg;
    cfg.load();
    PrefixLevelState state;
    state.level = g_level;
    state.numGroups = cfg.numGroups;
    state.totalN = cfg.totalN;
    uint block = groupID.x;
    uint lane = localID.x;
    uint levelSize = state.level_size(state.level);
    uint levelOffset = state.level_offset(state.level);
    bool isFirstLevel = state.level == 0u;
    uint globalIdx = block * BLOCK_SIZE + lane;
    gs_prefix[lane] = (globalIdx < levelSize) ? state.read_input(globalIdx) : 0u;
    GroupMemoryBarrierWithGroupSync();

    [unroll] for (uint stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
        uint val = (lane >= stride) ? gs_prefix[lane - stride] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gs_prefix[lane] += val;
        GroupMemoryBarrierWithGroupSync();
    }

    if (globalIdx >= levelSize) return;
    if (isFirstLevel)
    {
        if ((lane & 1u) == 0u)
        {
            uint hi = (lane + 1u < BLOCK_SIZE && globalIdx + 1u < levelSize) ? gs_prefix[lane + 1u] : 0u;
            g_Prefix[levelOffset + (globalIdx / RADIX_PAIR_SIZE)] = packPrefix16(gs_prefix[lane], hi);
        }
    } else {
        g_Prefix[levelOffset + globalIdx] = gs_prefix[lane];
    }
}
