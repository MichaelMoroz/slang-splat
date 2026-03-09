import "utility/radix_sort/common.slang";

static const uint QUARTER_SIZE = GROUP_SIZE / 4;
static const uint PACKED_QUARTER_COUNT = 3u;

uniform uint g_totalN;
uniform uint g_numLevels;

uniform StructuredBuffer<uint> g_KeysIn;
uniform StructuredBuffer<uint> g_ValuesIn;
uniform StructuredBuffer<uint> g_Prefix;
uniform RWStructuredBuffer<uint> g_KeysOut;
uniform RWStructuredBuffer<uint> g_ValuesOut;

groupshared uint tempBits[GROUP_SIZE];
groupshared uint halfCount[HISTOGRAM_SIZE];

struct PrefixHierarchy
{
    uint numGroups;
    uint totalN;
    uint numLevels;

    static uint derive_num_levels(uint totalN)
    {
        uint levels = 0;
        uint levelSize = totalN;
        for (uint i = 0u; i < RADIX_MAX_PREFIX_LEVELS && levelSize > 1u; ++i)
        {
            levels += 1u;
            levelSize = ceil_div_block(levelSize);
        }
        return max(levels + 1u, 1u);
    }

    uint get_global_offset(uint digit, uint group)
    {
        uint linearIdx = digit * numGroups + group;
        uint offset = 0u;
        uint lvlOff = 0u;
        uint lvlSz = totalN;
        uint idx = linearIdx;
        for (uint lvl = 0u; lvl < numLevels; lvl++)
        {
            if ((idx % BLOCK_SIZE) != 0u)
            {
                uint prev = idx - 1u;
                if (lvl == 0u) offset += unpackPrefix16(g_Prefix[lvlOff + prev / RADIX_PAIR_SIZE], prev & 1u);
                else offset += g_Prefix[lvlOff + prev];
            }
            lvlOff += (lvl == 0u) ? ceil_div2(lvlSz) : lvlSz;
            lvlSz = ceil_div_block(lvlSz);
            idx /= BLOCK_SIZE;
        }
        return offset;
    }
};

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void csScatter(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID)
{
    RadixDispatchConfig cfg;
    cfg.load();
    uint group = groupID.x;
    uint lane = localID.x;
    uint totalN = cfg.totalN;
    uint numLevels = cfg.numLevels;
    if (g_numLevels != 0u) numLevels = g_numLevels;
    if (g_totalN != 0u) totalN = g_totalN;
    if (numLevels == 0u) numLevels = PrefixHierarchy.derive_num_levels(totalN);
    PrefixHierarchy hierarchy;
    hierarchy.numGroups = cfg.numGroups;
    hierarchy.totalN = totalN;
    hierarchy.numLevels = numLevels;
    [unroll] for (uint idx = lane; idx < HISTOGRAM_SIZE; idx += GROUP_SIZE) halfCount[idx] = 0;

    uint globalIndex = group * GROUP_SIZE + lane;
    bool active = globalIndex < cfg.elementCount;
    uint bit = 0u;
    uint key = 0u;
    uint value = 0u;
    if (active)
    {
        key = g_KeysIn[globalIndex];
        bit = (key >> g_shift) & RADIX_BYTE_MASK;
        tempBits[lane] = bit;
        value = g_ValuesIn[globalIndex];
    } else tempBits[lane] = RADIX_INVALID_VALUE;
    GroupMemoryBarrierWithGroupSync();

    uint quarterIndex = lane / QUARTER_SIZE;
    if (active && quarterIndex < PACKED_QUARTER_COUNT)
    {
        uint inc = 0u;
        [unroll] for (uint packedQuarter = quarterIndex; packedQuarter < PACKED_QUARTER_COUNT; ++packedQuarter) inc |= (1u << (packedQuarter * RADIX_BYTE_SHIFT));
        InterlockedAdd(halfCount[bit], inc);
    }
    GroupMemoryBarrierWithGroupSync();

    if (active)
    {
        uint globalOff = hierarchy.get_global_offset(bit, group);
        uint quarterOffset = 0u;
        if (quarterIndex > 0u)
        {
            uint packed = halfCount[bit];
            uint shiftBytes = (min(quarterIndex, PACKED_QUARTER_COUNT) - 1u) * RADIX_BYTE_SHIFT;
            quarterOffset = (packed >> shiftBytes) & RADIX_BYTE_MASK;
        }
        uint begin = quarterIndex * QUARTER_SIZE;
        uint localCount = 0u;
        [unroll] for (uint t = begin; t < lane; ++t) localCount += (tempBits[t] == bit) ? 1u : 0u;
        uint dest = globalOff + quarterOffset + localCount;
        g_KeysOut[dest] = key;
        g_ValuesOut[dest] = value;
    }
}
