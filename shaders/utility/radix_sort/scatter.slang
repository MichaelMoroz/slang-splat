import "utility/radix_sort/common.slang";

uniform StructuredBuffer<uint> g_KeysIn;
uniform StructuredBuffer<uint> g_ValuesIn;
uniform RWStructuredBuffer<uint> g_KeysOut;
uniform RWStructuredBuffer<uint> g_ValuesOut;
uniform StructuredBuffer<uint> g_HistogramOffsets;
uniform uint g_ElementCount;
uniform uint g_NumGroups;
uniform uint g_TotalN;
uniform uint g_NumLevels;
uniform uint g_Shift;
uniform uint g_DigitMask;
uniform StructuredBuffer<uint> g_RadixParamsBuffer;
uniform uint g_UseRadixParams;

groupshared uint gsDigits[RADIX_GROUP_SIZE];
groupshared uint gsQuarterCounts[RADIX_BIN_COUNT * RADIX_PACKED_QUARTER_COUNT];

[shader("compute")]
[numthreads(RADIX_GROUP_SIZE, 1, 1)]
void csRadixScatter(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID)
{
    uint elementCount = load_radix_param(g_RadixParamsBuffer, g_UseRadixParams, PARAM_ELEMENT_COUNT, g_ElementCount);
    uint numGroups = load_radix_param(g_RadixParamsBuffer, g_UseRadixParams, PARAM_NUM_GROUPS, g_NumGroups);
    uint totalN = load_radix_param(g_RadixParamsBuffer, g_UseRadixParams, PARAM_TOTAL_N, g_TotalN != 0u ? g_TotalN : radix_total_n(numGroups));
    uint numLevels = load_radix_param(g_RadixParamsBuffer, g_UseRadixParams, PARAM_NUM_LEVELS, g_NumLevels != 0u ? g_NumLevels : radix_num_levels(totalN));
    uint lane = localID.x;
    uint index = groupID.x * RADIX_GROUP_SIZE + lane;
    bool active = index < elementCount;
    [unroll] for (uint digitIndex = lane; digitIndex < RADIX_BIN_COUNT * RADIX_PACKED_QUARTER_COUNT; digitIndex += RADIX_GROUP_SIZE) gsQuarterCounts[digitIndex] = 0u;
    uint digit = active ? ((g_KeysIn[index] >> g_Shift) & g_DigitMask) : RADIX_INVALID_VALUE;
    gsDigits[lane] = digit;
    GroupMemoryBarrierWithGroupSync();
    if (!active) return;

    uint quarterIndex = lane / RADIX_QUARTER_SIZE;
    if (quarterIndex < RADIX_PACKED_QUARTER_COUNT) InterlockedAdd(gsQuarterCounts[digit * RADIX_PACKED_QUARTER_COUNT + quarterIndex], 1u);
    GroupMemoryBarrierWithGroupSync();

    uint localOffset = 0u;
    uint begin = quarterIndex * RADIX_QUARTER_SIZE;
    [unroll] for (uint prev = begin; prev < lane; ++prev) localOffset += gsDigits[prev] == digit ? 1u : 0u;

    uint quarterOffset = 0u;
    if (quarterIndex > 0u)
    {
        [unroll] for (uint prevQuarter = 0u; prevQuarter < RADIX_PACKED_QUARTER_COUNT; ++prevQuarter)
        {
            if (prevQuarter >= quarterIndex) break;
            quarterOffset += gsQuarterCounts[digit * RADIX_PACKED_QUARTER_COUNT + prevQuarter];
        }
    }

    uint globalOffset = 0u;
    uint linearIndex = digit * numGroups + groupID.x;
    uint blockId = linearIndex / PREFIX_THREADS;
    if (linearIndex != 0u && (linearIndex % PREFIX_THREADS) != 0u) globalOffset = g_HistogramOffsets[linearIndex - 1u];
    if (numLevels > 1u && blockId > 0u)
    {
        uint level1Offset = radix_level_offset(totalN, 1u);
        globalOffset += g_HistogramOffsets[level1Offset + blockId - 1u];
    }
    uint dest = globalOffset + quarterOffset + localOffset;
    g_KeysOut[dest] = g_KeysIn[index];
    g_ValuesOut[dest] = g_ValuesIn[index];
}
