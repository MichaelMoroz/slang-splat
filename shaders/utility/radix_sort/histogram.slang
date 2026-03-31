import "utility/radix_sort/common.slang";

uniform StructuredBuffer<uint> g_KeysIn;
uniform RWStructuredBuffer<uint> g_Histogram;
uniform uint g_ElementCount;
uniform uint g_NumGroups;
uniform uint g_Shift;
uniform uint g_DigitMask;
uniform StructuredBuffer<uint> g_RadixParamsBuffer;
uniform uint g_UseRadixParams;

groupshared uint gsHistogram[PACKED_HIST_SIZE];

[shader("compute")]
[numthreads(RADIX_GROUP_SIZE, 1, 1)]
void csRadixHistogram(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID)
{
    uint elementCount = load_radix_param(g_RadixParamsBuffer, g_UseRadixParams, PARAM_ELEMENT_COUNT, g_ElementCount);
    uint numGroups = load_radix_param(g_RadixParamsBuffer, g_UseRadixParams, PARAM_NUM_GROUPS, g_NumGroups);
    uint lane = localID.x;
    if (lane < PACKED_HIST_SIZE) gsHistogram[lane] = 0u;
    GroupMemoryBarrierWithGroupSync();

    uint index = groupID.x * RADIX_GROUP_SIZE + lane;
    if (index < elementCount)
    {
        uint digit = (g_KeysIn[index] >> g_Shift) & g_DigitMask;
        InterlockedAdd(gsHistogram[digit >> RADIX_PACK_LANE_BITS], pack_count(digit, 1u));
    }
    GroupMemoryBarrierWithGroupSync();

    if (lane < PACKED_HIST_SIZE) g_Histogram[lane * numGroups + groupID.x] = gsHistogram[lane];
}
