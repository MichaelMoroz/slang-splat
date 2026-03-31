import "utility/radix_sort/common.slang";

uniform RWStructuredBuffer<uint> g_HistogramOffsets;
uniform StructuredBuffer<uint> g_Histogram;
uniform uint g_NumGroups;
uniform uint g_Shift;
uniform StructuredBuffer<uint> g_RadixParamsBuffer;
uniform uint g_UseRadixParams;

groupshared uint gsRadixPrefix[PREFIX_THREADS];

uint block_exclusive_scan_radix(uint value, uint localId, out uint blockTotal)
{
    uint waveSize = WaveGetLaneCount();
    uint laneId = WaveGetLaneIndex();
    uint waveId = (localId - laneId) / waveSize;
    uint numWaves = (PREFIX_THREADS + waveSize - 1u) / waveSize;
    uint wavePrefix = WavePrefixSum(value);
    uint waveTotal = WaveReadLaneAt(wavePrefix, waveSize - 1u) + WaveReadLaneAt(value, waveSize - 1u);

    if (WaveIsFirstLane()) gsRadixPrefix[waveId] = waveTotal;
    GroupMemoryBarrierWithGroupSync();

    if (waveId == 0u)
    {
        uint reshuffledTotal = localId < numWaves ? gsRadixPrefix[localId] : 0u;
        uint reshuffledPrefix = WavePrefixSum(reshuffledTotal);
        if (localId < numWaves) gsRadixPrefix[localId] = reshuffledPrefix;
    }
    GroupMemoryBarrierWithGroupSync();

    blockTotal = gsRadixPrefix[numWaves - 1u] + WaveReadLaneAt(waveTotal, 0u);
    uint waveBase = gsRadixPrefix[waveId];
    GroupMemoryBarrierWithGroupSync();
    return waveBase + wavePrefix;
}

[shader("compute")]
[numthreads(PREFIX_THREADS, 1, 1)]
void csRadixPrefixLevel(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID)
{
    uint lane = localID.x;
    uint numGroups = load_radix_param(g_RadixParamsBuffer, g_UseRadixParams, PARAM_NUM_GROUPS, g_NumGroups);
    uint totalN = load_radix_param(g_RadixParamsBuffer, g_UseRadixParams, PARAM_TOTAL_N, radix_total_n(numGroups));
    uint level = g_Shift;
    uint levelSize = radix_level_size(totalN, level);
    uint levelOffset = radix_level_offset(totalN, level);
    uint globalIndex = groupID.x * PREFIX_THREADS + lane;
    uint value = 0u;
    if (globalIndex < levelSize)
    {
        if (level == 0u)
        {
            uint digit = globalIndex / numGroups;
            uint group = globalIndex % numGroups;
            value = unpack_count(g_Histogram[(digit >> RADIX_PACK_LANE_BITS) * numGroups + group], digit);
        } else {
            uint prevLevel = level - 1u;
            uint prevLevelSize = radix_level_size(totalN, prevLevel);
            uint blockEnd = min((globalIndex + 1u) * PREFIX_THREADS, prevLevelSize) - 1u;
            value = g_HistogramOffsets[radix_level_offset(totalN, prevLevel) + blockEnd];
        }
    }

    uint blockTotal;
    uint inclusive = block_exclusive_scan_radix(value, lane, blockTotal) + value;
    gsRadixPrefix[lane] = inclusive;
    GroupMemoryBarrierWithGroupSync();
    if (globalIndex >= levelSize) return;
    g_HistogramOffsets[levelOffset + globalIndex] = gsRadixPrefix[lane];
}
