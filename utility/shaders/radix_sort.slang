import "slangpy";

static const uint PREFIX_THREADS = 256u;
static const uint RADIX_GROUP_SIZE = 512u;
static const uint RADIX_BIN_COUNT = 256u;
static const uint RADIX_PACK_LANE_BITS = 1u;
static const uint RADIX_PACK_LANE_MASK = 1u;
static const uint RADIX_PACK_SHIFT = 16u;
static const uint RADIX_PACK_MASK = 0xFFFFu;
static const uint RADIX_INVALID_VALUE = 0xFFFFFFFFu;
static const uint PACKED_HIST_SIZE = RADIX_BIN_COUNT >> RADIX_PACK_LANE_BITS;
static const uint RADIX_QUARTER_SIZE = RADIX_GROUP_SIZE / 4u;
static const uint RADIX_PACKED_QUARTER_COUNT = 3u;
static const uint RADIX_MAX_PREFIX_LEVELS = 4u;
static const uint INDIRECT_DISPATCH_ARG_STRIDE = 3u;
static const uint HISTOGRAM_ARGS_OFFSET = 0u;
static const uint PREFIX_L0_ARGS_OFFSET = HISTOGRAM_ARGS_OFFSET + INDIRECT_DISPATCH_ARG_STRIDE;
static const uint SCATTER_ARGS_OFFSET = PREFIX_L0_ARGS_OFFSET + RADIX_MAX_PREFIX_LEVELS * INDIRECT_DISPATCH_ARG_STRIDE;
static const uint BUILD_RANGE_ARGS_OFFSET = SCATTER_ARGS_OFFSET + INDIRECT_DISPATCH_ARG_STRIDE;
static const uint PARAM_ELEMENT_COUNT = BUILD_RANGE_ARGS_OFFSET + INDIRECT_DISPATCH_ARG_STRIDE;
static const uint PARAM_NUM_GROUPS = PARAM_ELEMENT_COUNT + 1u;
static const uint PARAM_TOTAL_N = PARAM_NUM_GROUPS + 1u;
static const uint PARAM_NUM_LEVELS = PARAM_TOTAL_N + 1u;

uint pack_count(uint digit, uint count) { return count << ((digit & RADIX_PACK_LANE_MASK) * RADIX_PACK_SHIFT); }
uint unpack_count(uint packed, uint digit) { return (packed >> ((digit & RADIX_PACK_LANE_MASK) * RADIX_PACK_SHIFT)) & RADIX_PACK_MASK; }
uint ceil_div_block(uint value) { return (value + PREFIX_THREADS - 1u) / PREFIX_THREADS; }

uniform Tensor<uint, 1> g_KeysIn;
uniform Tensor<uint, 1> g_ValuesIn;
uniform RWTensor<uint, 1> g_KeysOut;
uniform RWTensor<uint, 1> g_ValuesOut;
uniform RWTensor<uint, 1> g_Histogram;
uniform RWTensor<uint, 1> g_HistogramOffsets;
uniform uint g_ElementCount;
uniform uint g_NumGroups;
uniform uint g_Shift;
uniform uint g_DigitMask;
uniform Tensor<uint, 1> g_RadixParamsBuffer;
uniform uint g_UseRadixParams;
uniform uint g_UseRadixPrefixAdd;
uniform Tensor<uint, 1> g_CountBuffer;
uniform uint g_CountOffset;
uniform uint g_MaxElementCount;
uniform RWTensor<uint, 1> g_IndirectArgs;

groupshared uint gsHistogram[PACKED_HIST_SIZE];
groupshared uint gsDigits[RADIX_GROUP_SIZE];
groupshared uint gsQuarterCounts[RADIX_BIN_COUNT * RADIX_PACKED_QUARTER_COUNT];
groupshared uint gsRadixPrefix[PREFIX_THREADS];

uint radix_total_n(uint numGroups) { return numGroups * RADIX_BIN_COUNT; }
uint radix_level_size(uint totalN, uint level) { uint size = totalN; for (uint current = 0u; current < level; ++current) size = ceil_div_block(size); return size; }
uint radix_level_offset(uint totalN, uint level) { uint offset = 0u; uint size = totalN; for (uint current = 0u; current < level; ++current) { offset += size; size = ceil_div_block(size); } return offset; }
uint radix_num_levels(uint totalN) { uint levels = 0u; uint size = totalN; for (uint current = 0u; current < RADIX_MAX_PREFIX_LEVELS && size > 1u; ++current) { levels += 1u; size = ceil_div_block(size); } return max(levels, 1u); }
uint load_radix_param(uint index, uint fallback) { return g_UseRadixParams != 0u ? g_RadixParamsBuffer[index] : fallback; }

uint block_exclusive_scan_radix(uint value, uint localId, out uint blockTotal)
{
    uint waveSize = WaveGetLaneCount();
    uint laneId = WaveGetLaneIndex();
    uint waveId = (localId - laneId) / waveSize;
    uint numWaves = (PREFIX_THREADS + waveSize - 1u) / waveSize;
    uint wavePrefix = WavePrefixSum(value);
    uint waveTotal = WaveReadLaneAt(wavePrefix, waveSize - 1u) + WaveReadLaneAt(value, waveSize - 1u);

    if (WaveIsFirstLane()) {
        gsRadixPrefix[waveId] = waveTotal;
    }
    GroupMemoryBarrierWithGroupSync();

    if (waveId == 0u) {
        uint reshuffledTotal = localId < numWaves ? gsRadixPrefix[localId] : 0u;
        uint reshuffledPrefix = WavePrefixSum(reshuffledTotal);
        if (localId < numWaves) {
            gsRadixPrefix[localId] = reshuffledPrefix;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    blockTotal = gsRadixPrefix[numWaves - 1u] + (numWaves > 0u ? WaveReadLaneAt(waveTotal, 0u) : 0u);
    uint waveBase = gsRadixPrefix[waveId];
    GroupMemoryBarrierWithGroupSync();
    return waveBase + wavePrefix;
}

void write_dispatch_args_radix(RWTensor<uint, 1> indirectArgs, uint baseOffset, uint groupCount) {
    indirectArgs.store(int(baseOffset), groupCount);
    indirectArgs.store(int(baseOffset + 1u), 1u);
    indirectArgs.store(int(baseOffset + 2u), 1u);
}

[shader("compute")]
[numthreads(PREFIX_THREADS, 1, 1)]
void csRadixPrefixLevel(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID) {
    uint lane = localID.x;
    uint numGroups = load_radix_param(PARAM_NUM_GROUPS, g_NumGroups);
    uint totalN = radix_total_n(numGroups);
    uint level = g_Shift;
    uint levelSize = radix_level_size(totalN, level);
    uint levelOffset = radix_level_offset(totalN, level);
    uint globalIndex = groupID.x * PREFIX_THREADS + lane;
    uint value = 0u;
    if (globalIndex < levelSize) {
        if (level == 0u) {
            uint digit = globalIndex / numGroups;
            uint group = globalIndex % numGroups;
            value = unpack_count(g_Histogram[int((digit >> RADIX_PACK_LANE_BITS) * numGroups + group)], digit);
        } else {
            uint prevLevel = level - 1u;
            uint prevLevelSize = radix_level_size(totalN, prevLevel);
            uint blockEnd = min((globalIndex + 1u) * PREFIX_THREADS, prevLevelSize) - 1u;
            value = g_HistogramOffsets[int(radix_level_offset(totalN, prevLevel) + blockEnd)];
        }
    }

    uint blockTotal;
    uint inclusive = block_exclusive_scan_radix(value, lane, blockTotal) + value;
    gsRadixPrefix[lane] = inclusive;
    GroupMemoryBarrierWithGroupSync();
    if (globalIndex >= levelSize) return;
    g_HistogramOffsets.store(int(levelOffset + globalIndex), gsRadixPrefix[lane]);
}

[shader("compute")]
[numthreads(RADIX_GROUP_SIZE, 1, 1)]
void csRadixHistogram(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID) {
    uint elementCount = load_radix_param(PARAM_ELEMENT_COUNT, g_ElementCount);
    uint numGroups = load_radix_param(PARAM_NUM_GROUPS, g_NumGroups);
    uint lane = localID.x;
    if (lane < PACKED_HIST_SIZE) gsHistogram[lane] = 0u;
    GroupMemoryBarrierWithGroupSync();

    uint index = groupID.x * RADIX_GROUP_SIZE + lane;
    if (index < elementCount) {
        uint digit = (g_KeysIn[int(index)] >> g_Shift) & g_DigitMask;
        InterlockedAdd(gsHistogram[digit >> RADIX_PACK_LANE_BITS], pack_count(digit, 1u));
    }
    GroupMemoryBarrierWithGroupSync();

    if (lane < PACKED_HIST_SIZE) g_Histogram.store(int(lane * numGroups + groupID.x), gsHistogram[lane]);
}

[shader("compute")]
[numthreads(RADIX_GROUP_SIZE, 1, 1)]
void csRadixScatter(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID) {
    uint elementCount = load_radix_param(PARAM_ELEMENT_COUNT, g_ElementCount);
    uint numGroups = load_radix_param(PARAM_NUM_GROUPS, g_NumGroups);
    uint totalN = load_radix_param(PARAM_TOTAL_N, radix_total_n(g_NumGroups));
    uint numLevels = load_radix_param(PARAM_NUM_LEVELS, radix_num_levels(totalN));
    uint lane = localID.x;
    uint index = groupID.x * RADIX_GROUP_SIZE + lane;
    bool active = index < elementCount;
    [unroll] for (uint digitIndex = lane; digitIndex < RADIX_BIN_COUNT * RADIX_PACKED_QUARTER_COUNT; digitIndex += RADIX_GROUP_SIZE) gsQuarterCounts[digitIndex] = 0u;
    uint digit = active ? ((g_KeysIn[int(index)] >> g_Shift) & g_DigitMask) : RADIX_INVALID_VALUE;
    gsDigits[lane] = digit;
    GroupMemoryBarrierWithGroupSync();
    if (!active) return;

    uint quarterIndex = lane / RADIX_QUARTER_SIZE;
    if (quarterIndex < RADIX_PACKED_QUARTER_COUNT) {
        InterlockedAdd(gsQuarterCounts[digit * RADIX_PACKED_QUARTER_COUNT + quarterIndex], 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    uint localOffset = 0u;
    uint begin = quarterIndex * RADIX_QUARTER_SIZE;
    [unroll] for (uint prev = begin; prev < lane; ++prev) localOffset += gsDigits[prev] == digit ? 1u : 0u;

    uint quarterOffset = 0u;
    if (quarterIndex > 0u) {
        [unroll] for (uint prevQuarter = 0u; prevQuarter < RADIX_PACKED_QUARTER_COUNT; ++prevQuarter) {
            if (prevQuarter >= quarterIndex) break;
            quarterOffset += gsQuarterCounts[digit * RADIX_PACKED_QUARTER_COUNT + prevQuarter];
        }
    }

    uint globalOffset = 0u;
    uint idx = digit * numGroups + groupID.x;
    uint blockId = idx / PREFIX_THREADS;
    if (idx != 0u && (idx % PREFIX_THREADS) != 0u) {
        uint prev = idx - 1u;
        globalOffset = g_HistogramOffsets[int(prev)];
    }
    if (numLevels > 1u && blockId > 0u) {
        uint level1Offset = radix_level_offset(totalN, 1u);
        globalOffset += g_HistogramOffsets[int(level1Offset + blockId - 1u)];
    }
    uint dest = globalOffset + quarterOffset + localOffset;
    g_KeysOut.store(int(dest), g_KeysIn[int(index)]);
    g_ValuesOut.store(int(dest), g_ValuesIn[int(index)]);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void csComputeRadixIndirectArgsFromBuffer() {
    uint elementCount = g_CountBuffer[g_CountOffset];
    uint n = (g_MaxElementCount != 0u) ? min(elementCount, g_MaxElementCount) : elementCount;
    uint numGroups = max((n + RADIX_GROUP_SIZE - 1u) / RADIX_GROUP_SIZE, 1u);
    uint totalN = numGroups * RADIX_BIN_COUNT;
    write_dispatch_args_radix(g_IndirectArgs, HISTOGRAM_ARGS_OFFSET, numGroups);

    uint dispatchSize = totalN;
    uint argsOffset = PREFIX_L0_ARGS_OFFSET;
    for (uint i = 0u; i < RADIX_MAX_PREFIX_LEVELS; ++i)
    {
        write_dispatch_args_radix(g_IndirectArgs, argsOffset, max(ceil_div_block(dispatchSize), 1u));
        dispatchSize = ceil_div_block(dispatchSize);
        argsOffset += INDIRECT_DISPATCH_ARG_STRIDE;
    }

    write_dispatch_args_radix(g_IndirectArgs, SCATTER_ARGS_OFFSET, numGroups);
    write_dispatch_args_radix(g_IndirectArgs, BUILD_RANGE_ARGS_OFFSET, max(ceil_div_block(n), 1u));

    uint numLevels = 0u;
    uint levelSize = totalN;
    for (uint i = 0u; i < RADIX_MAX_PREFIX_LEVELS && levelSize > 1u; ++i)
    {
        numLevels += 1u;
        levelSize = ceil_div_block(levelSize);
    }
    numLevels = max(numLevels, 1u);

    g_IndirectArgs.store(int(PARAM_ELEMENT_COUNT), n);
    g_IndirectArgs.store(int(PARAM_NUM_GROUPS), numGroups);
    g_IndirectArgs.store(int(PARAM_TOTAL_N), totalN);
    g_IndirectArgs.store(int(PARAM_NUM_LEVELS), numLevels);
}
