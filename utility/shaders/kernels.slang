import "slangpy";

static const uint PREFIX_THREADS = 256u;
static const uint PREFIX_BLOCK_SIZE = PREFIX_THREADS * 2u;
static const uint RADIX_GROUP_SIZE = 256u;
static const uint RADIX_BIN_COUNT = 256u;
static const uint RADIX_PACK_LANE_BITS = 1u;
static const uint RADIX_PACK_LANE_MASK = 1u;
static const uint RADIX_PACK_SHIFT = 16u;
static const uint RADIX_PACK_MASK = 0xFFFFu;
static const uint RADIX_BYTE_SHIFT = 8u;
static const uint RADIX_BYTE_MASK = 0xFFu;
static const uint RADIX_INVALID_VALUE = 0xFFFFFFFFu;
static const uint PACKED_HIST_SIZE = RADIX_BIN_COUNT >> RADIX_PACK_LANE_BITS;
static const uint RADIX_QUARTER_SIZE = RADIX_GROUP_SIZE / 4u;
static const uint RADIX_PACKED_QUARTER_COUNT = 3u;
static const uint RADIX_PAIR_SIZE = 2u;
static const uint RADIX_PREFIX_SHIFT = 16u;
static const uint RADIX_PREFIX_MASK = 0xFFFFu;
static const uint RADIX_MAX_PREFIX_LEVELS = 4u;
static const uint PREFIX_MAX_LEVELS = 8u;
static const uint INDIRECT_DISPATCH_ARG_STRIDE = 3u;
static const uint PREFIX_SCAN_ARGS_OFFSET = 0u;
static const uint PREFIX_ADD_ARGS_OFFSET = PREFIX_SCAN_ARGS_OFFSET + PREFIX_MAX_LEVELS * INDIRECT_DISPATCH_ARG_STRIDE;
static const uint PREFIX_PARAM_COUNT_OFFSET = PREFIX_ADD_ARGS_OFFSET + PREFIX_MAX_LEVELS * INDIRECT_DISPATCH_ARG_STRIDE;
static const uint PREFIX_INDIRECT_ARGS_UINT_COUNT = PREFIX_PARAM_COUNT_OFFSET + 1u;
static const uint HISTOGRAM_ARGS_OFFSET = 0u;
static const uint PREFIX_L0_ARGS_OFFSET = HISTOGRAM_ARGS_OFFSET + INDIRECT_DISPATCH_ARG_STRIDE;
static const uint SCATTER_ARGS_OFFSET = PREFIX_L0_ARGS_OFFSET + RADIX_MAX_PREFIX_LEVELS * INDIRECT_DISPATCH_ARG_STRIDE;
static const uint BUILD_RANGE_ARGS_OFFSET = SCATTER_ARGS_OFFSET + INDIRECT_DISPATCH_ARG_STRIDE;
static const uint PARAM_ELEMENT_COUNT = BUILD_RANGE_ARGS_OFFSET + INDIRECT_DISPATCH_ARG_STRIDE;
static const uint PARAM_NUM_GROUPS = PARAM_ELEMENT_COUNT + 1u;
static const uint PARAM_TOTAL_N = PARAM_NUM_GROUPS + 1u;
static const uint PARAM_NUM_LEVELS = PARAM_TOTAL_N + 1u;
static const uint RADIX_INDIRECT_ARGS_UINT_COUNT = PARAM_NUM_LEVELS + 1u;

uniform Tensor<uint, 1> g_PrefixInput;
uniform RWTensor<uint, 1> g_PrefixOutput;
uniform RWTensor<uint, 1> g_PrefixBlockSums;
uniform Tensor<uint, 1> g_PrefixBlockOffsets;
uniform RWTensor<uint, 1> g_TotalOut;
uniform uint g_Count;
uniform uint g_Exclusive;
uniform Tensor<uint, 1> g_PrefixParamsBuffer;
uniform uint g_UsePrefixParams;
uniform uint g_Level;

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

uniform Tensor<uint, 1> g_DispatchCountBuffer;
uniform uint g_DispatchCountOffset;
uniform uint g_DispatchMaxCount;
uniform uint g_DispatchThreadsPerGroup;
uniform RWTensor<uint, 1> g_DispatchIndirectArgs;

uniform Tensor<uint, 1> g_CountBuffer;
uniform uint g_CountOffset;
uniform uint g_MaxElementCount;
uniform RWTensor<uint, 1> g_IndirectArgs;

groupshared uint gsPrefixWaveTotals[PREFIX_THREADS];
groupshared uint gsPrefixWaveScan[PREFIX_THREADS];
groupshared uint gsHistogram[PACKED_HIST_SIZE];
groupshared uint gsDigits[RADIX_GROUP_SIZE];
groupshared uint gsQuarterCounts[RADIX_BIN_COUNT];
groupshared uint gsRadixPrefix[PREFIX_THREADS];

uint pack_count(uint digit, uint count) {
    return count << ((digit & RADIX_PACK_LANE_MASK) * RADIX_PACK_SHIFT);
}

uint unpack_count(uint packed, uint digit) {
    return (packed >> ((digit & RADIX_PACK_LANE_MASK) * RADIX_PACK_SHIFT)) & RADIX_PACK_MASK;
}

uint pack_prefix16(uint lo, uint hi) {
    return (hi << RADIX_PREFIX_SHIFT) | (lo & RADIX_PREFIX_MASK);
}

uint unpack_prefix16(uint packed, uint odd) {
    return odd != 0u ? (packed >> RADIX_PREFIX_SHIFT) : (packed & RADIX_PREFIX_MASK);
}

uint ceil_div_block(uint value) {
    return (value + PREFIX_THREADS - 1u) / PREFIX_THREADS;
}

uint prefix_level_count(uint count, uint level) {
    uint size = max(count, 1u);
    for (uint current = 0u; current < level; ++current) {
        size = max((size + PREFIX_BLOCK_SIZE - 1u) / PREFIX_BLOCK_SIZE, 1u);
    }
    return size;
}

uint load_prefix_count() {
    return g_UsePrefixParams != 0u ? g_PrefixParamsBuffer[PREFIX_PARAM_COUNT_OFFSET] : g_Count;
}

uint block_exclusive_scan(uint value, uint localId, out uint blockTotal)
{
    uint waveSize = WaveGetLaneCount();
    uint laneId = WaveGetLaneIndex();
    uint waveId = (localId - laneId) / waveSize;
    uint numWaves = (PREFIX_THREADS + waveSize - 1u) / waveSize;

    uint wavePrefix = WavePrefixSum(value);
    uint waveTotal = WaveReadLaneAt(wavePrefix, waveSize - 1u) + WaveReadLaneAt(value, waveSize - 1u);

    if (WaveIsFirstLane())
    {
        gsPrefixWaveTotals[waveId] = waveTotal;
    }
    GroupMemoryBarrierWithGroupSync();

    if (waveId == 0u)
    {
        uint reshuffledTotal = localId < numWaves ? gsPrefixWaveTotals[localId] : 0u;
        uint reshuffledPrefix = WavePrefixSum(reshuffledTotal);
        if (localId < numWaves)
        {
            gsPrefixWaveScan[localId] = reshuffledPrefix;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    blockTotal = gsPrefixWaveScan[numWaves - 1u] + gsPrefixWaveTotals[numWaves - 1u];
    uint waveBase = gsPrefixWaveScan[waveId];
    GroupMemoryBarrierWithGroupSync();
    return waveBase + wavePrefix;
}

uint radix_total_n(uint numGroups) { return numGroups * RADIX_BIN_COUNT; }
uint radix_level_size(uint totalN, uint level) { uint size = totalN; for (uint current = 0u; current < level; ++current) size = ceil_div_block(size); return size; }
uint radix_level_offset(uint totalN, uint level) { uint offset = 0u; uint size = totalN; for (uint current = 0u; current < level; ++current) { offset += current == 0u ? (size + 1u) / 2u : size; size = ceil_div_block(size); } return offset; }
uint radix_num_levels(uint totalN) { uint levels = 0u; uint size = totalN; for (uint current = 0u; current < RADIX_MAX_PREFIX_LEVELS && size > 1u; ++current) { levels += 1u; size = ceil_div_block(size); } return max(levels, 1u); }
uint load_radix_param(uint index, uint fallback) { return g_UseRadixParams != 0u ? g_RadixParamsBuffer[index] : fallback; }

void write_dispatch_args(RWTensor<uint, 1> indirectArgs, uint baseOffset, uint groupCount) {
    indirectArgs.store(int(baseOffset), groupCount);
    indirectArgs.store(int(baseOffset + 1u), 1u);
    indirectArgs.store(int(baseOffset + 2u), 1u);
}

void write_radix_indirect_args(RWTensor<uint, 1> indirectArgs, uint elementCount, uint maxElementCount)
{
    uint n = (maxElementCount != 0u) ? min(elementCount, maxElementCount) : elementCount;
    uint numGroups = max((n + RADIX_GROUP_SIZE - 1u) / RADIX_GROUP_SIZE, 1u);
    uint totalN = numGroups * RADIX_BIN_COUNT;
    write_dispatch_args(indirectArgs, HISTOGRAM_ARGS_OFFSET, numGroups);

    uint dispatchSize = totalN;
    uint argsOffset = PREFIX_L0_ARGS_OFFSET;
    for (uint i = 0u; i < RADIX_MAX_PREFIX_LEVELS; ++i)
    {
        write_dispatch_args(indirectArgs, argsOffset, max(ceil_div_block(dispatchSize), 1u));
        dispatchSize = ceil_div_block(dispatchSize);
        argsOffset += INDIRECT_DISPATCH_ARG_STRIDE;
    }

    write_dispatch_args(indirectArgs, SCATTER_ARGS_OFFSET, numGroups);
    write_dispatch_args(indirectArgs, BUILD_RANGE_ARGS_OFFSET, max(ceil_div_block(n), 1u));

    uint numLevels = 0u;
    uint levelSize = totalN;
    for (uint i = 0u; i < RADIX_MAX_PREFIX_LEVELS && levelSize > 1u; ++i)
    {
        numLevels += 1u;
        levelSize = ceil_div_block(levelSize);
    }
    numLevels = max(numLevels, 1u);

    indirectArgs.store(int(PARAM_ELEMENT_COUNT), n);
    indirectArgs.store(int(PARAM_NUM_GROUPS), numGroups);
    indirectArgs.store(int(PARAM_TOTAL_N), totalN);
    indirectArgs.store(int(PARAM_NUM_LEVELS), numLevels);
}

[shader("compute")]
[numthreads(PREFIX_THREADS, 1, 1)]
void csPrefixScanBlocks(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID) {
    uint count = prefix_level_count(load_prefix_count(), g_Level);
    uint lane = localID.x;
    uint blockBase = groupID.x * PREFIX_BLOCK_SIZE;
    uint idx0 = blockBase + lane;
    uint idx1 = idx0 + PREFIX_THREADS;
    uint value0 = idx0 < count ? g_PrefixInput[int(idx0)] : 0u;
    uint value1 = idx1 < count ? g_PrefixInput[int(idx1)] : 0u;
    uint firstHalfTotal;
    uint secondHalfTotal;
    uint scan0 = block_exclusive_scan(value0, lane, firstHalfTotal);
    uint scan1 = block_exclusive_scan(value1, lane, secondHalfTotal) + firstHalfTotal;
    uint blockTotal = firstHalfTotal + secondHalfTotal;

    if (idx0 < count) g_PrefixOutput.store(int(idx0), scan0 + (g_Exclusive != 0u ? 0u : value0));
    if (idx1 < count) g_PrefixOutput.store(int(idx1), scan1 + (g_Exclusive != 0u ? 0u : value1));
    if (lane == 0u) g_PrefixBlockSums.store(int(groupID.x), blockTotal);
}

[shader("compute")]
[numthreads(PREFIX_THREADS, 1, 1)]
void csPrefixAddOffsets(uint3 dispatchThreadID : SV_DispatchThreadID) {
    if (g_UseRadixPrefixAdd != 0u) {
        uint numGroups = load_radix_param(PARAM_NUM_GROUPS, g_NumGroups);
        uint totalN = load_radix_param(PARAM_TOTAL_N, radix_total_n(numGroups));
        uint numLevels = load_radix_param(PARAM_NUM_LEVELS, radix_num_levels(totalN));
        if (g_Level == 0u || g_Level + 1u >= numLevels) return;

        uint count = radix_level_size(totalN, g_Level);
        uint index = dispatchThreadID.x;
        if (index >= count) return;

        uint blockId = index / PREFIX_THREADS;
        if (blockId == 0u) return;

        uint levelOffset = radix_level_offset(totalN, g_Level);
        uint nextLevelOffset = radix_level_offset(totalN, g_Level + 1u);
        uint blockOffset = g_HistogramOffsets[int(nextLevelOffset + blockId - 1u)];

        g_HistogramOffsets.store(int(levelOffset + index), g_HistogramOffsets[int(levelOffset + index)] + blockOffset);
        return;
    }

    uint count = prefix_level_count(load_prefix_count(), g_Level);
    uint index = dispatchThreadID.x;
    if (index >= count) return;
    uint blockId = index / PREFIX_BLOCK_SIZE;
    if (blockId == 0u) return;
    g_PrefixOutput.store(int(index), g_PrefixOutput[int(index)] + g_PrefixBlockOffsets[int(blockId)]);
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
            value = prevLevel == 0u
                ? unpack_prefix16(g_HistogramOffsets[int(radix_level_offset(totalN, prevLevel) + (blockEnd >> 1u))], blockEnd & 1u)
                : g_HistogramOffsets[int(radix_level_offset(totalN, prevLevel) + blockEnd)];
        }
    } else {
        value = 0u;
    }

    uint blockTotal;
    uint inclusive = block_exclusive_scan(value, lane, blockTotal) + value;
    gsRadixPrefix[lane] = inclusive;
    GroupMemoryBarrierWithGroupSync();
    if (level == 0u) {
        if (globalIndex >= levelSize) return;
        if ((lane & 1u) == 0u) {
            uint hi = (lane + 1u < PREFIX_THREADS && globalIndex + 1u < levelSize) ? gsRadixPrefix[lane + 1u] : 0u;
            g_HistogramOffsets.store(int(levelOffset + (globalIndex >> 1u)), pack_prefix16(gsRadixPrefix[lane], hi));
        }
    } else {
        if (globalIndex >= levelSize) return;
        g_HistogramOffsets.store(int(levelOffset + globalIndex), gsRadixPrefix[lane]);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void csPrefixWriteTotal(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint count = load_prefix_count();
    uint total = 0u;
    if (count != 0u) {
        total = g_PrefixOutput[int(count - 1u)];
        if (g_Exclusive != 0u) total += g_PrefixInput[int(count - 1u)];
    }
    g_TotalOut.store(0, total);
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
    [unroll] for (uint digitIndex = lane; digitIndex < RADIX_BIN_COUNT; digitIndex += RADIX_GROUP_SIZE) gsQuarterCounts[digitIndex] = 0u;
    uint digit = active ? ((g_KeysIn[int(index)] >> g_Shift) & g_DigitMask) : RADIX_INVALID_VALUE;
    gsDigits[lane] = digit;
    GroupMemoryBarrierWithGroupSync();
    if (!active) return;

    uint quarterIndex = lane / RADIX_QUARTER_SIZE;
    if (quarterIndex < RADIX_PACKED_QUARTER_COUNT) {
        uint increment = 0u;
        [unroll] for (uint packedQuarter = quarterIndex; packedQuarter < RADIX_PACKED_QUARTER_COUNT; ++packedQuarter) {
            increment |= 1u << (packedQuarter * RADIX_BYTE_SHIFT);
        }
        InterlockedAdd(gsQuarterCounts[digit], increment);
    }
    GroupMemoryBarrierWithGroupSync();

    uint localOffset = 0u;
    uint begin = quarterIndex * RADIX_QUARTER_SIZE;
    [unroll] for (uint prev = begin; prev < lane; ++prev) localOffset += gsDigits[prev] == digit ? 1u : 0u;

    uint quarterOffset = 0u;
    if (quarterIndex > 0u) {
        uint packed = gsQuarterCounts[digit];
        uint shiftBytes = (min(quarterIndex, RADIX_PACKED_QUARTER_COUNT) - 1u) * RADIX_BYTE_SHIFT;
        quarterOffset = (packed >> shiftBytes) & RADIX_BYTE_MASK;
    }

    uint globalOffset = 0u;
    uint idx = digit * numGroups + groupID.x;
    uint blockId = idx / PREFIX_THREADS;
    if (idx != 0u && (idx % PREFIX_THREADS) != 0u) {
        uint prev = idx - 1u;
        globalOffset = unpack_prefix16(g_HistogramOffsets[int(prev >> 1u)], prev & 1u);
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
[numthreads(PREFIX_THREADS, 1, 1)]
void csComputeDispatchArgsFromBuffer() {
    uint n = g_CountBuffer[g_CountOffset];
    if (g_MaxElementCount != 0u) n = min(n, g_MaxElementCount);
    uint threadsPerGroup = max(g_DispatchThreadsPerGroup, 1u);
    uint groupCount = max((n + threadsPerGroup - 1u) / threadsPerGroup, 1u);
    g_DispatchIndirectArgs.store(0, groupCount);
    g_DispatchIndirectArgs.store(1, 1u);
    g_DispatchIndirectArgs.store(2, 1u);
    g_DispatchIndirectArgs.store(3, n);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void csComputePrefixIndirectArgsFromBuffer() {
    uint n = g_CountBuffer[g_CountOffset];
    if (g_MaxElementCount != 0u) n = min(n, g_MaxElementCount);
    for (uint level = 0u; level < PREFIX_MAX_LEVELS; ++level) {
        uint levelCount = prefix_level_count(n, level);
        write_dispatch_args(g_IndirectArgs, PREFIX_SCAN_ARGS_OFFSET + level * INDIRECT_DISPATCH_ARG_STRIDE, max((levelCount + PREFIX_BLOCK_SIZE - 1u) / PREFIX_BLOCK_SIZE, 1u));
        write_dispatch_args(g_IndirectArgs, PREFIX_ADD_ARGS_OFFSET + level * INDIRECT_DISPATCH_ARG_STRIDE, max((levelCount + PREFIX_THREADS - 1u) / PREFIX_THREADS, 1u));
    }
    g_IndirectArgs.store(int(PREFIX_PARAM_COUNT_OFFSET), n);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void csComputeRadixIndirectArgsFromBuffer() {
    write_radix_indirect_args(g_IndirectArgs, g_CountBuffer[g_CountOffset], g_MaxElementCount);
}
