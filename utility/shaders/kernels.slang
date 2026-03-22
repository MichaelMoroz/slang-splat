import "slangpy";

static const uint PREFIX_THREADS = 256u;
static const uint PREFIX_BLOCK_SIZE = PREFIX_THREADS * 2u;
static const uint RADIX_GROUP_SIZE = 128u;
static const uint RADIX_BIN_COUNT = 256u;
static const uint RADIX_PACK_LANE_BITS = 2u;
static const uint RADIX_PACK_LANE_MASK = 3u;
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

uniform Tensor<uint, 1> g_PrefixInput;
uniform RWTensor<uint, 1> g_PrefixOutput;
uniform RWTensor<uint, 1> g_PrefixBlockSums;
uniform Tensor<uint, 1> g_PrefixBlockOffsets;
uniform RWTensor<uint, 1> g_TotalOut;
uniform uint g_Count;
uniform uint g_Exclusive;

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

groupshared uint gsPrefix[PREFIX_BLOCK_SIZE];
groupshared uint gsHistogram[PACKED_HIST_SIZE];
groupshared uint gsDigits[RADIX_GROUP_SIZE];
groupshared uint gsQuarterCounts[RADIX_BIN_COUNT];
groupshared uint gsRadixPrefix[PREFIX_THREADS];

uint pack_count(uint digit, uint count) {
    return count << ((digit & RADIX_PACK_LANE_MASK) * RADIX_BYTE_SHIFT);
}

uint unpack_count(uint packed, uint digit) {
    return (packed >> ((digit & RADIX_PACK_LANE_MASK) * RADIX_BYTE_SHIFT)) & RADIX_BYTE_MASK;
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

uint radix_total_n(uint numGroups) { return numGroups * RADIX_BIN_COUNT; }
uint radix_level_size(uint totalN, uint level) { uint size = totalN; for (uint current = 0u; current < level; ++current) size = ceil_div_block(size); return size; }
uint radix_level_offset(uint totalN, uint level) { uint offset = 0u; uint size = totalN; for (uint current = 0u; current < level; ++current) { offset += current == 0u ? (size + 1u) / 2u : size; size = ceil_div_block(size); } return offset; }
uint radix_num_levels(uint totalN) { uint levels = 0u; uint size = totalN; for (uint current = 0u; current < RADIX_MAX_PREFIX_LEVELS && size > 1u; ++current) { levels += 1u; size = ceil_div_block(size); } return max(levels, 1u); }

[shader("compute")]
[numthreads(PREFIX_THREADS, 1, 1)]
void csPrefixScanBlocks(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID) {
    uint lane = localID.x;
    uint blockBase = groupID.x * PREFIX_BLOCK_SIZE;
    uint idx0 = blockBase + lane;
    uint idx1 = idx0 + PREFIX_THREADS;
    uint value0 = idx0 < g_Count ? g_PrefixInput[int(idx0)] : 0u;
    uint value1 = idx1 < g_Count ? g_PrefixInput[int(idx1)] : 0u;
    gsPrefix[lane] = value0;
    gsPrefix[lane + PREFIX_THREADS] = value1;
    GroupMemoryBarrierWithGroupSync();

    [unroll] for (uint stride = 1u; stride < PREFIX_BLOCK_SIZE; stride <<= 1u) {
        uint i = (lane + 1u) * stride * 2u - 1u;
        if (i < PREFIX_BLOCK_SIZE) gsPrefix[i] += gsPrefix[i - stride];
        GroupMemoryBarrierWithGroupSync();
    }

    if (lane == 0u) g_PrefixBlockSums.store(int(groupID.x), gsPrefix[PREFIX_BLOCK_SIZE - 1u]);
    if (lane == 0u) gsPrefix[PREFIX_BLOCK_SIZE - 1u] = 0u;
    GroupMemoryBarrierWithGroupSync();

    [unroll] for (uint stride = PREFIX_BLOCK_SIZE >> 1u; stride > 0u; stride >>= 1u) {
        uint i = (lane + 1u) * stride * 2u - 1u;
        if (i < PREFIX_BLOCK_SIZE) {
            uint t = gsPrefix[i - stride];
            gsPrefix[i - stride] = gsPrefix[i];
            gsPrefix[i] += t;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (idx0 < g_Count) g_PrefixOutput.store(int(idx0), gsPrefix[lane] + (g_Exclusive != 0u ? 0u : value0));
    if (idx1 < g_Count) g_PrefixOutput.store(int(idx1), gsPrefix[lane + PREFIX_THREADS] + (g_Exclusive != 0u ? 0u : value1));
}

[shader("compute")]
[numthreads(PREFIX_THREADS, 1, 1)]
void csPrefixAddOffsets(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint index = dispatchThreadID.x;
    if (index >= g_Count) return;
    uint blockId = index / PREFIX_BLOCK_SIZE;
    if (blockId == 0u) return;
    g_PrefixOutput.store(int(index), g_PrefixOutput[int(index)] + g_PrefixBlockOffsets[int(blockId)]);
}

[shader("compute")]
[numthreads(PREFIX_THREADS, 1, 1)]
void csRadixPrefixLevel(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID) {
    uint lane = localID.x;
    uint totalN = radix_total_n(g_NumGroups);
    uint level = g_Shift;
    uint levelSize = radix_level_size(totalN, level);
    uint levelOffset = radix_level_offset(totalN, level);
    uint globalIndex = groupID.x * PREFIX_THREADS + lane;
    if (globalIndex < levelSize) {
        if (level == 0u) {
            uint digit = globalIndex / g_NumGroups;
            uint group = globalIndex % g_NumGroups;
            gsRadixPrefix[lane] = unpack_count(g_Histogram[int((digit >> RADIX_PACK_LANE_BITS) * g_NumGroups + group)], digit);
        } else {
            uint prevLevel = level - 1u;
            uint prevLevelSize = radix_level_size(totalN, prevLevel);
            uint blockEnd = min((globalIndex + 1u) * PREFIX_THREADS, prevLevelSize) - 1u;
            gsRadixPrefix[lane] = prevLevel == 0u
                ? unpack_prefix16(g_HistogramOffsets[int(radix_level_offset(totalN, prevLevel) + (blockEnd >> 1u))], blockEnd & 1u)
                : g_HistogramOffsets[int(radix_level_offset(totalN, prevLevel) + blockEnd)];
        }
    } else {
        gsRadixPrefix[lane] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    [unroll] for (uint stride = 1u; stride < PREFIX_THREADS; stride <<= 1u) {
        uint value = lane >= stride ? gsRadixPrefix[lane - stride] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gsRadixPrefix[lane] += value;
        GroupMemoryBarrierWithGroupSync();
    }

    if (globalIndex >= levelSize) return;
    if (level == 0u) {
        if ((lane & 1u) == 0u) {
            uint hi = (lane + 1u < PREFIX_THREADS && globalIndex + 1u < levelSize) ? gsRadixPrefix[lane + 1u] : 0u;
            g_HistogramOffsets.store(int(levelOffset + (globalIndex >> 1u)), pack_prefix16(gsRadixPrefix[lane], hi));
        }
    } else {
        g_HistogramOffsets.store(int(levelOffset + globalIndex), gsRadixPrefix[lane]);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void csPrefixWriteTotal(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint total = 0u;
    if (g_Count != 0u) {
        total = g_PrefixOutput[int(g_Count - 1u)];
        if (g_Exclusive != 0u) total += g_PrefixInput[int(g_Count - 1u)];
    }
    g_TotalOut.store(0, total);
}

[shader("compute")]
[numthreads(RADIX_GROUP_SIZE, 1, 1)]
void csRadixHistogram(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID) {
    uint lane = localID.x;
    if (lane < PACKED_HIST_SIZE) gsHistogram[lane] = 0u;
    GroupMemoryBarrierWithGroupSync();

    uint index = groupID.x * RADIX_GROUP_SIZE + lane;
    if (index < g_ElementCount) {
        uint digit = (g_KeysIn[int(index)] >> g_Shift) & g_DigitMask;
        InterlockedAdd(gsHistogram[digit >> RADIX_PACK_LANE_BITS], pack_count(digit, 1u));
    }
    GroupMemoryBarrierWithGroupSync();

    if (lane < PACKED_HIST_SIZE) g_Histogram.store(int(lane * g_NumGroups + groupID.x), gsHistogram[lane]);
}

[shader("compute")]
[numthreads(RADIX_GROUP_SIZE, 1, 1)]
void csRadixScatter(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID) {
    uint lane = localID.x;
    uint index = groupID.x * RADIX_GROUP_SIZE + lane;
    bool active = index < g_ElementCount;
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

    uint totalN = radix_total_n(g_NumGroups);
    uint numLevels = radix_num_levels(totalN);
    uint globalOffset = 0u;
    uint idx = digit * g_NumGroups + groupID.x;
    uint levelOffset = 0u;
    uint levelSize = totalN;
    for (uint level = 0u; level < numLevels; ++level) {
        if ((idx % PREFIX_THREADS) != 0u) {
            uint prev = idx - 1u;
            globalOffset += level == 0u
                ? unpack_prefix16(g_HistogramOffsets[int(levelOffset + (prev >> 1u))], prev & 1u)
                : g_HistogramOffsets[int(levelOffset + prev)];
        }
        levelOffset += level == 0u ? (levelSize + 1u) / 2u : levelSize;
        levelSize = ceil_div_block(levelSize);
        idx /= PREFIX_THREADS;
    }
    uint dest = globalOffset + quarterOffset + localOffset;
    g_KeysOut.store(int(dest), g_KeysIn[int(index)]);
    g_ValuesOut.store(int(dest), g_ValuesIn[int(index)]);
}
