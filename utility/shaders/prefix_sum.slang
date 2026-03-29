import "slangpy";

static const uint PREFIX_THREADS = 256u;
static const uint PREFIX_BLOCK_SIZE = PREFIX_THREADS * 2u;
static const uint RADIX_GROUP_SIZE = 512u;
static const uint RADIX_BIN_COUNT = 256u;
static const uint RADIX_MAX_PREFIX_LEVELS = 4u;
static const uint PREFIX_MAX_LEVELS = 8u;
static const uint INDIRECT_DISPATCH_ARG_STRIDE = 3u;
static const uint PREFIX_SCAN_ARGS_OFFSET = 0u;
static const uint PREFIX_ADD_ARGS_OFFSET = PREFIX_SCAN_ARGS_OFFSET + PREFIX_MAX_LEVELS * INDIRECT_DISPATCH_ARG_STRIDE;
static const uint PREFIX_PARAM_COUNT_OFFSET = PREFIX_ADD_ARGS_OFFSET + PREFIX_MAX_LEVELS * INDIRECT_DISPATCH_ARG_STRIDE;
static const uint HISTOGRAM_ARGS_OFFSET = 0u;
static const uint PREFIX_L0_ARGS_OFFSET = HISTOGRAM_ARGS_OFFSET + INDIRECT_DISPATCH_ARG_STRIDE;
static const uint SCATTER_ARGS_OFFSET = PREFIX_L0_ARGS_OFFSET + RADIX_MAX_PREFIX_LEVELS * INDIRECT_DISPATCH_ARG_STRIDE;
static const uint BUILD_RANGE_ARGS_OFFSET = SCATTER_ARGS_OFFSET + INDIRECT_DISPATCH_ARG_STRIDE;
static const uint PARAM_ELEMENT_COUNT = BUILD_RANGE_ARGS_OFFSET + INDIRECT_DISPATCH_ARG_STRIDE;
static const uint PARAM_NUM_GROUPS = PARAM_ELEMENT_COUNT + 1u;
static const uint PARAM_TOTAL_N = PARAM_NUM_GROUPS + 1u;
static const uint PARAM_NUM_LEVELS = PARAM_TOTAL_N + 1u;

uint ceil_div_block(uint value) { return (value + PREFIX_THREADS - 1u) / PREFIX_THREADS; }

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
uniform RWTensor<uint, 1> g_HistogramOffsets;
uniform uint g_NumGroups;
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

    if (WaveIsFirstLane()) {
        gsPrefixWaveTotals[waveId] = waveTotal;
    }
    GroupMemoryBarrierWithGroupSync();

    if (waveId == 0u) {
        uint reshuffledTotal = localId < numWaves ? gsPrefixWaveTotals[localId] : 0u;
        uint reshuffledPrefix = WavePrefixSum(reshuffledTotal);
        if (localId < numWaves) {
            gsPrefixWaveScan[localId] = reshuffledPrefix;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    blockTotal = gsPrefixWaveScan[numWaves - 1u] + gsPrefixWaveTotals[numWaves - 1u];
    uint waveBase = gsPrefixWaveScan[waveId];
    GroupMemoryBarrierWithGroupSync();
    return waveBase + wavePrefix;
}

void write_dispatch_args(RWTensor<uint, 1> indirectArgs, uint baseOffset, uint groupCount) {
    indirectArgs.store(int(baseOffset), groupCount);
    indirectArgs.store(int(baseOffset + 1u), 1u);
    indirectArgs.store(int(baseOffset + 2u), 1u);
}

uint radix_total_n(uint numGroups) { return numGroups * RADIX_BIN_COUNT; }
uint radix_level_size(uint totalN, uint level) { uint size = totalN; for (uint current = 0u; current < level; ++current) size = ceil_div_block(size); return size; }
uint radix_level_offset(uint totalN, uint level) { uint offset = 0u; uint size = totalN; for (uint current = 0u; current < level; ++current) { offset += size; size = ceil_div_block(size); } return offset; }
uint radix_num_levels(uint totalN) { uint levels = 0u; uint size = totalN; for (uint current = 0u; current < RADIX_MAX_PREFIX_LEVELS && size > 1u; ++current) { levels += 1u; size = ceil_div_block(size); } return max(levels, 1u); }
uint load_radix_param(uint index, uint fallback) { return g_UseRadixParams != 0u ? g_RadixParamsBuffer[index] : fallback; }

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
