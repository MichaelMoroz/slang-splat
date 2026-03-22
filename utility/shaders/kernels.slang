import "slangpy";

static const uint PREFIX_THREADS = 256u;
static const uint PREFIX_BLOCK_SIZE = PREFIX_THREADS * 2u;
static const uint RADIX_GROUP_SIZE = 128u;
static const uint RADIX_BIN_COUNT = 256u;

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
uniform Tensor<uint, 1> g_HistogramOffsets;
uniform uint g_ElementCount;
uniform uint g_NumGroups;
uniform uint g_Shift;
uniform uint g_DigitMask;

groupshared uint gsPrefix[PREFIX_BLOCK_SIZE];
groupshared uint gsHistogram[RADIX_BIN_COUNT];
groupshared uint gsDigits[RADIX_GROUP_SIZE];

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
    for (uint digit = lane; digit < RADIX_BIN_COUNT; digit += RADIX_GROUP_SIZE) gsHistogram[digit] = 0u;
    GroupMemoryBarrierWithGroupSync();

    uint index = groupID.x * RADIX_GROUP_SIZE + lane;
    if (index < g_ElementCount) {
        uint digit = (g_KeysIn[int(index)] >> g_Shift) & g_DigitMask;
        InterlockedAdd(gsHistogram[digit], 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint digit = lane; digit < RADIX_BIN_COUNT; digit += RADIX_GROUP_SIZE) {
        g_Histogram.store(int(digit * g_NumGroups + groupID.x), gsHistogram[digit]);
    }
}

[shader("compute")]
[numthreads(RADIX_GROUP_SIZE, 1, 1)]
void csRadixScatter(uint3 groupID : SV_GroupID, uint3 localID : SV_GroupThreadID) {
    uint lane = localID.x;
    uint index = groupID.x * RADIX_GROUP_SIZE + lane;
    bool active = index < g_ElementCount;
    uint digit = active ? ((g_KeysIn[int(index)] >> g_Shift) & g_DigitMask) : 0xFFFFFFFFu;
    gsDigits[lane] = digit;
    GroupMemoryBarrierWithGroupSync();
    if (!active) return;

    uint localOffset = 0u;
    [unroll] for (uint prev = 0u; prev < RADIX_GROUP_SIZE; prev++) {
        if (prev >= lane) break;
        localOffset += gsDigits[prev] == digit ? 1u : 0u;
    }

    uint dest = g_HistogramOffsets[int(digit * g_NumGroups + groupID.x)] + localOffset;
    g_KeysOut.store(int(dest), g_KeysIn[int(index)]);
    g_ValuesOut.store(int(dest), g_ValuesIn[int(index)]);
}
