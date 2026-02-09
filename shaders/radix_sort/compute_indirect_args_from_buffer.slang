import "common.slang";

static const uint HISTOGRAM_ARGS_OFFSET = 0;
static const uint PREFIX_L0_ARGS_OFFSET = 3;
static const uint PREFIX_L1_ARGS_OFFSET = 6;
static const uint PREFIX_L2_ARGS_OFFSET = 9;
static const uint PREFIX_L3_ARGS_OFFSET = 12;
static const uint SCATTER_ARGS_OFFSET = 15;
static const uint PARAM_ELEMENT_COUNT = 18;
static const uint PARAM_NUM_GROUPS = 19;
static const uint PARAM_TOTAL_N = 20;
static const uint PARAM_NUM_LEVELS = 21;

uniform StructuredBuffer<uint> g_ElementCountBuffer;
uniform uint g_elementCountOffset;
uniform RWStructuredBuffer<uint> g_IndirectArgs;

uint div_ceil(uint a, uint b) { return (a + b - 1) / b; }

[shader("compute")]
[numthreads(1, 1, 1)]
void csComputeIndirectArgsFromBuffer() {
    uint n = g_ElementCountBuffer[g_elementCountOffset];
    uint numGroups = max(div_ceil(n, GROUP_SIZE), 1u);
    uint total_n = numGroups * HISTOGRAM_SIZE;
    g_IndirectArgs[HISTOGRAM_ARGS_OFFSET] = numGroups; g_IndirectArgs[HISTOGRAM_ARGS_OFFSET + 1] = 1; g_IndirectArgs[HISTOGRAM_ARGS_OFFSET + 2] = 1;
    uint sz = total_n, argsOffset = PREFIX_L0_ARGS_OFFSET;
    for (uint i = 0; i < 4; i++) {
        uint dispatch = div_ceil(sz, BLOCK_SIZE);
        g_IndirectArgs[argsOffset] = dispatch; g_IndirectArgs[argsOffset + 1] = 1; g_IndirectArgs[argsOffset + 2] = 1;
        sz = dispatch;
        argsOffset += 3;
    }
    g_IndirectArgs[SCATTER_ARGS_OFFSET] = numGroups; g_IndirectArgs[SCATTER_ARGS_OFFSET + 1] = 1; g_IndirectArgs[SCATTER_ARGS_OFFSET + 2] = 1;
    g_IndirectArgs[PARAM_ELEMENT_COUNT] = n; g_IndirectArgs[PARAM_NUM_GROUPS] = numGroups;
    g_IndirectArgs[PARAM_TOTAL_N] = total_n; g_IndirectArgs[PARAM_NUM_LEVELS] = 0;
}
