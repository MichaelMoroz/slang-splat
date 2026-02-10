import "common.slang";

static const uint HISTOGRAM_ARGS_OFFSET = 0;
static const uint PREFIX_L0_ARGS_OFFSET = 3;
static const uint SCATTER_ARGS_OFFSET = 15;
static const uint PARAM_ELEMENT_COUNT = 18;
static const uint PARAM_NUM_GROUPS = 19;
static const uint PARAM_TOTAL_N = 20;
static const uint PARAM_NUM_LEVELS = 21;
static const uint BUILD_RANGE_ARGS_OFFSET = 22;

void write_indirect_args(RWStructuredBuffer<uint> indirectArgs, uint elementCount, uint maxElementCount)
{
    uint n = (maxElementCount != 0u) ? min(elementCount, maxElementCount) : elementCount;
    uint numGroups = max(ceil_div_group(n), 1u);
    uint totalN = numGroups * HISTOGRAM_SIZE;
    indirectArgs[HISTOGRAM_ARGS_OFFSET] = numGroups;
    indirectArgs[HISTOGRAM_ARGS_OFFSET + 1u] = 1u;
    indirectArgs[HISTOGRAM_ARGS_OFFSET + 2u] = 1u;

    uint numLevels = 0u;
    uint levelSize = totalN;
    for (uint i = 0u; i < 4u && levelSize > 1u; ++i)
    {
        numLevels += 1u;
        levelSize = ceil_div_block(levelSize);
    }
    numLevels = max(numLevels + 1u, 1u);

    uint dispatchSize = totalN;
    uint argsOffset = PREFIX_L0_ARGS_OFFSET;
    for (uint i = 0u; i < 4u; ++i)
    {
        uint dispatch = ceil_div_block(dispatchSize);
        indirectArgs[argsOffset] = dispatch;
        indirectArgs[argsOffset + 1u] = 1u;
        indirectArgs[argsOffset + 2u] = 1u;
        dispatchSize = dispatch;
        argsOffset += 3u;
    }

    indirectArgs[SCATTER_ARGS_OFFSET] = numGroups;
    indirectArgs[SCATTER_ARGS_OFFSET + 1u] = 1u;
    indirectArgs[SCATTER_ARGS_OFFSET + 2u] = 1u;

    uint buildDispatch = max(ceil_div_block(n), 1u);
    indirectArgs[BUILD_RANGE_ARGS_OFFSET] = buildDispatch;
    indirectArgs[BUILD_RANGE_ARGS_OFFSET + 1u] = 1u;
    indirectArgs[BUILD_RANGE_ARGS_OFFSET + 2u] = 1u;

    indirectArgs[PARAM_ELEMENT_COUNT] = n;
    indirectArgs[PARAM_NUM_GROUPS] = numGroups;
    indirectArgs[PARAM_TOTAL_N] = totalN;
    indirectArgs[PARAM_NUM_LEVELS] = numLevels;
}
