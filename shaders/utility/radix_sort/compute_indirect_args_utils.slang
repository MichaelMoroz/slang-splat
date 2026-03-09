import "utility/radix_sort/common.slang";

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
    for (uint i = 0u; i < RADIX_MAX_PREFIX_LEVELS && levelSize > 1u; ++i)
    {
        numLevels += 1u;
        levelSize = ceil_div_block(levelSize);
    }
    numLevels = max(numLevels + 1u, 1u);

    uint dispatchSize = totalN;
    uint argsOffset = PREFIX_L0_ARGS_OFFSET;
    for (uint i = 0u; i < RADIX_MAX_PREFIX_LEVELS; ++i)
    {
        uint dispatch = ceil_div_block(dispatchSize);
        indirectArgs[argsOffset] = dispatch;
        indirectArgs[argsOffset + 1u] = 1u;
        indirectArgs[argsOffset + 2u] = 1u;
        dispatchSize = dispatch;
        argsOffset += INDIRECT_DISPATCH_ARG_STRIDE;
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
