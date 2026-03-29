import "utility/radix_sort/common.slang";

void write_dispatch_args_radix(RWStructuredBuffer<uint> indirectArgs, uint baseOffset, uint groupCount)
{
    indirectArgs[baseOffset] = groupCount;
    indirectArgs[baseOffset + 1u] = 1u;
    indirectArgs[baseOffset + 2u] = 1u;
}

void write_indirect_args(RWStructuredBuffer<uint> indirectArgs, uint elementCount, uint maxElementCount)
{
    uint n = (maxElementCount != 0u) ? min(elementCount, maxElementCount) : elementCount;
    uint numGroups = max((n + RADIX_GROUP_SIZE - 1u) / RADIX_GROUP_SIZE, 1u);
    uint totalN = numGroups * RADIX_BIN_COUNT;
    write_dispatch_args_radix(indirectArgs, HISTOGRAM_ARGS_OFFSET, numGroups);

    uint dispatchSize = totalN;
    uint argsOffset = PREFIX_L0_ARGS_OFFSET;
    for (uint i = 0u; i < RADIX_MAX_PREFIX_LEVELS; ++i)
    {
        write_dispatch_args_radix(indirectArgs, argsOffset, max(ceil_div_block(dispatchSize), 1u));
        dispatchSize = ceil_div_block(dispatchSize);
        argsOffset += INDIRECT_DISPATCH_ARG_STRIDE;
    }

    write_dispatch_args_radix(indirectArgs, SCATTER_ARGS_OFFSET, numGroups);
    write_dispatch_args_radix(indirectArgs, BUILD_RANGE_ARGS_OFFSET, max(ceil_div_block(n), 1u));
    indirectArgs[PARAM_ELEMENT_COUNT] = n;
    indirectArgs[PARAM_NUM_GROUPS] = numGroups;
    indirectArgs[PARAM_TOTAL_N] = totalN;
    indirectArgs[PARAM_NUM_LEVELS] = radix_num_levels(totalN);
}
