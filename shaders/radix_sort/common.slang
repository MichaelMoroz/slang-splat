static const uint GROUP_SIZE = 128u;
static const uint HISTOGRAM_SIZE = 256u;
static const uint PACKED_HIST_SIZE = 64u;
static const uint BLOCK_SIZE = 256u;
static const uint RADIX_PAIR_SIZE = 2u;
static const uint RADIX_PACK_LANE_BITS = 2u;
static const uint RADIX_PACK_LANE_MASK = 3u;
static const uint RADIX_BYTE_SHIFT = 8u;
static const uint RADIX_PREFIX_SHIFT = 16u;
static const uint RADIX_BYTE_MASK = 0xFFu;
static const uint RADIX_PREFIX_MASK = 0xFFFFu;
static const uint RADIX_INVALID_VALUE = 0xFFFFFFFFu;
static const uint RADIX_MAX_PREFIX_LEVELS = 4u;
static const uint INDIRECT_DISPATCH_ARG_STRIDE = 3u;
static const uint HISTOGRAM_ARGS_OFFSET = 0u;
static const uint PREFIX_L0_ARGS_OFFSET = 3u;
static const uint SCATTER_ARGS_OFFSET = 15u;
static const uint PARAM_ELEMENT_COUNT = 18u;
static const uint PARAM_NUM_GROUPS = 19u;
static const uint PARAM_TOTAL_N = 20u;
static const uint PARAM_NUM_LEVELS = 21u;
static const uint BUILD_RANGE_ARGS_OFFSET = 22u;

uniform StructuredBuffer<uint> g_ParamsBuffer;
uniform uint g_useParamsBuffer, g_elementCount, g_numGroups, g_shift;

struct RadixDispatchConfig
{
    uint elementCount;
    uint numGroups;
    uint totalN;
    uint numLevels;

    [mutating]
    void load()
    {
        if (g_useParamsBuffer != 0u)
        {
            elementCount = g_ParamsBuffer[PARAM_ELEMENT_COUNT];
            numGroups = g_ParamsBuffer[PARAM_NUM_GROUPS];
            totalN = g_ParamsBuffer[PARAM_TOTAL_N];
            numLevels = g_ParamsBuffer[PARAM_NUM_LEVELS];
        } else {
            elementCount = g_elementCount;
            numGroups = g_numGroups;
            totalN = g_numGroups * HISTOGRAM_SIZE;
            numLevels = 0u;
        }
    }
};

uint ceil_div2(uint a) { return (a + RADIX_PAIR_SIZE - 1u) / RADIX_PAIR_SIZE; }
uint ceil_div_group(uint a) { return (a + GROUP_SIZE - 1u) / GROUP_SIZE; }
uint ceil_div_block(uint a) { return (a + BLOCK_SIZE - 1u) / BLOCK_SIZE; }

uint packCount(uint digit, uint count) { return count << ((digit & RADIX_PACK_LANE_MASK) * RADIX_BYTE_SHIFT); }
uint unpackCount(uint packed, uint digit) { return (packed >> ((digit & RADIX_PACK_LANE_MASK) * RADIX_BYTE_SHIFT)) & RADIX_BYTE_MASK; }

uint packPrefix16(uint lo, uint hi) { return (hi << RADIX_PREFIX_SHIFT) | (lo & RADIX_PREFIX_MASK); }
uint unpackPrefix16(uint packed, uint isOdd) { return (isOdd != 0u) ? (packed >> RADIX_PREFIX_SHIFT) : (packed & RADIX_PREFIX_MASK); }
