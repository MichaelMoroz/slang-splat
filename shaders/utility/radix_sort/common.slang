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
uint radix_total_n(uint numGroups) { return numGroups * RADIX_BIN_COUNT; }
uint radix_level_size(uint totalN, uint level) { uint size = totalN; for (uint current = 0u; current < level; ++current) size = ceil_div_block(size); return size; }
uint radix_level_offset(uint totalN, uint level) { uint offset = 0u; uint size = totalN; for (uint current = 0u; current < level; ++current) { offset += size; size = ceil_div_block(size); } return offset; }
uint radix_num_levels(uint totalN) { uint levels = 0u; uint size = totalN; for (uint current = 0u; current < RADIX_MAX_PREFIX_LEVELS && size > 1u; ++current) { levels += 1u; size = ceil_div_block(size); } return max(levels, 1u); }
uint load_radix_param(StructuredBuffer<uint> buffer, uint useParams, uint index, uint fallback) { return useParams != 0u ? buffer[index] : fallback; }
