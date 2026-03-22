static const uint PREFIX_THREADS = 256u;
static const uint PREFIX_BLOCK_SIZE = PREFIX_THREADS * 2u;
static const uint RADIX_GROUP_SIZE = 128u;
static const uint RADIX_BIN_COUNT = 256u;
static const uint RADIX_BITS_PER_PASS = 8u;

uint ceil_div_uint(uint a, uint b) { return (a + b - 1u) / b; }
