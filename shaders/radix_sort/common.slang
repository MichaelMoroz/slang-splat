// Common constants and utilities for radix sort
static const uint GROUP_SIZE = 128, HISTOGRAM_SIZE = 256, PACKED_HIST_SIZE = 64, BLOCK_SIZE = 256, BITS_PER_PASS = 8;

// Params buffer layout: [dispatch args (18), elementCount, numGroups, totalN, numLevels] at indices 18-21
uniform StructuredBuffer<uint> g_ParamsBuffer;
uniform uint g_useParamsBuffer, g_elementCount, g_numGroups, g_shift;

uint getElementCount() { return (g_useParamsBuffer != 0) ? g_ParamsBuffer[18] : g_elementCount; }
uint getNumGroups() { return (g_useParamsBuffer != 0) ? g_ParamsBuffer[19] : g_numGroups; }
uint getTotalN() { return (g_useParamsBuffer != 0) ? g_ParamsBuffer[20] : g_numGroups * HISTOGRAM_SIZE; }
uint getNumLevels() { return (g_useParamsBuffer != 0) ? g_ParamsBuffer[21] : 0; }

// Pack/unpack 4x8-bit histogram counters
uint packCount(uint digit, uint count) { return count << ((digit & 3) * 8); }
uint unpackCount(uint packed, uint digit) { return (packed >> ((digit & 3) * 8)) & 0xFF; }

// Pack/unpack 2x16-bit prefix sums (lo = even, hi = odd)
uint packPrefix16(uint lo, uint hi) { return (hi << 16) | (lo & 0xFFFF); }
uint unpackPrefixLo(uint packed) { return packed & 0xFFFF; }
uint unpackPrefixHi(uint packed) { return packed >> 16; }
uint unpackPrefix16(uint packed, uint isOdd) { return (isOdd != 0) ? (packed >> 16) : (packed & 0xFFFF); }
