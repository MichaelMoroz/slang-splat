import "utility/radix_sort/compute_indirect_args_utils.slang";

uniform StructuredBuffer<uint> g_CountBuffer;
uniform uint g_CountOffset;
uniform uint g_MaxElementCount;
uniform RWStructuredBuffer<uint> g_IndirectArgs;

[shader("compute")]
[numthreads(1, 1, 1)]
void csComputeIndirectArgsFromBuffer()
{
    uint n = g_CountBuffer[g_CountOffset];
    write_indirect_args(g_IndirectArgs, n, g_MaxElementCount);
}
