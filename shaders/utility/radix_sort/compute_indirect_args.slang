import "utility/radix_sort/compute_indirect_args_utils.slang";

uniform RWStructuredBuffer<uint> g_IndirectArgs;
uniform uint g_ElementCount;
uniform uint g_MaxElementCount;

[shader("compute")]
[numthreads(1, 1, 1)]
void csComputeIndirectArgs()
{
    write_indirect_args(g_IndirectArgs, g_ElementCount, g_MaxElementCount);
}
