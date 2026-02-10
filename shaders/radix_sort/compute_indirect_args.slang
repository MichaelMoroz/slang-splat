import "compute_indirect_args_utils.slang";

uniform RWStructuredBuffer<uint> g_IndirectArgs;
uniform uint g_elementCount;
uniform uint g_maxElementCount;

[shader("compute")]
[numthreads(1, 1, 1)]
void csComputeIndirectArgs() {
    write_indirect_args(g_IndirectArgs, g_elementCount, g_maxElementCount);
}
