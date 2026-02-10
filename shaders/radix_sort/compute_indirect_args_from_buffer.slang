import "compute_indirect_args_utils.slang";

uniform StructuredBuffer<uint> g_ElementCountBuffer;
uniform uint g_elementCountOffset;
uniform uint g_maxElementCount;
uniform RWStructuredBuffer<uint> g_IndirectArgs;

[shader("compute")]
[numthreads(1, 1, 1)]
void csComputeIndirectArgsFromBuffer() {
    uint n = g_ElementCountBuffer[g_elementCountOffset];
    write_indirect_args(g_IndirectArgs, n, g_maxElementCount);
}
