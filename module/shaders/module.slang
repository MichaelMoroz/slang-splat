import "slangpy";
import "common.slang";

[CUDAKernel]
void clear_ranges(RWTensor<uint2, 1> tileRanges)
{
    uint idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (idx >= tileRanges.shape[0]) return;
    int[1] at = { int(idx) };
    tileRanges.store(at, uint2(INVALID_START, 0));
}

[CUDAKernel]
void build_ranges(Tensor<uint, 1> sortedTileIds, RWTensor<uint2, 1> tileRanges, uint sortedCount)
{
    uint idx = cudaBlockIdx().x * cudaBlockDim().x + cudaThreadIdx().x;
    if (idx >= sortedCount) return;
    int[1] at = { int(idx) };
    uint tile = sortedTileIds.load(at);
    uint prevTile = INVALID_START;
    uint nextTile = INVALID_START;
    if (idx > 0u)
    {
        int[1] prevAt = { int(idx - 1u) };
        prevTile = sortedTileIds.load(prevAt);
    }
    if (idx + 1u < sortedCount)
    {
        int[1] nextAt = { int(idx + 1u) };
        nextTile = sortedTileIds.load(nextAt);
    }
    int[1] tileAt = { int(tile) };
    uint2 range = tileRanges.load(tileAt);
    if (idx == 0u || tile != prevTile) range.x = idx;
    if (idx + 1u == sortedCount || tile != nextTile) range.y = idx + 1u;
    tileRanges.store(tileAt, range);
}
