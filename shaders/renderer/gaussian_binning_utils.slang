import "gaussian_types.slang";
import "gaussian_scanline_utils.slang";

void emit_splat_bins(SplatBinRequest request)
{
    Ellipse ellipse = request.ellipse;
    if (!ellipse.isVisible(
        request.camDistance,
        request.depthValue,
        g_viewport,
        g_nearDepth,
        g_farDepth
    )) return;

    float tileSize = float(g_tileSize);
    float halfTile = 0.5 * tileSize;
    try_prepare_conic_for_binning(ellipse, halfTile);
    float2 boundsExtent = ellipse.bboxExtent + halfTile;
    int minX = int(floor((ellipse.centerPx.x - boundsExtent.x) / tileSize));
    int maxX = int(ceil((ellipse.centerPx.x + boundsExtent.x) / tileSize));
    int minY = int(floor((ellipse.centerPx.y - boundsExtent.y) / tileSize));
    int maxY = int(ceil((ellipse.centerPx.y + boundsExtent.y) / tileSize));
    minX = clamp(minX, 0, int(g_tileWidth) - 1);
    maxX = clamp(maxX, 0, int(g_tileWidth) - 1);
    minY = clamp(minY, 0, int(g_tileHeight) - 1);
    maxY = clamp(maxY, 0, int(g_tileHeight) - 1);
    if (minX > maxX || minY > maxY) return;

    ScanlineLoopConfig loopCfg = build_scanline_loop_config(ellipse, minX, maxX, minY, maxY);

    uint totalCount = 0;
    for (int lineCoord = loopCfg.lineStart; lineCoord <= loopCfg.lineEnd; lineCoord++)
    {
        ScanlineSpan span;
        if (!compute_scanline_tile_span_universal(
            ellipse,
            loopCfg,
            tileSize,
            loopCfg.lineCoordToUint(lineCoord),
            span
        )) continue;
        if (totalCount > UINT_MAX_VALUE - span.lineCount)
        {
            totalCount = UINT_MAX_VALUE;
            break;
        }
        totalCount += span.lineCount;
    }
    if (totalCount == 0u) return;

    uint baseIndex = 0;
    InterlockedAdd(g_ListCounter[0], totalCount, baseIndex);
    if (baseIndex >= g_maxListEntries) return;

    uint writeCount = min(totalCount, g_maxListEntries - baseIndex);
    if (writeCount == 0u) return;
    uint written = 0;
    for (int lineCoord = loopCfg.lineStart; lineCoord <= loopCfg.lineEnd && written < writeCount; lineCoord++)
    {
        ScanlineSpan span;
        uint lineCoordU = loopCfg.lineCoordToUint(lineCoord);
        if (!compute_scanline_tile_span_universal(
            ellipse,
            loopCfg,
            tileSize,
            lineCoordU,
            span
        )) continue;

        uint lineWriteCount = min(span.lineCount, writeCount - written);
        if (lineWriteCount == 0u) continue;

        uint tileBase = loopCfg.composeTileId(lineCoordU, span.lineStartMinor, g_tileWidth);
        uint tileStride = loopCfg.scanAlongX ? 1u : g_tileWidth;
        uint scanlineIndex = 0;
        InterlockedAdd(g_ScanlineCounter[0], 1u, scanlineIndex);
        if (scanlineIndex < g_maxScanlineEntries)
        {
            ScanlineWorkItem item;
            item.baseIndex = baseIndex + written;
            item.tileBase = tileBase;
            item.writeCount = lineWriteCount;
            item.splatId = request.splatId;
            item.depthCode = request.depthCode;
            item.tileStride = tileStride;
            item.reserved0 = 0u;
            item.reserved1 = 0u;
            g_ScanlineWorkItems[scanlineIndex] = item;
        }
        written += lineWriteCount;
    }
}
