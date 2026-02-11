import "gaussian_types.slang";
import "gaussian_scanline_utils.slang";

void emit_splat_bins(SplatBinRequest request)
{
    Ellipse ellipse = request.ellipse;
    if (!ellipse.isVisible(
        request.camDistance,
        request.depthValue,
        g_Camera.viewport,
        g_Camera.nearDepth,
        g_Camera.farDepth
    )) return;

    float tileSize = float(g_Prepass.tileSize);
    float halfTile = 0.5 * tileSize;
    if (!try_prepare_conic_for_binning(ellipse, halfTile)) return;
    float2 boundsExtent = ellipse.bboxExtent + halfTile;
    int minX = int(floor((ellipse.centerPx.x - boundsExtent.x) / tileSize));
    int maxX = int(ceil((ellipse.centerPx.x + boundsExtent.x) / tileSize));
    int minY = int(floor((ellipse.centerPx.y - boundsExtent.y) / tileSize));
    int maxY = int(ceil((ellipse.centerPx.y + boundsExtent.y) / tileSize));
    minX = clamp(minX, 0, int(g_Prepass.tileWidth) - 1);
    maxX = clamp(maxX, 0, int(g_Prepass.tileWidth) - 1);
    minY = clamp(minY, 0, int(g_Prepass.tileHeight) - 1);
    maxY = clamp(maxY, 0, int(g_Prepass.tileHeight) - 1);
    if (minX > maxX || minY > maxY) return;

    ScanlineLoopConfig loopCfg = build_scanline_loop_config(ellipse, minX, maxX, minY, maxY);
    uint scanlineCount = uint(loopCfg.lineEnd - loopCfg.lineStart + 1);
    if (scanlineCount == 0u) return;

    uint scanlineBase = 0;
    InterlockedAdd(g_ScanlineCounter[0], scanlineCount, scanlineBase);
    if (scanlineBase >= g_Prepass.maxScanlineEntries) return;

    uint scanlineWriteCount = min(scanlineCount, g_Prepass.maxScanlineEntries - scanlineBase);
    if (scanlineWriteCount == 0u) return;

    uint writtenListCount = 0u;
    for (uint i = 0; i < scanlineWriteCount; i++)
    {
        uint lineCoord = uint(loopCfg.lineStart + int(i));
        ScanlineSpan span;
        ScanlineWorkItem item;
        item.splatId = request.splatId;
        item.depthCode = request.depthCode;
        item.lineCoord = lineCoord;
        item.lineStartMinor = 0u;
        item.lineCount = 0u;
        item.splatListOffset = 0u;
        item.scanAlongX = loopCfg.scanAlongX ? 1u : 0u;
        item.reserved0 = 0u;
        bool hasSpan = compute_scanline_tile_span_universal(ellipse, loopCfg, tileSize, lineCoord, span);
        if (hasSpan && span.lineCount > 0u)
        {
            item.lineStartMinor = span.lineStartMinor;
            item.lineCount = span.lineCount;
            item.splatListOffset = writtenListCount;
            writtenListCount += span.lineCount;
        }

        uint scanlineIndex = scanlineBase + i;
        g_ScanlineWorkItems[scanlineIndex] = item;
    }
    if (writtenListCount == 0u) return;

    uint listBase = 0u;
    InterlockedAdd(g_ListCounter[0], writtenListCount, listBase);
    g_SplatListBases[request.splatId] = listBase;
}
