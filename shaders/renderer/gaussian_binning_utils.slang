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
    if (!try_prepare_conic_for_binning(ellipse, halfTile)) return;
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
    uint scanlineCount = uint(loopCfg.lineEnd - loopCfg.lineStart + 1);
    if (scanlineCount == 0u) return;

    uint scanlineBase = 0;
    InterlockedAdd(g_ScanlineCounter[0], scanlineCount, scanlineBase);
    if (scanlineBase >= g_maxScanlineEntries) return;

    uint scanlineWriteCount = min(scanlineCount, g_maxScanlineEntries - scanlineBase);
    for (uint i = 0; i < scanlineWriteCount; i++)
    {
        uint scanlineIndex = scanlineBase + i;
        int lineCoord = loopCfg.lineStart + int(i);
        ScanlineWorkItem item;
        item.splatId = request.splatId;
        item.depthCode = request.depthCode;
        item.lineCoord = uint(lineCoord);
        item.minorMin = uint(loopCfg.minorMin);
        item.minorMax = uint(loopCfg.minorMax);
        item.scanAlongX = loopCfg.scanAlongX ? 1u : 0u;
        item.reserved0 = 0u;
        item.reserved1 = 0u;
        g_ScanlineWorkItems[scanlineIndex] = item;
    }
}
