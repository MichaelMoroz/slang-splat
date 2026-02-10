import "gaussian_types.slang";
import "gaussian_math_utils.slang";
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
    ))
    {
        return;
    }

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
    if (minX > maxX || minY > maxY)
    {
        return;
    }

    ScanlineLoopConfig loopCfg = build_scanline_loop_config(ellipse, minX, maxX, minY, maxY);

    uint totalCount = 0;
    for (int lineCoord = loopCfg.lineStart; lineCoord <= loopCfg.lineEnd; lineCoord++)
    {
        ScanlineSpan span;
        if (!compute_scanline_tile_span_universal(
            ellipse,
            loopCfg,
            tileSize,
            halfTile,
            loopCfg.lineCoordToUint(lineCoord),
            span
        ))
        {
            continue;
        }
        if (totalCount > 0xFFFFFFFFu - span.lineCount)
        {
            totalCount = 0xFFFFFFFFu;
            break;
        }
        totalCount += span.lineCount;
    }
    if (totalCount == 0u)
    {
        return;
    }

    uint baseIndex = 0;
    InterlockedAdd(g_ListCounter[0], totalCount, baseIndex);
    if (baseIndex >= g_maxListEntries)
    {
        return;
    }

    uint writeCount = min(totalCount, g_maxListEntries - baseIndex);
    uint depthCode = request.depthCode;
    uint writeIndex = baseIndex;
    uint written = 0;
    for (int lineCoord = loopCfg.lineStart; lineCoord <= loopCfg.lineEnd && written < writeCount; lineCoord++)
    {
        ScanlineSpan span;
        uint lineCoordU = loopCfg.lineCoordToUint(lineCoord);
        if (!compute_scanline_tile_span_universal(
            ellipse,
            loopCfg,
            tileSize,
            halfTile,
            lineCoordU,
            span
        ))
        {
            continue;
        }
        if (!loopCfg.isLineCoordValid(lineCoordU))
        {
            continue;
        }

        uint lineWriteCount = min(span.lineCount, writeCount - written);
        for (uint i = 0; i < lineWriteCount; i++)
        {
            if (writeIndex >= g_maxListEntries)
            {
                break;
            }

            uint minorCoord = span.lineStartMinor + i;
            if (!loopCfg.isMinorCoordValid(minorCoord))
            {
                continue;
            }

            uint tileId = loopCfg.composeTileId(lineCoordU, minorCoord, g_tileWidth);
            if (tileId >= g_tileCount)
            {
                continue;
            }

            g_Keys[writeIndex] = (tileId << g_depthBits) | depthCode;
            g_Values[writeIndex] = request.splatId;
            writeIndex++;
        }
        written += lineWriteCount;
    }
}
