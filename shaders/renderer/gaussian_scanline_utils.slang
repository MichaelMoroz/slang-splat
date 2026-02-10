import "gaussian_types.slang";

bool try_prepare_conic_for_binning(inout Ellipse ellipse, float halfTile)
{
    ellipse.bboxExtent = float2(ellipse.radiusPx, ellipse.radiusPx);
    float3 conic = ellipse.conic;
    float detConic = conic.x * conic.z - conic.y * conic.y;
    bool useConic = all(isfinite(conic))
        && conic.x > 1e-10
        && conic.z > 1e-10
        && detConic > 1e-12;
    if (!useConic)
    {
        ellipse.useConic = 0u;
        return false;
    }

    float traceConic = conic.x + conic.z;
    float discConic = sqrt(max(0.25 * traceConic * traceConic - detConic, 0.0));
    float eigMin = 0.5 * traceConic - discConic;
    float eigMax = 0.5 * traceConic + discConic;
    float maxAxis = rsqrt(max(eigMin, 1e-20));
    float minAxis = rsqrt(max(eigMax, 1e-20));
    float radiusSafe = max(ellipse.radiusPx, 1.0);
    float axisLimit = radiusSafe * 2.0 + halfTile;
    useConic = isfinite(maxAxis)
        && isfinite(minAxis)
        && eigMin > 0.0
        && eigMax > 0.0
        && maxAxis <= axisLimit
        && minAxis >= 0.125;
    if (!useConic)
    {
        ellipse.useConic = 0u;
        return false;
    }

    float xExtent = sqrt(max(conic.z / detConic, 0.0));
    float yExtent = sqrt(max(conic.x / detConic, 0.0));
    if (!all(isfinite(float2(xExtent, yExtent))))
    {
        ellipse.useConic = 0u;
        return false;
    }

    ellipse.bboxExtent = clamp(
        float2(xExtent, yExtent),
        float2(1e-4, 1e-4),
        float2(ellipse.radiusPx, ellipse.radiusPx)
    );
    ellipse.useConic = 1u;
    return true;
}

ScanlineLoopConfig make_scanline_loop_config(
    bool scanAlongX,
    int minX,
    int maxX,
    int minY,
    int maxY
)
{
    ScanlineLoopConfig cfg;
    cfg.scanAlongX = scanAlongX;
    cfg.lineStart = cfg.scanAlongX ? minY : minX;
    cfg.lineEnd = cfg.scanAlongX ? maxY : maxX;
    cfg.minorMin = cfg.scanAlongX ? minX : minY;
    cfg.minorMax = cfg.scanAlongX ? maxX : maxY;
    cfg.minorLimit = cfg.scanAlongX ? g_tileWidth : g_tileHeight;
    cfg.lineLimit = cfg.scanAlongX ? g_tileHeight : g_tileWidth;
    return cfg;
}

ScanlineLoopConfig build_scanline_loop_config(
    Ellipse ellipse,
    int minX,
    int maxX,
    int minY,
    int maxY
)
{
    uint spanX = uint(maxX - minX + 1);
    uint spanY = uint(maxY - minY + 1);
    bool scanAlongX = ellipse.chooseScanAlongX(spanX, spanY);
    return make_scanline_loop_config(scanAlongX, minX, maxX, minY, maxY);
}

float eval_conic(float3 conic, float x, float y)
{
    return conic.x * x * x + 2.0 * conic.y * x * y + conic.z * y * y;
}

float min_conic_over_tile_box(float3 conic, float x0, float x1, float y0, float y1)
{
    float minVal = min(
        min(eval_conic(conic, x0, y0), eval_conic(conic, x1, y0)),
        min(eval_conic(conic, x0, y1), eval_conic(conic, x1, y1))
    );
    if (x0 <= 0.0 && 0.0 <= x1 && y0 <= 0.0 && 0.0 <= y1)
    {
        return 0.0;
    }

    float a = conic.x;
    float b = conic.y;
    float c = conic.z;
    if (c > 1e-12)
    {
        float yAtX0 = clamp(-b * x0 / c, y0, y1);
        float yAtX1 = clamp(-b * x1 / c, y0, y1);
        minVal = min(minVal, eval_conic(conic, x0, yAtX0));
        minVal = min(minVal, eval_conic(conic, x1, yAtX1));
    }
    if (a > 1e-12)
    {
        float xAtY0 = clamp(-b * y0 / a, x0, x1);
        float xAtY1 = clamp(-b * y1 / a, x0, x1);
        minVal = min(minVal, eval_conic(conic, xAtY0, y0));
        minVal = min(minVal, eval_conic(conic, xAtY1, y1));
    }
    return minVal;
}

bool tile_intersects_ellipse(
    Ellipse ellipse,
    ScanlineLoopConfig cfg,
    uint lineCoord,
    uint minorCoord,
    float tileSize
)
{
    float x0;
    float x1;
    float y0;
    float y1;
    if (cfg.scanAlongX)
    {
        x0 = float(minorCoord) * tileSize - ellipse.centerPx.x;
        x1 = float(minorCoord + 1u) * tileSize - ellipse.centerPx.x;
        y0 = float(lineCoord) * tileSize - ellipse.centerPx.y;
        y1 = float(lineCoord + 1u) * tileSize - ellipse.centerPx.y;
    }
    else
    {
        x0 = float(lineCoord) * tileSize - ellipse.centerPx.x;
        x1 = float(lineCoord + 1u) * tileSize - ellipse.centerPx.x;
        y0 = float(minorCoord) * tileSize - ellipse.centerPx.y;
        y1 = float(minorCoord + 1u) * tileSize - ellipse.centerPx.y;
    }

    if (ellipse.useConic != 0u)
    {
        float minVal = min_conic_over_tile_box(ellipse.conic, x0, x1, y0, y1);
        return minVal <= 1.0 + 1e-5;
    }

    float qx = clamp(0.0, x0, x1);
    float qy = clamp(0.0, y0, y1);
    float dist2 = qx * qx + qy * qy;
    return dist2 <= ellipse.radiusPx * ellipse.radiusPx;
}

bool compute_scanline_tile_span_universal(
    Ellipse ellipse,
    ScanlineLoopConfig cfg,
    float tileSize,
    float halfTile,
    uint lineCoord,
    out ScanlineSpan span
)
{
    span.lineStartMinor = 0;
    span.lineCount = 0;
    if (!cfg.isLineCoordValid(lineCoord))
    {
        return false;
    }

    int first = -1;
    int last = -1;
    for (int minor = cfg.minorMin; minor <= cfg.minorMax; minor++)
    {
        if (minor < 0)
        {
            continue;
        }
        uint minorU = uint(minor);
        if (!cfg.isMinorCoordValid(minorU))
        {
            continue;
        }
        if (!tile_intersects_ellipse(ellipse, cfg, lineCoord, minorU, tileSize))
        {
            continue;
        }
        if (first < 0)
        {
            first = minor;
        }
        last = minor;
    }

    if (first < 0 || last < first)
    {
        return false;
    }

    span.lineStartMinor = uint(first);
    span.lineCount = uint(last - first + 1);
    return true;
}
