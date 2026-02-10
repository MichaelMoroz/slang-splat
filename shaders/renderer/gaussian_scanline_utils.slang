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
    if (!useConic) { ellipse.useConic = 0u; return false; }

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
    if (!useConic) { ellipse.useConic = 0u; return false; }

    float xExtent = sqrt(max(conic.z / detConic, 0.0));
    float yExtent = sqrt(max(conic.x / detConic, 0.0));
    if (!all(isfinite(float2(xExtent, yExtent)))) { ellipse.useConic = 0u; return false; }

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

bool compute_scanline_tile_span_universal(
    Ellipse ellipse,
    ScanlineLoopConfig cfg,
    float tileSize,
    uint lineCoord,
    out ScanlineSpan span
)
{
    span.lineStartMinor = 0;
    span.lineCount = 0;
    if (!cfg.isLineCoordValid(lineCoord)) return false;

    int first = -1;
    int last = -1;
    for (int minor = cfg.minorMin; minor <= cfg.minorMax; minor++)
    {
        if (minor < 0) continue;
        uint minorU = uint(minor);
        if (!cfg.isMinorCoordValid(minorU)) continue;
        float2 boxMin;
        float2 boxMax;
        if (cfg.scanAlongX)
        {
            boxMin = float2(float(minorU) * tileSize, float(lineCoord) * tileSize) - ellipse.centerPx;
            boxMax = float2(float(minorU + 1u) * tileSize, float(lineCoord + 1u) * tileSize) - ellipse.centerPx;
        }
        else
        {
            boxMin = float2(float(lineCoord) * tileSize, float(minorU) * tileSize) - ellipse.centerPx;
            boxMax = float2(float(lineCoord + 1u) * tileSize, float(minorU + 1u) * tileSize) - ellipse.centerPx;
        }
        if (!ellipse.intersectsTileBox(boxMin, boxMax)) continue;
        if (first < 0) first = minor;
        last = minor;
    }

    if (first < 0 || last < first) return false;

    span.lineStartMinor = uint(first);
    span.lineCount = uint(last - first + 1);
    return true;
}
