import "gaussian_types.slang";

bool try_prepare_conic_for_binning(inout Ellipse ellipse, float halfTile)
{
    ellipse.bboxExtent = float2(ellipse.radiusPx, ellipse.radiusPx);
    float3 conic = ellipse.conic;
    float detConic = conic.x * conic.z - conic.y * conic.y;
    bool useConic = all(isfinite(conic))
        && conic.x > EPS_CONIC_POSITIVE
        && conic.z > EPS_CONIC_POSITIVE
        && detConic > EPS_CONIC_DETERMINANT;
    if (!useConic) { ellipse.useConic = 0u; return false; }

    float traceConic = conic.x + conic.z;
    float discConic = sqrt(max(0.25 * traceConic * traceConic - detConic, 0.0));
    float eigMin = 0.5 * traceConic - discConic;
    float eigMax = 0.5 * traceConic + discConic;
    float maxAxis = rsqrt(max(eigMin, EPS_CONIC_INV_AXIS));
    float minAxis = rsqrt(max(eigMax, EPS_CONIC_INV_AXIS));
    float radiusSafe = max(ellipse.radiusPx, MIN_RADIUS_PX);
    float axisLimit = radiusSafe * 2.0 + halfTile;
    useConic = isfinite(maxAxis)
        && isfinite(minAxis)
        && eigMin > 0.0
        && eigMax > 0.0
        && maxAxis <= axisLimit
        && minAxis >= MIN_AXIS_RATIO;
    if (!useConic) { ellipse.useConic = 0u; return false; }

    float xExtent = sqrt(max(conic.z / detConic, 0.0));
    float yExtent = sqrt(max(conic.x / detConic, 0.0));
    if (!all(isfinite(float2(xExtent, yExtent)))) { ellipse.useConic = 0u; return false; }

    ellipse.bboxExtent = clamp(
        float2(xExtent, yExtent),
        float2(MIN_BBOX_EXTENT_PX),
        float2(ellipse.radiusPx)
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

bool scanline_tile_hits_ellipse(
    Ellipse ellipse,
    ScanlineLoopConfig cfg,
    float tileSize,
    uint lineCoord,
    uint minorCoord
)
{
    float2 tileCoord = cfg.scanAlongX ? float2(float(minorCoord), float(lineCoord)) : float2(float(lineCoord), float(minorCoord));
    float2 tileMin = tileCoord * tileSize;
    float2 boxMin = tileMin - ellipse.centerPx;
    float2 boxMax = (tileMin + tileSize) - ellipse.centerPx;
    return ellipse.intersectsTileBox(boxMin, boxMax);
}

bool scanline_try_hit_minor(
    Ellipse ellipse,
    ScanlineLoopConfig cfg,
    float tileSize,
    uint lineCoord,
    int minor
)
{
    if (minor < 0) return false;
    uint minorU = uint(minor);
    if (!cfg.isMinorCoordValid(minorU)) return false;
    return scanline_tile_hits_ellipse(ellipse, cfg, tileSize, lineCoord, minorU);
}

int scanline_find_first_hit_binary(
    Ellipse ellipse,
    ScanlineLoopConfig cfg,
    float tileSize,
    uint lineCoord,
    int lowFalse,
    int highTrue
)
{
    while (highTrue - lowFalse > 1)
    {
        int mid = lowFalse + ((highTrue - lowFalse) >> 1);
        if (scanline_try_hit_minor(ellipse, cfg, tileSize, lineCoord, mid)) highTrue = mid; else lowFalse = mid;
    }
    return highTrue;
}

int scanline_find_last_hit_binary(
    Ellipse ellipse,
    ScanlineLoopConfig cfg,
    float tileSize,
    uint lineCoord,
    int lowTrue,
    int highFalse
)
{
    while (highFalse - lowTrue > 1)
    {
        int mid = lowTrue + ((highFalse - lowTrue) >> 1);
        if (scanline_try_hit_minor(ellipse, cfg, tileSize, lineCoord, mid)) lowTrue = mid; else highFalse = mid;
    }
    return lowTrue;
}

bool scanline_compute_span_linear_fallback(
    Ellipse ellipse,
    ScanlineLoopConfig cfg,
    float tileSize,
    uint lineCoord,
    out ScanlineSpan span
)
{
    span.lineStartMinor = 0;
    span.lineCount = 0;
    int first = -1;
    int last = -1;
    for (int minor = cfg.minorMin; minor <= cfg.minorMax; minor++)
    {
        if (!scanline_try_hit_minor(ellipse, cfg, tileSize, lineCoord, minor)) continue;
        if (first < 0) first = minor;
        last = minor;
    }
    if (first < 0 || last < first) return false;
    span.lineStartMinor = uint(first);
    span.lineCount = uint(last - first + 1);
    return true;
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
    if (cfg.minorMin > cfg.minorMax) return false;

    int minorMin = cfg.minorMin;
    int minorMax = cfg.minorMax;
    bool leftHit = scanline_try_hit_minor(ellipse, cfg, tileSize, lineCoord, minorMin);
    bool rightHit = scanline_try_hit_minor(ellipse, cfg, tileSize, lineCoord, minorMax);

    if (leftHit && rightHit)
    {
        span.lineStartMinor = uint(minorMin);
        span.lineCount = uint(minorMax - minorMin + 1);
        return true;
    }

    if (leftHit && !rightHit)
    {
        int last = scanline_find_last_hit_binary(ellipse, cfg, tileSize, lineCoord, minorMin, minorMax);
        span.lineStartMinor = uint(minorMin);
        span.lineCount = uint(last - minorMin + 1);
        return true;
    }

    if (!leftHit && rightHit)
    {
        int first = scanline_find_first_hit_binary(ellipse, cfg, tileSize, lineCoord, minorMin, minorMax);
        span.lineStartMinor = uint(first);
        span.lineCount = uint(minorMax - first + 1);
        return true;
    }

    float centerPxMinor = cfg.scanAlongX ? ellipse.centerPx.x : ellipse.centerPx.y;
    int centerMinor = clamp(int(floor(centerPxMinor / tileSize)), minorMin, minorMax);
    if (scanline_try_hit_minor(ellipse, cfg, tileSize, lineCoord, centerMinor))
    {
        int first = scanline_find_first_hit_binary(ellipse, cfg, tileSize, lineCoord, minorMin, centerMinor);
        int last = scanline_find_last_hit_binary(ellipse, cfg, tileSize, lineCoord, centerMinor, minorMax);
        span.lineStartMinor = uint(first);
        span.lineCount = uint(last - first + 1);
        return true;
    }

    int hitMinor = -1;
    int step = 1;
    int spanSize = minorMax - minorMin + 1;
    while (step < spanSize)
    {
        int leftProbe = centerMinor - step;
        if (leftProbe >= minorMin && scanline_try_hit_minor(ellipse, cfg, tileSize, lineCoord, leftProbe)) { hitMinor = leftProbe; break; }
        int rightProbe = centerMinor + step;
        if (rightProbe <= minorMax && scanline_try_hit_minor(ellipse, cfg, tileSize, lineCoord, rightProbe)) { hitMinor = rightProbe; break; }
        step <<= 1;
    }

    if (hitMinor >= 0)
    {
        int first = scanline_find_first_hit_binary(ellipse, cfg, tileSize, lineCoord, minorMin, hitMinor);
        int last = scanline_find_last_hit_binary(ellipse, cfg, tileSize, lineCoord, hitMinor, minorMax);
        span.lineStartMinor = uint(first);
        span.lineCount = uint(last - first + 1);
        return true;
    }

    return scanline_compute_span_linear_fallback(ellipse, cfg, tileSize, lineCoord, span);
}
