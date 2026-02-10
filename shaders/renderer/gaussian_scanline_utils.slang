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

void accumulate_conic_cross_section(
    float conicA, float conicB, float det, float invA,
    float centerMinor, float dM,
    inout float minPx, inout float maxPx
)
{
    float disc = conicA - det * dM * dM;
    if (disc <= 0) return;
    float half = sqrt(disc) * invA;
    float shift = -conicB * dM * invA;
    minPx = min(minPx, centerMinor + shift - half);
    maxPx = max(maxPx, centerMinor + shift + half);
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
    if (!cfg.isLineCoordValid(lineCoord) || cfg.minorMin > cfg.minorMax) return false;

    float centerMinor = cfg.scanAlongX ? ellipse.centerPx.x : ellipse.centerPx.y;
    float centerMajor = cfg.scanAlongX ? ellipse.centerPx.y : ellipse.centerPx.x;
    float dM0 = float(lineCoord) * tileSize - centerMajor;
    float dM1 = dM0 + tileSize;

    // Analytically compute the minor-axis pixel range of the ellipse over the major-axis band
    float minPx = 1e30;
    float maxPx = -1e30;

    if (ellipse.useConic != 0u)
    {
        // Conic: A*dx² + 2B*dx*dy + C*dy² = 1, solve quadratic at each band edge
        float A = cfg.scanAlongX ? ellipse.conic.x : ellipse.conic.z;
        float B = ellipse.conic.y;
        float C = cfg.scanAlongX ? ellipse.conic.z : ellipse.conic.x;
        float det = A * C - B * B;
        float invA = 1.0 / A;

        accumulate_conic_cross_section(A, B, det, invA, centerMinor, dM0, minPx, maxPx);
        accumulate_conic_cross_section(A, B, det, invA, centerMinor, dM1, minPx, maxPx);

        // Extremal dM where the boundary reaches bboxExtent: ±B/sqrt(det*C)
        float detC = det * C;
        if (detC > 0)
        {
            float dMCrit = B / sqrt(detC);
            if (dMCrit > dM0 && dMCrit < dM1)
                accumulate_conic_cross_section(A, B, det, invA, centerMinor, dMCrit, minPx, maxPx);
            if (-dMCrit > dM0 && -dMCrit < dM1)
                accumulate_conic_cross_section(A, B, det, invA, centerMinor, -dMCrit, minPx, maxPx);
        }
    } else {
        // Circle fallback: half-width = sqrt(r² - dM²)
        float rr = ellipse.radiusPx * ellipse.radiusPx;
        float disc0 = rr - dM0 * dM0;
        float disc1 = rr - dM1 * dM1;
        if (disc0 > 0) { float h = sqrt(disc0); minPx = min(minPx, centerMinor - h); maxPx = max(maxPx, centerMinor + h); }
        if (disc1 > 0) { float h = sqrt(disc1); minPx = min(minPx, centerMinor - h); maxPx = max(maxPx, centerMinor + h); }
        if (dM0 <= 0 && dM1 >= 0) { minPx = min(minPx, centerMinor - ellipse.radiusPx); maxPx = max(maxPx, centerMinor + ellipse.radiusPx); }
    }

    if (minPx > maxPx) return false;

    int first = max(int(floor(minPx / tileSize)), cfg.minorMin);
    int last = min(int(floor(maxPx / tileSize)), cfg.minorMax);
    if (first > last) return false;
    span.lineStartMinor = uint(first);
    span.lineCount = uint(last - first + 1);
    return true;
}
