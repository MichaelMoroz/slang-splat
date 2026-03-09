import "utility/math/constants.slang";
import "utility/math/matrix.slang";
import "utility/splatting/common.slang";

float screen_radius_cap(float2 viewport)
{
    return max(max(viewport.x, viewport.y), MIN_SCREEN_RADIUS_PX);
}

bool generate_silhouette_points_local5(float3 roLocal, float eps, out float3 points[MVEE_POINT_COUNT])
{
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) points[i] = 0.0;

    float d = length(roLocal);
    if (d <= 1.0 + eps) return false;

    float3 n = roLocal / d;
    float3 circleCenter = n / d;
    float circleRadius = sqrt(max(1.0 - 1.0 / (d * d), 0.0));
    float3 u;
    float3 v;
    if (n.z < SOUTH_POLE_DOT_FLOOR)
    {
        u = float3(0.0, -1.0, 0.0);
        v = float3(-1.0, 0.0, 0.0);
    } else {
        float a = 1.0 / (1.0 + n.z);
        float b = -n.x * n.y * a;
        u = float3(1.0 - n.x * n.x * a, b, -n.x);
        v = float3(b, 1.0 - n.y * n.y * a, -n.y);
    }
    if (!all(isfinite(u)) || !all(isfinite(v))) return false;

    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++)
    {
        float theta = TAU_RADIANS * (float(i) / float(MVEE_POINT_COUNT));
        float s, c;
        sincos(theta, s, c);
        points[i] = circleCenter + circleRadius * (c * u + s * v);
    }
    return true;
}

bool fit_mvee_5pt_fixed_iters(float2 points[MVEE_POINT_COUNT], uint mveeIters, float eps, out MveeFitResult fitOut)
{
    fitOut.clear();
    float2 pointsNorm[MVEE_POINT_COUNT];
    float2 normCenter = 0.0;
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) normCenter += points[i];
    normCenter *= (1.0 / float(MVEE_POINT_COUNT));

    float normScale = 0.0;
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) normScale = max(normScale, length(points[i] - normCenter));
    if (normScale <= eps) return false;
    float invNormScale = 1.0 / normScale;
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) pointsNorm[i] = (points[i] - normCenter) * invNormScale;

    float weights[MVEE_POINT_COUNT];
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) weights[i] = 1.0 / float(MVEE_POINT_COUNT);

    [unroll] for (uint iter = 0; iter < MVEE_MAX_ITERS; iter++)
    {
        if (iter >= mveeIters) break;

        float3x3 xMat = 0.0;
        [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++)
        {
            float3 q = float3(pointsNorm[i], 1.0);
            xMat += weights[i] * float3x3(
                q.x * q.x, q.x * q.y, q.x,
                q.y * q.x, q.y * q.y, q.y,
                q.x, q.y, 1.0
            );
        }

        float3x3 xInv = 0.0;
        if (!inverse3x3(xMat, eps, xInv)) return false;

        uint maxIdx = 0;
        float maxVal = -1.0;
        [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++)
        {
            float3 q = float3(pointsNorm[i], 1.0);
            float3 xq = mul(xInv, q);
            float m = dot(q, xq);
            if (m > maxVal)
            {
                maxVal = m;
                maxIdx = i;
            }
        }

        float denom = MVEE_AFFINE_DIMENSION * (maxVal - 1.0);
        if (denom <= eps) break;
        float alpha = saturate((maxVal - MVEE_AFFINE_DIMENSION) / denom);
        [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) weights[i] *= (1.0 - alpha);
        weights[maxIdx] += alpha;
    }

    float2 center = 0.0;
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) center += weights[i] * pointsNorm[i];

    float3 secondMoment = 0.0;
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++)
    {
        float2 d = pointsNorm[i] - center;
        secondMoment += weights[i] * float3(d.x * d.x, d.x * d.y, d.y * d.y);
    }
    secondMoment += float3(eps, 0.0, eps);

    float detS = secondMoment.x * secondMoment.z - secondMoment.y * secondMoment.y;
    if (detS <= eps) return false;

    float3 conicNorm = 0.5 * float3(secondMoment.z, -secondMoment.y, secondMoment.x) / detS;
    float trace = conicNorm.x + conicNorm.z;
    float detA = conicNorm.x * conicNorm.z - conicNorm.y * conicNorm.y;
    float disc = sqrt(max(0.25 * trace * trace - detA, 0.0));
    float l0 = max(0.5 * trace + disc, eps);
    float l1 = max(0.5 * trace - disc, eps);
    float axis0 = rsqrt(l0);
    float axis1 = rsqrt(l1);

    float maxQuad = 0.0;
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++)
    {
        float2 d = pointsNorm[i] - center;
        float quad = dot(float3(d.x * d.x, d.x * d.y, d.y * d.y), float3(conicNorm.x, 2.0 * conicNorm.y, conicNorm.z));
        maxQuad = max(maxQuad, quad);
    }
    float inflate = (maxQuad > 1.0) ? sqrt(maxQuad) : 1.0;
    float invInflate2 = 1.0 / max(inflate * inflate, eps);
    fitOut.centerPx = normCenter + center * normScale;
    fitOut.radiusPx = max(axis0, axis1) * inflate * normScale;
    fitOut.conic = conicNorm * (invInflate2 / max(normScale * normScale, eps));
    return all(isfinite(fitOut.centerPx)) && isfinite(fitOut.radiusPx) && fitOut.radiusPx > 0.0;
}

bool try_prepare_conic_for_binning(inout Ellipse ellipse, float halfTile)
{
    ellipse.bboxExtent = float2(ellipse.radiusPx, ellipse.radiusPx);
    float3 conic = ellipse.conic;
    float detConic = conic.x * conic.z - conic.y * conic.y;
    bool useConic = all(isfinite(conic))
        && conic.x > POSITIVE_CONIC_FLOOR
        && conic.z > POSITIVE_CONIC_FLOOR
        && detConic > CONIC_DETERMINANT_FLOOR;
    if (!useConic) return false;

    float traceConic = conic.x + conic.z;
    float discConic = sqrt(max(0.25 * traceConic * traceConic - detConic, 0.0));
    float eigMin = 0.5 * traceConic - discConic;
    float eigMax = 0.5 * traceConic + discConic;
    float maxAxis = rsqrt(max(eigMin, CONIC_AXIS_FLOOR));
    float minAxis = rsqrt(max(eigMax, CONIC_AXIS_FLOOR));
    float radiusSafe = max(ellipse.radiusPx, MIN_SCREEN_RADIUS_PX);
    float axisLimit = radiusSafe * 2.0 + halfTile;
    useConic = isfinite(maxAxis)
        && isfinite(minAxis)
        && eigMin > 0.0
        && eigMax > 0.0
        && maxAxis <= axisLimit;
    if (!useConic) return false;

    float xExtent = sqrt(max(conic.z / detConic, 0.0));
    float yExtent = sqrt(max(conic.x / detConic, 0.0));
    if (!all(isfinite(float2(xExtent, yExtent)))) return false;

    ellipse.bboxExtent = clamp(float2(xExtent, yExtent), float2(MIN_BBOX_EXTENT_PX), float2(ellipse.radiusPx));
    return true;
}

ScanlineLoopConfig make_scanline_loop_config(PrepassParams prepass, bool scanAlongX, int minX, int maxX, int minY, int maxY)
{
    ScanlineLoopConfig cfg;
    cfg.scanAlongX = scanAlongX;
    cfg.lineStart = cfg.scanAlongX ? minY : minX;
    cfg.lineEnd = cfg.scanAlongX ? maxY : maxX;
    cfg.minorMin = cfg.scanAlongX ? minX : minY;
    cfg.minorMax = cfg.scanAlongX ? maxX : maxY;
    cfg.minorLimit = cfg.scanAlongX ? prepass.tileWidth : prepass.tileHeight;
    cfg.lineLimit = cfg.scanAlongX ? prepass.tileHeight : prepass.tileWidth;
    return cfg;
}

ScanlineLoopConfig build_scanline_loop_config(PrepassParams prepass, Ellipse ellipse, int minX, int maxX, int minY, int maxY)
{
    uint spanX = uint(maxX - minX + 1);
    uint spanY = uint(maxY - minY + 1);
    bool scanAlongX = ellipse.chooseScanAlongX(spanX, spanY);
    return make_scanline_loop_config(prepass, scanAlongX, minX, maxX, minY, maxY);
}

void accumulate_conic_cross_section(float conicA, float conicB, float det, float invA, float centerMinor, float dM, inout float minPx, inout float maxPx)
{
    float disc = conicA - det * dM * dM;
    if (disc <= 0) return;
    float half = sqrt(disc) * invA;
    float shift = -conicB * dM * invA;
    minPx = min(minPx, centerMinor + shift - half);
    maxPx = max(maxPx, centerMinor + shift + half);
}

bool compute_scanline_tile_span_universal(Ellipse ellipse, ScanlineLoopConfig cfg, float tileSize, uint lineCoord, out ScanlineSpan span)
{
    span.lineStartMinor = 0;
    span.lineCount = 0;
    if (!cfg.isLineCoordValid(lineCoord) || cfg.minorMin > cfg.minorMax) return false;

    float centerMinor = cfg.scanAlongX ? ellipse.centerPx.x : ellipse.centerPx.y;
    float centerMajor = cfg.scanAlongX ? ellipse.centerPx.y : ellipse.centerPx.x;
    float dM0 = float(lineCoord) * tileSize - centerMajor;
    float dM1 = dM0 + tileSize;
    float minPx = 1e30;
    float maxPx = -1e30;

    float A = cfg.scanAlongX ? ellipse.conic.x : ellipse.conic.z;
    float B = ellipse.conic.y;
    float C = cfg.scanAlongX ? ellipse.conic.z : ellipse.conic.x;
    float det = A * C - B * B;
    float invA = 1.0 / A;

    accumulate_conic_cross_section(A, B, det, invA, centerMinor, dM0, minPx, maxPx);
    accumulate_conic_cross_section(A, B, det, invA, centerMinor, dM1, minPx, maxPx);

    float detC = det * C;
    if (detC > 0)
    {
        float dMCrit = B / sqrt(detC);
        if (dMCrit > dM0 && dMCrit < dM1) accumulate_conic_cross_section(A, B, det, invA, centerMinor, dMCrit, minPx, maxPx);
        if (-dMCrit > dM0 && -dMCrit < dM1) accumulate_conic_cross_section(A, B, det, invA, centerMinor, -dMCrit, minPx, maxPx);
    }

    if (minPx > maxPx) return false;

    int first = max(int(floor(minPx / tileSize)), cfg.minorMin);
    int last = min(int(floor(maxPx / tileSize)), cfg.minorMax);
    if (first > last) return false;
    span.lineStartMinor = uint(first);
    span.lineCount = uint(last - first + 1);
    return true;
}
