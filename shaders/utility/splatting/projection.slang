import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/splatting/common.slang";

float screen_radius_cap(float2 viewport)
{
    return max(max(viewport.x, viewport.y), MIN_SCREEN_RADIUS_PX);
}

void init_fullscreen_fallback_ellipse(float2 viewport, out Ellipse ellipse)
{
    float radius = screen_radius_cap(viewport);
    float invRadiusSq = 1.0 / max(radius * radius, ELLIPSE_EPS);
    ellipse.centerPx = 0.5 * viewport;
    ellipse.radiusPx = radius;
    ellipse.conic = float3(invRadiusSq, 0.0, invRadiusSq);
    ellipse.bboxExtent = float2(radius, radius);
}

bool fit_projected_outline_ellipse(float2 outlinePoints[ELLIPSE_POINT_COUNT], float2 outlineMin, float2 outlineMax, bool hasScreenCenter, float2 screenCenter, float2 viewport, out Ellipse ellipse)
{
    ellipse.centerPx = 0.0;
    ellipse.radiusPx = 0.0;
    ellipse.conic = 0.0;
    ellipse.bboxExtent = 0.0;

    if (hasScreenCenter)
    {
        outlineMin = min(outlineMin, screenCenter);
        outlineMax = max(outlineMax, screenCenter);
    }
    if (outlineMax.x < 0.0 || outlineMin.x >= viewport.x || outlineMax.y < 0.0 || outlineMin.y >= viewport.y) return false;

    float2 bboxCenter = 0.5 * (outlineMin + outlineMax);
    float2 bboxHalfExtent = max(0.5 * (outlineMax - outlineMin), float2(ELLIPSE_EPS));
    float2 normPoints[ELLIPSE_POINT_COUNT];
    [unroll] for (uint i = 0u; i < ELLIPSE_POINT_COUNT; i++) normPoints[i] = (outlinePoints[i] - bboxCenter) / bboxHalfExtent;

    float solution[5];
    if (!solve_conic_renorm(normPoints, ELLIPSE_EPS, solution)) return false;

    float3 conicNorm = float3(solution[0], solution[1], solution[2]);
    float2 linear = float2(solution[3], solution[4]);
    float detA = conicNorm.x * conicNorm.z - conicNorm.y * conicNorm.y;
    if (detA <= ELLIPSE_EPS) return false;

    float2 ellipseCenter = 0.5 * float2(
        (conicNorm.y * linear.y - conicNorm.z * linear.x) / detA,
        (conicNorm.y * linear.x - conicNorm.x * linear.y) / detA
    );
    float2 centerTimesA = float2(
        conicNorm.x * ellipseCenter.x + conicNorm.y * ellipseCenter.y,
        conicNorm.y * ellipseCenter.x + conicNorm.z * ellipseCenter.y
    );
    float centerScale = 1.0 + dot(ellipseCenter, centerTimesA);
    if (centerScale <= ELLIPSE_EPS) return false;
    conicNorm /= centerScale;

    float trace = conicNorm.x + conicNorm.z;
    detA = conicNorm.x * conicNorm.z - conicNorm.y * conicNorm.y;
    if (detA <= ELLIPSE_EPS || conicNorm.x <= ELLIPSE_EPS || conicNorm.z <= ELLIPSE_EPS) return false;
    float disc = sqrt(max(0.25 * trace * trace - detA, 0.0));
    float axis0 = rsqrt(max(0.5 * trace + disc, ELLIPSE_EPS));
    float axis1 = rsqrt(max(0.5 * trace - disc, ELLIPSE_EPS));

    ellipse.centerPx = bboxCenter + ellipseCenter * bboxHalfExtent;
    ellipse.radiusPx = max(axis0 * bboxHalfExtent.x, axis1 * bboxHalfExtent.y);
    ellipse.conic = float3(
        conicNorm.x / max(bboxHalfExtent.x * bboxHalfExtent.x, ELLIPSE_EPS),
        conicNorm.y / max(bboxHalfExtent.x * bboxHalfExtent.y, ELLIPSE_EPS),
        conicNorm.z / max(bboxHalfExtent.y * bboxHalfExtent.y, ELLIPSE_EPS)
    );
    ellipse.bboxExtent = bboxHalfExtent;
    return all(isfinite(ellipse.centerPx)) && isfinite(ellipse.radiusPx) && all(isfinite(ellipse.conic)) && ellipse.radiusPx > 0.0;
}

// 5-point conic fit via renormalization: maps two points to canonical positions,
// reducing the 5x5 system to 3x3. Returns coefficients for Ax^2+2Bxy+Cy^2+Dx+Ey = 1.
bool solve_conic_renorm(float2 points[ELLIPSE_POINT_COUNT], float eps, out float solution[5])
{
    [unroll] for (uint i = 0u; i < 5u; i++) solution[i] = 0.0;

    float sx = points[0].x - points[1].x;
    float sy = points[1].y - points[0].y;
    if (abs(sx) <= eps || abs(sy) <= eps) return false;

    float invSx = 1.0 / sx;
    float invSy = 1.0 / sy;
    float offsetX = points[1].x;
    float offsetY = points[0].y;

    float u2 = (points[2].x - offsetX) * invSx;
    float v2 = (points[2].y - offsetY) * invSy;
    float u3 = (points[3].x - offsetX) * invSx;
    float v3 = (points[3].y - offsetY) * invSy;
    float u4 = (points[4].x - offsetX) * invSx;
    float v4 = (points[4].y - offsetY) * invSy;

    float m00 = u2 * u2 - u2;
    float m01 = 2.0 * u2 * v2;
    float m02 = v2 * v2 - v2;
    float r0 = u2 + v2 - 1.0;
    float m10 = u3 * u3 - u3;
    float m11 = 2.0 * u3 * v3;
    float m12 = v3 * v3 - v3;
    float r1 = u3 + v3 - 1.0;
    float m20 = u4 * u4 - u4;
    float m21 = 2.0 * u4 * v4;
    float m22 = v4 * v4 - v4;
    float r2 = u4 + v4 - 1.0;

    // Forward elimination (no pivoting)
    if (abs(m00) <= eps) return false;
    float invM00 = 1.0 / m00;
    m01 *= invM00;
    m02 *= invM00;
    r0 *= invM00;
    float factor10 = m10;
    m11 -= factor10 * m01;
    m12 -= factor10 * m02;
    r1 -= factor10 * r0;
    float factor20 = m20;
    m21 -= factor20 * m01;
    m22 -= factor20 * m02;
    r2 -= factor20 * r0;
    if (abs(m11) <= eps) return false;
    float invM11 = 1.0 / m11;
    m12 *= invM11;
    r1 *= invM11;
    float factor21 = m21;
    m22 -= factor21 * m12;
    r2 -= factor21 * r1;
    if (abs(m22) <= eps) return false;
    r2 /= m22;

    // Back-substitution in renormalized space
    float conicC = r2;
    float conicB = r1 - m12 * conicC;
    float conicA = r0 - m01 * conicB - m02 * conicC;
    float conicD = -(conicA + 1.0);
    float conicE = -(conicC + 1.0);

    // Back-convert to original coordinate space
    float invSx2 = invSx * invSx;
    float invSy2 = invSy * invSy;
    float invSxSy = invSx * invSy;
    float coeffA = conicA * invSx2;
    float coeffB = conicB * invSxSy;
    float coeffC = conicC * invSy2;
    float coeffD = -2.0 * coeffA * offsetX - 2.0 * coeffB * offsetY + conicD * invSx;
    float coeffE = -2.0 * coeffB * offsetX - 2.0 * coeffC * offsetY + conicE * invSy;
    float coeffF = coeffA * offsetX * offsetX + 2.0 * coeffB * offsetX * offsetY + coeffC * offsetY * offsetY - conicD * invSx * offsetX - conicE * invSy * offsetY + 1.0;
    if (abs(coeffF) <= eps) return false;

    // Negate F: renorm internally solves RHS=-1, production convention is RHS=+1
    float invF = -1.0 / coeffF;
    solution[0] = coeffA * invF;
    solution[1] = coeffB * invF;
    solution[2] = coeffC * invF;
    solution[3] = coeffD * invF;
    solution[4] = coeffE * invF;
    return true;
}

bool compute_ellipsoid_outline_ellipse(Ellipsoid ellipsoid, PinholeCamera camera, out Ellipse ellipse)
{
    ellipse.centerPx = 0.0;
    ellipse.radiusPx = 0.0;
    ellipse.conic = 0.0;
    ellipse.bboxExtent = 0.0;

    float supportRadius = max(ellipsoid.projectedScale.x, max(ellipsoid.projectedScale.y, ellipsoid.projectedScale.z));
    float3 cameraCenter = camera.world_point_to_camera(ellipsoid.centerWorld);
    bool supportIntersectsFront = cameraCenter.z + supportRadius > DEPTH_VISIBILITY_FLOOR;

    float2 screenCenter = 0.0;
    bool hasScreenCenter = camera.project_world_to_screen(ellipsoid.centerWorld, screenCenter);
    if (!hasScreenCenter && !supportIntersectsFront) return false;

    // Build the tangent circle on the unit ellipsoid in support-local space.
    float3 viewOriginLocal = ellipsoid.projectedRoLocal;
    float viewDistance = length(viewOriginLocal);
    if (viewDistance <= 1.0 + ELLIPSE_EPS)
    {
        if (!supportIntersectsFront) return false;
        init_fullscreen_fallback_ellipse(camera.params.viewport, ellipse);
        return true;
    }

    float3 viewDirLocal = viewOriginLocal / viewDistance;
    float3 tangentCircleCenter = viewDirLocal / viewDistance;
    float tangentCircleRadius = sqrt(max(1.0 - 1.0 / (viewDistance * viewDistance), 0.0));
    float3 tangentAxis = abs(viewDirLocal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    float3 tangentBasisU = cross(tangentAxis, viewDirLocal);
    float tangentBasisULength = length(tangentBasisU);
    if (tangentBasisULength <= ELLIPSE_EPS || !isfinite(tangentBasisULength)) return false;
    tangentBasisU /= tangentBasisULength;
    float3 tangentBasisV = cross(viewDirLocal, tangentBasisU);
    float tangentBasisVLength = length(tangentBasisV);
    if (tangentBasisVLength <= ELLIPSE_EPS || !isfinite(tangentBasisVLength)) return false;
    tangentBasisV /= tangentBasisVLength;

    float2 outlinePoints[ELLIPSE_POINT_COUNT];
    float2 outlineMin = float2(1e30, 1e30);
    float2 outlineMax = float2(-1e30, -1e30);
    [unroll] for (uint i = 0u; i < ELLIPSE_POINT_COUNT; i++)
    {
        float theta = TAU_RADIANS * (float(i) / float(ELLIPSE_POINT_COUNT));
        float s, c;
        sincos(theta, s, c);
        float3 localPoint = tangentCircleCenter + tangentCircleRadius * (c * tangentBasisU + s * tangentBasisV);
        float3 worldPoint = local_point_to_world(localPoint, ellipsoid.centerWorld, ellipsoid.projectedInvScale, ellipsoid.rotation);
        float2 screenPoint = 0.0;
        if (!camera.project_world_to_screen(worldPoint, screenPoint))
        {
            if (!supportIntersectsFront) return false;
            init_fullscreen_fallback_ellipse(camera.params.viewport, ellipse);
            return true;
        }
        outlinePoints[i] = screenPoint;
        outlineMin = min(outlineMin, screenPoint);
        outlineMax = max(outlineMax, screenPoint);
    }
    return fit_projected_outline_ellipse(outlinePoints, outlineMin, outlineMax, hasScreenCenter, screenCenter, camera.params.viewport, ellipse);
}

bool compute_directional_outline_ellipse(float3 centerDir, float3 sigmaOrtho, float centerDistance, float opacity, float alphaCutoff, PinholeCamera camera, bool supportIntersectsFront, out Ellipse ellipse)
{
    ellipse.centerPx = 0.0;
    ellipse.radiusPx = 0.0;
    ellipse.conic = 0.0;
    ellipse.bboxExtent = 0.0;

    if (!all(isfinite(centerDir)) || !all(isfinite(sigmaOrtho)) || !isfinite(centerDistance) || !isfinite(opacity) || centerDistance <= SMALL_VALUE) return false;
    if (opacity < alphaCutoff) return false;

    float cutoffQuad = -2.0 * log(alphaCutoff / max(opacity, alphaCutoff));
    if (!(cutoffQuad > ELLIPSE_EPS) || !isfinite(cutoffQuad)) return false;

    float detFloor = RAY_DENOMINATOR_FLOOR * RAY_DENOMINATOR_FLOOR;
    float det = sigmaOrtho.x * sigmaOrtho.z - sigmaOrtho.y * sigmaOrtho.y;
    if (det <= detFloor || !isfinite(det)) return false;

    float2 screenCenter = 0.0;
    bool hasScreenCenter = camera.project_camera_to_screen(centerDir * centerDistance, screenCenter);
    if (!hasScreenCenter && !supportIntersectsFront) return false;

    float l00 = sqrt(max(sigmaOrtho.x, 0.0));
    if (!(l00 > RAY_DENOMINATOR_FLOOR) || !isfinite(l00)) return false;
    float l10 = sigmaOrtho.y / l00;
    float l11Sq = sigmaOrtho.z - l10 * l10;
    float l11 = sqrt(max(l11Sq, 0.0));
    if (!(l11 > RAY_DENOMINATOR_FLOOR) || !isfinite(l11)) return false;

    float cutoffRadius = sqrt(cutoffQuad);
    float3 tangentX;
    float3 tangentY;
    build_direction_frame(centerDir, tangentX, tangentY);

    float2 outlinePoints[ELLIPSE_POINT_COUNT];
    float2 outlineMin = float2(1e30, 1e30);
    float2 outlineMax = float2(-1e30, -1e30);
    [unroll] for (uint i = 0u; i < ELLIPSE_POINT_COUNT; i++)
    {
        float theta = TAU_RADIANS * (float(i) / float(ELLIPSE_POINT_COUNT));
        float s, c;
        sincos(theta, s, c);
        float2 eta = cutoffRadius * float2(l00 * c, l10 * c + l11 * s);
        float3 rayDirCamera = normalize(centerDir + tangentX * eta.x + tangentY * eta.y);
        float2 screenPoint = 0.0;
        if (!camera.project_camera_to_screen(rayDirCamera, screenPoint))
        {
            if (!supportIntersectsFront) return false;
            init_fullscreen_fallback_ellipse(camera.params.viewport, ellipse);
            return true;
        }
        outlinePoints[i] = screenPoint;
        outlineMin = min(outlineMin, screenPoint);
        outlineMax = max(outlineMax, screenPoint);
    }
    return fit_projected_outline_ellipse(outlinePoints, outlineMin, outlineMax, hasScreenCenter, screenCenter, camera.params.viewport, ellipse);
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
    float level = 1.0 + CONIC_INTERSECTION_SLACK;
    float xExtent = sqrt(max(level * conic.z / detConic, 0.0));
    float yExtent = sqrt(max(level * conic.x / detConic, 0.0));
    if (!all(isfinite(float2(xExtent, yExtent)))) return false;

    ellipse.bboxExtent = max(float2(xExtent, yExtent) + BBOX_PAD_PX, float2(MIN_BBOX_EXTENT_PX));
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

void tile_box_relative_to_ellipse(float2 centerPx, bool scanAlongX, float tileSize, uint lineCoord, uint minorCoord, out float2 boxMin, out float2 boxMax)
{
    float2 major = scanAlongX ? float2(minorCoord, lineCoord) : float2(lineCoord, minorCoord);
    boxMin = major * tileSize - centerPx;
    boxMax = (major + 1.0) * tileSize - centerPx;
}

bool tile_intersects_ellipse(Ellipse ellipse, bool scanAlongX, float tileSize, uint lineCoord, uint minorCoord)
{
    float2 boxMin;
    float2 boxMax;
    tile_box_relative_to_ellipse(ellipse.centerPx, scanAlongX, tileSize, lineCoord, minorCoord, boxMin, boxMax);
    return ellipse.intersectsTileBox(boxMin, boxMax);
}

bool compute_scanline_tile_span_universal(Ellipse ellipse, ScanlineLoopConfig cfg, float tileSize, uint lineCoord, out ScanlineSpan span)
{
    span.lineStartMinor = 0;
    span.lineCount = 0;
    if (!cfg.isLineCoordValid(lineCoord) || cfg.minorMin > cfg.minorMax) return false;

    bool found = false;
    uint firstHit = 0u;
    uint lastHit = 0u;
    uint minorStart = uint(max(cfg.minorMin, 0));
    uint minorEnd = uint(cfg.minorMax);
    for (uint minorCoord = minorStart; minorCoord <= minorEnd; minorCoord++)
    {
        if (!tile_intersects_ellipse(ellipse, cfg.scanAlongX, tileSize, lineCoord, minorCoord)) continue;
        if (!found)
        {
            found = true;
            firstHit = minorCoord;
        }
        lastHit = minorCoord;
    }
    if (!found) return false;

    span.lineStartMinor = firstHit;
    span.lineCount = lastHit - firstHit + 1u;
    return true;
}
