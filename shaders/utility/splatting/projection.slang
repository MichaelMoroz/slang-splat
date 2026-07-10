import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/splatting/common.slang";

static const uint EQUIRECTANGULAR_ELLIPSE_POINT_COUNT = 8u;

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

float support_plane_slack(float3 planeNormal, float supportRadius)
{
    return supportRadius * length(planeNormal);
}

bool support_sphere_intersects_view_frustum(PinholeCamera camera, float3 cameraCenter, float supportRadius)
{
    if (!(supportRadius > 0.0) || !all(isfinite(cameraCenter)) || !isfinite(supportRadius)) return false;
    if (camera.is_equirectangular()) return length(cameraCenter) + supportRadius > DEPTH_VISIBILITY_FLOOR;
    if (cameraCenter.z + supportRadius <= DEPTH_VISIBILITY_FLOOR) return false;

    float3 leftPlane = float3(camera.params.focalPixels.x, 0.0, camera.params.principalPoint.x);
    float3 rightPlane = float3(-camera.params.focalPixels.x, 0.0, camera.params.viewport.x - camera.params.principalPoint.x);
    float3 topPlane = float3(0.0, camera.params.focalPixels.y, camera.params.principalPoint.y);
    float3 bottomPlane = float3(0.0, -camera.params.focalPixels.y, camera.params.viewport.y - camera.params.principalPoint.y);
    if (dot(leftPlane, cameraCenter) < -support_plane_slack(leftPlane, supportRadius)) return false;
    if (dot(rightPlane, cameraCenter) < -support_plane_slack(rightPlane, supportRadius)) return false;
    if (dot(topPlane, cameraCenter) < -support_plane_slack(topPlane, supportRadius)) return false;
    if (dot(bottomPlane, cameraCenter) < -support_plane_slack(bottomPlane, supportRadius)) return false;
    return true;
}

bool equirectangular_outline_needs_fallback(PinholeCamera camera, float2 outlineMin, float2 outlineMax)
{
    if (!camera.is_equirectangular()) return false;
    // The seam-wrap detection compares against the longitude period (native full-sphere
    // width), not the viewport: a crop camera's viewport is much smaller than the period.
    float period = camera.equirect_period();
    float yTop = camera.equirect_principal_point().y - 0.5 * PI * camera.equirect_focal().y;
    float yBottom = camera.equirect_principal_point().y + 0.5 * PI * camera.equirect_focal().y;
    return (outlineMax.x - outlineMin.x) > 0.5 * period || outlineMin.y <= yTop || outlineMax.y >= yBottom;
}

bool equirectangular_support_needs_fallback(PinholeCamera camera, float3 cameraCenter, float supportRadius)
{
    if (!camera.is_equirectangular()) return false;
    float distance = length(cameraCenter);
    if (!(supportRadius > 0.0) || !all(isfinite(cameraCenter)) || !isfinite(supportRadius)) return false;
    if (distance <= DEPTH_VISIBILITY_FLOOR || supportRadius >= distance) return true;

    float horizontal = length(cameraCenter.xz);
    float theta = atan2(cameraCenter.x, cameraCenter.z);
    float phi = atan2(-cameraCenter.y, horizontal);
    float angularRadius = asin(saturate(supportRadius / distance));
    float angularSlack = TINY_VALUE;
    return abs(theta) + angularRadius + angularSlack >= PI || abs(phi) + angularRadius + angularSlack >= 0.5 * PI;
}

// Bounded binning rect for the support-sphere direction cap (half-angle asin(r/d)):
// latitude band center.y +- alpha/PI * H; longitude interval from the bounding-meridian
// identity sin(dTheta) = sin(alpha) / cos(phi0) = r / horizontal, or full width when the
// cap contains a pole (horizontal <= r). Encoded as conic = (-xExtent, 0, -yExtent) so
// every no-analytic-conic binning path picks it up without extra storage.
void init_equirectangular_cap_rect_ellipse(PinholeCamera camera, float3 cameraCenter, float supportRadius, float2 screenCenter, out Ellipse ellipse)
{
    float2 viewport = max(camera.params.viewport, float2(1.0));
    float distance = length(cameraCenter);
    if (!(distance > DEPTH_VISIBILITY_FLOOR) || !(supportRadius > 0.0) || supportRadius >= distance)
    {
        init_fullscreen_fallback_ellipse(camera.params.viewport, ellipse);
        return;
    }

    float horizontal = length(cameraCenter.xz);
    float angularRadius = asin(saturate(supportRadius / distance));
    // Angular extents convert to pixels through the equirect intrinsics so crop cameras
    // (native focal, small viewport) get native-scale extents rather than viewport-scaled.
    float2 focal = camera.equirect_focal();
    float yExtent = min(angularRadius, 0.5 * PI) * focal.y;
    float xExtent = horizontal <= supportRadius ? camera.equirect_period() : asin(saturate(supportRadius / horizontal)) * focal.x;
    ellipse.centerPx = screenCenter;
    ellipse.radiusPx = max(max(xExtent, yExtent), MIN_SCREEN_RADIUS_PX);
    ellipse.conic = float3(-xExtent, 0.0, -yExtent);
    ellipse.bboxExtent = float2(xExtent, yExtent);
}

bool solve_symmetric_3x3(
    float s00,
    float s01,
    float s02,
    float s11,
    float s12,
    float s22,
    float3 rhs,
    out float3 solution)
{
    solution = 0.0;
    float det = s00 * (s11 * s22 - s12 * s12)
        - s01 * (s01 * s22 - s12 * s02)
        + s02 * (s01 * s12 - s11 * s02);
    if (abs(det) <= ELLIPSE_EPS || !isfinite(det)) return false;

    solution.x = (
        rhs.x * (s11 * s22 - s12 * s12)
        - s01 * (rhs.y * s22 - s12 * rhs.z)
        + s02 * (rhs.y * s12 - s11 * rhs.z)
    ) / det;
    solution.y = (
        s00 * (rhs.y * s22 - s12 * rhs.z)
        - rhs.x * (s01 * s22 - s12 * s02)
        + s02 * (s01 * rhs.z - rhs.y * s02)
    ) / det;
    solution.z = (
        s00 * (s11 * rhs.z - rhs.y * s12)
        - s01 * (s01 * rhs.z - rhs.y * s02)
        + rhs.x * (s01 * s12 - s11 * s02)
    ) / det;
    return all(isfinite(solution));
}

bool fit_centered_projected_outline_ellipse(float2 outlinePoints[EQUIRECTANGULAR_ELLIPSE_POINT_COUNT], float2 screenCenter, out Ellipse ellipse)
{
    ellipse.centerPx = 0.0;
    ellipse.radiusPx = 0.0;
    ellipse.conic = 0.0;
    ellipse.bboxExtent = 0.0;

    float s00 = 0.0;
    float s01 = 0.0;
    float s02 = 0.0;
    float s11 = 0.0;
    float s12 = 0.0;
    float s22 = 0.0;
    float3 rhs = 0.0;
    float2 bboxHalfExtent = float2(ELLIPSE_EPS);
    [unroll] for (uint i = 0u; i < EQUIRECTANGULAR_ELLIPSE_POINT_COUNT; i++)
    {
        float2 d = outlinePoints[i] - screenCenter;
        bboxHalfExtent = max(bboxHalfExtent, abs(d));
    }

    [unroll] for (uint i = 0u; i < EQUIRECTANGULAR_ELLIPSE_POINT_COUNT; i++)
    {
        float2 d = (outlinePoints[i] - screenCenter) / bboxHalfExtent;
        float3 row = float3(d.x * d.x, 2.0 * d.x * d.y, d.y * d.y);
        s00 += row.x * row.x;
        s01 += row.x * row.y;
        s02 += row.x * row.z;
        s11 += row.y * row.y;
        s12 += row.y * row.z;
        s22 += row.z * row.z;
        rhs += row;
    }

    float3 conicNorm;
    if (!solve_symmetric_3x3(s00, s01, s02, s11, s12, s22, rhs, conicNorm)) return false;
    float detNorm = conicNorm.x * conicNorm.z - conicNorm.y * conicNorm.y;
    if (conicNorm.x <= ELLIPSE_EPS || conicNorm.z <= ELLIPSE_EPS || detNorm <= ELLIPSE_EPS) return false;

    // The least-squares fit straddles the sampled outline; scale the conic so every
    // sample lies inside the level-1 set, otherwise the far side of distorted
    // outlines gets tile-clipped during binning.
    float maxLevel = 1.0;
    [unroll] for (uint i = 0u; i < EQUIRECTANGULAR_ELLIPSE_POINT_COUNT; i++)
    {
        float2 d = (outlinePoints[i] - screenCenter) / bboxHalfExtent;
        maxLevel = max(maxLevel, conicNorm.x * d.x * d.x + 2.0 * conicNorm.y * d.x * d.y + conicNorm.z * d.y * d.y);
    }
    conicNorm /= maxLevel;

    float3 conic = float3(
        conicNorm.x / max(bboxHalfExtent.x * bboxHalfExtent.x, ELLIPSE_EPS),
        conicNorm.y / max(bboxHalfExtent.x * bboxHalfExtent.y, ELLIPSE_EPS),
        conicNorm.z / max(bboxHalfExtent.y * bboxHalfExtent.y, ELLIPSE_EPS)
    );
    float detConic = conic.x * conic.z - conic.y * conic.y;
    if (conic.x <= 0.0 || conic.z <= 0.0 || detConic <= 0.0 || !all(isfinite(conic)) || !isfinite(detConic)) return false;
    float trace = conic.x + conic.z;
    float disc = sqrt(max(0.25 * trace * trace - detConic, 0.0));
    float axis0 = rsqrt(max(0.5 * trace + disc, CONIC_AXIS_FLOOR));
    float axis1 = rsqrt(max(0.5 * trace - disc, CONIC_AXIS_FLOOR));
    float radius = max(axis0, axis1);
    ellipse.centerPx = screenCenter;
    ellipse.radiusPx = radius;
    ellipse.conic = conic;
    ellipse.bboxExtent = float2(radius, radius);
    return all(isfinite(ellipse.centerPx)) && isfinite(ellipse.radiusPx) && all(isfinite(ellipse.conic)) && ellipse.radiusPx > 0.0;
}

bool equirectangular_centered_ellipse_needs_fallback(PinholeCamera camera, Ellipse ellipse)
{
    float detConic = ellipse.conic.x * ellipse.conic.z - ellipse.conic.y * ellipse.conic.y;
    if (detConic <= 0.0 || !isfinite(detConic)) return true;
    float2 extent = sqrt(max(float2(ellipse.conic.z, ellipse.conic.x) / detConic, 0.0));
    if (!all(isfinite(extent))) return true;
    float detNorm = detConic * extent.x * extent.x * extent.y * extent.y;
    if (detNorm <= ELLIPSE_EPS) return true;
    float2 centerMin = ellipse.centerPx - extent;
    float2 centerMax = ellipse.centerPx + extent;
    return centerMin.x <= 0.0 || centerMax.x >= camera.params.viewport.x || centerMin.y <= 0.0 || centerMax.y >= camera.params.viewport.y;
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
    bool supportIntersectsFrustum = support_sphere_intersects_view_frustum(camera, cameraCenter, supportRadius);

    float2 screenCenter = 0.0;
    bool hasScreenCenter = camera.project_world_to_screen(ellipsoid.centerWorld, screenCenter);
    if (!hasScreenCenter && !supportIntersectsFrustum) return false;

    // Build the tangent circle on the unit ellipsoid in support-local space.
    float3 viewOriginLocal = ellipsoid.projectedRoLocal;
    float viewDistance = length(viewOriginLocal);
    if (viewDistance <= 1.0 + ELLIPSE_EPS)
    {
        if (!supportIntersectsFrustum) return false;
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

    if (camera.is_equirectangular())
    {
        if (!hasScreenCenter)
        {
            if (!supportIntersectsFrustum) return false;
            init_fullscreen_fallback_ellipse(camera.params.viewport, ellipse);
            return true;
        }
        if (equirectangular_support_needs_fallback(camera, cameraCenter, supportRadius))
        {
            if (!supportIntersectsFrustum) return false;
            init_equirectangular_cap_rect_ellipse(camera, cameraCenter, supportRadius, screenCenter, ellipse);
            return true;
        }
        float2 outlinePoints[EQUIRECTANGULAR_ELLIPSE_POINT_COUNT];
        float2 outlineMin = float2(1e30, 1e30);
        float2 outlineMax = float2(-1e30, -1e30);
        [unroll] for (uint i = 0u; i < EQUIRECTANGULAR_ELLIPSE_POINT_COUNT; i++)
        {
            float theta = TAU_RADIANS * (float(i) / float(EQUIRECTANGULAR_ELLIPSE_POINT_COUNT));
            float s, c;
            sincos(theta, s, c);
            float3 localPoint = tangentCircleCenter + tangentCircleRadius * (c * tangentBasisU + s * tangentBasisV);
            float3 worldPoint = local_point_to_world(localPoint, ellipsoid.centerWorld, ellipsoid.projectedInvScale, ellipsoid.rotation);
            float2 screenPoint = 0.0;
            if (!camera.project_world_to_screen(worldPoint, screenPoint))
            {
                if (!supportIntersectsFrustum) return false;
                init_equirectangular_cap_rect_ellipse(camera, cameraCenter, supportRadius, screenCenter, ellipse);
                return true;
            }
            outlinePoints[i] = screenPoint;
            outlineMin = min(outlineMin, screenPoint);
            outlineMax = max(outlineMax, screenPoint);
        }
        if (equirectangular_outline_needs_fallback(camera, outlineMin, outlineMax)
            || !fit_centered_projected_outline_ellipse(outlinePoints, screenCenter, ellipse)
            || equirectangular_centered_ellipse_needs_fallback(camera, ellipse))
        {
            init_equirectangular_cap_rect_ellipse(camera, cameraCenter, supportRadius, screenCenter, ellipse);
        }
        return true;
    }

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
            if (!supportIntersectsFrustum) return false;
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
    if (!all(isfinite(ellipse.centerPx)) || !isfinite(ellipse.radiusPx) || !(ellipse.radiusPx > 0.0)) return false;
    if (!ellipse.hasAnalyticConic())
    {
        float2 rawExtent = ellipse.isBBoxRect() ? ellipse.rectExtent() : float2(ellipse.radiusPx);
        ellipse.bboxExtent = max(rawExtent + BBOX_PAD_PX, float2(MIN_BBOX_EXTENT_PX));
        return true;
    }
    float3 conic = ellipse.conic;
    float detConic = ellipse.conicDeterminant();
    float level = 1.0 + CONIC_INTERSECTION_SLACK;
    float xExtent = sqrt(max(level * conic.z / detConic, 0.0));
    float yExtent = sqrt(max(level * conic.x / detConic, 0.0));
    if (!all(isfinite(float2(xExtent, yExtent))))
    {
        ellipse.bboxExtent = max(float2(ellipse.radiusPx + BBOX_PAD_PX), float2(MIN_BBOX_EXTENT_PX));
        return true;
    }

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

bool solve_quadratic_interval(float quadraticCoeff, float linearCoeff, float constantCoeff, float level, out float2 interval)
{
    interval = 0.0;
    float disc = linearCoeff * linearCoeff - quadraticCoeff * (constantCoeff - level);
    if (!(disc >= 0.0) || !isfinite(disc)) return false;
    float root = sqrt(max(disc, 0.0));
    interval = float2((-linearCoeff - root) / quadraticCoeff, (-linearCoeff + root) / quadraticCoeff);
    return all(isfinite(interval)) && interval.x <= interval.y;
}

void merge_interval(float2 candidate, inout bool found, inout float2 interval)
{
    if (!found)
    {
        interval = candidate;
        found = true;
        return;
    }
    interval.x = min(interval.x, candidate.x);
    interval.y = max(interval.y, candidate.y);
}

bool analytic_scanline_minor_interval(Ellipse ellipse, ScanlineLoopConfig cfg, float tileSize, uint lineCoord, float centerTightenPx, out float2 interval)
{
    interval = 0.0;
    if (!ellipse.hasAnalyticConic())
    {
        int firstHit = max(cfg.minorMin, 0);
        int lastHit = min(cfg.minorMax, int(cfg.minorLimit) - 1);
        interval = float2(float(firstHit), float(lastHit));
        return firstHit <= lastHit;
    }

    float minorCoeff = cfg.scanAlongX ? ellipse.conic.x : ellipse.conic.z;
    float lineCoeff = cfg.scanAlongX ? ellipse.conic.z : ellipse.conic.x;
    float mixedCoeff = ellipse.conic.y;
    float det = minorCoeff * lineCoeff - mixedCoeff * mixedCoeff;
    if (!(minorCoeff > CONIC_DETERMINANT_FLOOR) || !(lineCoeff > CONIC_DETERMINANT_FLOOR) || !(det > CONIC_DETERMINANT_FLOOR)) return false;

    float centerMinor = cfg.scanAlongX ? ellipse.centerPx.x : ellipse.centerPx.y;
    float centerLine = cfg.scanAlongX ? ellipse.centerPx.y : ellipse.centerPx.x;
    float lineStart = float(lineCoord) * tileSize + centerTightenPx - centerLine;
    float lineEnd = (float(lineCoord) + 1.0) * tileSize - centerTightenPx - centerLine;
    float level = 1.0 + CONIC_INTERSECTION_SLACK;

    bool found = false;
    float2 candidate = 0.0;
    if (solve_quadratic_interval(minorCoeff, mixedCoeff * lineStart, lineCoeff * lineStart * lineStart, level, candidate)) merge_interval(candidate, found, interval);
    if (solve_quadratic_interval(minorCoeff, mixedCoeff * lineEnd, lineCoeff * lineEnd * lineEnd, level, candidate)) merge_interval(candidate, found, interval);

    float interiorCoeff = det / lineCoeff;
    if (interiorCoeff > CONIC_DETERMINANT_FLOOR)
    {
        float interiorRadius = sqrt(level / interiorCoeff);
        float2 interiorInterval = float2(-interiorRadius, interiorRadius);
        if (abs(mixedCoeff) <= CONIC_DETERMINANT_FLOOR)
        {
            if (lineStart <= 0.0 && 0.0 <= lineEnd) merge_interval(interiorInterval, found, interval);
        }
        else
        {
            float stationaryMinorStart = -lineCoeff * lineStart / mixedCoeff;
            float stationaryMinorEnd = -lineCoeff * lineEnd / mixedCoeff;
            float2 stationaryInterval = float2(min(stationaryMinorStart, stationaryMinorEnd), max(stationaryMinorStart, stationaryMinorEnd));
            float2 clippedInterior = float2(max(interiorInterval.x, stationaryInterval.x), min(interiorInterval.y, stationaryInterval.y));
            if (clippedInterior.x <= clippedInterior.y) merge_interval(clippedInterior, found, interval);
        }
    }
    if (!found) return false;

    int firstHit = int(ceil((interval.x + centerMinor + centerTightenPx) / tileSize)) - 1;
    int lastHit = int(floor((interval.y + centerMinor - centerTightenPx) / tileSize));
    firstHit = max(firstHit, max(cfg.minorMin, 0));
    lastHit = min(lastHit, min(cfg.minorMax, int(cfg.minorLimit) - 1));
    interval = float2(float(firstHit), float(lastHit));
    return firstHit <= lastHit;
}

// centerTightenPx shrinks each tile box toward its pixel centers (0.5 px per side) so a
// splat whose exact conic region misses every pixel center of a tile is not binned into
// it. Only valid when the conic is exact - undistorted pinhole - since fitted conics
// (equirect, distorted) need the conservative full tile box; those pass 0.
bool compute_scanline_tile_span_universal(Ellipse ellipse, ScanlineLoopConfig cfg, float tileSize, uint lineCoord, float centerTightenPx, out ScanlineSpan span)
{
    span.lineStartMinor = 0;
    span.lineCount = 0;
    if (!cfg.isLineCoordValid(lineCoord) || cfg.minorMin > cfg.minorMax) return false;

    float2 minorInterval;
    if (!analytic_scanline_minor_interval(ellipse, cfg, tileSize, lineCoord, centerTightenPx, minorInterval)) return false;

    uint firstHit = uint(int(minorInterval.x));
    uint lastHit = uint(int(minorInterval.y));
    span.lineStartMinor = firstHit;
    span.lineCount = lastHit - firstHit + 1u;
    return true;
}
