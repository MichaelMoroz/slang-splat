import "slangpy";

static const uint TILE_SIZE = 8u;
static const uint PROJECT_GROUP_SIZE = 256u;
static const uint RASTER_BATCH = 64u;
static const uint RASTER_THREAD_TILE_DIM = TILE_SIZE;
static const uint RASTER_GROUP_THREADS = RASTER_THREAD_TILE_DIM * RASTER_THREAD_TILE_DIM;
static const uint MVEE_POINT_COUNT = 5u;
static const uint MVEE_MAX_ITERS = 8u;
static const float SMALL_VALUE = 1e-6;
static const float TINY_VALUE = 1e-8;
static const float RAY_DENOMINATOR_FLOOR = 1e-10;
static const float POSITIVE_CONIC_FLOOR = 1e-10;
static const float CONIC_DETERMINANT_FLOOR = 1e-12;
static const float CONIC_AXIS_FLOOR = 1e-20;
static const float CONIC_INTERSECTION_SLACK = 1e-5;
static const float DEPTH_VISIBILITY_FLOOR = 1e-4;
static const float SOUTH_POLE_DOT_FLOOR = -0.9999999;
static const float SPLAT_PIXEL_CLAMP_PX = 0.75;
static const float MIN_SCREEN_RADIUS_PX = 1.0;
static const float MIN_BBOX_EXTENT_PX = 1e-4;
static const float MVEE_AFFINE_DIMENSION = 3.0;
static const float OUTPUT_GAMMA = 2.2;
static const float TAU_RADIANS = 6.28318530718;
static const float GAUSSIAN_SUPPORT_SIGMA_RADIUS = 3.0;
static const uint PARAM_POSITION_X = 0u;
static const uint PARAM_POSITION_Y = 1u;
static const uint PARAM_POSITION_Z = 2u;
static const uint PARAM_SCALE_X = 3u;
static const uint PARAM_SCALE_Y = 4u;
static const uint PARAM_SCALE_Z = 5u;
static const uint PARAM_ROTATION_W = 6u;
static const uint PARAM_ROTATION_X = 7u;
static const uint PARAM_ROTATION_Y = 8u;
static const uint PARAM_ROTATION_Z = 9u;
static const uint PARAM_COLOR_R = 10u;
static const uint PARAM_COLOR_G = 11u;
static const uint PARAM_COLOR_B = 12u;
static const uint PARAM_RAW_OPACITY = 13u;
static const uint PARAM_COUNT = 14u;
static const uint INVALID_START = 0xFFFFFFFFu;

struct PackedSplat
{
    float px;
    float py;
    float pz;
    float sx;
    float sy;
    float sz;
    float qw;
    float qx;
    float qy;
    float qz;
    float r;
    float g;
    float b;
    float a;
};

struct ProjectionState
{
    float4 centerRadiusDepth;
    float4 conicOpacity;
};

struct RasterState
{
    float3 roLocal;
    float3 scale;
    float4 rotation;
    float4 colorAlpha;
};

struct TileEntry
{
    uint2 ids;
};

struct CameraParams
{
    float3 camPos;
    float3x3 camBasis;
    float2 focalPixels;
    float2 principalPoint;
    float2 viewport;
    float nearDepth;
    float farDepth;
    float k1;
    float k2;
};

struct Gaussian3D : IDifferentiable
{
    float3 position;
    float3 logScale;
    float4 rotation;
    float4 colorAlpha;
};

struct CachedEllipsoid : IDifferentiable
{
    float3 roLocal;
    float3 scale;
    float4 rotation;
};

struct Ellipse
{
    float2 centerPx;
    float radiusPx;
    float3 conic;
    float2 bboxExtent;
};

struct MveeFitResult
{
    float2 centerPx;
    float radiusPx;
    float3 conic;
};

bool inverse3x3(float3x3 m, float eps, out float3x3 outInv)
{
    float a = m[0][0], b = m[0][1], c = m[0][2];
    float d = m[1][0], e = m[1][1], f = m[1][2];
    float g = m[2][0], h = m[2][1], i = m[2][2];
    float A = e * i - f * h;
    float B = -(d * i - f * g);
    float C = d * h - e * g;
    float D = -(b * i - c * h);
    float E = a * i - c * g;
    float F = -(a * h - b * g);
    float G = b * f - c * e;
    float H = -(a * f - c * d);
    float I = a * e - b * d;
    float det = a * A + b * B + c * C;
    if (abs(det) <= eps) { outInv = 0.0; return false; }
    float invDet = 1.0 / det;
    outInv = float3x3(A, D, G, B, E, H, C, F, I) * invDet;
    return all(isfinite(outInv[0])) && all(isfinite(outInv[1])) && all(isfinite(outInv[2]));
}

[Differentiable]
float4 normalize_quaternion_safe(float4 q)
{
    float n = length(q);
    return (!isfinite(n) || n < TINY_VALUE) ? float4(1.0, 0.0, 0.0, 0.0) : q / n;
}

[Differentiable]
float3 rotate_by_quat(float3 v, float4 q)
{
    float3 qv = q.yzw;
    return v + 2.0 * cross(cross(v, qv) + q.x * v, qv);
}

[Differentiable]
float opacity_from_raw(float x)
{
    if (x >= 0.0) { float z = exp(-x); return 1.0 / (1.0 + z); }
    float z = exp(x);
    return z / (1.0 + z);
}

Gaussian3D load_gaussian(ITensor<PackedSplat, 1> splats, uint splatId)
{
    int[1] idx = { int(splatId) };
    PackedSplat s = splats.load(idx);
    Gaussian3D g;
    g.position = float3(s.px, s.py, s.pz);
    g.logScale = float3(s.sx, s.sy, s.sz);
    g.rotation = normalize_quaternion_safe(float4(s.qw, s.qx, s.qy, s.qz));
    g.colorAlpha = float4(s.r, s.g, s.b, s.a);
    return g;
}

float2 distort_normalized(CameraParams camera, float2 uv)
{
    float r2 = dot(uv, uv);
    return uv * (1.0 + camera.k1 * r2 + camera.k2 * r2 * r2);
}

float2 undistort_normalized(CameraParams camera, float2 uv)
{
    if (abs(camera.k1) <= SMALL_VALUE && abs(camera.k2) <= SMALL_VALUE) return uv;
    float rd = length(uv);
    if (rd <= SMALL_VALUE) return uv;
    float r = rd;
    [unroll] for (uint i = 0u; i < 6u; i++)
    {
        float r2 = r * r;
        float r4 = r2 * r2;
        float deriv = 1.0 + 3.0 * camera.k1 * r2 + 5.0 * camera.k2 * r4;
        if (abs(deriv) <= SMALL_VALUE) break;
        float nextR = r - (r * (1.0 + camera.k1 * r2 + camera.k2 * r4) - rd) / deriv;
        if (!isfinite(nextR) || nextR < 0.0) break;
        r = nextR;
    }
    return uv * (r / max(rd, SMALL_VALUE));
}

float3 world_to_camera(CameraParams camera, float3 worldVector) { return mul(camera.camBasis, worldVector); }
float3 camera_to_world(CameraParams camera, float3 cameraVector) { return mul(transpose(camera.camBasis), cameraVector); }
float3 world_point_to_camera(CameraParams camera, float3 worldPos) { return world_to_camera(camera, worldPos - camera.camPos); }
float3 screen_to_world_ray(CameraParams camera, float2 screenPos)
{
    float2 focalSafe = max(camera.focalPixels, float2(SMALL_VALUE));
    float2 uv = undistort_normalized(camera, (screenPos - camera.principalPoint) / focalSafe);
    return normalize(camera_to_world(camera, float3(uv, 1.0)));
}

bool project_camera_to_screen(CameraParams camera, float3 cameraPos, out float2 screenPos)
{
    if (cameraPos.z <= SMALL_VALUE) { screenPos = 0.0; return false; }
    float2 uv = distort_normalized(camera, cameraPos.xy / cameraPos.z);
    screenPos = uv * camera.focalPixels + camera.principalPoint;
    return all(isfinite(screenPos));
}

float screen_radius_cap(float2 viewport) { return max(max(viewport.x, viewport.y), MIN_SCREEN_RADIUS_PX); }

bool generate_silhouette_points_local5(float3 roLocal, float eps, out float3 points[MVEE_POINT_COUNT])
{
    [unroll] for (uint i = 0u; i < MVEE_POINT_COUNT; i++) points[i] = 0.0;
    float d = length(roLocal);
    if (d <= 1.0 + eps) return false;
    float3 n = roLocal / d;
    float3 circleCenter = n / d;
    float circleRadius = sqrt(max(1.0 - 1.0 / (d * d), 0.0));
    float3 u;
    float3 v;
    if (n.z < SOUTH_POLE_DOT_FLOOR) { u = float3(0.0, -1.0, 0.0); v = float3(-1.0, 0.0, 0.0); }
    else
    {
        float a = 1.0 / (1.0 + n.z);
        float b = -n.x * n.y * a;
        u = float3(1.0 - n.x * n.x * a, b, -n.x);
        v = float3(b, 1.0 - n.y * n.y * a, -n.y);
    }
    if (!all(isfinite(u)) || !all(isfinite(v))) return false;
    [unroll] for (uint i = 0u; i < MVEE_POINT_COUNT; i++)
    {
        float theta = TAU_RADIANS * (float(i) / float(MVEE_POINT_COUNT));
        float s, c;
        sincos(theta, s, c);
        points[i] = circleCenter + circleRadius * (c * u + s * v);
    }
    return true;
}

bool fit_mvee_5pt_fixed_iters(float2 points[MVEE_POINT_COUNT], uint iters, float eps, out MveeFitResult fit)
{
    fit.centerPx = 0.0;
    fit.radiusPx = 0.0;
    fit.conic = 0.0;
    float2 center = 0.0;
    [unroll] for (uint i = 0u; i < MVEE_POINT_COUNT; i++) center += points[i];
    center *= (1.0 / float(MVEE_POINT_COUNT));
    float normScale = 0.0;
    [unroll] for (uint i = 0u; i < MVEE_POINT_COUNT; i++) normScale = max(normScale, length(points[i] - center));
    if (normScale <= eps) return false;
    float2 normPts[MVEE_POINT_COUNT];
    [unroll] for (uint i = 0u; i < MVEE_POINT_COUNT; i++) normPts[i] = (points[i] - center) / normScale;
    float w[MVEE_POINT_COUNT];
    [unroll] for (uint i = 0u; i < MVEE_POINT_COUNT; i++) w[i] = 1.0 / float(MVEE_POINT_COUNT);
    [unroll] for (uint iter = 0u; iter < MVEE_MAX_ITERS; iter++)
    {
        if (iter >= iters) break;
        float3x3 xMat = 0.0;
        [unroll] for (uint i = 0u; i < MVEE_POINT_COUNT; i++)
        {
            float3 q = float3(normPts[i], 1.0);
            xMat += w[i] * float3x3(q.x * q.x, q.x * q.y, q.x, q.y * q.x, q.y * q.y, q.y, q.x, q.y, 1.0);
        }
        float3x3 xInv = 0.0;
        if (!inverse3x3(xMat, eps, xInv)) return false;
        uint maxIdx = 0u;
        float maxVal = -1.0;
        [unroll] for (uint i = 0u; i < MVEE_POINT_COUNT; i++)
        {
            float3 q = float3(normPts[i], 1.0);
            float m = dot(q, mul(xInv, q));
            if (m > maxVal) { maxVal = m; maxIdx = i; }
        }
        float denom = MVEE_AFFINE_DIMENSION * (maxVal - 1.0);
        if (denom <= eps) break;
        float alpha = saturate((maxVal - MVEE_AFFINE_DIMENSION) / denom);
        [unroll] for (uint i = 0u; i < MVEE_POINT_COUNT; i++) w[i] *= (1.0 - alpha);
        w[maxIdx] += alpha;
    }
    float2 ellipseCenter = 0.0;
    [unroll] for (uint i = 0u; i < MVEE_POINT_COUNT; i++) ellipseCenter += w[i] * normPts[i];
    float3 secondMoment = 0.0;
    [unroll] for (uint i = 0u; i < MVEE_POINT_COUNT; i++)
    {
        float2 d = normPts[i] - ellipseCenter;
        secondMoment += w[i] * float3(d.x * d.x, d.x * d.y, d.y * d.y);
    }
    secondMoment += float3(eps, 0.0, eps);
    float detS = secondMoment.x * secondMoment.z - secondMoment.y * secondMoment.y;
    if (detS <= eps) return false;
    float3 conicNorm = 0.5 * float3(secondMoment.z, -secondMoment.y, secondMoment.x) / detS;
    float trace = conicNorm.x + conicNorm.z;
    float detA = conicNorm.x * conicNorm.z - conicNorm.y * conicNorm.y;
    float disc = sqrt(max(0.25 * trace * trace - detA, 0.0));
    float axis0 = rsqrt(max(0.5 * trace + disc, eps));
    float axis1 = rsqrt(max(0.5 * trace - disc, eps));
    float maxQuad = 0.0;
    [unroll] for (uint i = 0u; i < MVEE_POINT_COUNT; i++)
    {
        float2 d = normPts[i] - ellipseCenter;
        float quad = dot(float3(d.x * d.x, d.x * d.y, d.y * d.y), float3(conicNorm.x, 2.0 * conicNorm.y, conicNorm.z));
        maxQuad = max(maxQuad, quad);
    }
    float inflate = maxQuad > 1.0 ? sqrt(maxQuad) : 1.0;
    fit.centerPx = center + ellipseCenter * normScale;
    fit.radiusPx = max(axis0, axis1) * inflate * normScale;
    fit.conic = conicNorm / max(inflate * inflate * normScale * normScale, eps);
    return all(isfinite(fit.centerPx)) && all(isfinite(fit.conic)) && isfinite(fit.radiusPx) && fit.radiusPx > 0.0;
}

bool prepare_conic(inout Ellipse e)
{
    e.bboxExtent = float2(e.radiusPx, e.radiusPx);
    float det = e.conic.x * e.conic.z - e.conic.y * e.conic.y;
    if (!all(isfinite(e.conic)) || e.conic.x <= POSITIVE_CONIC_FLOOR || e.conic.z <= POSITIVE_CONIC_FLOOR || det <= CONIC_DETERMINANT_FLOOR) return false;
    float xExtent = sqrt(max(e.conic.z / det, 0.0));
    float yExtent = sqrt(max(e.conic.x / det, 0.0));
    if (!all(isfinite(float2(xExtent, yExtent)))) return false;
    e.bboxExtent = clamp(float2(xExtent, yExtent), float2(MIN_BBOX_EXTENT_PX), float2(e.radiusPx));
    return true;
}

float eval_conic(Ellipse e, float2 d)
{
    return dot(d, mul(float2x2(e.conic.x, e.conic.y, e.conic.y, e.conic.z), d));
}

float min_conic_over_tile_box(Ellipse e, float2 boxMin, float2 boxMax)
{
    float minVal = min(min(eval_conic(e, float2(boxMin.x, boxMin.y)), eval_conic(e, float2(boxMax.x, boxMin.y))), min(eval_conic(e, float2(boxMin.x, boxMax.y)), eval_conic(e, float2(boxMax.x, boxMax.y))));
    if (boxMin.x <= 0.0 && 0.0 <= boxMax.x && boxMin.y <= 0.0 && 0.0 <= boxMax.y) return 0.0;
    if (e.conic.z > CONIC_DETERMINANT_FLOOR)
    {
        float y0 = clamp(-e.conic.y * boxMin.x / e.conic.z, boxMin.y, boxMax.y);
        float y1 = clamp(-e.conic.y * boxMax.x / e.conic.z, boxMin.y, boxMax.y);
        minVal = min(minVal, eval_conic(e, float2(boxMin.x, y0)));
        minVal = min(minVal, eval_conic(e, float2(boxMax.x, y1)));
    }
    if (e.conic.x > CONIC_DETERMINANT_FLOOR)
    {
        float x0 = clamp(-e.conic.y * boxMin.y / e.conic.x, boxMin.x, boxMax.x);
        float x1 = clamp(-e.conic.y * boxMax.y / e.conic.x, boxMin.x, boxMax.x);
        minVal = min(minVal, eval_conic(e, float2(x0, boxMin.y)));
        minVal = min(minVal, eval_conic(e, float2(x1, boxMax.y)));
    }
    return minVal;
}

bool tile_intersects(Ellipse e, uint tileX, uint tileY)
{
    float2 boxMin = float2(tileX * TILE_SIZE, tileY * TILE_SIZE) - e.centerPx;
    float2 boxMax = boxMin + float2(TILE_SIZE, TILE_SIZE);
    return min_conic_over_tile_box(e, boxMin, boxMax) <= 1.0 + CONIC_INTERSECTION_SLACK;
}

[Differentiable]
CachedEllipsoid build_cached_ellipsoid(no_diff CameraParams camera, no_diff float radiusScale, float3 position, float3 logScale, float4 rotation)
{
    float3 sigma = exp(logScale);
    float3 scale = max(sigma * (radiusScale * GAUSSIAN_SUPPORT_SIGMA_RADIUS), float3(SMALL_VALUE));
    float3 roLocal = rotate_by_quat(camera.camPos - position, rotation) / scale;
    return { roLocal, scale, rotation };
}

[Differentiable]
float build_raster_fused_opacity(no_diff CameraParams camera, no_diff float radiusScale, float3 position, float3 logScale, float rawOpacity)
{
    return opacity_from_raw(rawOpacity);
}

[Differentiable]
float raster_eval_alpha(float3 roLocal, float3 scale, float4 rotation, float fusedOpacity, no_diff float3 rayDir)
{
    float3 rayLocal = rotate_by_quat(rayDir, rotation) / max(scale, float3(SMALL_VALUE));
    float denom = dot(rayLocal, rayLocal);
    if (denom <= RAY_DENOMINATOR_FLOOR) return 0.0;
    float tClosest = -dot(rayLocal, roLocal) / denom;
    if (tClosest <= 0.0) return 0.0;
    float3 closest = roLocal + rayLocal * tClosest;
    float rho2 = max(dot(closest, closest), 0.0);
    return fusedOpacity * exp(-0.5 * GAUSSIAN_SUPPORT_SIGMA_RADIUS * GAUSSIAN_SUPPORT_SIGMA_RADIUS * rho2);
}

[Differentiable]
float4 raster_output(float3 accum, float trans, no_diff float3 background)
{
    float3 finalColor = accum + trans * background;
    return float4(pow(max(finalColor, 0.0), OUTPUT_GAMMA), 1.0 - trans);
}
