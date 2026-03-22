import "slangpy";

static const uint TILE_SIZE = 8u;
static const uint PROJECT_GROUP_SIZE = 256u;
static const uint RASTER_BATCH = 64u;
static const uint RASTER_THREAD_TILE_DIM = TILE_SIZE;
static const uint RASTER_GROUP_THREADS = RASTER_THREAD_TILE_DIM * RASTER_THREAD_TILE_DIM;
static const uint MVEE_POINT_COUNT = 5u;
static const uint MVEE_MAX_ITERS = 8u;
static const uint MVEE_ITERS = 6u;
static const float MVEE_SAFETY_SCALE = 1.0;
static const float MVEE_RADIUS_PAD_PX = 1.0;
static const float MVEE_EPS = 1e-6;
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

struct Gaussian3D : IDifferentiable {
    float3 position;
    float3 logScale;
    float4 rotation;
    float4 colorAlpha;

    static Gaussian3D load(Tensor<float, 1> splats, uint splatCount, uint splatId) {
        Gaussian3D gaussian;
        gaussian.position.x = splats[int(PARAM_POSITION_X * splatCount + splatId)];
        gaussian.position.y = splats[int(PARAM_POSITION_Y * splatCount + splatId)];
        gaussian.position.z = splats[int(PARAM_POSITION_Z * splatCount + splatId)];
        gaussian.logScale.x = splats[int(PARAM_SCALE_X * splatCount + splatId)];
        gaussian.logScale.y = splats[int(PARAM_SCALE_Y * splatCount + splatId)];
        gaussian.logScale.z = splats[int(PARAM_SCALE_Z * splatCount + splatId)];
        gaussian.rotation.x = splats[int(PARAM_ROTATION_W * splatCount + splatId)];
        gaussian.rotation.y = splats[int(PARAM_ROTATION_X * splatCount + splatId)];
        gaussian.rotation.z = splats[int(PARAM_ROTATION_Y * splatCount + splatId)];
        gaussian.rotation.w = splats[int(PARAM_ROTATION_Z * splatCount + splatId)];
        gaussian.colorAlpha.x = splats[int(PARAM_COLOR_R * splatCount + splatId)];
        gaussian.colorAlpha.y = splats[int(PARAM_COLOR_G * splatCount + splatId)];
        gaussian.colorAlpha.z = splats[int(PARAM_COLOR_B * splatCount + splatId)];
        gaussian.colorAlpha.w = splats[int(PARAM_RAW_OPACITY * splatCount + splatId)];
        return gaussian;
    }

    [Differentiable]
    static Gaussian3D build_cached_ellipsoid(Gaussian3D gaussian, no_diff Camera camera, no_diff float radiusScale) {
        Gaussian3D cached = gaussian;
        cached.logScale = max(exp(gaussian.logScale) * (radiusScale * GAUSSIAN_SUPPORT_SIGMA_RADIUS), float3(SMALL_VALUE));
        cached.position = rotate_by_quat(camera.camPos - gaussian.position, gaussian.rotation) / cached.logScale;
        return cached;
    }

    [Differentiable]
    float build_raster_fused_opacity() {
        return sigmoid(colorAlpha.w);
    }
};

struct ProjectionState {
    float4 centerRadiusDepth;
    float4 conicOpacity;
};

struct TileEntry {
    uint2 ids;
};

struct Camera {
    float3 camPos;
    float3x3 camBasis;
    float2 focalPixels;
    float2 principalPoint;
    float2 viewport;
    float nearDepth;
    float farDepth;
    float k1;
    float k2;

    float3 world_to_camera(float3 worldVector) {
        return mul(camBasis, worldVector);
    }

    float3 camera_to_world(float3 cameraVector) {
        return mul(transpose(camBasis), cameraVector);
    }

    float3 world_point_to_camera(float3 worldPos) {
        return world_to_camera(worldPos - camPos);
    }

    float3 camera_point_to_world(float3 cameraPos) {
        return camPos + camera_to_world(cameraPos);
    }

    float2 project_camera_to_screen_linear(float3 cameraPos) {
        float invDepth = 1.0 / max(cameraPos.z, SMALL_VALUE);
        return cameraPos.xy * (focalPixels * invDepth) + principalPoint;
    }

    float2 distort_normalized(float2 uv) {
        float r2 = dot(uv, uv);
        return uv * (1.0 + k1 * r2 + k2 * r2 * r2);
    }

    float2 undistort_normalized(float2 uvDistorted) {
        if (all(abs(float2(k1, k2)) <= SMALL_VALUE)) return uvDistorted;

        float radiusDistorted = length(uvDistorted);
        if (radiusDistorted <= SMALL_VALUE) return uvDistorted;

        float radius = radiusDistorted;
        [unroll] for (uint iter = 0u; iter < 6; iter++) {
            float r2 = radius * radius;
            float r4 = r2 * r2;
            float deriv = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * r4;
            if (abs(deriv) <= SMALL_VALUE) break;

            float nextRadius = radius - (radius * (1.0 + k1 * r2 + k2 * r4) - radiusDistorted) / deriv;
            if (!isfinite(nextRadius) || nextRadius < 0.0) break;
            radius = nextRadius;
        }
        return uvDistorted * (radius / max(radiusDistorted, SMALL_VALUE));
    }

    bool project_camera_to_screen(float3 cameraPos, out float2 screenPos) {
        if (cameraPos.z <= SMALL_VALUE) {
            screenPos = 0.0;
            return false;
        }

        float2 uv = distort_normalized(cameraPos.xy / cameraPos.z);
        screenPos = uv * focalPixels + principalPoint;
        return all(isfinite(screenPos));
    }

    bool project_world_to_screen(float3 worldPos, out float2 screenPos) {
        return project_camera_to_screen(world_point_to_camera(worldPos), screenPos);
    }

    float3 screen_to_world(float2 screenPos, float depth) {
        float2 focalSafe = max(focalPixels, float2(SMALL_VALUE));
        float depthSafe = max(depth, SMALL_VALUE);
        float2 uv = undistort_normalized((screenPos - principalPoint) / focalSafe);
        return camera_point_to_world(float3(uv * depthSafe, depthSafe));
    }

    float3 screen_to_world_ray(float2 screenPos) {
        float2 focalSafe = max(focalPixels, float2(SMALL_VALUE));
        float2 uv = undistort_normalized((screenPos - principalPoint) / focalSafe);
        return normalize(camera_to_world(float3(uv, 1.0)));
    }

    float pixel_world_size_max(float depth) {
        float depthSafe = max(depth, SMALL_VALUE);
        float minFocal = max(min(focalPixels.x, focalPixels.y), SMALL_VALUE);
        return SPLAT_PIXEL_CLAMP_PX * depthSafe / minFocal;
    }

    bool is_depth_in_range(float depthValue){
        return depthValue >= nearDepth && depthValue <= farDepth;
    }
};

struct Ellipse {
    float2 centerPx;
    float radiusPx;
    float3 conic;
    float2 bboxExtent;

    [mutating]
    bool prepare_conic() {
        bboxExtent = float2(radiusPx, radiusPx);
        float det = conic.x * conic.z - conic.y * conic.y;
        if (!all(isfinite(conic)) || conic.x <= POSITIVE_CONIC_FLOOR || conic.z <= POSITIVE_CONIC_FLOOR || det <= CONIC_DETERMINANT_FLOOR) return false;
        float xExtent = sqrt(max(conic.z / det, 0.0));
        float yExtent = sqrt(max(conic.x / det, 0.0));
        if (!all(isfinite(float2(xExtent, yExtent)))) return false;
        bboxExtent = clamp(float2(xExtent, yExtent), float2(MIN_BBOX_EXTENT_PX), float2(radiusPx));
        return true;
    }

    float eval_conic(float2 d) {
        return dot(d, mul(float2x2(conic.x, conic.y, conic.y, conic.z), d));
    }

    float min_conic_over_tile_box(float2 boxMin, float2 boxMax) {
        float minVal = min(min(eval_conic(float2(boxMin.x, boxMin.y)), eval_conic(float2(boxMax.x, boxMin.y))), min(eval_conic(float2(boxMin.x, boxMax.y)), eval_conic(float2(boxMax.x, boxMax.y))));
        if (boxMin.x <= 0.0 && 0.0 <= boxMax.x && boxMin.y <= 0.0 && 0.0 <= boxMax.y) return 0.0;
        if (conic.z > CONIC_DETERMINANT_FLOOR) {
            float y0 = clamp(-conic.y * boxMin.x / conic.z, boxMin.y, boxMax.y);
            float y1 = clamp(-conic.y * boxMax.x / conic.z, boxMin.y, boxMax.y);
            minVal = min(minVal, eval_conic(float2(boxMin.x, y0)));
            minVal = min(minVal, eval_conic(float2(boxMax.x, y1)));
        }
        if (conic.x > CONIC_DETERMINANT_FLOOR) {
            float x0 = clamp(-conic.y * boxMin.y / conic.x, boxMin.x, boxMax.x);
            float x1 = clamp(-conic.y * boxMax.y / conic.x, boxMin.x, boxMax.x);
            minVal = min(minVal, eval_conic(float2(x0, boxMin.y)));
            minVal = min(minVal, eval_conic(float2(x1, boxMax.y)));
        }
        return minVal;
    }

    bool tile_intersects(uint tileX, uint tileY) {
        float2 boxMin = float2(tileX * TILE_SIZE, tileY * TILE_SIZE) - centerPx;
        float2 boxMax = boxMin + float2(TILE_SIZE, TILE_SIZE);
        return min_conic_over_tile_box(boxMin, boxMax) <= 1.0 + CONIC_INTERSECTION_SLACK;
    }
};

struct MveeFitResult {
    float2 centerPx;
    float radiusPx;
    float3 conic;
};

bool inverse3x3(float3x3 m, float eps, out float3x3 mInv) {
    float3x3 cofactor = 0.0;
    [unroll] for (uint i = 0u; i < 3u; i++){
        [unroll] for (uint j = 0u; j < 3u; j++){
            float sign = ((i + j) & 1u) == 0u ? 1.0 : -1.0;
            uint2 idx0 = (uint2(i, j) + uint2(1, 1)) % 3;
            uint2 idx1 = (uint2(i, j) + uint2(2, 2)) % 3;
            float minorVal = m[idx0.x][idx0.y] * m[idx1.x][idx1.y] - m[idx0.x][idx1.y] * m[idx1.x][idx0.y];
            cofactor[i][j] = sign * minorVal;
        }
    }
    float det = dot(m[0], cofactor[0]);
    if (abs(det) <= eps) {
        mInv = 0.0;
        return false;
    }
    mInv = transpose(cofactor) * (1.0 / det);
    return true;
}

[Differentiable] vector<float, N> normalize_safe<let N : int>(vector<float, N> v) { float lenSq = dot(v, v); return lenSq > SMALL_VALUE ? v * rsqrt(lenSq) : 0.0; }
[Differentiable] float3 rotate_by_quat(float3 v, float4 q) { return v + 2.0 * cross(cross(v, q.yzw) + q.x * v, q.yzw); }
[Differentiable] float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

bool generate_silhouette_points_local5(float3 roLocal, float eps, out float3 points[MVEE_POINT_COUNT]) {
    [unroll] for (uint i = 0u; i < MVEE_POINT_COUNT; i++) points[i] = 0.0;
    float d = length(roLocal);
    if (d <= 1.0 + eps) return false;
    float3 n = roLocal / d;
    float3 circleCenter = n / d;
    float circleRadius = sqrt(max(1.0 - 1.0 / (d * d), 0.0));
    float3 u;
    float3 v;
    if (n.z < SOUTH_POLE_DOT_FLOOR) { u = float3(0.0, -1.0, 0.0); v = float3(-1.0, 0.0, 0.0); }
    else {
        float a = 1.0 / (1.0 + n.z);
        float b = -n.x * n.y * a;
        u = float3(1.0 - n.x * n.x * a, b, -n.x);
        v = float3(b, 1.0 - n.y * n.y * a, -n.y);
    }
    if (!all(isfinite(u)) || !all(isfinite(v))) return false;
    [unroll] for (uint i = 0u; i < MVEE_POINT_COUNT; i++) {
        float theta = TAU_RADIANS * (float(i) / float(MVEE_POINT_COUNT));
        float s, c;
        sincos(theta, s, c);
        points[i] = circleCenter + circleRadius * (c * u + s * v);
    }
    return true;
}

bool fit_mvee_5pt_fixed_iters(float2 points[MVEE_POINT_COUNT], uint iters, float eps, out MveeFitResult fit) {
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
    [unroll] for (uint iter = 0u; iter < MVEE_MAX_ITERS; iter++) {
        if (iter >= iters) break;
        float3x3 xMat = 0.0;
        [unroll] for (uint i = 0u; i < MVEE_POINT_COUNT; i++) {
            float3 q = float3(normPts[i], 1.0);
            xMat += w[i] * float3x3(q.x * q.x, q.x * q.y, q.x, q.y * q.x, q.y * q.y, q.y, q.x, q.y, 1.0);
        }
        float3x3 xInv = 0.0;
        if (!inverse3x3(xMat, eps, xInv)) return false;
        uint maxIdx = 0u;
        float maxVal = -1.0;
        [unroll] for (uint i = 0u; i < MVEE_POINT_COUNT; i++) {
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
    [unroll] for (uint i = 0u; i < MVEE_POINT_COUNT; i++) {
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
    [unroll] for (uint i = 0u; i < MVEE_POINT_COUNT; i++) {
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

[Differentiable]
float raster_eval_alpha(Gaussian3D gaussian, no_diff Camera camera, no_diff float radiusScale, no_diff float3 rayDir) {
    Gaussian3D cached = Gaussian3D.build_cached_ellipsoid(gaussian, camera, radiusScale);
    float3 rayLocal = rotate_by_quat(rayDir, cached.rotation) / max(cached.logScale, float3(SMALL_VALUE));
    float denom = dot(rayLocal, rayLocal);
    if (denom <= RAY_DENOMINATOR_FLOOR) return 0.0;
    float tClosest = -dot(rayLocal, cached.position) / denom;
    if (tClosest <= 0.0) return 0.0;
    float3 closest = cached.position + rayLocal * tClosest;
    float rho2 = max(dot(closest, closest), 0.0);
    return gaussian.colorAlpha.w * exp(-0.5 * GAUSSIAN_SUPPORT_SIGMA_RADIUS * GAUSSIAN_SUPPORT_SIGMA_RADIUS * rho2);
}

[Differentiable]
float4 raster_output(float3 accum, float trans, no_diff float3 background) {
    return float4(pow(max(accum + trans * background, 0.0), OUTPUT_GAMMA), 1.0 - trans);
}
