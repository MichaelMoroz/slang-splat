import "slangpy";

static const uint TILE_SIZE = 8u;
static const uint PROJECT_GROUP_SIZE = 256u;
static const uint RASTER_BATCH = 64u;
static const uint RASTER_THREAD_TILE_DIM = TILE_SIZE;
static const uint RASTER_GROUP_THREADS = RASTER_THREAD_TILE_DIM * RASTER_THREAD_TILE_DIM;
static const uint ELLIPSE_POINT_COUNT = 5u;
static const float ELLIPSE_SAFETY_SCALE = 1.0;
static const float ELLIPSE_RADIUS_PAD_PX = 1.0;
static const float ELLIPSE_EPS = 1e-6;
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
static const float BBOX_PAD_PX = 1.0;
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
        gaussian.position.x = splats[PARAM_POSITION_X * splatCount + splatId];
        gaussian.position.y = splats[PARAM_POSITION_Y * splatCount + splatId];
        gaussian.position.z = splats[PARAM_POSITION_Z * splatCount + splatId];
        gaussian.logScale.x = splats[PARAM_SCALE_X * splatCount + splatId];
        gaussian.logScale.y = splats[PARAM_SCALE_Y * splatCount + splatId];
        gaussian.logScale.z = splats[PARAM_SCALE_Z * splatCount + splatId];
        gaussian.rotation.x = splats[PARAM_ROTATION_W * splatCount + splatId];
        gaussian.rotation.y = splats[PARAM_ROTATION_X * splatCount + splatId];
        gaussian.rotation.z = splats[PARAM_ROTATION_Y * splatCount + splatId];
        gaussian.rotation.w = splats[PARAM_ROTATION_Z * splatCount + splatId];
        gaussian.colorAlpha.x = splats[PARAM_COLOR_R * splatCount + splatId];
        gaussian.colorAlpha.y = splats[PARAM_COLOR_G * splatCount + splatId];
        gaussian.colorAlpha.z = splats[PARAM_COLOR_B * splatCount + splatId];
        gaussian.colorAlpha.w = splats[PARAM_RAW_OPACITY * splatCount + splatId];
        return gaussian;
    }

    [Differentiable]
    static Gaussian3D camera_gaussian(Gaussian3D gaussian, no_diff Camera camera, no_diff float radiusScale, no_diff float supportSigmaRadius) {
        Gaussian3D cameraGaussian = gaussian;
        cameraGaussian.logScale = max(exp(gaussian.logScale) * (radiusScale * supportSigmaRadius), float3(SMALL_VALUE));
        cameraGaussian.position = rotate_by_quat(camera.camPos - gaussian.position, gaussian.rotation) / cameraGaussian.logScale;
        return cameraGaussian;
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
        bboxExtent = clamp(float2(xExtent, yExtent) + BBOX_PAD_PX, float2(MIN_BBOX_EXTENT_PX), float2(radiusPx));
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

struct LinearSystem<let N : int> {
    float aug[N * (N + 1)];

    int flat_index(int i0, int i1) {
        return i0 * (N + 1) + i1;
    }

    float load(int i0, int i1) {
        return aug[flat_index(i0, i1)];
    }

    [mutating]
    void store(int i0, int i1, float value) {
        aug[flat_index(i0, i1)] = value;
    }

    __subscript(int i0, int i1)->float {
        get { return this.load(i0, i1); }
        set { this.store(i0, i1, newValue); }
    }

    [mutating]
    bool solve(float eps, out float solution[N]) {
        [unroll] for (uint row = 0u; row < N; row++) solution[row] = 0.0;
        [unroll] for (uint pivot = 0u; pivot < N; pivot++) {
            uint bestRow = pivot;
            float bestAbs = abs(this.load(pivot, pivot));
            [unroll] for (uint row = pivot + 1u; row < N; row++) {
                float rowAbs = abs(this.load(row, pivot));
                if (rowAbs > bestAbs) { bestAbs = rowAbs; bestRow = row; }
            }
            if (bestAbs <= eps) return false;
            if (bestRow != pivot) {
                [unroll] for (uint col = pivot; col <= N; col++) {
                    float tmp = this.load(pivot, col);
                    this.store(pivot, col, this.load(bestRow, col));
                    this.store(bestRow, col, tmp);
                }
            }
            float invPivot = 1.0 / this.load(pivot, pivot);
            [unroll] for (uint col = pivot; col <= N; col++) this.store(pivot, col, this.load(pivot, col) * invPivot);
            [unroll] for (uint row = 0u; row < N; row++) {
                if (row == pivot) continue;
                float factor = this.load(row, pivot);
                [unroll] for (uint col = pivot; col <= N; col++) this.store(row, col, this.load(row, col) - factor * this.load(pivot, col));
            }
        }
        [unroll] for (uint row = 0u; row < N; row++) solution[row] = this.load(row, N);
        return true;
    }
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

bool compute_gaussian_outline_ellipse(
    Gaussian3D gaussian,
    Camera camera,
    float radiusScale,
    float supportSigmaRadius,
    out Ellipse ellipse)
{
    ellipse.centerPx = 0.0;
    ellipse.radiusPx = 0.0;
    ellipse.conic = 0.0;
    ellipse.bboxExtent = 0.0;
    Gaussian3D supportGaussian = Gaussian3D.camera_gaussian(gaussian, camera, radiusScale, supportSigmaRadius);

    // Build the tangent circle on the unit ellipsoid in support-local space.
    float3 viewOriginLocal = supportGaussian.position;
    float viewDistance = length(viewOriginLocal);
    if (viewDistance <= 1.0 + ELLIPSE_EPS) return false;
    float3 viewDirLocal = viewOriginLocal / viewDistance;
    float3 tangentCircleCenter = viewDirLocal / viewDistance;
    float tangentCircleRadius = sqrt(max(1.0 - 1.0 / (viewDistance * viewDistance), 0.0));
    float3 tangentBasisU;
    float3 tangentBasisV;
    if (viewDirLocal.z < SOUTH_POLE_DOT_FLOOR) {
        tangentBasisU = float3(0.0, -1.0, 0.0);
        tangentBasisV = float3(-1.0, 0.0, 0.0);
    } else {
        float invSouthPoleDistance = 1.0 / (1.0 + viewDirLocal.z);
        float xyMix = -viewDirLocal.x * viewDirLocal.y * invSouthPoleDistance;
        tangentBasisU = float3(1.0 - viewDirLocal.x * viewDirLocal.x * invSouthPoleDistance, xyMix, -viewDirLocal.x);
        tangentBasisV = float3(xyMix, 1.0 - viewDirLocal.y * viewDirLocal.y * invSouthPoleDistance, -viewDirLocal.y);
    }
    if (!all(isfinite(tangentBasisU)) || !all(isfinite(tangentBasisV))) return false;
    float3 worldCenter = gaussian.position;
    float2 outlinePoints[ELLIPSE_POINT_COUNT];
    float2 outlineMin = float2(1e30, 1e30);
    float2 outlineMax = float2(-1e30, -1e30);

    // Project the outline points first so off-screen splats can be rejected before the fit.
    [unroll] for (uint row = 0u; row < ELLIPSE_POINT_COUNT; row++) {
        float theta = TAU_RADIANS * (float(row) / float(ELLIPSE_POINT_COUNT));
        float s, c;
        sincos(theta, s, c);
        float3 supportPointLocal = tangentCircleCenter + tangentCircleRadius * (c * tangentBasisU + s * tangentBasisV);
        float3 worldPoint = worldCenter + rotate_by_quat(supportPointLocal * supportGaussian.logScale, float4(gaussian.rotation.x, -gaussian.rotation.yzw));
        float2 screenPoint = 0.0;
        if (!camera.project_world_to_screen(worldPoint, screenPoint)) return false;
        outlinePoints[row] = screenPoint;
        outlineMin = min(outlineMin, screenPoint);
        outlineMax = max(outlineMax, screenPoint);
    }

    // The projected outline never reaches the viewport.
    // WARNING: if the points are non-uniform in screen-space then bbox might not represent the actual ellipse extent
    if (outlineMax.x < 0.0 || outlineMin.x >= camera.viewport.x || outlineMax.y < 0.0 || outlineMin.y >= camera.viewport.y) return false;
    float2 visibleOutlineMin = max(outlineMin, 0.0);
    float2 visibleOutlineMax = min(outlineMax, camera.viewport - float2(TINY_VALUE, TINY_VALUE));
    int minTileX = int(floor(visibleOutlineMin.x / TILE_SIZE));
    int maxTileX = int(floor(visibleOutlineMax.x / TILE_SIZE));
    int minTileY = int(floor(visibleOutlineMin.y / TILE_SIZE));
    int maxTileY = int(floor(visibleOutlineMax.y / TILE_SIZE));

    // If the visible outline fits in one tile, an axis-aligned ellipse is enough, the exact fit will not help cull more tiles.
    if (minTileX == maxTileX && minTileY == maxTileY) {
        float2 halfExtent = max(0.5 * (outlineMax - outlineMin), float2(ELLIPSE_EPS, ELLIPSE_EPS));
        ellipse.centerPx = 0.5 * (outlineMin + outlineMax);
        ellipse.radiusPx = max(halfExtent.x, halfExtent.y);
        ellipse.conic = float3(1.0 / max(halfExtent.x * halfExtent.x, ELLIPSE_EPS), 0.0, 1.0 / max(halfExtent.y * halfExtent.y, ELLIPSE_EPS));
        ellipse.bboxExtent = halfExtent;
        return all(isfinite(ellipse.centerPx)) && all(isfinite(ellipse.conic)) && isfinite(ellipse.radiusPx) && ellipse.radiusPx > 0.0;
    }

    // Fit the screen-space ellipse directly from bbox-relative coordinates.
    float2 bboxCenter = 0.5 * (outlineMin + outlineMax);
    float2 bboxHalfExtent = max(0.5 * (outlineMax - outlineMin), float2(ELLIPSE_EPS, ELLIPSE_EPS));
    LinearSystem<ELLIPSE_POINT_COUNT> system;
    [unroll] for (uint row = 0u; row < ELLIPSE_POINT_COUNT; row++) {
        float2 norm = (outlinePoints[row] - bboxCenter) / bboxHalfExtent;
        system[row, 0] = norm.x * norm.x;
        system[row, 1] = 2.0 * norm.x * norm.y;
        system[row, 2] = norm.y * norm.y;
        system[row, 3] = norm.x;
        system[row, 4] = norm.y;
        system[row, ELLIPSE_POINT_COUNT] = 1.0;
    }

    // Convert the normalized implicit conic back into pixel-space ellipse data.
    float solution[ELLIPSE_POINT_COUNT];
    if (!system.solve(ELLIPSE_EPS, solution)) return false;
    float3 conicNorm = float3(solution[0], solution[1], solution[2]);
    float2 linear = float2(solution[3], solution[4]);
    float detA = conicNorm.x * conicNorm.z - conicNorm.y * conicNorm.y;
    if (detA <= ELLIPSE_EPS) return false;
    float2 ellipseCenter = 0.5 * float2(
        (conicNorm.y * linear.y - conicNorm.z * linear.x) / detA,
        (conicNorm.y * linear.x - conicNorm.x * linear.y) / detA
    );
    float centerScale = 1.0 + dot(ellipseCenter, float2(
        conicNorm.x * ellipseCenter.x + conicNorm.y * ellipseCenter.y,
        conicNorm.y * ellipseCenter.x + conicNorm.z * ellipseCenter.y
    ));
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
        conicNorm.z / max(bboxHalfExtent.y * bboxHalfExtent.y, ELLIPSE_EPS));
    ellipse.bboxExtent = float2(ellipse.radiusPx, ellipse.radiusPx);
    return all(isfinite(ellipse.centerPx)) && all(isfinite(ellipse.conic)) && isfinite(ellipse.radiusPx) && ellipse.radiusPx > 0.0;
}

[Differentiable]
float raster_eval_alpha(Gaussian3D gaussian, no_diff Camera camera, no_diff float radiusScale, no_diff float3 rayDir) {
    Gaussian3D cached = Gaussian3D.camera_gaussian(gaussian, camera, radiusScale, GAUSSIAN_SUPPORT_SIGMA_RADIUS);
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
