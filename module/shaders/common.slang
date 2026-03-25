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
static const float CONIC_DETERMINANT_FLOOR = 1e-20;
static const float CONIC_AXIS_FLOOR = 1e-20;
static const float CONIC_INTERSECTION_SLACK = 1e-5;
static const float DEPTH_VISIBILITY_FLOOR = 1e-4;
static const float SPLAT_PIXEL_CLAMP_PX = 0.75;
static const float MIN_SCREEN_RADIUS_PX = 1.0;
static const float MIN_BBOX_EXTENT_PX = 1e-4;
static const float BBOX_PAD_PX = 1.0;
static const float MIN_DEBUG_THICKNESS_PX = 0.25;
static const float DEBUG_ELLIPSE_THICKNESS_PX = 2.0;
static const float OUTPUT_GAMMA = 2.2;
static const uint FLOAT_SIGN_MASK = 0x80000000u;
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
static const uint DEBUG_MODE_NORMAL = 0u;
static const uint DEBUG_MODE_PROCESSED_COUNT = 1u;
static const uint DEBUG_MODE_DEPTH_MEAN = 2u;
static const uint DEBUG_MODE_DEPTH_STD = 3u;
static const uint DEBUG_MODE_ELLIPSE_OUTLINES = 4u;
static const uint DEBUG_MODE_SPLAT_SPATIAL_DENSITY = 5u;
static const uint DEBUG_MODE_SPLAT_SCREEN_DENSITY = 6u;

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
    static Gaussian3D camera_gaussian(Gaussian3D gaussian, no_diff Camera camera, no_diff float radiusScale, no_diff float supportSigmaRadius, no_diff float maxAnisotropy) {
        Gaussian3D cameraGaussian = gaussian;
        float3 supportScale = max(exp(gaussian.logScale) * (radiusScale * supportSigmaRadius), float3(SMALL_VALUE));
        float supportMaxScale = max(supportScale.x, max(supportScale.y, supportScale.z));
        float supportMinScale = supportMaxScale / max(maxAnisotropy, 1.0);
        cameraGaussian.logScale = max(supportScale, float3(supportMinScale));
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

struct ScanlineEntry {
    uint splatId;
    uint stripIndex;
    uint transpose;
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
        float level = 1.0 + CONIC_INTERSECTION_SLACK;
        float xExtent = sqrt(max(level * conic.z / det, 0.0));
        float yExtent = sqrt(max(level * conic.x / det, 0.0));
        if (!all(isfinite(float2(xExtent, yExtent)))) return false;
        // The axis-aligned bounds come from the fitted conic, not the scalar major-axis radius.
        // Clamping them back to radiusPx drops tiles for large rotated ellipses.
        bboxExtent = max(float2(xExtent, yExtent) + BBOX_PAD_PX, float2(MIN_BBOX_EXTENT_PX));
        return true;
    }

    void scanline_frame(out bool transpose, out float2 localCenter, out float2 localExtent, out float3 localConic) {
        transpose = bboxExtent.x < bboxExtent.y;
        localCenter = transpose ? centerPx.yx : centerPx;
        localExtent = transpose ? bboxExtent.yx : bboxExtent;
        localConic = transpose ? float3(conic.z, conic.y, conic.x) : conic;
    }

    bool range_for_orthogonal_coord(float3 stripConic, float coord, out float rangeMin, out float rangeMax) {
        float det = stripConic.x * stripConic.z - stripConic.y * stripConic.y;
        float radicand = stripConic.x * (1.0 + CONIC_INTERSECTION_SLACK) - det * coord * coord;
        if (radicand < 0.0) {
            rangeMin = 0.0;
            rangeMax = 0.0;
            return false;
        }
        float root = sqrt(max(radicand, 0.0));
        rangeMin = (-stripConic.y * coord - root) / stripConic.x;
        rangeMax = (-stripConic.y * coord + root) / stripConic.x;
        return true;
    }

    bool span_for_strip(float3 stripConic, float coord0, float coord1, out float spanMin, out float spanMax) {
        spanMin = 1e30;
        spanMax = -1e30;
        bool hit = false;
        float rangeMin, rangeMax;
        if (range_for_orthogonal_coord(stripConic, coord0, rangeMin, rangeMax)) { spanMin = min(spanMin, rangeMin); spanMax = max(spanMax, rangeMax); hit = true; }
        if (range_for_orthogonal_coord(stripConic, coord1, rangeMin, rangeMax)) { spanMin = min(spanMin, rangeMin); spanMax = max(spanMax, rangeMax); hit = true; }
        float det = stripConic.x * stripConic.z - stripConic.y * stripConic.y;
        float critDenomSq = stripConic.z * det;
        if (critDenomSq > 0.0) {
            // The interior x-extrema can be far from zero for huge screen-space ellipses.
            // Clamping this denominator to a large epsilon collapses those extrema and drops tiles.
            float coordCrit = stripConic.y * sqrt(1.0 + CONIC_INTERSECTION_SLACK) / sqrt(critDenomSq);
            if (coord0 <= coordCrit && coordCrit <= coord1 && range_for_orthogonal_coord(stripConic, coordCrit, rangeMin, rangeMax)) { spanMin = min(spanMin, rangeMin); spanMax = max(spanMax, rangeMax); hit = true; }
            if (coord0 <= -coordCrit && -coordCrit <= coord1 && range_for_orthogonal_coord(stripConic, -coordCrit, rangeMin, rangeMax)) { spanMin = min(spanMin, rangeMin); spanMax = max(spanMax, rangeMax); hit = true; }
        }
        return hit;
    }

    float eval_conic(float2 localPoint) {
        return conic.x * localPoint.x * localPoint.x + 2.0 * conic.y * localPoint.x * localPoint.y + conic.z * localPoint.y * localPoint.y;
    }

    bool tile_intersects(float2 boxMin, float2 boxMax) {
        float minValue = 1e30;
        minValue = min(minValue, eval_conic(float2(boxMin.x, boxMin.y)));
        minValue = min(minValue, eval_conic(float2(boxMin.x, boxMax.y)));
        minValue = min(minValue, eval_conic(float2(boxMax.x, boxMin.y)));
        minValue = min(minValue, eval_conic(float2(boxMax.x, boxMax.y)));

        float xAtMinY = clamp(-conic.y * boxMin.y / max(conic.x, POSITIVE_CONIC_FLOOR), boxMin.x, boxMax.x);
        float xAtMaxY = clamp(-conic.y * boxMax.y / max(conic.x, POSITIVE_CONIC_FLOOR), boxMin.x, boxMax.x);
        float yAtMinX = clamp(-conic.y * boxMin.x / max(conic.z, POSITIVE_CONIC_FLOOR), boxMin.y, boxMax.y);
        float yAtMaxX = clamp(-conic.y * boxMax.x / max(conic.z, POSITIVE_CONIC_FLOOR), boxMin.y, boxMax.y);

        minValue = min(minValue, eval_conic(float2(xAtMinY, boxMin.y)));
        minValue = min(minValue, eval_conic(float2(xAtMaxY, boxMax.y)));
        minValue = min(minValue, eval_conic(float2(boxMin.x, yAtMinX)));
        minValue = min(minValue, eval_conic(float2(boxMax.x, yAtMaxX)));

        if (boxMin.x <= 0.0 && 0.0 <= boxMax.x && boxMin.y <= 0.0 && 0.0 <= boxMax.y) minValue = 0.0;
        return minValue <= 1.0 + CONIC_INTERSECTION_SLACK;
    }

};

// 5-point conic fit via renormalization: maps two points to canonical positions,
// reducing the 5x5 system to 3x3. Returns coefficients for Ax^2+2Bxy+Cy^2+Dx+Ey = 1.
bool solve_conic_renorm(float2 pts[5], float eps, out float solution[5])
{
    [unroll] for (uint i = 0u; i < 5u; i++) solution[i] = 0.0;

    float sx = pts[0].x - pts[1].x;
    float sy = pts[1].y - pts[0].y;
    if (abs(sx) <= eps || abs(sy) <= eps) return false;

    float isx = 1.0 / sx;
    float isy = 1.0 / sy;
    float ox = pts[1].x;
    float oy = pts[0].y;

    float u2 = (pts[2].x - ox) * isx, v2 = (pts[2].y - oy) * isy;
    float u3 = (pts[3].x - ox) * isx, v3 = (pts[3].y - oy) * isy;
    float u4 = (pts[4].x - ox) * isx, v4 = (pts[4].y - oy) * isy;

    float m00 = u2*u2 - u2, m01 = 2.0*u2*v2, m02 = v2*v2 - v2, r0 = u2 + v2 - 1.0;
    float m10 = u3*u3 - u3, m11 = 2.0*u3*v3, m12 = v3*v3 - v3, r1 = u3 + v3 - 1.0;
    float m20 = u4*u4 - u4, m21 = 2.0*u4*v4, m22 = v4*v4 - v4, r2 = u4 + v4 - 1.0;

    // Forward elimination (no pivoting)
    if (abs(m00) <= eps) return false;
    { float inv = 1.0/m00; m01*=inv; m02*=inv; r0*=inv; }
    { float f=m10; m11-=f*m01; m12-=f*m02; r1-=f*r0; }
    { float f=m20; m21-=f*m01; m22-=f*m02; r2-=f*r0; }
    if (abs(m11) <= eps) return false;
    { float inv = 1.0/m11; m12*=inv; r1*=inv; }
    { float f=m21; m22-=f*m12; r2-=f*r1; }
    if (abs(m22) <= eps) return false;
    r2 /= m22;

    // Back-substitution in renormalized space
    float Cp = r2;
    float Bp = r1 - m12 * Cp;
    float Ap = r0 - m01 * Bp - m02 * Cp;
    float Dp = -(Ap + 1.0);
    float Ep = -(Cp + 1.0);

    // Back-convert to original coordinate space
    float isx2 = isx*isx, isy2 = isy*isy, isxisy = isx*isy;
    float A = Ap * isx2;
    float B = Bp * isxisy;
    float C = Cp * isy2;
    float d = -2.0*A*ox - 2.0*B*oy + Dp*isx;
    float e = -2.0*B*ox - 2.0*C*oy + Ep*isy;
    float F = A*ox*ox + 2.0*B*ox*oy + C*oy*oy - Dp*isx*ox - Ep*isy*oy + 1.0;
    if (abs(F) <= eps) return false;

    // Negate F: renorm internally solves RHS=-1, production convention is RHS=+1
    float invF = -1.0 / F;
    solution[0] = A * invF;
    solution[1] = B * invF;
    solution[2] = C * invF;
    solution[3] = d * invF;
    solution[4] = e * invF;
    return true;
}

[Differentiable] vector<float, N> normalize_safe<let N : int>(vector<float, N> v) { float lenSq = dot(v, v); return lenSq > SMALL_VALUE ? v * rsqrt(lenSq) : 0.0; }
[Differentiable] float3 rotate_by_quat(float3 v, float4 q) { return v + 2.0 * cross(cross(v, q.yzw) + q.x * v, q.yzw); }
[Differentiable] float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
uint float_to_ordered_uint(float value) { uint bits = asuint(value); return (bits & FLOAT_SIGN_MASK) != 0u ? ~bits : (bits | FLOAT_SIGN_MASK); }

uint hash_u32(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

uint hash_combine_u32(uint a, uint b) {
    return hash_u32(a ^ (b + 0x9e3779b9u + (a << 6u) + (a >> 2u)));
}

float hash_to_unit_float(uint x) {
    return float(x >> 8u) * (1.0 / 16777216.0);
}

float2 stochastic_subpixel_offset(uint pixelId, uint renderSeed, float strength) {
    uint seed = hash_combine_u32(pixelId, renderSeed);
    return saturate(strength) * float2(
        hash_to_unit_float(hash_u32(seed ^ 0xa511e9b3u)),
        hash_to_unit_float(hash_u32(seed ^ 0x63d83595u))
    );
}

float3 jet_colormap(float x) {
    float t = saturate(x);
    return float3(
        saturate(1.5 - abs(4.0 * t - 3.0)),
        saturate(1.5 - abs(4.0 * t - 2.0)),
        saturate(1.5 - abs(4.0 * t - 1.0))
    );
}

float3 raster_processed_count_color(uint processedCount, uint maxSplatSteps) {
    float logMax = log2(float(maxSplatSteps) + 1.0);
    float normalized = (logMax > 0.0) ? saturate(log2(float(processedCount) + 1.0) / logMax) : 0.0;
    return jet_colormap(normalized);
}

float debug_luminance(float3 rgb) {
    return dot(rgb, float3(0.2126, 0.7152, 0.0722));
}

float debug_normalize_range(float value, float2 range) {
    return saturate((value - range.x) / max(range.y - range.x, SMALL_VALUE));
}

float gaussian_scale_volume(float3 logScale) {
    float3 scale = exp(logScale);
    return max(scale.x * scale.y * scale.z, SMALL_VALUE);
}

float projected_ellipse_area(float3 conic) {
    float det = conic.x * conic.z - conic.y * conic.y;
    return max(0.5 * TAU_RADIANS / sqrt(max(det, CONIC_DETERMINANT_FLOOR)), SMALL_VALUE);
}

[Differentiable]
float3 depth_state_step(float weightSum, float depthMean, float depthM2, float contributionWeight, float depth) {
    if (contributionWeight <= SMALL_VALUE || depth <= 0.0) return float3(weightSum, depthMean, depthM2);
    float totalWeight = weightSum + contributionWeight;
    float delta = depth - depthMean;
    float nextDepthMean = depthMean + (contributionWeight / max(totalWeight, SMALL_VALUE)) * delta;
    float nextDepthM2 = depthM2 + weightSum * contributionWeight * delta * delta / max(totalWeight, SMALL_VALUE);
    return float3(totalWeight, nextDepthMean, nextDepthM2);
}

float3 depth_state_rewind(float weightSum, float depthMean, float depthM2, float contributionWeight, float depth) {
    if (contributionWeight <= SMALL_VALUE || depth <= 0.0) return float3(weightSum, depthMean, depthM2);
    float prevWeightSum = max(weightSum - contributionWeight, 0.0);
    if (prevWeightSum <= SMALL_VALUE) return 0.0;
    float prevDepthMean = (weightSum * depthMean - contributionWeight * depth) / prevWeightSum;
    float delta = depth - prevDepthMean;
    float prevDepthM2 = max(depthM2 - prevWeightSum * contributionWeight * delta * delta / max(weightSum, SMALL_VALUE), 0.0);
    return float3(prevWeightSum, prevDepthMean, prevDepthM2);
}

[Differentiable]
float raster_depth_output(float weightSum, float depthMean, float depthM2) {
    if (weightSum <= SMALL_VALUE) return 0.0;
    float depthStd = sqrt(max(depthM2 / max(weightSum, SMALL_VALUE), 0.0));
    float depthStdRatio = depthStd / max(abs(depthMean), SMALL_VALUE);
    return depthStdRatio;
}

bool compute_gaussian_outline_ellipse(
    Gaussian3D gaussian,
    Camera camera,
    float3 cameraCenter,
    float radiusScale,
    float supportSigmaRadius,
    float maxAnisotropy,
    out Ellipse ellipse)
{
    ellipse.centerPx = 0.0;
    ellipse.radiusPx = 0.0;
    ellipse.conic = 0.0;
    ellipse.bboxExtent = 0.0;
    Gaussian3D cameraGaussian = Gaussian3D.camera_gaussian(gaussian, camera, radiusScale, supportSigmaRadius, maxAnisotropy);
    float2 screenCenter = 0.0;
    if (!camera.project_camera_to_screen(cameraCenter, screenCenter)) return false;
    float4 invRotation = float4(gaussian.rotation.x, -gaussian.rotation.yzw);

    // Build the tangent circle on the unit ellipsoid in support-local space.
    float3 viewOriginLocal = cameraGaussian.position;
    float viewDistance = length(viewOriginLocal);
    if (viewDistance <= 1.0 + ELLIPSE_EPS) return false;
    float3 viewDirLocal = viewOriginLocal / viewDistance;
    float3 tangentCircleCenter = viewDirLocal / viewDistance;
    float tangentCircleRadius = sqrt(max(1.0 - 1.0 / (viewDistance * viewDistance), 0.0));
    float3 tangentAxis = abs(viewDirLocal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    float3 tangentBasisU = normalize_safe(cross(tangentAxis, viewDirLocal));
    float3 tangentBasisV = cross(viewDirLocal, tangentBasisU);
    float3 supportAxisX = camera.world_to_camera(rotate_by_quat(float3(cameraGaussian.logScale.x, 0.0, 0.0), invRotation));
    float3 supportAxisY = camera.world_to_camera(rotate_by_quat(float3(0.0, cameraGaussian.logScale.y, 0.0), invRotation));
    float3 supportAxisZ = camera.world_to_camera(rotate_by_quat(float3(0.0, 0.0, cameraGaussian.logScale.z), invRotation));

    float2 outlinePoints[ELLIPSE_POINT_COUNT];
    float2 outlineMin = float2(1e30, 1e30);
    float2 outlineMax = float2(-1e30, -1e30);

    [unroll] for (uint row = 0u; row < ELLIPSE_POINT_COUNT; row++) {
        float theta = TAU_RADIANS * (float(row) / float(ELLIPSE_POINT_COUNT));
        float s, c;
        sincos(theta, s, c);
        float3 supportPointLocal = tangentCircleCenter + tangentCircleRadius * (c * tangentBasisU + s * tangentBasisV);
        float3 cameraPoint = cameraCenter
            + supportPointLocal.x * supportAxisX
            + supportPointLocal.y * supportAxisY
            + supportPointLocal.z * supportAxisZ;
        float2 screenPoint = 0.0;
        if (!camera.project_camera_to_screen(cameraPoint, screenPoint)) return false;
        outlinePoints[row] = screenPoint;
        outlineMin = min(outlineMin, screenPoint);
        outlineMax = max(outlineMax, screenPoint);
    }
    outlineMin = min(outlineMin, screenCenter);
    outlineMax = max(outlineMax, screenCenter);
    float2 bboxCenter = 0.5 * (outlineMin + outlineMax);
    float2 bboxHalfExtent = max(0.5 * (outlineMax - outlineMin), float2(ELLIPSE_EPS, ELLIPSE_EPS));

    if (outlineMax.x < 0.0 || outlineMin.x >= camera.viewport.x || outlineMax.y < 0.0 || outlineMin.y >= camera.viewport.y) return false;

    // Solve for the ellipse conic coefficients in normalized space, then convert back to pixel space.
    float2 normPts[ELLIPSE_POINT_COUNT];
    [unroll] for (uint row = 0u; row < ELLIPSE_POINT_COUNT; row++)
        normPts[row] = (outlinePoints[row] - bboxCenter) / bboxHalfExtent;

    float solution[ELLIPSE_POINT_COUNT];
    if (!solve_conic_renorm(normPts, ELLIPSE_EPS, solution)) return false;
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
    ellipse.bboxExtent = bboxHalfExtent;
    return true;
}

[Differentiable]
float2 raster_eval_alpha(Gaussian3D gaussian, no_diff float3 rayDir) {
    float3 rayLocal = rotate_by_quat(rayDir, gaussian.rotation) / max(gaussian.logScale, float3(SMALL_VALUE));
    float denom = dot(rayLocal, rayLocal);
    if (denom <= RAY_DENOMINATOR_FLOOR) return 0.0;
    float tClosest = -dot(rayLocal, gaussian.position) / denom;
    if (tClosest <= 0.0) return 0.0;
    float3 closest = gaussian.position + rayLocal * tClosest;
    float rho2 = max(dot(closest, closest), 0.0);
    float alpha = gaussian.colorAlpha.w * exp(-0.5 * GAUSSIAN_SUPPORT_SIGMA_RADIUS * GAUSSIAN_SUPPORT_SIGMA_RADIUS * rho2);
    return float2(alpha, tClosest);
}

[Differentiable]
float raster_eval_alpha_only(Gaussian3D gaussian, no_diff float3 rayDir) {
    return raster_eval_alpha(gaussian, rayDir).x;
}

[Differentiable]
float raster_eval_depth_only(Gaussian3D gaussian, no_diff float3 rayDir) {
    return raster_eval_alpha(gaussian, rayDir).y;
}

[Differentiable]
float4 raster_output(float3 accum, float trans, no_diff float3 background) {
    return float4(pow(max(accum + trans * background, 0.0), OUTPUT_GAMMA), 1.0 - trans);
}
