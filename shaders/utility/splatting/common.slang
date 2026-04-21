import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/math/quaternion.slang";
import "utility/random/random.slang";

static const uint ELLIPSE_POINT_COUNT = 5u;
static const float GAUSSIAN_SUPPORT_SIGMA_RADIUS = 3.0;
static const uint DEBUG_MODE_NORMAL = 0u;
static const uint DEBUG_MODE_PROCESSED_COUNT = 1u;
static const uint DEBUG_MODE_SPLAT_AGE = 2u;
static const uint DEBUG_MODE_DEPTH_MEAN = 3u;
static const uint DEBUG_MODE_DEPTH_STD = 4u;
static const uint DEBUG_MODE_DEPTH_LOCAL_MISMATCH = 5u;
static const uint DEBUG_MODE_ELLIPSE_OUTLINES = 6u;
static const uint DEBUG_MODE_SPLAT_SPATIAL_DENSITY = 7u;
static const uint DEBUG_MODE_SPLAT_SCREEN_DENSITY = 8u;
static const uint DEBUG_MODE_GRAD_NORM = 9u;
static const uint DEBUG_MODE_SPLAT_DENSITY = 10u;
static const uint DEBUG_MODE_CONTRIBUTION_AMOUNT = 11u;
static const uint DEBUG_MODE_ADAM_MOMENTUM = 12u;
static const uint DEBUG_MODE_SH_VIEW_DEPENDENT = 13u;
static const uint DEBUG_MODE_SH_COEFFICIENT = 14u;
static const uint DEBUG_MODE_BLACK_NEGATIVE = 15u;
static const float DEBUG_CONTRIBUTION_AMOUNT_FLOOR = 1e-6;
static const uint DEBUG_SH_COEFF_COUNT = 16u;
static const float SH_C0 = 0.28209479177387814;
static const float SH_C1 = 0.4886025119029199;
static const float SH_C2_0 = 1.0925484305920792;
static const float SH_C2_1 = -1.0925484305920792;
static const float SH_C2_2 = 0.31539156525252005;
static const float SH_C2_3 = -1.0925484305920792;
static const float SH_C2_4 = 0.5462742152960396;
static const float SH_C3_0 = -0.5900435899266435;
static const float SH_C3_1 = 2.890611442640554;
static const float SH_C3_2 = -0.4570457994644658;
static const float SH_C3_3 = 0.3731763325901154;
static const float SH_C3_4 = -0.4570457994644658;
static const float SH_C3_5 = 1.445305721320277;
static const float SH_C3_6 = -0.5900435899266435;
static const uint TRAIN_SUBSAMPLE_HASH_STEP = 0x9e3779b9u;
static const uint TRAIN_SUBSAMPLE_HASH_FRAME = 0x85ebca6bu;
static const uint TRAIN_SUBSAMPLE_HASH_X = 0xc2b2ae35u;
static const uint TRAIN_SUBSAMPLE_HASH_Y = 0x27d4eb2fu;
static const uint TRAIN_SUBSAMPLE_HASH_FACTOR = 0x165667b1u;

struct TrainingSubsampleParams
{
    uint enabled;
    uint factor;
    uint nativeWidth;
    uint nativeHeight;
    uint frameIndex;
    uint stepIndex;
};

uint training_subsample_hash(uint2 lowResPixel, TrainingSubsampleParams params)
{
    uint seed = hash_u32(params.stepIndex + TRAIN_SUBSAMPLE_HASH_STEP);
    seed = hash_u32(seed ^ ((params.frameIndex + 1u) * TRAIN_SUBSAMPLE_HASH_FRAME));
    seed = hash_u32(seed ^ ((lowResPixel.x + 1u) * TRAIN_SUBSAMPLE_HASH_X));
    seed = hash_u32(seed ^ ((lowResPixel.y + 1u) * TRAIN_SUBSAMPLE_HASH_Y));
    return hash_u32(seed ^ (max(params.factor, 1u) * TRAIN_SUBSAMPLE_HASH_FACTOR));
}

uint2 training_sample_native_pixel(uint2 lowResPixel, TrainingSubsampleParams params)
{
    uint stride = max(params.factor, 1u);
    uint2 basePixel = lowResPixel * stride;
    if (params.enabled == 0u) return basePixel;

    uint blockWidth = max(min(stride, params.nativeWidth - min(basePixel.x, params.nativeWidth - 1u)), 1u);
    uint blockHeight = max(min(stride, params.nativeHeight - min(basePixel.y, params.nativeHeight - 1u)), 1u);
    uint seed = training_subsample_hash(lowResPixel, params);
    uint offsetX = seed % blockWidth;
    uint offsetY = hash_u32(seed ^ TRAIN_SUBSAMPLE_HASH_FACTOR) % blockHeight;
    return uint2(min(basePixel.x + offsetX, params.nativeWidth - 1u), min(basePixel.y + offsetY, params.nativeHeight - 1u));
}

[Differentiable]
float3 normalize_vector_safe(float3 value)
{
    return value * rsqrt(max(dot(value, value), SMALL_VALUE));
}

[Differentiable]
float3 evaluate_sh0_color_raw(float3 sh0)
{
    return 0.5 + SH_C0 * sh0;
}

[Differentiable]
float3 evaluate_sh_band_raw(
    float3 sh0,
    float3 sh1X,
    float3 sh1Y,
    float3 sh1Z,
    float3 sh2_0,
    float3 sh2_1,
    float3 sh2_2,
    float3 sh2_3,
    float3 sh2_4,
    float3 sh3_0,
    float3 sh3_1,
    float3 sh3_2,
    float3 sh3_3,
    float3 sh3_4,
    float3 sh3_5,
    float3 sh3_6,
    no_diff uint shBand,
    float3 viewDir
)
{
    if (shBand == 0u) return evaluate_sh0_color_raw(sh0);
    float3 dir = normalize_vector_safe(viewDir);
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float yz = y * z;
    float xz = x * z;
    float3 color = 0.5 + SH_C0 * sh0;
    if (shBand >= 1u)
    {
        color += -SH_C1 * y * sh1X
            + SH_C1 * z * sh1Y
            - SH_C1 * x * sh1Z;
    }
    if (shBand >= 2u)
    {
        color += SH_C2_0 * xy * sh2_0
            + SH_C2_1 * yz * sh2_1
            + SH_C2_2 * (2.0 * zz - xx - yy) * sh2_2
            + SH_C2_3 * xz * sh2_3
            + SH_C2_4 * (xx - yy) * sh2_4;
    }
    if (shBand >= 3u)
    {
        color += SH_C3_0 * y * (3.0 * xx - yy) * sh3_0
            + SH_C3_1 * xy * z * sh3_1
            + SH_C3_2 * y * (4.0 * zz - xx - yy) * sh3_2
            + SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh3_3
            + SH_C3_4 * x * (4.0 * zz - xx - yy) * sh3_4
            + SH_C3_5 * z * (xx - yy) * sh3_5
            + SH_C3_6 * x * (xx - 3.0 * yy) * sh3_6;
    }
    return color;
}

float3 evaluate_sh3_color(
    float3 sh0,
    float3 sh1X,
    float3 sh1Y,
    float3 sh1Z,
    float3 sh2_0,
    float3 sh2_1,
    float3 sh2_2,
    float3 sh2_3,
    float3 sh2_4,
    float3 sh3_0,
    float3 sh3_1,
    float3 sh3_2,
    float3 sh3_3,
    float3 sh3_4,
    float3 sh3_5,
    float3 sh3_6,
    float3 viewDir
)
{
    return evaluate_sh_band_raw(sh0, sh1X, sh1Y, sh1Z, sh2_0, sh2_1, sh2_2, sh2_3, sh2_4, sh3_0, sh3_1, sh3_2, sh3_3, sh3_4, sh3_5, sh3_6, 3u, viewDir);
}

[Differentiable]
float3 evaluate_gaussian_color(
    float3 sh0,
    float3 sh1X,
    float3 sh1Y,
    float3 sh1Z,
    float3 sh2_0,
    float3 sh2_1,
    float3 sh2_2,
    float3 sh2_3,
    float3 sh2_4,
    float3 sh3_0,
    float3 sh3_1,
    float3 sh3_2,
    float3 sh3_3,
    float3 sh3_4,
    float3 sh3_5,
    float3 sh3_6,
    no_diff float3 cameraPosition,
    float3 position,
    no_diff uint shBand
)
{
    return evaluate_sh_band_raw(sh0, sh1X, sh1Y, sh1Z, sh2_0, sh2_1, sh2_2, sh2_3, sh2_4, sh3_0, sh3_1, sh3_2, sh3_3, sh3_4, sh3_5, sh3_6, shBand, position - cameraPosition);
}

struct ScanlineWorkItem
{
    uint splatId;
    uint lineCoord;
    uint lineStartMinor;
    uint packedLineCountAxis;

    [mutating]
    void clear(bool scanAlongX)
    {
        lineStartMinor = 0u;
        packedLineCountAxis = scanAlongX ? 0x80000000u : 0u;
    }

    [mutating]
    void setSpan(uint lineStartMinorIn, uint lineCountIn, bool scanAlongX)
    {
        lineStartMinor = lineStartMinorIn;
        packedLineCountAxis = (lineCountIn & 0x7fffffffu) | (scanAlongX ? 0x80000000u : 0u);
    }

    uint lineCount()
    {
        return packedLineCountAxis & 0x7fffffffu;
    }

    bool scanAlongXAxis()
    {
        return (packedLineCountAxis & 0x80000000u) != 0u;
    }
};

struct PrepassParams
{
    uint splatCount;
    uint tileSize;
    uint tileWidth;
    uint tileHeight;
    uint tileCount;
    uint depthBits;
    uint sortedCountOffset;
    uint maxListEntries;
    uint maxScanlineEntries;
    float radiusScale;
};

struct RasterParams
{
    uint width;
    uint height;
    uint maxSplatSteps;
    float alphaCutoff;
    float transmittanceThreshold;
    float3 background;
    uint trainingBackgroundMode;
    uint trainingBackgroundSeed;
    uint debugMode;
    uint shBand;
    float debugGradNormThreshold;
    float debugEllipseThicknessPx;
    float2 debugSplatAgeRange;
    float2 debugDensityRange;
    float2 debugContributionRange;
    float debugContributionPercentScale;
    float2 debugAdamMomentumRange;
    float2 debugDepthMeanRange;
    float2 debugDepthStdRange;
    float2 debugDepthLocalMismatchRange;
    float debugDepthLocalMismatchSmoothRadius;
    float debugDepthLocalMismatchRejectRadius;
    uint debugSHCoeffIndex;
    float debugGaussianScaleMultiplier;
    float debugMinOpacity;
    float debugOpacityMultiplier;
    float debugEllipseScaleMultiplier;
};

float3 gaussian_sh_coeff(Gaussian3D gaussian, uint coeffIndex)
{
    switch (coeffIndex)
    {
    case 0u: return gaussian.sh0;
    case 1u: return gaussian.sh1X;
    case 2u: return gaussian.sh1Y;
    case 3u: return gaussian.sh1Z;
    case 4u: return gaussian.sh2_0;
    case 5u: return gaussian.sh2_1;
    case 6u: return gaussian.sh2_2;
    case 7u: return gaussian.sh2_3;
    case 8u: return gaussian.sh2_4;
    case 9u: return gaussian.sh3_0;
    case 10u: return gaussian.sh3_1;
    case 11u: return gaussian.sh3_2;
    case 12u: return gaussian.sh3_3;
    case 13u: return gaussian.sh3_4;
    case 14u: return gaussian.sh3_5;
    default: return gaussian.sh3_6;
    }
}

float3 gaussian_sh_view_dependent_component(Gaussian3D gaussian, float3 cameraPosition)
{
    return evaluate_sh_band_raw(
        gaussian.sh0,
        gaussian.sh1X,
        gaussian.sh1Y,
        gaussian.sh1Z,
        gaussian.sh2_0,
        gaussian.sh2_1,
        gaussian.sh2_2,
        gaussian.sh2_3,
        gaussian.sh2_4,
        gaussian.sh3_0,
        gaussian.sh3_1,
        gaussian.sh3_2,
        gaussian.sh3_3,
        gaussian.sh3_4,
        gaussian.sh3_5,
        gaussian.sh3_6,
        3u,
        gaussian.position - cameraPosition
    ) - evaluate_sh0_color_raw(gaussian.sh0);
}

struct RasterGaussian : IDifferentiable
{
    float3 roLocal;
    float3 scale;
    float4 rotation;
    float4 colorAlpha;
};

struct FixedRasterGaussianGrad
{
    int3 roLocal;
    int3 scale;
    int4 rotation;
    int4 colorAlpha;
};

struct Gaussian3D : IDifferentiable
{
    float3 position;
    float3 logScale;
    float4 rotation;
    float3 sh0;
    float3 sh1X;
    float3 sh1Y;
    float3 sh1Z;
    float3 sh2_0;
    float3 sh2_1;
    float3 sh2_2;
    float3 sh2_3;
    float3 sh2_4;
    float3 sh3_0;
    float3 sh3_1;
    float3 sh3_2;
    float3 sh3_3;
    float3 sh3_4;
    float3 sh3_5;
    float3 sh3_6;
    float rawOpacity;
};

struct Ellipsoid : IDifferentiable
{
    float3 centerWorld;
    float3 rasterScale;
    float3 rasterInvScale;
    float3 projectedScale;
    float3 projectedInvScale;
    float4 rotation;
    float3 rasterRoLocal;
    float3 projectedRoLocal;
    float opacityScale;
};

struct Ellipse
{
    float2 centerPx;
    float radiusPx;
    float3 conic;
    float2 bboxExtent;

    [mutating]
    void init(float2 centerPxIn, float radiusPxIn, float3 conicIn)
    {
        centerPx = centerPxIn;
        radiusPx = radiusPxIn;
        conic = conicIn;
        bboxExtent = float2(radiusPxIn, radiusPxIn);
    }

    bool isVisible(float camDistance, float depthValue, float2 viewport, float nearDepth, float farDepth)
    {
        return (centerPx.x + radiusPx) >= 0.0
            && (centerPx.x - radiusPx) < viewport.x
            && (centerPx.y + radiusPx) >= 0.0
            && (centerPx.y - radiusPx) < viewport.y;
    }

    float clippedVerticalLength(float x, float2 viewport, float det)
    {
        float dx = x - centerPx.x;
        float root = conic.z - det * dx * dx;
        if (!(root > 0.0)) return 0.0;
        float midY = centerPx.y - conic.y * dx / conic.z;
        float halfLength = sqrt(root) / conic.z;
        return max(min(midY + halfLength, viewport.y) - max(midY - halfLength, 0.0), 0.0);
    }

    float clippedHorizontalLength(float y, float2 viewport, float det)
    {
        float dy = y - centerPx.y;
        float root = conic.x - det * dy * dy;
        if (!(root > 0.0)) return 0.0;
        float midX = centerPx.x - conic.y * dy / conic.x;
        float halfLength = sqrt(root) / conic.x;
        return max(min(midX + halfLength, viewport.x) - max(midX - halfLength, 0.0), 0.0);
    }

    float clippedAreaPx(float2 viewport)
    {
        static const uint CLIPPED_AREA_SAMPLE_COUNT = 32u;
        float det = conic.x * conic.z - conic.y * conic.y;
        if (!all(isfinite(conic)) || !(conic.x > 0.0) || !(conic.z > 0.0) || !(det > CONIC_DETERMINANT_FLOOR)) return 0.0;
        float2 extent = sqrt(float2(conic.z, conic.x) / det);
        if (!all(isfinite(extent))) return 0.0;
        bool integrateX = extent.x <= extent.y;
        float2 bounds = integrateX
            ? float2(max(centerPx.x - extent.x, 0.0), min(centerPx.x + extent.x, viewport.x))
            : float2(max(centerPx.y - extent.y, 0.0), min(centerPx.y + extent.y, viewport.y));
        if (!(bounds.y > bounds.x)) return 0.0;
        float stepSize = (bounds.y - bounds.x) / float(CLIPPED_AREA_SAMPLE_COUNT);
        float area = 0.0;
        [unroll] for (uint i = 0u; i < CLIPPED_AREA_SAMPLE_COUNT; i++)
        {
            float sampleCoord = bounds.x + (float(i) + 0.5) * stepSize;
            area += integrateX ? clippedVerticalLength(sampleCoord, viewport, det) : clippedHorizontalLength(sampleCoord, viewport, det);
        }
        return area * stepSize;
    }

    bool chooseScanAlongX(uint spanX, uint spanY)
    {
        bool scanAlongX = bboxExtent.x >= bboxExtent.y;
        if (abs(bboxExtent.x - bboxExtent.y) < SMALL_VALUE) scanAlongX = spanX >= spanY;
        return scanAlongX;
    }

    float evalConic(float2 d)
    {
        return dot(d, mul(float2x2(conic.x, conic.y, conic.y, conic.z), d));
    }

    float minConicOverTileBox(float2 boxMin, float2 boxMax)
    {
        float minVal = min(
            min(evalConic(float2(boxMin.x, boxMin.y)), evalConic(float2(boxMax.x, boxMin.y))),
            min(evalConic(float2(boxMin.x, boxMax.y)), evalConic(float2(boxMax.x, boxMax.y)))
        );
        if (boxMin.x <= 0.0 && 0.0 <= boxMax.x && boxMin.y <= 0.0 && 0.0 <= boxMax.y) return 0.0;

        if (conic.z > CONIC_DETERMINANT_FLOOR)
        {
            float yAtMinX = clamp(-conic.y * boxMin.x / conic.z, boxMin.y, boxMax.y);
            float yAtMaxX = clamp(-conic.y * boxMax.x / conic.z, boxMin.y, boxMax.y);
            minVal = min(minVal, evalConic(float2(boxMin.x, yAtMinX)));
            minVal = min(minVal, evalConic(float2(boxMax.x, yAtMaxX)));
        }
        if (conic.x > CONIC_DETERMINANT_FLOOR)
        {
            float xAtMinY = clamp(-conic.y * boxMin.y / conic.x, boxMin.x, boxMax.x);
            float xAtMaxY = clamp(-conic.y * boxMax.y / conic.x, boxMin.x, boxMax.x);
            minVal = min(minVal, evalConic(float2(xAtMinY, boxMin.y)));
            minVal = min(minVal, evalConic(float2(xAtMaxY, boxMax.y)));
        }
        return minVal;
    }

    bool intersectsTileBox(float2 boxMin, float2 boxMax)
    {
        return minConicOverTileBox(boxMin, boxMax) <= 1.0 + CONIC_INTERSECTION_SLACK;
    }
};

struct ScanlineLoopConfig
{
    bool scanAlongX;
    int lineStart;
    int lineEnd;
    int minorMin;
    int minorMax;
    uint minorLimit;
    uint lineLimit;

    uint lineCoordToUint(int lineCoord)
    {
        return uint(lineCoord);
    }

    bool isLineCoordValid(uint lineCoord)
    {
        return lineCoord < lineLimit;
    }

    bool isMinorCoordValid(uint minorCoord)
    {
        return minorCoord < minorLimit;
    }

    uint composeTileId(uint lineCoord, uint minorCoord, uint tileWidth)
    {
        return scanAlongX ? (lineCoord * tileWidth + minorCoord) : (minorCoord * tileWidth + lineCoord);
    }
};

struct ScanlineSpan
{
    uint lineStartMinor;
    uint lineCount;
};

struct SplatBinRequest
{
    uint splatId;
    float camDistance;
    float depthValue;
    uint depthCode;
    Ellipse ellipse;
};

uint quantize_depth(CameraParams cameraParams, PrepassParams prepass, float depthValue)
{
    float denom = max(cameraParams.farDepth - cameraParams.nearDepth, SMALL_VALUE);
    float t = saturate((depthValue - cameraParams.nearDepth) / denom);
    uint maxValue = (1u << prepass.depthBits) - 1u;
    return min((uint)round(t * float(maxValue)), maxValue);
}

float3 local_point_to_world(float3 pLocal, float3 worldPos, float3 invScale, float4 q)
{
    float3 invScaleSafe = max(invScale, float3(SMALL_VALUE));
    float3 localUnscaled = pLocal / invScaleSafe;
    float4 qInv = float4(q.x, -q.y, -q.z, -q.w);
    return worldPos + rotate_by_quat(localUnscaled, qInv);
}

float debug_normalize_range(float value, float2 range)
{
    return saturate((value - range.x) / max(range.y - range.x, SMALL_VALUE));
}

float debug_normalize_log_range(float value, float valueFloor, float valueCeil)
{
    float lo = max(valueFloor, SMALL_VALUE);
    float hi = max(valueCeil, lo * (1.0 + SMALL_VALUE));
    float logLo = log2(lo);
    float logHi = log2(hi);
    float logValue = log2(max(value, lo));
    return saturate((logValue - logLo) / max(logHi - logLo, SMALL_VALUE));
}

float gaussian_scale_volume(float3 logScale)
{
    float3 scale = exp(logScale);
    return max(scale.x * scale.y * scale.z, SMALL_VALUE);
}

float projected_ellipse_area(float3 conic)
{
    float det = conic.x * conic.z - conic.y * conic.y;
    return max(0.5 * TAU_RADIANS / sqrt(max(det, CONIC_DETERMINANT_FLOOR)), SMALL_VALUE);
}

[Differentiable]
float3 depth_state_step(float3 state, float contributionWeight, float depth)
{
    if (contributionWeight <= SMALL_VALUE || depth <= 0.0) return state;
    float totalWeight = state.x + contributionWeight;
    float delta = depth - state.y;
    float nextDepthMean = state.y + (contributionWeight / max(totalWeight, SMALL_VALUE)) * delta;
    float nextDepthM2 = state.z + state.x * contributionWeight * delta * delta / max(totalWeight, SMALL_VALUE);
    return float3(totalWeight, nextDepthMean, nextDepthM2);
}

float3 depth_state_rewind(float3 state, float contributionWeight, float depth)
{
    if (contributionWeight <= SMALL_VALUE || depth <= 0.0) return state;
    float prevWeightSum = max(state.x - contributionWeight, 0.0);
    if (prevWeightSum <= SMALL_VALUE) return 0.0;
    float prevDepthMean = (state.x * state.y - contributionWeight * depth) / prevWeightSum;
    float delta = depth - prevDepthMean;
    float prevDepthM2 = max(state.z - prevWeightSum * contributionWeight * delta * delta / max(state.x, SMALL_VALUE), 0.0);
    return float3(prevWeightSum, prevDepthMean, prevDepthM2);
}

[Differentiable]
float raster_depth_output(float3 state)
{
    if (state.x <= SMALL_VALUE) return 0.0;
    float depthStd = sqrt(max(state.z / max(state.x, SMALL_VALUE), SMALL_VALUE));
    return depthStd / max(abs(state.y), SMALL_VALUE);
}

float3 jet_colormap(float value)
{
    float x = saturate(value);
    float r = saturate(1.5 - abs(4.0 * x - 3.0));
    float g = saturate(1.5 - abs(4.0 * x - 2.0));
    float b = saturate(1.5 - abs(4.0 * x - 1.0));
    return float3(r, g, b);
}

[Differentiable]
Ellipsoid build_ellipsoid_with_support(no_diff CameraParams cameraParams, no_diff float radiusScale, no_diff float supportSigmaRadius, no_diff float maxAnisotropy, Gaussian3D gaussian)
{
    PinholeCamera camera = PinholeCamera.from_params(cameraParams);
    float3 sigma = exp(gaussian.logScale);
    float3 rawSupportScale = sigma * (radiusScale * supportSigmaRadius);
    float supportMaxScale = max(rawSupportScale.x, max(rawSupportScale.y, rawSupportScale.z));
    float supportMinScale = supportMaxScale / max(maxAnisotropy, 1.0);
    float3 supportScale = max(rawSupportScale, float3(supportMinScale));
    Ellipsoid ellipsoid;
    ellipsoid.centerWorld = gaussian.position;
    ellipsoid.rasterScale = supportScale;
    ellipsoid.rasterInvScale = 1.0 / max(ellipsoid.rasterScale, float3(SMALL_VALUE));
    ellipsoid.projectedScale = ellipsoid.rasterScale;
    ellipsoid.projectedInvScale = ellipsoid.rasterInvScale;
    ellipsoid.rotation = gaussian.rotation;
    float3 cameraOffset = rotate_by_quat(camera.position_world() - ellipsoid.centerWorld, ellipsoid.rotation);
    ellipsoid.rasterRoLocal = cameraOffset * ellipsoid.rasterInvScale;
    ellipsoid.projectedRoLocal = ellipsoid.rasterRoLocal;
    ellipsoid.opacityScale = 1.0;
    return ellipsoid;
}

[Differentiable]
Ellipsoid build_ellipsoid(no_diff CameraParams cameraParams, no_diff float radiusScale, no_diff float maxAnisotropy, Gaussian3D gaussian)
{
    return build_ellipsoid_with_support(cameraParams, radiusScale, GAUSSIAN_SUPPORT_SIGMA_RADIUS, maxAnisotropy, gaussian);
}
