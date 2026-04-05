import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/math/quaternion.slang";

static const uint ELLIPSE_POINT_COUNT = 5u;
static const float GAUSSIAN_SUPPORT_SIGMA_RADIUS = 3.0;
static const uint DEBUG_MODE_NORMAL = 0u;
static const uint DEBUG_MODE_PROCESSED_COUNT = 1u;
static const uint DEBUG_MODE_CLONE_COUNT = 2u;
static const uint DEBUG_MODE_DEPTH_MEAN = 3u;
static const uint DEBUG_MODE_DEPTH_STD = 4u;
static const uint DEBUG_MODE_ELLIPSE_OUTLINES = 5u;
static const uint DEBUG_MODE_SPLAT_SPATIAL_DENSITY = 6u;
static const uint DEBUG_MODE_SPLAT_SCREEN_DENSITY = 7u;
static const uint DEBUG_MODE_GRAD_NORM = 8u;
static const uint DEBUG_MODE_SPLAT_DENSITY = 9u;
static const uint DEBUG_MODE_CONTRIBUTION_AMOUNT = 10u;
static const float DEBUG_CONTRIBUTION_AMOUNT_FLOOR = 1e-6;
static const float SH_C0 = 0.28209479177387814;
static const float SH_C1 = 0.4886025119029199;

[Differentiable]
float3 normalize_vector_safe(float3 value)
{
    return value * rsqrt(max(dot(value, value), SMALL_VALUE));
}

[Differentiable]
float4 sh0_sh1_basis(float3 viewDir)
{
    float3 dir = normalize_vector_safe(viewDir);
    return float4(SH_C0, -SH_C1 * dir.y, SH_C1 * dir.z, -SH_C1 * dir.x);
}

[Differentiable]
float3 evaluate_sh0_color_raw(float3 sh0)
{
    return 0.5 + SH_C0 * sh0;
}

[Differentiable]
float3 evaluate_sh0_sh1_raw(float3 sh0, float3 sh1X, float3 sh1Y, float3 sh1Z, float3 viewDir)
{
    float4 basis = sh0_sh1_basis(viewDir);
    return 0.5 + basis.x * sh0 + basis.y * sh1X + basis.z * sh1Y + basis.w * sh1Z;
}

float3 evaluate_sh0_sh1_color(float3 sh0, float3 sh1X, float3 sh1Y, float3 sh1Z, float3 viewDir)
{
    return saturate(evaluate_sh0_sh1_raw(sh0, sh1X, sh1Y, sh1Z, viewDir));
}

[Differentiable]
float3 evaluate_gaussian_color(float3 sh0, float3 sh1X, float3 sh1Y, float3 sh1Z, no_diff float3 cameraPosition, float3 position, no_diff uint useSH)
{
    float3 rawColor = useSH != 0u
        ? evaluate_sh0_sh1_raw(sh0, sh1X, sh1Y, sh1Z, cameraPosition - position)
        : evaluate_sh0_color_raw(sh0);
    return saturate(rawColor);
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
    uint debugMode;
    uint useSH;
    float debugGradNormThreshold;
    float debugEllipseThicknessPx;
    float2 debugCloneCountRange;
    float2 debugDensityRange;
    float2 debugDepthMeanRange;
    float2 debugDepthStdRange;
};

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
    float3 rawSupportScale = max(sigma * (radiusScale * supportSigmaRadius), float3(SMALL_VALUE));
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
