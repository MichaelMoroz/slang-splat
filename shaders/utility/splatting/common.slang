import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/math/quaternion.slang";

static const uint ELLIPSE_POINT_COUNT = 5u;
static const float GAUSSIAN_SUPPORT_SIGMA_RADIUS = 3.0;

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
    uint debugShowEllipses;
    uint debugShowProcessedCount;
    uint debugShowGradNorm;
    float debugGradNormThreshold;
    float debugEllipseThicknessPx;
};

struct Gaussian3D : IDifferentiable
{
    float3 position;
    float3 logScale;
    float4 rotation;
    float4 colorAlpha;
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
        return depthValue > DEPTH_VISIBILITY_FLOOR
            && (centerPx.x + radiusPx) >= 0.0
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

[Differentiable]
Ellipsoid build_ellipsoid_with_support(no_diff CameraParams cameraParams, no_diff float radiusScale, no_diff float supportSigmaRadius, Gaussian3D gaussian)
{
    PinholeCamera camera = PinholeCamera.from_params(cameraParams);
    float3 sigma = exp(gaussian.logScale);
    float3 rawSupportScale = max(sigma * (radiusScale * supportSigmaRadius), float3(SMALL_VALUE));
    Ellipsoid ellipsoid;
    ellipsoid.centerWorld = gaussian.position;
    ellipsoid.rasterScale = rawSupportScale;
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
Ellipsoid build_ellipsoid(no_diff CameraParams cameraParams, no_diff float radiusScale, Gaussian3D gaussian)
{
    return build_ellipsoid_with_support(cameraParams, radiusScale, GAUSSIAN_SUPPORT_SIGMA_RADIUS, gaussian);
}
