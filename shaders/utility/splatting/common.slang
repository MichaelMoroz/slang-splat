import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/math/matrix.slang";
import "utility/math/quaternion.slang";

static const uint MVEE_POINT_COUNT = 5u;
static const uint MVEE_MAX_ITERS = 8u;
static const float SCALE_AREA_PROXY_POWER = 2.0 / 3.0;
static const float SCALE_FLOOR_SMOOTH_RATIO = 1.0;

struct ScanlineWorkItem
{
    uint splatId;
    uint depthCode;
    uint lineCoord;
    uint lineStartMinor;
    uint lineCount;
    uint splatListOffset;
    uint scanAlongX;
    uint reserved0;
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
    uint sampled5MVEEIters;
    float sampled5SafetyScale;
    float sampled5RadiusPadPx;
    float sampled5Eps;
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
    float3 debugEllipseColor;
};

struct Gaussian3D : IDifferentiable
{
    float3 position;
    float3 scale;
    float4 rotation;
    float4 colorAlpha;
};

struct Ellipsoid : IDifferentiable
{
    float3 centerWorld;
    float3 scale;
    float3 invScale;
    float4 rotation;
    float3 roLocal;
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
            && camDistance > nearDepth
            && camDistance < farDepth
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

struct MveeFitResult
{
    float2 centerPx;
    float radiusPx;
    float3 conic;

    [mutating]
    void clear()
    {
        centerPx = 0.0;
        radiusPx = 0.0;
        conic = 0.0;
    }
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

float minimum_pixel_scale_world(CameraParams cameraParams, no_diff float3 centerWorld)
{
    PinholeCamera camera = PinholeCamera.from_params(cameraParams);
    float centerDepth = camera.camera_depth(centerWorld);
    return camera.pixel_world_size_max(centerDepth);
}

[Differentiable]
float scale_area_proxy(float3 scale)
{
    return pow(max(scale.x * scale.y * scale.z, SMALL_VALUE), SCALE_AREA_PROXY_POWER);
}

[Differentiable]
float3 smooth_max_scale(float3 rawScale, float3 minScale)
{
    float3 smoothness = max(minScale * SCALE_FLOOR_SMOOTH_RATIO, float3(SMALL_VALUE));
    float3 delta = rawScale - minScale;
    return 0.5 * (rawScale + minScale + sqrt(delta * delta + smoothness * smoothness));
}

[Differentiable]
float opacity_scale_from_clamp(float3 rawScale, float3 clampedScale)
{
    float rawArea = scale_area_proxy(rawScale);
    float clampedArea = scale_area_proxy(clampedScale);
    return min(rawArea / max(clampedArea, SMALL_VALUE), 1.0);
}

[Differentiable]
Ellipsoid build_ellipsoid(no_diff CameraParams cameraParams, no_diff float radiusScale, Gaussian3D gaussian)
{
    PinholeCamera camera = PinholeCamera.from_params(cameraParams);
    float minScaleWorld = minimum_pixel_scale_world(cameraParams, gaussian.position);
    float3 rawScale = max(gaussian.scale * radiusScale, float3(SMALL_VALUE));
    float3 clampedScale = smooth_max_scale(rawScale, float3(minScaleWorld));
    Ellipsoid ellipsoid;
    ellipsoid.centerWorld = gaussian.position;
    ellipsoid.scale = clampedScale;
    ellipsoid.invScale = 1.0 / max(ellipsoid.scale, float3(SMALL_VALUE));
    ellipsoid.rotation = gaussian.rotation;
    ellipsoid.roLocal = rotate_by_quat(camera.position_world() - ellipsoid.centerWorld, ellipsoid.rotation) * ellipsoid.invScale;
    ellipsoid.opacityScale = opacity_scale_from_clamp(rawScale, clampedScale);
    return ellipsoid;
}
