import "camera.slang";
import "math_constants.slang";

static const uint UINT_MAX_VALUE = 0xFFFFFFFFu;
static const uint INVALID_START = UINT_MAX_VALUE;
static const uint PROJECT_GROUP_SIZE = 256u;
static const uint TILE_GROUP_SIZE = 256u;
static const uint RASTER_THREAD_TILE_DIM = 8u;
static const uint RASTER_MICROTILE_DIM = 3u;
static const uint RASTER_MICROTILE_PIXELS = RASTER_MICROTILE_DIM * RASTER_MICROTILE_DIM;
static const uint RASTER_GROUP_THREADS = RASTER_THREAD_TILE_DIM * RASTER_THREAD_TILE_DIM;
static const uint RASTER_EFFECTIVE_TILE_SIZE = RASTER_THREAD_TILE_DIM * RASTER_MICROTILE_DIM;
static const uint RASTER_BATCH = 64u;
static const uint SPLAT_GRAD_COMPONENTS = 4u;
static const uint SPLAT_GRAD_STATE_POS_LOCAL_OFFSET = 0u;
static const uint SPLAT_GRAD_STATE_INV_SCALE_OFFSET = SPLAT_GRAD_STATE_POS_LOCAL_OFFSET + SPLAT_GRAD_COMPONENTS;
static const uint SPLAT_GRAD_STATE_QUAT_OFFSET = SPLAT_GRAD_STATE_INV_SCALE_OFFSET + SPLAT_GRAD_COMPONENTS;
static const uint SPLAT_GRAD_STATE_COLOR_ALPHA_OFFSET = SPLAT_GRAD_STATE_QUAT_OFFSET + SPLAT_GRAD_COMPONENTS;
static const uint SPLAT_GRAD_STATE_STRIDE = SPLAT_GRAD_STATE_COLOR_ALPHA_OFFSET + SPLAT_GRAD_COMPONENTS;
static const uint MVEE_POINT_COUNT = 5u;
static const uint MVEE_MAX_ITERS = 8u;

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

groupshared float4 gsRoLocal[RASTER_BATCH];
groupshared half4 gsInvScale[RASTER_BATCH];
groupshared half4 gsQuat[RASTER_BATCH];
groupshared half4 gsColorAlpha[RASTER_BATCH];
groupshared uint gsBatchSplatId[RASTER_BATCH];
groupshared uint gsStoppedMicrotiles;
groupshared float gsGradState[RASTER_BATCH * SPLAT_GRAD_STATE_STRIDE];

uniform StructuredBuffer<float4> g_Positions;
uniform StructuredBuffer<float4> g_Scales;
uniform StructuredBuffer<float4> g_Rotations;
uniform StructuredBuffer<float4> g_ColorAlpha;

uniform RWStructuredBuffer<float4> g_ScreenCenterRadiusDepth;
uniform RWStructuredBuffer<float4> g_ScreenColorAlpha;
uniform RWStructuredBuffer<float4> g_ScreenEllipseConic;

uniform RWStructuredBuffer<uint> g_Keys;
uniform RWStructuredBuffer<uint> g_Values;
uniform RWStructuredBuffer<uint> g_ListCounter;
uniform RWStructuredBuffer<uint> g_SplatListBases;
uniform RWStructuredBuffer<ScanlineWorkItem> g_ScanlineWorkItems;
uniform RWStructuredBuffer<uint> g_ScanlineCounter;
uniform RWStructuredBuffer<uint2> g_TileRanges;

uniform StructuredBuffer<uint> g_SortedKeys;
uniform StructuredBuffer<uint> g_SortedValues;
uniform StructuredBuffer<uint> g_PrepassParams;

uniform RWTexture2D<float4> g_Output;
uniform Texture2D<float4> g_OutputGrad;

uniform RWStructuredBuffer<float> g_GradPositions;
uniform RWStructuredBuffer<float> g_GradScales;
uniform RWStructuredBuffer<float> g_GradRotations;
uniform RWStructuredBuffer<float> g_GradColorAlpha;

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
    float debugEllipseThicknessPx;
    float3 debugEllipseColor;
};

uniform CameraParams g_Camera;
uniform PrepassParams g_Prepass;
uniform RasterParams g_Raster;

struct Gaussian3D : IDifferentiable
{
    float3 position;
    float3 scale;
    float4 rotation;
    float4 colorAlpha;

    static Gaussian3D load(uint splatId)
    {
        Gaussian3D g;
        g.position = g_Positions[splatId].xyz;
        g.scale = g_Scales[splatId].xyz;
        g.rotation = g_Rotations[splatId];
        g.colorAlpha = g_ColorAlpha[splatId];
        return g;
    }
};

struct Ellipsoid : IDifferentiable
{
    float3 centerWorld;
    float3 scale;
    float3 invScale;
    float4 rotation;
    float3 roLocal;
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

[Differentiable]
float3 rotate_by_quat(float3 v, float4 q)
{
    float3 qv = q.yzw;
    return v + 2.0 * cross(cross(v, qv) + q.x * v, qv);
}

uint quantizeDepth(float depthValue)
{
    float denom = max(g_Camera.farDepth - g_Camera.nearDepth, SMALL_VALUE);
    float t = saturate((depthValue - g_Camera.nearDepth) / denom);
    uint maxValue = (1u << g_Prepass.depthBits) - 1u;
    return min((uint)round(t * float(maxValue)), maxValue);
}

bool inverse3x3(float3x3 m, float eps, out float3x3 invOut)
{
    float det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
              - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
              + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if (abs(det) <= eps)
    {
        invOut = 0.0;
        return false;
    }
    float invDet = 1.0 / det;
    invOut[0] = float3(
        m[1][1] * m[2][2] - m[1][2] * m[2][1],
        m[0][2] * m[2][1] - m[0][1] * m[2][2],
        m[0][1] * m[1][2] - m[0][2] * m[1][1]
    ) * invDet;
    invOut[1] = float3(
        m[1][2] * m[2][0] - m[1][0] * m[2][2],
        m[0][0] * m[2][2] - m[0][2] * m[2][0],
        m[0][2] * m[1][0] - m[0][0] * m[1][2]
    ) * invDet;
    invOut[2] = float3(
        m[1][0] * m[2][1] - m[1][1] * m[2][0],
        m[0][1] * m[2][0] - m[0][0] * m[2][1],
        m[0][0] * m[1][1] - m[0][1] * m[1][0]
    ) * invDet;
    return true;
}

float3 local_point_to_world(float3 pLocal, float3 worldPos, float3 invScale, float4 q)
{
    float3 invScaleSafe = max(invScale, float3(SMALL_VALUE));
    float3 localUnscaled = pLocal / invScaleSafe;
    float4 qInv = float4(q.x, -q.y, -q.z, -q.w);
    return worldPos + rotate_by_quat(localUnscaled, qInv);
}

[Differentiable]
Ellipsoid build_ellipsoid(Gaussian3D gaussian)
{
    PinholeCamera camera = PinholeCamera.from_params(g_Camera);
    Ellipsoid ellipsoid;
    ellipsoid.centerWorld = gaussian.position;
    ellipsoid.scale = gaussian.scale * g_Prepass.radiusScale;
    ellipsoid.invScale = 1.0 / max(ellipsoid.scale, float3(SMALL_VALUE));
    ellipsoid.rotation = gaussian.rotation;
    ellipsoid.roLocal = rotate_by_quat(camera.position_world() - ellipsoid.centerWorld, ellipsoid.rotation) * ellipsoid.invScale;
    return ellipsoid;
}
