static const uint UINT_MAX_VALUE = 0xFFFFFFFFu;
static const uint INVALID_START = UINT_MAX_VALUE;
static const uint PROJECT_GROUP_SIZE = 256u;
static const uint TILE_GROUP_SIZE = 256u;
static const uint RASTER_TILE_DIM = 16u;
static const uint RASTER_BATCH = 256u;
static const uint MVEE_POINT_COUNT = 5u;
static const uint MVEE_MAX_ITERS = 8u;
static const float MVEE_FULL_TURN_RADIANS = 6.28318530718;
static const float MVEE_SOUTH_POLE_THRESHOLD = -0.9999999;
static const float MVEE_AUGMENTED_DIMENSION = 3.0;
static const float GAMMA_LINEAR_TO_OUTPUT = 2.2;
static const float EPS_DEPTH_VISIBILITY = 1e-4;
static const float EPS_SMALL = 1e-6;
static const float EPS_RAY_DENOM = 1e-10;
static const float EPS_CONIC_POSITIVE = 1e-10;
static const float EPS_CONIC_DETERMINANT = 1e-12;
static const float EPS_CONIC_INV_AXIS = 1e-20;
static const float EPS_CONIC_INTERSECTION = 1e-5;
static const float MIN_AXIS_RATIO = 0.125;
static const float MIN_RADIUS_PX = 1.0;
static const float MIN_BBOX_EXTENT_PX = 1e-4;
static const float MIN_DEBUG_THICKNESS_PX = 0.25;
static const float EPS_GRADIENT_LENGTH = 1e-4;
static const float SCALE_FLOOR_RATIO = 0.05;

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
groupshared uint gsStoppedPixels;

uniform StructuredBuffer<float4> g_Positions;
uniform StructuredBuffer<float4> g_Scales;
uniform StructuredBuffer<float4> g_Rotations;
uniform StructuredBuffer<float4> g_ColorAlpha;

uniform RWStructuredBuffer<float4> g_ScreenCenterRadiusDepth;
uniform RWStructuredBuffer<float4> g_ScreenColorAlpha;
uniform RWStructuredBuffer<float4> g_ScreenEllipseConic;
uniform RWStructuredBuffer<float4> g_SplatPosLocal;
uniform RWStructuredBuffer<float4> g_SplatInvScale;
uniform RWStructuredBuffer<float4> g_SplatQuat;

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

uniform uint g_splatCount;
uniform uint g_tileSize;
uniform uint g_tileWidth;
uniform uint g_tileHeight;
uniform uint g_tileCount;
uniform uint g_depthBits;
uniform uint g_sortedCountOffset;
uniform uint g_maxListEntries;
uniform uint g_maxScanlineEntries;
uniform uint g_width;
uniform uint g_height;
uniform uint g_maxSplatSteps;

uniform float2 g_viewport;
uniform float3 g_camPos;
uniform float3 g_camRight;
uniform float3 g_camUp;
uniform float3 g_camForward;
uniform float3 g_background;

uniform float g_focalPixels;
uniform float g_nearDepth;
uniform float g_farDepth;
uniform float g_radiusScale;
uniform float g_maxSplatRadiusPx;
uniform float g_alphaCutoff;
uniform float g_transmittanceThreshold;
uniform uint g_sampled5MVEEIters;
uniform float g_sampled5SafetyScale;
uniform float g_sampled5RadiusPadPx;
uniform float g_sampled5Eps;
uniform float g_projDistortionK1;
uniform float g_projDistortionK2;
uniform uint g_debugShowEllipses;
uniform uint g_debugShowProcessedCount;
uniform float g_debugEllipseThicknessPx;
uniform float3 g_debugEllipseColor;

struct Gaussian3D
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

struct Ellipsoid
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
        return depthValue > EPS_DEPTH_VISIBILITY
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
        if (abs(bboxExtent.x - bboxExtent.y) < EPS_SMALL) scanAlongX = spanX >= spanY;
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

        if (conic.z > EPS_CONIC_DETERMINANT)
        {
            float yAtMinX = clamp(-conic.y * boxMin.x / conic.z, boxMin.y, boxMax.y);
            float yAtMaxX = clamp(-conic.y * boxMax.x / conic.z, boxMin.y, boxMax.y);
            minVal = min(minVal, evalConic(float2(boxMin.x, yAtMinX)));
            minVal = min(minVal, evalConic(float2(boxMax.x, yAtMaxX)));
        }
        if (conic.x > EPS_CONIC_DETERMINANT)
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
        return minConicOverTileBox(boxMin, boxMax) <= 1.0 + EPS_CONIC_INTERSECTION;
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
