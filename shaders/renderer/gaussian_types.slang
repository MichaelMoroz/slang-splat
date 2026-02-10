static const uint INVALID_START = 0xFFFFFFFFu;
static const uint RASTER_BATCH = 256u;
static const uint MVEE_POINT_COUNT = 5u;
static const uint MVEE_MAX_ITERS = 8u;

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
uniform RWStructuredBuffer<uint2> g_TileRanges;

uniform StructuredBuffer<uint> g_SortedKeys;
uniform StructuredBuffer<uint> g_SortedValues;

uniform RWTexture2D<float4> g_Output;

uniform uint g_splatCount;
uniform uint g_tileSize;
uniform uint g_tileWidth;
uniform uint g_tileHeight;
uniform uint g_tileCount;
uniform uint g_depthBits;
uniform uint g_sortedCount;
uniform uint g_maxListEntries;
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
    uint useConic;

    [mutating]
    void init(float2 centerPxIn, float radiusPxIn, float3 conicIn)
    {
        centerPx = centerPxIn;
        radiusPx = radiusPxIn;
        conic = conicIn;
        bboxExtent = float2(radiusPxIn, radiusPxIn);
        useConic = 0u;
    }

    bool isVisible(float camDistance, float depthValue, float2 viewport, float nearDepth, float farDepth)
    {
        return depthValue > 1e-4
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
        if (abs(bboxExtent.x - bboxExtent.y) < 1e-6)
        {
            scanAlongX = spanX >= spanY;
        }
        return scanAlongX;
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
    Ellipse ellipse;
};
