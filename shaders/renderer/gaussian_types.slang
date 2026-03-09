import "utility/splatting/common.slang";
import "utility/math/camera.slang";

static const uint UINT_MAX_VALUE = 0xFFFFFFFFu;
static const uint INVALID_START = UINT_MAX_VALUE;
static const uint PROJECT_GROUP_SIZE = 256u;
static const uint TILE_GROUP_SIZE = 256u;
static const uint RASTER_THREAD_TILE_DIM = 8u;
static const uint RASTER_MICROTILE_DIM = 2u;
static const uint RASTER_MICROTILE_PIXELS = RASTER_MICROTILE_DIM * RASTER_MICROTILE_DIM;
static const uint RASTER_GROUP_THREADS = RASTER_THREAD_TILE_DIM * RASTER_THREAD_TILE_DIM;
static const uint RASTER_EFFECTIVE_TILE_SIZE = RASTER_THREAD_TILE_DIM * RASTER_MICROTILE_DIM;
static const uint RASTER_BATCH = 64u;
static const uint SPLAT_GRAD_COMPONENTS = 4u;

groupshared float4 gsRoLocal[RASTER_BATCH];
groupshared half4 gsInvScale[RASTER_BATCH];
groupshared half4 gsQuat[RASTER_BATCH];
groupshared half4 gsColorAlpha[RASTER_BATCH];
groupshared uint gsBatchSplatId[RASTER_BATCH];
groupshared uint gsStoppedMicrotiles;

uniform StructuredBuffer<float4> g_Positions;
uniform StructuredBuffer<float4> g_Scales;
uniform StructuredBuffer<float4> g_Rotations;
uniform StructuredBuffer<float4> g_ColorAlpha;

uniform RWStructuredBuffer<float4> g_ScreenCenterRadiusDepth;
uniform RWStructuredBuffer<float4> g_ScreenColorAlpha;
uniform RWStructuredBuffer<float4> g_ScreenEllipseConic;
uniform RWStructuredBuffer<uint> g_SplatVisible;

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
uniform StructuredBuffer<float> g_DebugGradNorm;

uniform RWTexture2D<float4> g_Output;
uniform Texture2D<float4> g_OutputGrad;

uniform RWStructuredBuffer<float> g_GradPositions;
uniform RWStructuredBuffer<float> g_GradScales;
uniform RWStructuredBuffer<float> g_GradRotations;
uniform RWStructuredBuffer<float> g_GradColorAlpha;

uniform CameraParams g_Camera;
uniform PrepassParams g_Prepass;
uniform RasterParams g_Raster;

Gaussian3D load_gaussian3d(uint splatId)
{
    Gaussian3D gaussian;
    gaussian.position = g_Positions[splatId].xyz;
    gaussian.scale = g_Scales[splatId].xyz;
    gaussian.rotation = g_Rotations[splatId];
    gaussian.colorAlpha = g_ColorAlpha[splatId];
    return gaussian;
}
