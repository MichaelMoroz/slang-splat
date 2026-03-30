import "utility/splatting/common.slang";
import "utility/splatting/params.slang";
import "utility/math/camera.slang";

static const uint UINT_MAX_VALUE = 0xFFFFFFFFu;
static const uint INVALID_START = UINT_MAX_VALUE;
static const uint PROJECT_GROUP_SIZE = 256u;
static const uint TILE_GROUP_SIZE = 256u;
static const uint RASTER_THREAD_TILE_DIM = 8u;
static const uint RASTER_GROUP_THREADS = RASTER_THREAD_TILE_DIM * RASTER_THREAD_TILE_DIM;
static const uint RASTER_TILE_SIZE = RASTER_THREAD_TILE_DIM;
static const uint RASTER_BATCH = 64u;
static const uint RASTER_GRAD_FIXED_FRAC_BITS = 12u;
static const uint RASTER_CACHE_RO_LOCAL_X = 0u;
static const uint RASTER_CACHE_RO_LOCAL_Y = 1u;
static const uint RASTER_CACHE_RO_LOCAL_Z = 2u;
static const uint RASTER_CACHE_SCALE_X = 3u;
static const uint RASTER_CACHE_SCALE_Y = 4u;
static const uint RASTER_CACHE_SCALE_Z = 5u;
static const uint RASTER_CACHE_QUAT_W = 6u;
static const uint RASTER_CACHE_QUAT_X = 7u;
static const uint RASTER_CACHE_QUAT_Y = 8u;
static const uint RASTER_CACHE_QUAT_Z = 9u;
static const uint RASTER_CACHE_COLOR_R = 10u;
static const uint RASTER_CACHE_COLOR_G = 11u;
static const uint RASTER_CACHE_COLOR_B = 12u;
static const uint RASTER_CACHE_FUSED_OPACITY = 13u;
static const uint RASTER_CACHE_PARAM_COUNT = 14u;
groupshared float3 gsRoLocal[RASTER_BATCH];
groupshared float3 gsScale[RASTER_BATCH];
groupshared float4 gsRotation[RASTER_BATCH];
groupshared half4 gsColorAlpha[RASTER_BATCH];
groupshared uint gsBatchSplatId[RASTER_BATCH];

uniform StructuredBuffer<float> g_SplatParams;

uniform RWStructuredBuffer<float4> g_ScreenCenterRadiusDepth;
uniform RWStructuredBuffer<float4> g_ScreenColorAlpha;
uniform RWStructuredBuffer<float4> g_ScreenEllipseConic;
uniform RWStructuredBuffer<uint> g_SplatVisible;
uniform RWStructuredBuffer<float> g_RasterCache;

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
uniform StructuredBuffer<float4> g_OutputGrad;
uniform RWStructuredBuffer<float4> g_TrainingForwardState;
uniform RWStructuredBuffer<uint> g_TrainingProcessedEnd;

uniform RWStructuredBuffer<float> g_ParamGrads;
uniform RWStructuredBuffer<int> g_CachedRasterGradsFixed;
uniform RWStructuredBuffer<float> g_CachedRasterGradsFloat;
uniform float g_RasterGradDecodeScale;
uniform float g_CachedRasterGradFixedROLocalRange;
uniform float g_CachedRasterGradFixedScaleRange;
uniform float g_CachedRasterGradFixedQuatRange;
uniform float g_CachedRasterGradFixedColorRange;
uniform float g_CachedRasterGradFixedOpacityRange;

uniform CameraParams g_Camera;
uniform PrepassParams g_Prepass;
uniform RasterParams g_Raster;

Gaussian3D load_gaussian3d(uint splatId)
{
    Gaussian3D gaussian;
    uint splatCount = g_Prepass.splatCount;
    gaussian.position = float3(
        load_splat_param(g_SplatParams, splatCount, PARAM_POSITION_X, splatId),
        load_splat_param(g_SplatParams, splatCount, PARAM_POSITION_Y, splatId),
        load_splat_param(g_SplatParams, splatCount, PARAM_POSITION_Z, splatId)
    );
    gaussian.logScale = float3(
        load_splat_param(g_SplatParams, splatCount, PARAM_SCALE_X, splatId),
        load_splat_param(g_SplatParams, splatCount, PARAM_SCALE_Y, splatId),
        load_splat_param(g_SplatParams, splatCount, PARAM_SCALE_Z, splatId)
    );
    gaussian.rotation = float4(
        load_splat_param(g_SplatParams, splatCount, PARAM_ROTATION_X, splatId),
        load_splat_param(g_SplatParams, splatCount, PARAM_ROTATION_Y, splatId),
        load_splat_param(g_SplatParams, splatCount, PARAM_ROTATION_Z, splatId),
        load_splat_param(g_SplatParams, splatCount, PARAM_ROTATION_W, splatId)
    );
    gaussian.colorAlpha = float4(
        load_splat_param(g_SplatParams, splatCount, PARAM_COLOR_R, splatId),
        load_splat_param(g_SplatParams, splatCount, PARAM_COLOR_G, splatId),
        load_splat_param(g_SplatParams, splatCount, PARAM_COLOR_B, splatId),
        load_splat_param(g_SplatParams, splatCount, PARAM_RAW_OPACITY, splatId)
    );
    return gaussian;
}

uint raster_cache_index(uint splatCount, uint fieldId, uint splatId)
{
    return fieldId * splatCount + splatId;
}
