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
static const uint RASTER_CACHE_CENTER_DIR_X = 0u;
static const uint RASTER_CACHE_CENTER_DIR_Y = 1u;
static const uint RASTER_CACHE_CENTER_DIR_Z = 2u;
static const uint RASTER_CACHE_SIGMA_ORTHO_X = 3u;
static const uint RASTER_CACHE_SIGMA_ORTHO_Y = 4u;
static const uint RASTER_CACHE_SIGMA_ORTHO_Z = 5u;
static const uint RASTER_CACHE_DEPTH_DISTANCE = 6u;
static const uint RASTER_CACHE_COLOR_R = 7u;
static const uint RASTER_CACHE_COLOR_G = 8u;
static const uint RASTER_CACHE_COLOR_B = 9u;
static const uint RASTER_CACHE_FUSED_OPACITY = 10u;
static const uint RASTER_GRAD_PARAM_COUNT = 10u;
static const uint RASTER_GAUSSIAN_PARAM_COUNT = 11u;
groupshared RasterGaussian gsGaussian[RASTER_BATCH];
groupshared uint gsBatchSplatId[RASTER_BATCH];
groupshared uint gsDoneCount;

uniform StructuredBuffer<float> g_SplatParams;

uniform RWStructuredBuffer<float4> g_ScreenCenterRadiusDepth;
uniform RWStructuredBuffer<float4> g_ScreenColorAlpha;
uniform RWStructuredBuffer<float4> g_ScreenEllipseConic;
uniform RWStructuredBuffer<uint> g_SplatVisible;
uniform RWStructuredBuffer<float> g_SplatVisibleAreaPx;
uniform RWStructuredBuffer<RasterGaussian> g_RasterCache;

uniform RWStructuredBuffer<uint> g_VisibleKeys;
uniform RWStructuredBuffer<uint> g_VisibleValues;
uniform RWStructuredBuffer<uint> g_VisibleCounter;
uniform RWStructuredBuffer<uint> g_ScanlineCounts;
uniform RWStructuredBuffer<uint> g_ScanlineOffsets;
uniform RWStructuredBuffer<uint> g_ScanlineTileCounts;
uniform RWStructuredBuffer<uint> g_ScanlineTileOffsets;
uniform RWStructuredBuffer<uint> g_Keys;
uniform RWStructuredBuffer<uint> g_Values;
uniform RWStructuredBuffer<uint> g_ListCounter;
uniform RWStructuredBuffer<ScanlineWorkItem> g_ScanlineWorkItems;
uniform RWStructuredBuffer<uint> g_ScanlineCounter;
uniform RWStructuredBuffer<uint2> g_TileRanges;

uniform StructuredBuffer<uint> g_SortedKeys;
uniform StructuredBuffer<uint> g_SortedValues;
uniform StructuredBuffer<uint> g_PrepassParams;
uniform StructuredBuffer<float> g_DebugGradNorm;
uniform StructuredBuffer<float> g_DebugGradStats;
uniform StructuredBuffer<float2> g_DebugAdamMoments;

uniform RWTexture2D<float4> g_Output;
uniform StructuredBuffer<float4> g_OutputGrad;
uniform RWStructuredBuffer<float4> g_TrainingForwardState;
uniform RWStructuredBuffer<float> g_TrainingDensity;
uniform StructuredBuffer<float> g_TrainingRegularizerGrad;
uniform RWStructuredBuffer<uint> g_TrainingProcessedEnd;
uniform RWStructuredBuffer<uint> g_TrainingBatchEnd;

uniform RWStructuredBuffer<float> g_ParamGrads;
uniform RWStructuredBuffer<int> g_CachedRasterGradsFixed;
uniform RWStructuredBuffer<float> g_CachedRasterGradsFloat;
uniform RWStructuredBuffer<float> g_GradientStats;
uniform float g_RasterGradDecodeScale;
uniform float g_CachedRasterGradFixedROLocalRange;
uniform float g_CachedRasterGradFixedScaleRange;
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
    gaussian.sh0 = float3(
        load_splat_param(g_SplatParams, splatCount, PARAM_SH0_R, splatId),
        load_splat_param(g_SplatParams, splatCount, PARAM_SH0_G, splatId),
        load_splat_param(g_SplatParams, splatCount, PARAM_SH0_B, splatId)
    );
    gaussian.sh1X = float3(
        load_splat_param(g_SplatParams, splatCount, PARAM_SH1_X_R, splatId),
        load_splat_param(g_SplatParams, splatCount, PARAM_SH1_X_G, splatId),
        load_splat_param(g_SplatParams, splatCount, PARAM_SH1_X_B, splatId)
    );
    gaussian.sh1Y = float3(
        load_splat_param(g_SplatParams, splatCount, PARAM_SH1_Y_R, splatId),
        load_splat_param(g_SplatParams, splatCount, PARAM_SH1_Y_G, splatId),
        load_splat_param(g_SplatParams, splatCount, PARAM_SH1_Y_B, splatId)
    );
    gaussian.sh1Z = float3(
        load_splat_param(g_SplatParams, splatCount, PARAM_SH1_Z_R, splatId),
        load_splat_param(g_SplatParams, splatCount, PARAM_SH1_Z_G, splatId),
        load_splat_param(g_SplatParams, splatCount, PARAM_SH1_Z_B, splatId)
    );
    gaussian.sh2_0 = float3(
        load_splat_param(g_SplatParams, splatCount, sh_param_id(4u, 0u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(4u, 1u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(4u, 2u), splatId)
    );
    gaussian.sh2_1 = float3(
        load_splat_param(g_SplatParams, splatCount, sh_param_id(5u, 0u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(5u, 1u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(5u, 2u), splatId)
    );
    gaussian.sh2_2 = float3(
        load_splat_param(g_SplatParams, splatCount, sh_param_id(6u, 0u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(6u, 1u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(6u, 2u), splatId)
    );
    gaussian.sh2_3 = float3(
        load_splat_param(g_SplatParams, splatCount, sh_param_id(7u, 0u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(7u, 1u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(7u, 2u), splatId)
    );
    gaussian.sh2_4 = float3(
        load_splat_param(g_SplatParams, splatCount, sh_param_id(8u, 0u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(8u, 1u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(8u, 2u), splatId)
    );
    gaussian.sh3_0 = float3(
        load_splat_param(g_SplatParams, splatCount, sh_param_id(9u, 0u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(9u, 1u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(9u, 2u), splatId)
    );
    gaussian.sh3_1 = float3(
        load_splat_param(g_SplatParams, splatCount, sh_param_id(10u, 0u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(10u, 1u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(10u, 2u), splatId)
    );
    gaussian.sh3_2 = float3(
        load_splat_param(g_SplatParams, splatCount, sh_param_id(11u, 0u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(11u, 1u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(11u, 2u), splatId)
    );
    gaussian.sh3_3 = float3(
        load_splat_param(g_SplatParams, splatCount, sh_param_id(12u, 0u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(12u, 1u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(12u, 2u), splatId)
    );
    gaussian.sh3_4 = float3(
        load_splat_param(g_SplatParams, splatCount, sh_param_id(13u, 0u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(13u, 1u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(13u, 2u), splatId)
    );
    gaussian.sh3_5 = float3(
        load_splat_param(g_SplatParams, splatCount, sh_param_id(14u, 0u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(14u, 1u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(14u, 2u), splatId)
    );
    gaussian.sh3_6 = float3(
        load_splat_param(g_SplatParams, splatCount, sh_param_id(15u, 0u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(15u, 1u), splatId),
        load_splat_param(g_SplatParams, splatCount, sh_param_id(15u, 2u), splatId)
    );
    gaussian.rawOpacity = load_splat_param(g_SplatParams, splatCount, PARAM_RAW_OPACITY, splatId);
    return gaussian;
}

uint cached_raster_grad_index(uint splatCount, uint paramId, uint splatId)
{
    return paramId * splatCount + splatId;
}
