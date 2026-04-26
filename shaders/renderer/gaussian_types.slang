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
static const uint RASTER_CACHE_RO_CAMERA_X = 0u;
static const uint RASTER_CACHE_RO_CAMERA_Y = 1u;
static const uint RASTER_CACHE_RO_CAMERA_Z = 2u;
static const uint RASTER_CACHE_SIGMA_XX = 3u;
static const uint RASTER_CACHE_SIGMA_YY = 4u;
static const uint RASTER_CACHE_SIGMA_ZZ = 5u;
static const uint RASTER_CACHE_SIGMA_XY = 6u;
static const uint RASTER_CACHE_SIGMA_XZ = 7u;
static const uint RASTER_CACHE_SIGMA_YZ = 8u;
static const uint RASTER_CACHE_COLOR_R = 9u;
static const uint RASTER_CACHE_COLOR_G = 10u;
static const uint RASTER_CACHE_COLOR_B = 11u;
static const uint RASTER_CACHE_FUSED_OPACITY = 12u;
static const uint RASTER_CACHE_PARAM_COUNT = 13u;
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
uniform RWTexture2D<float4> g_TrainingDepthStats;
uniform RWStructuredBuffer<float> g_TrainingDensity;
uniform StructuredBuffer<float2> g_TrainingRegularizerGrad;
uniform RWStructuredBuffer<uint> g_TrainingProcessedEnd;
uniform RWStructuredBuffer<uint> g_TrainingBatchEnd;

uniform RWStructuredBuffer<float> g_ParamGrads;
uniform RWStructuredBuffer<int> g_CachedRasterGradsFixed;
uniform RWStructuredBuffer<float> g_CachedRasterGradsFloat;
uniform RWStructuredBuffer<float> g_GradientStats;
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
    gaussian.position = load_splat_param_vector<3>(g_SplatParams, splatCount, PARAM_POSITION_X, splatId);
    gaussian.logScale = load_splat_param_vector<3>(g_SplatParams, splatCount, PARAM_SCALE_X, splatId);
    gaussian.rotation = load_splat_param_vector<4>(g_SplatParams, splatCount, PARAM_ROTATION_X, splatId);
    [unroll] for (uint coeffIndex = 0u; coeffIndex < SH_COEFF_COUNT; coeffIndex++)
    {
        gaussian.shCoeffs[coeffIndex] = load_splat_param_vector<3>(g_SplatParams, splatCount, sh_param_id(coeffIndex, 0u), splatId);
    }
    gaussian.rawOpacity = load_splat_param(g_SplatParams, splatCount, PARAM_RAW_OPACITY, splatId);
    return gaussian;
}

uint cached_raster_grad_index(uint splatCount, uint paramId, uint splatId)
{
    return paramId * splatCount + splatId;
}
