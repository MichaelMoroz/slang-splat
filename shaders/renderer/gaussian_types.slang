import "utility/splatting/common.slang";
import "utility/splatting/params.slang";
import "utility/math/camera.slang";

static const uint UINT_MAX_VALUE = 0xFFFFFFFFu;
static const uint INVALID_START = UINT_MAX_VALUE;
static const uint PROJECT_GROUP_SIZE = 256u;
static const uint TILE_GROUP_SIZE = 256u;
static const uint RASTER_THREAD_TILE_DIM = 8u;
static const uint RASTER_MICROTILE_DIM = 1u;
static const uint RASTER_MICROTILE_PIXELS = RASTER_MICROTILE_DIM * RASTER_MICROTILE_DIM;
static const uint RASTER_GROUP_THREADS = RASTER_THREAD_TILE_DIM * RASTER_THREAD_TILE_DIM;
static const uint RASTER_EFFECTIVE_TILE_SIZE = RASTER_THREAD_TILE_DIM * RASTER_MICROTILE_DIM;
static const uint RASTER_BATCH = 64u;
static const uint SPLAT_GRAD_STATE_RO_LOCAL_OPACITY_OFFSET = 0u;
static const uint SPLAT_GRAD_STATE_INV_SCALE_OFFSET = 4u;
static const uint SPLAT_GRAD_STATE_QUAT_OFFSET = 7u;
static const uint SPLAT_GRAD_STATE_COLOR_ALPHA_OFFSET = 11u;
static const uint SPLAT_GRAD_STATE_STRIDE = 15u;
groupshared float4 gsRoLocal[RASTER_BATCH];
groupshared half3 gsInvScale[RASTER_BATCH];
groupshared half4 gsQuat[RASTER_BATCH];
groupshared half4 gsColorAlpha[RASTER_BATCH];
groupshared uint gsBatchSplatId[RASTER_BATCH];
groupshared float gsGradState[RASTER_BATCH * SPLAT_GRAD_STATE_STRIDE];
groupshared uint gsStoppedMicrotiles;

uniform StructuredBuffer<float> g_SplatParams;

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
uniform StructuredBuffer<float4> g_OutputGrad;
uniform RWStructuredBuffer<float4> g_TrainingForwardState;
uniform RWStructuredBuffer<uint> g_TrainingProcessedEnd;

uniform RWStructuredBuffer<float> g_ParamGrads;

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
    gaussian.scale = float3(
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
