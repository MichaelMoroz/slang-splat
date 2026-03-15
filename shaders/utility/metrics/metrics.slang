import "utility/losses/losses.slang";
import "utility/math/constants.slang";
import "utility/math/numeric.slang";
import "utility/splatting/params.slang";

static const uint METRICS_GROUP_SIZE = 256u;
static const uint METRICS_TILE_DIM = 16u;
static const float METRICS_SANITIZE_ABS = 1e8;
static const uint IMAGE_METRIC_SLOT_MSE = 0u;

uniform uint g_ClearCount;
uniform RWStructuredBuffer<uint> g_ClearUIntBuffer;
uniform RWStructuredBuffer<float> g_ClearFloatBuffer;

uniform StructuredBuffer<float> g_SplatParams;
uniform uint g_SplatCount;
uniform uint g_BinCount;
uniform float g_Log10Min;
uniform float g_Log10InvBinSize;
uniform RWStructuredBuffer<uint> g_Histogram;

uniform uint g_Width;
uniform uint g_Height;
uniform float g_InvPixelCount;
uniform Texture2D<float4> g_Rendered;
uniform Texture2D<float4> g_Target;
uniform RWStructuredBuffer<float> g_ImageMetricBuffer;

groupshared float gsMetricTile[METRICS_TILE_DIM * METRICS_TILE_DIM];

float safe_log10(float value)
{
    return log2(max(value, TINY_VALUE)) * LOG2_TO_LOG10;
}

uint histogram_bin(float logValue)
{
    float scaled = (logValue - g_Log10Min) * g_Log10InvBinSize;
    int bin = int(floor(scaled));
    return uint(clamp(bin, 0, int(g_BinCount) - 1));
}

[shader("compute")]
[numthreads(METRICS_GROUP_SIZE, 1, 1)]
void csClearUIntBuffer(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx >= g_ClearCount) return;
    g_ClearUIntBuffer[idx] = 0u;
}

[shader("compute")]
[numthreads(METRICS_GROUP_SIZE, 1, 1)]
void csClearFloatBuffer(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx >= g_ClearCount) return;
    g_ClearFloatBuffer[idx] = 0.0;
}

[shader("compute")]
[numthreads(METRICS_GROUP_SIZE, 1, 1)]
void csHistogramScaleLog10(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount || g_BinCount == 0u) return;

    float3 logScale = float3(
        load_splat_param(g_SplatParams, g_SplatCount, PARAM_SCALE_X, splatId),
        load_splat_param(g_SplatParams, g_SplatCount, PARAM_SCALE_Y, splatId),
        load_splat_param(g_SplatParams, g_SplatCount, PARAM_SCALE_Z, splatId)
    );
    if (!all(isfinite(logScale))) return;

    float3 scale = exp(logScale);
    float scaleLog10 = (safe_log10(scale.x) + safe_log10(scale.y) + safe_log10(scale.z)) * (1.0 / 3.0);
    InterlockedAdd(g_Histogram[histogram_bin(scaleLog10)], 1u);
}

[shader("compute")]
[numthreads(METRICS_GROUP_SIZE, 1, 1)]
void csHistogramAnisotropyLog10(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount || g_BinCount == 0u) return;

    float3 logScale = float3(
        load_splat_param(g_SplatParams, g_SplatCount, PARAM_SCALE_X, splatId),
        load_splat_param(g_SplatParams, g_SplatCount, PARAM_SCALE_Y, splatId),
        load_splat_param(g_SplatParams, g_SplatCount, PARAM_SCALE_Z, splatId)
    );
    if (!all(isfinite(logScale))) return;

    float3 scale = exp(logScale);
    float minScale = max(min(scale.x, min(scale.y, scale.z)), TINY_VALUE);
    float maxScale = max(scale.x, max(scale.y, scale.z));
    InterlockedAdd(g_Histogram[histogram_bin(safe_log10(maxScale / minScale))], 1u);
}

[shader("compute")]
[numthreads(METRICS_TILE_DIM, METRICS_TILE_DIM, 1)]
void csAccumulateImageMSE(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID)
{
    uint localIndex = groupThreadID.y * METRICS_TILE_DIM + groupThreadID.x;
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;

    float localMSE = 0.0;
    if (px < g_Width && py < g_Height)
    {
        uint2 pixel = uint2(px, py);
        float3 rendered = sanitize_float3(g_Rendered[pixel].xyz, METRICS_SANITIZE_ABS);
        float3 target = sanitize_float3(g_Target[pixel].xyz, METRICS_SANITIZE_ABS);
        float mse = rgb_mse_loss(rendered, target, g_InvPixelCount);
        localMSE = isfinite(mse) ? mse : 0.0;
    }

    gsMetricTile[localIndex] = localMSE;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = (METRICS_TILE_DIM * METRICS_TILE_DIM) >> 1; stride > 0; stride >>= 1)
    {
        if (localIndex < stride) gsMetricTile[localIndex] += gsMetricTile[localIndex + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (localIndex == 0u && isfinite(gsMetricTile[0])) InterlockedAdd(g_ImageMetricBuffer[IMAGE_METRIC_SLOT_MSE], gsMetricTile[0]);
}
