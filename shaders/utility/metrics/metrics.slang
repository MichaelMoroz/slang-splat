import "utility/losses/losses.slang";
import "utility/math/constants.slang";
import "utility/math/numeric.slang";
import "utility/splatting/common.slang";
import "utility/splatting/opacity.slang";
import "utility/splatting/params.slang";

static const uint METRICS_GROUP_SIZE = 256u;
static const uint METRICS_TILE_DIM = 16u;
static const float METRICS_SANITIZE_ABS = 1e8;
static const uint IMAGE_METRIC_SLOT_MSE = 0u;
static const uint FLOAT_SIGN_MASK = 0x80000000u;
static const float POSITIVE_INFINITY_F = asfloat(0x7F800000u);
static const float NEGATIVE_INFINITY_F = asfloat(0xFF800000u);
static const uint SCENE_METRIC_POSITION_FIRST = 0u;
static const uint SCENE_METRIC_SCALE_FIRST = 3u;
static const uint SCENE_METRIC_ROTATION_FIRST = 6u;
static const uint SCENE_METRIC_BASE_COLOR_FIRST = 10u;
static const uint SCENE_METRIC_HIGHER_SH_FIRST = 13u;
static const uint SCENE_METRIC_OPACITY = 58u;
static const uint REFINEMENT_METRIC_CURRENT_FRAME_CONTRIBUTION = 0u;
static const uint REFINEMENT_METRIC_CONTRIBUTION = 1u;
static const uint REFINEMENT_METRIC_DISTRIBUTION = 2u;

uniform uint g_ClearCount;
uniform RWStructuredBuffer<uint> g_ClearUIntBuffer;
uniform RWStructuredBuffer<float> g_ClearFloatBuffer;

uniform StructuredBuffer<float> g_SplatParams;
uniform uint g_SplatCount;
uniform uint g_SplatPackedParamCount;
uniform StructuredBuffer<float> g_Tensor;
uniform StructuredBuffer<int> g_FixedIntBuffer;
uniform StructuredBuffer<uint4> g_SplatContributionInfo;
uniform StructuredBuffer<float> g_SplatViewedFractionHistory;
uniform StructuredBuffer<float> g_GradientStats;
uniform uint g_ParamCount;
uniform uint g_ItemCount;
uniform uint g_BinCount;
uniform float g_Log10Min;
uniform float g_Log10InvBinSize;
uniform float g_ValueMin;
uniform float g_ValueInvBinSize;
uniform float g_DecodeScale;
uniform float g_RefinementGradientVarianceWeightExponent;
uniform float g_RefinementContributionWeightExponent;
uniform float g_RefinementMinViewedFraction;
uniform StructuredBuffer<float2> g_ParamHistogramBounds;
uniform RWStructuredBuffer<uint> g_Histogram;
uniform RWStructuredBuffer<uint2> g_ParamRanges;
uniform RWStructuredBuffer<float> g_DecodedFloatBuffer;

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

uint value_histogram_bin(float value)
{
    float scaled = (value - g_ValueMin) * g_ValueInvBinSize;
    int bin = int(floor(scaled));
    return uint(clamp(bin, 0, int(g_BinCount) - 1));
}

uint bounded_histogram_bin(float value, float2 bounds)
{
    float scaled = (value - bounds.x) * bounds.y;
    int bin = int(floor(scaled));
    return uint(clamp(bin, 0, int(g_BinCount) - 1));
}

bool scene_metric_param_uses_log10(uint paramId)
{
    return (paramId >= SCENE_METRIC_SCALE_FIRST && paramId < SCENE_METRIC_ROTATION_FIRST) || paramId == SCENE_METRIC_OPACITY;
}

float scene_metric_param_value(uint paramId, uint splatId)
{
    if (paramId < SCENE_METRIC_SCALE_FIRST) return load_splat_param(g_SplatParams, g_SplatCount, PARAM_POSITION_X + paramId, splatId);
    if (paramId < SCENE_METRIC_ROTATION_FIRST) return exp(load_splat_param(g_SplatParams, g_SplatCount, PARAM_SCALE_X + paramId - SCENE_METRIC_SCALE_FIRST, splatId));
    if (paramId == SCENE_METRIC_ROTATION_FIRST) return load_splat_param(g_SplatParams, g_SplatCount, PARAM_ROTATION_W, splatId);
    if (paramId < SCENE_METRIC_BASE_COLOR_FIRST) return load_splat_param(g_SplatParams, g_SplatCount, PARAM_ROTATION_X + paramId - SCENE_METRIC_ROTATION_FIRST - 1u, splatId);
    if (paramId < SCENE_METRIC_HIGHER_SH_FIRST)
    {
        float3 sh0 = load_splat_sh_coeff(g_SplatParams, g_SplatCount, g_SplatPackedParamCount, 0u, splatId);
        float3 color = saturate(evaluate_sh0_color_raw(sh0));
        return color[paramId - SCENE_METRIC_BASE_COLOR_FIRST];
    }
    if (paramId < SCENE_METRIC_OPACITY)
    {
        uint shComponent = paramId - SCENE_METRIC_HIGHER_SH_FIRST + SH_PARAM_STRIDE;
        float3 coeff = load_splat_sh_coeff(g_SplatParams, g_SplatCount, g_SplatPackedParamCount, shComponent / SH_PARAM_STRIDE, splatId);
        return coeff[shComponent % SH_PARAM_STRIDE];
    }
    if (paramId == SCENE_METRIC_OPACITY) return opacity_from_raw(load_splat_raw_opacity(g_SplatParams, g_SplatCount, g_SplatPackedParamCount, splatId));
    return 0.0;
}

bool scene_metric_histogram_value(uint paramId, uint splatId, out float value)
{
    value = scene_metric_param_value(paramId, splatId);
    if (!isfinite(value)) return false;
    if (!scene_metric_param_uses_log10(paramId)) return true;
    if (value <= 0.0) return false;
    value = safe_log10(value);
    return true;
}

float refinement_metric_value(uint paramId, uint splatId)
{
    float currentContribution = float(g_SplatContributionInfo[splatId].x) / SPLAT_CONTRIBUTION_FIXED_SCALE;
    currentContribution = isfinite(currentContribution) ? max(currentContribution, 0.0) : 0.0;
    if (paramId == REFINEMENT_METRIC_CURRENT_FRAME_CONTRIBUTION) return currentContribution;
    float contribution = asfloat(g_SplatContributionInfo[splatId].z);
    contribution = isfinite(contribution) ? max(contribution, 0.0) : 0.0;
    if (paramId == REFINEMENT_METRIC_CONTRIBUTION) return contribution;
    if (paramId != REFINEMENT_METRIC_DISTRIBUTION) return 0.0;
    float viewedFraction = g_SplatViewedFractionHistory[splatId];
    viewedFraction = isfinite(viewedFraction) ? max(viewedFraction, 0.0) : 0.0;
    if (!(viewedFraction > max(g_RefinementMinViewedFraction, 0.0))) viewedFraction = 0.0;
    float sampleCount = float(g_SplatContributionInfo[splatId].y);
    float variance = sampleCount > 0.0 ? g_GradientStats[splatId * DEBUG_GRAD_STATS_STRIDE] / sampleCount : 0.0;
    variance = isfinite(variance) ? max(variance, 0.0) : 0.0;
    float distribution = pow(variance, g_RefinementGradientVarianceWeightExponent) * pow(viewedFraction, g_RefinementContributionWeightExponent);
    return isfinite(distribution) ? max(distribution, 0.0) : 0.0;
}

uint float_to_ordered_uint(float value)
{
    uint bits = asuint(value);
    return (bits & FLOAT_SIGN_MASK) != 0u ? ~bits : (bits | FLOAT_SIGN_MASK);
}

float ordered_uint_to_float(uint value)
{
    uint bits = (value & FLOAT_SIGN_MASK) != 0u ? (value & ~FLOAT_SIGN_MASK) : ~value;
    return asfloat(bits);
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

    float3 logScale = load_splat_param_vector<3>(g_SplatParams, g_SplatCount, PARAM_SCALE_X, splatId);
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

    float3 logScale = load_splat_param_vector<3>(g_SplatParams, g_SplatCount, PARAM_SCALE_X, splatId);
    if (!all(isfinite(logScale))) return;

    float3 scale = exp(logScale);
    float minScale = max(min(scale.x, min(scale.y, scale.z)), TINY_VALUE);
    float maxScale = max(scale.x, max(scale.y, scale.z));
    InterlockedAdd(g_Histogram[histogram_bin(safe_log10(maxScale / minScale))], 1u);
}

[shader("compute")]
[numthreads(METRICS_TILE_DIM, METRICS_TILE_DIM, 1)]
void csHistogramParamTensorLog10(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint itemId = dispatchThreadID.x;
    uint paramId = dispatchThreadID.y;
    if (paramId >= g_ParamCount || itemId >= g_ItemCount || g_BinCount == 0u) return;

    float value = g_Tensor[paramId * g_ItemCount + itemId];
    if (!isfinite(value) || value == 0.0) return;
    InterlockedAdd(g_Histogram[paramId * g_BinCount + histogram_bin(safe_log10(abs(value)))], 1u);
}

[shader("compute")]
[numthreads(METRICS_TILE_DIM, METRICS_TILE_DIM, 1)]
void csHistogramParamTensorLinear(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint itemId = dispatchThreadID.x;
    uint paramId = dispatchThreadID.y;
    if (paramId >= g_ParamCount || itemId >= g_ItemCount || g_BinCount == 0u) return;

    float value = g_Tensor[paramId * g_ItemCount + itemId];
    if (!isfinite(value)) return;
    InterlockedAdd(g_Histogram[paramId * g_BinCount + value_histogram_bin(value)], 1u);
}

[shader("compute")]
[numthreads(METRICS_TILE_DIM, METRICS_TILE_DIM, 1)]
void csHistogramSceneParamsLinear(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    uint paramId = dispatchThreadID.y;
    if (paramId >= g_ParamCount || splatId >= g_SplatCount || g_BinCount == 0u) return;

    float value;
    if (!scene_metric_histogram_value(paramId, splatId, value)) return;
    InterlockedAdd(g_Histogram[paramId * g_BinCount + bounded_histogram_bin(value, g_ParamHistogramBounds[paramId])], 1u);
}

[shader("compute")]
[numthreads(METRICS_TILE_DIM, METRICS_TILE_DIM, 1)]
void csHistogramRefinementDistributionsLog10(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    uint paramId = dispatchThreadID.y;
    if (paramId >= g_ParamCount || splatId >= g_ItemCount || g_BinCount == 0u) return;

    float value = refinement_metric_value(paramId, splatId);
    if (!isfinite(value) || value <= 0.0) return;
    InterlockedAdd(g_Histogram[paramId * g_BinCount + bounded_histogram_bin(safe_log10(value), g_ParamHistogramBounds[paramId])], 1u);
}

[shader("compute")]
[numthreads(METRICS_GROUP_SIZE, 1, 1)]
void csInitParamTensorRanges(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint paramId = dispatchThreadID.x;
    if (paramId >= g_ParamCount) return;
    g_ParamRanges[paramId] = uint2(float_to_ordered_uint(POSITIVE_INFINITY_F), float_to_ordered_uint(NEGATIVE_INFINITY_F));
}

[shader("compute")]
[numthreads(METRICS_TILE_DIM, METRICS_TILE_DIM, 1)]
void csRangeParamTensor(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint itemId = dispatchThreadID.x;
    uint paramId = dispatchThreadID.y;
    if (paramId >= g_ParamCount || itemId >= g_ItemCount) return;

    float value = g_Tensor[paramId * g_ItemCount + itemId];
    if (!isfinite(value)) return;
    uint ordered = float_to_ordered_uint(value);
    InterlockedMin(g_ParamRanges[paramId].x, ordered);
    InterlockedMax(g_ParamRanges[paramId].y, ordered);
}

[shader("compute")]
[numthreads(METRICS_TILE_DIM, METRICS_TILE_DIM, 1)]
void csRangeSceneParams(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    uint paramId = dispatchThreadID.y;
    if (paramId >= g_ParamCount || splatId >= g_SplatCount) return;

    float value;
    if (!scene_metric_histogram_value(paramId, splatId, value)) return;
    uint ordered = float_to_ordered_uint(value);
    InterlockedMin(g_ParamRanges[paramId].x, ordered);
    InterlockedMax(g_ParamRanges[paramId].y, ordered);
}

[shader("compute")]
[numthreads(METRICS_TILE_DIM, METRICS_TILE_DIM, 1)]
void csRangeRefinementDistributions(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    uint paramId = dispatchThreadID.y;
    if (paramId >= g_ParamCount || splatId >= g_ItemCount) return;

    float value = refinement_metric_value(paramId, splatId);
    if (!isfinite(value) || value <= 0.0) return;
    uint ordered = float_to_ordered_uint(safe_log10(value));
    InterlockedMin(g_ParamRanges[paramId].x, ordered);
    InterlockedMax(g_ParamRanges[paramId].y, ordered);
}

[shader("compute")]
[numthreads(METRICS_GROUP_SIZE, 1, 1)]
void csDecodeFixedIntBufferToFloat(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    if (idx >= g_ClearCount) return;
    g_DecodedFloatBuffer[idx] = float(g_FixedIntBuffer[idx]) * g_DecodeScale;
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
