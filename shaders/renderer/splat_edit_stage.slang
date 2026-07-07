// GPU splat-editor kernels. Operate directly on the planar `splat_params` buffer so
// selection, property edits, and resampling never round-trip through the CPU. The
// selection mask is the same per-splat buffer the renderer reads as the highlight, so
// selecting a splat highlights it with no extra upload.
import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/math/quaternion.slang";
import "utility/random/random.slang";
import "utility/splatting/opacity.slang";
import "utility/splatting/params.slang";

static const uint EDIT_GROUP_SIZE = 256u;

static const uint SELECT_MODE_REPLACE = 0u;
static const uint SELECT_MODE_ADD = 1u;
static const uint SELECT_MODE_SUBTRACT = 2u;
static const uint SELECT_MODE_INTERSECT = 3u;

static const uint SELECT_SCALAR_SCALE = 0u;
static const uint SELECT_SCALAR_OPACITY = 1u;
static const uint SELECT_SCALAR_COLOR = 2u;

static const uint EDIT_FLAG_COLOR = 1u;
static const uint EDIT_FLAG_OPACITY = 2u;
static const uint EDIT_FLAG_SCALE = 4u;

static const float EDIT_SH_C0 = 0.28209479177387814;
static const float3 EDIT_LUMA_WEIGHTS = float3(0.2126, 0.7152, 0.0722);
static const float EDIT_DENSIFY_SCALE_SHRINK = 1.6;
static const uint EDIT_RESAMPLE_HASH = 0x9e3779b9u;

uniform RWStructuredBuffer<float> g_SplatParams;
uniform RWStructuredBuffer<uint> g_SelectionMask;
uniform uint g_EditSplatCount;
uniform uint g_PackedParamCount;

uniform uint g_SelectMode;
uniform uint g_SelectScalarKind;
uniform float2 g_SelectRange;
uniform float3 g_SelectBoxCenter;
uniform float3 g_SelectBoxAxisX;
uniform float3 g_SelectBoxAxisY;
uniform float3 g_SelectBoxAxisZ;
uniform float3 g_SelectBoxHalfExtent;
uniform uint g_SelectSetValue;

uniform uint g_EditFlags;
uniform float3 g_EditColor;
uniform float g_EditOpacity;
uniform float g_EditTargetScale;

uniform RWStructuredBuffer<float> g_SrcParams;
uniform RWStructuredBuffer<float> g_DstParams;
uniform RWStructuredBuffer<uint> g_SrcMask;
uniform RWStructuredBuffer<uint> g_DstMask;
uniform RWStructuredBuffer<uint> g_KeepFlags;
uniform RWStructuredBuffer<uint> g_KeepPrefix;
uniform RWStructuredBuffer<uint> g_SelectedIds;
uniform RWStructuredBuffer<uint> g_SelectedPrefix;
uniform RWStructuredBuffer<uint> g_ResampleCounter;
uniform uint g_ResampleSrcCount;
uniform uint g_ResampleSurvivorCount;
uniform uint g_ResampleChildCount;
uniform uint g_ResampleSelectedCount;
uniform float g_ResampleRatio;
uniform uint g_ResampleSeed;

float edit_selection_scalar(uint splatId)
{
    if (g_SelectScalarKind == SELECT_SCALAR_OPACITY)
    {
        return opacity_from_raw(load_splat_raw_opacity(g_SplatParams, g_EditSplatCount, g_PackedParamCount, splatId));
    }
    if (g_SelectScalarKind == SELECT_SCALAR_COLOR)
    {
        float3 sh0 = load_splat_param_vector<3>(g_SplatParams, g_EditSplatCount, PARAM_SH0_R, splatId);
        float3 color = saturate(0.5 + EDIT_SH_C0 * sh0);
        return dot(color, EDIT_LUMA_WEIGHTS);
    }
    float3 logScale = load_splat_param_vector<3>(g_SplatParams, g_EditSplatCount, PARAM_SCALE_X, splatId);
    return exp((logScale.x + logScale.y + logScale.z) / 3.0);
}

uint combine_selection(uint current, bool hit)
{
    if (g_SelectMode == SELECT_MODE_REPLACE) return hit ? 1u : 0u;
    if (g_SelectMode == SELECT_MODE_SUBTRACT) return (current != 0u && !hit) ? 1u : 0u;
    if (g_SelectMode == SELECT_MODE_INTERSECT) return (current != 0u && hit) ? 1u : 0u;
    return (current != 0u || hit) ? 1u : 0u;
}

[shader("compute")]
[numthreads(EDIT_GROUP_SIZE, 1, 1)]
void csSelectBox(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_EditSplatCount) return;
    float3 offset = load_splat_param_vector<3>(g_SplatParams, g_EditSplatCount, PARAM_POSITION_X, splatId) - g_SelectBoxCenter;
    float3 local = float3(dot(g_SelectBoxAxisX, offset), dot(g_SelectBoxAxisY, offset), dot(g_SelectBoxAxisZ, offset));
    bool hit = all(abs(local) <= g_SelectBoxHalfExtent);
    g_SelectionMask[splatId] = combine_selection(g_SelectionMask[splatId], hit);
}

[shader("compute")]
[numthreads(EDIT_GROUP_SIZE, 1, 1)]
void csSelectRange(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_EditSplatCount) return;
    float value = edit_selection_scalar(splatId);
    bool hit = value >= g_SelectRange.x && value <= g_SelectRange.y;
    g_SelectionMask[splatId] = combine_selection(g_SelectionMask[splatId], hit);
}

[shader("compute")]
[numthreads(EDIT_GROUP_SIZE, 1, 1)]
void csSelectSetAll(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_EditSplatCount) return;
    // g_SelectSetValue: 0 clears, 1 selects all, 2 inverts the current mask.
    uint current = g_SelectionMask[splatId];
    g_SelectionMask[splatId] = g_SelectSetValue == 2u ? (current != 0u ? 0u : 1u) : g_SelectSetValue;
}

[shader("compute")]
[numthreads(EDIT_GROUP_SIZE, 1, 1)]
void csCountSelected(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_EditSplatCount) return;
    if (g_SelectionMask[splatId] != 0u) InterlockedAdd(g_ResampleCounter[0], 1u);
}

[shader("compute")]
[numthreads(EDIT_GROUP_SIZE, 1, 1)]
void csEditProperties(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_EditSplatCount || g_SelectionMask[splatId] == 0u) return;

    if ((g_EditFlags & EDIT_FLAG_COLOR) != 0u)
    {
        float3 sh0 = (saturate(g_EditColor) - 0.5) / EDIT_SH_C0;
        store_splat_param_vector<3>(g_SplatParams, g_EditSplatCount, PARAM_SH0_R, splatId, sh0);
    }
    if ((g_EditFlags & EDIT_FLAG_OPACITY) != 0u)
    {
        store_splat_raw_opacity(g_SplatParams, g_EditSplatCount, g_PackedParamCount, splatId, raw_opacity_from_alpha(g_EditOpacity));
    }
    if ((g_EditFlags & EDIT_FLAG_SCALE) != 0u)
    {
        float3 logScale = load_splat_param_vector<3>(g_SplatParams, g_EditSplatCount, PARAM_SCALE_X, splatId);
        float targetLog = log(max(g_EditTargetScale, 1e-12));
        float shift = targetLog - (logScale.x + logScale.y + logScale.z) / 3.0;
        store_splat_param_vector<3>(g_SplatParams, g_EditSplatCount, PARAM_SCALE_X, splatId, logScale + shift);
    }
}

float resample_hash01(uint splatId)
{
    uint state = hash_u32(g_ResampleSeed ^ ((splatId + 1u) * EDIT_RESAMPLE_HASH));
    return rng_uniform_open01(state);
}

[shader("compute")]
[numthreads(EDIT_GROUP_SIZE, 1, 1)]
void csResampleMarkKeep(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_ResampleSrcCount) return;
    bool selected = g_SrcMask[splatId] != 0u;
    // Unselected splats always survive; selected splats survive with probability = ratio,
    // so ratio 0 deletes the whole selection and ratio 1 keeps it.
    bool keep = !selected || resample_hash01(splatId) < g_ResampleRatio;
    g_KeepFlags[splatId] = keep ? 1u : 0u;
    g_SelectedIds[splatId] = selected ? 1u : 0u;
}

[shader("compute")]
[numthreads(EDIT_GROUP_SIZE, 1, 1)]
void csResampleCompactSurvivors(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_ResampleSrcCount || g_KeepFlags[splatId] == 0u) return;
    uint dstId = g_KeepPrefix[splatId];
    [loop] for (uint paramId = 0u; paramId < g_PackedParamCount; paramId++)
    {
        g_DstParams[splat_param_index(g_ResampleSurvivorCount + g_ResampleChildCount, paramId, dstId)] =
            g_SrcParams[splat_param_index(g_ResampleSrcCount, paramId, splatId)];
    }
    g_DstMask[dstId] = g_SrcMask[splatId];
}

// Compacts the ids of selected splats into g_SelectedIds[0 .. selectedCount) so children
// can pick a random parent. g_SelectedPrefix is the exclusive scan of the selected flags.
[shader("compute")]
[numthreads(EDIT_GROUP_SIZE, 1, 1)]
void csResampleCompactSelected(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_ResampleSrcCount || g_SrcMask[splatId] == 0u) return;
    g_SelectedIds[g_SelectedPrefix[splatId]] = splatId;
}

[shader("compute")]
[numthreads(EDIT_GROUP_SIZE, 1, 1)]
void csResampleEmitChildren(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint childId = dispatchThreadID.x;
    if (childId >= g_ResampleChildCount || g_ResampleSelectedCount == 0u) return;

    uint totalCount = g_ResampleSurvivorCount + g_ResampleChildCount;
    uint rngState = hash_u32(g_ResampleSeed ^ ((childId + 1u) * EDIT_RESAMPLE_HASH));
    uint parentSlot = uint(rng_uniform_open01(rngState) * float(g_ResampleSelectedCount));
    parentSlot = min(parentSlot, g_ResampleSelectedCount - 1u);
    uint parentId = g_SelectedIds[parentSlot];
    uint dstId = g_ResampleSurvivorCount + childId;

    [loop] for (uint paramId = 0u; paramId < g_PackedParamCount; paramId++)
    {
        g_DstParams[splat_param_index(totalCount, paramId, dstId)] =
            g_SrcParams[splat_param_index(g_ResampleSrcCount, paramId, parentId)];
    }

    float3 position = load_splat_param_vector<3>(g_SrcParams, g_ResampleSrcCount, PARAM_POSITION_X, parentId);
    float3 logScale = load_splat_param_vector<3>(g_SrcParams, g_ResampleSrcCount, PARAM_SCALE_X, parentId);
    float4 rotation = load_splat_param_vector<4>(g_SrcParams, g_ResampleSrcCount, PARAM_ROTATION_X, parentId);
    float3 localOffset = rng_normal3_box_muller(rngState) * exp(logScale);
    float3 worldOffset = rotate_by_quat(localOffset, float4(rotation.x, -rotation.y, -rotation.z, -rotation.w));
    store_splat_param_vector<3>(g_DstParams, totalCount, PARAM_POSITION_X, dstId, position + worldOffset);
    store_splat_param_vector<3>(g_DstParams, totalCount, PARAM_SCALE_X, dstId, logScale - log(EDIT_DENSIFY_SCALE_SHRINK));
    g_DstMask[dstId] = 1u;
}
