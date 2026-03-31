import "utility/losses/losses.slang";
import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/math/numeric.slang";
import "utility/math/quaternion.slang";
import "utility/splatting/common.slang";
import "utility/splatting/opacity.slang";
import "utility/splatting/params.slang";

static const uint TRAIN_TILE_DIM = 16u;
static const uint LOSS_SLOT_TOTAL = 0u;
static const uint LOSS_SLOT_MSE = 1u;
static const uint LOSS_SLOT_DEPTH_RATIO = 2u;
static const float GOLDEN_ANGLE_RADIANS = 2.39996322972865332;

uniform uint g_Width;
uniform uint g_Height;
uniform float g_InvPixelCount;
uniform float g_LossGradClip;
uniform float g_HugeValue;
uniform uint g_BatchStepIndex;

uniform Texture2D<float4> g_Rendered;
uniform Texture2D<float4> g_Target;
uniform StructuredBuffer<float> g_RenderedDepthRatio;
uniform RWStructuredBuffer<float4> g_OutputGrad;
uniform RWStructuredBuffer<float> g_DepthRatioGrad;
uniform RWStructuredBuffer<float> g_LossBuffer;
uniform RWStructuredBuffer<float> g_DepthRatioStats;
uniform RWStructuredBuffer<float4> g_BatchStepInfo;
uniform float g_DepthRatioWeight;

uniform Texture2D<float4> g_SourceTarget;
uniform RWTexture2D<float4> g_DownscaledTarget;
uniform uint g_SourceWidth;
uniform uint g_SourceHeight;
uniform uint g_TargetWidth;
uniform uint g_TargetHeight;
uniform uint g_DownscaleFactor;

uniform Texture2D<float4> g_DebugRendered;
uniform Texture2D<float4> g_DebugTarget;
uniform RWTexture2D<float4> g_DebugOutput;
uniform uint g_DebugWidth;
uniform uint g_DebugHeight;
uniform float g_DebugDiffScale;
uniform Texture2D<float4> g_LetterboxSource;
uniform RWTexture2D<float4> g_LetterboxOutput;
uniform uint g_LetterboxSourceWidth;
uniform uint g_LetterboxSourceHeight;
uniform uint g_LetterboxOutputWidth;
uniform uint g_LetterboxOutputHeight;

uniform RWStructuredBuffer<float> g_SrcSplatParams;
uniform StructuredBuffer<float2> g_SrcAdamMoments;
uniform RWStructuredBuffer<float> g_DstSplatParams;
uniform RWStructuredBuffer<float2> g_DstAdamMoments;
uniform RWStructuredBuffer<float> g_AppendParams;
uniform RWStructuredBuffer<uint> g_CloneCounts;
uniform RWStructuredBuffer<uint> g_TotalCloneCounter;
uniform RWStructuredBuffer<uint> g_AppendCounter;
uniform StructuredBuffer<float4> g_MaintenanceCameraRows;
uniform uint g_SrcSplatCount;
uniform uint g_DstSplatCount;
uniform uint g_AppendSplatCount;
uniform uint g_SurvivorCount;
uniform uint g_MaintenanceSeed;
uniform uint g_MaintenanceCameraCount;
uniform float g_MaintenanceAlphaCullThreshold;
uniform float g_MaintenanceRadiusScale;

groupshared float gsLoss[TRAIN_TILE_DIM * TRAIN_TILE_DIM];
groupshared float gsMSE[TRAIN_TILE_DIM * TRAIN_TILE_DIM];
groupshared float gsDepthRatioLoss[TRAIN_TILE_DIM * TRAIN_TILE_DIM];

struct SplatState
{
    float3 position;
    float3 logScale;
    float4 rotation;
    float4 colorAlpha;
};

static const uint MAINTENANCE_CAMERA_ROW_COUNT = 6u;

SplatState load_splat_state(RWStructuredBuffer<float> params, uint splatCount, uint splatId)
{
    SplatState state;
    state.position = float3(
        load_splat_param(params, splatCount, PARAM_POSITION_X, splatId),
        load_splat_param(params, splatCount, PARAM_POSITION_Y, splatId),
        load_splat_param(params, splatCount, PARAM_POSITION_Z, splatId));
    state.logScale = float3(
        load_splat_param(params, splatCount, PARAM_SCALE_X, splatId),
        load_splat_param(params, splatCount, PARAM_SCALE_Y, splatId),
        load_splat_param(params, splatCount, PARAM_SCALE_Z, splatId));
    state.rotation = normalize_quaternion_safe(float4(
        load_splat_param(params, splatCount, PARAM_ROTATION_X, splatId),
        load_splat_param(params, splatCount, PARAM_ROTATION_Y, splatId),
        load_splat_param(params, splatCount, PARAM_ROTATION_Z, splatId),
        load_splat_param(params, splatCount, PARAM_ROTATION_W, splatId)));
    state.colorAlpha = float4(
        load_splat_param(params, splatCount, PARAM_COLOR_R, splatId),
        load_splat_param(params, splatCount, PARAM_COLOR_G, splatId),
        load_splat_param(params, splatCount, PARAM_COLOR_B, splatId),
        load_splat_param(params, splatCount, PARAM_RAW_OPACITY, splatId));
    return state;
}

void store_splat_state(RWStructuredBuffer<float> params, uint splatCount, uint splatId, SplatState state)
{
    store_splat_param(params, splatCount, PARAM_POSITION_X, splatId, state.position.x);
    store_splat_param(params, splatCount, PARAM_POSITION_Y, splatId, state.position.y);
    store_splat_param(params, splatCount, PARAM_POSITION_Z, splatId, state.position.z);
    store_splat_param(params, splatCount, PARAM_SCALE_X, splatId, state.logScale.x);
    store_splat_param(params, splatCount, PARAM_SCALE_Y, splatId, state.logScale.y);
    store_splat_param(params, splatCount, PARAM_SCALE_Z, splatId, state.logScale.z);
    store_splat_param(params, splatCount, PARAM_ROTATION_X, splatId, state.rotation.x);
    store_splat_param(params, splatCount, PARAM_ROTATION_Y, splatId, state.rotation.y);
    store_splat_param(params, splatCount, PARAM_ROTATION_Z, splatId, state.rotation.z);
    store_splat_param(params, splatCount, PARAM_ROTATION_W, splatId, state.rotation.w);
    store_splat_param(params, splatCount, PARAM_COLOR_R, splatId, state.colorAlpha.x);
    store_splat_param(params, splatCount, PARAM_COLOR_G, splatId, state.colorAlpha.y);
    store_splat_param(params, splatCount, PARAM_COLOR_B, splatId, state.colorAlpha.z);
    store_splat_param(params, splatCount, PARAM_RAW_OPACITY, splatId, state.colorAlpha.w);
}

uint adam_moment_index(uint splatCount, uint paramId, uint splatId)
{
    return paramId * splatCount + splatId;
}

float2 load_adam_moment(StructuredBuffer<float2> moments, uint splatCount, uint paramId, uint splatId)
{
    return moments[adam_moment_index(splatCount, paramId, splatId)];
}

void store_adam_moment(RWStructuredBuffer<float2> moments, uint splatCount, uint paramId, uint splatId, float2 value)
{
    moments[adam_moment_index(splatCount, paramId, splatId)] = value;
}

PinholeCamera maintenance_camera(uint cameraIndex)
{
    uint baseRow = cameraIndex * MAINTENANCE_CAMERA_ROW_COUNT;
    float4 row0 = g_MaintenanceCameraRows[baseRow + 0u];
    float4 row1 = g_MaintenanceCameraRows[baseRow + 1u];
    float4 row2 = g_MaintenanceCameraRows[baseRow + 2u];
    float4 row3 = g_MaintenanceCameraRows[baseRow + 3u];
    float4 row4 = g_MaintenanceCameraRows[baseRow + 4u];
    float4 row5 = g_MaintenanceCameraRows[baseRow + 5u];

    CameraParams params;
    params.viewport = row0.xy;
    params.camPos = float3(row0.z, row0.w, row1.x);
    params.camBasis = float3x3(
        float3(row3.y, row3.z, row3.w),
        float3(row4.x, row4.y, row4.z),
        float3(row4.w, row5.x, row5.y)
    );
    params.focalPixels = row1.yz;
    params.principalPoint = float2(row1.w, row2.x);
    params.nearDepth = row2.y;
    params.farDepth = row2.z;
    params.projDistortionK1 = row2.w;
    params.projDistortionK2 = row3.x;
    return PinholeCamera.from_params(params);
}

float3 apply_support_radius_lower_bound(float3 logScale, float supportRadius)
{
    float currentMaxScale = max(exp(logScale.x), max(exp(logScale.y), exp(logScale.z)));
    if (!(currentMaxScale > SMALL_VALUE) || !(supportRadius > SMALL_VALUE)) return logScale;

    float currentSupportRadius = currentMaxScale * max(g_MaintenanceRadiusScale, SMALL_VALUE) * GAUSSIAN_SUPPORT_SIGMA_RADIUS;
    if (currentSupportRadius >= supportRadius) return logScale;
    return logScale + log(supportRadius / currentSupportRadius);
}

void copy_adam_state(uint srcSplatId, uint dstSplatId)
{
    [unroll] for (uint paramId = 0u; paramId < TRAINABLE_PARAM_COUNT; paramId++)
    {
        store_adam_moment(
            g_DstAdamMoments,
            g_DstSplatCount,
            paramId,
            dstSplatId,
            load_adam_moment(g_SrcAdamMoments, g_SrcSplatCount, paramId, srcSplatId)
        );
    }
}

float clone_unit_float(uint value)
{
    return (float(value) + 0.5) * 2.3283064365386963e-10;
}

float3 split_direction(uint splatId, uint slot, uint familySize)
{
    if (slot == 0u || familySize <= 1u) return 0.0;
    float index = float(slot - 1u);
    float count = max(float(familySize - 1u), 1.0);
    float z = 1.0 - 2.0 * ((index + 0.5) / count);
    float r = sqrt(max(1.0 - z * z, 0.0));
    float angle = GOLDEN_ANGLE_RADIANS * (index + clone_unit_float(splatId ^ g_MaintenanceSeed));
    float s;
    float c;
    sincos(angle, s, c);
    return float3(r * c, r * s, z);
}

SplatState split_family_member(SplatState source, uint splatId, uint slot, uint familySize)
{
    if (familySize <= 1u) return source;
    float family = float(familySize);
    float shrink = log(family) / 3.0;
    float3 parentScale = exp(source.logScale);
    float3 childScale = exp(source.logScale - shrink);
    float alpha = opacity_from_raw(source.colorAlpha.w);
    float childAlpha = 1.0 - pow(max(1.0 - alpha, TINY_VALUE), 1.0 / family);
    float4 qInv = float4(source.rotation.x, -source.rotation.y, -source.rotation.z, -source.rotation.w);
    float3 localOffset = split_direction(splatId, slot, familySize) * max(parentScale - childScale, 0.0);
    SplatState child = source;
    child.position = source.position + rotate_by_quat(localOffset, qInv);
    child.logScale = source.logScale - shrink;
    child.colorAlpha.w = raw_opacity_from_alpha(childAlpha);
    return child;
}

[shader("compute")]
[numthreads(TRAIN_TILE_DIM, TRAIN_TILE_DIM, 1)]
void csResampleDownscaledTargetNearest(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= g_TargetWidth || py >= g_TargetHeight) return;

    uint factor = max(g_DownscaleFactor, 1u);
    uint srcX0 = px * factor;
    uint srcY0 = py * factor;
    uint srcX1 = min(srcX0 + factor, g_SourceWidth);
    uint srcY1 = min(srcY0 + factor, g_SourceHeight);
    float4 accum = 0.0;
    uint sampleCount = 0u;
    for (uint srcY = srcY0; srcY < srcY1; srcY++)
    {
        for (uint srcX = srcX0; srcX < srcX1; srcX++)
        {
            accum += g_SourceTarget[uint2(srcX, srcY)];
            sampleCount += 1u;
        }
    }
    g_DownscaledTarget[uint2(px, py)] = accum / max(float(sampleCount), 1.0);
}

[shader("compute")]
[numthreads(TRAIN_TILE_DIM, TRAIN_TILE_DIM, 1)]
void csClearLossBuffer(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px == 0u && py == 0u)
    {
        g_LossBuffer[LOSS_SLOT_TOTAL] = 0.0;
        g_LossBuffer[LOSS_SLOT_MSE] = 0.0;
        g_LossBuffer[LOSS_SLOT_DEPTH_RATIO] = 0.0;
        g_DepthRatioStats[0] = 0.0;
        g_DepthRatioStats[1] = 0.0;
        g_DepthRatioStats[2] = 0.0;
        g_DepthRatioStats[3] = 0.0;
    }
}

[shader("compute")]
[numthreads(TRAIN_TILE_DIM, TRAIN_TILE_DIM, 1)]
void csComputeL1LossForward(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupThreadID : SV_GroupThreadID
)
{
    uint localIndex = groupThreadID.y * TRAIN_TILE_DIM + groupThreadID.x;
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;

    float localLoss = 0.0;
    float localMSE = 0.0;
    float localDepthRatioLoss = 0.0;
    if (px < g_Width && py < g_Height)
    {
        uint2 pixel = uint2(px, py);
        float3 rendered = sanitize_float3(g_Rendered[pixel].xyz, g_HugeValue);
        float3 target = sanitize_float3(g_Target[pixel].xyz, g_HugeValue);
        float loss = rgb_l1_loss(rendered, target, g_InvPixelCount);
        float mse = rgb_mse_loss(rendered, target, g_InvPixelCount);
        float depthRatio = sanitize_scalar(g_RenderedDepthRatio[py * g_Width + px], g_HugeValue);
        localLoss = isfinite(loss) ? loss : 0.0;
        localMSE = isfinite(mse) ? mse : 0.0;
        if (g_DepthRatioWeight > 0.0 && depthRatio > 0.0)
        {
            float depthLoss = g_DepthRatioWeight * depthRatio * depthRatio * g_InvPixelCount;
            localDepthRatioLoss = isfinite(depthLoss) ? depthLoss : 0.0;
        }
    }

    gsLoss[localIndex] = localLoss;
    gsMSE[localIndex] = localMSE;
    gsDepthRatioLoss[localIndex] = localDepthRatioLoss;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = (TRAIN_TILE_DIM * TRAIN_TILE_DIM) >> 1; stride > 0; stride >>= 1)
    {
        if (localIndex < stride) gsLoss[localIndex] += gsLoss[localIndex + stride];
        if (localIndex < stride) gsMSE[localIndex] += gsMSE[localIndex + stride];
        if (localIndex < stride) gsDepthRatioLoss[localIndex] += gsDepthRatioLoss[localIndex + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (localIndex == 0u)
    {
        if (isfinite(gsLoss[0])) InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], gsLoss[0]);
        if (isfinite(gsMSE[0])) InterlockedAdd(g_LossBuffer[LOSS_SLOT_MSE], gsMSE[0]);
        if (isfinite(gsDepthRatioLoss[0]))
        {
            InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], gsDepthRatioLoss[0]);
            InterlockedAdd(g_LossBuffer[LOSS_SLOT_DEPTH_RATIO], gsDepthRatioLoss[0]);
        }
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void csFinalizeDepthRatioLoss(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u || dispatchThreadID.y != 0u || dispatchThreadID.z != 0u) return;
    g_DepthRatioStats[3] = g_LossBuffer[LOSS_SLOT_DEPTH_RATIO];
}

[shader("compute")]
[numthreads(TRAIN_TILE_DIM, TRAIN_TILE_DIM, 1)]
void csComputeL1LossBackward(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= g_Width || py >= g_Height) return;

    uint2 pixel = uint2(px, py);
    float3 rendered = sanitize_float3(g_Rendered[pixel].xyz, g_HugeValue);
    float3 target = sanitize_float3(g_Target[pixel].xyz, g_HugeValue);
    float3 dOut = clamp(sanitize_float3(rgb_l1_grad(rendered, target) * g_InvPixelCount, g_LossGradClip), -g_LossGradClip, g_LossGradClip);
    g_OutputGrad[py * g_Width + px] = float4(dOut, 0.0);
    float depthRatio = sanitize_scalar(g_RenderedDepthRatio[py * g_Width + px], g_HugeValue);
    if (g_DepthRatioWeight <= 0.0 || depthRatio <= 0.0)
    {
        g_DepthRatioGrad[py * g_Width + px] = 0.0;
        return;
    }

    float grad = 2.0 * g_DepthRatioWeight * depthRatio * g_InvPixelCount;
    g_DepthRatioGrad[py * g_Width + px] = clamp(sanitize_scalar(grad, g_LossGradClip), -g_LossGradClip, g_LossGradClip);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void csCacheTrainingStepInfo(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u || dispatchThreadID.y != 0u || dispatchThreadID.z != 0u) return;
    g_BatchStepInfo[g_BatchStepIndex] = float4(g_LossBuffer[LOSS_SLOT_TOTAL], g_LossBuffer[LOSS_SLOT_MSE], g_LossBuffer[LOSS_SLOT_DEPTH_RATIO], 0.0);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csClearCloneCounts(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId == 0u)
    {
        g_TotalCloneCounter[0] = 0u;
        g_AppendCounter[0] = 0u;
    }
    if (splatId >= g_SrcSplatCount) return;
    g_CloneCounts[splatId] = 0u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csClearMaintenanceCounters(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u) return;
    g_TotalCloneCounter[0] = 0u;
    g_AppendCounter[0] = 0u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csPrepareMaintenanceCounts(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SrcSplatCount) return;
    float alpha = opacity_from_raw(load_splat_param(g_SrcSplatParams, g_SrcSplatCount, PARAM_RAW_OPACITY, splatId));
    if (alpha < g_MaintenanceAlphaCullThreshold) return;
    uint _;
    InterlockedAdd(g_AppendCounter[0], 1u, _);
    uint cloneCount = g_CloneCounts[splatId];
    if (cloneCount > 0u) InterlockedAdd(g_TotalCloneCounter[0], cloneCount, _);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csClampMaintenanceMinScreenSize(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SrcSplatCount) return;
    if (g_MaintenanceCameraCount == 0u) return;

    SplatState source = load_splat_state(g_SrcSplatParams, g_SrcSplatCount, splatId);
    float minVisibleSupportRadius = 0.0;
    bool hasVisibleCamera = false;
    [loop] for (uint cameraIndex = 0u; cameraIndex < g_MaintenanceCameraCount; cameraIndex++)
    {
        float supportRadius;
        if (!world_support_radius_from_screen_circle(maintenance_camera(cameraIndex), source.position, MAINTENANCE_MIN_SCREEN_RADIUS_PX, supportRadius)) continue;
        if (!hasVisibleCamera || supportRadius < minVisibleSupportRadius)
        {
            minVisibleSupportRadius = supportRadius;
            hasVisibleCamera = true;
        }
    }
    if (!hasVisibleCamera) return;

    source.logScale = apply_support_radius_lower_bound(source.logScale, minVisibleSupportRadius);
    store_splat_state(g_SrcSplatParams, g_SrcSplatCount, splatId, source);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csRewriteMaintenanceSplats(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SrcSplatCount) return;

    SplatState source = load_splat_state(g_SrcSplatParams, g_SrcSplatCount, splatId);
    float alpha = opacity_from_raw(source.colorAlpha.w);
    if (alpha < g_MaintenanceAlphaCullThreshold) return;

    uint cloneCount = g_CloneCounts[splatId];
    uint parentIndex = 0u;
    InterlockedAdd(g_AppendCounter[0], 1u, parentIndex);

    uint emittedCloneCount = 0u;
    if (cloneCount > 0u)
    {
        uint childBase = 0u;
        InterlockedAdd(g_TotalCloneCounter[0], cloneCount, childBase);
        uint availableChildren = childBase < g_AppendSplatCount ? (g_AppendSplatCount - childBase) : 0u;
        emittedCloneCount = min(cloneCount, availableChildren);
        uint familySize = emittedCloneCount + 1u;
        store_splat_state(g_DstSplatParams, g_DstSplatCount, parentIndex, split_family_member(source, splatId, 0u, familySize));
        copy_adam_state(splatId, parentIndex);
        [loop] for (uint childSlot = 0u; childSlot < emittedCloneCount; childSlot++)
        {
            SplatState child = split_family_member(source, splatId, childSlot + 1u, familySize);
            uint childIndex = g_SurvivorCount + childBase + childSlot;
            store_splat_state(g_AppendParams, g_AppendSplatCount, childBase + childSlot, child);
            store_splat_state(g_DstSplatParams, g_DstSplatCount, childIndex, child);
            copy_adam_state(splatId, childIndex);
        }
        return;
    }

    store_splat_state(g_DstSplatParams, g_DstSplatCount, parentIndex, source);
    copy_adam_state(splatId, parentIndex);
}

[shader("compute")]
[numthreads(TRAIN_TILE_DIM, TRAIN_TILE_DIM, 1)]
void csComposeAbsDiffDebug(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= g_DebugWidth || py >= g_DebugHeight) return;
    uint2 pixel = uint2(px, py);
    float3 rendered = g_DebugRendered[pixel].xyz;
    float3 target = g_DebugTarget[pixel].xyz;
    float3 diff = abs(sanitize_float3(rendered - target, g_HugeValue)) * max(g_DebugDiffScale, 0.0);
    g_DebugOutput[pixel] = float4(diff, 1.0);
}

[shader("compute")]
[numthreads(TRAIN_TILE_DIM, TRAIN_TILE_DIM, 1)]
void csComposeLetterboxDebug(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= g_LetterboxOutputWidth || py >= g_LetterboxOutputHeight) return;

    float srcW = max(float(g_LetterboxSourceWidth), 1.0);
    float srcH = max(float(g_LetterboxSourceHeight), 1.0);
    float dstW = max(float(g_LetterboxOutputWidth), 1.0);
    float dstH = max(float(g_LetterboxOutputHeight), 1.0);
    float srcAspect = srcW / srcH;
    float dstAspect = dstW / dstH;

    float fitW = dstW;
    float fitH = dstH;
    if (dstAspect > srcAspect) fitW = dstH * srcAspect;
    else fitH = dstW / srcAspect;

    float offsetX = 0.5 * (dstW - fitW);
    float offsetY = 0.5 * (dstH - fitH);
    float sampleX = float(px) + 0.5;
    float sampleY = float(py) + 0.5;
    bool inside = sampleX >= offsetX && sampleX < (offsetX + fitW) && sampleY >= offsetY && sampleY < (offsetY + fitH);
    if (!inside)
    {
        g_LetterboxOutput[uint2(px, py)] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    float u = (sampleX - offsetX) / max(fitW, TINY_VALUE);
    float v = (sampleY - offsetY) / max(fitH, TINY_VALUE);
    uint srcX = min(uint(u * srcW), g_LetterboxSourceWidth - 1u);
    uint srcY = min(uint(v * srcH), g_LetterboxSourceHeight - 1u);
    g_LetterboxOutput[uint2(px, py)] = g_LetterboxSource[uint2(srcX, srcY)];
}
