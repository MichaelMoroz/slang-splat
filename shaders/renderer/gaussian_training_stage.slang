import "utility/losses/losses.slang";
import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/math/numeric.slang";
import "utility/math/quaternion.slang";
import "utility/random/random.slang";
import "utility/splatting/common.slang";
import "utility/splatting/opacity.slang";
import "utility/splatting/params.slang";

static const uint TRAIN_TILE_DIM = 16u;
static const uint LOSS_SLOT_TOTAL = 0u;
static const uint LOSS_SLOT_MSE = 1u;
static const uint LOSS_SLOT_DENSITY = 2u;
static const float POSITION_RANDOM_STEP_MAX_ANISOTROPY = 4.0;
static const float DENSIFY_SCALE_SHRINK_EXPONENT = -1.0 / 3.0;
static const float DENSIFY_MAX_STANDARDIZED_SIGMA = 3.0;
static const float SRGB_LINEAR_THRESHOLD = 0.0031308;
static const float SRGB_LINEAR_SCALE = 12.92;
static const float SRGB_GAMMA_SCALE = 1.055;
static const float SRGB_GAMMA_OFFSET = 0.055;
static const float SRGB_INV_GAMMA = 1.0 / 2.4;
static const uint MAINTENANCE_HASH_INIT = 0x9e3779b9u;
static const uint MAINTENANCE_HASH_MIX = 0x85ebca6bu;
uniform uint g_Width;
uniform uint g_Height;
uniform float g_InvPixelCount;
uniform float g_LossGradClip;
uniform float g_HugeValue;
uniform uint g_BatchStepIndex;

uniform Texture2D<float4> g_Rendered;
uniform Texture2D<float4> g_Target;
uniform StructuredBuffer<float> g_RenderedDensity;
uniform RWStructuredBuffer<float4> g_OutputGrad;
uniform RWStructuredBuffer<float> g_DensityGrad;
uniform RWStructuredBuffer<float> g_LossBuffer;
uniform RWStructuredBuffer<float4> g_BatchStepInfo;
uniform float g_DensityRegularizer;
uniform float g_MaxAllowedDensity;
uniform RWStructuredBuffer<float> g_PositionRandomStepParams;
uniform uint g_PositionRandomStepSplatCount;
uniform uint g_PositionRandomStepSeed;
uniform float g_PositionRandomStepNoiseLr;
uniform float g_PositionRandomStepPositionLr;
uniform float g_PositionRandomStepOpacityGateCenter;
uniform float g_PositionRandomStepOpacityGateSharpness;

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
uniform RWStructuredBuffer<uint> g_SplatContribution;
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
uniform uint g_MaintenanceContributionCullThreshold;
uniform float g_MaintenanceRadiusScale;

groupshared float gsLoss[TRAIN_TILE_DIM * TRAIN_TILE_DIM];
groupshared float gsMSE[TRAIN_TILE_DIM * TRAIN_TILE_DIM];
groupshared float gsDensityLoss[TRAIN_TILE_DIM * TRAIN_TILE_DIM];

struct SplatState
{
    float3 position;
    float3 logScale;
    float4 rotation;
    float3 sh0;
    float3 sh1X;
    float3 sh1Y;
    float3 sh1Z;
    float rawOpacity;
};

static const uint MAINTENANCE_CAMERA_ROW_COUNT = 6u;

float linear_to_srgb_safe(float value)
{
    if (!(value > 0.0)) return 0.0;
    if (value <= SRGB_LINEAR_THRESHOLD) return SRGB_LINEAR_SCALE * value;
    return SRGB_GAMMA_SCALE * pow(value, SRGB_INV_GAMMA) - SRGB_GAMMA_OFFSET;
}

float3 linear_to_srgb_safe(float3 value)
{
    return float3(linear_to_srgb_safe(value.x), linear_to_srgb_safe(value.y), linear_to_srgb_safe(value.z));
}

bool should_cull_maintenance_splat(float alpha, uint contribution)
{
    return alpha < g_MaintenanceAlphaCullThreshold || contribution < g_MaintenanceContributionCullThreshold;
}

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
    state.sh0 = float3(
        load_splat_param(params, splatCount, PARAM_SH0_R, splatId),
        load_splat_param(params, splatCount, PARAM_SH0_G, splatId),
        load_splat_param(params, splatCount, PARAM_SH0_B, splatId));
    state.sh1X = float3(
        load_splat_param(params, splatCount, PARAM_SH1_X_R, splatId),
        load_splat_param(params, splatCount, PARAM_SH1_X_G, splatId),
        load_splat_param(params, splatCount, PARAM_SH1_X_B, splatId));
    state.sh1Y = float3(
        load_splat_param(params, splatCount, PARAM_SH1_Y_R, splatId),
        load_splat_param(params, splatCount, PARAM_SH1_Y_G, splatId),
        load_splat_param(params, splatCount, PARAM_SH1_Y_B, splatId));
    state.sh1Z = float3(
        load_splat_param(params, splatCount, PARAM_SH1_Z_R, splatId),
        load_splat_param(params, splatCount, PARAM_SH1_Z_G, splatId),
        load_splat_param(params, splatCount, PARAM_SH1_Z_B, splatId));
    state.rawOpacity = load_splat_param(params, splatCount, PARAM_RAW_OPACITY, splatId);
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
    store_splat_param(params, splatCount, PARAM_SH0_R, splatId, state.sh0.x);
    store_splat_param(params, splatCount, PARAM_SH0_G, splatId, state.sh0.y);
    store_splat_param(params, splatCount, PARAM_SH0_B, splatId, state.sh0.z);
    store_splat_param(params, splatCount, PARAM_SH1_X_R, splatId, state.sh1X.x);
    store_splat_param(params, splatCount, PARAM_SH1_X_G, splatId, state.sh1X.y);
    store_splat_param(params, splatCount, PARAM_SH1_X_B, splatId, state.sh1X.z);
    store_splat_param(params, splatCount, PARAM_SH1_Y_R, splatId, state.sh1Y.x);
    store_splat_param(params, splatCount, PARAM_SH1_Y_G, splatId, state.sh1Y.y);
    store_splat_param(params, splatCount, PARAM_SH1_Y_B, splatId, state.sh1Y.z);
    store_splat_param(params, splatCount, PARAM_SH1_Z_R, splatId, state.sh1Z.x);
    store_splat_param(params, splatCount, PARAM_SH1_Z_G, splatId, state.sh1Z.y);
    store_splat_param(params, splatCount, PARAM_SH1_Z_B, splatId, state.sh1Z.z);
    store_splat_param(params, splatCount, PARAM_RAW_OPACITY, splatId, state.rawOpacity);
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

uint maintenance_hash_combine(uint seed, uint value)
{
    return hash_u32(seed ^ (value * MAINTENANCE_HASH_MIX + MAINTENANCE_HASH_INIT));
}

uint maintenance_family_frame_hash(uint splatId)
{
    uint seed = maintenance_hash_combine(g_MaintenanceSeed, splatId + 1u);
    if (g_MaintenanceCameraCount == 0u) return seed;
    uint cameraIndex = seed % g_MaintenanceCameraCount;
    uint baseRow = cameraIndex * MAINTENANCE_CAMERA_ROW_COUNT;
    return maintenance_hash_combine(asuint(g_MaintenanceCameraRows[baseRow + 5u].z), cameraIndex + 1u);
}

uint maintenance_sample_seed(uint frameHash, uint splatId, uint subsampleId)
{
    return maintenance_hash_combine(maintenance_hash_combine(frameHash, splatId + 1u), subsampleId + 1u);
}

float3 maintenance_gaussian_sample_raw(uint frameHash, uint splatId, uint subsampleId)
{
    uint rngState = maintenance_sample_seed(frameHash, splatId, subsampleId);
    return rng_normal3_box_muller(rngState);
}

float3 maintenance_gaussian_family_mean(uint frameHash, uint splatId, uint familySize)
{
    if (familySize <= 1u) return 0.0;
    float3 mean = 0.0;
    [loop] for (uint slot = 0u; slot < familySize; slot++) mean += maintenance_gaussian_sample_raw(frameHash, splatId, slot);
    return mean / max(float(familySize), 1.0);
}

float3 maintenance_centered_gaussian_sample(uint frameHash, uint splatId, uint slot, uint familySize, float3 familyMean)
{
    if (familySize <= 1u) return 0.0;
    float varianceRestore = sqrt(float(familySize) / float(familySize - 1u));
    float3 sample = (maintenance_gaussian_sample_raw(frameHash, splatId, slot) - familyMean) * varianceRestore;
    float sampleLength = length(sample);
    return sampleLength > DENSIFY_MAX_STANDARDIZED_SIGMA ? sample * (DENSIFY_MAX_STANDARDIZED_SIGMA / sampleLength) : sample;
}

float position_random_step_gate(float opacity)
{
    float x = g_PositionRandomStepOpacityGateSharpness * (g_PositionRandomStepOpacityGateCenter - opacity);
    return rcp(1.0 + exp(-x));
}

float3 position_random_step_world_delta(SplatState state, inout uint rngState)
{
    float opacity = opacity_from_raw(state.rawOpacity);
    float gate = position_random_step_gate(opacity);
    float stepScale = gate * max(g_PositionRandomStepNoiseLr, 0.0) * max(g_PositionRandomStepPositionLr, 0.0);
    if (!(stepScale > 0.0)) return 0.0;

    float3 scale = exp(state.logScale);
    float maxScale = max(scale.x, max(scale.y, scale.z));
    float3 clampedScale = max(scale, float3(maxScale / POSITION_RANDOM_STEP_MAX_ANISOTROPY));
    float3 localDelta = rng_normal3_box_muller(rngState) * clampedScale * clampedScale * stepScale;
    float4 qInv = float4(state.rotation.x, -state.rotation.y, -state.rotation.z, -state.rotation.w);
    return rotate_by_quat(localDelta, qInv);
}

SplatState split_family_member(SplatState source, uint splatId, uint slot, uint familySize, uint frameHash, float3 familyMean)
{
    if (familySize <= 1u) return source;
    float family = float(familySize);
    float shrink = pow(family, DENSIFY_SCALE_SHRINK_EXPONENT);
    float3 parentScale = exp(source.logScale);
    float shrinkOffset = sqrt(max(1.0 - shrink * shrink, 0.0));
    float4 qInv = float4(source.rotation.x, -source.rotation.y, -source.rotation.z, -source.rotation.w);
    float3 localOffset = maintenance_centered_gaussian_sample(frameHash, splatId, slot, familySize, familyMean) * parentScale * shrinkOffset;
    SplatState child = source;
    child.position = source.position + rotate_by_quat(localOffset, qInv);
    child.logScale = source.logScale + log(shrink);
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
        g_LossBuffer[LOSS_SLOT_DENSITY] = 0.0;
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
    float localDensityLoss = 0.0;
    if (px < g_Width && py < g_Height)
    {
        uint2 pixel = uint2(px, py);
        float3 rendered = sanitize_float3(g_Rendered[pixel].xyz, g_HugeValue);
        float3 target = sanitize_float3(g_Target[pixel].xyz, g_HugeValue);
        float loss = rgb_l1_loss(rendered, target, g_InvPixelCount);
        float mse = rgb_mse_loss(rendered, target, g_InvPixelCount);
        float density = sanitize_scalar(g_RenderedDensity[py * g_Width + px], g_HugeValue);
        localLoss = isfinite(loss) ? loss : 0.0;
        localMSE = isfinite(mse) ? mse : 0.0;
        if (g_DensityRegularizer > 0.0)
        {
            float densityExcess = max(density - g_MaxAllowedDensity, 0.0);
            float densityLoss = g_DensityRegularizer * densityExcess * g_InvPixelCount;
            localDensityLoss = isfinite(densityLoss) ? densityLoss : 0.0;
        }
    }

    gsLoss[localIndex] = localLoss;
    gsMSE[localIndex] = localMSE;
    gsDensityLoss[localIndex] = localDensityLoss;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = (TRAIN_TILE_DIM * TRAIN_TILE_DIM) >> 1; stride > 0; stride >>= 1)
    {
        if (localIndex < stride) gsLoss[localIndex] += gsLoss[localIndex + stride];
        if (localIndex < stride) gsMSE[localIndex] += gsMSE[localIndex + stride];
        if (localIndex < stride) gsDensityLoss[localIndex] += gsDensityLoss[localIndex + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (localIndex == 0u)
    {
        if (isfinite(gsLoss[0])) InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], gsLoss[0]);
        if (isfinite(gsMSE[0])) InterlockedAdd(g_LossBuffer[LOSS_SLOT_MSE], gsMSE[0]);
        if (isfinite(gsDensityLoss[0]))
        {
            InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], gsDensityLoss[0]);
            InterlockedAdd(g_LossBuffer[LOSS_SLOT_DENSITY], gsDensityLoss[0]);
        }
    }
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
    float density = sanitize_scalar(g_RenderedDensity[py * g_Width + px], g_HugeValue);
    if (g_DensityRegularizer <= 0.0 || density <= g_MaxAllowedDensity)
    {
        g_DensityGrad[py * g_Width + px] = 0.0;
        return;
    }

    float densityGrad = g_DensityRegularizer * g_InvPixelCount;
    g_DensityGrad[py * g_Width + px] = clamp(sanitize_scalar(densityGrad, g_LossGradClip), -g_LossGradClip, g_LossGradClip);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void csCacheTrainingStepInfo(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u || dispatchThreadID.y != 0u || dispatchThreadID.z != 0u) return;
    g_BatchStepInfo[g_BatchStepIndex] = float4(g_LossBuffer[LOSS_SLOT_TOTAL], g_LossBuffer[LOSS_SLOT_MSE], g_LossBuffer[LOSS_SLOT_DENSITY], 0.0);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csApplyPositionRandomSteps(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_PositionRandomStepSplatCount) return;
    if (!(g_PositionRandomStepNoiseLr > 0.0) || !(g_PositionRandomStepPositionLr > 0.0)) return;

    SplatState state = load_splat_state(g_PositionRandomStepParams, g_PositionRandomStepSplatCount, splatId);
    uint rngState = hash_u32(g_PositionRandomStepSeed ^ (0x9e3779b9u * (splatId + 1u)));
    state.position += position_random_step_world_delta(state, rngState);
    store_splat_state(g_PositionRandomStepParams, g_PositionRandomStepSplatCount, splatId, state);
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
    g_SplatContribution[splatId] = 0u;
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
    if (should_cull_maintenance_splat(alpha, g_SplatContribution[splatId])) return;
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
    float alpha = opacity_from_raw(source.rawOpacity);
    if (should_cull_maintenance_splat(alpha, g_SplatContribution[splatId])) return;

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
        uint frameHash = maintenance_family_frame_hash(splatId);
        float3 familyMean = maintenance_gaussian_family_mean(frameHash, splatId, familySize);
        store_splat_state(g_DstSplatParams, g_DstSplatCount, parentIndex, split_family_member(source, splatId, 0u, familySize, frameHash, familyMean));
        copy_adam_state(splatId, parentIndex);
        [loop] for (uint childSlot = 0u; childSlot < emittedCloneCount; childSlot++)
        {
            SplatState child = split_family_member(source, splatId, childSlot + 1u, familySize, frameHash, familyMean);
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
    float4 sampled = g_LetterboxSource[uint2(srcX, srcY)];
    g_LetterboxOutput[uint2(px, py)] = float4(linear_to_srgb_safe(sampled.xyz), 1.0);
}
