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
static const uint LOSS_SLOT_RGB = 3u;
static const float POSITION_RANDOM_STEP_MAX_ANISOTROPY = 4.0;
static const float DENSIFY_SCALE_SHRINK_EXPONENT = -1.0 / 3.0;
static const float DENSIFY_MAX_STANDARDIZED_SIGMA = 3.0;
static const float REFINEMENT_CHILD_ANISOTROPY_RATIO_SCALE = 0.5;
static const float REFINEMENT_FIBONACCI_RADIUS = 2.0;
static const float REFINEMENT_FIBONACCI_JITTER_RATIO = 0.2;
static const float REFINEMENT_FIBONACCI_GOLDEN_ANGLE = 2.39996322972865332;
static const float SRGB_LINEAR_THRESHOLD = 0.0031308;
static const float SRGB_LINEAR_SCALE = 12.92;
static const float SRGB_GAMMA_SCALE = 1.055;
static const float SRGB_GAMMA_OFFSET = 0.055;
static const float SRGB_INV_GAMMA = 1.0 / 2.4;
static const float DEPTH_RATIO_GRAD_MIN_BAND_WIDTH = 1e-4;
static const float DEPTH_RATIO_GRAD_SOFTNESS_RATIO = 0.1;
static const float DEPTH_RATIO_GRAD_SOFTNESS_FLOOR = 1e-4;
static const float DEBUG_EDGE_GRADIENT_DISPLAY_MAX = 4.0;
static const uint SSIM_FEATURE_CHANNELS = 15u;
static const uint SSIM_FEATURE_STRIDE = 3u;
static const uint SSIM_FEATURE_X_OFFSET = 0u;
static const uint SSIM_FEATURE_Y_OFFSET = SSIM_FEATURE_X_OFFSET + SSIM_FEATURE_STRIDE;
static const uint SSIM_FEATURE_X2_OFFSET = SSIM_FEATURE_Y_OFFSET + SSIM_FEATURE_STRIDE;
static const uint SSIM_FEATURE_Y2_OFFSET = SSIM_FEATURE_X2_OFFSET + SSIM_FEATURE_STRIDE;
static const uint SSIM_FEATURE_XY_OFFSET = SSIM_FEATURE_Y2_OFFSET + SSIM_FEATURE_STRIDE;
static const float SSIM_C1 = 0.0001;
static const float SSIM_C2 = 0.0009;
static const float3 SSIM_YCBCR_WEIGHTS = float3(4.0, 1.0, 1.0) * (1.0 / 6.0);
static const uint REFINEMENT_HASH_INIT = 0x9e3779b9u;
static const uint REFINEMENT_HASH_MIX = 0x85ebca6bu;
uniform uint g_Width;
uniform uint g_Height;
uniform float g_InvPixelCount;
uniform float g_LossGradClip;
uniform float g_HugeValue;
uniform uint g_UseTargetAlphaMask;
uniform uint g_BatchStepIndex;
uniform TrainingSubsampleParams g_TrainingSubsample;
uniform float g_SSIMWeight;

uniform Texture2D<float4> g_Rendered;
uniform Texture2D<float4> g_Target;
uniform Texture2D<float4> g_RenderedDepthStats;
uniform StructuredBuffer<float> g_RenderedDensity;
uniform RWStructuredBuffer<float4> g_OutputGrad;
uniform RWStructuredBuffer<float2> g_RegularizerGrad;
uniform RWStructuredBuffer<float> g_LossBuffer;
uniform RWStructuredBuffer<float4> g_BatchStepInfo;
uniform RWStructuredBuffer<float> g_TrainingRgbLoss;
uniform RWStructuredBuffer<float> g_TrainingRgbLossTotal;
uniform RWStructuredBuffer<float> g_TrainingTargetEdge;
uniform RWStructuredBuffer<float> g_TrainingTargetEdgeTotal;
uniform RWStructuredBuffer<float> g_SSIMMoments;
uniform StructuredBuffer<float> g_SSIMBlurredMoments;
uniform RWStructuredBuffer<float> g_SSIMBlurredFeatureGrads;
uniform StructuredBuffer<float> g_SSIMFeatureGrads;
uniform float g_DensityRegularizer;
uniform float g_DepthRatioWeight;
uniform float g_RefinementLossWeight;
uniform float g_RefinementTargetEdgeWeight;
uniform float g_DepthRatioGradMin;
uniform float g_DepthRatioGradMax;
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
uniform StructuredBuffer<float4> g_RefinementCameraRows;
uniform uint g_SrcSplatCount;
uniform uint g_DstSplatCount;
uniform uint g_AppendSplatCount;
uniform uint g_SurvivorCount;
uniform uint g_RefinementSeed;
uniform uint g_RefinementCameraCount;
uniform float g_RefinementAlphaCullThreshold;
uniform uint g_RefinementMinContributionThreshold;
uniform float g_RefinementOpacityMul;
uniform float g_RefinementRadiusScale;

groupshared float gsLoss[TRAIN_TILE_DIM * TRAIN_TILE_DIM];
groupshared float gsMSE[TRAIN_TILE_DIM * TRAIN_TILE_DIM];
groupshared float gsDensityLoss[TRAIN_TILE_DIM * TRAIN_TILE_DIM];
groupshared float gsRgbLoss[TRAIN_TILE_DIM * TRAIN_TILE_DIM];
groupshared float gsTargetEdge[TRAIN_TILE_DIM * TRAIN_TILE_DIM];

struct SSIMMoments : IDifferentiable
{
    float3 x;
    float3 y;
    float3 x2;
    float3 y2;
    float3 xy;
};

struct SplatState
{
    float3 position;
    float3 logScale;
    float4 rotation;
    float3 sh0;
    float3 sh1X;
    float3 sh1Y;
    float3 sh1Z;
    float3 sh2_0;
    float3 sh2_1;
    float3 sh2_2;
    float3 sh2_3;
    float3 sh2_4;
    float3 sh3_0;
    float3 sh3_1;
    float3 sh3_2;
    float3 sh3_3;
    float3 sh3_4;
    float3 sh3_5;
    float3 sh3_6;
    float rawOpacity;
};

static const uint REFINEMENT_CAMERA_ROW_COUNT = 6u;

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

bool should_cull_refinement_splat(float alpha, uint contribution)
{
    return alpha < g_RefinementAlphaCullThreshold || contribution < g_RefinementMinContributionThreshold;
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
    state.sh2_0 = float3(
        load_splat_param(params, splatCount, sh_param_id(4u, 0u), splatId),
        load_splat_param(params, splatCount, sh_param_id(4u, 1u), splatId),
        load_splat_param(params, splatCount, sh_param_id(4u, 2u), splatId));
    state.sh2_1 = float3(
        load_splat_param(params, splatCount, sh_param_id(5u, 0u), splatId),
        load_splat_param(params, splatCount, sh_param_id(5u, 1u), splatId),
        load_splat_param(params, splatCount, sh_param_id(5u, 2u), splatId));
    state.sh2_2 = float3(
        load_splat_param(params, splatCount, sh_param_id(6u, 0u), splatId),
        load_splat_param(params, splatCount, sh_param_id(6u, 1u), splatId),
        load_splat_param(params, splatCount, sh_param_id(6u, 2u), splatId));
    state.sh2_3 = float3(
        load_splat_param(params, splatCount, sh_param_id(7u, 0u), splatId),
        load_splat_param(params, splatCount, sh_param_id(7u, 1u), splatId),
        load_splat_param(params, splatCount, sh_param_id(7u, 2u), splatId));
    state.sh2_4 = float3(
        load_splat_param(params, splatCount, sh_param_id(8u, 0u), splatId),
        load_splat_param(params, splatCount, sh_param_id(8u, 1u), splatId),
        load_splat_param(params, splatCount, sh_param_id(8u, 2u), splatId));
    state.sh3_0 = float3(
        load_splat_param(params, splatCount, sh_param_id(9u, 0u), splatId),
        load_splat_param(params, splatCount, sh_param_id(9u, 1u), splatId),
        load_splat_param(params, splatCount, sh_param_id(9u, 2u), splatId));
    state.sh3_1 = float3(
        load_splat_param(params, splatCount, sh_param_id(10u, 0u), splatId),
        load_splat_param(params, splatCount, sh_param_id(10u, 1u), splatId),
        load_splat_param(params, splatCount, sh_param_id(10u, 2u), splatId));
    state.sh3_2 = float3(
        load_splat_param(params, splatCount, sh_param_id(11u, 0u), splatId),
        load_splat_param(params, splatCount, sh_param_id(11u, 1u), splatId),
        load_splat_param(params, splatCount, sh_param_id(11u, 2u), splatId));
    state.sh3_3 = float3(
        load_splat_param(params, splatCount, sh_param_id(12u, 0u), splatId),
        load_splat_param(params, splatCount, sh_param_id(12u, 1u), splatId),
        load_splat_param(params, splatCount, sh_param_id(12u, 2u), splatId));
    state.sh3_4 = float3(
        load_splat_param(params, splatCount, sh_param_id(13u, 0u), splatId),
        load_splat_param(params, splatCount, sh_param_id(13u, 1u), splatId),
        load_splat_param(params, splatCount, sh_param_id(13u, 2u), splatId));
    state.sh3_5 = float3(
        load_splat_param(params, splatCount, sh_param_id(14u, 0u), splatId),
        load_splat_param(params, splatCount, sh_param_id(14u, 1u), splatId),
        load_splat_param(params, splatCount, sh_param_id(14u, 2u), splatId));
    state.sh3_6 = float3(
        load_splat_param(params, splatCount, sh_param_id(15u, 0u), splatId),
        load_splat_param(params, splatCount, sh_param_id(15u, 1u), splatId),
        load_splat_param(params, splatCount, sh_param_id(15u, 2u), splatId));
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
    store_splat_param(params, splatCount, sh_param_id(4u, 0u), splatId, state.sh2_0.x);
    store_splat_param(params, splatCount, sh_param_id(4u, 1u), splatId, state.sh2_0.y);
    store_splat_param(params, splatCount, sh_param_id(4u, 2u), splatId, state.sh2_0.z);
    store_splat_param(params, splatCount, sh_param_id(5u, 0u), splatId, state.sh2_1.x);
    store_splat_param(params, splatCount, sh_param_id(5u, 1u), splatId, state.sh2_1.y);
    store_splat_param(params, splatCount, sh_param_id(5u, 2u), splatId, state.sh2_1.z);
    store_splat_param(params, splatCount, sh_param_id(6u, 0u), splatId, state.sh2_2.x);
    store_splat_param(params, splatCount, sh_param_id(6u, 1u), splatId, state.sh2_2.y);
    store_splat_param(params, splatCount, sh_param_id(6u, 2u), splatId, state.sh2_2.z);
    store_splat_param(params, splatCount, sh_param_id(7u, 0u), splatId, state.sh2_3.x);
    store_splat_param(params, splatCount, sh_param_id(7u, 1u), splatId, state.sh2_3.y);
    store_splat_param(params, splatCount, sh_param_id(7u, 2u), splatId, state.sh2_3.z);
    store_splat_param(params, splatCount, sh_param_id(8u, 0u), splatId, state.sh2_4.x);
    store_splat_param(params, splatCount, sh_param_id(8u, 1u), splatId, state.sh2_4.y);
    store_splat_param(params, splatCount, sh_param_id(8u, 2u), splatId, state.sh2_4.z);
    store_splat_param(params, splatCount, sh_param_id(9u, 0u), splatId, state.sh3_0.x);
    store_splat_param(params, splatCount, sh_param_id(9u, 1u), splatId, state.sh3_0.y);
    store_splat_param(params, splatCount, sh_param_id(9u, 2u), splatId, state.sh3_0.z);
    store_splat_param(params, splatCount, sh_param_id(10u, 0u), splatId, state.sh3_1.x);
    store_splat_param(params, splatCount, sh_param_id(10u, 1u), splatId, state.sh3_1.y);
    store_splat_param(params, splatCount, sh_param_id(10u, 2u), splatId, state.sh3_1.z);
    store_splat_param(params, splatCount, sh_param_id(11u, 0u), splatId, state.sh3_2.x);
    store_splat_param(params, splatCount, sh_param_id(11u, 1u), splatId, state.sh3_2.y);
    store_splat_param(params, splatCount, sh_param_id(11u, 2u), splatId, state.sh3_2.z);
    store_splat_param(params, splatCount, sh_param_id(12u, 0u), splatId, state.sh3_3.x);
    store_splat_param(params, splatCount, sh_param_id(12u, 1u), splatId, state.sh3_3.y);
    store_splat_param(params, splatCount, sh_param_id(12u, 2u), splatId, state.sh3_3.z);
    store_splat_param(params, splatCount, sh_param_id(13u, 0u), splatId, state.sh3_4.x);
    store_splat_param(params, splatCount, sh_param_id(13u, 1u), splatId, state.sh3_4.y);
    store_splat_param(params, splatCount, sh_param_id(13u, 2u), splatId, state.sh3_4.z);
    store_splat_param(params, splatCount, sh_param_id(14u, 0u), splatId, state.sh3_5.x);
    store_splat_param(params, splatCount, sh_param_id(14u, 1u), splatId, state.sh3_5.y);
    store_splat_param(params, splatCount, sh_param_id(14u, 2u), splatId, state.sh3_5.z);
    store_splat_param(params, splatCount, sh_param_id(15u, 0u), splatId, state.sh3_6.x);
    store_splat_param(params, splatCount, sh_param_id(15u, 1u), splatId, state.sh3_6.y);
    store_splat_param(params, splatCount, sh_param_id(15u, 2u), splatId, state.sh3_6.z);
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

PinholeCamera refinement_camera(uint cameraIndex)
{
    uint baseRow = cameraIndex * REFINEMENT_CAMERA_ROW_COUNT;
    float4 row0 = g_RefinementCameraRows[baseRow + 0u];
    float4 row1 = g_RefinementCameraRows[baseRow + 1u];
    float4 row2 = g_RefinementCameraRows[baseRow + 2u];
    float4 row3 = g_RefinementCameraRows[baseRow + 3u];
    float4 row4 = g_RefinementCameraRows[baseRow + 4u];
    float4 row5 = g_RefinementCameraRows[baseRow + 5u];

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

    float currentSupportRadius = currentMaxScale * max(g_RefinementRadiusScale, SMALL_VALUE) * GAUSSIAN_SUPPORT_SIGMA_RADIUS;
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

uint refinement_hash_combine(uint seed, uint value)
{
    return hash_u32(seed ^ (value * REFINEMENT_HASH_MIX + REFINEMENT_HASH_INIT));
}

uint refinement_family_frame_hash(uint splatId)
{
    uint seed = refinement_hash_combine(g_RefinementSeed, splatId + 1u);
    if (g_RefinementCameraCount == 0u) return seed;
    uint cameraIndex = seed % g_RefinementCameraCount;
    uint baseRow = cameraIndex * REFINEMENT_CAMERA_ROW_COUNT;
    return refinement_hash_combine(asuint(g_RefinementCameraRows[baseRow + 5u].z), cameraIndex + 1u);
}

uint refinement_sample_seed(uint frameHash, uint splatId, uint subsampleId)
{
    return refinement_hash_combine(refinement_hash_combine(frameHash, splatId + 1u), subsampleId + 1u);
}

float refinement_family_root(uint familySize)
{
    return pow(max(float(familySize), 1.0), 1.0 / 3.0);
}

float3 refinement_fibonacci_direction(uint subsampleId, uint familySize)
{
    float count = max(float(familySize), 1.0);
    float index = float(subsampleId) + 0.5;
    float y = 1.0 - 2.0 * index / count;
    float radial = sqrt(max(1.0 - y * y, 0.0));
    float angle = REFINEMENT_FIBONACCI_GOLDEN_ANGLE * index;
    return float3(cos(angle) * radial, y, sin(angle) * radial);
}

float3 refinement_volume_sample_raw(uint frameHash, uint splatId, uint subsampleId, uint familySize)
{
    uint sampleSeed = refinement_sample_seed(frameHash, splatId, subsampleId);
    float radius = REFINEMENT_FIBONACCI_RADIUS * refinement_family_root(subsampleId + 1u) / refinement_family_root(familySize);
    float3 sample = refinement_fibonacci_direction(subsampleId, familySize) * radius;
    uint jitterState = refinement_hash_combine(sampleSeed, 0x7f4a7c15u);
    float jitterScale = REFINEMENT_FIBONACCI_RADIUS * REFINEMENT_FIBONACCI_JITTER_RATIO / refinement_family_root(familySize);
    return sample + rng_normal3_box_muller(jitterState) * jitterScale;
}

float3 refinement_volume_family_mean(uint frameHash, uint splatId, uint familySize)
{
    if (familySize <= 1u) return 0.0;
    float3 mean = 0.0;
    [loop] for (uint slot = 0u; slot < familySize; slot++) mean += refinement_volume_sample_raw(frameHash, splatId, slot, familySize);
    return mean / max(float(familySize), 1.0);
}

float3 refinement_centered_volume_sample(uint frameHash, uint splatId, uint slot, uint familySize, float3 familyMean)
{
    if (familySize <= 1u) return 0.0;
    float3 sample = refinement_volume_sample_raw(frameHash, splatId, slot, familySize) - familyMean;
    float sampleLength = length(sample);
    return sampleLength > DENSIFY_MAX_STANDARDIZED_SIGMA ? sample * (DENSIFY_MAX_STANDARDIZED_SIGMA / sampleLength) : sample;
}

float refinement_axis_value(uint axisIndex, uint minIndex, uint midIndex, float minValue, float midValue, float maxValue)
{
    if (axisIndex == minIndex) return minValue;
    if (axisIndex == midIndex) return midValue;
    return maxValue;
}

float3 refinement_child_log_scale(float3 parentLogScale, float shrink)
{
    float3 childLogScale = parentLogScale + log(shrink);
    float sortedMin = parentLogScale.x;
    float sortedMid = parentLogScale.y;
    float sortedMax = parentLogScale.z;
    uint minIndex = 0u;
    uint midIndex = 1u;
    uint maxIndex = 2u;
    if (sortedMin > sortedMid)
    {
        float tempValue = sortedMin;
        sortedMin = sortedMid;
        sortedMid = tempValue;
        uint tempIndex = minIndex;
        minIndex = midIndex;
        midIndex = tempIndex;
    }
    if (sortedMid > sortedMax)
    {
        float tempValue = sortedMid;
        sortedMid = sortedMax;
        sortedMax = tempValue;
        uint tempIndex = midIndex;
        midIndex = maxIndex;
        maxIndex = tempIndex;
    }
    if (sortedMin > sortedMid)
    {
        float tempValue = sortedMin;
        sortedMin = sortedMid;
        sortedMid = tempValue;
        uint tempIndex = minIndex;
        minIndex = midIndex;
        midIndex = tempIndex;
    }

    float parentSpan = sortedMax - sortedMin;
    if (!(parentSpan > SMALL_VALUE)) return childLogScale;

    float targetRatio = max(exp(parentSpan) * REFINEMENT_CHILD_ANISOTROPY_RATIO_SCALE, 1.0);
    float targetSpan = log(targetRatio);
    if (!(targetSpan > SMALL_VALUE))
    {
        float meanLog = (childLogScale.x + childLogScale.y + childLogScale.z) / 3.0;
        return float3(meanLog);
    }

    float midT = min(max((sortedMid - sortedMin) / parentSpan, 0.0), 1.0);
    float meanLog = (childLogScale.x + childLogScale.y + childLogScale.z) / 3.0;
    sortedMin = meanLog - ((1.0 + midT) * targetSpan) / 3.0;
    sortedMid = sortedMin + midT * targetSpan;
    sortedMax = sortedMin + targetSpan;
    return float3(
        refinement_axis_value(0u, minIndex, midIndex, sortedMin, sortedMid, sortedMax),
        refinement_axis_value(1u, minIndex, midIndex, sortedMin, sortedMid, sortedMax),
        refinement_axis_value(2u, minIndex, midIndex, sortedMin, sortedMid, sortedMax)
    );
}

float position_random_step_gate(float opacity)
{
    float x = g_PositionRandomStepOpacityGateSharpness * (g_PositionRandomStepOpacityGateCenter - opacity);
    return rcp(1.0 + exp(-x));
}

float depth_ratio_sigmoid(float x)
{
    if (x >= 0.0)
    {
        float z = exp(-x);
        return rcp(1.0 + z);
    }
    float z = exp(x);
    return z / (1.0 + z);
}

float depth_ratio_softplus(float x)
{
    if (x >= SOFTPLUS_INPUT_LIMIT) return x;
    if (x <= -SOFTPLUS_INPUT_LIMIT) return exp(x);
    return log(1.0 + exp(x));
}

float4 depth_ratio_window_params(float depthRatio)
{
    float x = max(depthRatio, 0.0);
    float bandMin = max(g_DepthRatioGradMin, 0.0);
    float bandMax = max(g_DepthRatioGradMax, bandMin + DEPTH_RATIO_GRAD_MIN_BAND_WIDTH);
    float softness = max((bandMax - bandMin) * DEPTH_RATIO_GRAD_SOFTNESS_RATIO, DEPTH_RATIO_GRAD_SOFTNESS_FLOOR);
    return float4(x, bandMin, bandMax, softness);
}

float depth_ratio_window_loss(float depthRatio)
{
    float4 params = depth_ratio_window_params(depthRatio);
    float scaledLo = (params.x - params.y) / params.w;
    float scaledHi = (params.x - params.z) / params.w;
    return g_DepthRatioWeight * params.w * (depth_ratio_softplus(scaledLo) - depth_ratio_softplus(scaledHi)) * g_InvPixelCount;
}

float depth_ratio_window_grad(float depthRatio)
{
    float4 params = depth_ratio_window_params(depthRatio);
    float scaledLo = (params.x - params.y) / params.w;
    float scaledHi = (params.x - params.z) / params.w;
    return g_DepthRatioWeight * (depth_ratio_sigmoid(scaledLo) - depth_ratio_sigmoid(scaledHi)) * g_InvPixelCount;
}

uint2 training_target_pixel(uint2 pixel)
{
    return g_TrainingSubsample.enabled == 0u ? pixel : training_sample_native_pixel(pixel, g_TrainingSubsample);
}

float4 training_target_rgba(uint2 pixel)
{
    return sanitize_float4(g_Target[training_target_pixel(pixel)], g_HugeValue);
}

float training_target_mask(float alpha)
{
    return g_UseTargetAlphaMask == 0u ? 1.0 : saturate(alpha);
}

float3 training_target_rgb(uint2 pixel)
{
    return training_target_rgba(pixel).xyz;
}

uint2 training_clamped_pixel(int2 pixel)
{
    return uint2(
        uint(clamp(pixel.x, 0, int(g_Width) - 1)),
        uint(clamp(pixel.y, 0, int(g_Height) - 1))
    );
}

float training_target_luminance(uint2 pixel)
{
    return dot(max(training_target_rgb(pixel), 0.0), float3(0.2126, 0.7152, 0.0722));
}

float training_target_edge_intensity(uint2 pixel)
{
    uint2 px00 = training_clamped_pixel(int2(pixel) + int2(-1, -1));
    uint2 px10 = training_clamped_pixel(int2(pixel) + int2(0, -1));
    uint2 px20 = training_clamped_pixel(int2(pixel) + int2(1, -1));
    uint2 px01 = training_clamped_pixel(int2(pixel) + int2(-1, 0));
    uint2 px21 = training_clamped_pixel(int2(pixel) + int2(1, 0));
    uint2 px02 = training_clamped_pixel(int2(pixel) + int2(-1, 1));
    uint2 px12 = training_clamped_pixel(int2(pixel) + int2(0, 1));
    uint2 px22 = training_clamped_pixel(int2(pixel) + int2(1, 1));

    float tl = training_target_luminance(px00);
    float tc = training_target_luminance(px10);
    float tr = training_target_luminance(px20);
    float ml = training_target_luminance(px01);
    float mr = training_target_luminance(px21);
    float bl = training_target_luminance(px02);
    float bc = training_target_luminance(px12);
    float br = training_target_luminance(px22);

    float gx = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br;
    float gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
    return length(float2(gx, gy));
}

uint ssim_feature_index(uint2 pixel, uint channel)
{
    return (pixel.y * g_Width + pixel.x) * SSIM_FEATURE_CHANNELS + channel;
}

void store_ssim_feature3(RWStructuredBuffer<float> buffer, uint2 pixel, uint channelOffset, float3 value)
{
    buffer[ssim_feature_index(pixel, channelOffset + 0u)] = value.x;
    buffer[ssim_feature_index(pixel, channelOffset + 1u)] = value.y;
    buffer[ssim_feature_index(pixel, channelOffset + 2u)] = value.z;
}

float3 load_ssim_feature3(StructuredBuffer<float> buffer, uint2 pixel, uint channelOffset)
{
    return float3(
        buffer[ssim_feature_index(pixel, channelOffset + 0u)],
        buffer[ssim_feature_index(pixel, channelOffset + 1u)],
        buffer[ssim_feature_index(pixel, channelOffset + 2u)]
    );
}

void clear_ssim_feature_grads(RWStructuredBuffer<float> buffer, uint2 pixel)
{
    [unroll] for (uint channel = 0u; channel < SSIM_FEATURE_CHANNELS; channel++) buffer[ssim_feature_index(pixel, channel)] = 0.0;
}

[Differentiable]
float3 rgb_to_ycbcr_bt601(float3 rgb)
{
    return float3(
        dot(rgb, float3(0.299, 0.587, 0.114)),
        dot(rgb, float3(-0.168736, -0.331264, 0.5)) + 0.5,
        dot(rgb, float3(0.5, -0.418688, -0.081312)) + 0.5
    );
}

[Differentiable]
SSIMMoments ssim_feature_moments(float3 renderedRgb, no_diff float3 targetRgb)
{
    float3 rendered = rgb_to_ycbcr_bt601(renderedRgb);
    float3 target = rgb_to_ycbcr_bt601(targetRgb);
    SSIMMoments moments;
    moments.x = rendered;
    moments.y = target;
    moments.x2 = rendered * rendered;
    moments.y2 = target * target;
    moments.xy = rendered * target;
    return moments;
}

[Differentiable]
float ssim_dssim_from_blurred_moments(float3 x, no_diff float3 y, float3 x2, no_diff float3 y2, float3 xy)
{
    float3 sigmaX2 = max(x2 - x * x, 0.0);
    float3 sigmaY2 = max(y2 - y * y, 0.0);
    float3 sigmaXY = xy - x * y;
    float3 numer = (2.0 * x * y + SSIM_C1) * (2.0 * sigmaXY + SSIM_C2);
    float3 denom = max((x * x + y * y + SSIM_C1) * (sigmaX2 + sigmaY2 + SSIM_C2), float3(SMALL_VALUE));
    float3 ssim = numer / denom;
    return 0.5 * (1.0 - dot(ssim, SSIM_YCBCR_WEIGHTS));
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
    float3 localOffset = refinement_centered_volume_sample(frameHash, splatId, slot, familySize, familyMean) * parentScale * shrinkOffset;
    SplatState child = source;
    child.position = source.position + rotate_by_quat(localOffset, qInv);
    child.logScale = refinement_child_log_scale(source.logScale, shrink);
    return child;
}

void apply_refinement_opacity_mul(inout SplatState state)
{
    float alpha = opacity_from_raw(state.rawOpacity);
    float scaledAlpha = alpha * saturate(g_RefinementOpacityMul);
    state.rawOpacity = raw_opacity_from_alpha(max(scaledAlpha, min(0.5, alpha)));
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
        g_LossBuffer[LOSS_SLOT_RGB] = 0.0;
        g_TrainingRgbLossTotal[0] = 0.0;
        g_TrainingTargetEdgeTotal[0] = 0.0;
    }
}

[shader("compute")]
[numthreads(TRAIN_TILE_DIM, TRAIN_TILE_DIM, 1)]
void csComputeSSIMFeatures(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= g_Width || py >= g_Height) return;

    uint2 pixel = uint2(px, py);
    float3 rendered = sanitize_float3(g_Rendered[pixel].xyz, g_HugeValue);
    float3 target = training_target_rgb(pixel);
    SSIMMoments moments = ssim_feature_moments(rendered, target);
    store_ssim_feature3(g_SSIMMoments, pixel, SSIM_FEATURE_X_OFFSET, moments.x);
    store_ssim_feature3(g_SSIMMoments, pixel, SSIM_FEATURE_Y_OFFSET, moments.y);
    store_ssim_feature3(g_SSIMMoments, pixel, SSIM_FEATURE_X2_OFFSET, moments.x2);
    store_ssim_feature3(g_SSIMMoments, pixel, SSIM_FEATURE_Y2_OFFSET, moments.y2);
    store_ssim_feature3(g_SSIMMoments, pixel, SSIM_FEATURE_XY_OFFSET, moments.xy);
}

[shader("compute")]
[numthreads(TRAIN_TILE_DIM, TRAIN_TILE_DIM, 1)]
void csComputeBlendedLossForward(
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
    float localRgbLoss = 0.0;
    float localTargetEdge = 0.0;
    if (px < g_Width && py < g_Height)
    {
        uint2 pixel = uint2(px, py);
        uint pixelIndex = py * g_Width + px;
        float3 rendered = sanitize_float3(g_Rendered[pixel].xyz, g_HugeValue);
        float4 targetRgba = training_target_rgba(pixel);
        float3 target = targetRgba.xyz;
        float targetMask = training_target_mask(targetRgba.w);
        float l1Loss = rgb_l1_loss(rendered, target, g_InvPixelCount) * targetMask;
        float dssimLoss = ssim_dssim_from_blurred_moments(
            load_ssim_feature3(g_SSIMBlurredMoments, pixel, SSIM_FEATURE_X_OFFSET),
            load_ssim_feature3(g_SSIMBlurredMoments, pixel, SSIM_FEATURE_Y_OFFSET),
            load_ssim_feature3(g_SSIMBlurredMoments, pixel, SSIM_FEATURE_X2_OFFSET),
            load_ssim_feature3(g_SSIMBlurredMoments, pixel, SSIM_FEATURE_Y2_OFFSET),
            load_ssim_feature3(g_SSIMBlurredMoments, pixel, SSIM_FEATURE_XY_OFFSET)
        ) * (g_InvPixelCount * targetMask);
        float blendedRgbLoss = lerp(l1Loss, dssimLoss, g_SSIMWeight);
        float mse = rgb_mse_loss(rendered, target, g_InvPixelCount) * targetMask;
        float depthRatio = sanitize_scalar(g_RenderedDepthStats[pixel].w, g_HugeValue);
        float density = sanitize_scalar(g_RenderedDensity[py * g_Width + px], g_HugeValue);
        localRgbLoss = isfinite(blendedRgbLoss) ? blendedRgbLoss : 0.0;
        localTargetEdge = isfinite(g_RefinementTargetEdgeWeight) && g_RefinementTargetEdgeWeight > 0.0 ? max(training_target_edge_intensity(pixel), 0.0) * targetMask : 0.0;
        localLoss = localRgbLoss;
        localMSE = isfinite(mse) ? mse : 0.0;
        if (g_DepthRatioWeight > 0.0)
        {
            float depthRatioLoss = depth_ratio_window_loss(depthRatio) * targetMask;
            if (isfinite(depthRatioLoss)) localLoss += depthRatioLoss;
        }
        if (g_DensityRegularizer > 0.0)
        {
            float densityExcess = max(density - g_MaxAllowedDensity, 0.0);
            float densityLoss = g_DensityRegularizer * densityExcess * g_InvPixelCount * targetMask;
            localDensityLoss = isfinite(densityLoss) ? densityLoss : 0.0;
        }
        g_TrainingRgbLoss[pixelIndex] = localRgbLoss;
        g_TrainingTargetEdge[pixelIndex] = localTargetEdge;
    }

    gsLoss[localIndex] = localLoss;
    gsMSE[localIndex] = localMSE;
    gsDensityLoss[localIndex] = localDensityLoss;
    gsRgbLoss[localIndex] = localRgbLoss;
    gsTargetEdge[localIndex] = localTargetEdge;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint stride = (TRAIN_TILE_DIM * TRAIN_TILE_DIM) >> 1; stride > 0; stride >>= 1)
    {
        if (localIndex < stride) gsLoss[localIndex] += gsLoss[localIndex + stride];
        if (localIndex < stride) gsMSE[localIndex] += gsMSE[localIndex + stride];
        if (localIndex < stride) gsDensityLoss[localIndex] += gsDensityLoss[localIndex + stride];
        if (localIndex < stride) gsRgbLoss[localIndex] += gsRgbLoss[localIndex + stride];
        if (localIndex < stride) gsTargetEdge[localIndex] += gsTargetEdge[localIndex + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (localIndex == 0u)
    {
        if (isfinite(gsLoss[0])) InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], gsLoss[0]);
        if (isfinite(gsMSE[0])) InterlockedAdd(g_LossBuffer[LOSS_SLOT_MSE], gsMSE[0]);
        if (isfinite(gsRgbLoss[0]))
        {
            InterlockedAdd(g_LossBuffer[LOSS_SLOT_RGB], gsRgbLoss[0]);
            InterlockedAdd(g_TrainingRgbLossTotal[0], gsRgbLoss[0]);
        }
        if (isfinite(gsTargetEdge[0])) InterlockedAdd(g_TrainingTargetEdgeTotal[0], gsTargetEdge[0]);
        if (isfinite(gsDensityLoss[0]))
        {
            InterlockedAdd(g_LossBuffer[LOSS_SLOT_TOTAL], gsDensityLoss[0]);
            InterlockedAdd(g_LossBuffer[LOSS_SLOT_DENSITY], gsDensityLoss[0]);
        }
    }
}

[shader("compute")]
[numthreads(TRAIN_TILE_DIM, TRAIN_TILE_DIM, 1)]
void csComputeSSIMBlurredGradients(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= g_Width || py >= g_Height) return;

    uint2 pixel = uint2(px, py);
    clear_ssim_feature_grads(g_SSIMBlurredFeatureGrads, pixel);
    if (!(g_SSIMWeight > 0.0)) return;

    float targetMask = training_target_mask(training_target_rgba(pixel).w);
    if (!(targetMask > 0.0)) return;

    float3 x = load_ssim_feature3(g_SSIMBlurredMoments, pixel, SSIM_FEATURE_X_OFFSET);
    float3 y = load_ssim_feature3(g_SSIMBlurredMoments, pixel, SSIM_FEATURE_Y_OFFSET);
    float3 x2 = load_ssim_feature3(g_SSIMBlurredMoments, pixel, SSIM_FEATURE_X2_OFFSET);
    float3 y2 = load_ssim_feature3(g_SSIMBlurredMoments, pixel, SSIM_FEATURE_Y2_OFFSET);
    float3 xy = load_ssim_feature3(g_SSIMBlurredMoments, pixel, SSIM_FEATURE_XY_OFFSET);
    float lossScale = g_SSIMWeight * g_InvPixelCount * targetMask;
    if (!(lossScale > 0.0)) return;

    var dpX = diffPair(x);
    var dpX2 = diffPair(x2);
    var dpXY = diffPair(xy);
    bwd_diff(ssim_dssim_from_blurred_moments)(dpX, y, dpX2, y2, dpXY, lossScale);
    store_ssim_feature3(g_SSIMBlurredFeatureGrads, pixel, SSIM_FEATURE_X_OFFSET, dpX.getDifferential());
    store_ssim_feature3(g_SSIMBlurredFeatureGrads, pixel, SSIM_FEATURE_X2_OFFSET, dpX2.getDifferential());
    store_ssim_feature3(g_SSIMBlurredFeatureGrads, pixel, SSIM_FEATURE_XY_OFFSET, dpXY.getDifferential());
}

[shader("compute")]
[numthreads(TRAIN_TILE_DIM, TRAIN_TILE_DIM, 1)]
void csComputeBlendedLossBackward(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= g_Width || py >= g_Height) return;

    uint2 pixel = uint2(px, py);
    float3 rendered = sanitize_float3(g_Rendered[pixel].xyz, g_HugeValue);
    float4 targetRgba = training_target_rgba(pixel);
    float3 target = targetRgba.xyz;
    float targetMask = training_target_mask(targetRgba.w);
    float3 dL1 = rgb_l1_grad(rendered, target) * ((1.0 - g_SSIMWeight) * g_InvPixelCount * targetMask);
    SSIMMoments dMoments;
    dMoments.x = load_ssim_feature3(g_SSIMFeatureGrads, pixel, SSIM_FEATURE_X_OFFSET);
    dMoments.y = load_ssim_feature3(g_SSIMFeatureGrads, pixel, SSIM_FEATURE_Y_OFFSET);
    dMoments.x2 = load_ssim_feature3(g_SSIMFeatureGrads, pixel, SSIM_FEATURE_X2_OFFSET);
    dMoments.y2 = load_ssim_feature3(g_SSIMFeatureGrads, pixel, SSIM_FEATURE_Y2_OFFSET);
    dMoments.xy = load_ssim_feature3(g_SSIMFeatureGrads, pixel, SSIM_FEATURE_XY_OFFSET);
    var dpRendered = diffPair(rendered);
    bwd_diff(ssim_feature_moments)(dpRendered, target, dMoments);
    float3 dOut = clamp(sanitize_float3(dL1 + dpRendered.getDifferential(), g_LossGradClip), -g_LossGradClip, g_LossGradClip);
    g_OutputGrad[py * g_Width + px] = float4(dOut, 0.0);
    float densityGrad = 0.0;
    float density = sanitize_scalar(g_RenderedDensity[py * g_Width + px], g_HugeValue);
    if (g_DensityRegularizer > 0.0 && density > g_MaxAllowedDensity)
    {
        densityGrad = g_DensityRegularizer * g_InvPixelCount * targetMask;
    }

    float depthRatioGrad = 0.0;
    if (g_DepthRatioWeight > 0.0)
    {
        float depthRatio = sanitize_scalar(g_RenderedDepthStats[pixel].w, g_HugeValue);
        depthRatioGrad = depth_ratio_window_grad(depthRatio) * targetMask;
    }
    g_RegularizerGrad[py * g_Width + px] = float2(
        clamp(sanitize_scalar(densityGrad, g_LossGradClip), -g_LossGradClip, g_LossGradClip),
        clamp(sanitize_scalar(depthRatioGrad, g_LossGradClip), -g_LossGradClip, g_LossGradClip)
    );
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
void csClearRefinementCounters(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if (dispatchThreadID.x != 0u) return;
    g_TotalCloneCounter[0] = 0u;
    g_AppendCounter[0] = 0u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csPrepareRefinementCounts(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SrcSplatCount) return;
    float alpha = opacity_from_raw(load_splat_param(g_SrcSplatParams, g_SrcSplatCount, PARAM_RAW_OPACITY, splatId));
    if (should_cull_refinement_splat(alpha, g_SplatContribution[splatId])) return;
    uint _;
    InterlockedAdd(g_AppendCounter[0], 1u, _);
    uint cloneCount = g_CloneCounts[splatId];
    if (cloneCount > 0u) InterlockedAdd(g_TotalCloneCounter[0], cloneCount, _);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csClampRefinementMinScreenSize(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SrcSplatCount) return;
    if (g_RefinementCameraCount == 0u) return;

    SplatState source = load_splat_state(g_SrcSplatParams, g_SrcSplatCount, splatId);
    float minVisibleSupportRadius = 0.0;
    bool hasVisibleCamera = false;
    [loop] for (uint cameraIndex = 0u; cameraIndex < g_RefinementCameraCount; cameraIndex++)
    {
        float supportRadius;
        if (!world_support_radius_from_screen_circle(refinement_camera(cameraIndex), source.position, REFINEMENT_MIN_SCREEN_RADIUS_PX, supportRadius)) continue;
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
void csRewriteRefinementSplats(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SrcSplatCount) return;

    SplatState source = load_splat_state(g_SrcSplatParams, g_SrcSplatCount, splatId);
    float alpha = opacity_from_raw(source.rawOpacity);
    if (should_cull_refinement_splat(alpha, g_SplatContribution[splatId])) return;
    apply_refinement_opacity_mul(source);

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
        uint frameHash = refinement_family_frame_hash(splatId);
        float3 familyMean = refinement_volume_family_mean(frameHash, splatId, familySize);
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
void csComposeEdgeDebug(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= g_DebugWidth || py >= g_DebugHeight) return;

    uint px0 = px > 0u ? px - 1u : 0u;
    uint py0 = py > 0u ? py - 1u : 0u;
    uint px2 = min(px + 1u, g_DebugWidth - 1u);
    uint py2 = min(py + 1u, g_DebugHeight - 1u);

    float tl = dot(max(sanitize_float3(g_DebugRendered[uint2(px0, py0)].xyz, g_HugeValue), 0.0), float3(0.2126, 0.7152, 0.0722));
    float tc = dot(max(sanitize_float3(g_DebugRendered[uint2(px, py0)].xyz, g_HugeValue), 0.0), float3(0.2126, 0.7152, 0.0722));
    float tr = dot(max(sanitize_float3(g_DebugRendered[uint2(px2, py0)].xyz, g_HugeValue), 0.0), float3(0.2126, 0.7152, 0.0722));
    float ml = dot(max(sanitize_float3(g_DebugRendered[uint2(px0, py)].xyz, g_HugeValue), 0.0), float3(0.2126, 0.7152, 0.0722));
    float mr = dot(max(sanitize_float3(g_DebugRendered[uint2(px2, py)].xyz, g_HugeValue), 0.0), float3(0.2126, 0.7152, 0.0722));
    float bl = dot(max(sanitize_float3(g_DebugRendered[uint2(px0, py2)].xyz, g_HugeValue), 0.0), float3(0.2126, 0.7152, 0.0722));
    float bc = dot(max(sanitize_float3(g_DebugRendered[uint2(px, py2)].xyz, g_HugeValue), 0.0), float3(0.2126, 0.7152, 0.0722));
    float br = dot(max(sanitize_float3(g_DebugRendered[uint2(px2, py2)].xyz, g_HugeValue), 0.0), float3(0.2126, 0.7152, 0.0722));

    float gx = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br;
    float gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
    float gradient = length(float2(gx, gy));
    float normalized = saturate(gradient / DEBUG_EDGE_GRADIENT_DISPLAY_MAX);
    g_DebugOutput[uint2(px, py)] = float4(jet_colormap(normalized), 1.0);
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
