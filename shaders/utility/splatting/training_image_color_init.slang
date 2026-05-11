import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/splatting/params.slang";

static const uint TRAINING_IMAGE_COLOR_INIT_GROUP_SIZE = 256u;
static const float TRAINING_IMAGE_COLOR_INIT_INF = 3.402823466e38;
static const float TRAINING_IMAGE_COLOR_INIT_SH_C0 = 0.28209479177387814;

uniform RWStructuredBuffer<float> g_SplatParams;
uniform RWStructuredBuffer<float> g_BestDistance;
uniform Texture2D<float4> g_TrainingImage;
uniform CameraParams g_Camera;
uniform uint g_SplatCount;
uniform uint g_PackedParamCount;
uniform uint g_ImageWidth;
uniform uint g_ImageHeight;

float4 sample_training_image_linear(float2 screenPos)
{
    float x = clamp(screenPos.x, 0.0, float(g_ImageWidth - 1u));
    float y = clamp(screenPos.y, 0.0, float(g_ImageHeight - 1u));
    uint x0 = uint(floor(x));
    uint y0 = uint(floor(y));
    uint x1 = min(x0 + 1u, g_ImageWidth - 1u);
    uint y1 = min(y0 + 1u, g_ImageHeight - 1u);
    float tx = x - float(x0);
    float ty = y - float(y0);
    float4 top = lerp(g_TrainingImage[uint2(x0, y0)], g_TrainingImage[uint2(x1, y0)], tx);
    float4 bottom = lerp(g_TrainingImage[uint2(x0, y1)], g_TrainingImage[uint2(x1, y1)], tx);
    return lerp(top, bottom, ty);
}

void store_splat_rgb_as_sh0(uint splatId, float3 rgb)
{
    float3 sh0 = (saturate(rgb) - 0.5) / TRAINING_IMAGE_COLOR_INIT_SH_C0;
    store_splat_sh_coeff(g_SplatParams, g_SplatCount, g_PackedParamCount, 0u, splatId, sh0);
    uint coeffCount = packed_sh_coeff_count(g_PackedParamCount);
    for (uint coeffIndex = 1u; coeffIndex < coeffCount; coeffIndex++) store_splat_sh_coeff(g_SplatParams, g_SplatCount, g_PackedParamCount, coeffIndex, splatId, float3(0.0, 0.0, 0.0));
}

[shader("compute")]
[numthreads(TRAINING_IMAGE_COLOR_INIT_GROUP_SIZE, 1, 1)]
void csClearTrainingImageColorInit(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount) return;
    g_BestDistance[splatId] = TRAINING_IMAGE_COLOR_INIT_INF;
}

[shader("compute")]
[numthreads(TRAINING_IMAGE_COLOR_INIT_GROUP_SIZE, 1, 1)]
void csSampleTrainingImageColorInit(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_SplatCount || g_ImageWidth == 0u || g_ImageHeight == 0u) return;

    float3 worldPos = load_splat_param_vector<3>(g_SplatParams, g_SplatCount, PARAM_POSITION_X, splatId);
    PinholeCamera camera = PinholeCamera.from_params(g_Camera);
    float3 cameraPos = camera.world_point_to_camera(worldPos);
    if (cameraPos.z <= SMALL_VALUE) return;

    float2 screenPos;
    if (!camera.project_camera_to_screen(cameraPos, screenPos)) return;
    if (screenPos.x < 0.0 || screenPos.y < 0.0 || screenPos.x > float(g_ImageWidth - 1u) || screenPos.y > float(g_ImageHeight - 1u)) return;

    float distanceToCamera = length(worldPos - camera.position_world());
    if (!(distanceToCamera < g_BestDistance[splatId])) return;

    float4 sampled = sample_training_image_linear(screenPos);
    if (!all(isfinite(sampled)) || sampled.w <= SMALL_VALUE) return;

    g_BestDistance[splatId] = distanceToCamera;
    store_splat_rgb_as_sh0(splatId, sampled.xyz);
}
