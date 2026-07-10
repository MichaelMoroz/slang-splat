import "utility/splatting/common.slang";
import "renderer/gaussian_types.slang";
import "utility/splatting/raster.slang";
import "renderer/gaussian_raster_stage.slang";

static const uint PROBE_GROUP_SIZE = 64u;
static const uint PROBE_RASTER_COMPONENT_COUNT = 13u;
static const uint PROBE_GAUSSIAN_GRAD_COMPONENT_COUNT = 11u;

uniform StructuredBuffer<RasterGaussian> g_InputRasterGaussian;
uniform StructuredBuffer<float3> g_InputRayDirs;
uniform StructuredBuffer<float2> g_InputEvalGrad;
uniform StructuredBuffer<RasterGaussian> g_InputRasterGrad;

uniform RWStructuredBuffer<float> g_OutputEval;
uniform RWStructuredBuffer<float> g_OutputRasterGaussian;
uniform RWStructuredBuffer<float> g_OutputGaussianGrad;

void store_output_eval(uint index, float2 value)
{
    uint baseIndex = index * 2u;
    g_OutputEval[baseIndex + 0u] = value.x;
    g_OutputEval[baseIndex + 1u] = value.y;
}

void store_output_raster_gaussian(uint index, RasterGaussian value)
{
    uint baseIndex = index * PROBE_RASTER_COMPONENT_COUNT;
    g_OutputRasterGaussian[baseIndex + 0u] = value.roCamera.x;
    g_OutputRasterGaussian[baseIndex + 1u] = value.roCamera.y;
    g_OutputRasterGaussian[baseIndex + 2u] = value.roCamera.z;
    g_OutputRasterGaussian[baseIndex + 3u] = value.sigmaDiag.x;
    g_OutputRasterGaussian[baseIndex + 4u] = value.sigmaDiag.y;
    g_OutputRasterGaussian[baseIndex + 5u] = value.sigmaDiag.z;
    g_OutputRasterGaussian[baseIndex + 6u] = value.sigmaOffDiag.x;
    g_OutputRasterGaussian[baseIndex + 7u] = value.sigmaOffDiag.y;
    g_OutputRasterGaussian[baseIndex + 8u] = value.sigmaOffDiag.z;
    g_OutputRasterGaussian[baseIndex + 9u] = value.colorAlpha.x;
    g_OutputRasterGaussian[baseIndex + 10u] = value.colorAlpha.y;
    g_OutputRasterGaussian[baseIndex + 11u] = value.colorAlpha.z;
    g_OutputRasterGaussian[baseIndex + 12u] = value.colorAlpha.w;
}

void store_output_gaussian_grad(uint index, Gaussian3D.Differential value)
{
    uint baseIndex = index * PROBE_GAUSSIAN_GRAD_COMPONENT_COUNT;
    g_OutputGaussianGrad[baseIndex + 0u] = value.position.x;
    g_OutputGaussianGrad[baseIndex + 1u] = value.position.y;
    g_OutputGaussianGrad[baseIndex + 2u] = value.position.z;
    g_OutputGaussianGrad[baseIndex + 3u] = value.logScale.x;
    g_OutputGaussianGrad[baseIndex + 4u] = value.logScale.y;
    g_OutputGaussianGrad[baseIndex + 5u] = value.logScale.z;
    g_OutputGaussianGrad[baseIndex + 6u] = value.rotation.x;
    g_OutputGaussianGrad[baseIndex + 7u] = value.rotation.y;
    g_OutputGaussianGrad[baseIndex + 8u] = value.rotation.z;
    g_OutputGaussianGrad[baseIndex + 9u] = value.rotation.w;
    g_OutputGaussianGrad[baseIndex + 10u] = value.rawOpacity;
}

RasterGaussian.Differential raster_differential_from_value(RasterGaussian value)
{
    RasterGaussian.Differential diff;
    diff.roCamera = value.roCamera;
    diff.sigmaDiag = value.sigmaDiag;
    diff.sigmaOffDiag = value.sigmaOffDiag;
    diff.colorAlpha = value.colorAlpha;
    return diff;
}

[shader("compute")]
[numthreads(PROBE_GROUP_SIZE, 1, 1)]
void csEvalRasterAlphaDepth(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint index = dispatchThreadID.x;
    if (index >= g_Prepass.splatCount) return;

    store_output_eval(index, raster_eval_alpha_depth_gaussian(g_InputRasterGaussian[index], g_InputRayDirs[index]));
}

[shader("compute")]
[numthreads(PROBE_GROUP_SIZE, 1, 1)]
void csBackpropRasterEvalAlphaDepth(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint index = dispatchThreadID.x;
    if (index >= g_Prepass.splatCount) return;

    RasterGaussian gaussian = g_InputRasterGaussian[index];
    var dpGaussian = diffPair(gaussian);
    bwd_diff(raster_eval_alpha_depth_gaussian)(dpGaussian, g_InputRayDirs[index], g_InputEvalGrad[index]);
    store_output_raster_gaussian(index, dpGaussian.getDifferential());
}

[shader("compute")]
[numthreads(PROBE_GROUP_SIZE, 1, 1)]
void csBuildRasterGaussianForward(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_Prepass.splatCount) return;

    Gaussian3D gaussian = load_gaussian3d(splatId);
    store_output_raster_gaussian(splatId, build_raster_gaussian(g_Camera, g_Prepass.radiusScale, g_MaxAnisotropy, g_Camera.camPos, g_Raster.shBand, gaussian));
}

[shader("compute")]
[numthreads(PROBE_GROUP_SIZE, 1, 1)]
void csBackpropBuildRasterGaussian(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_Prepass.splatCount) return;

    Gaussian3D gaussian = load_gaussian3d(splatId);
    var dpGaussian = diffPair(gaussian);
    bwd_diff(build_raster_gaussian)(g_Camera, g_Prepass.radiusScale, g_MaxAnisotropy, g_Camera.camPos, g_Raster.shBand, dpGaussian, raster_differential_from_value(g_InputRasterGrad[splatId]));
    store_output_gaussian_grad(splatId, dpGaussian.getDifferential());
}

[shader("compute")]
[numthreads(PROBE_GROUP_SIZE, 1, 1)]
void csFlushCachedRasterGradFloatFromInput(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_Prepass.splatCount) return;

    RasterGaussian.Differential dGaussian = raster_differential_from_value(g_InputRasterGrad[splatId]);
    float contributionNormSq = raster_grad_contribution_norm_sq(dGaussian);
    if (isfinite(contributionNormSq) && contributionNormSq > 0.0) atomic_add_gradient_stat(splatId, 1u, contributionNormSq);
    uint splatCount = g_Prepass.splatCount;
    bool includeDepth = raster_grad_include_depth();
    add_cached_raster_grad_components(splatCount, uint3(RASTER_CACHE_RO_CAMERA_X, RASTER_CACHE_RO_CAMERA_Y, RASTER_CACHE_RO_CAMERA_Z), splatId, dGaussian.roCamera, includeDepth);
    add_cached_raster_grad_components(splatCount, uint3(RASTER_CACHE_SIGMA_XX, RASTER_CACHE_SIGMA_YY, RASTER_CACHE_SIGMA_ZZ), splatId, dGaussian.sigmaDiag, includeDepth);
    add_cached_raster_grad_components(splatCount, uint3(RASTER_CACHE_SIGMA_XY, RASTER_CACHE_SIGMA_XZ, RASTER_CACHE_SIGMA_YZ), splatId, dGaussian.sigmaOffDiag, includeDepth);
    add_cached_raster_grad_components(splatCount, uint4(RASTER_CACHE_COLOR_R, RASTER_CACHE_COLOR_G, RASTER_CACHE_COLOR_B, RASTER_CACHE_FUSED_OPACITY), splatId, dGaussian.colorAlpha, includeDepth);
}

[shader("compute")]
[numthreads(PROBE_GROUP_SIZE, 1, 1)]
void csFlushCachedRasterGradFixedFromInput(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_Prepass.splatCount) return;

    flush_cached_raster_grad_fixed(splatId, g_RasterCache[splatId], raster_differential_from_value(g_InputRasterGrad[splatId]));
}

[shader("compute")]
[numthreads(PROBE_GROUP_SIZE, 1, 1)]
void csLoadCachedRasterGradFloatToOutput(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_Prepass.splatCount) return;

    store_output_raster_gaussian(splatId, load_cached_raster_grad_float(splatId));
}

[shader("compute")]
[numthreads(PROBE_GROUP_SIZE, 1, 1)]
void csLoadCachedRasterGradFixedToOutput(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_Prepass.splatCount) return;

    store_output_raster_gaussian(splatId, load_cached_raster_grad_fixed(splatId));
}