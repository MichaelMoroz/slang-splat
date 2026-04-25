import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/math/quaternion.slang";
import "utility/splatting/common.slang";
import "utility/splatting/opacity.slang";

static const float ENVELOPE_SIGMA_RADIUS = GAUSSIAN_SUPPORT_SIGMA_RADIUS;
static const float ENVELOPE_EXPONENT_SCALE = 0.5 * ENVELOPE_SIGMA_RADIUS * ENVELOPE_SIGMA_RADIUS;

struct CachedEllipsoid : IDifferentiable
{
    float3 roLocal;
    float3 scale;
    float4 rotation;
};

struct CachedDirectionalGaussian : IDifferentiable
{
    float3 centerDir;
    float3 sigmaOrtho;
    float centerDistance;
};

[Differentiable]
float3 world_to_local_dir(float3 dir, float3 scale, float4 rotation)
{
    return rotate_by_quat(dir, rotation) / max(scale, float3(SMALL_VALUE));
}

[Differentiable]
CachedEllipsoid build_cached_ellipsoid(no_diff CameraParams cameraParams, no_diff float radiusScale, no_diff float maxAnisotropy, float3 position, float3 logScale, float4 rotation)
{
    Gaussian3D gaussian;
    gaussian.position = position;
    gaussian.logScale = logScale;
    gaussian.rotation = rotation;
    gaussian.sh0 = 0.0;
    gaussian.sh1X = 0.0;
    gaussian.sh1Y = 0.0;
    gaussian.sh1Z = 0.0;
    gaussian.sh2_0 = 0.0;
    gaussian.sh2_1 = 0.0;
    gaussian.sh2_2 = 0.0;
    gaussian.sh2_3 = 0.0;
    gaussian.sh2_4 = 0.0;
    gaussian.sh3_0 = 0.0;
    gaussian.sh3_1 = 0.0;
    gaussian.sh3_2 = 0.0;
    gaussian.sh3_3 = 0.0;
    gaussian.sh3_4 = 0.0;
    gaussian.sh3_5 = 0.0;
    gaussian.sh3_6 = 0.0;
    gaussian.rawOpacity = 0.0;
    Ellipsoid ellipsoid = build_ellipsoid(cameraParams, radiusScale, maxAnisotropy, gaussian);
    return {
        ellipsoid.rasterRoLocal,
        ellipsoid.rasterScale,
        ellipsoid.rotation
    };
}

[Differentiable]
CachedDirectionalGaussian build_cached_directional_gaussian(no_diff CameraParams cameraParams, no_diff float radiusScale, no_diff float maxAnisotropy, float3 position, float3 logScale, float4 rotation)
{
    CachedDirectionalGaussian result;
    result.centerDir = float3(0.0, 0.0, 1.0);
    result.sigmaOrtho = 0.0;
    result.centerDistance = 0.0;

    CachedEllipsoid cached = build_cached_ellipsoid(cameraParams, radiusScale, maxAnisotropy, position, logScale, rotation);
    float3 cameraMean = mul(cameraParams.camBasis, position - cameraParams.camPos);
    float centerDistance = length(cameraMean);
    if (centerDistance <= SMALL_VALUE || !all(isfinite(cameraMean))) return result;

    float3 centerDir = cameraMean / centerDistance;
    float3 tangentX;
    float3 tangentY;
    build_direction_frame(centerDir, tangentX, tangentY);

    float3 worldTangentX = mul(transpose(cameraParams.camBasis), tangentX);
    float3 worldTangentY = mul(transpose(cameraParams.camBasis), tangentY);
    float3 localTangentX = rotate_by_quat(worldTangentX, cached.rotation);
    float3 localTangentY = rotate_by_quat(worldTangentY, cached.rotation);
    float3 sigma = cached.scale / ENVELOPE_SIGMA_RADIUS;
    float3 variance = sigma * sigma;
    float invCenterDistance2 = 1.0 / max(centerDistance * centerDistance, SMALL_VALUE);

    result.centerDir = centerDir;
    result.sigmaOrtho = float3(
        dot(localTangentX * variance, localTangentX),
        dot(localTangentX * variance, localTangentY),
        dot(localTangentY * variance, localTangentY)
    ) * invCenterDistance2;
    result.centerDistance = centerDistance;
    return result;
}

[Differentiable]
float build_raster_fused_opacity(no_diff CameraParams cameraParams, no_diff float radiusScale, float3 position, float3 logScale, float rawOpacity)
{
    return opacity_from_raw(rawOpacity);
}

[Differentiable]
float envelope(float rho)
{
    if (rho >= 1.0) return 0.0;
    return exp(-ENVELOPE_EXPONENT_SCALE * rho * rho);
}

[Differentiable]
float raster_eval_alpha(float3 roLocal, float3 scale, float4 rotation, float fusedOpacity, no_diff float3 rayDir, no_diff float alphaCutoff)
{
    float3 rayLocal = world_to_local_dir(rayDir, scale, rotation);
    float denom = dot(rayLocal, rayLocal);
    if (denom <= RAY_DENOMINATOR_FLOOR) return 0.0;

    float numer = dot(rayLocal, roLocal);
    float tClosest = -numer / denom;
    if (tClosest <= 0.0) return 0.0;

    float3 closest = roLocal + rayLocal * tClosest;
    float rho2 = dot(closest, closest);
    if (rho2 < 0.0) rho2 = 0.0;
    float alpha = fusedOpacity * exp(-ENVELOPE_EXPONENT_SCALE * rho2);
    if (alpha < alphaCutoff) return 0.0;
    return alpha;
}

[Differentiable]
float2 project_ray_to_direction_frame_camera(float3 centerDir, float3 rayDirCamera)
{
    float3 tangentX;
    float3 tangentY;
    build_direction_frame(centerDir, tangentX, tangentY);
    float denom = dot(centerDir, rayDirCamera);
    if (denom <= RAY_DENOMINATOR_FLOOR) return 0.0;
    return float2(dot(tangentX, rayDirCamera), dot(tangentY, rayDirCamera)) / denom;
}

[Differentiable]
float raster_eval_directional_alpha(no_diff CameraParams cameraParams, float3 centerDir, float3 sigmaOrtho, float opacity, no_diff float3 rayDirWorld, no_diff float alphaCutoff)
{
    float det = sigmaOrtho.x * sigmaOrtho.z - sigmaOrtho.y * sigmaOrtho.y;
    if (det <= (RAY_DENOMINATOR_FLOOR * RAY_DENOMINATOR_FLOOR) || !isfinite(det)) return 0.0;

    float3 rayDirCamera = mul(cameraParams.camBasis, rayDirWorld);
    float2 eta = project_ray_to_direction_frame_camera(centerDir, rayDirCamera);
    if (!all(isfinite(eta))) return 0.0;

    float invDet = 1.0 / det;
    float2 quadVec = float2(
        sigmaOrtho.z * eta.x - sigmaOrtho.y * eta.y,
        -sigmaOrtho.y * eta.x + sigmaOrtho.x * eta.y
    ) * invDet;
    float quad = dot(eta, quadVec);
    float alpha = opacity * exp(-0.5 * quad);
    if (alpha < alphaCutoff) return 0.0;
    return alpha;
}
