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
    gaussian.rawOpacity = 0.0;
    Ellipsoid ellipsoid = build_ellipsoid(cameraParams, radiusScale, maxAnisotropy, gaussian);
    return {
        ellipsoid.rasterRoLocal,
        ellipsoid.rasterScale,
        ellipsoid.rotation
    };
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
