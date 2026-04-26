import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/math/quaternion.slang";
import "utility/splatting/common.slang";
import "utility/splatting/opacity.slang";

static const float ENVELOPE_SIGMA_RADIUS = GAUSSIAN_SUPPORT_SIGMA_RADIUS;
static const float ENVELOPE_EXPONENT_SCALE = 0.5 * ENVELOPE_SIGMA_RADIUS * ENVELOPE_SIGMA_RADIUS;

[Differentiable]
float3 world_to_local_dir(float3 dir, float3 scale, float4 rotation)
{
    return rotate_by_quat(dir, rotation) / max(scale, float3(SMALL_VALUE));
}

[Differentiable]
RasterGaussian build_raster_gaussian(no_diff CameraParams cameraParams, no_diff float radiusScale, no_diff float maxAnisotropy, no_diff float3 cameraPosition, no_diff uint shBand, Gaussian3D gaussian)
{
    Ellipsoid ellipsoid = build_ellipsoid(cameraParams, radiusScale, maxAnisotropy, gaussian);
    return {
        ellipsoid.rasterRoLocal,
        ellipsoid.rasterScale,
        ellipsoid.rotation,
        float4(
            evaluate_gaussian_color(
                gaussian.position,
                gaussian.shCoeffs,
                cameraPosition,
                shBand
            ),
            opacity_from_raw(gaussian.rawOpacity)
        )
    };
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
