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
    float3 logInvScale;
    float4 quat;
};

[Differentiable]
CachedEllipsoid build_cached_ellipsoid(no_diff CameraParams cameraParams, no_diff float radiusScale, float3 position, float3 logScale, float4 rotation)
{
    Gaussian3D gaussian;
    gaussian.position = position;
    gaussian.logScale = logScale;
    gaussian.rotation = rotation;
    gaussian.colorAlpha = 0.0;
    Ellipsoid ellipsoid = build_ellipsoid(cameraParams, radiusScale, gaussian);
    return { ellipsoid.rasterRoLocal, log(max(ellipsoid.rasterInvScale, float3(SMALL_VALUE))), ellipsoid.rotation };
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
float raster_eval_alpha(float3 roLocal, float4 quat, float3 logInvScale, float fusedOpacity, no_diff float3 rayDir, no_diff float alphaCutoff)
{
    float3 invScale = exp(logInvScale);
    float3 rdLocal = rotate_by_quat(rayDir, quat) * invScale;
    float denom = dot(rdLocal, rdLocal);
    if (denom <= RAY_DENOMINATOR_FLOOR) return 0.0;

    float tClosest = dot(rdLocal, -roLocal) / denom;
    if (tClosest <= 0.0) return 0.0;

    float3 closest = roLocal + rdLocal * tClosest;
    float rho = length(closest);
    float alpha = fusedOpacity * envelope(rho);
    if (alpha < alphaCutoff) return 0.0;
    return alpha;
}
