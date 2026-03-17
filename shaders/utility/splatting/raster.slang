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
    float3 logLDiag;
    float3 lOffDiag;
};

[Differentiable]
float3 rotate_and_scale_axis(no_diff float3 axis, float3 invScale, float4 rotation)
{
    return rotate_by_quat(axis, rotation) * invScale;
}

[Differentiable]
float3x3 build_shape_matrix(float3 logLDiag, float3 lOffDiag)
{
    float3 lDiag = exp(logLDiag);
    float3x3 lower = float3x3(
        lDiag.x, 0.0, 0.0,
        lOffDiag.x, lDiag.y, 0.0,
        lOffDiag.y, lOffDiag.z, lDiag.z
    );
    float3x3 lowerTranspose = transpose(lower);
    return mul(lower, lowerTranspose);
}

[Differentiable]
float3x3 build_shape_matrix_from_axes(float3 axisX, float3 axisY, float3 axisZ)
{
    return float3x3(
        dot(axisX, axisX), dot(axisX, axisY), dot(axisX, axisZ),
        dot(axisX, axisY), dot(axisY, axisY), dot(axisY, axisZ),
        dot(axisX, axisZ), dot(axisY, axisZ), dot(axisZ, axisZ)
    );
}

[Differentiable]
CachedEllipsoid build_cached_ellipsoid(no_diff CameraParams cameraParams, no_diff float radiusScale, float3 position, float3 logScale, float4 rotation)
{
    Gaussian3D gaussian;
    gaussian.position = position;
    gaussian.logScale = logScale;
    gaussian.rotation = rotation;
    gaussian.colorAlpha = 0.0;
    Ellipsoid ellipsoid = build_ellipsoid(cameraParams, radiusScale, gaussian);
    PinholeCamera camera = PinholeCamera.from_params(cameraParams);
    float3 invScale = ellipsoid.rasterInvScale;
    float3 axisX = rotate_and_scale_axis(float3(1.0, 0.0, 0.0), invScale, ellipsoid.rotation);
    float3 axisY = rotate_and_scale_axis(float3(0.0, 1.0, 0.0), invScale, ellipsoid.rotation);
    float3 axisZ = rotate_and_scale_axis(float3(0.0, 0.0, 1.0), invScale, ellipsoid.rotation);
    float3x3 shape = build_shape_matrix_from_axes(axisX, axisY, axisZ);
    float l00 = sqrt(max(shape[0][0], SMALL_VALUE));
    float l10 = shape[1][0] / l00;
    float l20 = shape[2][0] / l00;
    float l11 = sqrt(max(shape[1][1] - l10 * l10, SMALL_VALUE));
    float l21 = (shape[2][1] - l20 * l10) / l11;
    float l22 = sqrt(max(shape[2][2] - l20 * l20 - l21 * l21, SMALL_VALUE));
    return {
        camera.position_world() - ellipsoid.centerWorld,
        log(max(float3(l00, l11, l22), float3(SMALL_VALUE))),
        float3(l10, l20, l21)
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
float raster_eval_alpha(float3 roLocal, float3 logLDiag, float3 lOffDiag, float fusedOpacity, no_diff float3 rayDir, no_diff float alphaCutoff)
{
    float3x3 shape = build_shape_matrix(logLDiag, lOffDiag);
    float3 shapeRay = mul(shape, rayDir);
    float denom = dot(rayDir, shapeRay);
    if (denom <= RAY_DENOMINATOR_FLOOR) return 0.0;

    float3 shapeRo = mul(shape, roLocal);
    float numer = dot(rayDir, shapeRo);
    float tClosest = -numer / denom;
    if (tClosest <= 0.0) return 0.0;

    float rho2 = dot(roLocal, shapeRo) - numer * numer / denom;
    if (rho2 < 0.0) rho2 = 0.0;
    float alpha = fusedOpacity * exp(-ENVELOPE_EXPONENT_SCALE * rho2);
    if (alpha < alphaCutoff) return 0.0;
    return alpha;
}
