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
float3 symmetric_matrix3_mul(float3 diag, float3 offDiag, float3 value)
{
    return float3(
        diag.x * value.x + offDiag.x * value.y + offDiag.y * value.z,
        offDiag.x * value.x + diag.y * value.y + offDiag.z * value.z,
        offDiag.y * value.x + offDiag.z * value.y + diag.z * value.z
    );
}

[Differentiable]
float symmetric_matrix3_quadratic(float3 diag, float3 offDiag, float3 value)
{
    return dot(value, symmetric_matrix3_mul(diag, offDiag, value));
}

[Differentiable]
float symmetric_matrix3_determinant(float3 diag, float3 offDiag)
{
    return diag.x * diag.y * diag.z + 2.0 * offDiag.x * offDiag.y * offDiag.z - diag.x * offDiag.z * offDiag.z - diag.y * offDiag.y * offDiag.y - diag.z * offDiag.x * offDiag.x;
}

[Differentiable]
float symmetric_matrix3_diag_characteristic_inv_scale(float3 diag)
{
    return pow(max(diag.x * diag.y * diag.z, SMALL_VALUE), 1.0 / 6.0);
}

[Differentiable]
RasterGaussian build_raster_gaussian(no_diff CameraParams cameraParams, no_diff float radiusScale, no_diff float maxAnisotropy, no_diff float3 cameraPosition, no_diff uint shBand, Gaussian3D gaussian)
{
    Ellipsoid ellipsoid = build_ellipsoid(cameraParams, radiusScale, maxAnisotropy, gaussian);
    no_diff float3x3 cameraToWorld = transpose(cameraParams.camBasis);
    float3 axisX = world_to_local_dir(mul(cameraToWorld, float3(1.0, 0.0, 0.0)), ellipsoid.rasterScale, ellipsoid.rotation);
    float3 axisY = world_to_local_dir(mul(cameraToWorld, float3(0.0, 1.0, 0.0)), ellipsoid.rasterScale, ellipsoid.rotation);
    float3 axisZ = world_to_local_dir(mul(cameraToWorld, float3(0.0, 0.0, 1.0)), ellipsoid.rasterScale, ellipsoid.rotation);
    return {
        mul(cameraParams.camBasis, cameraParams.camPos - ellipsoid.centerWorld),
        float3(dot(axisX, axisX), dot(axisY, axisY), dot(axisZ, axisZ)),
        float3(dot(axisX, axisY), dot(axisX, axisZ), dot(axisY, axisZ)),
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
float raster_eval_alpha(float3 roCamera, float3 sigmaDiag, float3 sigmaOffDiag, float fusedOpacity, no_diff float3 rayDirCamera, no_diff float alphaCutoff)
{
    float denom = symmetric_matrix3_quadratic(sigmaDiag, sigmaOffDiag, rayDirCamera);
    if (denom <= RAY_DENOMINATOR_FLOOR) return 0.0;

    float numer = dot(rayDirCamera, symmetric_matrix3_mul(sigmaDiag, sigmaOffDiag, roCamera));
    float tClosest = -numer / denom;
    if (tClosest <= 0.0) return 0.0;

    float3 closest = roCamera + rayDirCamera * tClosest;
    float rho2 = symmetric_matrix3_quadratic(sigmaDiag, sigmaOffDiag, closest);
    if (rho2 < 0.0) rho2 = 0.0;
    float alpha = fusedOpacity * exp(-ENVELOPE_EXPONENT_SCALE * rho2);
    if (alpha < alphaCutoff) return 0.0;
    return alpha;
}
