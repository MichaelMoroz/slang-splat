import "gaussian_types.slang";

float3 rotate_by_quat(float3 v, float4 q)
{
    float3 qv = q.yzw;
    return v + 2.0 * cross(cross(v, qv) + q.x * v, qv);
}

float3 world_to_camera(float3 v)
{
    return float3(dot(v, g_camRight), dot(v, g_camUp), dot(v, g_camForward));
}

uint quantizeDepth(float depthValue)
{
    float denom = max(g_farDepth - g_nearDepth, EPS_SMALL);
    float t = saturate((depthValue - g_nearDepth) / denom);
    uint maxValue = (1u << g_depthBits) - 1u;
    return min((uint)round(t * float(maxValue)), maxValue);
}

bool inverse3x3(float3x3 m, float eps, out float3x3 invOut)
{
    float a00 = m._m00;
    float a01 = m._m01;
    float a02 = m._m02;
    float a10 = m._m10;
    float a11 = m._m11;
    float a12 = m._m12;
    float a20 = m._m20;
    float a21 = m._m21;
    float a22 = m._m22;
    float det =
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20);
    if (abs(det) <= eps)
    {
        invOut = 0.0;
        return false;
    }
    float invDet = 1.0 / det;
    invOut[0] = float3(
        (a11 * a22 - a12 * a21),
        (a02 * a21 - a01 * a22),
        (a01 * a12 - a02 * a11)
    ) * invDet;
    invOut[1] = float3(
        (a12 * a20 - a10 * a22),
        (a00 * a22 - a02 * a20),
        (a02 * a10 - a00 * a12)
    ) * invDet;
    invOut[2] = float3(
        (a10 * a21 - a11 * a20),
        (a01 * a20 - a00 * a21),
        (a00 * a11 - a01 * a10)
    ) * invDet;
    return true;
}

bool project_world_to_screen(float3 worldPos, out float2 screenPos)
{
    float3 cam = world_to_camera(worldPos - g_camPos);
    if (cam.z <= EPS_SMALL)
    {
        screenPos = 0.0;
        return false;
    }

    float2 uv = cam.xy / cam.z;
    float r2 = dot(uv, uv);
    float distortion = 1.0 + g_projDistortionK1 * r2 + g_projDistortionK2 * r2 * r2;
    uv *= distortion;
    screenPos = uv * g_focalPixels + 0.5 * g_viewport;
    return all(isfinite(screenPos));
}

float3 local_point_to_world(float3 pLocal, float3 worldPos, float3 invScale, float4 q)
{
    float3 invScaleSafe = max(invScale, float3(EPS_SMALL));
    float3 localUnscaled = pLocal / invScaleSafe;
    float4 qInv = float4(q.x, -q.y, -q.z, -q.w);
    return worldPos + rotate_by_quat(localUnscaled, qInv);
}

Ellipsoid build_ellipsoid(Gaussian3D gaussian)
{
    Ellipsoid ellipsoid;
    ellipsoid.centerWorld = gaussian.position;
    float maxScale = max(gaussian.scale.x, max(gaussian.scale.y, gaussian.scale.z));
    ellipsoid.scale = max(gaussian.scale, maxScale * SCALE_FLOOR_RATIO) * g_radiusScale;
    ellipsoid.invScale = 1.0 / max(ellipsoid.scale, float3(EPS_SMALL));
    ellipsoid.rotation = gaussian.rotation;
    ellipsoid.roLocal = rotate_by_quat(g_camPos - ellipsoid.centerWorld, ellipsoid.rotation) * ellipsoid.invScale;
    return ellipsoid;
}
