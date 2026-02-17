import "gaussian_types.slang";

[Differentiable]
float3 rotate_by_quat(float3 v, float4 q)
{
    float3 qv = q.yzw;
    return v + 2.0 * cross(cross(v, qv) + q.x * v, qv);
}

float3 world_to_camera(float3 v)
{
    return mul(g_Camera.camBasis, v);
}

float3 camera_to_world(float3 v)
{
    return mul(transpose(g_Camera.camBasis), v);
}

uint quantizeDepth(float depthValue)
{
    float denom = max(g_Camera.farDepth - g_Camera.nearDepth, EPS_SMALL);
    float t = saturate((depthValue - g_Camera.nearDepth) / denom);
    uint maxValue = (1u << g_Prepass.depthBits) - 1u;
    return min((uint)round(t * float(maxValue)), maxValue);
}

bool inverse3x3(float3x3 m, float eps, out float3x3 invOut)
{
    float det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
              - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
              + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if (abs(det) <= eps)
    {
        invOut = 0.0;
        return false;
    }
    float invDet = 1.0 / det;
    invOut[0] = float3(
        m[1][1] * m[2][2] - m[1][2] * m[2][1],
        m[0][2] * m[2][1] - m[0][1] * m[2][2],
        m[0][1] * m[1][2] - m[0][2] * m[1][1]
    ) * invDet;
    invOut[1] = float3(
        m[1][2] * m[2][0] - m[1][0] * m[2][2],
        m[0][0] * m[2][2] - m[0][2] * m[2][0],
        m[0][2] * m[1][0] - m[0][0] * m[1][2]
    ) * invDet;
    invOut[2] = float3(
        m[1][0] * m[2][1] - m[1][1] * m[2][0],
        m[0][1] * m[2][0] - m[0][0] * m[2][1],
        m[0][0] * m[1][1] - m[0][1] * m[1][0]
    ) * invDet;
    return true;
}

bool project_world_to_screen(float3 worldPos, out float2 screenPos)
{
    float3 cam = world_to_camera(worldPos - g_Camera.camPos);
    if (cam.z <= EPS_SMALL)
    {
        screenPos = 0.0;
        return false;
    }

    float2 uv = cam.xy / cam.z;
    float r2 = dot(uv, uv);
    float distortion = 1.0 + g_Camera.projDistortionK1 * r2 + g_Camera.projDistortionK2 * r2 * r2;
    uv *= distortion;
    screenPos = uv * g_Camera.focalPixels + g_Camera.principalPoint;
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
    ellipsoid.scale = max(gaussian.scale, maxScale * SCALE_FLOOR_RATIO) * g_Prepass.radiusScale;
    ellipsoid.invScale = 1.0 / max(ellipsoid.scale, float3(EPS_SMALL));
    ellipsoid.rotation = gaussian.rotation;
    ellipsoid.roLocal = rotate_by_quat(g_Camera.camPos - ellipsoid.centerWorld, ellipsoid.rotation) * ellipsoid.invScale;
    return ellipsoid;
}
