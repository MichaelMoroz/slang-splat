import "utility/math/constants.slang";

float4 normalize_quaternion_safe(float4 q)
{
    float normQ = length(q);
    if (!isfinite(normQ) || normQ < TINY_VALUE) return float4(1.0, 0.0, 0.0, 0.0);
    return q / normQ;
}

[Differentiable]
float3 rotate_by_quat(float3 v, float4 q)
{
    float3 qv = q.yzw;
    return v + 2.0 * cross(cross(v, qv) + q.x * v, qv);
}
