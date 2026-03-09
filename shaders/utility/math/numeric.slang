import "utility/math/constants.slang";

float sanitize_scalar(float value, float clampAbs)
{
    if (!isfinite(value)) return 0.0;
    return clamp(value, -clampAbs, clampAbs);
}

float3 sanitize_float3(float3 value, float clampAbs)
{
    return float3(
        sanitize_scalar(value.x, clampAbs),
        sanitize_scalar(value.y, clampAbs),
        sanitize_scalar(value.z, clampAbs)
    );
}

float4 sanitize_float4(float4 value, float clampAbs)
{
    return float4(
        sanitize_scalar(value.x, clampAbs),
        sanitize_scalar(value.y, clampAbs),
        sanitize_scalar(value.z, clampAbs),
        sanitize_scalar(value.w, clampAbs)
    );
}

float3 clip_norm3(float3 value, float maxNorm)
{
    if (maxNorm <= 0.0) return value;
    float len = length(value);
    if (!isfinite(len) || len <= maxNorm) return value;
    return value * (maxNorm / max(len, TINY_VALUE));
}

float4 clip_norm4(float4 value, float maxNorm)
{
    if (maxNorm <= 0.0) return value;
    float len = length(value);
    if (!isfinite(len) || len <= maxNorm) return value;
    return value * (maxNorm / max(len, TINY_VALUE));
}
