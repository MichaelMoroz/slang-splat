import "constants.slang";

[Differentiable]
inline float GammaToLinearSpaceExact(float value)
{
    if (value <= SRGB_TO_LINEAR_THRESHOLD) return value / SRGB_LINEAR_SCALE;
    if (value < 1.0) return pow((value + SRGB_GAMMA_OFFSET) / SRGB_GAMMA_SCALE, SRGB_TO_LINEAR_EXPONENT);
    return pow(value, SRGB_HDR_TO_LINEAR_EXPONENT);
}

[Differentiable]
inline float3 GammaToLinearSpaceExact(float3 value)
{
    return float3(
        GammaToLinearSpaceExact(value.x),
        GammaToLinearSpaceExact(value.y),
        GammaToLinearSpaceExact(value.z)
    );
}

[Differentiable]
inline float LinearToGammaSpaceExact(float value)
{
    if (value <= 0.0) return 0.0;
    if (value <= LINEAR_TO_SRGB_THRESHOLD) return SRGB_LINEAR_SCALE * value;
    if (value < 1.0) return SRGB_GAMMA_SCALE * pow(value, LINEAR_TO_SRGB_EXPONENT) - SRGB_GAMMA_OFFSET;
    return pow(value, LINEAR_TO_SRGB_HDR_EXPONENT);
}

[Differentiable]
inline float3 LinearToGammaSpaceExact(float3 value)
{
    return float3(
        LinearToGammaSpaceExact(value.x),
        LinearToGammaSpaceExact(value.y),
        LinearToGammaSpaceExact(value.z)
    );
}
