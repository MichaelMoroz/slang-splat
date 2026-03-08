import "math_constants.slang";

[Differentiable]
float opacity_from_raw(float rawOpacity)
{
    if (rawOpacity >= 0.0)
    {
        float z = exp(-rawOpacity);
        return 1.0 / (1.0 + z);
    } else {
        float z = exp(rawOpacity);
        return z / (1.0 + z);
    }
}

float raw_opacity_from_alpha(float opacity)
{
    float alpha = clamp(opacity, TINY_VALUE, 1.0 - TINY_VALUE);
    return log(alpha) - log(1.0 - alpha);
}
