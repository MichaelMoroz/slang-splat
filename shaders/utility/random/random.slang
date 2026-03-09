import "utility/math/constants.slang";

static const uint RNG_LINEAR_A = 1664525u;
static const uint RNG_LINEAR_C = 1013904223u;

uint rng_next(inout uint state)
{
    state = RNG_LINEAR_A * state + RNG_LINEAR_C;
    return state;
}

uint hash_u32(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float rng_uniform_open01(inout uint state)
{
    uint bits = rng_next(state);
    return (float(bits) + 1.0) * RNG_OPEN01_SCALE;
}

float2 rng_normal2_box_muller(inout uint state)
{
    float u1 = max(rng_uniform_open01(state), TINY_VALUE);
    float u2 = rng_uniform_open01(state);
    float radius = sqrt(max(-2.0 * log(u1), 0.0));
    float s, c;
    sincos(TAU_RADIANS * u2, s, c);
    return radius * float2(c, s);
}

float3 rng_normal3_box_muller(inout uint state)
{
    float2 n01 = rng_normal2_box_muller(state);
    float2 n23 = rng_normal2_box_muller(state);
    return float3(n01.x, n01.y, n23.x);
}

float4 rng_normal4_box_muller(inout uint state)
{
    float2 n01 = rng_normal2_box_muller(state);
    float2 n23 = rng_normal2_box_muller(state);
    return float4(n01.x, n01.y, n23.x, n23.y);
}
