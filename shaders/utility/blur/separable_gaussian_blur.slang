static const uint BLUR_TILE_DIM = 16u;
static const uint GAUSSIAN_BLUR_RADIUS = 5u;
static const uint GAUSSIAN_BLUR_TAP_COUNT = GAUSSIAN_BLUR_RADIUS * 2u + 1u;
static const float GAUSSIAN_BLUR_WEIGHTS[GAUSSIAN_BLUR_TAP_COUNT] = {
    0.00102838008447911,
    0.00759875813523919,
    0.03600077212843083,
    0.10936068950970002,
    0.21300553771125369,
    0.26601172486179436,
    0.21300553771125369,
    0.10936068950970002,
    0.03600077212843083,
    0.00759875813523919,
    0.00102838008447911
};

uniform Texture2D<float4> g_BlurInput;
uniform RWTexture2D<float4> g_BlurOutput;
uniform uint g_BlurWidth;
uniform uint g_BlurHeight;

uint2 clamp_blur_pixel(int2 pixel)
{
    return uint2(clamp(pixel.x, 0, int(g_BlurWidth) - 1), clamp(pixel.y, 0, int(g_BlurHeight) - 1));
}

[Differentiable]
float4 gaussian_blur_taps(float4 taps[GAUSSIAN_BLUR_TAP_COUNT])
{
    float4 accum = 0.0;
    [unroll] for (uint tap = 0u; tap < GAUSSIAN_BLUR_TAP_COUNT; ++tap) accum += taps[tap] * GAUSSIAN_BLUR_WEIGHTS[tap];
    return accum;
}

[shader("compute")]
[numthreads(BLUR_TILE_DIM, BLUR_TILE_DIM, 1)]
void csGaussianBlurHorizontal(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= g_BlurWidth || py >= g_BlurHeight) return;

    float4 taps[GAUSSIAN_BLUR_TAP_COUNT];
    [unroll] for (uint tap = 0u; tap < GAUSSIAN_BLUR_TAP_COUNT; ++tap)
    {
        int offset = int(tap) - int(GAUSSIAN_BLUR_RADIUS);
        taps[tap] = g_BlurInput[clamp_blur_pixel(int2(int(px) + offset, int(py)))];
    }
    g_BlurOutput[uint2(px, py)] = gaussian_blur_taps(taps);
}

[shader("compute")]
[numthreads(BLUR_TILE_DIM, BLUR_TILE_DIM, 1)]
void csGaussianBlurVertical(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= g_BlurWidth || py >= g_BlurHeight) return;

    float4 taps[GAUSSIAN_BLUR_TAP_COUNT];
    [unroll] for (uint tap = 0u; tap < GAUSSIAN_BLUR_TAP_COUNT; ++tap)
    {
        int offset = int(tap) - int(GAUSSIAN_BLUR_RADIUS);
        taps[tap] = g_BlurInput[clamp_blur_pixel(int2(int(px), int(py) + offset))];
    }
    g_BlurOutput[uint2(px, py)] = gaussian_blur_taps(taps);
}
