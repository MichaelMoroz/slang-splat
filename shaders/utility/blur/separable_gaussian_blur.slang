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

uniform StructuredBuffer<float> g_BlurInput;
uniform RWStructuredBuffer<float> g_BlurOutput;
uniform uint g_BlurWidth;
uniform uint g_BlurHeight;
uniform uint g_BlurChannelCount;

uint clamp_blur_index(int2 pixel, uint channel)
{
    uint2 clamped = uint2(clamp(pixel.x, 0, int(g_BlurWidth) - 1), clamp(pixel.y, 0, int(g_BlurHeight) - 1));
    return ((clamped.y * g_BlurWidth) + clamped.x) * g_BlurChannelCount + channel;
}

[shader("compute")]
[numthreads(BLUR_TILE_DIM, BLUR_TILE_DIM, 1)]
void csGaussianBlurHorizontal(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    uint channel = dispatchThreadID.z;
    if (px >= g_BlurWidth || py >= g_BlurHeight || channel >= g_BlurChannelCount) return;

    float accum = 0.0;
    [unroll] for (uint tap = 0u; tap < GAUSSIAN_BLUR_TAP_COUNT; ++tap)
    {
        int offset = int(tap) - int(GAUSSIAN_BLUR_RADIUS);
        accum += g_BlurInput[clamp_blur_index(int2(int(px) + offset, int(py)), channel)] * GAUSSIAN_BLUR_WEIGHTS[tap];
    }
    g_BlurOutput[((py * g_BlurWidth) + px) * g_BlurChannelCount + channel] = accum;
}

[shader("compute")]
[numthreads(BLUR_TILE_DIM, BLUR_TILE_DIM, 1)]
void csGaussianBlurVertical(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    uint channel = dispatchThreadID.z;
    if (px >= g_BlurWidth || py >= g_BlurHeight || channel >= g_BlurChannelCount) return;

    float accum = 0.0;
    [unroll] for (uint tap = 0u; tap < GAUSSIAN_BLUR_TAP_COUNT; ++tap)
    {
        int offset = int(tap) - int(GAUSSIAN_BLUR_RADIUS);
        accum += g_BlurInput[clamp_blur_index(int2(int(px), int(py) + offset), channel)] * GAUSSIAN_BLUR_WEIGHTS[tap];
    }
    g_BlurOutput[((py * g_BlurWidth) + px) * g_BlurChannelCount + channel] = accum;
}
