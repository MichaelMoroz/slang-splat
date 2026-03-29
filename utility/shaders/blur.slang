import "slangpy";

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

bool blur_in_bounds(int2 pixel)
{
    return pixel.x >= 0 && pixel.y >= 0 && pixel.x < int(g_BlurWidth) && pixel.y < int(g_BlurHeight);
}

uint blur_index(int2 pixel, uint channel)
{
    return ((uint(pixel.y) * g_BlurWidth) + uint(pixel.x)) * g_BlurChannelCount + channel;
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
        int2 sample = int2(int(px) + offset, int(py));
        if (blur_in_bounds(sample)) accum += g_BlurInput[blur_index(sample, channel)] * GAUSSIAN_BLUR_WEIGHTS[tap];
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
        int2 sample = int2(int(px), int(py) + offset);
        if (blur_in_bounds(sample)) accum += g_BlurInput[blur_index(sample, channel)] * GAUSSIAN_BLUR_WEIGHTS[tap];
    }
    g_BlurOutput[((py * g_BlurWidth) + px) * g_BlurChannelCount + channel] = accum;
}
