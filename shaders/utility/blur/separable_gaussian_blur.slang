static const uint BLUR_TILE_DIM = 16u;
static const uint GAUSSIAN_BLUR_RADIUS = 5u;
static const uint GAUSSIAN_BLUR_TAP_COUNT = GAUSSIAN_BLUR_RADIUS * 2u + 1u;
static const uint BLUR_SHARED_STRIDE = BLUR_TILE_DIM + GAUSSIAN_BLUR_RADIUS * 2u;
static const uint BLUR_SHARED_SIZE = BLUR_TILE_DIM * BLUR_SHARED_STRIDE;
static const uint MAX_GAUSSIAN_TEXTURE_BLUR_RADIUS = 24u;
static const uint MAX_GAUSSIAN_TEXTURE_BLUR_TAP_COUNT = MAX_GAUSSIAN_TEXTURE_BLUR_RADIUS * 2u + 1u;
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
static const float GAUSSIAN_BLUR_EDGE_ACCUM_WEIGHTS[GAUSSIAN_BLUR_RADIUS + 1u] = {
    0.6330058624308972,
    0.3669941375691028,
    0.15398859985784916,
    0.04462791034814913,
    0.0086271382197183,
    0.00102838008447911
};

uniform StructuredBuffer<float> g_BlurInput;
uniform RWStructuredBuffer<float> g_BlurOutput;
uniform uint g_BlurWidth;
uniform uint g_BlurHeight;
uniform uint g_BlurChannelCount;
uniform StructuredBuffer<float> g_BlurWeightsDynamic;
uniform uint g_BlurDynamicRadius;
uniform Texture2D<float4> g_BlurTextureInput;
uniform RWTexture2D<float4> g_BlurTextureOutput;

groupshared float gsBlurTile[BLUR_SHARED_SIZE];

uint clamp_blur_index(int2 pixel, uint channel)
{
    uint2 clamped = uint2(clamp(pixel.x, 0, int(g_BlurWidth) - 1), clamp(pixel.y, 0, int(g_BlurHeight) - 1));
    return ((clamped.y * g_BlurWidth) + clamped.x) * g_BlurChannelCount + channel;
}

uint blur_index(uint2 pixel, uint channel)
{
    return ((pixel.y * g_BlurWidth) + pixel.x) * g_BlurChannelCount + channel;
}

uint blur_thread_index(uint2 groupThreadID)
{
    return groupThreadID.y * BLUR_TILE_DIM + groupThreadID.x;
}

uint horizontal_shared_index(uint localX, uint localY)
{
    return localY * BLUR_SHARED_STRIDE + localX;
}

uint vertical_shared_index(uint localX, uint localY)
{
    return localY * BLUR_TILE_DIM + localX;
}

void load_horizontal_blur_tile(uint2 groupID, uint2 groupThreadID, uint channel)
{
    uint threadIndex = blur_thread_index(groupThreadID);
    uint2 groupOrigin = groupID * BLUR_TILE_DIM;
    for (uint linear = threadIndex; linear < BLUR_SHARED_SIZE; linear += BLUR_TILE_DIM * BLUR_TILE_DIM)
    {
        uint localY = linear / BLUR_SHARED_STRIDE;
        uint localX = linear - localY * BLUR_SHARED_STRIDE;
        if (channel >= g_BlurChannelCount) gsBlurTile[linear] = 0.0;
        else
        {
            int2 sample = int2(int(groupOrigin.x + localX) - int(GAUSSIAN_BLUR_RADIUS), int(groupOrigin.y + localY));
            gsBlurTile[linear] = g_BlurInput[clamp_blur_index(sample, channel)];
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

void load_vertical_blur_tile(uint2 groupID, uint2 groupThreadID, uint channel)
{
    uint threadIndex = blur_thread_index(groupThreadID);
    uint2 groupOrigin = groupID * BLUR_TILE_DIM;
    for (uint linear = threadIndex; linear < BLUR_SHARED_SIZE; linear += BLUR_TILE_DIM * BLUR_TILE_DIM)
    {
        uint localY = linear / BLUR_TILE_DIM;
        uint localX = linear - localY * BLUR_TILE_DIM;
        if (channel >= g_BlurChannelCount) gsBlurTile[linear] = 0.0;
        else
        {
            int2 sample = int2(int(groupOrigin.x + localX), int(groupOrigin.y + localY) - int(GAUSSIAN_BLUR_RADIUS));
            gsBlurTile[linear] = g_BlurInput[clamp_blur_index(sample, channel)];
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

float horizontal_blur_from_shared(uint2 groupThreadID)
{
    float accum = 0.0;
    [unroll] for (uint tap = 0u; tap < GAUSSIAN_BLUR_TAP_COUNT; ++tap)
        accum += gsBlurTile[horizontal_shared_index(groupThreadID.x + tap, groupThreadID.y)] * GAUSSIAN_BLUR_WEIGHTS[tap];
    return accum;
}

float vertical_blur_from_shared(uint2 groupThreadID)
{
    float accum = 0.0;
    [unroll] for (uint tap = 0u; tap < GAUSSIAN_BLUR_TAP_COUNT; ++tap)
        accum += gsBlurTile[vertical_shared_index(groupThreadID.x, groupThreadID.y + tap)] * GAUSSIAN_BLUR_WEIGHTS[tap];
    return accum;
}

float horizontal_blur_adjoint_from_shared(uint2 groupThreadID, uint2 groupID, uint px)
{
    if (g_BlurWidth == 1u) return gsBlurTile[horizontal_shared_index(GAUSSIAN_BLUR_RADIUS, groupThreadID.y)];

    uint maxX = g_BlurWidth - 1u;
    if (px == 0u || px == maxX)
    {
        float accum = 0.0;
        uint groupOriginX = groupID.x * BLUR_TILE_DIM;
        uint edgeSpan = min(GAUSSIAN_BLUR_RADIUS, maxX);
        [unroll] for (uint distance = 0u; distance <= GAUSSIAN_BLUR_RADIUS; ++distance)
        {
            if (distance > edgeSpan) continue;
            uint sampleX = px == 0u ? distance : maxX - distance;
            uint localX = uint(int(sampleX) - int(groupOriginX) + int(GAUSSIAN_BLUR_RADIUS));
            accum += gsBlurTile[horizontal_shared_index(localX, groupThreadID.y)] * GAUSSIAN_BLUR_EDGE_ACCUM_WEIGHTS[distance];
        }
        return accum;
    }

    float accum = 0.0;
    uint groupOriginX = groupID.x * BLUR_TILE_DIM;
    [unroll] for (uint tap = 0u; tap < GAUSSIAN_BLUR_TAP_COUNT; ++tap)
    {
        int sampleX = int(px) - (int(tap) - int(GAUSSIAN_BLUR_RADIUS));
        if (sampleX < 0 || sampleX >= int(g_BlurWidth)) continue;
        uint localX = uint(sampleX - int(groupOriginX) + int(GAUSSIAN_BLUR_RADIUS));
        accum += gsBlurTile[horizontal_shared_index(localX, groupThreadID.y)] * GAUSSIAN_BLUR_WEIGHTS[tap];
    }
    return accum;
}

float vertical_blur_adjoint_from_shared(uint2 groupThreadID, uint2 groupID, uint py)
{
    if (g_BlurHeight == 1u) return gsBlurTile[vertical_shared_index(groupThreadID.x, GAUSSIAN_BLUR_RADIUS)];

    uint maxY = g_BlurHeight - 1u;
    if (py == 0u || py == maxY)
    {
        float accum = 0.0;
        uint groupOriginY = groupID.y * BLUR_TILE_DIM;
        uint edgeSpan = min(GAUSSIAN_BLUR_RADIUS, maxY);
        [unroll] for (uint distance = 0u; distance <= GAUSSIAN_BLUR_RADIUS; ++distance)
        {
            if (distance > edgeSpan) continue;
            uint sampleY = py == 0u ? distance : maxY - distance;
            uint localY = uint(int(sampleY) - int(groupOriginY) + int(GAUSSIAN_BLUR_RADIUS));
            accum += gsBlurTile[vertical_shared_index(groupThreadID.x, localY)] * GAUSSIAN_BLUR_EDGE_ACCUM_WEIGHTS[distance];
        }
        return accum;
    }

    float accum = 0.0;
    uint groupOriginY = groupID.y * BLUR_TILE_DIM;
    [unroll] for (uint tap = 0u; tap < GAUSSIAN_BLUR_TAP_COUNT; ++tap)
    {
        int sampleY = int(py) - (int(tap) - int(GAUSSIAN_BLUR_RADIUS));
        if (sampleY < 0 || sampleY >= int(g_BlurHeight)) continue;
        uint localY = uint(sampleY - int(groupOriginY) + int(GAUSSIAN_BLUR_RADIUS));
        accum += gsBlurTile[vertical_shared_index(groupThreadID.x, localY)] * GAUSSIAN_BLUR_WEIGHTS[tap];
    }
    return accum;
}

[shader("compute")]
[numthreads(BLUR_TILE_DIM, BLUR_TILE_DIM, 1)]
void csGaussianBlurHorizontal(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    uint channel = dispatchThreadID.z;
    load_horizontal_blur_tile(groupID.xy, groupThreadID.xy, channel);
    if (px >= g_BlurWidth || py >= g_BlurHeight || channel >= g_BlurChannelCount) return;
    g_BlurOutput[blur_index(uint2(px, py), channel)] = horizontal_blur_from_shared(groupThreadID.xy);
}

[shader("compute")]
[numthreads(BLUR_TILE_DIM, BLUR_TILE_DIM, 1)]
void csGaussianBlurVertical(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    uint channel = dispatchThreadID.z;
    load_vertical_blur_tile(groupID.xy, groupThreadID.xy, channel);
    if (px >= g_BlurWidth || py >= g_BlurHeight || channel >= g_BlurChannelCount) return;
    g_BlurOutput[blur_index(uint2(px, py), channel)] = vertical_blur_from_shared(groupThreadID.xy);
}

[shader("compute")]
[numthreads(BLUR_TILE_DIM, BLUR_TILE_DIM, 1)]
void csGaussianBlurHorizontalAdjoint(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    uint channel = dispatchThreadID.z;
    load_horizontal_blur_tile(groupID.xy, groupThreadID.xy, channel);
    if (px >= g_BlurWidth || py >= g_BlurHeight || channel >= g_BlurChannelCount) return;
    g_BlurOutput[blur_index(uint2(px, py), channel)] = horizontal_blur_adjoint_from_shared(groupThreadID.xy, groupID.xy, px);
}

[shader("compute")]
[numthreads(BLUR_TILE_DIM, BLUR_TILE_DIM, 1)]
void csGaussianBlurVerticalAdjoint(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    uint channel = dispatchThreadID.z;
    load_vertical_blur_tile(groupID.xy, groupThreadID.xy, channel);
    if (px >= g_BlurWidth || py >= g_BlurHeight || channel >= g_BlurChannelCount) return;
    g_BlurOutput[blur_index(uint2(px, py), channel)] = vertical_blur_adjoint_from_shared(groupThreadID.xy, groupID.xy, py);
}

[shader("compute")]
[numthreads(BLUR_TILE_DIM, BLUR_TILE_DIM, 1)]
void csGaussianBlurTextureHorizontal(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= g_BlurWidth || py >= g_BlurHeight) return;

    uint radius = min(g_BlurDynamicRadius, MAX_GAUSSIAN_TEXTURE_BLUR_RADIUS);
    float4 accum = 0.0;
    for (int offset = -int(radius); offset <= int(radius); ++offset)
    {
        uint tapIndex = uint(offset + int(MAX_GAUSSIAN_TEXTURE_BLUR_RADIUS));
        uint sampleX = uint(clamp(int(px) + offset, 0, int(g_BlurWidth) - 1));
        accum += g_BlurTextureInput[uint2(sampleX, py)] * g_BlurWeightsDynamic[tapIndex];
    }
    g_BlurTextureOutput[uint2(px, py)] = accum;
}

[shader("compute")]
[numthreads(BLUR_TILE_DIM, BLUR_TILE_DIM, 1)]
void csGaussianBlurTextureVertical(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= g_BlurWidth || py >= g_BlurHeight) return;

    uint radius = min(g_BlurDynamicRadius, MAX_GAUSSIAN_TEXTURE_BLUR_RADIUS);
    float4 accum = 0.0;
    for (int offset = -int(radius); offset <= int(radius); ++offset)
    {
        uint tapIndex = uint(offset + int(MAX_GAUSSIAN_TEXTURE_BLUR_RADIUS));
        uint sampleY = uint(clamp(int(py) + offset, 0, int(g_BlurHeight) - 1));
        accum += g_BlurTextureInput[uint2(px, sampleY)] * g_BlurWeightsDynamic[tapIndex];
    }
    g_BlurTextureOutput[uint2(px, py)] = accum;
}
