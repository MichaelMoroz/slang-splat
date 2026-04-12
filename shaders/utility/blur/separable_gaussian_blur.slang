static const uint BLUR_TILE_DIM = 16u;
static const uint BLUR_GROUP_SIZE = 64u;
static const uint GAUSSIAN_BLUR_RADIUS = 5u;
static const uint GAUSSIAN_BLUR_TAP_COUNT = GAUSSIAN_BLUR_RADIUS * 2u + 1u;
static const uint BLUR_SHARED_STRIDE = BLUR_GROUP_SIZE + GAUSSIAN_BLUR_RADIUS * 2u;
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

groupshared float gsHorizontalBlurRow[BLUR_SHARED_STRIDE];
groupshared float gsVerticalBlurColumn[BLUR_SHARED_STRIDE];

uint clamp_blur_index(int2 pixel, uint channel)
{
    uint2 clamped = uint2(clamp(pixel.x, 0, int(g_BlurWidth) - 1), clamp(pixel.y, 0, int(g_BlurHeight) - 1));
    return ((clamped.y * g_BlurWidth) + clamped.x) * g_BlurChannelCount + channel;
}

uint blur_index(uint2 pixel, uint channel)
{
    return ((pixel.y * g_BlurWidth) + pixel.x) * g_BlurChannelCount + channel;
}

void load_horizontal_blur_row(uint3 groupID, uint3 groupThreadID, uint channel)
{
    uint pxBase = groupID.x * BLUR_GROUP_SIZE;
    uint py = groupID.y;
    for (uint localX = groupThreadID.x; localX < BLUR_SHARED_STRIDE; localX += BLUR_GROUP_SIZE)
    {
        if (channel >= g_BlurChannelCount) gsHorizontalBlurRow[localX] = 0.0;
        else
        {
            int2 sample = int2(int(pxBase + localX) - int(GAUSSIAN_BLUR_RADIUS), int(py));
            gsHorizontalBlurRow[localX] = g_BlurInput[clamp_blur_index(sample, channel)];
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

void load_vertical_blur_column(uint3 groupID, uint3 groupThreadID, uint channel)
{
    uint px = groupID.x;
    uint pyBase = groupID.y * BLUR_GROUP_SIZE;
    for (uint localY = groupThreadID.y; localY < BLUR_SHARED_STRIDE; localY += BLUR_GROUP_SIZE)
    {
        if (channel >= g_BlurChannelCount) gsVerticalBlurColumn[localY] = 0.0;
        else
        {
            int2 sample = int2(int(px), int(pyBase + localY) - int(GAUSSIAN_BLUR_RADIUS));
            gsVerticalBlurColumn[localY] = g_BlurInput[clamp_blur_index(sample, channel)];
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

float horizontal_blur_from_shared(uint localThreadX)
{
    float accum = 0.0;
    [unroll] for (uint tap = 0u; tap < GAUSSIAN_BLUR_TAP_COUNT; ++tap) accum += gsHorizontalBlurRow[localThreadX + tap] * GAUSSIAN_BLUR_WEIGHTS[tap];
    return accum;
}

float vertical_blur_from_shared(uint localThreadY)
{
    float accum = 0.0;
    [unroll] for (uint tap = 0u; tap < GAUSSIAN_BLUR_TAP_COUNT; ++tap) accum += gsVerticalBlurColumn[localThreadY + tap] * GAUSSIAN_BLUR_WEIGHTS[tap];
    return accum;
}

float horizontal_blur_adjoint_from_shared(uint localThreadX, uint groupBaseX, uint px)
{
    if (g_BlurWidth == 1u) return gsHorizontalBlurRow[GAUSSIAN_BLUR_RADIUS];

    uint maxX = g_BlurWidth - 1u;
    if (px == 0u || px == maxX)
    {
        float accum = 0.0;
        uint edgeSpan = min(GAUSSIAN_BLUR_RADIUS, maxX);
        [unroll] for (uint distance = 0u; distance <= GAUSSIAN_BLUR_RADIUS; ++distance)
        {
            if (distance > edgeSpan) continue;
            uint sampleX = px == 0u ? distance : maxX - distance;
            uint localX = uint(int(sampleX) - int(groupBaseX) + int(GAUSSIAN_BLUR_RADIUS));
            accum += gsHorizontalBlurRow[localX] * GAUSSIAN_BLUR_EDGE_ACCUM_WEIGHTS[distance];
        }
        return accum;
    }

    float accum = 0.0;
    [unroll] for (uint tap = 0u; tap < GAUSSIAN_BLUR_TAP_COUNT; ++tap)
    {
        int sampleX = int(px) - (int(tap) - int(GAUSSIAN_BLUR_RADIUS));
        if (sampleX < 0 || sampleX >= int(g_BlurWidth)) continue;
        uint localX = uint(sampleX - int(groupBaseX) + int(GAUSSIAN_BLUR_RADIUS));
        accum += gsHorizontalBlurRow[localX] * GAUSSIAN_BLUR_WEIGHTS[tap];
    }
    return accum;
}

float vertical_blur_adjoint_from_shared(uint localThreadY, uint groupBaseY, uint py)
{
    if (g_BlurHeight == 1u) return gsVerticalBlurColumn[GAUSSIAN_BLUR_RADIUS];

    uint maxY = g_BlurHeight - 1u;
    if (py == 0u || py == maxY)
    {
        float accum = 0.0;
        uint edgeSpan = min(GAUSSIAN_BLUR_RADIUS, maxY);
        [unroll] for (uint distance = 0u; distance <= GAUSSIAN_BLUR_RADIUS; ++distance)
        {
            if (distance > edgeSpan) continue;
            uint sampleY = py == 0u ? distance : maxY - distance;
            uint localY = uint(int(sampleY) - int(groupBaseY) + int(GAUSSIAN_BLUR_RADIUS));
            accum += gsVerticalBlurColumn[localY] * GAUSSIAN_BLUR_EDGE_ACCUM_WEIGHTS[distance];
        }
        return accum;
    }

    float accum = 0.0;
    [unroll] for (uint tap = 0u; tap < GAUSSIAN_BLUR_TAP_COUNT; ++tap)
    {
        int sampleY = int(py) - (int(tap) - int(GAUSSIAN_BLUR_RADIUS));
        if (sampleY < 0 || sampleY >= int(g_BlurHeight)) continue;
        uint localY = uint(sampleY - int(groupBaseY) + int(GAUSSIAN_BLUR_RADIUS));
        accum += gsVerticalBlurColumn[localY] * GAUSSIAN_BLUR_WEIGHTS[tap];
    }
    return accum;
}

[shader("compute")]
[numthreads(BLUR_GROUP_SIZE, 1, 1)]
void csGaussianBlurHorizontal(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    uint channel = dispatchThreadID.z;
    load_horizontal_blur_row(groupID, groupThreadID, channel);
    if (px >= g_BlurWidth || py >= g_BlurHeight || channel >= g_BlurChannelCount) return;
    g_BlurOutput[blur_index(uint2(px, py), channel)] = horizontal_blur_from_shared(groupThreadID.x);
}

[shader("compute")]
[numthreads(1, BLUR_GROUP_SIZE, 1)]
void csGaussianBlurVertical(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    uint channel = dispatchThreadID.z;
    load_vertical_blur_column(groupID, groupThreadID, channel);
    if (px >= g_BlurWidth || py >= g_BlurHeight || channel >= g_BlurChannelCount) return;
    g_BlurOutput[blur_index(uint2(px, py), channel)] = vertical_blur_from_shared(groupThreadID.y);
}

[shader("compute")]
[numthreads(BLUR_GROUP_SIZE, 1, 1)]
void csGaussianBlurHorizontalAdjoint(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    uint channel = dispatchThreadID.z;
    load_horizontal_blur_row(groupID, groupThreadID, channel);
    if (px >= g_BlurWidth || py >= g_BlurHeight || channel >= g_BlurChannelCount) return;
    g_BlurOutput[blur_index(uint2(px, py), channel)] = horizontal_blur_adjoint_from_shared(groupThreadID.x, groupID.x * BLUR_GROUP_SIZE, px);
}

[shader("compute")]
[numthreads(1, BLUR_GROUP_SIZE, 1)]
void csGaussianBlurVerticalAdjoint(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID, uint3 groupID : SV_GroupID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    uint channel = dispatchThreadID.z;
    load_vertical_blur_column(groupID, groupThreadID, channel);
    if (px >= g_BlurWidth || py >= g_BlurHeight || channel >= g_BlurChannelCount) return;
    g_BlurOutput[blur_index(uint2(px, py), channel)] = vertical_blur_adjoint_from_shared(groupThreadID.y, groupID.y * BLUR_GROUP_SIZE, py);
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
