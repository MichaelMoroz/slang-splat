import "slangpy";

uniform Texture2D<float4> g_TorchInput;
uniform RWStructuredBuffer<float4> g_TorchOutput;
uniform uint g_TorchWidth;
uniform uint g_TorchHeight;

static const uint TORCH_PACK_TILE_SIZE = 8u;

[shader("compute")]
[numthreads(TORCH_PACK_TILE_SIZE, TORCH_PACK_TILE_SIZE, 1)]
void csPackOutputTexture(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    if (pixel.x >= g_TorchWidth || pixel.y >= g_TorchHeight) return;
    g_TorchOutput[pixel.y * g_TorchWidth + pixel.x] = g_TorchInput[pixel];
}
