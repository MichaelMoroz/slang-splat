import "slangpy";

uniform Tensor<float4, 2> g_ViewImage;
uniform RWTexture2D<float4> g_Surface;
uniform uint2 g_Viewport;

[shader("compute")]
[numthreads(8, 8, 1)]
void csBlitOutput(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint2 pixel = dispatchThreadID.xy;
    if (pixel.x >= g_Viewport.x || pixel.y >= g_Viewport.y) return;
    g_Surface[pixel] = g_ViewImage[int2(int(pixel.x), int(pixel.y))];
}
