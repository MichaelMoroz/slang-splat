import "slangpy";

uniform Tensor<float4, 2> g_ViewImage;
uniform RWTexture2D<float4> g_Surface;
uniform uint2 g_Viewport;
uniform Tensor<float4, 2> g_LetterboxSource;
uniform uint g_LetterboxSourceWidth;
uniform uint g_LetterboxSourceHeight;
uniform uint g_LetterboxOutputWidth;
uniform uint g_LetterboxOutputHeight;
uniform Tensor<float4, 2> g_DebugRendered;
uniform Tensor<float4, 2> g_DebugTarget;
uniform float g_DebugDiffScale;

static float3 _loss_color(float loss) {
    float t = saturate(loss);
    return lerp(float3(0.0, 0.0, 0.0), float3(1.0, 0.92, 0.28), sqrt(t));
}

static float2 _letterbox_uv(uint2 pixel, uint sourceWidth, uint sourceHeight, uint outputWidth, uint outputHeight) {
    float sourceAspect = float(sourceWidth) / max(float(sourceHeight), 1.0);
    float outputAspect = float(outputWidth) / max(float(outputHeight), 1.0);
    float2 uv = (float2(pixel) + 0.5) / float2(max(outputWidth, 1), max(outputHeight, 1));
    if (sourceAspect > outputAspect) {
        float scaledHeight = outputAspect / max(sourceAspect, 1e-8);
        float y = (uv.y - 0.5) / max(scaledHeight, 1e-8) + 0.5;
        return float2(uv.x, y);
    }
    float scaledWidth = sourceAspect / max(outputAspect, 1e-8);
    float x = (uv.x - 0.5) / max(scaledWidth, 1e-8) + 0.5;
    return float2(x, uv.y);
}

static bool _inside_letterbox(float2 uv) {
    return all(uv >= 0.0.xx) && all(uv <= 1.0.xx);
}

static int2 _sample_coord(float2 uv, uint width, uint height) {
    return int2(
        clamp(int(uv.x * max(int(width), 1)), 0, max(int(width) - 1, 0)),
        clamp(int(uv.y * max(int(height), 1)), 0, max(int(height) - 1, 0))
    );
}

static uint2 _pixel_from_index(uint idx, uint width) {
    return uint2(idx % width, idx / max(width, 1u));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csBlitOutput(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint idx = dispatchThreadID.x;
    uint total = g_Viewport.x * g_Viewport.y;
    if (idx >= total) return;
    uint2 pixel = _pixel_from_index(idx, g_Viewport.x);
    if (pixel.x >= g_Viewport.x || pixel.y >= g_Viewport.y) return;
    g_Surface[pixel] = g_ViewImage[int2(int(pixel.x), int(pixel.y))];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csDebugLetterboxTensor(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint idx = dispatchThreadID.x;
    uint total = g_LetterboxOutputWidth * g_LetterboxOutputHeight;
    if (idx >= total) return;
    uint2 pixel = _pixel_from_index(idx, g_LetterboxOutputWidth);
    if (pixel.x >= g_LetterboxOutputWidth || pixel.y >= g_LetterboxOutputHeight) return;
    float2 uv = _letterbox_uv(pixel, g_LetterboxSourceWidth, g_LetterboxSourceHeight, g_LetterboxOutputWidth, g_LetterboxOutputHeight);
    if (!_inside_letterbox(uv)) {
        g_Surface[pixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    g_Surface[pixel] = g_LetterboxSource[_sample_coord(uv, g_LetterboxSourceWidth, g_LetterboxSourceHeight)];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csDebugAbsDiffLetterbox(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint idx = dispatchThreadID.x;
    uint total = g_LetterboxOutputWidth * g_LetterboxOutputHeight;
    if (idx >= total) return;
    uint2 pixel = _pixel_from_index(idx, g_LetterboxOutputWidth);
    if (pixel.x >= g_LetterboxOutputWidth || pixel.y >= g_LetterboxOutputHeight) return;
    float2 uv = _letterbox_uv(pixel, g_LetterboxSourceWidth, g_LetterboxSourceHeight, g_LetterboxOutputWidth, g_LetterboxOutputHeight);
    if (!_inside_letterbox(uv)) {
        g_Surface[pixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    int2 coord = _sample_coord(uv, g_LetterboxSourceWidth, g_LetterboxSourceHeight);
    float3 rendered = g_DebugRendered[coord].xyz;
    float3 target = g_DebugTarget[coord].xyz;
    g_Surface[pixel] = float4(abs(rendered - target) * g_DebugDiffScale, 1.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csDebugLossLetterbox(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint idx = dispatchThreadID.x;
    uint total = g_LetterboxOutputWidth * g_LetterboxOutputHeight;
    if (idx >= total) return;
    uint2 pixel = _pixel_from_index(idx, g_LetterboxOutputWidth);
    if (pixel.x >= g_LetterboxOutputWidth || pixel.y >= g_LetterboxOutputHeight) return;
    float2 uv = _letterbox_uv(pixel, g_LetterboxSourceWidth, g_LetterboxSourceHeight, g_LetterboxOutputWidth, g_LetterboxOutputHeight);
    if (!_inside_letterbox(uv)) {
        g_Surface[pixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    int2 coord = _sample_coord(uv, g_LetterboxSourceWidth, g_LetterboxSourceHeight);
    float3 rendered = g_DebugRendered[coord].xyz;
    float3 target = g_DebugTarget[coord].xyz;
    float loss = dot(abs(rendered - target), 1.0.xxx / 3.0) * g_DebugDiffScale;
    g_Surface[pixel] = float4(_loss_color(loss), 1.0);
}
