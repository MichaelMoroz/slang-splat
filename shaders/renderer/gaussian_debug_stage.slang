import "gaussian_types.slang";
import "utility/math/constants.slang";
import "utility/splatting/common.slang";
import "utility/splatting/opacity.slang";

static const float DEBUG_GRAD_NORM_FLOOR = 1e-12;
static const float DEBUG_GRAD_ALPHA_SCALE = 0.1;
static const float LOG2_TO_LOG10 = 0.30102999566;
static const float DEBUG_GRAD_THRESHOLD_MIN_SCALE = 0.001;
static const float DEBUG_GRAD_THRESHOLD_MAX_SCALE = 10.0;

uniform Texture2D<float4> g_BaseOutput;

struct DebugSplat
{
    float2 centerPx;
    float radiusPx;
    float3 conic;
    float4 colorAlpha;
    float gradNorm;

    static DebugSplat load(uint splatId)
    {
        float4 centerRadiusDepth = g_ScreenCenterRadiusDepth[splatId];
        float4 conic = g_ScreenEllipseConic[splatId];
        float gradNorm = g_DebugGradNorm[splatId];
        if (!isfinite(gradNorm)) gradNorm = 0.0;
        return {
            centerRadiusDepth.xy,
            centerRadiusDepth.z,
            conic.xyz,
            g_ScreenColorAlpha[splatId],
            max(gradNorm, 0.0)
        };
    }

    bool is_valid()
    {
        return radiusPx > 0.0;
    }

    bool is_near_sample(float2 samplePoint, float radiusPad)
    {
        float2 delta = samplePoint - centerPx;
        return all(abs(delta) <= radiusPx + radiusPad);
    }

    float eval_quad(float2 samplePoint)
    {
        float2 delta = samplePoint - centerPx;
        return dot(delta, mul(float2x2(conic.x, conic.y, conic.y, conic.z), delta));
    }

    bool contains(float2 samplePoint)
    {
        return eval_quad(samplePoint) <= 1.0 + CONIC_INTERSECTION_SLACK;
    }

    float eval_alpha(float2 samplePoint)
    {
        float quad = eval_quad(samplePoint);
        if (quad >= 1.0) return 0.0;
        float coverage = saturate(1.0 - quad);
        coverage = coverage * coverage * (3.0 - 2.0 * coverage);
        float alpha = opacity_from_raw(colorAlpha.w) * coverage;
        return alpha >= g_Raster.alphaCutoff ? alpha : 0.0;
    }

    float eval_edge(float2 samplePoint, float thicknessPx)
    {
        float2 delta = samplePoint - centerPx;
        float2 grad = mul(float2x2(conic.x, conic.y, conic.y, conic.z), delta);
        float quad = dot(delta, grad);
        float gradLen = max(length(2.0 * grad), GRADIENT_LENGTH_FLOOR);
        float edgeDistance = abs(quad - 1.0) / gradLen;
        return saturate(1.0 - edgeDistance / thicknessPx);
    }
};

float3 jet_colormap(float x)
{
    float t = saturate(x);
    return float3(
        saturate(1.5 - abs(4.0 * t - 3.0)),
        saturate(1.5 - abs(4.0 * t - 2.0)),
        saturate(1.5 - abs(4.0 * t - 1.0))
    );
}

float3 render_processed_count(float2 samplePoint)
{
    uint processedCount = 0u;
    for (uint splatId = 0u; splatId < g_Prepass.splatCount; splatId++)
    {
        if (g_SplatVisible[splatId] == 0u) continue;
        DebugSplat splat = DebugSplat.load(splatId);
        if (!splat.is_valid() || !splat.is_near_sample(samplePoint, 0.0) || !splat.contains(samplePoint)) continue;
        processedCount += 1u;
    }
    float logMax = log2(float(g_Raster.maxSplatSteps) + 1.0);
    float normalized = (logMax > 0.0) ? saturate(log2(float(processedCount) + 1.0) / logMax) : 0.0;
    return float3(normalized);
}

float3 render_grad_norm(float2 samplePoint)
{
    float accum = 0.0;
    float trans = 1.0;
    for (uint splatId = 0u; splatId < g_Prepass.splatCount; splatId++)
    {
        if (g_SplatVisible[splatId] == 0u) continue;
        DebugSplat splat = DebugSplat.load(splatId);
        if (!splat.is_valid() || !splat.is_near_sample(samplePoint, 0.0)) continue;
        float alpha = DEBUG_GRAD_ALPHA_SCALE * splat.eval_alpha(samplePoint);
        if (alpha <= 0.0) continue;
        accum += trans * alpha * splat.gradNorm;
        trans *= 1.0 - alpha;
        if (trans < g_Raster.transmittanceThreshold) break;
    }

    float gradValue = max(accum, DEBUG_GRAD_NORM_FLOOR);
    float gradLog10 = log2(gradValue) * LOG2_TO_LOG10;
    float gradThreshold = max(g_Raster.debugGradNormThreshold, DEBUG_GRAD_NORM_FLOOR);
    float gradLog10Min = log2(max(gradThreshold * DEBUG_GRAD_THRESHOLD_MIN_SCALE, DEBUG_GRAD_NORM_FLOOR)) * LOG2_TO_LOG10;
    float gradLog10Max = log2(max(gradThreshold * DEBUG_GRAD_THRESHOLD_MAX_SCALE, DEBUG_GRAD_NORM_FLOOR)) * LOG2_TO_LOG10;
    float normalized = saturate((gradLog10 - gradLog10Min) / max(gradLog10Max - gradLog10Min, TINY_VALUE));
    return jet_colormap(normalized);
}

float4 render_ellipse_overlay(float2 samplePoint, uint2 pixel)
{
    float4 base = g_BaseOutput[pixel];
    float maxEdge = 0.0;
    float3 edgeColor = g_Raster.debugEllipseColor;
    float thicknessPx = max(g_Raster.debugEllipseThicknessPx, MIN_DEBUG_THICKNESS_PX);
    for (uint splatId = 0u; splatId < g_Prepass.splatCount; splatId++)
    {
        if (g_SplatVisible[splatId] == 0u) continue;
        DebugSplat splat = DebugSplat.load(splatId);
        if (!splat.is_valid() || !splat.is_near_sample(samplePoint, thicknessPx)) continue;
        float edge = splat.eval_edge(samplePoint, thicknessPx);
        if (edge <= maxEdge) continue;
        maxEdge = edge;
        edgeColor = splat.colorAlpha.xyz;
    }
    return float4(lerp(base.xyz, edgeColor, maxEdge), base.w);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void csRenderDebug(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    if (pixel.x >= g_Raster.width || pixel.y >= g_Raster.height) return;

    float2 samplePoint = float2(pixel) + 0.5;
    if (g_Raster.debugShowProcessedCount != 0u)
    {
        g_Output[pixel] = float4(render_processed_count(samplePoint), 1.0);
        return;
    }
    if (g_Raster.debugShowGradNorm != 0u)
    {
        g_Output[pixel] = float4(render_grad_norm(samplePoint), 1.0);
        return;
    }
    g_Output[pixel] = render_ellipse_overlay(samplePoint, pixel);
}
