import "gaussian_types.slang";
import "gaussian_math_utils.slang";

void rasterize_impl(uint3 dispatchThreadID, uint lane)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    bool pixelActive = px < g_width && py < g_height;
    uint samplePx = min(px, g_width - 1u);
    uint samplePy = min(py, g_height - 1u);

    uint tileX = samplePx / g_tileSize;
    uint tileY = samplePy / g_tileSize;
    uint tileId = tileY * g_tileWidth + tileX;
    uint2 range = g_TileRanges[tileId];
    bool hasTileRange = range.x != INVALID_START && range.y > range.x;
    float4 outputValue = float4(0.0, 0.0, 0.0, 1.0);
    if (hasTileRange)
    {
        uint start = range.x;
        uint end = min(range.y, start + g_maxSplatSteps);
        float3 accum = 0.0;
        float trans = 1.0;
        uint processedCount = 0u;
        float2 rayUV = (float2(samplePx, samplePy) + 0.5 - 0.5 * g_viewport) / g_focalPixels;
        float3 rayDir = normalize(g_camForward + rayUV.x * g_camRight + rayUV.y * g_camUp);
        bool pixelDone = !pixelActive;
        if (lane == 0u) gsStoppedPixels = 0u;
        GroupMemoryBarrierWithGroupSync();
        if (pixelDone) InterlockedAdd(gsStoppedPixels, 1u);

        for (uint batchStart = start; batchStart < end; batchStart += RASTER_BATCH)
        {
            uint idx = batchStart + lane;
            if (idx < end)
            {
                uint splatId = g_SortedValues[idx];
                gsRoLocal[lane] = g_SplatPosLocal[splatId];
                gsInvScale[lane] = half4(g_SplatInvScale[splatId]);
                gsQuat[lane] = half4(g_SplatQuat[splatId]);
                gsColorAlpha[lane] = half4(g_ScreenColorAlpha[splatId]);
            }
            GroupMemoryBarrierWithGroupSync();

            uint batchCount = min(RASTER_BATCH, end - batchStart);
            for (uint localIdx = 0; localIdx < batchCount; localIdx++)
            {
                if (pixelDone) continue;
                processedCount += 1u;

                float3 roLocal = gsRoLocal[localIdx].xyz;
                float3 invScale = float3(gsInvScale[localIdx].xyz);
                float4 q = float4(gsQuat[localIdx]);
                float3 rdLocal = rotate_by_quat(rayDir, q) * invScale;
                float denom = dot(rdLocal, rdLocal);
                if (denom <= EPS_RAY_DENOM) continue;
                float tClosest = dot(rdLocal, -roLocal) / denom;
                if (tClosest <= 0.0) continue;
                float3 closest = roLocal + rdLocal * tClosest;
                float rho = length(closest);
                if (rho >= 1.0) continue;

                float4 colorAlpha = float4(gsColorAlpha[localIdx]);
                float coverage = saturate(1.0 - rho);
                coverage = coverage * coverage * (3.0 - 2.0 * coverage);
                float alpha = saturate(colorAlpha.w * coverage);
                if (alpha < g_alphaCutoff) continue;

                accum += trans * alpha * colorAlpha.xyz;
                trans *= 1.0 - alpha;
                if (trans < g_transmittanceThreshold)
                {
                    pixelDone = true;
                    InterlockedAdd(gsStoppedPixels, 1u);
                }
            }
            GroupMemoryBarrierWithGroupSync();
            if (gsStoppedPixels == RASTER_BATCH) break;
        }

        if (pixelActive)
        {
            if (g_debugShowProcessedCount != 0u)
            {
                float processed = float(processedCount);
                float logMax = log2(float(g_maxSplatSteps) + 1.0);
                float normalized = (logMax > 0.0) ? saturate(log2(processed + 1.0) / logMax) : 0.0;
                float3 debugColor = float3(normalized);
                outputValue = float4(debugColor, 1.0);
            } else {
                float3 finalColor = accum + trans * g_background;
                if (g_debugShowEllipses != 0u)
                {
                    float2 samplePoint = float2(px, py) + 0.5;
                    float maxEdge = 0.0;
                    float3 edgeColor = g_debugEllipseColor;
                    float thicknessPx = max(g_debugEllipseThicknessPx, MIN_DEBUG_THICKNESS_PX);
                    for (uint idx = start; idx < end; idx++)
                    {
                        uint splatId = g_SortedValues[idx];
                        float4 centerRadiusDepth = g_ScreenCenterRadiusDepth[splatId];
                        float2 d = samplePoint - centerRadiusDepth.xy;
                        float bound = centerRadiusDepth.z + thicknessPx;
                        if (any(abs(d) > bound)) continue;

                        float4 conic = g_ScreenEllipseConic[splatId];
                        if (conic.w <= 0.0) continue;
                        float2x2 conicMatrix = float2x2(conic.x, conic.y, conic.y, conic.z);
                        float2 grad = mul(conicMatrix, d);
                        float quad = dot(d, grad);
                        float gradLen = max(length(2.0 * grad), EPS_GRADIENT_LENGTH);
                        float edgeDistance = abs(quad - 1.0) / gradLen;
                        float edge = saturate(1.0 - edgeDistance / thicknessPx);
                        if (edge > maxEdge)
                        {
                            maxEdge = edge;
                            edgeColor = g_ScreenColorAlpha[splatId].xyz;
                        }
                    }
                    finalColor = lerp(finalColor, edgeColor, maxEdge);
                }

                float3 linearOut = pow(max(finalColor, 0.0), GAMMA_LINEAR_TO_OUTPUT);
                outputValue = float4(linearOut, 1.0 - trans);
            }
        }
    } else if (pixelActive) {
        float3 bgColor = (g_debugShowProcessedCount != 0u) ? float3(0.0) : pow(max(g_background, 0.0), GAMMA_LINEAR_TO_OUTPUT);
        outputValue = float4(bgColor, 1.0);
    }

    if (pixelActive) g_Output[uint2(px, py)] = outputValue;
}
