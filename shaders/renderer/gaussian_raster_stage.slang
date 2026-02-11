import "gaussian_types.slang";
import "gaussian_math_utils.slang";

struct RasterState : IDifferentiable
{
    float3 accum;
    float trans;
};

void atomic_add_grad_pos(uint splatId, float4 grad)
{
    uint base = splatId * 4u;
    float old = 0.0;
    if (grad.x != 0.0) InterlockedAdd(g_GradSplatPosLocal[base + 0u], grad.x, old);
    if (grad.y != 0.0) InterlockedAdd(g_GradSplatPosLocal[base + 1u], grad.y, old);
    if (grad.z != 0.0) InterlockedAdd(g_GradSplatPosLocal[base + 2u], grad.z, old);
    if (grad.w != 0.0) InterlockedAdd(g_GradSplatPosLocal[base + 3u], grad.w, old);
}

void atomic_add_grad_inv_scale(uint splatId, float4 grad)
{
    uint base = splatId * 4u;
    float old = 0.0;
    if (grad.x != 0.0) InterlockedAdd(g_GradSplatInvScale[base + 0u], grad.x, old);
    if (grad.y != 0.0) InterlockedAdd(g_GradSplatInvScale[base + 1u], grad.y, old);
    if (grad.z != 0.0) InterlockedAdd(g_GradSplatInvScale[base + 2u], grad.z, old);
    if (grad.w != 0.0) InterlockedAdd(g_GradSplatInvScale[base + 3u], grad.w, old);
}

void atomic_add_grad_quat(uint splatId, float4 grad)
{
    uint base = splatId * 4u;
    float old = 0.0;
    if (grad.x != 0.0) InterlockedAdd(g_GradSplatQuat[base + 0u], grad.x, old);
    if (grad.y != 0.0) InterlockedAdd(g_GradSplatQuat[base + 1u], grad.y, old);
    if (grad.z != 0.0) InterlockedAdd(g_GradSplatQuat[base + 2u], grad.z, old);
    if (grad.w != 0.0) InterlockedAdd(g_GradSplatQuat[base + 3u], grad.w, old);
}

void atomic_add_grad_color_alpha(uint splatId, float4 grad)
{
    uint base = splatId * 4u;
    float old = 0.0;
    if (grad.x != 0.0) InterlockedAdd(g_GradScreenColorAlpha[base + 0u], grad.x, old);
    if (grad.y != 0.0) InterlockedAdd(g_GradScreenColorAlpha[base + 1u], grad.y, old);
    if (grad.z != 0.0) InterlockedAdd(g_GradScreenColorAlpha[base + 2u], grad.z, old);
    if (grad.w != 0.0) InterlockedAdd(g_GradScreenColorAlpha[base + 3u], grad.w, old);
}

[BackwardDerivative(load_splat_pos_global_bwd)]
float4 load_splat_pos_global(uint splatId)
{
    return g_SplatPosLocal[splatId];
}

void load_splat_pos_global_bwd(uint splatId, float4 dOut)
{
    atomic_add_grad_pos(splatId, dOut);
}

[BackwardDerivative(load_splat_inv_scale_global_bwd)]
float4 load_splat_inv_scale_global(uint splatId)
{
    return g_SplatInvScale[splatId];
}

void load_splat_inv_scale_global_bwd(uint splatId, float4 dOut)
{
    atomic_add_grad_inv_scale(splatId, dOut);
}

[BackwardDerivative(load_splat_quat_global_bwd)]
float4 load_splat_quat_global(uint splatId)
{
    return g_SplatQuat[splatId];
}

void load_splat_quat_global_bwd(uint splatId, float4 dOut)
{
    atomic_add_grad_quat(splatId, dOut);
}

[BackwardDerivative(load_splat_color_alpha_global_bwd)]
float4 load_splat_color_alpha_global(uint splatId)
{
    return g_ScreenColorAlpha[splatId];
}

void load_splat_color_alpha_global_bwd(uint splatId, float4 dOut)
{
    atomic_add_grad_color_alpha(splatId, dOut);
}

[BackwardDerivative(load_splat_pos_shared_bwd)]
float4 load_splat_pos_shared(uint localIdx)
{
    return gsRoLocal[localIdx];
}

void load_splat_pos_shared_bwd(uint localIdx, float4 dOut)
{
    atomic_add_grad_pos(gsBatchSplatId[localIdx], dOut);
}

[BackwardDerivative(load_splat_inv_scale_shared_bwd)]
float4 load_splat_inv_scale_shared(uint localIdx)
{
    return float4(gsInvScale[localIdx]);
}

void load_splat_inv_scale_shared_bwd(uint localIdx, float4 dOut)
{
    atomic_add_grad_inv_scale(gsBatchSplatId[localIdx], dOut);
}

[BackwardDerivative(load_splat_quat_shared_bwd)]
float4 load_splat_quat_shared(uint localIdx)
{
    return float4(gsQuat[localIdx]);
}

void load_splat_quat_shared_bwd(uint localIdx, float4 dOut)
{
    atomic_add_grad_quat(gsBatchSplatId[localIdx], dOut);
}

[BackwardDerivative(load_splat_color_alpha_shared_bwd)]
float4 load_splat_color_alpha_shared(uint localIdx)
{
    return float4(gsColorAlpha[localIdx]);
}

void load_splat_color_alpha_shared_bwd(uint localIdx, float4 dOut)
{
    atomic_add_grad_color_alpha(gsBatchSplatId[localIdx], dOut);
}

[Differentiable]
float raster_eval_alpha(uint localIdx, no_diff float3 rayDir)
{
    float3 roLocal = load_splat_pos_shared(localIdx).xyz;
    float3 invScale = load_splat_inv_scale_shared(localIdx).xyz;
    float4 q = load_splat_quat_shared(localIdx);
    float4 colorAlpha = load_splat_color_alpha_shared(localIdx);

    float3 rdLocal = rotate_by_quat(rayDir, q) * invScale;
    float denom = dot(rdLocal, rdLocal);
    if (denom <= EPS_RAY_DENOM) return 0.0;

    float tClosest = dot(rdLocal, -roLocal) / denom;
    if (tClosest <= 0.0) return 0.0;

    float3 closest = roLocal + rdLocal * tClosest;
    float rho = length(closest);
    if (rho >= 1.0) return 0.0;

    float coverage = saturate(1.0 - rho);
    coverage = coverage * coverage * (3.0 - 2.0 * coverage);
    float alpha = saturate(colorAlpha.w * coverage);
    if (alpha < g_alphaCutoff) return 0.0;
    return alpha;
}

[Differentiable]
RasterState raster_blend_step(RasterState state, uint localIdx, float alpha)
{
    float3 color = load_splat_color_alpha_shared(localIdx).xyz;
    return { state.accum + state.trans * alpha * color, state.trans * (1.0 - alpha) };
}

RasterState raster_undo_blend_step(RasterState nextState, float4 colorAlpha, float alpha)
{
    float denom = max(1.0 - alpha, EPS_SMALL);
    float prevTrans = nextState.trans / denom;
    float3 prevAccum = nextState.accum - prevTrans * alpha * colorAlpha.xyz;
    return { prevAccum, prevTrans };
}

void prefetch_batch_from_global(uint batchStart, uint batchEnd, uint lane)
{
    uint idx = batchStart + lane;
    if (idx >= batchEnd) return;

    uint splatId = g_SortedValues[idx];
    gsBatchSplatId[lane] = splatId;
    gsRoLocal[lane] = load_splat_pos_global(splatId);
    gsInvScale[lane] = half4(load_splat_inv_scale_global(splatId));
    gsQuat[lane] = half4(load_splat_quat_global(splatId));
    gsColorAlpha[lane] = half4(load_splat_color_alpha_global(splatId));
}

RasterState raster_forward_replay(
    uint start,
    uint end,
    bool pixelActive,
    float3 rayDir,
    uint lane,
    out uint processedCountOut
)
{
    RasterState state = { 0.0, 1.0 };
    processedCountOut = 0u;
    bool pixelDone = !pixelActive;

    if (lane == 0u) gsStoppedPixels = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (pixelDone) InterlockedAdd(gsStoppedPixels, 1u);

    for (uint batchStart = start; batchStart < end; batchStart += RASTER_BATCH)
    {
        prefetch_batch_from_global(batchStart, end, lane);
        GroupMemoryBarrierWithGroupSync();

        uint batchCount = min(RASTER_BATCH, end - batchStart);
        for (uint localIdx = 0u; localIdx < batchCount; localIdx++)
        {
            if (pixelDone) continue;
            processedCountOut += 1u;

            float alpha = raster_eval_alpha(localIdx, rayDir);
            if (alpha <= 0.0) continue;

            state = raster_blend_step(state, localIdx, alpha);
            if (state.trans < g_transmittanceThreshold)
            {
                pixelDone = true;
                InterlockedAdd(gsStoppedPixels, 1u);
            }
        }

        GroupMemoryBarrierWithGroupSync();
        if (gsStoppedPixels == RASTER_BATCH) break;
    }

    return state;
}

float3 output_to_linear_grad(float3 finalColor, float3 dOutputRgb)
{
    float3 dLinear = 0.0;
    if (finalColor.x > 0.0) dLinear.x = dOutputRgb.x * GAMMA_LINEAR_TO_OUTPUT * pow(finalColor.x, GAMMA_LINEAR_TO_OUTPUT - 1.0);
    if (finalColor.y > 0.0) dLinear.y = dOutputRgb.y * GAMMA_LINEAR_TO_OUTPUT * pow(finalColor.y, GAMMA_LINEAR_TO_OUTPUT - 1.0);
    if (finalColor.z > 0.0) dLinear.z = dOutputRgb.z * GAMMA_LINEAR_TO_OUTPUT * pow(finalColor.z, GAMMA_LINEAR_TO_OUTPUT - 1.0);
    return dLinear;
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csClearRasterGrads(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint gradCount = g_splatCount * 4u;
    uint idx = dispatchThreadID.x;
    if (idx >= gradCount) return;

    g_GradSplatPosLocal[idx] = 0.0;
    g_GradSplatInvScale[idx] = 0.0;
    g_GradSplatQuat[idx] = 0.0;
    g_GradScreenColorAlpha[idx] = 0.0;
}

[shader("compute")]
[numthreads(RASTER_TILE_DIM, RASTER_TILE_DIM, 1)]
void csRasterize(uint3 dispatchThreadID : SV_DispatchThreadID, uint lane : SV_GroupIndex)
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
                gsBatchSplatId[lane] = splatId;
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

[shader("compute")]
[numthreads(RASTER_TILE_DIM, RASTER_TILE_DIM, 1)]
void csRasterizeBackward(uint3 dispatchThreadID : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    if (g_debugShowProcessedCount != 0u || g_debugShowEllipses != 0u) return;

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

    uint start = range.x;
    uint end = hasTileRange ? min(range.y, start + g_maxSplatSteps) : start;
    float2 rayUV = (float2(samplePx, samplePy) + 0.5 - 0.5 * g_viewport) / g_focalPixels;
    float3 rayDir = normalize(g_camForward + rayUV.x * g_camRight + rayUV.y * g_camUp);

    uint processedCount = 0u;
    RasterState state = { 0.0, 1.0 };
    if (hasTileRange) state = raster_forward_replay(start, end, pixelActive, rayDir, lane, processedCount);

    RasterState.Differential dState;
    dState.accum = 0.0;
    dState.trans = 0.0;
    if (pixelActive && hasTileRange)
    {
        float4 dOut = g_OutputGrad[uint2(px, py)];
        float3 finalColor = state.accum + state.trans * g_background;
        float3 dFinalColor = output_to_linear_grad(finalColor, dOut.xyz);
        dState.accum = dFinalColor;
        dState.trans = dot(dFinalColor, g_background) - dOut.w;
    }

    uint processedEnd = start + processedCount;
    for (uint batchBase = end; batchBase > start;)
    {
        uint batchStart = (batchBase > start + RASTER_BATCH) ? (batchBase - RASTER_BATCH) : start;
        uint batchCount = batchBase - batchStart;

        prefetch_batch_from_global(batchStart, batchBase, lane);
        GroupMemoryBarrierWithGroupSync();

        for (uint rev = 0u; rev < batchCount; rev++)
        {
            uint localIdx = batchCount - 1u - rev;
            uint globalIdx = batchStart + localIdx;
            if (!pixelActive || !hasTileRange || globalIdx >= processedEnd) continue;

            float alpha = raster_eval_alpha(localIdx, rayDir);
            if (alpha <= 0.0) continue;

            float4 colorAlpha = float4(gsColorAlpha[localIdx]);
            RasterState prevState = raster_undo_blend_step(state, colorAlpha, alpha);

            var dpPrevState = diffPair(prevState);
            var dpAlpha = diffPair(alpha);
            bwd_diff(raster_blend_step)(dpPrevState, localIdx, dpAlpha, dState);
            bwd_diff(raster_eval_alpha)(localIdx, rayDir, dpAlpha.getDifferential());

            state = prevState;
            dState = dpPrevState.getDifferential();
        }

        GroupMemoryBarrierWithGroupSync();
        batchBase = batchStart;
    }
}
