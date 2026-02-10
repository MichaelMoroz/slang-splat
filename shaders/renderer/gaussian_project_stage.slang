import "gaussian_types.slang";
import "gaussian_math_utils.slang";
import "gaussian_mvee_utils.slang";
import "gaussian_binning_utils.slang";
import "gaussian_raster_stage.slang";

void project_and_bin_impl(uint3 dispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_splatCount)
    {
        return;
    }

    Gaussian3D gaussian = load_gaussian3d(splatId);
    Ellipsoid ellipsoid = build_ellipsoid(gaussian);
    float3 worldPos = ellipsoid.centerWorld;
    float3 rel = worldPos - g_camPos;
    float3 camPos = world_to_camera(rel);
    float camDistance = length(rel);
    float depthValue = camPos.z;

    float invDepth = 1.0 / max(depthValue, 1e-6);
    float px = camPos.x * g_focalPixels * invDepth + 0.5 * g_viewport.x;
    float py = camPos.y * g_focalPixels * invDepth + 0.5 * g_viewport.y;

    float4 q = ellipsoid.rotation;
    float3 invScale = ellipsoid.invScale;
    float3 roLocal = ellipsoid.roLocal;

    float2 fitCenter = float2(px, py);
    float fitRadius = g_maxSplatRadiusPx;
    float invFitRadius2 = 1.0 / max(fitRadius * fitRadius, 1e-6);
    float3 fitConic = float3(invFitRadius2, 0.0, invFitRadius2);

    float3 localPoints[MVEE_POINT_COUNT];
    bool silhouetteOk = generate_silhouette_points_local5(roLocal, g_sampled5Eps, localPoints);
    if (silhouetteOk)
    {
        float2 screenPoints[MVEE_POINT_COUNT];
        [unroll]
        for (uint i = 0; i < MVEE_POINT_COUNT; i++)
        {
            float3 worldPt = local_point_to_world(localPoints[i], worldPos, invScale, q);
            float2 screenPt = 0.0;
            bool pointOk = project_world_to_screen(worldPt, screenPt);
            screenPoints[i] = screenPt;
            silhouetteOk = silhouetteOk && pointOk;
        }

        if (silhouetteOk)
        {
            float2 mveeCenter = 0.0;
            float mveeRadius = 0.0;
            float3 mveeConic = 0.0;
            bool fitOk = fit_mvee_5pt_fixed_iters(
                screenPoints,
                min(g_sampled5MVEEIters, MVEE_MAX_ITERS),
                g_sampled5Eps,
                mveeCenter,
                mveeRadius,
                mveeConic
            );
            if (fitOk && all(isfinite(mveeCenter)) && isfinite(mveeRadius) && all(isfinite(mveeConic)))
            {
                fitCenter = mveeCenter;
                fitRadius = clamp(mveeRadius * g_sampled5SafetyScale + g_sampled5RadiusPadPx, 1.0, g_maxSplatRadiusPx);
                float conicScale = fitRadius / max(mveeRadius, g_sampled5Eps);
                float invScale2 = 1.0 / max(conicScale * conicScale, g_sampled5Eps);
                fitConic = mveeConic * invScale2;
            }
        }
    }

    g_ScreenCenterRadiusDepth[splatId] = float4(fitCenter, fitRadius, camDistance);
    g_ScreenColorAlpha[splatId] = gaussian.colorAlpha;
    g_ScreenEllipseConic[splatId] = float4(fitConic, 1.0);
    g_SplatPosLocal[splatId] = float4(roLocal, 0.0);
    g_SplatInvScale[splatId] = float4(invScale, 0.0);
    g_SplatQuat[splatId] = q;
    SplatBinRequest binRequest;
    binRequest.splatId = splatId;
    binRequest.camDistance = camDistance;
    binRequest.depthValue = depthValue;
    binRequest.depthCode = quantizeDepth(camDistance);
    binRequest.ellipse.init(fitCenter, fitRadius, fitConic);
    emit_splat_bins(binRequest);
}

void clear_tile_ranges_impl(uint3 dispatchThreadID)
{
    uint tileId = dispatchThreadID.x;
    if (tileId >= g_tileCount)
    {
        return;
    }
    g_TileRanges[tileId] = uint2(INVALID_START, 0);
}

void build_tile_ranges_impl(uint3 dispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    uint sortedCount = g_PrepassParams[g_sortedCountOffset];
    if (idx >= sortedCount)
    {
        return;
    }

    uint tile = g_SortedKeys[idx] >> g_depthBits;
    if (tile >= g_tileCount)
    {
        return;
    }
    uint prevTile = INVALID_START;
    if (idx > 0)
    {
        prevTile = g_SortedKeys[idx - 1] >> g_depthBits;
        if (prevTile >= g_tileCount)
        {
            prevTile = INVALID_START;
        }
    }
    uint nextTile = INVALID_START;
    if (idx + 1 < sortedCount)
    {
        nextTile = g_SortedKeys[idx + 1] >> g_depthBits;
        if (nextTile >= g_tileCount)
        {
            nextTile = INVALID_START;
        }
    }

    if (idx == 0 || tile != prevTile)
    {
        g_TileRanges[tile].x = idx;
    }
    if (idx + 1 == sortedCount || tile != nextTile)
    {
        g_TileRanges[tile].y = idx + 1;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csProjectAndBin(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    project_and_bin_impl(dispatchThreadID);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csClearTileRanges(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    clear_tile_ranges_impl(dispatchThreadID);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void csBuildTileRanges(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    build_tile_ranges_impl(dispatchThreadID);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void csRasterize(uint3 dispatchThreadID : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    rasterize_impl(dispatchThreadID, lane);
}
