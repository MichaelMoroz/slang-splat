import "gaussian_types.slang";
import "gaussian_math_utils.slang";
import "gaussian_mvee_utils.slang";
import "gaussian_binning_utils.slang";
import "gaussian_scanline_utils.slang";

[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csProjectAndBin(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_splatCount) return;

    Gaussian3D gaussian = Gaussian3D.load(splatId);
    Ellipsoid ellipsoid = build_ellipsoid(gaussian);
    float3 cameraOffset = ellipsoid.centerWorld - g_camPos;
    float3 camPos = world_to_camera(cameraOffset);
    float camDistance = length(cameraOffset);
    float depthValue = camPos.z;

    float invDepth = 1.0 / max(depthValue, EPS_SMALL);
    float2 fitCenter = camPos.xy * (g_focalPixels * invDepth) + 0.5 * g_viewport;
    float fitRadius = g_maxSplatRadiusPx;
    float invFitRadius2 = 1.0 / max(fitRadius * fitRadius, EPS_SMALL);
    float3 fitConic = float3(invFitRadius2, 0.0, invFitRadius2);

    float3 localPoints[MVEE_POINT_COUNT];
    bool silhouetteOk = generate_silhouette_points_local5(ellipsoid.roLocal, g_sampled5Eps, localPoints);
    if (silhouetteOk)
    {
        float2 screenPoints[MVEE_POINT_COUNT];
        [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++)
        {
            float3 worldPt = local_point_to_world(localPoints[i], ellipsoid.centerWorld, ellipsoid.invScale, ellipsoid.rotation);
            float2 screenPt = 0.0;
            bool pointOk = project_world_to_screen(worldPt, screenPt);
            screenPoints[i] = screenPt;
            silhouetteOk = silhouetteOk && pointOk;
        }

        if (silhouetteOk)
        {
            MveeFitResult mveeFit;
            bool fitOk = fit_mvee_5pt_fixed_iters(
                screenPoints,
                min(g_sampled5MVEEIters, MVEE_MAX_ITERS),
                g_sampled5Eps,
                mveeFit
            );
            if (fitOk && all(isfinite(mveeFit.centerPx)) && isfinite(mveeFit.radiusPx) && all(isfinite(mveeFit.conic)))
            {
                fitCenter = mveeFit.centerPx;
                fitRadius = clamp(mveeFit.radiusPx * g_sampled5SafetyScale + g_sampled5RadiusPadPx, MIN_RADIUS_PX, g_maxSplatRadiusPx);
                float conicScale = fitRadius / max(mveeFit.radiusPx, g_sampled5Eps);
                float invScale2 = 1.0 / max(conicScale * conicScale, g_sampled5Eps);
                fitConic = mveeFit.conic * invScale2;
            }
        }
    }

    g_ScreenCenterRadiusDepth[splatId] = float4(fitCenter, fitRadius, camDistance);
    g_ScreenColorAlpha[splatId] = gaussian.colorAlpha;
    g_ScreenEllipseConic[splatId] = float4(fitConic, 1.0);
    g_SplatPosLocal[splatId] = float4(ellipsoid.roLocal, 0.0);
    g_SplatInvScale[splatId] = float4(ellipsoid.invScale, 0.0);
    g_SplatQuat[splatId] = ellipsoid.rotation;
    SplatBinRequest binRequest;
    binRequest.splatId = splatId;
    binRequest.camDistance = camDistance;
    binRequest.depthValue = depthValue;
    binRequest.depthCode = quantizeDepth(camDistance);
    binRequest.ellipse.init(fitCenter, fitRadius, fitConic);
    emit_splat_bins(binRequest);
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csComposeScanlineKeyValues(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint scanlineId = dispatchThreadID.x;
    if (scanlineId >= g_maxScanlineEntries) return;
    uint scanlineCount = g_ScanlineCounter[0];
    if (scanlineId >= scanlineCount) return;

    ScanlineWorkItem item = g_ScanlineWorkItems[scanlineId];
    float4 centerRadiusDepth = g_ScreenCenterRadiusDepth[item.splatId];
    float4 conicData = g_ScreenEllipseConic[item.splatId];
    Ellipse ellipse;
    ellipse.init(centerRadiusDepth.xy, centerRadiusDepth.z, conicData.xyz);
    float tileSize = float(g_tileSize);
    float halfTile = 0.5 * tileSize;
    if (!try_prepare_conic_for_binning(ellipse, halfTile)) return;

    ScanlineLoopConfig cfg;
    cfg.scanAlongX = item.scanAlongX != 0u;
    cfg.lineStart = 0;
    cfg.lineEnd = 0;
    cfg.minorMin = int(item.minorMin);
    cfg.minorMax = int(item.minorMax);
    cfg.minorLimit = cfg.scanAlongX ? g_tileWidth : g_tileHeight;
    cfg.lineLimit = cfg.scanAlongX ? g_tileHeight : g_tileWidth;

    ScanlineSpan span;
    if (!compute_scanline_tile_span_universal(ellipse, cfg, tileSize, item.lineCoord, span)) return;
    if (span.lineCount == 0u) return;

    uint baseIndex = 0;
    InterlockedAdd(g_ListCounter[0], span.lineCount, baseIndex);
    if (baseIndex >= g_maxListEntries) return;

    uint writeCount = min(span.lineCount, g_maxListEntries - baseIndex);
    for (uint i = 0; i < writeCount; i++)
    {
        uint minorCoord = span.lineStartMinor + i;
        if (!cfg.isMinorCoordValid(minorCoord)) continue;
        uint tileId = cfg.composeTileId(item.lineCoord, minorCoord, g_tileWidth);
        if (tileId >= g_tileCount) continue;
        uint writeIndex = baseIndex + i;
        g_Keys[writeIndex] = (tileId << g_depthBits) | item.depthCode;
        g_Values[writeIndex] = item.splatId;
    }
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csClearTileRanges(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tileId = dispatchThreadID.x;
    if (tileId >= g_tileCount) return;
    g_TileRanges[tileId] = uint2(INVALID_START, 0);
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csBuildTileRanges(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    uint sortedCount = g_PrepassParams[g_sortedCountOffset];
    if (idx >= sortedCount) return;

    uint tile = g_SortedKeys[idx] >> g_depthBits;
    if (tile >= g_tileCount) return;
    uint prevTile = INVALID_START;
    if (idx > 0)
    {
        prevTile = g_SortedKeys[idx - 1] >> g_depthBits;
        if (prevTile >= g_tileCount) prevTile = INVALID_START;
    }
    uint nextTile = INVALID_START;
    if (idx + 1 < sortedCount)
    {
        nextTile = g_SortedKeys[idx + 1] >> g_depthBits;
        if (nextTile >= g_tileCount) nextTile = INVALID_START;
    }

    if (idx == 0 || tile != prevTile) g_TileRanges[tile].x = idx;
    if (idx + 1 == sortedCount || tile != nextTile) g_TileRanges[tile].y = idx + 1;
}

