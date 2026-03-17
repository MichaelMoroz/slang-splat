import "gaussian_types.slang";
import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/splatting/common.slang";
import "utility/splatting/projection.slang";
import "utility/splatting/raster.slang";

void store_raster_cache_zero(uint splatId)
{
    uint splatCount = g_Prepass.splatCount;
    [unroll] for (uint fieldId = 0u; fieldId < RASTER_CACHE_PARAM_COUNT; fieldId++)
    {
        g_RasterCache[raster_cache_index(splatCount, fieldId, splatId)] = 0.0;
    }
}

void store_raster_cache(uint splatId, CachedEllipsoid cached, float4 colorAlpha)
{
    uint splatCount = g_Prepass.splatCount;
    g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_RO_LOCAL_X, splatId)] = cached.roLocal.x;
    g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_RO_LOCAL_Y, splatId)] = cached.roLocal.y;
    g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_RO_LOCAL_Z, splatId)] = cached.roLocal.z;
    g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_INV_SCALE_X, splatId)] = cached.logInvScale.x;
    g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_INV_SCALE_Y, splatId)] = cached.logInvScale.y;
    g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_INV_SCALE_Z, splatId)] = cached.logInvScale.z;
    g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_QUAT_X, splatId)] = cached.quat.x;
    g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_QUAT_Y, splatId)] = cached.quat.y;
    g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_QUAT_Z, splatId)] = cached.quat.z;
    g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_QUAT_W, splatId)] = cached.quat.w;
    g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_COLOR_R, splatId)] = colorAlpha.x;
    g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_COLOR_G, splatId)] = colorAlpha.y;
    g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_COLOR_B, splatId)] = colorAlpha.z;
    g_RasterCache[raster_cache_index(splatCount, RASTER_CACHE_FUSED_OPACITY, splatId)] = colorAlpha.w;
}

bool emit_splat_bins(SplatBinRequest request)
{
    Ellipse ellipse = request.ellipse;
    if (!ellipse.isVisible(
        request.camDistance,
        request.depthValue,
        g_Camera.viewport,
        g_Camera.nearDepth,
        g_Camera.farDepth
    )) return false;

    float tileSize = float(g_Prepass.tileSize);
    float halfTile = 0.5 * tileSize;
    if (!try_prepare_conic_for_binning(ellipse, halfTile)) return false;
    float2 boundsExtent = ellipse.bboxExtent + halfTile;
    int minX = int(floor((ellipse.centerPx.x - boundsExtent.x) / tileSize));
    int maxX = int(ceil((ellipse.centerPx.x + boundsExtent.x) / tileSize));
    int minY = int(floor((ellipse.centerPx.y - boundsExtent.y) / tileSize));
    int maxY = int(ceil((ellipse.centerPx.y + boundsExtent.y) / tileSize));
    minX = clamp(minX, 0, int(g_Prepass.tileWidth) - 1);
    maxX = clamp(maxX, 0, int(g_Prepass.tileWidth) - 1);
    minY = clamp(minY, 0, int(g_Prepass.tileHeight) - 1);
    maxY = clamp(maxY, 0, int(g_Prepass.tileHeight) - 1);
    if (minX > maxX || minY > maxY) return false;

    ScanlineLoopConfig loopCfg = build_scanline_loop_config(g_Prepass, ellipse, minX, maxX, minY, maxY);
    uint scanlineCount = uint(loopCfg.lineEnd - loopCfg.lineStart + 1);
    if (scanlineCount == 0u) return false;

    uint scanlineBase = 0;
    InterlockedAdd(g_ScanlineCounter[0], scanlineCount, scanlineBase);
    if (scanlineBase >= g_Prepass.maxScanlineEntries) return false;

    uint scanlineWriteCount = min(scanlineCount, g_Prepass.maxScanlineEntries - scanlineBase);
    if (scanlineWriteCount == 0u) return false;

    uint writtenListCount = 0u;
    for (uint i = 0; i < scanlineWriteCount; i++)
    {
        uint lineCoord = uint(loopCfg.lineStart + int(i));
        ScanlineSpan span;
        ScanlineWorkItem item;
        item.splatId = request.splatId;
        item.depthCode = request.depthCode;
        item.lineCoord = lineCoord;
        item.lineStartMinor = 0u;
        item.lineCount = 0u;
        item.splatListOffset = 0u;
        item.scanAlongX = loopCfg.scanAlongX ? 1u : 0u;
        item.reserved0 = 0u;
        bool hasSpan = compute_scanline_tile_span_universal(ellipse, loopCfg, tileSize, lineCoord, span);
        if (hasSpan && span.lineCount > 0u)
        {
            item.lineStartMinor = span.lineStartMinor;
            item.lineCount = span.lineCount;
            item.splatListOffset = writtenListCount;
            writtenListCount += span.lineCount;
        }

        uint scanlineIndex = scanlineBase + i;
        g_ScanlineWorkItems[scanlineIndex] = item;
    }
    if (writtenListCount == 0u) return false;

    uint listBase = 0u;
    InterlockedAdd(g_ListCounter[0], writtenListCount, listBase);
    g_SplatListBases[request.splatId] = listBase;
    return true;
}

[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csProjectAndBin(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_Prepass.splatCount) return;
    g_ScreenCenterRadiusDepth[splatId] = 0.0;
    g_ScreenColorAlpha[splatId] = 0.0;
    g_ScreenEllipseConic[splatId] = 0.0;
    g_SplatVisible[splatId] = 0u;
    store_raster_cache_zero(splatId);

    PinholeCamera camera = PinholeCamera.from_params(g_Camera);
    Gaussian3D gaussian = load_gaussian3d(splatId);
    Ellipsoid ellipsoid = build_ellipsoid(g_Camera, g_Prepass.radiusScale, gaussian);
    CachedEllipsoid cached = { ellipsoid.rasterRoLocal, log(max(ellipsoid.rasterInvScale, float3(SMALL_VALUE))), ellipsoid.rotation };
    float4 rasterColorAlpha = float4(
        gaussian.colorAlpha.xyz,
        build_raster_fused_opacity(g_Camera, g_Prepass.radiusScale, gaussian.position, gaussian.logScale, gaussian.colorAlpha.w)
    );
    float3 cameraOffset = ellipsoid.centerWorld - camera.position_world();
    float3 camPos = camera.world_to_camera(cameraOffset);
    float camDistance = length(cameraOffset);
    float depthValue = camPos.z;

    float invDepth = 1.0 / max(depthValue, SMALL_VALUE);
    float2 fitCenter = camera.project_camera_to_screen_linear(camPos);
    float maxAxisWorld = max(ellipsoid.projectedScale.x, max(ellipsoid.projectedScale.y, ellipsoid.projectedScale.z));
    float fallbackFocal = max(g_Camera.focalPixels.x, g_Camera.focalPixels.y);
    float radiusCap = screen_radius_cap(g_Camera.viewport);
    float fallbackRadius = fallbackFocal * maxAxisWorld * invDepth;
    float fitRadius = clamp(
        fallbackRadius * g_Prepass.sampled5SafetyScale + g_Prepass.sampled5RadiusPadPx,
        MIN_SCREEN_RADIUS_PX,
        radiusCap
    );
    float invFitRadius2 = 1.0 / max(fitRadius * fitRadius, SMALL_VALUE);
    float3 fitConic = float3(invFitRadius2, 0.0, invFitRadius2);

    float3 localPoints[MVEE_POINT_COUNT];
    bool silhouetteOk = generate_silhouette_points_local5(ellipsoid.projectedRoLocal, g_Prepass.sampled5Eps, localPoints);
    if (silhouetteOk)
    {
        float2 screenPoints[MVEE_POINT_COUNT];
        [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++)
        {
            float3 worldPt = local_point_to_world(localPoints[i], ellipsoid.centerWorld, ellipsoid.projectedInvScale, ellipsoid.rotation);
            float2 screenPt = 0.0;
            bool pointOk = camera.project_world_to_screen(worldPt, screenPt);
            screenPoints[i] = screenPt;
            silhouetteOk = silhouetteOk && pointOk;
        }

        if (silhouetteOk)
        {
            MveeFitResult mveeFit;
            bool fitOk = fit_mvee_5pt_fixed_iters(screenPoints, min(g_Prepass.sampled5MVEEIters, MVEE_MAX_ITERS), g_Prepass.sampled5Eps, mveeFit);
            if (fitOk && all(isfinite(mveeFit.centerPx)) && isfinite(mveeFit.radiusPx) && all(isfinite(mveeFit.conic)))
            {
                fitCenter = mveeFit.centerPx;
                fitRadius = clamp(
                    mveeFit.radiusPx * g_Prepass.sampled5SafetyScale + g_Prepass.sampled5RadiusPadPx,
                    MIN_SCREEN_RADIUS_PX,
                    radiusCap
                );
                float conicScale = fitRadius / max(mveeFit.radiusPx, g_Prepass.sampled5Eps);
                float invScale2 = 1.0 / max(conicScale * conicScale, g_Prepass.sampled5Eps);
                fitConic = mveeFit.conic * invScale2;
            }
        }
    }

    SplatBinRequest binRequest;
    binRequest.splatId = splatId;
    binRequest.camDistance = camDistance;
    binRequest.depthValue = depthValue;
    binRequest.depthCode = quantize_depth(g_Camera, g_Prepass, camDistance);
    binRequest.ellipse.init(fitCenter, fitRadius, fitConic);
    if (!emit_splat_bins(binRequest)) return;
    g_ScreenCenterRadiusDepth[splatId] = float4(fitCenter, fitRadius, camDistance);
    g_ScreenColorAlpha[splatId] = gaussian.colorAlpha;
    g_ScreenEllipseConic[splatId] = float4(fitConic, ellipsoid.opacityScale);
    g_SplatVisible[splatId] = 1u;
    store_raster_cache(splatId, cached, rasterColorAlpha);
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csComposeScanlineKeyValues(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint scanlineId = dispatchThreadID.x;
    uint scanlineCount = g_ScanlineCounter[0];
    if (scanlineId >= scanlineCount || scanlineId >= g_Prepass.maxScanlineEntries) return;

    ScanlineWorkItem item = g_ScanlineWorkItems[scanlineId];
    uint baseIndex = g_SplatListBases[item.splatId] + item.splatListOffset;
    if (baseIndex >= g_Prepass.maxListEntries || item.lineCount == 0u) return;

    uint writeCount = min(item.lineCount, g_Prepass.maxListEntries - baseIndex);
    uint lineCoord = item.lineCoord;
    bool scanAlongX = item.scanAlongX != 0u;
    for (uint i = 0; i < writeCount; i++)
    {
        uint minorCoord = item.lineStartMinor + i;
        uint tileId = scanAlongX
            ? (lineCoord * g_Prepass.tileWidth + minorCoord)
            : (minorCoord * g_Prepass.tileWidth + lineCoord);
        uint writeIndex = baseIndex + i;
        g_Keys[writeIndex] = (tileId << g_Prepass.depthBits) | item.depthCode;
        g_Values[writeIndex] = item.splatId;
    }
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csClearTileRanges(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tileId = dispatchThreadID.x;
    if (tileId >= g_Prepass.tileCount) return;
    g_TileRanges[tileId] = uint2(INVALID_START, 0);
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csBuildTileRanges(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    uint sortedCount = g_PrepassParams[g_Prepass.sortedCountOffset];
    if (idx >= sortedCount) return;

    uint tile = g_SortedKeys[idx] >> g_Prepass.depthBits;
    if (tile >= g_Prepass.tileCount) return;
    uint prevTile = INVALID_START;
    if (idx > 0)
    {
        prevTile = g_SortedKeys[idx - 1] >> g_Prepass.depthBits;
        if (prevTile >= g_Prepass.tileCount) prevTile = INVALID_START;
    }
    uint nextTile = INVALID_START;
    if (idx + 1 < sortedCount)
    {
        nextTile = g_SortedKeys[idx + 1] >> g_Prepass.depthBits;
        if (nextTile >= g_Prepass.tileCount) nextTile = INVALID_START;
    }

    if (idx == 0 || tile != prevTile) g_TileRanges[tile].x = idx;
    if (idx + 1 == sortedCount || tile != nextTile) g_TileRanges[tile].y = idx + 1;
}
