import "gaussian_types.slang";
import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/splatting/common.slang";
import "utility/splatting/projection.slang";
import "utility/splatting/raster.slang";

uniform float g_MaxAnisotropy;

uint float_to_ordered_uint(float value)
{
    uint bits = asuint(value);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}

void store_raster_cache(uint splatId, CachedEllipsoid cached, float4 colorAlpha)
{
    RasterGaussian gaussian;
    gaussian.roLocal = cached.roLocal;
    gaussian.scale = cached.scale;
    gaussian.rotation = cached.rotation;
    gaussian.colorAlpha = colorAlpha;
    g_RasterCache[splatId] = gaussian;
}

void append_visible_splat(uint distanceKey, uint splatId)
{
    uint writeIndex = 0u;
    InterlockedAdd(g_VisibleCounter[0], 1u, writeIndex);
    if (writeIndex >= g_Prepass.splatCount) return;
    g_VisibleKeys[writeIndex] = distanceKey;
    g_VisibleValues[writeIndex] = splatId;
}

bool load_projected_ellipse(uint splatId, out Ellipse ellipse)
{
    ellipse.init(0.0, 0.0, 0.0);
    float4 centerRadiusDepth = g_ScreenCenterRadiusDepth[splatId];
    if (centerRadiusDepth.z <= 0.0 || !all(isfinite(centerRadiusDepth))) return false;
    float4 conicOpacity = g_ScreenEllipseConic[splatId];
    ellipse.init(centerRadiusDepth.xy, centerRadiusDepth.z, conicOpacity.xyz);
    return all(isfinite(ellipse.centerPx)) && isfinite(ellipse.radiusPx) && all(isfinite(ellipse.conic));
}

bool prepare_ellipse_for_scanlines(Ellipse ellipseIn, out Ellipse ellipseOut, out ScanlineLoopConfig loopCfg)
{
    ellipseOut = ellipseIn;
    loopCfg = make_scanline_loop_config(g_Prepass, false, 0, -1, 0, -1);
    float tileSize = float(g_Prepass.tileSize);
    float halfTile = 0.5 * tileSize;
    if (!try_prepare_conic_for_binning(ellipseOut, halfTile)) return false;

    float2 boundsExtent = ellipseOut.bboxExtent + halfTile;
    int minX = clamp(int(floor((ellipseOut.centerPx.x - boundsExtent.x) / tileSize)), 0, int(g_Prepass.tileWidth) - 1);
    int maxX = clamp(int(ceil((ellipseOut.centerPx.x + boundsExtent.x) / tileSize)), 0, int(g_Prepass.tileWidth) - 1);
    int minY = clamp(int(floor((ellipseOut.centerPx.y - boundsExtent.y) / tileSize)), 0, int(g_Prepass.tileHeight) - 1);
    int maxY = clamp(int(ceil((ellipseOut.centerPx.y + boundsExtent.y) / tileSize)), 0, int(g_Prepass.tileHeight) - 1);
    if (minX > maxX || minY > maxY) return false;

    loopCfg = build_scanline_loop_config(g_Prepass, ellipseOut, minX, maxX, minY, maxY);
    return loopCfg.lineEnd >= loopCfg.lineStart;
}

[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csProjectVisibleSplats(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_Prepass.splatCount) return;
    g_ScreenCenterRadiusDepth[splatId] = 0.0;
    g_ScreenColorAlpha[splatId] = 0.0;
    g_ScreenEllipseConic[splatId] = 0.0;
    g_SplatVisible[splatId] = 0u;

    Gaussian3D gaussian = load_gaussian3d(splatId);
    float fusedOpacity = build_raster_fused_opacity(g_Camera, g_Prepass.radiusScale, gaussian.position, gaussian.logScale, gaussian.rawOpacity);
    if (fusedOpacity < g_Raster.alphaCutoff) return;

    // Project only splats whose cutoff isosurface can contribute within the camera frustum.
    PinholeCamera camera = PinholeCamera.from_params(g_Camera);
    float supportSigmaRadius = sqrt(max(-2.0 * log(g_Raster.alphaCutoff / fusedOpacity), 0.0));
    Ellipsoid projectionEllipsoid = build_ellipsoid_with_support(g_Camera, g_Prepass.radiusScale, supportSigmaRadius, g_MaxAnisotropy, gaussian);
    float3 cameraOffset = projectionEllipsoid.centerWorld - camera.position_world();
    float3 camPos = camera.world_to_camera(cameraOffset);
    float camDistance = length(cameraOffset);
    float depthValue = camPos.z;
    float supportRadius = max(projectionEllipsoid.projectedScale.x, max(projectionEllipsoid.projectedScale.y, projectionEllipsoid.projectedScale.z));
    if (depthValue + supportRadius <= DEPTH_VISIBILITY_FLOOR) return;

    Ellipse projectedEllipse;
    if (!compute_ellipsoid_outline_ellipse(projectionEllipsoid, camera, projectedEllipse)) return;
    projectedEllipse.radiusPx = max(projectedEllipse.radiusPx * ELLIPSE_SAFETY_SCALE + ELLIPSE_RADIUS_PAD_PX, MIN_SCREEN_RADIUS_PX);
    projectedEllipse.bboxExtent = float2(projectedEllipse.radiusPx, projectedEllipse.radiusPx);
    if (!projectedEllipse.isVisible(camDistance, depthValue, camera.params.viewport, g_Camera.nearDepth, g_Camera.farDepth)) return;
    float3 evaluatedColor = evaluate_gaussian_color(gaussian.sh0, gaussian.sh1X, gaussian.sh1Y, gaussian.sh1Z, camera.position_world(), projectionEllipsoid.centerWorld, g_Raster.useSH);
    g_ScreenCenterRadiusDepth[splatId] = float4(projectedEllipse.centerPx, projectedEllipse.radiusPx, camDistance);
    g_ScreenColorAlpha[splatId] = float4(evaluatedColor, gaussian.rawOpacity);
    g_ScreenEllipseConic[splatId] = float4(projectedEllipse.conic, projectionEllipsoid.opacityScale);
    g_SplatVisible[splatId] = 1u;
    CachedEllipsoid cached = build_cached_ellipsoid(g_Camera, g_Prepass.radiusScale, g_MaxAnisotropy, gaussian.position, gaussian.logScale, gaussian.rotation);
    store_raster_cache(splatId, cached, float4(evaluatedColor, fusedOpacity));
    append_visible_splat(float_to_ordered_uint(camDistance), splatId);
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csCountVisibleScanlines(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint visibleSplatId = dispatchThreadID.x;
    uint visibleCount = g_VisibleCounter[0];
    if (visibleSplatId >= visibleCount) return;

    uint splatId = g_VisibleValues[visibleSplatId];
    Ellipse ellipse;
    Ellipse preparedEllipse;
    ScanlineLoopConfig loopCfg;
    uint scanlineCount = 0u;
    if (load_projected_ellipse(splatId, ellipse) && prepare_ellipse_for_scanlines(ellipse, preparedEllipse, loopCfg))
    {
        scanlineCount = uint(loopCfg.lineEnd - loopCfg.lineStart + 1);
    }
    g_ScanlineCounts[visibleSplatId] = scanlineCount;
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csEmitScanlines(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint visibleSplatId = dispatchThreadID.x;
    uint visibleCount = g_VisibleCounter[0];
    if (visibleSplatId >= visibleCount) return;

    uint splatId = g_VisibleValues[visibleSplatId];
    Ellipse ellipse;
    Ellipse preparedEllipse;
    ScanlineLoopConfig loopCfg;
    if (!load_projected_ellipse(splatId, ellipse) || !prepare_ellipse_for_scanlines(ellipse, preparedEllipse, loopCfg)) return;

    uint scanlineBase = g_ScanlineOffsets[visibleSplatId];
    uint scanlineCount = g_ScanlineCounts[visibleSplatId];
    if (scanlineBase >= g_Prepass.maxScanlineEntries || scanlineCount == 0u) return;
    uint scanlineWriteCount = min(scanlineCount, g_Prepass.maxScanlineEntries - scanlineBase);
    float tileSize = float(g_Prepass.tileSize);
    for (uint i = 0u; i < scanlineWriteCount; i++)
    {
        uint lineCoord = uint(loopCfg.lineStart + int(i));
        ScanlineSpan span;
        ScanlineWorkItem item;
        item.splatId = splatId;
        item.lineCoord = lineCoord;
        item.clear(loopCfg.scanAlongX);
        bool hasSpan = compute_scanline_tile_span_universal(preparedEllipse, loopCfg, tileSize, lineCoord, span);
        if (hasSpan && span.lineCount > 0u) item.setSpan(span.lineStartMinor, span.lineCount, loopCfg.scanAlongX);
        g_ScanlineWorkItems[scanlineBase + i] = item;
    }
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csCountScanlineTiles(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint scanlineId = dispatchThreadID.x;
    uint scanlineCount = min(g_ScanlineCounter[0], g_Prepass.maxScanlineEntries);
    if (scanlineId >= scanlineCount) return;
    g_ScanlineTileCounts[scanlineId] = g_ScanlineWorkItems[scanlineId].lineCount();
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csEmitTileEntries(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint scanlineId = dispatchThreadID.x;
    uint scanlineCount = min(g_ScanlineCounter[0], g_Prepass.maxScanlineEntries);
    if (scanlineId >= scanlineCount) return;

    ScanlineWorkItem item = g_ScanlineWorkItems[scanlineId];
    uint lineCount = item.lineCount();
    if (lineCount == 0u) return;

    uint baseIndex = g_ScanlineTileOffsets[scanlineId];
    if (baseIndex >= g_Prepass.maxListEntries) return;
    uint writeCount = min(lineCount, g_Prepass.maxListEntries - baseIndex);
    uint lineCoord = item.lineCoord;
    bool scanAlongX = item.scanAlongXAxis();
    for (uint i = 0u; i < writeCount; i++)
    {
        uint minorCoord = item.lineStartMinor + i;
        uint tileId = scanAlongX
            ? (lineCoord * g_Prepass.tileWidth + minorCoord)
            : (minorCoord * g_Prepass.tileWidth + lineCoord);
        uint writeIndex = baseIndex + i;
        g_Keys[writeIndex] = tileId;
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

    uint tile = g_SortedKeys[idx];
    if (tile >= g_Prepass.tileCount) return;
    uint prevTile = INVALID_START;
    if (idx > 0)
    {
        prevTile = g_SortedKeys[idx - 1];
        if (prevTile >= g_Prepass.tileCount) prevTile = INVALID_START;
    }
    uint nextTile = INVALID_START;
    if (idx + 1 < sortedCount)
    {
        nextTile = g_SortedKeys[idx + 1];
        if (nextTile >= g_Prepass.tileCount) nextTile = INVALID_START;
    }

    if (idx == 0 || tile != prevTile) g_TileRanges[tile].x = idx;
    if (idx + 1 == sortedCount || tile != nextTile) g_TileRanges[tile].y = idx + 1;
}
