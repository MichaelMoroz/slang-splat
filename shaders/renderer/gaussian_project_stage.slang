import "gaussian_types.slang";
import "utility/math/camera.slang";
import "utility/math/constants.slang";
import "utility/random/random.slang";
import "utility/splatting/common.slang";
import "utility/splatting/projection.slang";
import "utility/splatting/raster.slang";

static const uint SORT_CAMERA_DITHER_SPLAT_HASH = 0x6d2b79f5u;
static const uint SORT_CAMERA_DITHER_SEED_HASH = 0x9e3779b9u;
static const uint SORT_SPLAT_KEY_DISTANCE_TO_CAMERA = 0u;
static const uint SORT_SPLAT_KEY_Z_DEPTH = 1u;

uniform float g_MaxAnisotropy;
uniform float3 g_SortCameraPosition;
uniform float g_SortCameraDitherSigma;
uniform uint g_SortCameraDitherSeed;
uniform uint g_SortSplatKeyMode;

// Splat-editor selection highlight. When enabled, splats flagged in
// g_SplatHighlight have their projected color blended toward g_HighlightColor.
uniform uint g_HighlightEnabled;
uniform float3 g_HighlightColor;
uniform float g_HighlightMix;
uniform StructuredBuffer<uint> g_SplatHighlight;

uint encode_sort_key_float(float value)
{
    uint bits = asuint(value);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}

void make_conservative_circle_binning_ellipse(inout Ellipse ellipse)
{
    float radius = max(ellipse.radiusPx, MIN_SCREEN_RADIUS_PX);
    float invRadiusSq = 1.0 / max(radius * radius, ELLIPSE_EPS);
    ellipse.conic = float3(invRadiusSq, 0.0, invRadiusSq);
    ellipse.bboxExtent = float2(radius);
}

bool prepare_projected_ellipse_loop(inout Ellipse ellipse, out ScanlineLoopConfig loopCfg)
{
    loopCfg = make_scanline_loop_config(g_Prepass, false, 0, -1, 0, -1);
    float tileSize = float(g_Prepass.tileSize);
    float halfTile = 0.5 * tileSize;
    if (!try_prepare_conic_for_binning(ellipse, halfTile)) return false;

    float2 boundsExtent = ellipse.bboxExtent + halfTile;
    int minX = clamp(int(floor((ellipse.centerPx.x - boundsExtent.x) / tileSize)), 0, int(g_Prepass.tileWidth) - 1);
    int maxX = clamp(int(ceil((ellipse.centerPx.x + boundsExtent.x) / tileSize)), 0, int(g_Prepass.tileWidth) - 1);
    int minY = clamp(int(floor((ellipse.centerPx.y - boundsExtent.y) / tileSize)), 0, int(g_Prepass.tileHeight) - 1);
    int maxY = clamp(int(ceil((ellipse.centerPx.y + boundsExtent.y) / tileSize)), 0, int(g_Prepass.tileHeight) - 1);
    if (minX > maxX || minY > maxY) return false;

    loopCfg = build_scanline_loop_config(g_Prepass, ellipse, minX, maxX, minY, maxY);
    return loopCfg.lineEnd >= loopCfg.lineStart;
}

uint load_prepared_projected_ellipse_variants(
    uint splatId,
    out Ellipse ellipse0,
    out ScanlineLoopConfig loopCfg0,
    out Ellipse ellipse1,
    out ScanlineLoopConfig loopCfg1)
{
    ellipse0.init(0.0, 0.0, 0.0);
    ellipse1.init(0.0, 0.0, 0.0);
    loopCfg0 = make_scanline_loop_config(g_Prepass, false, 0, -1, 0, -1);
    loopCfg1 = make_scanline_loop_config(g_Prepass, false, 0, -1, 0, -1);
    float4 centerRadiusDepth = g_ScreenCenterRadiusDepth[splatId];
    if (centerRadiusDepth.z <= 0.0 || !all(isfinite(centerRadiusDepth))) return 0u;
    float4 conicOpacity = g_ScreenEllipseConic[splatId];
    ellipse0.init(centerRadiusDepth.xy, centerRadiusDepth.z, conicOpacity.xyz);
    if (!all(isfinite(ellipse0.centerPx)) || !isfinite(ellipse0.radiusPx) || !all(isfinite(ellipse0.conic))) return 0u;

    float viewportWidth = max(g_Camera.viewport.x, 1.0);
    bool equirectangular = g_Camera.projectionModel == CAMERA_PROJECTION_EQUIRECTANGULAR;
    bool wrapsLeft = equirectangular && ellipse0.centerPx.x - ellipse0.radiusPx <= 0.0 && ellipse0.centerPx.x + ellipse0.radiusPx < viewportWidth;
    bool wrapsRight = equirectangular && ellipse0.centerPx.x + ellipse0.radiusPx >= viewportWidth && ellipse0.centerPx.x - ellipse0.radiusPx > 0.0;
    bool hasWrappedVariant = wrapsLeft || wrapsRight;
    if (hasWrappedVariant) make_conservative_circle_binning_ellipse(ellipse0);

    uint variantCount = prepare_projected_ellipse_loop(ellipse0, loopCfg0) ? 1u : 0u;
    if (hasWrappedVariant)
    {
        ellipse1 = ellipse0;
        ellipse1.centerPx.x += wrapsLeft ? viewportWidth : -viewportWidth;
        if (prepare_projected_ellipse_loop(ellipse1, loopCfg1)) variantCount++;
    }
    return variantCount;
}

uint scanline_loop_count(ScanlineLoopConfig loopCfg)
{
    return loopCfg.lineEnd >= loopCfg.lineStart ? uint(loopCfg.lineEnd - loopCfg.lineStart + 1) : 0u;
}

uint emit_projected_ellipse_scanlines(uint splatId, Ellipse preparedEllipse, ScanlineLoopConfig loopCfg, uint writeIndex, uint writeEnd)
{
    float tileSize = float(g_Prepass.tileSize);
    uint loopCount = scanline_loop_count(loopCfg);
    for (uint i = 0u; i < loopCount && writeIndex < writeEnd; i++)
    {
        uint lineCoord = uint(loopCfg.lineStart + int(i));
        ScanlineSpan span;
        ScanlineWorkItem item;
        item.splatId = splatId;
        item.clear(loopCfg.scanAlongX);
        if (compute_scanline_tile_span_universal(preparedEllipse, loopCfg, tileSize, lineCoord, span) && span.lineCount > 0u)
        {
            uint startTileId = loopCfg.scanAlongX
                ? (lineCoord * g_Prepass.tileWidth + span.lineStartMinor)
                : (span.lineStartMinor * g_Prepass.tileWidth + lineCoord);
            item.setSpan(startTileId, span.lineCount, loopCfg.scanAlongX);
        }
        g_ScanlineWorkItems[writeIndex] = item.packed();
        writeIndex++;
    }
    return writeIndex;
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
    g_SplatVisibleAreaPx[splatId] = 0.0;

    Gaussian3D gaussian = load_gaussian3d(splatId);
    PinholeCamera camera = PinholeCamera.from_params(g_Camera);
    float3 cameraPosition = camera.position_world();
    RasterGaussian rasterGaussian = build_raster_gaussian(g_Camera, g_Prepass.radiusScale, g_MaxAnisotropy, cameraPosition, g_Raster.shBand, gaussian);
    if (g_HighlightEnabled != 0u && g_SplatHighlight[splatId] != 0u)
        rasterGaussian.colorAlpha.xyz = lerp(rasterGaussian.colorAlpha.xyz, g_HighlightColor, saturate(g_HighlightMix));
    float fusedOpacity = rasterGaussian.colorAlpha.w;
    if (fusedOpacity < g_Raster.alphaCutoff) return;

    // Project only splats whose cutoff isosurface can contribute within the camera frustum.
    float supportSigmaRadius = sqrt(max(-2.0 * log(g_Raster.alphaCutoff / fusedOpacity), 0.0));
    Ellipsoid projectionEllipsoid = build_ellipsoid_with_support(g_Camera, g_Prepass.radiusScale, supportSigmaRadius, g_MaxAnisotropy, gaussian);
    float3 cameraOffset = projectionEllipsoid.centerWorld - cameraPosition;
    float3 camPos = camera.world_to_camera(cameraOffset);
    float camDistance = length(cameraOffset);
    if (camDistance < g_Camera.minCameraDistance) return;
    float depthValue = camPos.z;
    float supportRadius = max(projectionEllipsoid.projectedScale.x, max(projectionEllipsoid.projectedScale.y, projectionEllipsoid.projectedScale.z));
    bool hasProjectionDistortion = camera.is_equirectangular() || any(abs(g_Camera.projDistortionK1K2P1P2) > SMALL_VALUE) || any(abs(g_Camera.projDistortionK3K4K5K6) > SMALL_VALUE);
    if (!hasProjectionDistortion && !support_sphere_intersects_view_frustum(camera, camPos, supportRadius)) return;

    Ellipse projectedEllipse;
    if (!compute_ellipsoid_outline_ellipse(projectionEllipsoid, camera, projectedEllipse)) return;
    projectedEllipse.radiusPx = max(projectedEllipse.radiusPx * ELLIPSE_SAFETY_SCALE + ELLIPSE_RADIUS_PAD_PX, MIN_SCREEN_RADIUS_PX);
    projectedEllipse.bboxExtent = float2(projectedEllipse.radiusPx, projectedEllipse.radiusPx);
    if (!projectedEllipse.isVisible(camDistance, depthValue, camera.params.viewport, g_Camera.nearDepth, g_Camera.farDepth)) return;
    g_ScreenCenterRadiusDepth[splatId] = float4(projectedEllipse.centerPx, projectedEllipse.radiusPx, camDistance);
    g_ScreenColorAlpha[splatId] = float4(rasterGaussian.colorAlpha.xyz, gaussian.rawOpacity);
    g_ScreenEllipseConic[splatId] = float4(projectedEllipse.conic, projectionEllipsoid.opacityScale);
    g_SplatVisible[splatId] = 1u;
    g_SplatVisibleAreaPx[splatId] = projectedEllipse.clippedAreaPx(camera.params.viewport);
    g_RasterCache[splatId] = rasterGaussian;
    float3 sortCameraPosition = g_SortCameraPosition;
    if (g_SortCameraDitherSigma > 0.0)
    {
        uint rngState = hash_u32(g_SortCameraDitherSeed ^ ((splatId + 1u) * SORT_CAMERA_DITHER_SPLAT_HASH));
        rngState = hash_u32(rngState ^ SORT_CAMERA_DITHER_SEED_HASH);
        sortCameraPosition += rng_normal3_box_muller(rngState) * g_SortCameraDitherSigma;
    }
    float3 sortCameraOffset = projectionEllipsoid.centerWorld - sortCameraPosition;
    float sortKey = (g_SortSplatKeyMode == SORT_SPLAT_KEY_Z_DEPTH && !camera.is_equirectangular()) ? camera.world_to_camera(sortCameraOffset).z : length(sortCameraOffset);
    uint writeIndex = 0u;
    InterlockedAdd(g_VisibleCounter[0], 1u, writeIndex);
    if (writeIndex >= g_Prepass.splatCount) return;
    g_VisibleKeys[writeIndex] = encode_sort_key_float(sortKey);
    g_VisibleValues[writeIndex] = splatId;
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csCountVisibleScanlines(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint visibleSplatId = dispatchThreadID.x;
    uint visibleCount = g_VisibleCounter[0];
    if (visibleSplatId >= visibleCount) return;

    uint splatId = g_VisibleValues[visibleSplatId];
    Ellipse ellipse0, ellipse1;
    ScanlineLoopConfig loopCfg0, loopCfg1;
    uint scanlineCount = 0u;
    uint variantCount = load_prepared_projected_ellipse_variants(splatId, ellipse0, loopCfg0, ellipse1, loopCfg1);
    if (variantCount > 0u)
    {
        scanlineCount += scanline_loop_count(loopCfg0);
        if (variantCount > 1u) scanlineCount += scanline_loop_count(loopCfg1);
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
    Ellipse ellipse0, ellipse1;
    ScanlineLoopConfig loopCfg0, loopCfg1;
    uint variantCount = load_prepared_projected_ellipse_variants(splatId, ellipse0, loopCfg0, ellipse1, loopCfg1);
    if (variantCount == 0u) return;

    uint scanlineBase = g_ScanlineOffsets[visibleSplatId];
    uint scanlineCount = g_ScanlineCounts[visibleSplatId];
    if (scanlineBase >= g_Prepass.maxScanlineEntries || scanlineCount == 0u) return;
    uint writeEnd = min(scanlineBase + scanlineCount, g_Prepass.maxScanlineEntries);
    uint writeIndex = emit_projected_ellipse_scanlines(splatId, ellipse0, loopCfg0, scanlineBase, writeEnd);
    if (variantCount > 1u) emit_projected_ellipse_scanlines(splatId, ellipse1, loopCfg1, writeIndex, writeEnd);
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csCountScanlineTiles(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint scanlineId = dispatchThreadID.x;
    uint scanlineCount = min(g_ScanlineCounter[0], g_Prepass.maxScanlineEntries);
    if (scanlineId >= scanlineCount) return;
    ScanlineWorkItem item;
    item.loadPacked(g_ScanlineWorkItems[scanlineId]);
    g_ScanlineTileCounts[scanlineId] = item.lineCount();
}

[shader("compute")]
[numthreads(TILE_GROUP_SIZE, 1, 1)]
void csEmitTileEntries(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint scanlineId = dispatchThreadID.x;
    uint scanlineCount = min(g_ScanlineCounter[0], g_Prepass.maxScanlineEntries);
    if (scanlineId >= scanlineCount) return;

    ScanlineWorkItem item;
    item.loadPacked(g_ScanlineWorkItems[scanlineId]);
    uint lineCount = item.lineCount();
    if (lineCount == 0u) return;

    uint baseIndex = g_ScanlineTileOffsets[scanlineId];
    if (baseIndex >= g_Prepass.maxListEntries) return;
    uint writeCount = min(lineCount, g_Prepass.maxListEntries - baseIndex);
    uint startTileId = item.startTileId();
    uint stride = item.scanAlongXAxis() ? 1u : g_Prepass.tileWidth;
    for (uint i = 0u; i < writeCount; i++)
    {
        uint tileId = startTileId + i * stride;
        g_Keys[baseIndex + i] = tileId;
        g_Values[baseIndex + i] = item.splatId;
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
