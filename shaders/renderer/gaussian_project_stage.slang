import "gaussian_types.slang";
import "gaussian_math_utils.slang";
import "gaussian_mvee_utils.slang";
import "gaussian_binning_utils.slang";
import "gaussian_raster_stage.slang";

void project_and_bin_legacy_impl(uint3 dispatchThreadID)
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
    float4 qInv = float4(q.x, -q.y, -q.z, -q.w);
    float3 scale = ellipsoid.scale;

    float3 axis0 = world_to_camera(rotate_by_quat(float3(1.0, 0.0, 0.0), qInv)) * scale.x;
    float3 axis1 = world_to_camera(rotate_by_quat(float3(0.0, 1.0, 0.0), qInv)) * scale.y;
    float3 axis2 = world_to_camera(rotate_by_quat(float3(0.0, 0.0, 1.0), qInv)) * scale.z;

    float2 center = float2(px, py);
    float2 extent = float2(0.0, 0.0);
    float2 p0p = project_cam(camPos + axis0);
    float2 p0m = project_cam(camPos - axis0);
    float2 p1p = project_cam(camPos + axis1);
    float2 p1m = project_cam(camPos - axis1);
    float2 p2p = project_cam(camPos + axis2);
    float2 p2m = project_cam(camPos - axis2);
    extent = max(extent, abs(p0p - center));
    extent = max(extent, abs(p0m - center));
    extent = max(extent, abs(p1p - center));
    extent = max(extent, abs(p1m - center));
    extent = max(extent, abs(p2p - center));
    extent = max(extent, abs(p2m - center));
    float radiusPx = clamp(max(extent.x, extent.y) + 1.0, 1.0, g_maxSplatRadiusPx);

    float2 e0 = 0.5 * (p0p - p0m);
    float2 e1 = 0.5 * (p1p - p1m);
    float2 e2 = 0.5 * (p2p - p2m);
    float s00 = dot(float3(e0.x, e1.x, e2.x), float3(e0.x, e1.x, e2.x)) + 1e-6;
    float s01 = dot(float3(e0.x, e1.x, e2.x), float3(e0.y, e1.y, e2.y));
    float s11 = dot(float3(e0.y, e1.y, e2.y), float3(e0.y, e1.y, e2.y)) + 1e-6;
    float detS = s00 * s11 - s01 * s01;
    float invR2 = 1.0 / max(radiusPx * radiusPx, 1e-6);
    float3 legacyConic = float3(invR2, 0.0, invR2);
    if (detS > 1e-8)
    {
        legacyConic = float3(s11 / detS, -s01 / detS, s00 / detS);
    }

    float3 invScale = ellipsoid.invScale;
    float3 roLocal = ellipsoid.roLocal;

    g_ScreenCenterRadiusDepth[splatId] = float4(px, py, radiusPx, camDistance);
    g_ScreenColorAlpha[splatId] = gaussian.colorAlpha;
    g_ScreenEllipseConic[splatId] = float4(legacyConic, 1.0);
    g_SplatPosLocal[splatId] = float4(roLocal, 0.0);
    g_SplatInvScale[splatId] = float4(invScale, 0.0);
    g_SplatQuat[splatId] = q;
    SplatBinRequest binRequest;
    binRequest.splatId = splatId;
    binRequest.camDistance = camDistance;
    binRequest.depthValue = depthValue;
    binRequest.ellipse.init(float2(px, py), radiusPx, legacyConic);
    emit_splat_bins(binRequest);
}

void project_and_bin_sampled5_mvee_impl(uint3 dispatchThreadID)
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
    float3 scale = ellipsoid.scale;
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
    if (idx >= g_sortedCount)
    {
        return;
    }

    uint tile = g_SortedKeys[idx] >> g_depthBits;
    uint prevTile = (idx > 0) ? (g_SortedKeys[idx - 1] >> g_depthBits) : INVALID_START;
    uint nextTile = (idx + 1 < g_sortedCount) ? (g_SortedKeys[idx + 1] >> g_depthBits) : INVALID_START;

    if (idx == 0 || tile != prevTile)
    {
        g_TileRanges[tile].x = idx;
    }
    if (idx + 1 == g_sortedCount || tile != nextTile)
    {
        g_TileRanges[tile].y = idx + 1;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csProjectAndBin(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    project_and_bin_legacy_impl(dispatchThreadID);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void csProjectAndBinSampled5MVEE(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    project_and_bin_sampled5_mvee_impl(dispatchThreadID);
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
