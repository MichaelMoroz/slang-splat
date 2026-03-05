import "gaussian_types.slang";
import "camera.slang";

float screen_radius_cap()
{
    return max(max(g_Camera.viewport.x, g_Camera.viewport.y), MIN_RADIUS_PX);
}

bool generate_silhouette_points_local5(float3 roLocal, float eps, out float3 points[MVEE_POINT_COUNT])
{
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) points[i] = 0.0;

    float d = length(roLocal);
    if (d <= 1.0 + eps) return false;

    float3 n = roLocal / d;
    float3 circleCenter = n / d;
    float circleRadius = sqrt(max(1.0 - 1.0 / (d * d), 0.0));
    float3 u;
    float3 v;
    if (n.z < MVEE_SOUTH_POLE_THRESHOLD)
    {
        u = float3(0.0, -1.0, 0.0);
        v = float3(-1.0, 0.0, 0.0);
    } else {
        float a = 1.0 / (1.0 + n.z);
        float b = -n.x * n.y * a;
        u = float3(1.0 - n.x * n.x * a, b, -n.x);
        v = float3(b, 1.0 - n.y * n.y * a, -n.y);
    }
    if (!all(isfinite(u)) || !all(isfinite(v))) return false;

    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++)
    {
        float theta = MVEE_FULL_TURN_RADIANS * (float(i) / float(MVEE_POINT_COUNT));
        float s, c;
        sincos(theta, s, c);
        points[i] = circleCenter + circleRadius * (c * u + s * v);
    }
    return true;
}

bool fit_mvee_5pt_fixed_iters(
    float2 points[MVEE_POINT_COUNT],
    uint mveeIters,
    float eps,
    out MveeFitResult fitOut
)
{
    fitOut.clear();
    float2 pointsNorm[MVEE_POINT_COUNT];
    float2 normCenter = 0.0;
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) normCenter += points[i];
    normCenter *= (1.0 / float(MVEE_POINT_COUNT));

    float normScale = 0.0;
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) normScale = max(normScale, length(points[i] - normCenter));
    if (normScale <= eps) return false;
    float invNormScale = 1.0 / normScale;
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) pointsNorm[i] = (points[i] - normCenter) * invNormScale;

    float weights[MVEE_POINT_COUNT];
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) weights[i] = 1.0 / float(MVEE_POINT_COUNT);

    [unroll] for (uint iter = 0; iter < MVEE_MAX_ITERS; iter++)
    {
        if (iter >= mveeIters) break;

        float3x3 xMat = 0.0;
        [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++)
        {
            float3 q = float3(pointsNorm[i], 1.0);
            xMat += weights[i] * float3x3(
                q.x * q.x, q.x * q.y, q.x,
                q.y * q.x, q.y * q.y, q.y,
                q.x, q.y, 1.0
            );
        }

        float3x3 xInv = 0.0;
        if (!inverse3x3(xMat, eps, xInv)) return false;

        uint maxIdx = 0;
        float maxVal = -1.0;
        [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++)
        {
            float3 q = float3(pointsNorm[i], 1.0);
            float3 xq = mul(xInv, q);
            float m = dot(q, xq);
            if (m > maxVal)
            {
                maxVal = m;
                maxIdx = i;
            }
        }

        float denom = MVEE_AUGMENTED_DIMENSION * (maxVal - 1.0);
        if (denom <= eps) break;
        float alpha = saturate((maxVal - MVEE_AUGMENTED_DIMENSION) / denom);
        [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) weights[i] *= (1.0 - alpha);
        weights[maxIdx] += alpha;
    }

    float2 center = 0.0;
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++) center += weights[i] * pointsNorm[i];

    float3 secondMoment = 0.0;
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++)
    {
        float2 d = pointsNorm[i] - center;
        secondMoment += weights[i] * float3(d.x * d.x, d.x * d.y, d.y * d.y);
    }
    secondMoment += float3(eps, 0.0, eps);

    float detS = secondMoment.x * secondMoment.z - secondMoment.y * secondMoment.y;
    if (detS <= eps) return false;

    float3 conicNorm = 0.5 * float3(secondMoment.z, -secondMoment.y, secondMoment.x) / detS;
    float trace = conicNorm.x + conicNorm.z;
    float detA = conicNorm.x * conicNorm.z - conicNorm.y * conicNorm.y;
    float disc = sqrt(max(0.25 * trace * trace - detA, 0.0));
    float l0 = max(0.5 * trace + disc, eps);
    float l1 = max(0.5 * trace - disc, eps);
    float axis0 = rsqrt(l0);
    float axis1 = rsqrt(l1);

    float maxQuad = 0.0;
    [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++)
    {
        float2 d = pointsNorm[i] - center;
        float quad = dot(float3(d.x * d.x, d.x * d.y, d.y * d.y), float3(conicNorm.x, 2.0 * conicNorm.y, conicNorm.z));
        maxQuad = max(maxQuad, quad);
    }
    float inflate = (maxQuad > 1.0) ? sqrt(maxQuad) : 1.0;
    float invInflate2 = 1.0 / max(inflate * inflate, eps);
    fitOut.centerPx = normCenter + center * normScale;
    fitOut.radiusPx = max(axis0, axis1) * inflate * normScale;
    fitOut.conic = conicNorm * (invInflate2 / max(normScale * normScale, eps));
    return all(isfinite(fitOut.centerPx)) && isfinite(fitOut.radiusPx) && fitOut.radiusPx > 0.0;
}

bool try_prepare_conic_for_binning(inout Ellipse ellipse, float halfTile)
{
    ellipse.bboxExtent = float2(ellipse.radiusPx, ellipse.radiusPx);
    float3 conic = ellipse.conic;
    float detConic = conic.x * conic.z - conic.y * conic.y;
    bool useConic = all(isfinite(conic))
        && conic.x > EPS_CONIC_POSITIVE
        && conic.z > EPS_CONIC_POSITIVE
        && detConic > EPS_CONIC_DETERMINANT;
    if (!useConic) return false;

    float traceConic = conic.x + conic.z;
    float discConic = sqrt(max(0.25 * traceConic * traceConic - detConic, 0.0));
    float eigMin = 0.5 * traceConic - discConic;
    float eigMax = 0.5 * traceConic + discConic;
    float maxAxis = rsqrt(max(eigMin, EPS_CONIC_INV_AXIS));
    float minAxis = rsqrt(max(eigMax, EPS_CONIC_INV_AXIS));
    float radiusSafe = max(ellipse.radiusPx, MIN_RADIUS_PX);
    float axisLimit = radiusSafe * 2.0 + halfTile;
    useConic = isfinite(maxAxis)
        && isfinite(minAxis)
        && eigMin > 0.0
        && eigMax > 0.0
        && maxAxis <= axisLimit;
    if (!useConic) return false;

    float xExtent = sqrt(max(conic.z / detConic, 0.0));
    float yExtent = sqrt(max(conic.x / detConic, 0.0));
    if (!all(isfinite(float2(xExtent, yExtent)))) return false;

    ellipse.bboxExtent = clamp(
        float2(xExtent, yExtent),
        float2(MIN_BBOX_EXTENT_PX),
        float2(ellipse.radiusPx)
    );
    return true;
}

ScanlineLoopConfig make_scanline_loop_config(
    bool scanAlongX,
    int minX,
    int maxX,
    int minY,
    int maxY
)
{
    ScanlineLoopConfig cfg;
    cfg.scanAlongX = scanAlongX;
    cfg.lineStart = cfg.scanAlongX ? minY : minX;
    cfg.lineEnd = cfg.scanAlongX ? maxY : maxX;
    cfg.minorMin = cfg.scanAlongX ? minX : minY;
    cfg.minorMax = cfg.scanAlongX ? maxX : maxY;
    cfg.minorLimit = cfg.scanAlongX ? g_Prepass.tileWidth : g_Prepass.tileHeight;
    cfg.lineLimit = cfg.scanAlongX ? g_Prepass.tileHeight : g_Prepass.tileWidth;
    return cfg;
}

ScanlineLoopConfig build_scanline_loop_config(
    Ellipse ellipse,
    int minX,
    int maxX,
    int minY,
    int maxY
)
{
    uint spanX = uint(maxX - minX + 1);
    uint spanY = uint(maxY - minY + 1);
    bool scanAlongX = ellipse.chooseScanAlongX(spanX, spanY);
    return make_scanline_loop_config(scanAlongX, minX, maxX, minY, maxY);
}

void accumulate_conic_cross_section(
    float conicA, float conicB, float det, float invA,
    float centerMinor, float dM,
    inout float minPx, inout float maxPx
)
{
    float disc = conicA - det * dM * dM;
    if (disc <= 0) return;
    float half = sqrt(disc) * invA;
    float shift = -conicB * dM * invA;
    minPx = min(minPx, centerMinor + shift - half);
    maxPx = max(maxPx, centerMinor + shift + half);
}

bool compute_scanline_tile_span_universal(
    Ellipse ellipse,
    ScanlineLoopConfig cfg,
    float tileSize,
    uint lineCoord,
    out ScanlineSpan span
)
{
    span.lineStartMinor = 0;
    span.lineCount = 0;
    if (!cfg.isLineCoordValid(lineCoord) || cfg.minorMin > cfg.minorMax) return false;

    float centerMinor = cfg.scanAlongX ? ellipse.centerPx.x : ellipse.centerPx.y;
    float centerMajor = cfg.scanAlongX ? ellipse.centerPx.y : ellipse.centerPx.x;
    float dM0 = float(lineCoord) * tileSize - centerMajor;
    float dM1 = dM0 + tileSize;
    float minPx = 1e30;
    float maxPx = -1e30;

    float A = cfg.scanAlongX ? ellipse.conic.x : ellipse.conic.z;
    float B = ellipse.conic.y;
    float C = cfg.scanAlongX ? ellipse.conic.z : ellipse.conic.x;
    float det = A * C - B * B;
    float invA = 1.0 / A;

    accumulate_conic_cross_section(A, B, det, invA, centerMinor, dM0, minPx, maxPx);
    accumulate_conic_cross_section(A, B, det, invA, centerMinor, dM1, minPx, maxPx);

    float detC = det * C;
    if (detC > 0)
    {
        float dMCrit = B / sqrt(detC);
        if (dMCrit > dM0 && dMCrit < dM1) accumulate_conic_cross_section(A, B, det, invA, centerMinor, dMCrit, minPx, maxPx);
        if (-dMCrit > dM0 && -dMCrit < dM1) accumulate_conic_cross_section(A, B, det, invA, centerMinor, -dMCrit, minPx, maxPx);
    }

    if (minPx > maxPx) return false;

    int first = max(int(floor(minPx / tileSize)), cfg.minorMin);
    int last = min(int(floor(maxPx / tileSize)), cfg.minorMax);
    if (first > last) return false;
    span.lineStartMinor = uint(first);
    span.lineCount = uint(last - first + 1);
    return true;
}

void emit_splat_bins(SplatBinRequest request)
{
    Ellipse ellipse = request.ellipse;
    if (!ellipse.isVisible(
        request.camDistance,
        request.depthValue,
        g_Camera.viewport,
        g_Camera.nearDepth,
        g_Camera.farDepth
    )) return;

    float tileSize = float(g_Prepass.tileSize);
    float halfTile = 0.5 * tileSize;
    if (!try_prepare_conic_for_binning(ellipse, halfTile)) return;
    float2 boundsExtent = ellipse.bboxExtent + halfTile;
    int minX = int(floor((ellipse.centerPx.x - boundsExtent.x) / tileSize));
    int maxX = int(ceil((ellipse.centerPx.x + boundsExtent.x) / tileSize));
    int minY = int(floor((ellipse.centerPx.y - boundsExtent.y) / tileSize));
    int maxY = int(ceil((ellipse.centerPx.y + boundsExtent.y) / tileSize));
    minX = clamp(minX, 0, int(g_Prepass.tileWidth) - 1);
    maxX = clamp(maxX, 0, int(g_Prepass.tileWidth) - 1);
    minY = clamp(minY, 0, int(g_Prepass.tileHeight) - 1);
    maxY = clamp(maxY, 0, int(g_Prepass.tileHeight) - 1);
    if (minX > maxX || minY > maxY) return;

    ScanlineLoopConfig loopCfg = build_scanline_loop_config(ellipse, minX, maxX, minY, maxY);
    uint scanlineCount = uint(loopCfg.lineEnd - loopCfg.lineStart + 1);
    if (scanlineCount == 0u) return;

    uint scanlineBase = 0;
    InterlockedAdd(g_ScanlineCounter[0], scanlineCount, scanlineBase);
    if (scanlineBase >= g_Prepass.maxScanlineEntries) return;

    uint scanlineWriteCount = min(scanlineCount, g_Prepass.maxScanlineEntries - scanlineBase);
    if (scanlineWriteCount == 0u) return;

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
    if (writtenListCount == 0u) return;

    uint listBase = 0u;
    InterlockedAdd(g_ListCounter[0], writtenListCount, listBase);
    g_SplatListBases[request.splatId] = listBase;
}

[shader("compute")]
[numthreads(PROJECT_GROUP_SIZE, 1, 1)]
void csProjectAndBin(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint splatId = dispatchThreadID.x;
    if (splatId >= g_Prepass.splatCount) return;

    PinholeCamera camera = PinholeCamera.from_params(g_Camera);
    Gaussian3D gaussian = Gaussian3D.load(splatId);
    Ellipsoid ellipsoid = build_ellipsoid(gaussian);
    float3 cameraOffset = ellipsoid.centerWorld - camera.position_world();
    float3 camPos = camera.world_to_camera(cameraOffset);
    float camDistance = length(cameraOffset);
    float depthValue = camPos.z;

    float invDepth = 1.0 / max(depthValue, EPS_SMALL);
    float2 fitCenter = camera.project_camera_to_screen_linear(camPos);
    float maxAxisWorld = max(ellipsoid.scale.x, max(ellipsoid.scale.y, ellipsoid.scale.z));
    float fallbackFocal = max(g_Camera.focalPixels.x, g_Camera.focalPixels.y);
    float radiusCap = screen_radius_cap();
    float fallbackRadius = fallbackFocal * maxAxisWorld * invDepth;
    float fitRadius = clamp(
        fallbackRadius * g_Prepass.sampled5SafetyScale + g_Prepass.sampled5RadiusPadPx,
        MIN_RADIUS_PX,
        radiusCap
    );
    float invFitRadius2 = 1.0 / max(fitRadius * fitRadius, EPS_SMALL);
    float3 fitConic = float3(invFitRadius2, 0.0, invFitRadius2);

    float3 localPoints[MVEE_POINT_COUNT];
    bool silhouetteOk = generate_silhouette_points_local5(ellipsoid.roLocal, g_Prepass.sampled5Eps, localPoints);
    if (silhouetteOk)
    {
        float2 screenPoints[MVEE_POINT_COUNT];
        [unroll] for (uint i = 0; i < MVEE_POINT_COUNT; i++)
        {
            float3 worldPt = local_point_to_world(localPoints[i], ellipsoid.centerWorld, ellipsoid.invScale, ellipsoid.rotation);
            float2 screenPt = 0.0;
            bool pointOk = camera.project_world_to_screen(worldPt, screenPt);
            screenPoints[i] = screenPt;
            silhouetteOk = silhouetteOk && pointOk;
        }

        if (silhouetteOk)
        {
            MveeFitResult mveeFit;
            bool fitOk = fit_mvee_5pt_fixed_iters(
                screenPoints,
                min(g_Prepass.sampled5MVEEIters, MVEE_MAX_ITERS),
                g_Prepass.sampled5Eps,
                mveeFit
            );
            if (fitOk && all(isfinite(mveeFit.centerPx)) && isfinite(mveeFit.radiusPx) && all(isfinite(mveeFit.conic)))
            {
                fitCenter = mveeFit.centerPx;
                fitRadius = clamp(
                    mveeFit.radiusPx * g_Prepass.sampled5SafetyScale + g_Prepass.sampled5RadiusPadPx,
                    MIN_RADIUS_PX,
                    radiusCap
                );
                float conicScale = fitRadius / max(mveeFit.radiusPx, g_Prepass.sampled5Eps);
                float invScale2 = 1.0 / max(conicScale * conicScale, g_Prepass.sampled5Eps);
                fitConic = mveeFit.conic * invScale2;
            }
        }
    }

    g_ScreenCenterRadiusDepth[splatId] = float4(fitCenter, fitRadius, camDistance);
    g_ScreenColorAlpha[splatId] = gaussian.colorAlpha;
    g_ScreenEllipseConic[splatId] = float4(fitConic, 1.0);
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

